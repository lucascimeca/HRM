from typing import Optional, List
import math
import numpy as np

import torch
import torch.distributed as dist
import wandb

# Optional matplotlib for labeled heatmap with colorbar; fall back gracefully if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class MoEUsageLogger:
    """
    Low-bandwidth MoE usage logger:
      - Per-layer scalars: neff_frac, entropy01, gini, cv, max_share, topk_share
      - Optional fixed-bin histogram (no raw samples uploaded)
      - Heatmap (layers x maxE) with labels and colorbar when matplotlib is available; compact image fallback otherwise
    """
    def __init__(
        self,
        experts_per_layer: List[int],
        ema_beta: float = 0.95,
        topk_share_k: int = 4,
        log_every: int = 100,
        make_histogram: bool = False,
        hist_bins: int = 16,
        image_gamma: float = 0.5,
        name_prefix: str = "moe",
    ):
        self.E = list(experts_per_layer)
        self.L = len(self.E)
        self.maxE = max(self.E) if self.E else 0
        self.beta = float(ema_beta)
        self.topk_share_k = int(topk_share_k)
        self.log_every = int(log_every)
        self.make_hist = bool(make_histogram)
        self.hist_bins = int(hist_bins)
        self.gamma = float(image_gamma)
        self.prefix = name_prefix
        self.step = 0
        self.layer_labels = [f"H{idx+1}" for idx in range(self.L)]
        # Use 1-based expert labels for readability (E1..Emax)
        self.expert_labels = [f"E{j+1}" for j in range(self.maxE)]
        self.ema_counts = [torch.zeros(e, dtype=torch.float32) for e in self.E]
        if self.make_hist:
            self.bin_edges = np.linspace(0.0, 1.0, self.hist_bins + 1, dtype=np.float32)

    @staticmethod
    def _allreduce_sum_(t: torch.Tensor) -> None:
        """All-reduce sum into-place on tensor t regardless of backend/device.
        - If backend is NCCL, reduce on current CUDA device then copy back to t's device.
        - Else (e.g., Gloo), reduce directly on t (CPU tensors supported).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return

        # Determine backend robustly
        backend = None
        try:
            backend = dist.get_backend()
        except Exception:
            backend = None

        # Ensure we have a float32 buffer to reduce
        original_dtype = t.dtype
        src = t if t.dtype == torch.float32 else t.to(torch.float32)

        def _copy_back(src_reduced: torch.Tensor) -> None:
            # Copy reduced values back into original tensor on its original device/dtype
            if src_reduced.device != t.device or src_reduced.dtype != original_dtype:
                t.copy_(src_reduced.to(device=t.device, dtype=original_dtype))
            else:
                # Same storage: if src is t, reduction already updated t in-place
                if src_reduced.data_ptr() != t.data_ptr():
                    t.copy_(src_reduced)

        # Handle NCCL (GPU-only) vs others (e.g., Gloo CPU)
        is_nccl = False
        try:
            # dist.Backend.NCCL is an Enum in recent PyTorch; compare both styles for safety
            is_nccl = (backend == dist.Backend.NCCL) or (str(backend).lower() == "nccl")
        except Exception:
            is_nccl = str(backend).lower() == "nccl"

        if is_nccl:
            # Move to current CUDA device for reduction, then bring back
            if not torch.cuda.is_available():
                # Fallback: attempt CPU all_reduce (will fail for NCCL, but keeps behavior explicit)
                dist.all_reduce(src, op=dist.ReduceOp.SUM)
                _copy_back(src)
                return
            cuda_dev = torch.device("cuda", torch.cuda.current_device())
            tmp = src if src.device.type == "cuda" else src.to(cuda_dev, non_blocking=True)
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            _copy_back(tmp)
        else:
            # Gloo path: CPU tensors supported directly
            dist.all_reduce(src, op=dist.ReduceOp.SUM)
            _copy_back(src)

    @staticmethod
    def _norm_counts(counts: torch.Tensor) -> torch.Tensor:
        s = counts.sum()
        if s <= 0:
            return torch.full_like(counts, 1.0 / max(1, counts.numel()))
        return counts / s

    @staticmethod
    def _entropy01(p: torch.Tensor) -> float:
        e = max(1, p.numel())
        plogp = (p * (p.clamp_min(1e-12)).log()).sum().item()
        h = -plogp
        return float(h / (math.log(e) if e > 1 else 1.0))

    @staticmethod
    def _gini(p: torch.Tensor) -> float:
        e = p.numel()
        if e <= 1:
            return 0.0
        ps = torch.sort(p)[0]
        idx = torch.arange(1, e + 1, dtype=torch.float32)
        g = 1.0 - 2.0 * torch.sum((e - idx + 0.5) * ps) / (e * 1.0)
        return float(g.item())

    @staticmethod
    def _neff_frac(p: torch.Tensor) -> float:
        e = max(1, p.numel())
        hhi = torch.sum(p * p).item()
        if hhi <= 0:
            return 0.0
        neff = 1.0 / hhi
        return float(neff / e)

    @staticmethod
    def _cv(p: torch.Tensor) -> float:
        e = max(1, p.numel())
        mean = 1.0 / e
        std = float(p.std(unbiased=False).item())
        return float(std / (mean + 1e-12))

    @staticmethod
    def _topk_share(p: torch.Tensor, k: int) -> float:
        k = max(1, min(k, p.numel()))
        return float(torch.topk(p, k).values.sum().item())

    def _build_heatmap_matrix(self) -> np.ndarray:
        """Build L x maxE array of usage probabilities; pads with zeros for layers with fewer experts."""
        H = np.zeros((self.L, self.maxE), dtype=np.float32)
        for l in range(self.L):
            e = self.E[l]
            p = self._norm_counts(self.ema_counts[l]).clamp_min(0).cpu().numpy()
            H[l, :e] = p
        # Optional gamma correction for visual contrast
        if self.gamma != 1.0 and self.gamma > 0.0:
            H = np.power(H, self.gamma, dtype=np.float32)
            vmax = H.max()
            if vmax > 0:
                H = H / vmax
        return H

    def _log_heatmap(self, step: int) -> None:
        if self.L == 0 or self.maxE == 0:
            return
        H = self._build_heatmap_matrix()
        if _HAS_MPL:
            # Size heuristics to keep readability for larger matrices
            fig_h = max(2.2, 0.35 * self.L + 1.0)
            fig_w = max(3.0, 0.5 * self.maxE + 1.2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
            im = ax.imshow(H, interpolation="nearest", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert")
            ax.set_ylabel("Layer (H-level)")
            # Manage tick density for readability
            if self.maxE <= 30:
                ax.set_xticks(range(self.maxE))
                ax.set_xticklabels(self.expert_labels, fontsize=8)
            else:
                step_x = 5 if self.maxE <= 80 else 10
                xs = list(range(0, self.maxE, step_x))
                ax.set_xticks(xs)
                # +1 in label to reflect 1-based expert labels when subsampling
                ax.set_xticklabels([f"E{x+1}" for x in xs], fontsize=8)
            ax.set_yticks(range(self.L))
            ax.set_yticklabels(self.layer_labels, fontsize=8)
            # Faint separators where layers have fewer experts than maxE
            for li, e in enumerate(self.E):
                if e < self.maxE:
                    ax.plot([e - 0.5, e - 0.5], [li - 0.5, li + 0.5], color="w", alpha=0.25, linewidth=0.6)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("usage share (0..1)", rotation=270, labelpad=8)
            ax.set_title("MoE expert usage (rows=H-level layers, cols=experts)")
            try:
                wandb.log({f"{self.prefix}/usage_heatmap": wandb.Image(fig)}, step=step)
            finally:
                plt.close(fig)
        else:
            # Fallback: compact image with explicit caption describing axes
            img_u8 = (H * 255.0 + 0.5).astype(np.uint8)
            caption = (
                f"MoE usage heatmap @ step {step}. Rows=layers (H1..H{self.L}), "
                f"Cols=experts (E1..E{self.maxE}). Values are normalized shares per layer (sum to 1)."
            )
            wandb.log({f"{self.prefix}/usage_heatmap": wandb.Image(img_u8, caption=caption)}, step=step)

    def update(
        self,
        counts_per_layer: List[torch.Tensor],
        *,
        step: Optional[int] = None,
        sync_dist: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if step is not None:
            self.step = int(step)
        else:
            self.step += 1

        # 1) Aggregate across ranks
        reduced = []
        for c in counts_per_layer:
            c = (c.detach().to("cpu", dtype=torch.float32)).clone()
            if sync_dist and world_size > 1 and dist.is_initialized():
                self._allreduce_sum_(c)
            reduced.append(c)

        # 2) EMA update
        for l in range(self.L):
            self.ema_counts[l].mul_(self.beta).add_(reduced[l], alpha=(1.0 - self.beta))

        # 3) Log on cadence from rank 0
        if (self.step % self.log_every) != 0 or rank != 0:
            return

        logs = {}
        all_probs = [] if self.make_hist else None

        for l in range(self.L):
            p = self._norm_counts(self.ema_counts[l])
            prefix = f"{self.prefix}/L{l}"
            logs[f"{prefix}/neff_frac"] = self._neff_frac(p)
            logs[f"{prefix}/entropy01"] = self._entropy01(p)
            logs[f"{prefix}/gini"] = self._gini(p)
            logs[f"{prefix}/cv"] = self._cv(p)
            logs[f"{prefix}/max_share"] = float(p.max().item())
            logs[f"{prefix}/top{self.topk_share_k}_share"] = self._topk_share(p, self.topk_share_k)
            if all_probs is not None:
                all_probs.append(p.numpy())

        if all_probs is not None and len(all_probs):
            flat = np.concatenate(all_probs, axis=0)
            counts, edges = np.histogram(flat, bins=np.linspace(0.0, 1.0, self.hist_bins + 1, dtype=np.float32))
            logs[f"{self.prefix}/usage_hist"] = wandb.Histogram(np_histogram=(counts, edges))

        try:
            wandb.log(logs, step=self.step)
        except Exception as e:
            print(f"[W&B] Skipping MoE usage metrics logging due to error: {e}")

        # 4) Labeled heatmap with colorbar (or fallback image)
        try:
            self._log_heatmap(step=self.step)
        except Exception as e:
            print(f"[W&B] Skipping MoE heatmap logging due to error: {e}")


def build_moe_usage_logger(train_state, log_every: int = 100) -> Optional[MoEUsageLogger]:
    """Create a MoEUsageLogger if H-level MoE layers exist; otherwise return None."""
    try:
        base_model = getattr(train_state.model, "model", None)
        inner = getattr(base_model, "inner", None)
        H_level = getattr(inner, "H_level", None)
        layers = getattr(H_level, "layers", None)
    except Exception:
        return None
    if layers is None:
        return None
    experts_per_layer: List[int] = []
    for layer in layers:
        moe_mlp = getattr(layer, "moe_mlp", None)
        if moe_mlp is not None and getattr(moe_mlp, "experts", None) is not None:
            experts_per_layer.append(len(moe_mlp.experts))
    if not len(experts_per_layer):
        return None
    return MoEUsageLogger(experts_per_layer=experts_per_layer, log_every=log_every)


def collect_moe_counts_per_layer(train_state) -> Optional[List[torch.Tensor]]:
    """Collect last per-expert counts for each H-level MoE layer; returns None if no MoE."""
    try:
        base_model = getattr(train_state.model, "model", None)
        inner = getattr(base_model, "inner", None)
        H_level = getattr(inner, "H_level", None)
        layers = getattr(H_level, "layers", None)
    except Exception:
        return None
    if layers is None:
        return None

    counts_list: List[torch.Tensor] = []
    for layer in layers:
        moe_mlp = getattr(layer, "moe_mlp", None)
        if moe_mlp is None or getattr(moe_mlp, "experts", None) is None:
            continue
        E = len(moe_mlp.experts)
        counts = getattr(layer, "last_counts", None)
        if counts is None:
            counts = torch.zeros(E, dtype=torch.float32)
        counts_list.append(counts)
    if not len(counts_list):
        return None
    return counts_list
