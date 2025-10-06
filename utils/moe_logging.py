from typing import Optional, List
import math
import numpy as np

import torch
import torch.distributed as dist
import wandb


class MoEUsageLogger:
    """
    Low-bandwidth MoE usage logger:
      - Per-layer scalars: neff_frac, entropy01, gini, cv, max_share, topk_share
      - Optional fixed-bin histogram (no raw samples uploaded)
      - Single compact heatmap image (layers x maxE), 8-bit
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
        self.beta = float(ema_beta)
        self.topk_share_k = int(topk_share_k)
        self.log_every = int(log_every)
        self.make_hist = bool(make_histogram)
        self.hist_bins = int(hist_bins)
        self.gamma = float(image_gamma)
        self.prefix = name_prefix
        self.step = 0
        self.ema_counts = [torch.zeros(e, dtype=torch.float32) for e in self.E]
        if self.make_hist:
            self.bin_edges = np.linspace(0.0, 1.0, self.hist_bins + 1, dtype=np.float32)

    @staticmethod
    def _allreduce_sum_(t: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized():
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

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

        # Single compact heatmap (L x maxE), uint8
        maxE = max(self.E) if self.E else 0
        if maxE > 0:
            img = np.zeros((self.L, maxE), dtype=np.float32)
            for l in range(self.L):
                p = self._norm_counts(self.ema_counts[l]).numpy()
                img[l, : self.E[l]] = p
            img = np.power(img, self.gamma, dtype=np.float32)
            vmax = img.max()
            if vmax > 0:
                img /= vmax
            img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
            logs[f"{self.prefix}/usage_heatmap"] = wandb.Image(
                img_u8, caption=f"MoE usage (rows=layers, cols=experts) @ {self.step}"
            )

        try:
            wandb.log(logs, step=self.step)
        except Exception as e:
            print(f"[W&B] Skipping MoE usage logging due to error: {e}")


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

