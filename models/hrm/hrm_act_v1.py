from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Optional H-level MoE routing config (disabled by default)
    use_H_moe: bool = False
    H_moe_num_experts: int = 0
    H_moe_top_k: int = 2
    H_moe_hidden_ratio: Optional[float] = None  # unused in SOTA MoE, kept for config compatibility
    H_moe_aux_loss_weight: float = 0.01


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


# --- SOTA-style token-level MoE in FFN (Switch-style) ---
class TokenTopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = max(1, int(top_k))
        self.gate = CastedLinear(hidden_size, num_experts, bias=False)
        # Observability for logging
        self.last_importance: Optional[torch.Tensor] = None  # [E]
        self.last_load: Optional[torch.Tensor] = None        # [E]
        self.last_counts: Optional[torch.Tensor] = None      # [E]
        self.last_tokens: Optional[int] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, S, H]
        B, S, H = x.shape
        logits = self.gate(x)  # [B, S, E]
        probs = F.softmax(logits.to(torch.float32), dim=-1)  # [B, S, E]

        topk_vals, topk_idx = torch.topk(probs, k=min(self.top_k, self.num_experts), dim=-1)  # [B, S, K]
        # Renormalize over selected experts
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, S, K]

        # Aux loss (Switch): importance and load
        E = self.num_experts
        importance = probs.sum(dim=(0, 1)) / max(B * S, 1)
        sel_mask = torch.zeros((B, S, E), dtype=probs.dtype, device=probs.device)
        sel_mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_vals))
        load = sel_mask.sum(dim=(0, 1)) / max(B * S * topk_idx.shape[-1], 1)
        aux_loss = (E * (importance * load)).sum()

        # Log state
        self.last_importance = importance.detach()
        self.last_load = load.detach()
        self.last_counts = sel_mask.sum(dim=(0, 1)).detach()
        self.last_tokens = int(B * S)

        return topk_idx, topk_weights, aux_loss


class MoEMLP(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, num_experts: int, top_k: int, norm_eps: float):
        super().__init__()
        self.router = TokenTopKRouter(hidden_size, num_experts, top_k)
        # Experts are standard SwiGLU FFNs at the same hidden size (no adapters)
        self.experts = nn.ModuleList([SwiGLU(hidden_size=hidden_size, expansion=expansion) for _ in range(num_experts)])
        self.norm_eps = norm_eps
        # Expose last usage
        self.last_load: Optional[torch.Tensor] = None
        self.last_counts: Optional[torch.Tensor] = None
        self.last_tokens: Optional[int] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, S, H]
        B, S, H = x.shape
        topk_idx, topk_weights, aux_loss = self.router(x)  # [B, S, K], [B, S, K]
        E = len(self.experts)
        K = topk_idx.shape[-1]

        # Dense per-expert weights W_full: [B, S, E]
        W_full = x.new_zeros((B, S, E), dtype=topk_weights.dtype)
        W_full.scatter_add_(dim=-1, index=topk_idx, src=topk_weights)

        mixed = torch.zeros_like(x)
        x_flat = x.reshape(B * S, H)
        mixed_flat = mixed.view(B * S, H)
        W_flat = W_full.view(B * S, E)

        for e, expert in enumerate(self.experts):
            w_e = W_flat[:, e]
            sel = w_e > 0
            if sel.any():
                idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
                x_e = x_flat.index_select(dim=0, index=idx)
                y_e = expert(x_e)  # [N_e, H]
                w = w_e.index_select(0, idx).unsqueeze(-1).to(y_e.dtype)
                mixed_flat.index_add_(0, idx, y_e * w)

        out = mixed  # [B, S, H]

        # Log last usage
        self.last_load = self.router.last_load
        self.last_counts = self.router.last_counts
        self.last_tokens = self.router.last_tokens

        return out, aux_loss


class HierarchicalReasoningModel_ACTV1MoEBlock(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config):
        super().__init__()
        H = config.hidden_size
        self.self_attn = Attention(
            hidden_size=H,
            head_dim=H // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        # Expert FFN size control: effective_expert_expansion = base_expansion * H_moe_hidden_ratio (multiplier)
        # Rule-of-thumb: per-token active FFN compute ~ top_k * H_moe_hidden_ratio of the dense baseline.
        expert_mult = 1.0 if (config.H_moe_hidden_ratio is None) else float(config.H_moe_hidden_ratio)
        effective_expansion = config.expansion * expert_mult
        self.moe_mlp = MoEMLP(hidden_size=H, expansion=effective_expansion, num_experts=config.H_moe_num_experts, top_k=config.H_moe_top_k, norm_eps=config.rms_norm_eps)
        self.norm_eps = config.rms_norm_eps
        # Expose aux and usage
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_load: Optional[torch.Tensor] = None
        self.last_counts: Optional[torch.Tensor] = None
        self.last_tokens: Optional[int] = None
        self.expert_expansion_multiplier: float = expert_mult

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention + residual + norm
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # MoE MLP + residual + norm
        mlp_out, aux = self.moe_mlp(hidden_states)
        self.last_aux_loss = aux
        self.last_load = self.moe_mlp.last_load
        self.last_counts = self.moe_mlp.last_counts
        self.last_tokens = self.moe_mlp.last_tokens
        hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1MoEReasoningModule(nn.Module):
    """Drop-in replacement for H-level reasoning using MoE blocks in layers 2..N.
    Input injection is applied once at the module entrance to match baseline semantics.
    """
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config):
        super().__init__()
        self.config = config
        self.layers: nn.ModuleList = nn.ModuleList()
        if config.H_layers > 0:
            # Layer 1: standard block
            self.layers.append(HierarchicalReasoningModel_ACTV1Block(config))
        # Layers 2..N: MoE blocks
        for _ in range(max(config.H_layers - 1, 0)):
            self.layers.append(HierarchicalReasoningModel_ACTV1MoEBlock(config))
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_usage_per_layer: Optional[List[torch.Tensor]] = None

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        cos_sin = kwargs.get("cos_sin", None)
        hidden_states = hidden_states + input_injection

        total_aux = None
        usage_list: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
            if isinstance(layer, HierarchicalReasoningModel_ACTV1MoEBlock):
                aux = layer.last_aux_loss
                if aux is not None:
                    total_aux = aux if total_aux is None else (total_aux + aux)
                if layer.last_load is not None:
                    usage_list.append(layer.last_load)
                else:
                    usage_list.append(torch.zeros((layer.moe_mlp.router.num_experts,), device=hidden_states.device, dtype=torch.float32))
            else:
                if self.config.H_moe_num_experts > 0:
                    usage_list.append(torch.zeros((self.config.H_moe_num_experts,), device=hidden_states.device, dtype=torch.float32))

        # Average aux over MoE layers for stability
        num_moe_layers = sum(1 for l in self.layers if isinstance(l, HierarchicalReasoningModel_ACTV1MoEBlock))
        if total_aux is not None and num_moe_layers > 0:
            total_aux = total_aux / num_moe_layers
        self.last_aux_loss = total_aux
        self.last_usage_per_layer = usage_list if len(usage_list) > 0 else None
        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        if self.config.use_H_moe and (self.config.H_moe_num_experts > 0):
            self.H_level = HierarchicalReasoningModel_ACTV1MoEReasoningModule(self.config)
        else:
            self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        # Attach MoE aux loss and usage if present (only H-level MoE produces it)
        if getattr(self.inner, "H_level", None) is not None and hasattr(self.inner.H_level, "last_aux_loss"):
            aux = self.inner.H_level.last_aux_loss
            if aux is not None:
                outputs["moe_aux_loss"] = aux

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
