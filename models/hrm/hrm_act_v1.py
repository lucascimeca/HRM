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


class MoEConfig(BaseModel):
    batch_size: int
    seq_len: int
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

    puzzle_emb_ndim: int = 0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Configuration for DeepSeekMoE
    num_shared_experts: int = 2
    num_routed_experts: int = 160
    num_experts_per_tok: int = 6  # K_r in the paper
    expert_intermediate_size: int = 1536
    num_devices: int = 8
    max_devices_per_token: int = 3  # M in the paper

    # Balance loss coefficients
    expert_balance_factor: float = 0.003  # α1
    device_balance_factor: float = 0.05  # α2
    comm_balance_factor: float = 0.02  # α3

    # Other configs from your existing Block
    # num_heads already declared above
    use_token_dropping: bool = True
    capacity_factor: float = 1.0


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: MoEConfig, moe:bool=False) -> None:
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

        self.moe = None
        if moe:
            self.moe = DeepSeekMoE(config)

        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)

        if self.moe is None:
            # Fully Connected
            hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
            aux_losses = None
        else:
            # DeepSeekMoE with post-norm
            moe_output, aux_losses = self.moe(hidden_states)
            hidden_states = rms_norm(hidden_states + moe_output, variance_epsilon=self.norm_eps)

        return hidden_states, aux_losses


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input injection (add)
        hidden_states = hidden_states + input_injection

        # Layers
        aux_loss_agg = 0.0
        for layer in self.layers:
            # Each layer returns (hidden_states_after_layer, aux_losses)
            hidden_states, aux_losses = layer(hidden_states=hidden_states, **kwargs)
            if aux_losses is not None:
                # aux_losses expected to be a dict with key 'total_aux_loss'
                aux_loss_agg = aux_loss_agg + aux_losses['total_aux_loss']

        return hidden_states, aux_loss_agg


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
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
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config, moe=_i > 0) for _i in range(self.config.H_layers)])  # add moe layers
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        # Register initial state vectors as proper buffers so they move with the module/device
        self.register_buffer('H_init', trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.register_buffer('L_init', trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

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

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
                        z_L, _ = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H, _ = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L, _ = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H, aux_loss = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, aux_loss, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = MoEConfig(**config_dict)
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
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor], torch.Tensor]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, aux_loss, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

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

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs, aux_loss




# ------------------------------------------------------


class SharedExperts(nn.Module):
    """Shared experts that are always activated"""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.experts = nn.ModuleList([
            SwiGLU(config.hidden_size, config.expansion)
            for _ in range(config.num_shared_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Sum outputs from all shared experts
        output = torch.zeros_like(hidden_states)
        for expert in self.experts:
            output = output + expert(hidden_states)
        return output / len(self.experts)


class RoutedExperts(nn.Module):
    """Routed experts with top-K selection"""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.max_devices_per_token = config.max_devices_per_token
        self.num_devices = config.num_devices

        # Expert FFN size: allow a separate intermediate size for routed experts
        expert_expansion_ratio = max(1.0, float(config.expert_intermediate_size) / float(config.hidden_size))

        # Create expert modules
        self.experts = nn.ModuleList([
            SwiGLU(config.hidden_size, expert_expansion_ratio)
            for _ in range(self.num_experts)
        ])

        # Router: expert centroids for computing affinity scores
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        # Initialize router with small weights for stability
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)

        # Ensure router dtype matches model dtype if specified
        if hasattr(config, 'dtype'):
            self.router = self.router.to(dtype=config.dtype)

        # Device assignment: evenly distribute experts to devices (robust to remainders)
        # Example: E experts across D devices -> device id = expert_idx % D
        self.expert_to_device = torch.arange(self.num_experts) % max(1, self.num_devices)

    def compute_routing_scores(
            self,
            hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing scores with device-limited routing (with safe fallbacks).

        Returns:
            expert_indices: [batch_size, seq_len, num_experts_per_tok]
            expert_weights: [batch_size, seq_len, num_experts_per_tok]
            router_logits: [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Router logits in float32 for numerical stability, then clamp to avoid overflow in softmax
        router_weight = self.router.weight.to(torch.float32)
        router_logits = F.linear(hidden_states.to(torch.float32), router_weight)  # [B, S, E]
        router_logits = router_logits.clamp(-20.0, 20.0)
        routing_scores = F.softmax(router_logits, dim=-1)  # float32

        # Decide whether to apply device-limited routing
        apply_device_limit = self.max_devices_per_token < self.num_devices
        if apply_device_limit:
            # Lower-bound of experts per selected device (some devices may have one extra expert)
            min_experts_per_device = self.num_experts // max(1, self.num_devices)
            # If not enough experts would be available under device limit, skip masking to keep K valid
            if self.max_devices_per_token * max(1, min_experts_per_device) < self.num_experts_per_tok:
                apply_device_limit = False

        if apply_device_limit:
            # Device-limited routing (VECTORIZED)
            expert_to_device = self.expert_to_device.to(hidden_states.device)

            # Aggregate scores per device using scatter_add
            device_scores = torch.zeros(
                batch_size, seq_len, self.num_devices,
                device=hidden_states.device,
                dtype=routing_scores.dtype
            )
            device_indices = expert_to_device.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_len, -1
            )
            device_scores.scatter_add_(
                dim=2,
                index=device_indices,
                src=routing_scores
            )

            # Select top M devices per token
            _, selected_devices = torch.topk(
                device_scores,
                k=self.max_devices_per_token,
                dim=-1
            )  # [B, S, M]

            # Create expert mask: True if expert's device is selected
            expert_devices_expanded = expert_to_device.unsqueeze(0).unsqueeze(0)
            selected_devices_expanded = selected_devices.unsqueeze(-1)
            device_mask = (expert_devices_expanded == selected_devices_expanded).any(dim=2)  # [B, S, E]

            # Mask out experts not on selected devices
            masked_scores = routing_scores.masked_fill(~device_mask, float('-inf'))
        else:
            masked_scores = routing_scores

        # Top-K experts among allowed experts
        # Note: if all masked (shouldn't happen), we will fallback below.
        expert_weights, expert_indices = torch.topk(
            masked_scores,
            k=min(self.num_experts_per_tok, self.num_experts),
            dim=-1
        )  # [B, S, K]

        # Fallback if a token had no finite scores after masking (all -inf)
        finite_mask = torch.isfinite(expert_weights)
        any_finite = finite_mask.any(dim=-1, keepdim=True)  # [B, S, 1]
        if not torch.all(any_finite):
            # Recompute using unmasked scores for the affected tokens
            fallback_scores = routing_scores  # [B, S, E]
            fw, fi = torch.topk(
                fallback_scores,
                k=min(self.num_experts_per_tok, self.num_experts),
                dim=-1
            )
            # Replace only where none finite
            replace_mask = ~any_finite
            expert_weights = torch.where(replace_mask, fw, expert_weights)
            expert_indices = torch.where(replace_mask, fi, expert_indices)

        # Stable softmax on selected weights (float32)
        w = expert_weights.to(torch.float32)
        w = w - w.amax(dim=-1, keepdim=True)
        exp_w = torch.exp(w)
        denom = exp_w.sum(dim=-1, keepdim=True) + 1e-9
        expert_weights = (exp_w / denom).to(hidden_states.dtype)

        return expert_indices, expert_weights, router_logits.to(hidden_states.dtype)

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with top-K routing.

        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss_dict: Dictionary containing auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get routing decisions
        expert_indices, expert_weights, router_logits = self.compute_routing_scores(hidden_states)

        # Flatten for easier processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*S, H]
        expert_indices_flat = expert_indices.view(-1, self.num_experts_per_tok)  # [B*S, K]
        expert_weights_flat = expert_weights.view(-1, self.num_experts_per_tok)  # [B*S, K]

        # Initialize output
        output = torch.zeros_like(hidden_states_flat)  # [B*S, H]

        # Process tokens per expert
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices_flat == expert_idx)  # [B*S, K]
            if not expert_mask.any():
                continue

            token_indices, k_indices = torch.where(expert_mask)
            expert_inputs = hidden_states_flat[token_indices]  # [N, H]
            expert_weights_batch = expert_weights_flat[token_indices, k_indices]  # [N]

            expert_outputs = self.experts[expert_idx](expert_inputs)  # [N, H]
            weighted_outputs = expert_outputs * expert_weights_batch.unsqueeze(-1)

            output.index_add_(0, token_indices, weighted_outputs)

        output = output.view(batch_size, seq_len, hidden_size)

        # Compute auxiliary losses for load balancing (use float32 for stability)
        aux_losses = self._compute_auxiliary_losses(
            router_logits.to(torch.float32),
            expert_indices,
            expert_weights.to(torch.float32)
        )

        return output, aux_losses

    def _compute_auxiliary_losses(
            self,
            router_logits: torch.Tensor,
            expert_indices: torch.Tensor,
            expert_weights: torch.Tensor
    ) -> dict:
        """Compute the three auxiliary losses with basic stability tweaks"""
        batch_size, seq_len, num_experts = router_logits.shape
        total_tokens = max(1, batch_size * seq_len)

        routing_probs = F.softmax(router_logits.clamp(-20.0, 20.0), dim=-1)

        # 1. Expert-Level Balance Loss
        expert_mask = F.one_hot(expert_indices, num_classes=num_experts).to(router_logits.dtype)
        f_i = expert_mask.sum(dim=[0, 1, 2]) / (total_tokens * max(1, self.num_experts_per_tok))
        P_i = routing_probs.mean(dim=[0, 1])
        expert_balance_loss = (f_i * P_i).sum() * self.config.expert_balance_factor

        # 2. Device-Level Balance Loss
        device_assignment = self.expert_to_device.to(router_logits.device)
        device_f = torch.zeros(self.num_devices, device=router_logits.device, dtype=router_logits.dtype)
        device_P = torch.zeros(self.num_devices, device=router_logits.device, dtype=router_logits.dtype)

        for device_id in range(self.num_devices):
            device_experts = (device_assignment == device_id)
            experts_on_device = int(device_experts.sum().item())
            if experts_on_device > 0:
                device_f[device_id] = f_i[device_experts].mean()
                device_P[device_id] = P_i[device_experts].mean() * experts_on_device

        device_balance_loss = (device_f * device_P).sum() * self.config.device_balance_factor

        # 3. Communication Balance Loss
        device_token_counts = torch.zeros(self.num_devices, device=router_logits.device, dtype=router_logits.dtype)
        for device_id in range(self.num_devices):
            device_experts = (device_assignment == device_id)
            if device_experts.any():
                device_expert_mask = expert_mask[:, :, :, device_experts].sum(dim=-1).clamp(min=0, max=1)
                device_token_counts[device_id] = device_expert_mask.sum()

        f_comm = device_token_counts / (total_tokens * max(1, self.max_devices_per_token))
        comm_balance_loss = (f_comm * device_P).sum() * self.config.comm_balance_factor

        total_aux = expert_balance_loss + device_balance_loss + comm_balance_loss
        return {
            'expert_balance_loss': expert_balance_loss,
            'device_balance_loss': device_balance_loss,
            'comm_balance_loss': comm_balance_loss,
            'total_aux_loss': total_aux
        }


class DeepSeekMoE(nn.Module):
    """Complete DeepSeekMoE module combining shared and routed experts"""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.shared_experts = SharedExperts(config)
        self.routed_experts = RoutedExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass combining shared and routed experts.

        Returns the MoE transformation WITHOUT residual connection.
        The residual is handled by the Block layer for consistency with standard FFN.

        output = Σ FFN_shared(u_t) + Σ g_i,t * FFN_routed(u_t)
        """
        # Shared experts (always activated)
        shared_output = self.shared_experts(hidden_states)

        # Routed experts (sparse activation)
        routed_output, aux_losses = self.routed_experts(hidden_states)

        # Combine WITHOUT residual (Block layer handles residual + norm)
        output = shared_output + routed_output

        return output, aux_losses


class MoEBlock(nn.Module):
    """
    Transformer block with DeepSeekMoE replacing the standard FFN.

    Architecture:
        Input -> RMSNorm -> Attention -> Residual
                                      -> RMSNorm -> DeepSeekMoE -> Residual -> Output
    """

    def __init__(self, config: MoEConfig, attention_module):
        super().__init__()
        self.self_attn = attention_module
        self.moe = DeepSeekMoE(config)
        self.norm_eps = config.rms_norm_eps

    def rms_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """RMS normalization"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.norm_eps)
        return hidden_states.to(input_dtype)

    def forward(
            self,
            cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
            hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with post-norm architecture.

        Returns:
            output: Transformed hidden states
            aux_losses: Dictionary of auxiliary losses for training
        """
        # Self Attention with post-norm
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = self.rms_norm(hidden_states + attn_output)

        # DeepSeekMoE with post-norm (add residual then normalize)
        moe_output, aux_losses = self.moe(hidden_states)
        hidden_states = self.rms_norm(hidden_states + moe_output)

        return hidden_states, aux_losses
