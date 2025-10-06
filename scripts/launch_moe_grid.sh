#!/bin/bash
# Submit a set of MoE ablations as individual sbatch jobs.
# Usage (from repo root or from scripts/):
#   bash scripts/launch_moe_grid.sh
# Each line launches one job via conda_run.sh with MoE overrides.
#
# Notation:
# - H_moe_hidden_ratio scales each expert's FFN expansion vs. dense baseline.
#   Rough rule: active FFN compute per MoE block ≈ top_k * H_moe_hidden_ratio of dense.
# - Choose more experts and bigger top_k for smaller experts; choose fewer experts and k=1 for larger experts.

# Resolve conda_run.sh relative to this script's directory for robustness
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SH="${SCRIPT_DIR}/conda_run.sh"
SBATCH_CHDIR=(--chdir "$SCRIPT_DIR")

# --- Small experts (ratio ~ 0.25) -> pick many ---
# Parity around k≈4 (0.25*4 ≈ 1x), explore more/less k and E
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=128 H_moe_top_k=8 H_moe_hidden_ratio=0.25 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=64  H_moe_top_k=8 H_moe_hidden_ratio=0.25 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=64  H_moe_top_k=4 H_moe_hidden_ratio=0.25 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=32  H_moe_top_k=4 H_moe_hidden_ratio=0.25 H_moe_aux_loss_weight=0.01

# --- Medium experts (ratio ~ 0.50) -> pick some ---
# Parity around k≈2 (0.5*2 ≈ 1x), explore E variety
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=64  H_moe_top_k=4 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=64  H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=32  H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=16  H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.01

# --- Normal experts (ratio ~ 1.0) -> pick one ---
# Parity at k=1 (1.0*1 ≈ 1x), explore E variety including minimal choice
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=32  H_moe_top_k=1 H_moe_hidden_ratio=1.00 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=16  H_moe_top_k=1 H_moe_hidden_ratio=1.00 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=8   H_moe_top_k=1 H_moe_hidden_ratio=1.00 H_moe_aux_loss_weight=0.01
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=2   H_moe_top_k=1 H_moe_hidden_ratio=1.00 H_moe_aux_loss_weight=0.01

# --- Auxiliary loss ablations near parity (medium size) ---
# Keep E and k fixed; vary load balancing strength
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=32  H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.0
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=32  H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.001
sbatch "${SBATCH_CHDIR[@]}" "$RUN_SH" use_H_moe=True H_moe_num_experts=32  H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.05
