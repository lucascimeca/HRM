#!/bin/bash
# Submit a grid of MoE ablations using sbatch and scripts/conda_run.sh
# Usage:
#   bash scripts/launch_moe_grid.sh
# Optional:
#   export PARTITION=main GPUS=2 TIME=24:00:00 to override SBATCH defaults in conda_run.sh if you clone/edit it.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
JOB_SCRIPT="${SCRIPT_DIR}/conda_run.sh"

# Only require the script file to exist; sbatch doesn't need it to be executable
if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "ERROR: $JOB_SCRIPT not found" >&2
  exit 1
fi

# Helper to normalize float token for run names (replace '.' with 'p')
_norm() {
  echo "$1" | sed -e 's/\./p/g'
}

# Define experiments: name and overrides
# Notes:
# - Parity configs roughly match active params to non-MoE baseline.
# - Under-parity use smaller hidden_ratio than parity to reduce active params.
# - Aux ablations vary load balancing strength.
# - Extreme case matches your example (very small active params).

declare -a EXPERIMENTS=(
  # Balanced parity (E=64, K=4, r=0.42, aux=0.01)
  "E64_K4_r0.42_aux0.01 use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.01"
  # More smaller experts (parity)
  "E128_K8_r0.28_aux0.01 use_H_moe=True H_moe_num_experts=128 H_moe_top_k=8 H_moe_hidden_ratio=0.28 H_moe_aux_loss_weight=0.01"
  # Fewer larger experts (parity)
  "E32_K2_r0.63_aux0.01 use_H_moe=True H_moe_num_experts=32 H_moe_top_k=2 H_moe_hidden_ratio=0.63 H_moe_aux_loss_weight=0.01"
  # Under-parity variants (~80% of parity r)
  "E64_K4_r0.33_aux0.01 use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.33 H_moe_aux_loss_weight=0.01"
  "E32_K2_r0.50_aux0.01 use_H_moe=True H_moe_num_experts=32 H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.01"
  "E128_K8_r0.22_aux0.01 use_H_moe=True H_moe_num_experts=128 H_moe_top_k=8 H_moe_hidden_ratio=0.22 H_moe_aux_loss_weight=0.01"
  # Aux load balancing ablations (around parity)
  "E64_K4_r0.42_aux0.00 use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.0"
  "E64_K4_r0.42_aux0.001 use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.001"
  "E64_K4_r0.42_aux0.05 use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.05"
  # Extreme under-parameterized (your example)
  "E50_K20_r0.01_aux0.01 use_H_moe=True H_moe_num_experts=50 H_moe_top_k=20 H_moe_hidden_ratio=0.01 H_moe_aux_loss_weight=0.01"
)

for exp in "${EXPERIMENTS[@]}"; do
  name="${exp%% *}"              # first token before space
  overrides="${exp#* }"         # rest after first space
  run_name="MoE_${name}"
  echo "Submitting: ${run_name} :: ${overrides}"

  # Forward a deterministic run_name so runs are labeled by config
  sbatch "$JOB_SCRIPT" $overrides
  # Add a tiny sleep to avoid hammering the scheduler
  sleep 0.5
done

echo "All jobs submitted. Use 'squeue -u $USER' to monitor."
