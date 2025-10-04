#!/bin/bash
# Submit a set of MoE ablations as individual sbatch jobs.
# Usage (from repo root):
#   bash scripts/launch_moe_grid.sh
# Each line below launches one job via scripts/conda_run.sh with MoE overrides.

# Balanced parity (E=64, K=4, r=0.42, aux=0.01)
sbatch conda_run.sh use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.01

# More smaller experts (parity)
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=128 H_moe_top_k=8 H_moe_hidden_ratio=0.28 H_moe_aux_loss_weight=0.01

# Fewer larger experts (parity)
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=32 H_moe_top_k=2 H_moe_hidden_ratio=0.63 H_moe_aux_loss_weight=0.01

# Under-parity variants (~80% of parity r)
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.33 H_moe_aux_loss_weight=0.01
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=32 H_moe_top_k=2 H_moe_hidden_ratio=0.50 H_moe_aux_loss_weight=0.01
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=128 H_moe_top_k=8 H_moe_hidden_ratio=0.22 H_moe_aux_loss_weight=0.01

# Aux load balancing ablations (around parity)
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.0
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.001
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=64 H_moe_top_k=4 H_moe_hidden_ratio=0.42 H_moe_aux_loss_weight=0.05

# Extreme under-parameterized (your example)
sbatch scripts/conda_run.sh use_H_moe=True H_moe_num_experts=50 H_moe_top_k=20 H_moe_hidden_ratio=0.01 H_moe_aux_loss_weight=0.01
