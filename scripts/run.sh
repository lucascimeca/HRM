#!/bin/bash
#SBATCH --job-name=hrm_ddp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --constraint="40gb|48gb|80gb"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o %x-%j.out

set -euo pipefail

# 1) load singularity (Mila shows singularity/3.7.1 in your logs)
module load singularity/3.7.1

# 2) paths (EDIT these to your actual paths)
SIF="$SCRATCH/images/hrm_nvidia.25.08.sif"    # your .sif image
CODE="$HOME/projects/HRM"                  # your repo directory on the cluster
DATA="$SCRATCH/data"                       # datasets/checkpoints/output directory

# 3) optional: forward some env vars into the container (W&B, etc.)
#    Any var prefixed with SINGULARITYENV_ shows up inside the container.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"
fi
export SINGULARITYENV_WANDB_PROJECT="hrm"
export SINGULARITYENV_WANDB_DIR="/workspace/wandb"
export SINGULARITYENV_TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0  # optional: silence GEMM autotune warning

# 4) determine local world size = # GPUs on this node
#    Prefer Slurm's view; fallback to nvidia-smi count if unset.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
# check it's a positive integer
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$GPUS_PER_NODE" -lt 1 ]]; then
  echo "ERROR: could not determine GPUS_PER_NODE (got: '$GPUS_PER_NODE')" >&2
  exit 1
fi

echo "GPUS_PER_NODE=${GPUS_PER_NODE}"

# 5) run inside the container with GPU support (--nv) and bind mounts (-B)
#    Use srun so Slurm tracks the task.
srun singularity exec --nv \
     -B "${CODE}:/workspace","${DATA}:/data" \
     "$SIF" \
     bash -lc "
       set -euo pipefail
       cd /workspace
       echo 'Launching torchrun with --nproc_per_node=${GPUS_PER_NODE}'
       torchrun --nproc_per_node=${GPUS_PER_NODE} pretrain.py --data /data
     "

