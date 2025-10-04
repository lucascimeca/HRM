#!/bin/bash -l
#SBATCH --job-name=hrm_ddp
#SBATCH --constraint="80gb"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=42G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --partition=main

# or > salloc --gres=gpu:1 --constraint="80gb" --cpus-per-task=6 --mem=32G  --time=12:00:00 --nodes=1 --partition=main

set -euo pipefail
# optional: echo commands as they run
# set -x

# Resolve repository root
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Prefer the directory where sbatch was invoked
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  # Try git toplevel from the script dir; else fallback to parent of script dir
  if GIT_TOP=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
    REPO_ROOT="$GIT_TOP"
  else
    REPO_ROOT="${SCRIPT_DIR}/.."
  fi
fi

# If pretrain.py is not in REPO_ROOT, try parent (handles sbatch from scripts/)
if [[ ! -f "$REPO_ROOT/pretrain.py" && -d "$REPO_ROOT/.." && -f "$REPO_ROOT/../pretrain.py" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi

# Ensure we run from the repo root for relative paths (e.g., checkpoints/)
cd "$REPO_ROOT"

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "Repo root: $REPO_ROOT"

# Ensure only anaconda/3 module loaded.
module --quiet purge || true
# This example uses Conda to manage package dependencies.
module load anaconda/3 || true
module load cuda/11.8 || true

# Activate pre-existing environment.
conda activate pytorch

# Prefer pure PyTorch AdamAtan2 implementation to avoid CUDA extension incompatibilities on clusters
export FORCE_PY_ADAM_ATAN2=1

# Optional: cap threads per worker unless overridden
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID:-$$} | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES || true

# Determine local world size = # GPUs on this node
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
# Pick a writable place
SHIM_BASE="${SLURM_TMPDIR:-$PWD}"
SHIM_DIR="${SHIM_BASE}/libcuda_shim"
mkdir -p "$SHIM_DIR"

# Link a real libcuda if present (workaround for library path issues on some clusters)
if [ -f /usr/local/cuda/compat/lib/libcuda.so.1 ]; then
  ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 "$SHIM_DIR/libcuda.so.1"
  ln -sf /usr/local/cuda/compat/lib/libcuda.so "$SHIM_DIR/libcuda.so"
elif [ -f /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 "$SHIM_DIR/libcuda.so.1"
  ln -sf /lib/x86_64-linux-gnu/libcuda.so "$SHIM_DIR/libcuda.so"
fi

export LD_LIBRARY_PATH="$SHIM_DIR:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Default overrides (can be overridden by passing hydra args to this script)
DEFAULT_OVERRIDES=(
    data_path=$SCRATCH/projects/HRM/data/sudoku-extreme-1k-aug-1000
    epochs=20000
    eval_interval=2000
    global_batch_size=384
    lr=7e-5
    puzzle_emb_lr=7e-5
    weight_decay=1.0
    puzzle_emb_weight_decay=1.0
)

# Forward any additional overrides (e.g., use_H_moe, H_moe_*). Caller can pass: use_H_moe=True H_moe_num_experts=64 ...
EXTRA_OVERRIDES=("$@")

# Use absolute path to pretrain.py to avoid torchrun running in /var/spool/slurmd
PRETRAIN_ABS="$REPO_ROOT/pretrain.py"
if [[ ! -f "$PRETRAIN_ABS" ]]; then
  echo "ERROR: pretrain.py not found at $PRETRAIN_ABS" >&2
  exit 1
fi

# Launch
exec torchrun --nproc_per_node=${GPUS_PER_NODE} "$PRETRAIN_ABS" \
    "${DEFAULT_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
