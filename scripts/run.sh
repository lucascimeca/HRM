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
CODE="$SCRATCH/projects/HRM"                  # your repo directory on the cluster
DATA="$SCRATCH/projects/HRM/dataset"                       # datasets/checkpoints/output directory

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
export SINGULARITYENV_GPUS_PER_NODE="${GPUS_PER_NODE}"

srun singularity exec --nv \
  -B "${CODE}:/workspace","${DATA}:/data" \
  "$SIF" \
  bash -lc '
    set -euo pipefail

    # 1) Locate the driver .so that --nv provides
    CANDIDATES=( "/.singularity.d/libs" "/usr/local/cuda/compat/lib" "/usr/local/nvidia/lib64" )
    CUDA_DIR=""
    for d in "${CANDIDATES[@]}"; do
      if [ -r "$d/libcuda.so.1" ]; then CUDA_DIR="$d"; break; fi
    done
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"
    echo "libcuda search hit: ${CUDA_DIR:-<none>}"

    # 2) Create a shim in a writable place (your bind mount)
    if [ -z "${CUDA_DIR}" ]; then
      echo "ERROR: libcuda.so.1 not found; disabling Inductor to proceed." >&2
      export TORCHDYNAMO_DISABLE=1
    else
      SHIM_DIR="/workspace/.libcuda_shim"
      mkdir -p "$SHIM_DIR"
      ln -sf "${CUDA_DIR}/libcuda.so.1" "${SHIM_DIR}/libcuda.so"
      export LD_LIBRARY_PATH="${SHIM_DIR}:${CUDA_DIR}:${LD_LIBRARY_PATH-}"
      echo "Shim at ${SHIM_DIR}/libcuda.so -> ${CUDA_DIR}/libcuda.so.1"
    fi

    # Optional: quieter warnings + set W&B dir
    export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
    export WANDB_DIR=/workspace/wandb

    # 3) Quick sanity
    python - << "PY"
import ctypes, os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
ctypes.CDLL("libcuda.so")
print("CDLL(libcuda.so) OK")
PY

    cd /workspace
    echo "Launching torchrun (GPUs per node = ${GPUS_PER_NODE})"
    torchrun --nproc_per_node=${GPUS_PER_NODE} pretrain.py \
      data_path=data/sudoku-extreme-1k-aug-1000 \
      epochs=20000 eval_interval=2000 global_batch_size=384 \
      lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
  '




