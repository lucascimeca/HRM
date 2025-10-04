#!/bin/bash -l
#SBATCH --job-name=hrm_ddp
#SBATCH --constraint="80gb"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=42G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out

# or > salloc --gres=gpu:1 --constraint="80gb" --cpus-per-task=6 --mem=32G  --time=12:00:00 --nodes=1 --partition=main
cd "${SLURM_SUBMIT_DIR:-$HOME}"  # fallback to $HOME if unset

# Compute the output directory after the SBATCH directives
export OUTPUT_DIR=$HOME/script_outputs
# Update the output path of the SLURM output file
export SLURM_JOB_OUTPUT=${OUTPUT_DIR}/$(basename ${SLURM_JOB_OUTPUT})

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.8

# Activate pre-existing environment.
conda activate pytorch

#git pull

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

set -euo pipefail
# optional: echo commands as they run
# set -x

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
# Pick a writable place
SHIM_BASE="${SLURM_TMPDIR:-$PWD}"
SHIM_DIR="${SHIM_BASE}/libcuda_shim"
mkdir -p "$SHIM_DIR"

# Link a real libcuda if present
if [ -f /usr/local/cuda/compat/lib/libcuda.so.1 ]; then
  ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 "$SHIM_DIR/libcuda.so.1"
  ln -sf /usr/local/cuda/compat/lib/libcuda.so"
elif [ -f /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 "$SHIM_DIR/libcuda.so.1"
  ln -sf /lib/x86_64-linux-gnu/libcuda.so"
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

# Launch
torchrun --nproc_per_node=${GPUS_PER_NODE} ${SLURM_SUBMIT_DIR}/../pretrain.py \
    "${DEFAULT_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
