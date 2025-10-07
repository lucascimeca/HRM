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

# Compute the output directory after the SBATCH directives
#export OUTPUT_DIR=$HOME/script_outputs
# Update the output path of the SLURM output file
#export SLURM_JOB_OUTPUT=${OUTPUT_DIR}/$(basename ${SLURM_JOB_OUTPUT})

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

git pull

# Rendezvous settings: pick a unique, free port and pass it explicitly to torchrun
export MASTER_ADDR="127.0.0.1"
# Base on SLURM job id but probe forward to avoid collisions
BASE_PORT=$((10000 + ${SLURM_JOBID:-0} % 50000))
CANDIDATE_PORT=${BASE_PORT}
# probe up to 50 ports ahead
for _ in $(seq 1 50); do
  python - <<'PY'
import os, socket, sys
port = int(os.environ.get('CANDIDATE_PORT', '29500'))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
  if [[ "$?" -eq 0 ]]; then
    export MASTER_PORT=${CANDIDATE_PORT}
    break
  fi
  CANDIDATE_PORT=$((CANDIDATE_PORT + 1))
done
# Fallback if loop failed
export MASTER_PORT=${MASTER_PORT:-29500}
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

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

# Collect any additional Hydra overrides passed to this script and forward them to pretrain.py
EXTRA_OVERRIDES=("$@")

# inside your job, before launching torchrun
mkdir -p "$PWD/libcuda_shim"

# prefer a real, existing .so.1
if [ -f /usr/local/cuda/compat/lib/libcuda.so.1 ]; then
  ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 "$PWD/libcuda_shim/libcuda.so.1"
  ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 "$PWD/libcuda_shim/libcuda.so"
elif [ -f /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 "$PWD/libcuda_shim/libcuda.so.1"
  ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 "$PWD/libcuda_shim/libcuda.so"
fi

export LD_LIBRARY_PATH="$PWD/libcuda_shim:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"


torchrun --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    ../pretrain.py \
    data_path=$SCRATCH/projects/HRM/data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    "${EXTRA_OVERRIDES[@]}"
