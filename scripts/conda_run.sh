#!/bin/bash
#SBATCH --constraint="80gb"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=42G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out

# or > salloc --gres=gpu:1 --constraint="80gb" --cpus-per-task=6 --mem=32G  --time=12:00:00 --nodes=1 --partition=main

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

git pull

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

# now assign positional args
DATASET="$1"
FREQ_SLOPE="$2"
EXP_NAME="$3"
SEED="$4"

echo "[DEBUG] 1=$1, 2=$2, 3=$3, 4=$4" >&2

torchrun --nproc_per_node=${GPUS_PER_NODE} pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
