cat > run_ddp.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=hrm_ddp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o %x-%j.out

module load singularity
SIF=$SCRATCH/images/hrm_nvidia.25.08.sif
CODE=$HOME/projects/HRM
DATA=$SCRATCH/data

srun singularity exec --nv -B ${CODE}:/workspace,${DATA}:/data ${SIF} \
     bash -lc 'cd /workspace && \
       torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE ../pretrain.py \
         --data_path data/sudoku-extreme-1k-aug-1000  \
         --epochs 20000  \
         --eval_interval 2000  \
         --global_batch_size 384  \
         --lr 7e-5  \
         --puzzle_emb_lr 7e-5  \
         --weight_decay 1.0  \
         --puzzle_emb_weight_decay 1.0'
EOF

sbatch run_ddp.sbatch
