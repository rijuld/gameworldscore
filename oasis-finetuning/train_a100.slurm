#!/bin/bash
#SBATCH --job-name=oasis_grpo
#SBATCH --account=csci_ga_3033-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Run training inside Singularity container
singularity exec --bind /scratch --nv \
    --overlay /scratch/rd3629/overlay-25GB-500K.ext3:ro \
    /scratch/rd3629/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate bdml_env

# Navigate to project directory
cd /scratch/rd3629/oasis-finetuning

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
python train.py
"
