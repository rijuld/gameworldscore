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

conda activate my_new_env

# Navigate to project directory
cd ~/../../scratch/rd3629/RL-Project/oasis-finetuning

# Run training
python train.py
"
