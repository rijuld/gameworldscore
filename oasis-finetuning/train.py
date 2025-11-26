#!/usr/bin/env python3
"""
Oasis RL Finetuning - Main Training Script

This script provides the entry point for RL finetuning of the Oasis world model
using GameWorldScore reward (ground-truth-free).

Usage:
    python train.py --oasis-ckpt path/to/oasis.safetensors --vae-ckpt path/to/vae.safetensors
    
    # With custom config
    python train.py --config config/custom.yaml
    
    # Override specific parameters
    python train.py --oasis-ckpt oasis.safetensors --learning-rate 5e-6 --use-wandb

Example:
    python train.py \
        --oasis-ckpt oasis500m.safetensors \
        --vae-ckpt vit-l-20.safetensors \
        --data-dir open-oasis/sample_data \
        --total-steps 10000 \
        --use-wandb
"""

import os
import sys
import argparse
from pathlib import Path
from pprint import pprint

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oasis RL Finetuning with GameWorldScore Reward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model paths
    parser.add_argument(
        "--oasis-ckpt",
        type=str,
        default="checkpoints/oasis500m.safetensors",
        help="Path to Oasis DiT checkpoint",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="checkpoints/vit-l-20.safetensors",
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--reward-models-dir",
        type=str,
        default="models_for_rl_finetuning",
        help="Directory containing reward model checkpoints",
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Dataset/MiDaS-60_small",
        help="Directory containing training data (default: Dataset/MiDaS-60_small)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["auto", "video", "midas"],
        help="Dataset type: auto (detect), video (mp4 files), midas (image folders)",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=360,
        help="Target frame height",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Target frame width",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--n-prompt-frames",
        type=int,
        default=1,
        help="Number of prompt frames",
    )
    parser.add_argument(
        "--max-gen-frames",
        type=int,
        default=31,
        help="Maximum frames to generate per rollout",
    )
    
    # Training
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000,
        help="Total training steps",
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=10,
        help="Total training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=1,
        help="PPO update epochs per step",
    )
    parser.add_argument(
        "--clip-ratio",
        type=float,
        default=0.2,
        help="PPO clip ratio",
    )
    
    # Reward weights
    parser.add_argument(
        "--rik-weight",
        type=float,
        default=1.0,
        help="Weight for RIK (action fidelity) reward",
    )
    parser.add_argument(
        "--rtc-weight",
        type=float,
        default=1.0,
        help="Weight for RTC (temporal consistency) reward",
    )
    parser.add_argument(
        "--raq-weight",
        type=float,
        default=1.0,
        help="Weight for RAQ (aesthetic quality) reward",
    )
    
    # KL regularization
    parser.add_argument(
        "--use-kl",
        action="store_true",
        default=True,
        help="Use KL penalty in reward",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.001,
        help="KL penalty coefficient",
    )
    
    # Advantage estimation
    parser.add_argument(
        "--adv-estimator",
        type=str,
        default="grpo",
        choices=["gae", "grpo", "reinforce_plus_plus", "rloo"],
        help="Advantage estimator type",
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/oasis_ppo",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Checkpoint save frequency (steps)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    
    # Logging
    parser.add_argument(
        "--project-name",
        type=str,
        default="oasis_rl_finetuning",
        help="W&B project name",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="gameworldscore",
        help="Experiment name",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides defaults)",
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print("=" * 60)
    print("Oasis RL Finetuning with GameWorldScore Reward")
    print("=" * 60)
    print("\nConfiguration:")
    pprint(vars(args))
    print()
    
    # Check model files
    if not os.path.exists(args.oasis_ckpt):
        print(f"Warning: Oasis checkpoint not found at {args.oasis_ckpt}")
        print("Please download the model or provide correct path.")
    
    if not os.path.exists(args.vae_ckpt):
        print(f"Warning: VAE checkpoint not found at {args.vae_ckpt}")
        print("Please download the model or provide correct path.")
    
    # Import trainer
    from trainer.oasis_ppo_trainer import OasisPPOConfig, OasisPPOTrainer
    
    # Create config
    config = OasisPPOConfig(
        oasis_ckpt=args.oasis_ckpt,
        vae_ckpt=args.vae_ckpt,
        reward_models_dir=args.reward_models_dir,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        frame_size=(args.frame_height, args.frame_width),
        total_epochs=args.total_epochs,
        total_training_steps=args.total_steps,
        train_batch_size=args.batch_size,
        n_prompt_frames=args.n_prompt_frames,
        max_gen_frames=args.max_gen_frames,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        clip_ratio=args.clip_ratio,
        rik_weight=args.rik_weight,
        rtc_weight=args.rtc_weight,
        raq_weight=args.raq_weight,
        use_kl_in_reward=args.use_kl and not args.no_wandb,
        kl_coeff=args.kl_coeff,
        adv_estimator=args.adv_estimator,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb and not args.no_wandb,
        device=args.device,
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = OasisPPOTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    print("\nStarting training...")
    trainer.fit()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

