#!/usr/bin/env python3
"""
Oasis RL Finetuning - Main Training Script

All configuration is loaded from config/default.yaml (single source of truth).
Command-line arguments can override specific values.

Usage:
    # Use default config
    python train.py
    
    # With custom config file
    python train.py --config config/custom.yaml
    
    # Override specific parameters
    python train.py --learning-rate 5e-5 --no-wandb
"""

import os
import sys
import argparse
from pathlib import Path
import glob

# Add the oasis-finetuning directory to sys.path for proper imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

# Import config loader
from config.loader import load_config, DEFAULT_CONFIG_PATH


def parse_args():
    """Parse command-line arguments. All values come from YAML by default."""
    parser = argparse.ArgumentParser(
        description="Oasis RL Finetuning - Config loaded from YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file (default: config/default.yaml)",
    )
    
    # Common overrides (optional - all default to None = use YAML value)
    parser.add_argument("--oasis-ckpt", type=str, default=None)
    parser.add_argument("--vae-ckpt", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # Boolean flags
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-kl", action="store_true", help="Disable KL penalty")
    parser.add_argument("--no-rik", action="store_true", help="Disable RIK reward")
    parser.add_argument("--no-rtc", action="store_true", help="Disable RTC reward")
    parser.add_argument("--no-raq", action="store_true", help="Disable RAQ reward")
    
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


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
        
    # Look for .pt files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not checkpoints:
        return None
        
    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Build overrides dict from command-line args
    overrides = {}
    if args.oasis_ckpt: overrides['oasis_ckpt'] = args.oasis_ckpt
    if args.vae_ckpt: overrides['vae_ckpt'] = args.vae_ckpt
    if args.data_dir: overrides['data_dir'] = args.data_dir
    if args.learning_rate: overrides['learning_rate'] = args.learning_rate
    if args.total_steps: overrides['total_training_steps'] = args.total_steps
    if args.group_size: overrides['group_size'] = args.group_size
    if args.reward_scale: overrides['reward_scale'] = args.reward_scale
    if args.checkpoint_dir: overrides['checkpoint_dir'] = args.checkpoint_dir
    if args.device: overrides['device'] = args.device
    
    # Boolean overrides
    if args.no_wandb: overrides['use_wandb'] = False
    if args.no_kl: overrides['use_kl_in_reward'] = False
    if args.no_rik: overrides['rik_weight'] = 0.0
    if args.no_rtc: overrides['rtc_weight'] = 0.0
    if args.no_raq: overrides['raq_weight'] = 0.0
    
    # Load config from YAML with overrides
    print("=" * 60)
    print("Oasis GRPO Finetuning with GameWorldScore Reward")
    print(f"Loading config from: {args.config}")
    print("=" * 60)
    
    config = load_config(args.config, **overrides)
    
    # Check device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
    
    # Check model files
    if not os.path.exists(config.oasis_ckpt):
        print(f"Warning: Oasis checkpoint not found at {config.oasis_ckpt}")
        print("Please download the model or provide correct path.")
    
    if not os.path.exists(config.vae_ckpt):
        print(f"Warning: VAE checkpoint not found at {config.vae_ckpt}")
        print("Please download the model or provide correct path.")
    
    # Import trainer (config already loaded from YAML)
    from trainer.oasis_grpo_trainer import OasisGRPOTrainer
    
    # Create trainer with config from YAML
    print("\nInitializing trainer...")
    trainer = OasisGRPOTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)
    elif config.auto_resume:
        # Try to find latest checkpoint
        latest_ckpt = get_latest_checkpoint(config.checkpoint_dir)
        if latest_ckpt:
            print(f"Auto-resuming from latest checkpoint: {latest_ckpt}")
            trainer.load_checkpoint(latest_ckpt)
    
    # Start training
    print("\nStarting training...")
    trainer.fit()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

