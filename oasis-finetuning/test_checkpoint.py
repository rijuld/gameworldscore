#!/usr/bin/env python3
"""
Test script for evaluating the most recent checkpoint on test data.

This script:
- Finds the most recent checkpoint in the checkpoint directory
- Loads the checkpoint into the trained model
- Evaluates on test data
- Computes GameWorldScore rewards and other metrics
- Saves sample videos and evaluation results

Usage:
    # Test most recent checkpoint with default config
    python test_checkpoint.py

    # Test specific checkpoint
    python test_checkpoint.py --checkpoint checkpoints/oasis_grpo/step_1000

    # Test with custom config
    python test_checkpoint.py --config config/custom.yaml

    # Limit number of test samples
    python test_checkpoint.py --max-samples 50

    # Save videos
    python test_checkpoint.py --save-videos
"""

import os
import sys
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add the oasis-finetuning directory to sys.path for proper imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config.loader import load_config, DEFAULT_CONFIG_PATH
from models.oasis_policy import OasisPolicy
from rewards.game_world_score import create_game_world_score_reward
from data.minecraft_dataset import create_minecraft_dataloader
from utils.video_utils import frames_to_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the most recent checkpoint on test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to specific checkpoint directory or file. If not provided, uses most recent.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Maximum number of test samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--save-videos", action="store_true",
        help="Save generated videos to output directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="test_results",
        help="Directory to save test results and videos",
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir: Root directory containing checkpoint subdirectories
        
    Returns:
        Path to the most recent checkpoint directory, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint directories (step_*)
    checkpoint_dirs = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            checkpoint_file = item / "checkpoint.pt"
            if checkpoint_file.exists():
                checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        # Also check for .pt files directly in checkpoint_dir
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            # Return the directory containing the most recent file
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(checkpoint_files[0].parent)
        return None
    
    # Sort by modification time (most recent first)
    checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoint_dirs[0])


def load_checkpoint_into_policy(
    policy: OasisPolicy,
    checkpoint_path: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load checkpoint into policy model.
    
    Args:
        policy: OasisPolicy instance
        checkpoint_path: Path to checkpoint file or directory
        device: Device to load on
        
    Returns:
        Dictionary with checkpoint metadata
    """
    # Handle both file path and directory path
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    policy.dit.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded model state")
    
    # Extract metadata
    metadata = {
        'global_step': checkpoint.get('global_step', 0),
        'checkpoint_path': checkpoint_path,
    }
    
    if 'config' in checkpoint:
        metadata['config'] = checkpoint['config']
    
    print(f"  ✓ Checkpoint from step {metadata['global_step']}")
    
    return metadata


@torch.no_grad()
def evaluate_checkpoint(
    config,
    checkpoint_path: str,
    max_samples: int = 0,
    save_videos: bool = False,
    output_dir: str = "test_results",
) -> Dict[str, Any]:
    """
    Evaluate a checkpoint on test data.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint
        max_samples: Maximum number of samples to evaluate (0 = all)
        save_videos: Whether to save generated videos
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = config.device
    if device == "cuda" and (not torch.cuda.is_available()):
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_dir = output_path / "videos"
    if save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Initializing Model and Reward Function")
    print("="*60)
    
    # Initialize policy
    print("Loading Oasis policy...")
    policy = OasisPolicy(
        oasis_ckpt=config.oasis_ckpt,
        vae_ckpt=config.vae_ckpt,
        device=device,
        ddim_steps=config.ddim_steps,
    )
    policy.eval_mode()
    
    # Load checkpoint
    checkpoint_metadata = load_checkpoint_into_policy(policy, checkpoint_path, device)
    
    # Initialize reward function
    print("\nInitializing GameWorldScore reward...")
    reward_fn = create_game_world_score_reward(
        models_dir=config.reward_models_dir,
        device=device,
        rik_weight=config.rik_weight,
        rtc_weight=config.rtc_weight,
        raq_weight=config.raq_weight,
        rrg_weight=config.rrg_weight,
        anti_drift_weight=config.anti_drift_weight,
        require_vpt=config.require_vpt,
    )
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    clip_length = config.n_prompt_frames + config.max_gen_frames
    
    # Get num_workers from config (may be dataloader_num_workers or num_workers)
    num_workers = getattr(config, 'dataloader_num_workers', getattr(config, 'num_workers', 4))
    
    test_dataloader = create_minecraft_dataloader(
        data_dir=str(config.data_dir),
        batch_size=1,  # Use batch size 1 for testing
        clip_length=clip_length,
        dataset_type=config.dataset_type,
        frame_size=(config.frame_height, config.frame_width),
        split="test",  # Use test split
        num_workers=num_workers,
    )
    
    if test_dataloader is None:
        print("Warning: Test dataloader is None. Trying 'train' split...")
        test_dataloader = create_minecraft_dataloader(
            data_dir=str(config.data_dir),
            batch_size=1,
            clip_length=clip_length,
            dataset_type=config.dataset_type,
            frame_size=(config.frame_height, config.frame_width),
            split="train",  # Fallback to train split
            num_workers=num_workers,
        )
    
    if test_dataloader is None:
        raise RuntimeError("Could not create test dataloader. Check data_dir and dataset_type in config.")
    
    print(f"Test dataloader created with {len(test_dataloader)} batches")
    
    # Evaluation loop
    print("\n" + "="*60)
    print("Evaluating on Test Data")
    print("="*60)
    
    all_rewards = []
    all_reward_info = defaultdict(list)
    num_samples = 0
    
    # Limit number of samples if specified
    max_batches = max_samples if max_samples > 0 else len(test_dataloader)
    
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing", total=max_batches)):
        if max_samples > 0 and num_samples >= max_samples:
            break
        
        # Prepare inputs
        if 'initial_frame' in batch:
            initial_frames = batch['initial_frame'].to(device)
        else:
            frames = batch['frames'].to(device)
            initial_frames = frames[:, :config.n_prompt_frames]
        
        actions = batch['actions'].to(device)
        target_actions = actions[:, :config.max_gen_frames]
        
        # Generate frames
        generated_frames = policy.generate_sequence(
            initial_frames=initial_frames,
            actions=target_actions,
            num_frames=config.max_gen_frames,
        )
        
        # Concatenate initial and generated frames
        all_frames = torch.cat([initial_frames, generated_frames], dim=1)
        
        # Compute rewards
        rewards, reward_info = reward_fn.compute_sequence_reward(
            all_frames,
            target_actions,
            return_per_frame=True,
        )
        
        # Aggregate metrics
        all_rewards.append(rewards.cpu().numpy())
        for key, value in reward_info.items():
            if isinstance(value, (int, float)):
                all_reward_info[key].append(value)
        
        # Save sample videos
        if save_videos and batch_idx < 10:  # Save first 10 samples
            sample_frames = all_frames[0].cpu()  # First sample
            video_path = video_dir / f"sample_{batch_idx:04d}_step_{checkpoint_metadata['global_step']}.mp4"
            frames_to_video(sample_frames, str(video_path), fps=10)
        
        num_samples += 1
        
        # Clean up
        del initial_frames, actions, target_actions, generated_frames, all_frames, rewards
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute aggregate statistics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    all_rewards = np.concatenate(all_rewards, axis=0)  # (num_samples, num_frames)
    
    results = {
        'checkpoint_step': checkpoint_metadata['global_step'],
        'checkpoint_path': checkpoint_metadata['checkpoint_path'],
        'num_samples': num_samples,
        'reward_stats': {
            'mean': float(np.mean(all_rewards)),
            'std': float(np.std(all_rewards)),
            'min': float(np.min(all_rewards)),
            'max': float(np.max(all_rewards)),
            'per_frame_mean': all_rewards.mean(axis=0).tolist(),
            'per_frame_std': all_rewards.std(axis=0).tolist(),
        },
        'component_rewards': {
            key: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
            for key, values in all_reward_info.items()
            if len(values) > 0
        },
    }
    
    # Print results
    print(f"\nCheckpoint: Step {results['checkpoint_step']}")
    print(f"Number of samples evaluated: {results['num_samples']}")
    print(f"\nOverall Reward Statistics:")
    print(f"  Mean: {results['reward_stats']['mean']:.4f} ± {results['reward_stats']['std']:.4f}")
    print(f"  Range: [{results['reward_stats']['min']:.4f}, {results['reward_stats']['max']:.4f}]")
    
    if len(results['reward_stats']['per_frame_mean']) > 1:
        print(f"\nPer-Frame Reward (showing degradation over time):")
        for i, (mean, std) in enumerate(zip(
            results['reward_stats']['per_frame_mean'],
            results['reward_stats']['per_frame_std']
        )):
            print(f"  Frame {i+1}: {mean:.4f} ± {std:.4f}")
    
    print(f"\nComponent Rewards:")
    for key, stats in results['component_rewards'].items():
        print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Save results to JSON
    results_path = output_path / f"test_results_step_{checkpoint_metadata['global_step']}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    if save_videos:
        print(f"Videos saved to: {video_dir}")
    
    return results


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    print("="*60)
    print("Oasis Checkpoint Testing")
    print(f"Loading config from: {args.config}")
    print("="*60)
    
    config = load_config(args.config)
    
    # Resolve device
    if args.no_cuda:
        config.device = "cpu"
    elif config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {config.checkpoint_dir}. "
                "Train a model first or specify --checkpoint."
            )
        print(f"\nFound most recent checkpoint: {checkpoint_path}")
    
    # Evaluate
    results = evaluate_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        max_samples=args.max_samples,
        save_videos=args.save_videos,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

