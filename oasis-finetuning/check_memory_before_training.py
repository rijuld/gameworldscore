#!/usr/bin/env python3
"""
Pre-training memory check for Oasis GRPO.
Run this to see how much memory is used just by loading the models.
"""

import os
import sys

# Set memory config before importing torch
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import torch
from pathlib import Path

def format_bytes(bytes_val):
    """Format bytes to GB."""
    return f"{bytes_val / 1e9:.2f} GB"

def check_gpu_memory():
    """Check GPU memory availability."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    print(f"\n{'='*70}")
    print(f"GPU Device: {torch.cuda.get_device_name(device)}")
    print(f"Total Memory: {format_bytes(total_memory)}")
    print(f"{'='*70}\n")
    
    return True

def load_models_and_check():
    """Load models incrementally and check memory after each step."""
    
    # Add parent directories to path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    sys.path.insert(0, str(project_root / "oasis-finetuning"))
    
    print("Step 1: Initial GPU memory")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
    
    print("\nStep 2: Loading Oasis Policy...")
    try:
        from models.oasis_policy import OasisPolicy
        
        policy = OasisPolicy(
            oasis_ckpt="/home/rd3629/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/oasis500m.safetensors",
            vae_ckpt="/home/rd3629/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/vit-l-20.safetensors",
            device="cuda",
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - (torch.cuda.memory_reserved() / 1e9)
            print(f"  ✓ Policy loaded")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Free: {free:.2f} GB ({(free/total)*100:.1f}%)")
            
            if free < 5:
                print(f"  ⚠️  WARNING: Less than 5 GB free!")
    except Exception as e:
        print(f"  ❌ Failed to load policy: {e}")
        return
    
    print("\nStep 3: Loading Reward Models (CPU)...")
    try:
        from rewards.game_world_score import create_game_world_score_reward
        
        reward_fn = create_game_world_score_reward(
            models_dir="models_for_rl_finetuning",
            device="cpu",  # Load on CPU
            rik_weight=1.0,
            rtc_weight=1.0,
            raq_weight=1.0,
            require_vpt=True,
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - (torch.cuda.memory_reserved() / 1e9)
            print(f"  ✓ Reward models loaded on CPU")
            print(f"  GPU Allocated: {allocated:.2f} GB")
            print(f"  GPU Free: {free:.2f} GB ({(free/total)*100:.1f}%)")
    except Exception as e:
        print(f"  ⚠️  Could not load reward models: {e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - reserved
        
        print(f"Total GPU Memory:     {total:.2f} GB")
        print(f"Allocated:            {allocated:.2f} GB ({(allocated/total)*100:.1f}%)")
        print(f"Reserved:             {reserved:.2f} GB ({(reserved/total)*100:.1f}%)")
        print(f"Free:                 {free:.2f} GB ({(free/total)*100:.1f}%)")
        print("="*70)
        
        if free < 5:
            print("\n🚨 CRITICAL: Less than 5 GB free after loading models!")
            print("   Training will likely OOM. Recommendations:")
            print("   1. Reduce group_size to 1")
            print("   2. Reduce max_gen_frames to 1")
            print("   3. Use CPU offloading for rewards (already enabled)")
        elif free < 10:
            print("\n⚠️  WARNING: Less than 10 GB free.")
            print("   Training might OOM with current settings.")
            print("   Consider reducing group_size=2, max_gen_frames=2")
        else:
            print(f"\n✅ {free:.2f} GB free - should be enough for training!")

if __name__ == "__main__":
    if check_gpu_memory():
        load_models_and_check()
