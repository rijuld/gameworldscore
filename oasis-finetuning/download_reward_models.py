#!/usr/bin/env python3
"""
Download reward models for Oasis RL finetuning.

This script downloads the pre-trained models needed for computing
GameWorldScore reward:
1. CLIP (for Temporal Consistency and Aesthetic Quality)
2. Aesthetic Predictor (for Aesthetic Quality)
3. IDM (Inverse Dynamics Model for Action Fidelity)

Usage:
    python download_reward_models.py
    python download_reward_models.py --output-dir custom_models/
"""

import os
import argparse
import requests
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor


def download_file(url: str, filename: str, desc: str = None):
    """Download a file with progress bar."""
    from tqdm import tqdm
    
    print(f"Downloading {desc or url} to {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or "Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Download reward models for Oasis RL finetuning")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_for_rl_finetuning",
        help="Output directory for models",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP download",
    )
    parser.add_argument(
        "--skip-aesthetic",
        action="store_true",
        help="Skip aesthetic predictor download",
    )
    parser.add_argument(
        "--skip-idm",
        action="store_true",
        help="Skip IDM download",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Reward Models for Oasis RL Finetuning")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir.absolute()}\n")
    
    # 1. CLIP (Temporal Consistency)
    if not args.skip_clip:
        print("1. Downloading CLIP (openai/clip-vit-base-patch32)...")
        clip_dir = output_dir / "clip-vit-base-patch32"
        
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            model.save_pretrained(str(clip_dir))
            processor.save_pretrained(str(clip_dir))
            print(f"   Saved to {clip_dir}")
        except Exception as e:
            print(f"   Failed to download CLIP: {e}")
    else:
        print("1. Skipping CLIP...")
    
    # 2. Aesthetic Predictor (Aesthetic Quality)
    if not args.skip_aesthetic:
        print("\n2. Downloading Aesthetic Predictor...")
        aesthetic_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        aesthetic_path = output_dir / "aesthetic_predictor.pth"
        
        try:
            download_file(
                aesthetic_url,
                str(aesthetic_path),
                "Aesthetic Predictor",
            )
        except Exception as e:
            print(f"   Failed to download aesthetic predictor: {e}")
    else:
        print("\n2. Skipping Aesthetic Predictor...")
    
    # 3. IDM (Inverse Dynamics Model)
    if not args.skip_idm:
        print("\n3. Downloading IDM (4x_idm)...")
        idm_model_url = "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model"
        idm_weights_url = "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights"
        
        try:
            download_file(
                idm_model_url,
                str(output_dir / "4x_idm.model"),
                "IDM Model",
            )
            download_file(
                idm_weights_url,
                str(output_dir / "4x_idm.weights"),
                "IDM Weights",
            )
        except Exception as e:
            print(f"   Failed to download IDM: {e}")
    else:
        print("\n3. Skipping IDM...")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # List downloaded files
    print("\nDownloaded files:")
    for path in output_dir.rglob("*"):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(output_dir)}: {size_mb:.2f} MB")
    
    print(f"\nYou can now use these models with:")
    print(f"  python train.py --reward-models-dir {output_dir}")


if __name__ == "__main__":
    main()

