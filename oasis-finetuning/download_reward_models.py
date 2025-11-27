#!/usr/bin/env python3
"""
Download reward models for Oasis RL finetuning.

Downloads:
1. CLIP (for Temporal Consistency)
2. Aesthetic Predictor (for Aesthetic Quality)
3. IDM (Inverse Dynamics Model for Action Fidelity)

Usage:
    python download_reward_models.py
"""

import os
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("models_for_rl_finetuning")

# Model URLs
AESTHETIC_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
IDM_MODEL_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model"
IDM_WEIGHTS_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights"


def download(url: str, path: Path):
    """Download a file."""
    print(f"Downloading {path.name}...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")

    # 1. CLIP (ViT-L/14 to match aesthetic predictor embeddings - 768 dim)
    print("1. Downloading CLIP ViT-L/14...")
    from transformers import CLIPModel, CLIPProcessor
    clip_dir = OUTPUT_DIR / "clip-vit-large-patch14"
    CLIPModel.from_pretrained("openai/clip-vit-large-patch14").save_pretrained(str(clip_dir))
    CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").save_pretrained(str(clip_dir))
    print(f"  Saved to {clip_dir}")

    # 2. Aesthetic Predictor
    print("\n2. Downloading Aesthetic Predictor...")
    download(AESTHETIC_URL, OUTPUT_DIR / "aesthetic_predictor.pth")

    # 3. IDM
    print("\n3. Downloading IDM...")
    download(IDM_MODEL_URL, OUTPUT_DIR / "4x_idm.model")
    download(IDM_WEIGHTS_URL, OUTPUT_DIR / "4x_idm.weights")

    print("\nDone! Use with: python train.py --reward-models-dir models_for_rl_finetuning")


if __name__ == "__main__":
    main()
