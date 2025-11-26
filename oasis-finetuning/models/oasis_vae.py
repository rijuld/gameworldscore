"""
VAE wrapper for Oasis, importing from the open-oasis repository.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import autocast
from einops import rearrange
from safetensors.torch import load_model

# Add open-oasis to path for imports
OASIS_PATH = Path(__file__).parent.parent.parent / "open-oasis"
if str(OASIS_PATH) not in sys.path:
    sys.path.insert(0, str(OASIS_PATH))

from vae import VAE_models


class OasisVAE(nn.Module):
    """
    Wrapper around the Oasis VAE for encoding/decoding frames.
    
    This class provides a clean interface for:
    - Encoding RGB frames to latent representations
    - Decoding latent representations back to RGB frames
    """
    
    def __init__(
        self,
        vae_ckpt: str,
        vae_type: str = "vit-l-20-shallow-encoder",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.scaling_factor = 0.07843137255
        
        # Load VAE model from open-oasis
        self.vae = VAE_models[vae_type]()
        
        if vae_ckpt.endswith(".pt"):
            vae_state = torch.load(vae_ckpt, weights_only=True)
            self.vae.load_state_dict(vae_state)
        elif vae_ckpt.endswith(".safetensors"):
            load_model(self.vae, vae_ckpt)
        
        self.vae = self.vae.to(device).eval()
        self.patch_size = self.vae.patch_size
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB frames to latent representations.
        
        Args:
            frames: (B, T, C, H, W) or (B, C, H, W) tensor of RGB frames in [0, 1]
            
        Returns:
            latents: (B, T, h, w, C) or (B, h, w, C) tensor of latent representations
        """
        has_time_dim = frames.dim() == 5
        
        if has_time_dim:
            B, T, C, H, W = frames.shape
            frames = rearrange(frames, "b t c h w -> (b t) c h w")
        else:
            B = frames.shape[0]
            T = 1
            H, W = frames.shape[-2:]
        
        with autocast(self.device, dtype=self.dtype):
            # Normalize to [-1, 1] and encode
            latents = self.vae.encode(frames * 2 - 1).mean * self.scaling_factor
        
        # Reshape latents
        h, w = H // self.patch_size, W // self.patch_size
        latents = rearrange(latents, "(b t) (h w) c -> b t c h w", t=T, h=h, w=w)
        
        if not has_time_dim:
            latents = latents.squeeze(1)
        
        return latents
    
    @torch.no_grad() 
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to RGB frames.
        
        Args:
            latents: (B, T, C, H, W) or (B, C, H, W) tensor of latent representations
            
        Returns:
            frames: (B, T, C, H, W) or (B, C, H, W) tensor of RGB frames in [0, 1]
        """
        has_time_dim = latents.dim() == 5
        
        if has_time_dim:
            B, T = latents.shape[:2]
            latents = rearrange(latents, "b t c h w -> (b t) (h w) c")
        else:
            B = latents.shape[0]
            T = 1
            latents = rearrange(latents, "b c h w -> b (h w) c")
        
        with autocast(self.device, dtype=self.dtype):
            frames = (self.vae.decode(latents / self.scaling_factor) + 1) / 2
        
        frames = torch.clamp(frames, 0, 1)
        
        if has_time_dim:
            frames = rearrange(frames, "(b t) c h w -> b t c h w", t=T)
        
        return frames


def load_oasis_vae(vae_ckpt: str, device: str = "cuda") -> OasisVAE:
    """
    Convenience function to load Oasis VAE.
    
    Args:
        vae_ckpt: Path to VAE checkpoint
        device: Device to load model on
        
    Returns:
        OasisVAE instance
    """
    return OasisVAE(vae_ckpt=vae_ckpt, device=device)

