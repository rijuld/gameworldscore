"""
Oasis DiT wrapper as an RL-compatible policy.

This module wraps the Oasis diffusion transformer to provide:
- Forward pass for computing log probabilities
- Policy update methods compatible with RLVR-World training
- Latent diffusion sampling for rollouts
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from einops import rearrange
from safetensors.torch import load_model

# Add open-oasis to path for imports
OASIS_PATH = Path(__file__).parent.parent.parent / "open-oasis"
if str(OASIS_PATH) not in sys.path:
    sys.path.insert(0, str(OASIS_PATH))

from dit import DiT_models

from .oasis_vae import OasisVAE


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    Sigmoid beta schedule for diffusion.
    From open-oasis/utils.py
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class OasisPolicy(nn.Module):
    """
    Oasis DiT wrapped as an RL policy.
    
    This class provides the interface expected by RLVR-World's PPO trainer:
    - compute_log_prob: Compute log probability of generated frames
    - generate: Generate frames autoregressively
    - update: Update policy parameters with PPO loss
    
    The policy operates in latent space using the VAE encoder/decoder.
    """
    
    def __init__(
        self,
        oasis_ckpt: str,
        vae_ckpt: str,
        dit_type: str = "DiT-S/2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_noise_level: int = 1000,
        ddim_steps: int = 10,
        noise_abs_max: float = 20.0,
        stabilization_level: int = 15,
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.max_noise_level = max_noise_level
        self.ddim_steps = ddim_steps
        self.noise_abs_max = noise_abs_max
        self.stabilization_level = stabilization_level
        
        # Load DiT model
        self.dit = DiT_models[dit_type]()
        if oasis_ckpt.endswith(".pt"):
            ckpt = torch.load(oasis_ckpt, weights_only=True)
            self.dit.load_state_dict(ckpt, strict=False)
        elif oasis_ckpt.endswith(".safetensors"):
            load_model(self.dit, oasis_ckpt)
        self.dit = self.dit.to(device)
        
        # Load VAE
        self.vae = OasisVAE(vae_ckpt=vae_ckpt, device=device, dtype=dtype)
        
        # Precompute diffusion schedule
        self._setup_diffusion_schedule()
        
        # Max frames the model can process
        self.max_frames = self.dit.max_frames
        
    def _setup_diffusion_schedule(self):
        """Precompute diffusion noise schedule."""
        betas = sigmoid_beta_schedule(self.max_noise_level).float().to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', rearrange(alphas_cumprod, "T -> T 1 1 1"))
        
        # DDIM noise range
        noise_range = torch.linspace(-1, self.max_noise_level - 1, self.ddim_steps + 1)
        self.register_buffer('noise_range', noise_range)
    
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode RGB frames to latent space."""
        return self.vae.encode(frames)
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to RGB frames."""
        return self.vae.decode(latents)
    
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through DiT to predict velocity.
        
        Args:
            latents: (B, T, C, H, W) latent frames
            timesteps: (B, T) diffusion timesteps
            actions: (B, T, action_dim) one-hot encoded actions
            
        Returns:
            velocity: (B, T, C, H, W) predicted velocity
        """
        with autocast(self.device, dtype=self.dtype):
            velocity = self.dit(latents, timesteps, actions)
        return velocity
    
    @torch.no_grad()
    def generate_next_frame(
        self,
        context_latents: torch.Tensor,
        action: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Generate the next frame given context frames and action.
        
        Args:
            context_latents: (B, T, C, H, W) context latent frames
            action: (B, 1, action_dim) action for the new frame
            return_intermediates: Whether to return intermediate denoising states
            
        Returns:
            new_latent: (B, 1, C, H, W) generated latent frame
            intermediates: Optional dict of intermediate states
        """
        B = context_latents.shape[0]
        T = context_latents.shape[1]
        
        # Initialize with noise
        new_frame = torch.randn(
            (B, 1, *context_latents.shape[-3:]),
            device=self.device
        )
        new_frame = torch.clamp(new_frame, -self.noise_abs_max, self.noise_abs_max)
        
        # Concatenate context and new frame
        x = torch.cat([context_latents, new_frame], dim=1)
        
        # Concatenate actions
        all_actions = action  # actions for all frames including new one
        if context_latents.shape[1] > 0:
            # Pad context actions (use zeros or repeat last action)
            context_actions = torch.zeros(
                B, T, action.shape[-1], device=self.device
            )
            all_actions = torch.cat([context_actions, action], dim=1)
        
        # Determine start frame for sliding window
        start_frame = max(0, T + 1 - self.max_frames)
        
        intermediates = [] if return_intermediates else None
        
        # DDIM denoising loop
        for noise_idx in reversed(range(1, self.ddim_steps + 1)):
            # Set up noise timesteps
            t_ctx = torch.full(
                (B, T), self.stabilization_level - 1,
                dtype=torch.long, device=self.device
            )
            t = torch.full(
                (B, 1), self.noise_range[noise_idx].long(),
                dtype=torch.long, device=self.device
            )
            t_next = torch.full(
                (B, 1), self.noise_range[noise_idx - 1].long(),
                dtype=torch.long, device=self.device
            )
            t_next = torch.where(t_next < 0, t, t_next)
            
            t_full = torch.cat([t_ctx, t], dim=1)
            t_next_full = torch.cat([t_ctx, t_next], dim=1)
            
            # Apply sliding window
            x_curr = x[:, start_frame:].clone()
            t_curr = t_full[:, start_frame:]
            t_next_curr = t_next_full[:, start_frame:]
            actions_curr = all_actions[:, start_frame:T + 1]
            
            # Get model prediction
            with autocast(self.device, dtype=self.dtype):
                v = self.dit(x_curr, t_curr, actions_curr)
            
            # DDIM update
            alpha = self.alphas_cumprod[t_curr]
            x_start = alpha.sqrt() * x_curr - (1 - alpha).sqrt() * v
            x_noise = ((1 / alpha).sqrt() * x_curr - x_start) / (1 / alpha - 1).sqrt()
            
            alpha_next = self.alphas_cumprod[t_next_curr]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]
            
            if return_intermediates:
                intermediates.append(x[:, -1:].clone())
        
        new_latent = x[:, -1:]
        
        if return_intermediates:
            return new_latent, {'intermediates': intermediates}
        return new_latent, None
    
    @torch.no_grad()
    def generate_sequence(
        self,
        initial_frames: torch.Tensor,
        actions: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Generate a sequence of frames autoregressively.
        
        Args:
            initial_frames: (B, T, C, H, W) initial RGB frames
            actions: (B, num_frames, action_dim) actions for each generated frame
            num_frames: Number of frames to generate
            
        Returns:
            generated_frames: (B, num_frames, C, H, W) generated RGB frames
        """
        # Encode initial frames to latent space
        latents = self.encode_frames(initial_frames)
        
        generated_latents = []
        
        for i in range(num_frames):
            # Get action for this frame
            action = actions[:, i:i+1]
            
            # Generate next frame
            new_latent, _ = self.generate_next_frame(latents, action)
            generated_latents.append(new_latent)
            
            # Update context (sliding window)
            latents = torch.cat([latents, new_latent], dim=1)
            if latents.shape[1] > self.max_frames:
                latents = latents[:, -self.max_frames:]
        
        # Stack and decode
        generated_latents = torch.cat(generated_latents, dim=1)
        generated_frames = self.decode_latents(generated_latents)
        
        return generated_frames
    
    def compute_log_prob(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        target_latents: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of target latents given context.
        
        This approximates the log probability using the diffusion loss,
        which serves as a proxy for the true log probability in latent space.
        
        Args:
            latents: (B, T, C, H, W) context latent frames  
            actions: (B, T+1, action_dim) actions
            target_latents: (B, 1, C, H, W) target latent frames to score
            timesteps: Optional specific timesteps to evaluate at
            
        Returns:
            log_probs: (B,) log probability scores
        """
        B = latents.shape[0]
        
        if timesteps is None:
            # Sample random timesteps for evaluation
            timesteps = torch.randint(
                0, self.max_noise_level,
                (B,), device=self.device
            )
        
        # Add noise to target at given timesteps
        noise = torch.randn_like(target_latents)
        alpha = self.alphas_cumprod[timesteps].view(B, 1, 1, 1, 1)
        
        noisy_target = alpha.sqrt() * target_latents + (1 - alpha).sqrt() * noise
        
        # Concatenate context and noisy target
        x = torch.cat([latents, noisy_target], dim=1)
        
        # Create timestep tensor
        T = latents.shape[1]
        t_ctx = torch.full((B, T), 0, dtype=torch.long, device=self.device)
        t = torch.cat([t_ctx, timesteps.unsqueeze(1)], dim=1)
        
        # Forward pass
        with autocast(self.device, dtype=self.dtype):
            v_pred = self.dit(x, t, actions)
        
        # Compute target velocity (ensure same dtype as v_pred)
        v_target = alpha.sqrt() * noise - (1 - alpha).sqrt() * target_latents
        v_target = v_target.to(v_pred.dtype)
        
        # MSE loss as negative log probability proxy
        # Use float32 for stable loss computation
        mse = F.mse_loss(v_pred[:, -1:].float(), v_target.float(), reduction='none')
        mse = mse.view(B, -1).mean(dim=-1)
        
        # Convert to log prob (negative loss) - keep as float32
        log_probs = -mse
        
        return log_probs
    
    def get_trainable_parameters(self):
        """Get trainable parameters (DiT only, VAE is frozen)."""
        return self.dit.parameters()
    
    def train_mode(self):
        """Set model to training mode."""
        self.dit.train()
        self.vae.eval()  # Keep VAE frozen
        
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.dit.eval()
        self.vae.eval()


def load_oasis_policy(
    oasis_ckpt: str,
    vae_ckpt: str,
    device: str = "cuda",
) -> OasisPolicy:
    """
    Convenience function to load Oasis policy.
    
    Args:
        oasis_ckpt: Path to Oasis DiT checkpoint
        vae_ckpt: Path to VAE checkpoint
        device: Device to load model on
        
    Returns:
        OasisPolicy instance
    """
    return OasisPolicy(oasis_ckpt=oasis_ckpt, vae_ckpt=vae_ckpt, device=device)

