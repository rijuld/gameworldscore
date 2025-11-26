"""
Diffusion utilities for Oasis RL finetuning.

Provides noise scheduling and DDIM sampling functions.
"""

import torch
import math
from typing import Tuple, Optional


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    """
    Sigmoid beta schedule for diffusion.
    
    From open-oasis/utils.py - better for images > 64x64.
    
    Args:
        timesteps: Number of diffusion steps
        start: Start value for sigmoid
        end: End value for sigmoid
        tau: Temperature parameter
        clamp_min: Minimum value to clamp betas
        
    Returns:
        betas: (timesteps,) tensor of beta values
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule as proposed in Improved DDPM.
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent singularity at t=0
        
    Returns:
        betas: (timesteps,) tensor of beta values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear beta schedule.
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        betas: (timesteps,) tensor of beta values
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def get_diffusion_params(
    betas: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute diffusion parameters from beta schedule.
    
    Args:
        betas: (T,) beta schedule
        device: Device to place tensors on
        
    Returns:
        alphas: (T,) alpha values
        alphas_cumprod: (T,) cumulative product of alphas
        alphas_cumprod_prev: (T,) shifted cumulative product
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    return (
        alphas.to(device),
        alphas_cumprod.to(device),
        alphas_cumprod_prev.to(device),
    )


def ddim_sample(
    model: callable,
    x_T: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    stabilization_level: int = 15,
) -> torch.Tensor:
    """
    DDIM sampling for diffusion models.
    
    Args:
        model: Diffusion model that predicts velocity/noise
        x_T: (B, ...) initial noise
        timesteps: (num_steps,) timestep schedule for sampling
        alphas_cumprod: (T,) cumulative product of alphas
        condition: Optional conditioning tensor
        eta: DDIM stochasticity (0 = deterministic)
        stabilization_level: Noise level for context frames
        
    Returns:
        x_0: (B, ...) denoised sample
    """
    device = x_T.device
    B = x_T.shape[0]
    
    x = x_T.clone()
    
    for i in range(len(timesteps) - 1, 0, -1):
        t = timesteps[i]
        t_prev = timesteps[i - 1]
        
        # Current and previous alpha values
        alpha = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(alpha)
        
        # Model prediction
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        
        if condition is not None:
            v_pred = model(x, t_tensor, condition)
        else:
            v_pred = model(x, t_tensor)
        
        # Convert velocity to noise and x_0 prediction
        x_0_pred = alpha.sqrt() * x - (1 - alpha).sqrt() * v_pred
        eps_pred = ((1 / alpha).sqrt() * x - x_0_pred) / (1 / alpha - 1).sqrt()
        
        # DDIM update
        sigma = eta * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
        noise = torch.randn_like(x) if eta > 0 else 0
        
        x = alpha_prev.sqrt() * x_0_pred + (1 - alpha_prev - sigma**2).sqrt() * eps_pred + sigma * noise
    
    return x


def add_noise(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Add noise to samples for training.
    
    Args:
        x_0: (B, ...) clean samples
        noise: (B, ...) noise to add
        timesteps: (B,) timestep for each sample
        alphas_cumprod: (T,) cumulative product of alphas
        
    Returns:
        x_t: (B, ...) noisy samples
    """
    alpha = alphas_cumprod[timesteps]
    while alpha.dim() < x_0.dim():
        alpha = alpha.unsqueeze(-1)
    
    return alpha.sqrt() * x_0 + (1 - alpha).sqrt() * noise


def compute_velocity_target(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Compute velocity prediction target for v-prediction.
    
    v = sqrt(alpha) * eps - sqrt(1-alpha) * x_0
    
    Args:
        x_0: (B, ...) clean samples
        noise: (B, ...) noise added
        timesteps: (B,) timestep for each sample
        alphas_cumprod: (T,) cumulative product of alphas
        
    Returns:
        v: (B, ...) velocity target
    """
    alpha = alphas_cumprod[timesteps]
    while alpha.dim() < x_0.dim():
        alpha = alpha.unsqueeze(-1)
    
    return alpha.sqrt() * noise - (1 - alpha).sqrt() * x_0

