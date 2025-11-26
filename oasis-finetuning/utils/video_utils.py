"""
Video utilities for Oasis RL finetuning.

Provides functions for video I/O and frame manipulation.
"""

from typing import Tuple, Optional, Union
from pathlib import Path

import torch
import numpy as np


def frames_to_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 20,
) -> None:
    """
    Save frames as a video file.
    
    Args:
        frames: (T, C, H, W) or (T, H, W, C) tensor of frames in [0, 1]
        output_path: Path to save video
        fps: Frames per second
    """
    from torchvision.io import write_video
    
    # Ensure (T, H, W, C) format
    if frames.dim() == 4:
        if frames.shape[1] == 3:  # (T, C, H, W)
            frames = frames.permute(0, 2, 3, 1)
    
    # Convert to uint8
    frames = (frames.clamp(0, 1) * 255).byte()
    
    write_video(output_path, frames.cpu(), fps=fps)


def video_to_frames(
    video_path: str,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index
        num_frames: Number of frames to load (None = all)
        frame_size: Optional (H, W) to resize frames
        
    Returns:
        frames: (T, C, H, W) tensor of frames in [0, 1]
    """
    from torchvision.io import read_video
    from torchvision.transforms.functional import resize
    
    video, _, _ = read_video(video_path, pts_unit="sec")
    
    # Slice frames
    if num_frames is not None:
        video = video[start_frame:start_frame + num_frames]
    else:
        video = video[start_frame:]
    
    # Convert to (T, C, H, W) and normalize
    frames = video.permute(0, 3, 1, 2).float() / 255.0
    
    # Resize if needed
    if frame_size is not None:
        frames = torch.stack([resize(f, frame_size) for f in frames])
    
    return frames


def create_comparison_grid(
    real_frames: torch.Tensor,
    generated_frames: torch.Tensor,
    num_cols: int = 4,
) -> torch.Tensor:
    """
    Create a comparison grid of real vs generated frames.
    
    Args:
        real_frames: (T, C, H, W) real frames
        generated_frames: (T, C, H, W) generated frames
        num_cols: Number of columns in grid
        
    Returns:
        grid: (C, H', W') comparison grid image
    """
    T = min(real_frames.shape[0], generated_frames.shape[0])
    T = min(T, num_cols * 2)  # Limit for visualization
    
    # Select frames to show
    indices = torch.linspace(0, T - 1, min(T, num_cols * 2)).long()
    
    real_selected = real_frames[indices[:num_cols]]
    gen_selected = generated_frames[indices[:num_cols]]
    
    # Create rows
    real_row = torch.cat(list(real_selected), dim=-1)
    gen_row = torch.cat(list(gen_selected), dim=-1)
    
    # Stack vertically
    grid = torch.cat([real_row, gen_row], dim=-2)
    
    return grid


def compute_video_metrics(
    real_frames: torch.Tensor,
    generated_frames: torch.Tensor,
) -> dict:
    """
    Compute video quality metrics.
    
    Args:
        real_frames: (T, C, H, W) ground truth frames
        generated_frames: (T, C, H, W) generated frames
        
    Returns:
        Dict with PSNR, SSIM, etc.
    """
    import torch.nn.functional as F
    
    T = min(real_frames.shape[0], generated_frames.shape[0])
    real = real_frames[:T]
    gen = generated_frames[:T]
    
    # MSE
    mse = F.mse_loss(gen, real).item()
    
    # PSNR
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    
    # MAE
    mae = F.l1_loss(gen, real).item()
    
    # Frame-wise metrics
    frame_mses = []
    for t in range(T):
        frame_mse = F.mse_loss(gen[t], real[t]).item()
        frame_mses.append(frame_mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'frame_mses': frame_mses,
        'temporal_consistency': 1.0 - np.std(frame_mses) if len(frame_mses) > 1 else 1.0,
    }


def sample_frames_uniform(
    video: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """
    Sample frames uniformly from video.
    
    Args:
        video: (T, ...) video tensor
        num_frames: Number of frames to sample
        
    Returns:
        sampled: (num_frames, ...) sampled frames
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_frames).long()
    return video[indices]


def pad_video(
    video: torch.Tensor,
    target_length: int,
    mode: str = "replicate",
) -> torch.Tensor:
    """
    Pad video to target length.
    
    Args:
        video: (T, C, H, W) video tensor
        target_length: Target number of frames
        mode: Padding mode - "replicate" or "zero"
        
    Returns:
        padded: (target_length, C, H, W) padded video
    """
    T = video.shape[0]
    
    if T >= target_length:
        return video[:target_length]
    
    pad_length = target_length - T
    
    if mode == "replicate":
        # Repeat last frame
        pad = video[-1:].repeat(pad_length, 1, 1, 1)
    elif mode == "zero":
        pad = torch.zeros(pad_length, *video.shape[1:], device=video.device)
    else:
        raise ValueError(f"Unknown padding mode: {mode}")
    
    return torch.cat([video, pad], dim=0)

