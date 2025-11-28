"""
Temporal Consistency Score (RTC) for measuring smoothness.

Uses Optical Flow Warping Consistency (Photometric Loss) to
measure temporal consistency and motion smoothness.

This is the standard approach for enforcing geometric coherence.
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NEW IMPORTS FOR OPTICAL FLOW AND WARPING ---
try:
    import torchvision.models.optical_flow as optical_flow_models
except ImportError:
    print("Warning: torchvision not found. Install with: pip install torchvision")
    optical_flow_models = type('optical_flow_models', (object,), {'raft_large': None})
# ------------------------------------------------


class TemporalConsistencyReward(nn.Module):
    """
    Computes the Temporal Consistency Score (RTC) reward based on Optical Flow.
    
    RTC uses the photometric warping consistency loss (Warping Loss) 
    between consecutive frames, which is the negative of the reward.
    
    Higher scores (lower loss) indicate better temporal consistency.
    """
    
    def __init__(
        self,
        # Note: clip_model_path is unused but kept for compatibility/re-purposing
        clip_model_path: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        raft_model_name: str = "raft_large", # Use raft_large for high accuracy
    ):
        super().__init__()
        self.device = device
        
        # --- OPTICAL FLOW MODEL INITIALIZATION (Replaces CLIP) ---
        print(f"Loading RAFT Optical Flow Model ({raft_model_name})...")
        if raft_model_name == "raft_large":
            self.flow_model = optical_flow_models.raft_large(pretrained=True).to(device).eval()
        elif raft_model_name == "raft_small":
            self.flow_model = optical_flow_models.raft_small(pretrained=True).to(device).eval()
        else:
             raise ValueError(f"Unknown RAFT model name: {raft_model_name}")
             
        # Freeze RAFT model
        for param in self.flow_model.parameters():
            param.requires_grad = False
            
        # RAFT is typically run at full precision or its own autocast logic
        self.use_half_precision = False 

    # --- CLIP-related methods are removed: _preprocess_for_clip, extract_features, compute_clip_similarity ---

    def _warp_frame(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp an image using optical flow.
        
        Args:
            image: (B, C, H, W) image to warp
            flow: (B, 2, H, W) optical flow field (dx, dy)
            
        Returns:
            warped_image: (B, C, H, W) warped image
        """
        B, C, H, W = image.shape
        
        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=image.device), torch.arange(W, device=image.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # (2, H, W)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
        
        # Add flow to grid
        # Flow is typically (dx, dy) in pixels
        vgrid = grid + flow
        
        # Normalize to [-1, 1] for grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # Permute to (B, H, W, 2)
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        # Sample
        warped = F.grid_sample(image, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    @torch.no_grad()
    def _compute_warping_loss(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Photometric Warping Loss and the flow vectors.
        
        NOTE: This implements Photometric Consistency Loss: || Warp(frame_t, flow) - frame_t1 ||
        This is distinct from Flow Matching Loss: || Flow(gen) - Flow(gt) || which requires ground truth.
        For RL finetuning without ground truth, Photometric Consistency is the correct metric.
        
        Args:
            frame_t: (B, C, H, W) frame at time t (normalized to [0, 1])
            frame_t1: (B, C, H, W) frame at time t+1 (normalized to [0, 1])
            
        Returns:
            warping_loss: (B,) MSE loss for each sequence
            flow_vectors: (B, 2, H, W) Estimated flow V_t->t+1
        """
        B, C, H, W = frame_t.shape
        
        # 1. Estimate Optical Flow V_t->t+1
        # Input to RAFT should be [0, 255] roughly, but it handles [0, 1] okay usually.
        # Ideally we scale to [0, 255] and normalize, but for reward signal [0, 1] is often sufficient.
        # We will use the inputs as-is (assuming [0, 1]).
        
        # flow is (B, 2, H, W)
        flow_vectors = self.flow_model(frame_t, frame_t1)[-1] 
        
        # 2. Warp frame_t to predicted frame_t+1 (Warped_t1)
        warped_t1 = self._warp_frame(frame_t, flow_vectors)

        # 3. Calculate MSE (Photometric Warping Loss)
        # Loss between the warped prediction and the actual next frame
        loss = F.mse_loss(warped_t1, frame_t1, reduction='none')
        # Sum over C, H, W and average over C to get (B,) loss per sequence
        warping_loss = loss.view(B, -1).mean(dim=-1)

        return warping_loss, flow_vectors
    
    @torch.no_grad()
    def compute_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        # frame_t2 is no longer needed for a simple transition reward
        frame_t2: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RTC reward for a frame transition based on Warping Loss.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) frame at time t+1
            
        Returns:
            reward: (B,) RTC reward normalized to [0, 1] (Higher is better)
            info: Dict with metrics
        """
        # Calculate the Warping Loss (Loss is lower -> Consistency is better)
        warping_loss, _ = self._compute_warping_loss(frame_t, frame_t1)
        
        # Convert Loss to Reward: Reward = 1 / (1 + Loss) 
        # This converts a loss (0 to infinity) into a reward (0 to 1) where 1 is perfect.
        reward = 1.0 / (1.0 + warping_loss)
        
        info = {
            'rtc_warping_loss': warping_loss.mean().item(),
            'rtc_normalized_reward': reward.mean().item(),
        }
        
        return reward, info
    
    @torch.no_grad()
    def compute_sequence_reward(
        self,
        frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RTC reward for a sequence of frames (OPTIMIZED: batched).
        
        Args:
            frames: (B, T, C, H, W) sequence of frames
            
        Returns:
            rewards: (B, T-1) RTC reward for each transition
            info: Dict with aggregated metrics
        """
        # Ensure inputs are on the correct device
        if frames.device.type != self.device:
            frames = frames.to(self.device)
        
        B, T = frames.shape[:2]
        
        if T < 2:
            return torch.zeros(B, 0, device=self.device), {'rtc_warping_loss': 0.0}
        
        # Batch process all transitions at once
        frame_t_batch = frames[:, :-1].contiguous()  # (B, T-1, C, H, W)
        frame_t1_batch = frames[:, 1:].contiguous()   # (B, T-1, C, H, W)
        
        # Reshape to (B*(T-1), C, H, W) for batch processing
        B_seq = B * (T - 1)
        frame_t_flat = frame_t_batch.view(B_seq, *frame_t_batch.shape[2:])
        frame_t1_flat = frame_t1_batch.view(B_seq, *frame_t1_batch.shape[2:])
        
        # Calculate Warping Loss for all transitions
        warping_losses_flat, _ = self._compute_warping_loss(frame_t_flat, frame_t1_flat)
        
        # Convert Loss to Reward: Reward = 1 / (1 + Loss) 
        rewards_flat = 1.0 / (1.0 + warping_losses_flat)
        
        # Reshape back to (B, T-1)
        rewards = rewards_flat.view(B, T - 1)
        warping_losses = warping_losses_flat.view(B, T - 1)
        
        info = {
            'rtc_warping_loss': warping_losses.mean().item(),
            'rtc_normalized_reward': rewards.mean().item(),
        }
        
        # The old 'motion smoothness' term (linear interpolation) is deprecated 
        # as the Warping Loss is the superior measure of geometric motion coherence.
        
        return rewards, info


def load_temporal_consistency_reward(
    # Note: clip_model_path is unused but kept for compatibility/re-purposing
    clip_model_path: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
) -> TemporalConsistencyReward:
    """
    Load temporal consistency reward module.
    
    Args:
        clip_model_path: Path to CLIP model (ignored)
        device: Device to load on
        
    Returns:
        TemporalConsistencyReward instance using RAFT Optical Flow.
    """
    return TemporalConsistencyReward(
        device=device,
    )