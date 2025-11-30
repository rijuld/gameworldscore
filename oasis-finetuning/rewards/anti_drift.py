"""
Anti-Drift Reward for Minecraft World Model Finetuning

Specifically designed to:
1. Prevent cascading blur/degradation
2. Maintain motion dynamics from original model
3. Avoid motion collapse and grid artifacts

Key insight: Don't just penalize blur - reward clarity AND motion simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class AntiDriftReward(nn.Module):
    """
    Prevents cascading error while maintaining motion dynamics.
    
    Components:
    1. Sharpness Reward: Penalizes blur accumulation
    2. Motion Preservation: Ensures actions still cause movement
    3. Texture Quality: Maintains block texture clarity
    4. Anti-Grid: Penalizes checkerboard/grid artifacts
    """
    
    def __init__(
        self,
        device: str = "cuda",
        # Component weights
        sharpness_weight: float = 1.0,
        motion_weight: float = 1.5,        # Higher to prevent motion collapse
        texture_weight: float = 0.8,
        anti_grid_weight: float = 0.5,
        # Sharpness parameters
        blur_threshold: float = 0.15,       # Penalize if below this
        target_sharpness: float = 0.25,     # Good Minecraft sharpness
        # Motion parameters
        min_motion_threshold: float = 0.02, # Minimum expected motion
        motion_history_len: int = 3,        # Track motion over time
    ):
        super().__init__()
        self.device = device
        
        self.sharpness_weight = sharpness_weight
        self.motion_weight = motion_weight
        self.texture_weight = texture_weight
        self.anti_grid_weight = anti_grid_weight
        
        self.blur_threshold = blur_threshold
        self.target_sharpness = target_sharpness
        self.min_motion_threshold = min_motion_threshold
        self.motion_history_len = motion_history_len
        
        # Laplacian kernel for sharpness detection
        laplacian = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))
        
        # High-pass filter for texture quality
        highpass = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32) / 9.0
        self.register_buffer('highpass', highpass.view(1, 1, 3, 3))
        
        # Checkerboard detection kernel
        checkerboard = torch.tensor([
            [1, -1, 1],
            [-1, 1, -1],
            [1, -1, 1]
        ], dtype=torch.float32)
        self.register_buffer('checkerboard', checkerboard.view(1, 1, 3, 3))
    
    def compute_sharpness(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute image sharpness using Laplacian variance.
        Higher = sharper (less blurry).
        
        This directly measures blur accumulation.
        """
        B = frames.shape[0]
        
        # Convert to grayscale
        gray = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)
        
        # Apply Laplacian
        laplacian = F.conv2d(gray, self.laplacian.to(frames.device), padding=1)
        
        # Variance of Laplacian (higher = sharper edges)
        sharpness = laplacian.var(dim=[2, 3])  # (B, 1)
        sharpness = sharpness.squeeze(1)  # (B,)
        
        return sharpness
    
    def compute_motion_magnitude(
        self, 
        frame_t: torch.Tensor, 
        frame_t1: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actual motion between frames.
        Returns both pixel difference and action-conditioned motion.
        
        Key: We want motion when actions are taken, not random drift.
        """
        B = frame_t.shape[0]
        
        # Pixel-level motion (simple but effective)
        pixel_diff = (frame_t1 - frame_t).abs().mean(dim=[1, 2, 3])  # (B,)
        
        # Action magnitude (how much action was taken)
        # Assuming action is (B, action_dim) with one-hot or continuous values
        from data.action_utils import ACTION_KEYS
        
        # Camera actions (most important for motion)
        camera_x_idx = ACTION_KEYS.index("cameraX") if "cameraX" in ACTION_KEYS else 0
        camera_y_idx = ACTION_KEYS.index("cameraY") if "cameraY" in ACTION_KEYS else 1
        
        camera_motion = (
            action[:, camera_x_idx].abs() + 
            action[:, camera_y_idx].abs()
        )
        
        # Movement actions
        movement_keys = ["forward", "back", "left", "right", "jump", "sneak"]
        movement_motion = torch.zeros(B, device=action.device)
        for key in movement_keys:
            if key in ACTION_KEYS:
                idx = ACTION_KEYS.index(key)
                movement_motion += action[:, idx].abs()
        
        total_action_magnitude = camera_motion + movement_motion * 0.3
        
        # Expected motion given action (more action = more motion expected)
        expected_motion = torch.clamp(total_action_magnitude * 0.1, 0, 1)
        
        return pixel_diff, expected_motion
    
    def compute_texture_quality(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Measure texture quality (block detail preservation).
        Minecraft blocks have high-frequency texture patterns.
        """
        B = frames.shape[0]
        
        # Convert to grayscale
        gray = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray = gray.unsqueeze(1)
        
        # Apply high-pass filter
        high_freq = F.conv2d(gray, self.highpass.to(frames.device), padding=1)
        
        # Texture energy (should be moderate for Minecraft)
        texture_energy = high_freq.abs().mean(dim=[2, 3]).squeeze(1)  # (B,)
        
        # Normalize: 0.1-0.4 is good range for Minecraft textures
        # Too low = blurry, too high = noise
        optimal_range = (0.1, 0.4)
        texture_score = 1.0 - torch.clamp(
            (texture_energy - optimal_range[0]).abs() / optimal_range[1],
            0, 1
        )
        
        return texture_score
    
    def detect_grid_artifacts(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Detect checkerboard/grid artifacts (common in unstable training).
        Lower = fewer artifacts.
        """
        B = frames.shape[0]
        
        # Convert to grayscale
        gray = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        gray = gray.unsqueeze(1)
        
        # Apply checkerboard detection
        checker_response = F.conv2d(
            gray, 
            self.checkerboard.to(frames.device), 
            padding=1
        )
        
        # Strong checkerboard pattern = high response
        grid_score = checker_response.abs().mean(dim=[2, 3]).squeeze(1)  # (B,)
        
        # Penalize high grid scores
        grid_penalty = torch.clamp(grid_score / 2.0, 0, 1)
        
        return grid_penalty
    
    def compute_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute anti-drift reward for a single transition.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) generated frame at t+1
            action: (B, action_dim) action taken
            
        Returns:
            reward: (B,) total reward
            info: Dict with component scores
        """
        B = frame_t.shape[0]
        
        # 1. Sharpness reward (prevent blur accumulation)
        sharpness = self.compute_sharpness(frame_t1)
        
        # Reward: higher sharpness = better, but clamp to reasonable range
        # Good Minecraft frames: 0.15 - 0.35 Laplacian variance
        sharpness_reward = torch.clamp(
            (sharpness - self.blur_threshold) / self.target_sharpness,
            0, 1
        )
        
        # 2. Motion preservation (ensure actions cause movement)
        pixel_motion, expected_motion = self.compute_motion_magnitude(
            frame_t, frame_t1, action
        )
        
        # If action is taken, we want motion
        # If no action, low motion is okay
        motion_reward = torch.where(
            expected_motion > self.min_motion_threshold,
            # Action taken: reward if pixel motion matches expectation
            1.0 - (pixel_motion - expected_motion).abs(),
            # No action: reward low motion (static is okay)
            1.0 - torch.clamp(pixel_motion / 0.1, 0, 1)
        )
        motion_reward = torch.clamp(motion_reward, 0, 1)
        
        # 3. Texture quality (maintain block details)
        texture_reward = self.compute_texture_quality(frame_t1)
        
        # 4. Anti-grid (penalize artifacts)
        grid_penalty = self.detect_grid_artifacts(frame_t1)
        anti_grid_reward = 1.0 - grid_penalty
        
        # Combine rewards
        total_reward = (
            self.sharpness_weight * sharpness_reward +
            self.motion_weight * motion_reward +
            self.texture_weight * texture_reward +
            self.anti_grid_weight * anti_grid_reward
        )
        
        # Normalize
        total_weight = (
            self.sharpness_weight + 
            self.motion_weight + 
            self.texture_weight + 
            self.anti_grid_weight
        )
        total_reward = total_reward / total_weight
        
        info = {
            'anti_drift_total': total_reward.mean().item(),
            'sharpness': sharpness.mean().item(),
            'sharpness_reward': sharpness_reward.mean().item(),
            'motion_magnitude': pixel_motion.mean().item(),
            'motion_reward': motion_reward.mean().item(),
            'texture_reward': texture_reward.mean().item(),
            'anti_grid_reward': anti_grid_reward.mean().item(),
            'grid_penalty': grid_penalty.mean().item(),
        }
        
        return total_reward, info
    
    @torch.no_grad()
    def compute_sequence_reward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        return_per_frame: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute anti-drift reward for a sequence.
        
        Key: Penalize accumulating degradation over time.
        
        Args:
            frames: (B, T, C, H, W) sequence of frames
            actions: (B, T-1, action_dim) actions
            return_per_frame: Return per-frame rewards
            
        Returns:
            rewards: (B,) or (B, T-1)
            info: Dict with metrics
        """
        B, T = frames.shape[:2]
        
        if T < 2:
            return torch.zeros(B, 0, device=self.device), {}
        
        frame_rewards = []
        all_sharpness = []
        all_motion = []
        all_texture = []
        all_anti_grid = []
        
        # Initial sharpness (reference)
        initial_sharpness = self.compute_sharpness(frames[:, 0])
        
        for t in range(T - 1):
            reward, info = self.compute_reward(
                frames[:, t],
                frames[:, t + 1],
                actions[:, t],
            )
            frame_rewards.append(reward)
            all_sharpness.append(info['sharpness'])
            all_motion.append(info['motion_magnitude'])
            all_texture.append(info['texture_reward'])
            all_anti_grid.append(info['anti_grid_reward'])
        
        rewards = torch.stack(frame_rewards, dim=1)  # (B, T-1)
        
        # Additional penalty for sharpness degradation over time
        final_sharpness = self.compute_sharpness(frames[:, -1])
        sharpness_degradation = torch.clamp(
            (initial_sharpness - final_sharpness) / initial_sharpness,
            0, 1
        )
        
        # Apply degradation penalty to later frames
        degradation_penalty = sharpness_degradation.unsqueeze(1)  # (B, 1)
        time_weights = torch.linspace(0, 1, T-1, device=self.device).view(1, -1)
        rewards = rewards - degradation_penalty * time_weights * 0.2
        
        info = {
            'anti_drift_total': rewards.mean().item(),
            'sharpness_mean': sum(all_sharpness) / len(all_sharpness),
            'sharpness_initial': initial_sharpness.mean().item(),
            'sharpness_final': final_sharpness.mean().item(),
            'sharpness_degradation': sharpness_degradation.mean().item(),
            'motion_mean': sum(all_motion) / len(all_motion),
            'texture_mean': sum(all_texture) / len(all_texture),
            'anti_grid_mean': sum(all_anti_grid) / len(all_anti_grid),
        }
        
        if return_per_frame:
            return rewards, info
        else:
            return rewards.sum(dim=1), info


def create_anti_drift_reward(
    device: str = "cuda",
    # Tune these based on your specific degradation pattern
    sharpness_weight: float = 1.0,
    motion_weight: float = 1.5,  # Increase if motion collapse occurs
    texture_weight: float = 0.8,
    anti_grid_weight: float = 0.5,
) -> AntiDriftReward:
    """
    Create anti-drift reward module.
    
    Tuning guide:
    - Motion collapse: Increase motion_weight to 2.0+
    - Still too blurry: Increase sharpness_weight to 1.5+
    - Grid artifacts: Increase anti_grid_weight to 1.0+
    - Texture loss: Increase texture_weight to 1.0+
    """
    return AntiDriftReward(
        device=device,
        sharpness_weight=sharpness_weight,
        motion_weight=motion_weight,
        texture_weight=texture_weight,
        anti_grid_weight=anti_grid_weight,
    )