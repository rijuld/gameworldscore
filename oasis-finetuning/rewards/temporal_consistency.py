"""
Temporal Consistency Score (RTC) for measuring smoothness.

Uses CLIP feature similarity between consecutive frames to
measure temporal consistency and motion smoothness.

From GameWorldScore benchmark:
- CLIP feature cosine similarity between frames
- Motion smoothness via frame interpolation consistency
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class TemporalConsistencyReward(nn.Module):
    """
    Computes the Temporal Consistency Score (RTC) reward.
    
    RTC combines:
    1. CLIP feature similarity between consecutive frames
    2. Motion smoothness via optical flow consistency (optional)
    
    Higher scores indicate better temporal consistency.
    """
    
    def __init__(
        self,
        clip_model_path: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        use_motion_smoothness: bool = False,
    ):
        super().__init__()
        self.device = device
        self.use_motion_smoothness = use_motion_smoothness
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.clip_model = self.clip_model.to(device).eval()
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Normalization for CLIP input
        self.register_buffer(
            'mean', 
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
    
    def _preprocess_for_clip(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP.
        
        Args:
            images: (B, C, H, W) images in [0, 1]
            
        Returns:
            preprocessed: (B, C, 224, 224) preprocessed images
        """
        # Resize to CLIP input size
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize
        images = (images - self.mean) / self.std
        
        return images
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP visual features.
        
        Args:
            images: (B, C, H, W) images in [0, 1]
            
        Returns:
            features: (B, feature_dim) CLIP features
        """
        images = self._preprocess_for_clip(images)
        features = self.clip_model.get_image_features(images)
        features = F.normalize(features, p=2, dim=-1)
        return features
    
    @torch.no_grad()
    def compute_clip_similarity(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP feature similarity between consecutive frames.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) frame at time t+1
            
        Returns:
            similarity: (B,) cosine similarity scores
        """
        feat_t = self.extract_features(frame_t)
        feat_t1 = self.extract_features(frame_t1)
        
        # Cosine similarity (features already normalized)
        similarity = (feat_t * feat_t1).sum(dim=-1)
        
        return similarity
    
    @torch.no_grad()
    def compute_motion_smoothness(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        frame_t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute motion smoothness using frame interpolation.
        
        The middle frame should be consistent with interpolating
        between the first and last frames.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) frame at time t+1 (middle)
            frame_t2: (B, C, H, W) frame at time t+2
            
        Returns:
            smoothness: (B,) motion smoothness scores
        """
        # Simple linear interpolation baseline
        interpolated = (frame_t + frame_t2) / 2
        
        # Measure similarity with actual middle frame
        diff = F.mse_loss(frame_t1, interpolated, reduction='none')
        diff = diff.view(diff.shape[0], -1).mean(dim=-1)
        
        # Convert to reward (higher is better)
        smoothness = 1.0 / (1.0 + diff)
        
        return smoothness
    
    @torch.no_grad()
    def compute_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        frame_t2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RTC reward for a frame transition.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) frame at time t+1
            frame_t2: Optional (B, C, H, W) frame at time t+2 for motion smoothness
            
        Returns:
            reward: (B,) RTC reward
            info: Dict with metrics
        """
        # CLIP similarity
        clip_sim = self.compute_clip_similarity(frame_t, frame_t1)
        
        info = {'rtc_clip_similarity': clip_sim.mean().item()}
        
        reward = clip_sim
        
        # Add motion smoothness if enabled and we have 3 frames
        if self.use_motion_smoothness and frame_t2 is not None:
            smoothness = self.compute_motion_smoothness(frame_t, frame_t1, frame_t2)
            reward = (reward + smoothness) / 2
            info['rtc_motion_smoothness'] = smoothness.mean().item()
        
        return reward, info
    
    @torch.no_grad()
    def compute_sequence_reward(
        self,
        frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RTC reward for a sequence of frames.
        
        Args:
            frames: (B, T, C, H, W) sequence of frames
            
        Returns:
            rewards: (B, T-1) RTC reward for each transition
            info: Dict with aggregated metrics
        """
        B, T = frames.shape[:2]
        
        rewards = []
        clip_sims = []
        smoothness_scores = []
        
        for t in range(T - 1):
            frame_t2 = frames[:, t + 2] if (t + 2 < T and self.use_motion_smoothness) else None
            
            reward, info = self.compute_reward(
                frames[:, t],
                frames[:, t + 1],
                frame_t2,
            )
            rewards.append(reward)
            clip_sims.append(info['rtc_clip_similarity'])
            if 'rtc_motion_smoothness' in info:
                smoothness_scores.append(info['rtc_motion_smoothness'])
        
        rewards = torch.stack(rewards, dim=1)
        
        info = {
            'rtc_clip_similarity': sum(clip_sims) / len(clip_sims),
        }
        if smoothness_scores:
            info['rtc_motion_smoothness'] = sum(smoothness_scores) / len(smoothness_scores)
        
        return rewards, info


def load_temporal_consistency_reward(
    clip_model_path: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
) -> TemporalConsistencyReward:
    """
    Load temporal consistency reward module.
    
    Args:
        clip_model_path: Path to CLIP model
        device: Device to load on
        
    Returns:
        TemporalConsistencyReward instance
    """
    return TemporalConsistencyReward(
        clip_model_path=clip_model_path,
        device=device,
    )

