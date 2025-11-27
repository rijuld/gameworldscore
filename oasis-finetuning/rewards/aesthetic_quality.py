"""
Aesthetic Quality Score (RAQ) for visual quality assessment.

Combines:
1. MUSIQ (Multi-scale Image Quality Transformer) for technical quality
2. LAION Aesthetic Predictor for aesthetic appeal

Both models predict quality/aesthetic scores for generated frames.
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AestheticPredictor(nn.Module):
    """
    LAION Aesthetic Predictor.
    
    Predicts aesthetic scores based on CLIP embeddings.
    Uses a linear probe trained on the AVA dataset.
    """
    
    def __init__(
        self,
        clip_embedding_dim: int = 768,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Simple MLP for aesthetic prediction
        self.mlp = nn.Sequential(
            nn.Linear(clip_embedding_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, path: str):
        """Load pretrained weights."""
        state_dict = torch.load(path, weights_only=True)
        
        # Handle different checkpoint formats
        # Some checkpoints have 'layers.' prefix, some don't
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'layers.' prefix if present
            new_key = key.replace('layers.', '') if key.startswith('layers.') else key
            new_state_dict[new_key] = value
        
        self.mlp.load_state_dict(new_state_dict)
    
    def forward(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict aesthetic score from CLIP embeddings.
        
        Args:
            clip_embeddings: (B, embedding_dim) CLIP image embeddings
            
        Returns:
            scores: (B,) aesthetic scores
        """
        return self.mlp(clip_embeddings).squeeze(-1)


class SimpleMUSIQ(nn.Module):
    """
    Simplified MUSIQ-like quality predictor.
    
    Uses a CNN backbone to predict image quality scores.
    In production, replace with actual MUSIQ implementation.
    """
    
    def __init__(self):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict quality scores.
        
        Args:
            images: (B, C, H, W) images in [0, 1]
            
        Returns:
            scores: (B,) quality scores
        """
        return self.backbone(images).squeeze(-1)


class AestheticQualityReward(nn.Module):
    """
    Computes the Aesthetic Quality Score (RAQ) reward.
    
    RAQ = w_musiq * MUSIQ_score + w_aesthetic * Aesthetic_score
    
    Both components are normalized to [0, 1] range.
    """
    
    def __init__(
        self,
        clip_model_path: str = "openai/clip-vit-large-patch14",
        aesthetic_checkpoint: Optional[str] = None,
        device: str = "cuda",
        musiq_weight: float = 0.5,
        aesthetic_weight: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.musiq_weight = musiq_weight
        self.aesthetic_weight = aesthetic_weight
        
        # Load CLIP for aesthetic prediction embeddings
        from transformers import CLIPModel, CLIPProcessor
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.clip_model = self.clip_model.to(device).eval()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # CLIP normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        
        # Aesthetic predictor
        clip_dim = self.clip_model.config.projection_dim
        self.aesthetic_predictor = AestheticPredictor(
            clip_embedding_dim=clip_dim,
            checkpoint_path=aesthetic_checkpoint,
        ).to(device).eval()
        
        for param in self.aesthetic_predictor.parameters():
            param.requires_grad = False
        
        # Image quality predictor
        self.musiq = SimpleMUSIQ().to(device).eval()
        
        for param in self.musiq.parameters():
            param.requires_grad = False
    
    def _preprocess_for_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for CLIP."""
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = (images - self.mean) / self.std
        return images
    
    @torch.no_grad()
    def compute_aesthetic_score(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute aesthetic score using CLIP + aesthetic predictor.
        
        Args:
            images: (B, C, H, W) images in [0, 1]
            
        Returns:
            scores: (B,) aesthetic scores
        """
        # Get CLIP embeddings
        images_clip = self._preprocess_for_clip(images)
        embeddings = self.clip_model.get_image_features(images_clip)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Predict aesthetic score
        scores = self.aesthetic_predictor(embeddings)
        
        # Normalize to [0, 1] using sigmoid
        scores = torch.sigmoid(scores / 5.0)  # Assuming raw scores around [-10, 10]
        
        return scores
    
    @torch.no_grad()
    def compute_quality_score(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute image quality score using MUSIQ-like model.
        
        Args:
            images: (B, C, H, W) images in [0, 1]
            
        Returns:
            scores: (B,) quality scores
        """
        # Resize for quality prediction
        images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        
        scores = self.musiq(images)
        
        # Normalize to [0, 1]
        scores = torch.sigmoid(scores)
        
        return scores
    
    @torch.no_grad()
    def compute_reward(
        self,
        frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RAQ reward for a batch of frames.
        
        Args:
            frame: (B, C, H, W) frames in [0, 1]
            
        Returns:
            reward: (B,) RAQ reward
            info: Dict with metrics
        """
        aesthetic_score = self.compute_aesthetic_score(frame)
        quality_score = self.compute_quality_score(frame)
        
        # Weighted combination
        reward = (
            self.musiq_weight * quality_score +
            self.aesthetic_weight * aesthetic_score
        )
        
        info = {
            'raq_aesthetic_score': aesthetic_score.mean().item(),
            'raq_quality_score': quality_score.mean().item(),
        }
        
        return reward, info
    
    @torch.no_grad()
    def compute_sequence_reward(
        self,
        frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RAQ reward for a sequence of frames.
        
        Args:
            frames: (B, T, C, H, W) sequence of frames
            
        Returns:
            rewards: (B, T) RAQ reward for each frame
            info: Dict with aggregated metrics
        """
        B, T = frames.shape[:2]
        
        rewards = []
        aesthetic_scores = []
        quality_scores = []
        
        for t in range(T):
            reward, info = self.compute_reward(frames[:, t])
            rewards.append(reward)
            aesthetic_scores.append(info['raq_aesthetic_score'])
            quality_scores.append(info['raq_quality_score'])
        
        rewards = torch.stack(rewards, dim=1)
        
        info = {
            'raq_aesthetic_score': sum(aesthetic_scores) / len(aesthetic_scores),
            'raq_quality_score': sum(quality_scores) / len(quality_scores),
        }
        
        return rewards, info


def load_aesthetic_quality_reward(
    clip_model_path: str = "openai/clip-vit-large-patch14",
    aesthetic_checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> AestheticQualityReward:
    """
    Load aesthetic quality reward module.
    
    Args:
        clip_model_path: Path to CLIP model
        aesthetic_checkpoint: Path to aesthetic predictor checkpoint
        device: Device to load on
        
    Returns:
        AestheticQualityReward instance
    """
    return AestheticQualityReward(
        clip_model_path=clip_model_path,
        aesthetic_checkpoint=aesthetic_checkpoint,
        device=device,
    )

