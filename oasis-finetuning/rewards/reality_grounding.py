"""
Reality Grounding Reward (RRG) - Anti-Drift Reward for World Models

Prevents world model drift by anchoring generated frames to the real
Minecraft visual manifold. Combines multiple techniques:

1. MineCLIP/Vision Embedding Similarity - Semantic grounding
2. Style/Texture Consistency - Block textures, edges, color distribution  
3. FFT Frequency Matching - Minecraft's characteristic frequency patterns
4. Structural Similarity (SSIM) - Low-level visual consistency

R_reality = w1 * embed_sim + w2 * texture_score + w3 * fft_score + w4 * ssim_score

This reward penalizes:
- Hallucinated non-Minecraft textures
- Smooth/anti-aliased surfaces (Minecraft has hard block edges)
- Frequency drift (wrong texture patterns)
- Semantic drift from Minecraft visual distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("Warning: transformers not available, MineCLIP similarity disabled")


@dataclass
class RealityGroundingWeights:
    """Weights for reality grounding components."""
    embed_sim: float = 1.0      # Vision embedding similarity
    texture: float = 0.5        # Texture/style consistency
    fft: float = 0.3            # FFT frequency matching
    edge: float = 0.2           # Edge sharpness (Minecraft has hard edges)
    
    def normalize(self) -> 'RealityGroundingWeights':
        total = self.embed_sim + self.texture + self.fft + self.edge
        if total == 0:
            return RealityGroundingWeights(0, 0, 0, 0)
        return RealityGroundingWeights(
            embed_sim=self.embed_sim / total,
            texture=self.texture / total,
            fft=self.fft / total,
            edge=self.edge / total,
        )


class RealityGroundingReward(nn.Module):
    """
    Reality Grounding Reward to prevent world model drift.
    
    Uses multiple signals to anchor generated frames to the real
    Minecraft visual distribution:
    
    1. Embedding Similarity: CLIP/MineCLIP cosine similarity
    2. Texture Consistency: Color histogram and texture pattern matching
    3. FFT Matching: Frequency domain similarity (Minecraft has specific patterns)
    4. Edge Sharpness: Minecraft has hard block edges, not smooth gradients
    """
    
    def __init__(
        self,
        clip_model_path: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        # Weights
        embed_weight: float = 1.0,
        texture_weight: float = 0.5,
        fft_weight: float = 0.3,
        edge_weight: float = 0.2,
        normalize_weights: bool = True,
        # Reference distribution
        use_reference_bank: bool = True,
        reference_bank_size: int = 256,
        # Thresholds
        min_edge_sharpness: float = 0.1,  # Minecraft has sharp edges
        max_smooth_ratio: float = 0.3,    # Penalize too-smooth regions
    ):
        super().__init__()
        
        self.device = device
        self.use_reference_bank = use_reference_bank
        self.reference_bank_size = reference_bank_size
        self.min_edge_sharpness = min_edge_sharpness
        self.max_smooth_ratio = max_smooth_ratio
        
        # Set up weights
        self.weights = RealityGroundingWeights(
            embed_sim=embed_weight,
            texture=texture_weight,
            fft=fft_weight,
            edge=edge_weight,
        )
        if normalize_weights:
            self.weights = self.weights.normalize()
        
        # Initialize CLIP for embedding similarity
        self.clip_model = None
        self.clip_processor = None
        if HAS_CLIP and embed_weight > 0:
            try:
                print(f"  Loading CLIP model for reality grounding: {clip_model_path}")
                self.clip_model = CLIPModel.from_pretrained(clip_model_path)
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
                self.clip_model = self.clip_model.to(device)
                self.clip_model.eval()
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                print("  ✓ CLIP model loaded for reality grounding")
            except Exception as e:
                print(f"  Warning: Could not load CLIP model: {e}")
                self.clip_model = None
        
        # Reference embedding bank (running average of real Minecraft frames)
        self.register_buffer(
            'reference_embeddings',
            torch.zeros(reference_bank_size, 512)  # CLIP embedding dim
        )
        self.register_buffer('reference_count', torch.tensor(0))
        self.register_buffer('reference_ptr', torch.tensor(0))
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        
        # Minecraft color palette (approximate block colors for histogram matching)
        # These are common Minecraft colors in RGB [0, 1]
        minecraft_colors = torch.tensor([
            [0.33, 0.49, 0.27],  # Grass green
            [0.53, 0.36, 0.26],  # Dirt brown
            [0.50, 0.50, 0.50],  # Stone gray
            [0.36, 0.25, 0.20],  # Wood brown
            [0.00, 0.47, 0.75],  # Water blue
            [0.53, 0.81, 0.92],  # Sky blue
            [0.13, 0.55, 0.13],  # Leaves green
            [0.96, 0.87, 0.70],  # Sand
            [0.20, 0.20, 0.20],  # Coal/dark
            [1.00, 1.00, 1.00],  # Snow/white
        ], dtype=torch.float32)
        self.register_buffer('minecraft_palette', minecraft_colors)
        
    @torch.no_grad()
    def update_reference_bank(self, real_frames: torch.Tensor):
        """
        Update the reference embedding bank with real Minecraft frames.
        Call this periodically with frames from your dataset.
        
        Args:
            real_frames: (B, C, H, W) real Minecraft frames [0, 1]
        """
        if self.clip_model is None:
            return
            
        # Get embeddings
        embeddings = self._get_clip_embeddings(real_frames)
        
        # Add to bank (circular buffer)
        B = embeddings.shape[0]
        for i in range(B):
            idx = self.reference_ptr.item()
            self.reference_embeddings[idx] = embeddings[i]
            self.reference_ptr = (self.reference_ptr + 1) % self.reference_bank_size
            self.reference_count = min(self.reference_count + 1, self.reference_bank_size)
    
    def _get_clip_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """Get CLIP image embeddings for frames."""
        if self.clip_model is None:
            return torch.zeros(frames.shape[0], 512, device=self.device)
        
        # Frames are (B, C, H, W) in [0, 1]
        # CLIP expects (B, C, H, W) in specific normalization
        B = frames.shape[0]
        
        # Resize to CLIP input size (224x224)
        frames_resized = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize for CLIP (approximate)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        frames_norm = (frames_resized - mean) / std
        
        # Get embeddings
        outputs = self.clip_model.get_image_features(pixel_values=frames_norm)
        embeddings = F.normalize(outputs, dim=-1)
        
        return embeddings
    
    def _compute_embedding_similarity(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between generated frames and reference Minecraft distribution.
        
        Returns:
            similarity: (B,) cosine similarity scores in [0, 1]
        """
        if self.clip_model is None or self.reference_count == 0:
            return torch.ones(frames.shape[0], device=self.device)
        
        # Get embeddings for generated frames
        gen_embeddings = self._get_clip_embeddings(frames)
        
        # Compare to reference bank
        ref_embeddings = self.reference_embeddings[:self.reference_count.item()]
        
        # Compute similarity to mean reference embedding
        ref_mean = F.normalize(ref_embeddings.mean(dim=0, keepdim=True), dim=-1)
        similarity = (gen_embeddings * ref_mean).sum(dim=-1)
        
        # Also compute max similarity to any reference (catches diverse valid states)
        all_sims = torch.mm(gen_embeddings, ref_embeddings.t())
        max_sim = all_sims.max(dim=-1)[0]
        
        # Combine mean and max similarity
        combined_sim = 0.5 * similarity + 0.5 * max_sim
        
        return combined_sim.clamp(0, 1)
    
    def _compute_edge_sharpness(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute edge sharpness score. Minecraft has hard block edges.
        
        Returns:
            sharpness: (B,) edge sharpness scores in [0, 1]
        """
        B = frames.shape[0]
        
        # Apply Sobel filters
        edges_x = F.conv2d(frames, self.sobel_x, padding=1, groups=3)
        edges_y = F.conv2d(frames, self.sobel_y, padding=1, groups=3)
        
        # Edge magnitude
        edge_magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        
        # Mean edge strength (Minecraft should have strong edges)
        mean_edge = edge_magnitude.view(B, -1).mean(dim=-1)
        
        # Edge bimodality: Minecraft edges are either strong or zero (not gradual)
        # Compute ratio of strong edges to weak edges
        strong_edges = (edge_magnitude > 0.3).float().view(B, -1).mean(dim=-1)
        weak_edges = ((edge_magnitude > 0.05) & (edge_magnitude < 0.15)).float().view(B, -1).mean(dim=-1)
        
        # High bimodality = good (strong or no edges, not gradual)
        bimodality = strong_edges / (weak_edges + 0.1)
        bimodality = torch.clamp(bimodality / 5.0, 0, 1)  # Normalize
        
        # Combine mean edge strength and bimodality
        sharpness = 0.5 * torch.clamp(mean_edge * 5, 0, 1) + 0.5 * bimodality
        
        return sharpness
    
    def _compute_texture_consistency(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute texture consistency with Minecraft visual style.
        
        Returns:
            consistency: (B,) texture consistency scores in [0, 1]
        """
        B = frames.shape[0]
        
        # 1. Color histogram similarity to Minecraft palette
        # Compute color histogram
        frames_flat = frames.view(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)
        
        # Distance to nearest Minecraft palette color
        palette = self.minecraft_palette.unsqueeze(0).unsqueeze(0)  # (1, 1, 10, 3)
        frames_exp = frames_flat.unsqueeze(2)  # (B, N, 1, 3)
        
        color_distances = ((frames_exp - palette) ** 2).sum(dim=-1)  # (B, N, 10)
        min_distances = color_distances.min(dim=-1)[0]  # (B, N)
        
        # Fraction of pixels close to Minecraft colors
        close_to_palette = (min_distances < 0.1).float().mean(dim=-1)  # (B,)
        
        # 2. Check for unnatural smoothness (Minecraft is blocky)
        # Compute local variance (low variance = too smooth)
        local_var = self._compute_local_variance(frames)
        
        # Minecraft should have some variance (texture patterns)
        variance_ok = torch.clamp(local_var * 10, 0, 1)
        
        # Combine
        consistency = 0.6 * close_to_palette + 0.4 * variance_ok
        
        return consistency
    
    def _compute_local_variance(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute local variance (detects smooth regions)."""
        B = frames.shape[0]
        
        # Use average pooling to get local mean
        kernel_size = 8
        local_mean = F.avg_pool2d(frames, kernel_size, stride=1, padding=kernel_size//2)
        
        # Local variance
        local_var = F.avg_pool2d((frames - local_mean) ** 2, kernel_size, stride=1, padding=kernel_size//2)
        
        return local_var.view(B, -1).mean(dim=-1)
    
    def _compute_fft_similarity(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT frequency pattern similarity.
        Minecraft has characteristic frequency patterns (block textures).
        
        Returns:
            similarity: (B,) FFT similarity scores in [0, 1]
        """
        B = frames.shape[0]
        
        # Convert to grayscale
        gray = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        
        # Compute 2D FFT
        fft = torch.fft.fft2(gray)
        fft_mag = torch.abs(fft)
        
        # Shift zero frequency to center
        fft_shifted = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        
        # Minecraft has strong low-mid frequencies (block patterns ~16 pixels)
        # and weaker high frequencies (no fine gradients)
        H, W = fft_shifted.shape[-2:]
        cy, cx = H // 2, W // 2
        
        # Create frequency bands
        y_coords = torch.arange(H, device=self.device).view(-1, 1) - cy
        x_coords = torch.arange(W, device=self.device).view(1, -1) - cx
        radius = torch.sqrt(y_coords ** 2 + x_coords ** 2)
        
        # Low freq (blocks), mid freq (textures), high freq (should be low)
        low_mask = (radius < min(H, W) * 0.1).float()
        mid_mask = ((radius >= min(H, W) * 0.1) & (radius < min(H, W) * 0.3)).float()
        high_mask = (radius >= min(H, W) * 0.3).float()
        
        # Compute energy in each band
        low_energy = (fft_shifted * low_mask).view(B, -1).mean(dim=-1)
        mid_energy = (fft_shifted * mid_mask).view(B, -1).mean(dim=-1)
        high_energy = (fft_shifted * high_mask).view(B, -1).mean(dim=-1)
        
        # Minecraft: strong low+mid, weak high
        # Score = (low + mid) / (high + eps)
        ratio = (low_energy + mid_energy) / (high_energy + 1e-6)
        
        # Normalize to [0, 1]
        score = torch.clamp(ratio / 10.0, 0, 1)
        
        return score
    
    def compute_reward(
        self,
        generated_frames: torch.Tensor,
        real_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute reality grounding reward.
        
        Args:
            generated_frames: (B, C, H, W) generated frames in [0, 1]
            real_frames: Optional (B, C, H, W) real frames to update reference bank
            
        Returns:
            reward: (B,) reality grounding reward in [0, 1]
            info: Dict with component scores
        """
        B = generated_frames.shape[0]
        
        # Ensure on correct device
        if generated_frames.device != self.device:
            generated_frames = generated_frames.to(self.device)
        
        # Update reference bank if real frames provided
        if real_frames is not None and self.use_reference_bank:
            self.update_reference_bank(real_frames)
        
        # Compute component scores
        embed_sim = self._compute_embedding_similarity(generated_frames)
        texture_score = self._compute_texture_consistency(generated_frames)
        fft_score = self._compute_fft_similarity(generated_frames)
        edge_score = self._compute_edge_sharpness(generated_frames)
        
        # Weighted combination
        reward = (
            self.weights.embed_sim * embed_sim +
            self.weights.texture * texture_score +
            self.weights.fft * fft_score +
            self.weights.edge * edge_score
        )
        
        info = {
            'reality_embed_sim': embed_sim.mean().item(),
            'reality_texture': texture_score.mean().item(),
            'reality_fft': fft_score.mean().item(),
            'reality_edge': edge_score.mean().item(),
            'reality_total': reward.mean().item(),
            'reference_bank_size': self.reference_count.item(),
        }
        
        return reward, info
    
    def compute_sequence_reward(
        self,
        frames: torch.Tensor,
        real_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute reality grounding reward for a sequence of frames.
        
        Args:
            frames: (B, T, C, H, W) frame sequence
            real_frames: Optional (B, T, C, H, W) real frames
            
        Returns:
            rewards: (B, T) per-frame rewards
            info: Dict with aggregated metrics
        """
        B, T = frames.shape[:2]
        
        all_rewards = []
        all_info = {
            'reality_embed_sim': [],
            'reality_texture': [],
            'reality_fft': [],
            'reality_edge': [],
        }
        
        for t in range(T):
            frame_t = frames[:, t]
            real_t = real_frames[:, t] if real_frames is not None else None
            
            reward_t, info_t = self.compute_reward(frame_t, real_t)
            all_rewards.append(reward_t)
            
            for k in all_info:
                if k in info_t:
                    all_info[k].append(info_t[k])
        
        rewards = torch.stack(all_rewards, dim=1)  # (B, T)
        
        # Aggregate info
        info = {k: np.mean(v) if v else 0.0 for k, v in all_info.items()}
        info['reality_total'] = rewards.mean().item()
        info['reference_bank_size'] = self.reference_count.item()
        
        return rewards, info


def create_reality_grounding_reward(
    clip_model_path: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    embed_weight: float = 1.0,
    texture_weight: float = 0.5,
    fft_weight: float = 0.3,
    edge_weight: float = 0.2,
) -> RealityGroundingReward:
    """
    Create a reality grounding reward module.
    
    Args:
        clip_model_path: Path to CLIP model for embedding similarity
        device: Device to run on
        embed_weight: Weight for embedding similarity
        texture_weight: Weight for texture consistency
        fft_weight: Weight for FFT frequency matching
        edge_weight: Weight for edge sharpness
        
    Returns:
        RealityGroundingReward instance
    """
    return RealityGroundingReward(
        clip_model_path=clip_model_path,
        device=device,
        embed_weight=embed_weight,
        texture_weight=texture_weight,
        fft_weight=fft_weight,
        edge_weight=edge_weight,
    )
