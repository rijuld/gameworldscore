"""
GameWorldScore: Unified ground-truth-free reward function.

Combines four components for world model training:
1. RIK (Inverse Kinematics Score) - Action fidelity
2. RTC (Temporal Consistency Score) - Temporal smoothness
3. RAQ (Aesthetic Quality Score) - Visual quality
4. RRG (Reality Grounding Score) - Anti-drift anchoring

R_total = w1 * RIK + w2 * RTC + w3 * RAQ + w4 * RRG

All components are normalized to [0, 1] range:
- RIK: exp(-cross_entropy_loss), where 1 = perfect action match
- RTC: (cosine_similarity + 1) / 2, where 1 = identical frames
- RAQ: sigmoid-normalized aesthetic/quality scores
- RRG: sigmoid-normalized reality/grounding scores

This enables meaningful weight combinations and consistent training.
Ideal total reward: ~0.7-0.9 for well-trained models.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from .inverse_kinematics import InverseKinematicsReward
from .temporal_consistency import TemporalConsistencyRewardV2 as TemporalConsistencyReward
from .temporal_consistency import load_temporal_consistency_reward_v2
from .aesthetic_quality import AestheticQualityReward
from .reality_grounding import RealityGroundingReward
from .anti_drift import AntiDriftReward


@dataclass
class RewardWeights:
    """Weights for combining reward components."""
    rik: float = 1.0
    rtc: float = 1.0
    raq: float = 1.0
    rrg: float = 0.0  # Reality grounding (anti-drift)
    anti_drift: float = 0.0  # New Anti-Drift Reward
    
    def normalize(self) -> 'RewardWeights':
        """Normalize weights to sum to 1."""
        total = self.rik + self.rtc + self.raq + self.rrg + self.anti_drift
        if total == 0:
            return RewardWeights(0.0, 0.0, 0.0, 0.0, 0.0)
        return RewardWeights(
            rik=self.rik / total,
            rtc=self.rtc / total,
            raq=self.raq / total,
            rrg=self.rrg / total,
            anti_drift=self.anti_drift / total,
        )


class GameWorldScoreReward(nn.Module):
    """
    GameWorldScore: Ground-truth-free reward for world model training.
    
    This unified reward function evaluates generated frames based on:
    - Action fidelity (RIK): Does the transition match the intended action?
    - Temporal consistency (RTC): Are consecutive frames smooth and coherent?
    - Aesthetic quality (RAQ): Does the frame look visually good?
    - Reality Grounding (RRG): Anti-drift anchoring.
    - Anti-Drift (AD): Sharpness, motion, texture, anti-grid.
    """
    
    def __init__(
        self,
        # Model paths
        idm_model_path: Optional[str] = None,
        idm_weights_path: Optional[str] = None,
        clip_model_path: str = "openai/clip-vit-large-patch14",
        aesthetic_checkpoint: Optional[str] = None,
        # Weights
        rik_weight: float = 1.0,
        rtc_weight: float = 1.0,
        raq_weight: float = 1.0,
        rrg_weight: float = 0.0,  # Reality grounding (anti-drift)
        anti_drift_weight: float = 0.0, # New Anti-Drift
        normalize_weights: bool = True,
        # Settings
        device: str = "cuda",
        action_dim: int = 25,
        require_vpt: bool = True,
    ):
        super().__init__()
        self.device = device
        
        # Store weights
        self.weights = RewardWeights(
            rik=rik_weight, rtc=rtc_weight, raq=raq_weight, rrg=rrg_weight, anti_drift=anti_drift_weight
        )
        if normalize_weights:
            self.weights = self.weights.normalize()
        
        # Initialize reward components
        self.rik = InverseKinematicsReward(
            idm_model_path=idm_model_path,
            idm_weights_path=idm_weights_path,
            device=device,
            action_dim=action_dim,
            require_vpt=require_vpt,
        )
        
        self.rtc = TemporalConsistencyReward(
            device=device,
            alpha=30.0, # Default alpha
        )
        
        # Note: RTC now uses RAFT optical flow instead of CLIP
        # RAQ uses CLIP for aesthetic scoring, so it loads its own CLIP model
        self.raq = AestheticQualityReward(
            clip_model_path=clip_model_path,
            aesthetic_checkpoint=aesthetic_checkpoint,
            device=device,
            shared_clip_model=None,  # RAQ will load its own CLIP model
        )
        
        # RRG: Reality Grounding Reward (anti-drift)
        # Uses lighter CLIP model to save memory
        self.rrg = None
        if rrg_weight > 0:
            self.rrg = RealityGroundingReward(
                clip_model_path="openai/clip-vit-base-patch32",  # Lighter model
                device=device,
                embed_weight=1.0,
                texture_weight=0.5,
                fft_weight=0.05,  # FIXED: Reduced to prevent grid artifacts
                edge_weight=1.0,  # FIXED: Increased to force sharpness (anti-blur)
            )
            
        # Anti-Drift Reward
        self.anti_drift = None
        if anti_drift_weight > 0:
            self.anti_drift = AntiDriftReward(
                device=device,
                sharpness_weight=1.0,
                motion_weight=2.0,
                texture_weight=0.8,
                anti_grid_weight=0.5,
            )
    
    @torch.no_grad()
    def compute_frame_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        action: torch.Tensor,
        frame_t2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GameWorldScore reward for a single transition.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) generated frame at time t+1
            action: (B, action_dim) action taken from t to t+1
            frame_t2: Optional (B, C, H, W) frame at t+2 for motion smoothness
            
        Returns:
            reward: (B,) total GameWorldScore reward
            info: Dict with component scores and metrics
        """
        # Compute RIK (action fidelity)
        rik_reward, rik_info = self.rik.compute_reward(frame_t, frame_t1, action)
        
        # Compute RTC (temporal consistency)
        rtc_reward, rtc_info = self.rtc.compute_reward(frame_t, frame_t1, frame_t2)
        
        # Compute RAQ (aesthetic quality) - only for generated frame
        raq_reward, raq_info = self.raq.compute_reward(frame_t1)
        
        # Combine rewards
        total_reward = (
            self.weights.rik * rik_reward +
            self.weights.rtc * rtc_reward +
            self.weights.raq * raq_reward
        )
        
        # Aggregate info
        info = {
            'game_world_score': total_reward.mean().item(),
            'rik_reward': rik_reward.mean().item(),
            'rtc_reward': rtc_reward.mean().item(),
            'raq_reward': raq_reward.mean().item(),
            **rik_info,
            **rtc_info,
            **raq_info,
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
        Compute GameWorldScore reward for a sequence of frames.
        
        OPTIMIZED: Uses batched operations, ensures all computations stay on GPU.
        
        Args:
            frames: (B, T, C, H, W) sequence of frames (first is context)
            actions: (B, T-1, action_dim) actions for each transition
            return_per_frame: If True, return rewards for each frame
            
        Returns:
            reward: (B,) or (B, T-1) total/per-frame rewards
            info: Dict with aggregated metrics (includes timing)
        """
        # Ensure inputs are on GPU (reward models will handle device internally)
        if frames.device.type != self.device:
            frames = frames.to(self.device)
        if actions.device.type != self.device:
            actions = actions.to(self.device)
        
        import time
        
        # Time RIK (batched)
        rik_start = time.perf_counter()
        if self.weights.rik > 0:
            rik_rewards, rik_info = self.rik.compute_sequence_reward(frames, actions)
        else:
            rik_rewards = torch.zeros(frames.shape[0], frames.shape[1]-1, device=self.device)
            rik_info = {}
        rik_time_total = time.perf_counter() - rik_start
        
        # Time RTC (batched)
        rtc_start = time.perf_counter()
        if self.weights.rtc > 0:
            rtc_rewards, rtc_info = self.rtc.compute_sequence_reward(frames)
        else:
            rtc_rewards = torch.zeros(frames.shape[0], frames.shape[1]-1, device=self.device)
            rtc_info = {}
        rtc_time_total = time.perf_counter() - rtc_start
        
        # Time RAQ (batched)
        raq_start = time.perf_counter()
        if self.weights.raq > 0:
            raq_rewards_all, raq_info = self.raq.compute_sequence_reward(frames)
            # RAQ is (B, T), but we need (B, T-1) to match transitions
            raq_rewards = raq_rewards_all[:, 1:]  # Skip first frame (context)
        else:
            raq_rewards = torch.zeros(frames.shape[0], frames.shape[1]-1, device=self.device)
            raq_info = {}
        raq_time_total = time.perf_counter() - raq_start
        
        # Time RRG (Reality Grounding - anti-drift)
        rrg_start = time.perf_counter()
        if self.weights.rrg > 0 and self.rrg is not None:
            # RRG evaluates generated frames (skip first context frame)
            gen_frames = frames[:, 1:]  # (B, T-1, C, H, W)
            B, T_gen = gen_frames.shape[:2]
            # Flatten for batch processing
            gen_frames_flat = gen_frames.reshape(B * T_gen, *gen_frames.shape[2:])
            rrg_rewards_flat, rrg_info = self.rrg.compute_reward(gen_frames_flat)
            rrg_rewards = rrg_rewards_flat.reshape(B, T_gen)
        else:
            rrg_rewards = torch.zeros(frames.shape[0], frames.shape[1]-1, device=self.device)
            rrg_info = {}
        rrg_time_total = time.perf_counter() - rrg_start
        
        # Time Anti-Drift
        ad_start = time.perf_counter()
        if self.weights.anti_drift > 0 and self.anti_drift is not None:
            # User's AntiDriftReward requires return_per_frame=True to get (B, T-1)
            ad_rewards, ad_info = self.anti_drift.compute_sequence_reward(
                frames, actions, return_per_frame=True
            )
        else:
            ad_rewards = torch.zeros(frames.shape[0], frames.shape[1]-1, device=self.device)
            ad_info = {}
        ad_time_total = time.perf_counter() - ad_start
        
        # Weighted combination: (B, T-1)
        frame_rewards = (
            self.weights.rik * rik_rewards +
            self.weights.rtc * rtc_rewards +
            self.weights.raq * raq_rewards +
            self.weights.rrg * rrg_rewards +
            self.weights.anti_drift * ad_rewards
        )
        
        # Aggregate info
        info = {
            'game_world_score': frame_rewards.mean().item(),
            'rik_reward': rik_rewards.mean().item(),
            'rtc_reward': rtc_rewards.mean().item(),
            'raq_reward': raq_rewards.mean().item(),
            'rrg_reward': rrg_rewards.mean().item(),
            'anti_drift_reward': ad_rewards.mean().item(),
            # Timing info
            'time_rik_sec': rik_time_total,
            'time_rtc_sec': rtc_time_total,
            'time_raq_sec': raq_time_total,
            'time_rrg_sec': rrg_time_total,
            'time_ad_sec': ad_time_total,
        }
        
        # Add Anti-Drift sub-info
        for k, v in ad_info.items():
            info[k] = v
        
        # Add sub-reward info if available
        if 'rik_ce_loss' in rik_info:
            info['rik_ce_loss'] = rik_info.get('rik_ce_loss', 0.0)
        if 'rik_accuracy' in rik_info:
            info['rik_accuracy'] = rik_info.get('rik_accuracy', 0.0)
        if 'rtc_clip_similarity' in rtc_info:
            info['rtc_clip_similarity'] = rtc_info.get('rtc_clip_similarity', 0.0)
        if 'raq_aesthetic_score' in raq_info:
            info['raq_aesthetic_score'] = raq_info.get('raq_aesthetic_score', 0.0)
        if 'raq_quality_score' in raq_info:
            info['raq_quality_score'] = raq_info.get('raq_quality_score', 0.0)
        # Add RRG sub-info
        for key in ['reality_embed_sim', 'reality_texture', 'reality_fft', 'reality_edge']:
            if key in rrg_info:
                info[key] = rrg_info[key]
        
        if return_per_frame:
            return frame_rewards, info
        else:
            return frame_rewards.sum(dim=1), info
    
    def compute_token_level_reward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        response_length: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute token-level rewards for RLVR-World compatibility.
        
        In RLVR-World's framework, rewards are typically sparse and
        placed at specific positions. For Oasis, we assign rewards
        at frame boundaries.
        
        Args:
            frames: (B, T, C, H, W) generated frames
            actions: (B, T-1, action_dim) actions
            response_length: Length of response sequence (for padding)
            
        Returns:
            token_level_rewards: (B, response_length) reward tensor
            info: Dict with metrics
        """
        B = frames.shape[0]
        
        # Compute per-frame rewards
        frame_rewards, info = self.compute_sequence_reward(
            frames, actions, return_per_frame=True
        )  # (B, T-1)
        
        # Create token-level reward tensor
        # For Oasis, each frame corresponds to some number of tokens
        # We place the reward at the end of each frame's tokens
        token_level_rewards = torch.zeros(B, response_length, device=self.device)
        
        T = frames.shape[1] - 1  # Number of generated frames
        tokens_per_frame = response_length // T if T > 0 else response_length
        
        for t in range(T):
            # Place reward at the last token of each frame
            reward_position = min((t + 1) * tokens_per_frame - 1, response_length - 1)
            token_level_rewards[:, reward_position] = frame_rewards[:, t]
        
        return token_level_rewards, info


def create_game_world_score_reward(
    models_dir: str = "models_for_rl_finetuning",
    device: str = "cuda",
    rik_weight: float = 1.0,
    rtc_weight: float = 1.0,
    raq_weight: float = 1.0,
    rrg_weight: float = 0.0,
    anti_drift_weight: float = 0.0,
    require_vpt: bool = True,
) -> GameWorldScoreReward:
    """
    Create GameWorldScore reward with models from the specified directory.
    
    Expected directory structure:
    - models_dir/clip-vit-large-patch14/  (CLIP model, ViT-L/14 for 768-dim embeddings)
    - models_dir/aesthetic_predictor.pth  (Aesthetic predictor weights)
    - models_dir/4x_idm.model  (IDM model definition)
    - models_dir/4x_idm.weights  (IDM weights)
    
    Args:
        models_dir: Directory containing reward model checkpoints
        device: Device to load models on
        rik_weight: Weight for RIK component
        rtc_weight: Weight for RTC component
        raq_weight: Weight for RAQ component
        rrg_weight: Weight for RRG component (Reality Grounding, anti-drift)
        anti_drift_weight: Weight for Anti-Drift component
        require_vpt: If True, raise error when VPT IDM cannot be loaded.
                     If False, fall back to SimpleIDM (less accurate).
        
    Returns:
        GameWorldScoreReward instance
    """
    import os
    
    # Use ViT-L/14 to match aesthetic predictor (768-dim embeddings)
    clip_path = os.path.join(models_dir, "clip-vit-large-patch14")
    if not os.path.exists(clip_path):
        clip_path = "openai/clip-vit-large-patch14"  # Fall back to hub
    
    aesthetic_path = os.path.join(models_dir, "aesthetic_predictor.pth")
    if not os.path.exists(aesthetic_path):
        aesthetic_path = None
    
    # VPT IDM (downloaded by download_reward_models.py)
    idm_model_path = os.path.join(models_dir, "4x_idm.model")
    idm_weights_path = os.path.join(models_dir, "4x_idm.weights")
    
    # If not found and require_vpt=True, InverseKinematicsReward will raise an error
    
    return GameWorldScoreReward(
        idm_model_path=idm_model_path,
        idm_weights_path=idm_weights_path,
        clip_model_path=clip_path,
        aesthetic_checkpoint=aesthetic_path,
        rik_weight=rik_weight,
        rtc_weight=rtc_weight,
        raq_weight=raq_weight,
        rrg_weight=rrg_weight,
        anti_drift_weight=anti_drift_weight,
        device=device,
        require_vpt=require_vpt,
    )

