"""
GameWorldScore: Unified ground-truth-free reward function.

Combines three components from the Matrix-Game GameWorldScore benchmark:
1. RIK (Inverse Kinematics Score) - Action fidelity
2. RTC (Temporal Consistency Score) - Temporal smoothness
3. RAQ (Aesthetic Quality Score) - Visual quality

R_total = w1 * RIK + w2 * RTC + w3 * RAQ

All components are normalized to [0, 1] range:
- RIK: exp(-cross_entropy_loss), where 1 = perfect action match
- RTC: (cosine_similarity + 1) / 2, where 1 = identical frames
- RAQ: sigmoid-normalized aesthetic/quality scores

This enables meaningful weight combinations and consistent training.
Ideal total reward: ~0.7-0.9 for well-trained models.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from .inverse_kinematics import InverseKinematicsReward
from .temporal_consistency import TemporalConsistencyReward
from .aesthetic_quality import AestheticQualityReward


@dataclass
class RewardWeights:
    """Weights for combining reward components."""
    rik: float = 1.0
    rtc: float = 1.0
    raq: float = 1.0
    
    def normalize(self) -> 'RewardWeights':
        """Normalize weights to sum to 1."""
        total = self.rik + self.rtc + self.raq
        if total == 0:
            return RewardWeights(0.0, 0.0, 0.0)
        return RewardWeights(
            rik=self.rik / total,
            rtc=self.rtc / total,
            raq=self.raq / total,
        )


class GameWorldScoreReward(nn.Module):
    """
    GameWorldScore: Ground-truth-free reward for world model training.
    
    This unified reward function evaluates generated frames based on:
    - Action fidelity (RIK): Does the transition match the intended action?
    - Temporal consistency (RTC): Are consecutive frames smooth and coherent?
    - Aesthetic quality (RAQ): Does the frame look visually good?
    
    Unlike ground-truth comparison methods, this reward can be computed
    purely from generated outputs, enabling RL training on long-horizon
    rollouts without access to future ground-truth frames.
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
        normalize_weights: bool = True,
        # Settings
        device: str = "cuda",
        action_dim: int = 25,
        use_motion_smoothness: bool = False,
        require_vpt: bool = True,
    ):
        super().__init__()
        self.device = device
        
        # Store weights
        self.weights = RewardWeights(rik=rik_weight, rtc=rtc_weight, raq=raq_weight)
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
            clip_model_path=clip_model_path,
            device=device,
            use_motion_smoothness=use_motion_smoothness,
        )
        
        # Share CLIP model between RTC and RAQ to save ~1.7GB GPU memory
        self.raq = AestheticQualityReward(
            clip_model_path=clip_model_path,
            aesthetic_checkpoint=aesthetic_checkpoint,
            device=device,
            shared_clip_model=self.rtc.clip_model,  # Reuse RTC's CLIP
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
        
        # Weighted combination: (B, T-1)
        frame_rewards = (
            self.weights.rik * rik_rewards +
            self.weights.rtc * rtc_rewards +
            self.weights.raq * raq_rewards
        )
        
        # Aggregate info
        info = {
            'game_world_score': frame_rewards.mean().item(),
            'rik_reward': rik_rewards.mean().item(),
            'rtc_reward': rtc_rewards.mean().item(),
            'raq_reward': raq_rewards.mean().item(),
            # Timing info
            'time_rik_sec': rik_time_total,
            'time_rtc_sec': rtc_time_total,
            'time_raq_sec': raq_time_total,
        }
        
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
    require_vpt: bool = True,
    use_motion_smoothness: bool = False,
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
        device=device,
        require_vpt=require_vpt,
        use_motion_smoothness=use_motion_smoothness,
    )

