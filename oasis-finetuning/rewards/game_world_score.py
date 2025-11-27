"""
GameWorldScore: Unified ground-truth-free reward function.

Combines three components from the Matrix-Game GameWorldScore benchmark:
1. RIK (Inverse Kinematics Score) - Action fidelity
2. RTC (Temporal Consistency Score) - Temporal smoothness
3. RAQ (Aesthetic Quality Score) - Visual quality

R_total = w1 * RIK + w2 * RTC + w3 * RAQ

This reward function enables RL finetuning without ground-truth frames.
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
        
        Args:
            frames: (B, T, C, H, W) sequence of frames (first is context)
            actions: (B, T-1, action_dim) actions for each transition
            return_per_frame: If True, return rewards for each frame
            
        Returns:
            reward: (B,) or (B, T-1) total/per-frame rewards
            info: Dict with aggregated metrics (includes timing)
        """
        import time
        
        B, T = frames.shape[:2]
        
        frame_rewards = []
        rik_rewards = []
        rtc_rewards = []
        raq_rewards = []
        
        # Timing accumulators
        rik_time_total = 0.0
        rtc_time_total = 0.0
        raq_time_total = 0.0
        
        all_info = {
            'rik_ce_loss': [],
            'rik_accuracy': [],
            'rtc_clip_similarity': [],
            'raq_aesthetic_score': [],
            'raq_quality_score': [],
        }
        
        for t in range(T - 1):
            frame_t = frames[:, t]
            frame_t1 = frames[:, t + 1]
            action = actions[:, t]
            frame_t2 = frames[:, t + 2] if t + 2 < T else None
            
            # Time RIK
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            rik_start = time.perf_counter()
            rik_reward, rik_info = self.rik.compute_reward(frame_t, frame_t1, action)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            rik_time_total += time.perf_counter() - rik_start
            
            # Time RTC
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            rtc_start = time.perf_counter()
            rtc_reward, rtc_info = self.rtc.compute_reward(frame_t, frame_t1, frame_t2)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            rtc_time_total += time.perf_counter() - rtc_start
            
            # Time RAQ
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            raq_start = time.perf_counter()
            raq_reward, raq_info = self.raq.compute_reward(frame_t1)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            raq_time_total += time.perf_counter() - raq_start
            
            # Weighted combination
            total_reward = (
                self.weights.rik * rik_reward +
                self.weights.rtc * rtc_reward +
                self.weights.raq * raq_reward
            )
            
            frame_rewards.append(total_reward)
            rik_rewards.append(rik_reward.mean().item())
            rtc_rewards.append(rtc_reward.mean().item())
            raq_rewards.append(raq_reward.mean().item())
            
            for key in all_info:
                if key.startswith('rik_') and key in rik_info:
                    all_info[key].append(rik_info[key])
                elif key.startswith('rtc_') and key in rtc_info:
                    all_info[key].append(rtc_info[key])
                elif key.startswith('raq_') and key in raq_info:
                    all_info[key].append(raq_info[key])
        
        frame_rewards = torch.stack(frame_rewards, dim=1)  # (B, T-1)
        
        # Aggregate info
        info = {
            'game_world_score': frame_rewards.mean().item(),
            'rik_reward': sum(rik_rewards) / len(rik_rewards),
            'rtc_reward': sum(rtc_rewards) / len(rtc_rewards),
            'raq_reward': sum(raq_rewards) / len(raq_rewards),
            # Timing info
            'time_rik_sec': rik_time_total,
            'time_rtc_sec': rtc_time_total,
            'time_raq_sec': raq_time_total,
        }
        
        for key, values in all_info.items():
            if values:
                info[key] = sum(values) / len(values)
        
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
    
    idm_model_path = os.path.join(models_dir, "4x_idm.model")
    idm_weights_path = os.path.join(models_dir, "4x_idm.weights")
    if not os.path.exists(idm_model_path):
        idm_model_path = None
        idm_weights_path = None
    
    return GameWorldScoreReward(
        idm_model_path=idm_model_path,
        idm_weights_path=idm_weights_path,
        clip_model_path=clip_path,
        aesthetic_checkpoint=aesthetic_path,
        rik_weight=rik_weight,
        rtc_weight=rtc_weight,
        raq_weight=raq_weight,
        device=device,
    )

