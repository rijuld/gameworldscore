"""
Oasis RL Finetuning Pipeline

This package provides a unified RL finetuning pipeline for the Oasis world model,
integrating components from the Oasis repository and RLVR-World training infrastructure.

Key Components:
- OasisPolicy: Wrapper around Oasis DiT for RL-compatible policy interface
- GameWorldScoreReward: Ground-truth-free reward composed of RIK, RTC, and RAQ
- OasisRolloutWorker: Long-horizon rollout worker for Oasis
- OasisPPOTrainer: PPO/GRPO trainer for Oasis world model
"""

from .models.oasis_policy import OasisPolicy
from .rewards.game_world_score import GameWorldScoreReward
from .trainer.oasis_ppo_trainer import OasisPPOTrainer

__version__ = "0.1.0"
__all__ = ["OasisPolicy", "GameWorldScoreReward", "OasisPPOTrainer"]

