"""
Training components for Oasis GRPO finetuning.
"""

from .oasis_grpo_trainer import OasisGRPOTrainer, OasisGRPOConfig, create_oasis_grpo_trainer

# Backwards compatibility aliases
OasisPPOTrainer = OasisGRPOTrainer
OasisPPOConfig = OasisGRPOConfig

__all__ = [
    "OasisGRPOTrainer",
    "OasisGRPOConfig", 
    "create_oasis_grpo_trainer",
    # Backwards compatibility
    "OasisPPOTrainer",
    "OasisPPOConfig",
]

