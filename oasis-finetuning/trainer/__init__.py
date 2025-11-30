"""
Training components for Oasis GRPO finetuning.
Config is loaded from config/default.yaml (single source of truth).
"""

from .oasis_grpo_trainer import OasisGRPOTrainer
from config.loader import OasisGRPOConfig, load_config, print_config

# Backwards compatibility aliases
OasisPPOTrainer = OasisGRPOTrainer
OasisPPOConfig = OasisGRPOConfig

__all__ = [
    "OasisGRPOTrainer",
    "OasisGRPOConfig",
    "load_config",
    "print_config",
    # Backwards compatibility
    "OasisPPOTrainer",
    "OasisPPOConfig",
]

