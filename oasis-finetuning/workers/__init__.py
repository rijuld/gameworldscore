"""
Worker classes for Oasis RL finetuning.
"""

from .oasis_actor import OasisActorWorker
from .oasis_rollout import OasisRolloutWorker

__all__ = ["OasisActorWorker", "OasisRolloutWorker"]

