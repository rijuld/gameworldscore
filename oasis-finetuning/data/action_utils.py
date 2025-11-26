"""
Action utilities for Minecraft gameplay.

Provides encoding/decoding for Minecraft actions compatible with
the VPT (Video Pre-Training) format used by Oasis.
"""

from typing import Mapping, Sequence, List, Dict
import torch


# Minecraft action keys (from open-oasis/utils.py)
ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


def one_hot_actions(actions: Sequence[Mapping[str, int]]) -> torch.Tensor:
    """
    Convert action dictionaries to one-hot encoded tensor.
    
    From open-oasis/utils.py - encodes VPT-style action dicts.
    
    Args:
        actions: List of action dictionaries with keys from ACTION_KEYS
        
    Returns:
        actions_one_hot: (T, num_actions) tensor of encoded actions
    """
    actions_one_hot = torch.zeros(len(actions), len(ACTION_KEYS))
    
    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                value = (value - num_buckets) / num_buckets
                assert -1 - 1e-3 <= value <= 1 + 1e-3, \
                    f"Camera action value must be in [-1, 1], got {value}"
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            actions_one_hot[i, j] = value
    
    return actions_one_hot


def decode_actions(actions_tensor: torch.Tensor) -> List[Dict]:
    """
    Decode one-hot encoded actions back to dictionaries.
    
    Args:
        actions_tensor: (T, num_actions) tensor of encoded actions
        
    Returns:
        actions: List of action dictionaries
    """
    actions = []
    
    for i in range(actions_tensor.shape[0]):
        action_dict = {}
        camera = [0, 0]
        
        for j, action_key in enumerate(ACTION_KEYS):
            value = actions_tensor[i, j].item()
            
            if action_key == "cameraX":
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                camera[0] = value * num_buckets + num_buckets
            elif action_key == "cameraY":
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                camera[1] = value * num_buckets + num_buckets
            else:
                action_dict[action_key] = int(round(value))
        
        action_dict["camera"] = camera
        actions.append(action_dict)
    
    return actions


def sample_random_action(batch_size: int = 1) -> torch.Tensor:
    """
    Sample random Minecraft actions.
    
    Args:
        batch_size: Number of actions to sample
        
    Returns:
        actions: (batch_size, num_actions) tensor of random actions
    """
    actions = torch.zeros(batch_size, len(ACTION_KEYS))
    
    for i in range(batch_size):
        # Random discrete actions (binary)
        for j, key in enumerate(ACTION_KEYS):
            if key in ["cameraX", "cameraY"]:
                # Camera is continuous [-1, 1]
                actions[i, j] = torch.rand(1).item() * 2 - 1
            else:
                # Binary actions
                actions[i, j] = float(torch.rand(1).item() > 0.5)
    
    return actions


def action_to_readable(action: torch.Tensor) -> str:
    """
    Convert action tensor to human-readable string.
    
    Args:
        action: (num_actions,) action tensor
        
    Returns:
        readable: Human-readable action description
    """
    parts = []
    
    for j, key in enumerate(ACTION_KEYS):
        value = action[j].item()
        if key in ["cameraX", "cameraY"]:
            if abs(value) > 0.1:
                parts.append(f"{key}={value:.2f}")
        elif value > 0.5:
            parts.append(key)
    
    return " ".join(parts) if parts else "noop"

