"""
Inverse Kinematics Score (RIK) for action fidelity.

Uses a pre-trained Inverse Dynamics Model (IDM) to verify that
the generated frame transition is consistent with the intended action.

The reward is the negative cross-entropy between the intended action
and the IDM's predicted action from the generated transition.
"""

import os
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IDMModel(nn.Module):
    """
    Inverse Dynamics Model for Minecraft.
    
    Predicts the action taken between two consecutive frames.
    Uses the VPT (Video Pre-Training) IDM architecture.
    """
    
    def __init__(
        self,
        model_path: str,
        weights_path: str,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        
        # Load the IDM model using the VPT format
        self.model = self._load_idm_model(model_path, weights_path)
        self.model = self.model.to(device).eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_idm_model(self, model_path: str, weights_path: str):
        """
        Load IDM model from VPT checkpoint format.
        
        The IDM model architecture is defined in the .model file,
        and weights are loaded from the .weights file.
        """
        # For now, create a simple CNN-based IDM
        # In production, this would load the actual VPT IDM architecture
        return SimpleIDM()
    
    @torch.no_grad()
    def forward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict action probabilities from frame transition.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) frame at time t+1
            
        Returns:
            action_logits: (B, num_actions) predicted action logits
        """
        # Concatenate frames channel-wise
        x = torch.cat([frame_t, frame_t1], dim=1)
        logits = self.model(x)
        return logits


class SimpleIDM(nn.Module):
    """
    Simple CNN-based IDM for development/testing.
    
    In production, replace with the actual VPT IDM architecture.
    """
    
    def __init__(
        self,
        in_channels: int = 6,  # Two RGB frames concatenated
        num_actions: int = 25,  # Minecraft action space
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class InverseKinematicsReward(nn.Module):
    """
    Computes the Inverse Kinematics Score (RIK) reward.
    
    RIK = -CrossEntropy(IDM(s_t, s_{t+1}), a_t)
    
    This measures how well the generated transition reflects
    the intended action.
    """
    
    def __init__(
        self,
        idm_model_path: Optional[str] = None,
        idm_weights_path: Optional[str] = None,
        device: str = "cuda",
        action_dim: int = 25,
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        
        if idm_model_path is not None and idm_weights_path is not None:
            self.idm = IDMModel(
                model_path=idm_model_path,
                weights_path=idm_weights_path,
                device=device,
            )
        else:
            # Use simple IDM for development
            self.idm = SimpleIDM(num_actions=action_dim).to(device).eval()
            for param in self.idm.parameters():
                param.requires_grad = False
    
    @torch.no_grad()
    def compute_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        intended_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RIK reward for a batch of transitions.
        
        Args:
            frame_t: (B, C, H, W) frame at time t, values in [0, 1]
            frame_t1: (B, C, H, W) frame at time t+1, values in [0, 1]
            intended_action: (B, action_dim) one-hot encoded intended action
            
        Returns:
            reward: (B,) RIK reward for each sample
            info: Dict with additional metrics
        """
        B = frame_t.shape[0]
        
        # Predict action from transition
        action_logits = self.idm(frame_t, frame_t1)
        
        # Compute cross-entropy with intended action
        if intended_action.dim() == 1:
            # Action indices
            action_idx = intended_action
        else:
            # One-hot or continuous action
            if intended_action.shape[-1] == self.action_dim:
                # One-hot encoded - convert to indices
                action_idx = intended_action.argmax(dim=-1)
            else:
                # Continuous action space - use MSE instead
                action_pred = F.softmax(action_logits, dim=-1)
                mse = F.mse_loss(action_pred, intended_action, reduction='none')
                reward = -mse.mean(dim=-1)
                return reward, {'mse': mse.mean().item()}
        
        # Cross-entropy loss (lower is better, so negate)
        ce_loss = F.cross_entropy(action_logits, action_idx, reduction='none')
        reward = -ce_loss
        
        # Compute accuracy for logging
        pred_action = action_logits.argmax(dim=-1)
        accuracy = (pred_action == action_idx).float().mean()
        
        info = {
            'rik_ce_loss': ce_loss.mean().item(),
            'rik_accuracy': accuracy.item(),
        }
        
        return reward, info
    
    @torch.no_grad()
    def compute_sequence_reward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RIK reward for a sequence of frames.
        
        Args:
            frames: (B, T, C, H, W) sequence of frames
            actions: (B, T-1, action_dim) actions for each transition
            
        Returns:
            rewards: (B, T-1) RIK reward for each transition
            info: Dict with aggregated metrics
        """
        B, T = frames.shape[:2]
        
        rewards = []
        ce_losses = []
        accuracies = []
        
        for t in range(T - 1):
            reward, info = self.compute_reward(
                frames[:, t],
                frames[:, t + 1],
                actions[:, t],
            )
            rewards.append(reward)
            ce_losses.append(info['rik_ce_loss'])
            accuracies.append(info['rik_accuracy'])
        
        rewards = torch.stack(rewards, dim=1)
        
        info = {
            'rik_ce_loss': np.mean(ce_losses),
            'rik_accuracy': np.mean(accuracies),
        }
        
        return rewards, info


def load_idm_from_vpt(
    model_path: str,
    weights_path: str,
    device: str = "cuda",
) -> InverseKinematicsReward:
    """
    Load IDM from VPT checkpoint files.
    
    Args:
        model_path: Path to .model file
        weights_path: Path to .weights file
        device: Device to load on
        
    Returns:
        InverseKinematicsReward instance
    """
    return InverseKinematicsReward(
        idm_model_path=model_path,
        idm_weights_path=weights_path,
        device=device,
    )

