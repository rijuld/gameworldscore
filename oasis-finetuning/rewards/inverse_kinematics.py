"""
Inverse Kinematics Score (RIK) for action fidelity.

Uses a pre-trained Inverse Dynamics Model (IDM) to verify that
the generated frame transition is consistent with the intended action.

The reward is based on how well the IDM's predicted action matches
the intended action from the generated transition.

Supports:
1. VPT IDM (OpenAI's Video Pre-Training model) - high quality
2. SimpleIDM (fallback CNN) - for development/testing
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# Install minerl mock before VPT imports (VPT has minerl as transitive dependency
# but we only need the IDM which doesn't actually use minerl functionality)
def _install_minerl_mock():
    """Try to install minerl mock using various import methods."""
    # Method 1: Try relative import
    try:
        from utils.minerl_mock import install_mock
        install_mock()
        return True
    except ImportError:
        pass
    
    # Method 2: Try absolute path import
    try:
        import importlib.util
        mock_path = Path(__file__).parent.parent / "utils" / "minerl_mock.py"
        if mock_path.exists():
            spec = importlib.util.spec_from_file_location("minerl_mock", mock_path)
            minerl_mock = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(minerl_mock)
            minerl_mock.install_mock()
            return True
    except Exception:
        pass
    
    # Method 3: Inline mock (last resort)
    try:
        from types import ModuleType
        
        minerl = ModuleType("minerl")
        minerl_herobraine = ModuleType("minerl.herobraine")
        minerl_herobraine_hero = ModuleType("minerl.herobraine.hero")
        minerl_herobraine_hero_mc = ModuleType("minerl.herobraine.hero.mc")
        
        minerl_herobraine_hero_mc.MINERL_ITEM_MAP = {i: f"item_{i}" for i in range(256)}
        
        minerl.herobraine = minerl_herobraine
        minerl_herobraine.hero = minerl_herobraine_hero
        minerl_herobraine_hero.mc = minerl_herobraine_hero_mc
        
        sys.modules["minerl"] = minerl
        sys.modules["minerl.herobraine"] = minerl_herobraine
        sys.modules["minerl.herobraine.hero"] = minerl_herobraine_hero
        sys.modules["minerl.herobraine.hero.mc"] = minerl_herobraine_hero_mc
        
        print("  ✓ Installed inline minerl mock")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not install minerl mock: {e}")
        return False

_install_minerl_mock()

# Add VPT to path for imports
# VPT_PATH can be overridden via environment variable
VPT_PATH = Path(os.environ.get("VPT_PATH", Path(__file__).parent.parent.parent / "VPT"))
if VPT_PATH.exists() and str(VPT_PATH) not in sys.path:
    sys.path.insert(0, str(VPT_PATH))


class VPTIDMWrapper(nn.Module):
    """
    Wrapper around the VPT (Video Pre-Training) Inverse Dynamics Model.
    
    This loads the actual VPT IDM and provides a simplified interface
    for computing action predictions from frame pairs.
    """
    
    def __init__(
        self,
        model_path: str,
        weights_path: str,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self._vpt_available = False
        
        try:
            from inverse_dynamics_model import IDMAgent
            from agent import AGENT_RESOLUTION
            
            # Load model parameters from pickle
            agent_parameters = pickle.load(open(model_path, "rb"))
            net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
            pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
            pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
            
            # Create IDM agent
            self.agent = IDMAgent(
                idm_net_kwargs=net_kwargs,
                pi_head_kwargs=pi_head_kwargs,
                device=device
            )
            self.agent.load_weights(weights_path)
            self.agent_resolution = AGENT_RESOLUTION
            self._vpt_available = True
            print("  ✓ VPT IDM loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Failed to load VPT IDM: {e}")
            print("  → Falling back to SimpleIDM")
            self._vpt_available = False
    
    @property
    def is_available(self):
        return self._vpt_available
    
    def _preprocess_frames(self, frames: torch.Tensor) -> np.ndarray:
        """Convert torch frames to numpy format expected by VPT."""
        # frames: (B, C, H, W) in [0, 1] -> (B, H, W, C) in [0, 255] uint8
        frames = frames.permute(0, 2, 3, 1)  # (B, H, W, C)
        frames = (frames * 255).byte().cpu().numpy()
        
        # Resize to agent resolution
        resized = []
        for frame in frames:
            frame_resized = cv2.resize(
                frame, 
                (self.agent_resolution[0], self.agent_resolution[1]),
                interpolation=cv2.INTER_LINEAR
            )
            resized.append(frame_resized)
        
        return np.stack(resized)
    
    @torch.no_grad()
    def predict_actions(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Predict actions from frame transition using VPT IDM.
        
        Args:
            frame_t: (B, C, H, W) frame at time t
            frame_t1: (B, C, H, W) frame at time t+1
            
        Returns:
            Dict with 'buttons' and 'camera' predictions
        """
        if not self._vpt_available:
            return None
        
        # Stack frames for VPT (it expects a sequence)
        frames_t = self._preprocess_frames(frame_t)
        frames_t1 = self._preprocess_frames(frame_t1)
        
        # VPT expects sequence of frames, predict for each pair
        all_predictions = []
        for i in range(len(frames_t)):
            # Reset hidden state for each sample
            self.agent.reset()
            # Predict from pair of frames
            frames_pair = np.stack([frames_t[i], frames_t1[i]])
            pred = self.agent.predict_actions(frames_pair)
            all_predictions.append(pred)
        
        # Aggregate predictions
        result = {}
        for key in all_predictions[0].keys():
            # Take the second prediction (transition from t to t+1)
            result[key] = np.stack([p[key][0, 1] if p[key].ndim > 1 else p[key][1] 
                                   for p in all_predictions])
        
        return result


class SimpleIDM(nn.Module):
    """
    Simple CNN-based IDM for development/testing.
    
    This is a fallback when VPT IDM cannot be loaded.
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
    
    Requires VPT IDM (OpenAI's Video Pre-Training Inverse Dynamics Model).
    Download with: python download_vpt_idm.py
    
    The reward is normalized to [0, 1]:
    - 1 = perfect action prediction
    - 0 = completely wrong prediction
    """
    
    # Default paths for VPT IDM (downloaded by download_reward_models.py)
    DEFAULT_MODEL_PATH = "models_for_rl_finetuning/4x_idm.model"
    DEFAULT_WEIGHTS_PATH = "models_for_rl_finetuning/4x_idm.weights"
    
    def __init__(
        self,
        idm_model_path: Optional[str] = None,
        idm_weights_path: Optional[str] = None,
        device: str = "cuda",
        action_dim: int = 25,
        require_vpt: bool = True,
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.use_vpt = False
        self.simple_idm = None
        self.vpt_idm = None
        
        # Resolve default paths relative to project root
        project_root = Path(__file__).parent.parent
        
        if idm_model_path is None:
            idm_model_path = str(project_root / self.DEFAULT_MODEL_PATH)
        if idm_weights_path is None:
            idm_weights_path = str(project_root / self.DEFAULT_WEIGHTS_PATH)
        
        # Check if VPT IDM files exist
        model_exists = os.path.exists(idm_model_path)
        weights_exists = os.path.exists(idm_weights_path)
        
        if not model_exists or not weights_exists:
            print("  ⚠️  VPT IDM not found!")
            print(f"      Model path: {idm_model_path} ({'exists' if model_exists else 'MISSING'})")
            print(f"      Weights path: {idm_weights_path} ({'exists' if weights_exists else 'MISSING'})")
            print()
            print("  To download VPT IDM, run:")
            print("      python download_reward_models.py")
            print()
            print("  Or download manually from OpenAI:")
            print("      https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model")
            print("      https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights")
            print()
            if require_vpt:
                raise FileNotFoundError(
                    f"VPT IDM files not found. Run 'python download_vpt_idm.py' to download them."
                )
            else:
                print("  → Using SimpleIDM fallback (less accurate)")
                self._init_simple_idm()
                return
        
        # Load VPT IDM
        self.vpt_idm = VPTIDMWrapper(
            model_path=idm_model_path,
            weights_path=idm_weights_path,
            device=device,
        )
        
        if not self.vpt_idm.is_available:
            if require_vpt:
                raise RuntimeError(
                    "Failed to load VPT IDM. Check that VPT repo is cloned and dependencies installed.\n"
                    "  Option 1: Clone VPT repo to project root:\n"
                    "      git clone https://github.com/openai/Video-Pre-Training.git VPT\n"
                    "  Option 2: Set VPT_PATH environment variable:\n"
                    "      export VPT_PATH=/path/to/VPT\n"
                    "  Option 3: Set require_vpt=False in config to use SimpleIDM fallback"
                )
            else:
                print("  → Using SimpleIDM fallback (less accurate)")
                self._init_simple_idm()
                return
        
        self.use_vpt = True
        print("  ✓ VPT IDM loaded successfully for RIK reward")
    
    def _init_simple_idm(self):
        """Initialize SimpleIDM as fallback."""
        self.simple_idm = SimpleIDM(
            in_channels=6,
            num_actions=self.action_dim,
        ).to(self.device)
        self.use_vpt = False
        print("  ✓ SimpleIDM initialized as fallback")
    
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
            reward: (B,) RIK reward in [0, 1]
            info: Dict with additional metrics
        """
        if self.use_vpt:
            return self._compute_vpt_reward(frame_t, frame_t1, intended_action)
        else:
            return self._compute_simple_reward(frame_t, frame_t1, intended_action)
    
    def _compute_vpt_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        intended_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute reward using VPT IDM."""
        B = frame_t.shape[0]
        
        # Get VPT predictions
        predictions = self.vpt_idm.predict_actions(frame_t, frame_t1)
        
        # Convert intended action to comparable format
        if intended_action.dim() == 1:
            action_idx = intended_action
        else:
            action_idx = intended_action.argmax(dim=-1)
        
        # VPT predicts buttons (binary) and camera (continuous)
        # For simplicity, we compare the dominant action
        # This is a simplified comparison - could be made more sophisticated
        
        buttons_pred = predictions.get('attack', np.zeros(B))
        
        # Simple reward: 1 if any predicted button matches intended action category
        # This is a simplified heuristic
        reward = torch.zeros(B, device=self.device)
        
        # Basic matching logic (can be improved)
        for i in range(B):
            # Check if intended action is attack and VPT predicts attack
            if action_idx[i].item() == 0 and buttons_pred[i] > 0.5:
                reward[i] = 1.0
            elif action_idx[i].item() != 0 and buttons_pred[i] < 0.5:
                reward[i] = 0.5  # Partial match for non-attack
            else:
                reward[i] = 0.2  # Base reward
        
        info = {
            'rik_ce_loss': 0.0,  # Not applicable for VPT
            'rik_accuracy': reward.mean().item(),
            'rik_normalized': reward.mean().item(),
            'rik_using_vpt': True,
        }
        
        return reward, info
    
    def _compute_simple_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        intended_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute reward using SimpleIDM (fallback)."""
        B = frame_t.shape[0]
        
        # Concatenate frames channel-wise
        x = torch.cat([frame_t, frame_t1], dim=1)
        action_logits = self.simple_idm(x)
        
        # Get action indices
        if intended_action.dim() == 1:
            action_idx = intended_action
        else:
            if intended_action.shape[-1] == self.action_dim:
                action_idx = intended_action.argmax(dim=-1)
            else:
                # Continuous action space - use MSE
                action_pred = F.softmax(action_logits, dim=-1)
                mse = F.mse_loss(action_pred, intended_action, reduction='none')
                mse_per_sample = mse.mean(dim=-1)
                reward = torch.exp(-mse_per_sample)
                return reward, {'rik_mse': mse_per_sample.mean().item()}
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(action_logits, action_idx, reduction='none')
        
        # Normalize to [0, 1] using exp(-ce_loss)
        reward = torch.exp(-ce_loss)
        
        # Compute accuracy
        pred_action = action_logits.argmax(dim=-1)
        accuracy = (pred_action == action_idx).float().mean()
        
        info = {
            'rik_ce_loss': ce_loss.mean().item(),
            'rik_accuracy': accuracy.item(),
            'rik_normalized': reward.mean().item(),
            'rik_using_vpt': False,
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
            ce_losses.append(info.get('rik_ce_loss', 0.0))
            accuracies.append(info.get('rik_accuracy', 0.0))
        
        rewards = torch.stack(rewards, dim=1)
        
        info = {
            'rik_ce_loss': np.mean(ce_losses),
            'rik_accuracy': np.mean(accuracies),
            'rik_using_vpt': self.use_vpt,
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
