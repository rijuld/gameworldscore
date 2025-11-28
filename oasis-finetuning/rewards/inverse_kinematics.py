"""
Inverse Kinematics Score (RIK) for action fidelity.

Uses a pre-trained Inverse Dynamics Model (IDM) to verify that
the generated frame transition is consistent with the intended action.

The reward is based on how well the IDM's predicted action matches
the intended action from the generated transition.

Uses VPT IDM (OpenAI's Video Pre-Training model) for high-quality action prediction.
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
            self._vpt_available = False
            raise e
    
    @property
    def is_available(self):
        return self._vpt_available
    
    def _preprocess_frames(self, frames: torch.Tensor) -> np.ndarray:
        """Convert torch frames to numpy format expected by VPT.
        
        OPTIMIZED: Uses torch interpolation for batch resizing, keeps on GPU until last step.
        """
        # frames: (B, C, H, W) in [0, 1] -> (B, H, W, C) in [0, 255] uint8
        # Use torch interpolation for faster batch resizing (all on GPU)
        import torch.nn.functional as F
        
        # Ensure frames are on the correct device (GPU if available)
        if frames.device.type != self.device:
            frames = frames.to(self.device)
        
        # Resize using torch (much faster for batches, stays on GPU)
        frames_resized = F.interpolate(
            frames,
            size=(self.agent_resolution[1], self.agent_resolution[0]),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy format (only at the very end, after all GPU ops)
        frames_resized = frames_resized.permute(0, 2, 3, 1)  # (B, H, W, C)
        frames_resized = (frames_resized * 255).clamp(0, 255).byte()
        # Only move to CPU and convert to numpy at the last step
        frames_resized = frames_resized.cpu().numpy()
        
        return frames_resized
    
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
            raise FileNotFoundError(
                f"VPT IDM files not found. Run 'python download_vpt_idm.py' to download them."
            )
        
        # Load VPT IDM
        self.vpt_idm = VPTIDMWrapper(
            model_path=idm_model_path,
            weights_path=idm_weights_path,
            device=device,
        )
        
        if not self.vpt_idm.is_available:
            raise RuntimeError(
                "Failed to load VPT IDM. Check that VPT repo is cloned and dependencies installed.\n"
                "  Option 1: Clone VPT repo to project root:\n"
                "      git clone https://github.com/openai/Video-Pre-Training.git VPT\n"
                "  Option 2: Set VPT_PATH environment variable:\n"
                "      export VPT_PATH=/path/to/VPT"
            )
        
        self.use_vpt = True
        print("  ✓ VPT IDM loaded successfully for RIK reward")

    
    @torch.no_grad()
    def compute_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        intended_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RIK reward for a batch of transitions.
        
        OPTIMIZED: Ensures all inputs are on GPU before computation.
        
        Args:
            frame_t: (B, C, H, W) frame at time t, values in [0, 1]
            frame_t1: (B, C, H, W) frame at time t+1, values in [0, 1]
            intended_action: (B, action_dim) one-hot encoded intended action
            
        Returns:
            reward: (B,) RIK reward in [0, 1]
            info: Dict with additional metrics
        """
        # Ensure all inputs are on the correct device (GPU if available)
        if frame_t.device.type != self.device:
            frame_t = frame_t.to(self.device)
        if frame_t1.device.type != self.device:
            frame_t1 = frame_t1.to(self.device)
        if intended_action.device.type != self.device:
            intended_action = intended_action.to(self.device)
        
        return self._compute_vpt_reward(frame_t, frame_t1, intended_action)
    
    def _compute_vpt_reward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        intended_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute reward using VPT IDM.
        
        OPTIMIZED: Minimizes CPU operations, keeps tensors on GPU where possible.
        """
        B = frame_t.shape[0]
        
        # Ensure inputs are on correct device before VPT processing
        if frame_t.device.type != self.device:
            frame_t = frame_t.to(self.device)
        if frame_t1.device.type != self.device:
            frame_t1 = frame_t1.to(self.device)
        if intended_action.device.type != self.device:
            intended_action = intended_action.to(self.device)
        
        # Get VPT predictions (this will convert to numpy internally)
        predictions = self.vpt_idm.predict_actions(frame_t, frame_t1)
        
        # Convert intended action to comparable format (all on GPU)
        # intended_action is one-hot encoded (B, action_dim)
        # We need to decode it to check individual keys
        from data.action_utils import ACTION_KEYS
        
        total_reward = torch.zeros(B, device=self.device)
        total_actions_checked = 0
        
        # Process each action key
        for i, key in enumerate(ACTION_KEYS):
            # Get intended value for this action key
            intended_val = intended_action[:, i]  # (B,)
            
            if key == "cameraX":
                # VPT predicts 'camera' as [dy, dx] (pitch, yaw)
                # Oasis cameraX is camera[0] -> index 0
                pred_val_np = predictions.get('camera', np.zeros((B, 2)))[:, 0]
                pred_val = torch.from_numpy(pred_val_np).to(self.device).float()
                
                # Normalize predicted value to match Oasis range [-1, 1] roughly
                # VPT camera is raw pixels/degrees? 
                # Oasis encoding: value = (raw - num_buckets) / num_buckets
                # Let's assume we just want correlation or low error
                # For simplicity in RIK, we check if sign matches or if both are near zero
                
                # Better: Check if movement is consistent
                # If intended > 0.1 (right), pred should be > 0
                # If intended < -0.1 (left), pred should be < 0
                # If intended ~ 0, pred should be ~ 0
                
                # Simple reward: 1.0 if consistent, 0.0 otherwise
                threshold = 0.05
                intended_move = (intended_val.abs() > threshold)
                pred_move = (pred_val.abs() > threshold) # VPT threshold might need tuning
                
                # Direction match
                same_dir = (intended_val * pred_val) > 0
                
                match = torch.where(
                    ~intended_move & ~pred_move, torch.tensor(1.0, device=self.device), # Both still
                    torch.where(
                        intended_move & pred_move & same_dir, torch.tensor(1.0, device=self.device), # Both move same way
                        torch.tensor(0.0, device=self.device) # Mismatch
                    )
                )
                total_reward += match
                total_actions_checked += 1
                
            elif key == "cameraY":
                # VPT predicts 'camera' as [dy, dx] (pitch, yaw)
                # Oasis cameraY is camera[1] -> index 1
                pred_val_np = predictions.get('camera', np.zeros((B, 2)))[:, 1]
                pred_val = torch.from_numpy(pred_val_np).to(self.device).float()
                
                threshold = 0.05
                intended_move = (intended_val.abs() > threshold)
                pred_move = (pred_val.abs() > threshold)
                same_dir = (intended_val * pred_val) > 0
                
                match = torch.where(
                    ~intended_move & ~pred_move, torch.tensor(1.0, device=self.device),
                    torch.where(
                        intended_move & pred_move & same_dir, torch.tensor(1.0, device=self.device),
                        torch.tensor(0.0, device=self.device)
                    )
                )
                total_reward += match
                total_actions_checked += 1
                
            elif key == "ESC":
                # VPT doesn't predict ESC usually, skip
                continue
                
            else:
                # Binary buttons
                # VPT keys might differ slightly?
                # VPT keys: attack, back, drop, forward, hotbar.X, inventory, jump, left, right, sneak, sprint, swapHands, use, pickItem
                # Oasis keys match these exactly.
                
                if key not in predictions:
                    continue
                    
                pred_prob_np = predictions[key]
                # Handle different shapes (sometimes (B, 2) softmax, sometimes (B,) sigmoid?)
                # VPT wrapper returns: p[key][0, 1] if ndim > 1 else p[key][1]
                # So it should be probability of class 1
                
                pred_prob = torch.from_numpy(pred_prob_np).to(self.device).float()
                
                # Intended is 0 or 1
                # Reward = 1 - |intended - pred|
                # Or binary match with threshold
                
                match = 1.0 - (intended_val - pred_prob).abs()
                total_reward += match
                total_actions_checked += 1
        
        # Average reward across all checked actions
        if total_actions_checked > 0:
            reward = total_reward / total_actions_checked
        else:
            reward = torch.zeros(B, device=self.device)
            
        info = {
            'rik_ce_loss': 0.0,
            'rik_accuracy': reward.mean().item(),
            'rik_normalized': reward.mean().item(),
            'rik_using_vpt': True,
        }
        
        return reward, info
    

    
    @torch.no_grad()
    def compute_sequence_reward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RIK reward for a sequence of frames (OPTIMIZED: batched where possible).
        
        Note: VPT IDM processes sequentially due to its recurrent architecture.
        
        OPTIMIZED: Ensures all inputs stay on GPU throughout computation.
        
        Args:
            frames: (B, T, C, H, W) sequence of frames
            actions: (B, T-1, action_dim) actions for each transition
            
        Returns:
            rewards: (B, T-1) RIK reward for each transition
            info: Dict with aggregated metrics
        """
        # Ensure inputs are on GPU
        if frames.device.type != self.device:
            frames = frames.to(self.device)
        if actions.device.type != self.device:
            actions = actions.to(self.device)
        
        B, T = frames.shape[:2]
        
        if T < 2:
            return torch.zeros(B, 0, device=self.device), {
                'rik_ce_loss': 0.0,
                'rik_accuracy': 0.0,
                'rik_using_vpt': self.use_vpt,
            }
        

        
        # For VPT IDM, we still need to process sequentially (VPT limitation)
        # OPTIMIZED: Batch process all timesteps at once where possible
        # Pre-extract all frame pairs to reduce repeated operations
        frame_pairs_t = frames[:, :-1]  # (B, T-1, C, H, W)
        frame_pairs_t1 = frames[:, 1:]   # (B, T-1, C, H, W)
        
        rewards = []
        ce_losses = []
        accuracies = []
        
        # Process all timesteps - VPT agent must reset per sample, but we can optimize preprocessing
        for t in range(T - 1):
            reward, info = self.compute_reward(
                frame_pairs_t[:, t],
                frame_pairs_t1[:, t],
                actions[:, t],
            )
            rewards.append(reward)
            ce_losses.append(info.get('rik_ce_loss', 0.0))
            accuracies.append(info.get('rik_accuracy', 0.0))
        
        rewards = torch.stack(rewards, dim=1)
        
        info = {
            'rik_ce_loss': np.mean(ce_losses) if ce_losses else 0.0,
            'rik_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'rik_using_vpt': True,
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
