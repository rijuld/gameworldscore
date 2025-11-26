"""
Oasis Rollout Worker for long-horizon generation.

Provides rollout functionality for Oasis world model,
compatible with RLVR-World's rollout worker interface.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

# Add RLVR-World to path
RLVR_PATH = Path(__file__).parent.parent.parent / "RLVR-World" / "vid_wm" / "verl"
if str(RLVR_PATH) not in sys.path:
    sys.path.insert(0, str(RLVR_PATH))

try:
    from verl import DataProto
except ImportError:
    DataProto = None

from ..models.oasis_policy import OasisPolicy
from ..rewards.game_world_score import GameWorldScoreReward


@dataclass
class OasisRolloutConfig:
    """Configuration for Oasis rollout worker."""
    oasis_ckpt: str
    vae_ckpt: str
    dit_type: str = "DiT-S/2"
    device: str = "cuda"
    dtype: str = "float16"
    
    # Generation settings
    temperature: float = 1.0
    ddim_steps: int = 10
    max_frames: int = 32
    n_prompt_frames: int = 1
    
    # Rollout settings
    n_rollouts: int = 1  # Number of parallel rollouts per prompt
    response_length: int = 31  # Frames to generate


class OasisRolloutWorker(nn.Module):
    """
    Oasis Rollout Worker for generating frame sequences.
    
    Provides:
    - generate_sequences: Generate frame sequences from prompts
    - Long-horizon rollout with action conditioning
    """
    
    def __init__(self, config: OasisRolloutConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(config.dtype, torch.float16)
        
        # Load policy
        self.policy = OasisPolicy(
            oasis_ckpt=config.oasis_ckpt,
            vae_ckpt=config.vae_ckpt,
            dit_type=config.dit_type,
            device=config.device,
            dtype=self.dtype,
            ddim_steps=config.ddim_steps,
        )
        self.policy.eval_mode()
    
    def init_model(self):
        """Initialize model (called by RLVR-World worker setup)."""
        self.policy.eval_mode()
    
    @torch.no_grad()
    def generate_sequences(self, data: 'DataProto') -> 'DataProto':
        """
        Generate frame sequences from prompts.
        
        Args:
            data: DataProto containing:
                - prompts or initial_frames: (B, T, C, H, W) prompt frames
                - actions: (B, num_gen, action_dim) actions for generation
                
        Returns:
            DataProto with:
                - responses or generated_frames: (B, num_gen, C, H, W)
                - log_probs for each generated frame
        """
        self.policy.eval_mode()
        
        # Extract inputs
        if 'prompts' in data.batch:
            prompts = data.batch['prompts']
        elif 'initial_frames' in data.batch:
            prompts = data.batch['initial_frames']
        elif 'frames' in data.batch:
            # Use first frames as prompts
            prompts = data.batch['frames'][:, :self.config.n_prompt_frames]
        else:
            raise ValueError("No prompt frames found in data")
        
        if 'actions' in data.batch:
            actions = data.batch['actions']
        elif 'action_ids' in data.batch:
            actions = data.batch['action_ids']
        else:
            raise ValueError("No actions found in data")
        
        B = prompts.shape[0]
        num_gen = min(actions.shape[1], self.config.response_length)
        
        # Generate frames
        generated_frames = self.policy.generate_sequence(
            initial_frames=prompts,
            actions=actions[:, :num_gen],
            num_frames=num_gen,
        )
        
        # Compute log probs for generated frames
        all_frames = torch.cat([prompts, generated_frames], dim=1)
        latents = self.policy.encode_frames(all_frames)
        
        log_probs = []
        T_prompt = prompts.shape[1]
        
        for t in range(T_prompt, T_prompt + num_gen):
            context = latents[:, :t]
            target = latents[:, t:t+1]
            
            # Get action for this transition
            action_idx = t - T_prompt
            if action_idx < actions.shape[1]:
                action = actions[:, action_idx:action_idx+1]
            else:
                action = torch.zeros(B, 1, actions.shape[-1], device=self.device)
            
            log_prob = self.policy.compute_log_prob(context, action, target)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)  # (B, num_gen)
        
        # Prepare output
        result_dict = {
            'responses': generated_frames,
            'generated_frames': generated_frames,
            'old_log_probs': log_probs,
        }
        
        # Create attention mask (all ones for generated frames)
        total_length = T_prompt + num_gen
        attention_mask = torch.ones(B, total_length, device=self.device)
        result_dict['attention_mask'] = attention_mask
        
        # Response mask (ones for generated, zeros for prompt)
        response_mask = torch.zeros(B, total_length, device=self.device)
        response_mask[:, T_prompt:] = 1.0
        result_dict['response_mask'] = response_mask
        
        if DataProto is not None:
            result = DataProto.from_dict(result_dict)
            result.meta_info = {
                'num_generated_frames': num_gen,
                'prompt_length': T_prompt,
            }
        else:
            result = result_dict
        
        return result
    
    @torch.no_grad()
    def generate_interactive(
        self,
        initial_frames: torch.Tensor,
        action_generator: callable,
        num_frames: int = 32,
        reward_fn: Optional[GameWorldScoreReward] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate frames interactively with action feedback.
        
        This enables closed-loop generation where actions can depend
        on previously generated frames.
        
        Args:
            initial_frames: (B, T, C, H, W) initial context frames
            action_generator: Function that takes frames and returns actions
            num_frames: Number of frames to generate
            reward_fn: Optional reward function for computing rewards
            
        Returns:
            Dict with frames, actions, rewards, and log_probs
        """
        self.policy.eval_mode()
        
        B = initial_frames.shape[0]
        
        # Encode initial frames
        latents = self.policy.encode_frames(initial_frames)
        
        generated_frames = []
        actions_taken = []
        log_probs = []
        rewards = []
        
        # Keep track of all frames for action generation
        all_frames = [initial_frames]
        
        for i in tqdm(range(num_frames), desc="Generating frames"):
            # Get current frame sequence
            current_frames = torch.cat(all_frames, dim=1)
            
            # Generate action based on current frames
            action = action_generator(current_frames)  # (B, 1, action_dim)
            actions_taken.append(action)
            
            # Generate next frame
            new_latent, _ = self.policy.generate_next_frame(latents, action)
            
            # Decode to pixel space
            new_frame = self.policy.decode_latents(new_latent)
            generated_frames.append(new_frame)
            all_frames.append(new_frame)
            
            # Compute log prob
            log_prob = self.policy.compute_log_prob(latents, action, new_latent)
            log_probs.append(log_prob)
            
            # Update latents for next iteration
            latents = torch.cat([latents, new_latent], dim=1)
            if latents.shape[1] > self.policy.max_frames:
                latents = latents[:, -self.policy.max_frames:]
            
            # Compute reward if function provided
            if reward_fn is not None:
                prev_frame = all_frames[-2][:, -1] if len(all_frames) > 1 else initial_frames[:, -1]
                reward, _ = reward_fn.compute_frame_reward(
                    prev_frame,
                    new_frame.squeeze(1),
                    action.squeeze(1),
                )
                rewards.append(reward)
        
        # Stack outputs
        result = {
            'generated_frames': torch.cat(generated_frames, dim=1),
            'actions': torch.cat(actions_taken, dim=1),
            'log_probs': torch.stack(log_probs, dim=1),
        }
        
        if rewards:
            result['rewards'] = torch.stack(rewards, dim=1)
        
        return result
    
    def compute_log_prob(self, data: 'DataProto') -> 'DataProto':
        """
        Compute log probabilities for existing frame sequence.
        
        Compatible with RLVR-World's log_prob computation interface.
        """
        frames = data.batch['frames']
        actions = data.batch['actions']
        
        B, T = frames.shape[:2]
        
        latents = self.policy.encode_frames(frames)
        
        log_probs = []
        for t in range(1, T):
            context = latents[:, :t]
            target = latents[:, t:t+1]
            action = actions[:, t-1:t] if t-1 < actions.shape[1] else torch.zeros(
                B, 1, actions.shape[-1], device=self.device
            )
            
            log_prob = self.policy.compute_log_prob(context, action, target)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)
        
        if DataProto is not None:
            result = DataProto.from_dict({
                'old_log_probs': log_probs,
            })
        else:
            result = {'old_log_probs': log_probs}
        
        return result


def create_oasis_rollout(
    oasis_ckpt: str,
    vae_ckpt: str,
    device: str = "cuda",
    **kwargs,
) -> OasisRolloutWorker:
    """
    Create Oasis rollout worker.
    
    Args:
        oasis_ckpt: Path to Oasis DiT checkpoint
        vae_ckpt: Path to VAE checkpoint
        device: Device to load on
        **kwargs: Additional config options
        
    Returns:
        OasisRolloutWorker instance
    """
    config = OasisRolloutConfig(
        oasis_ckpt=oasis_ckpt,
        vae_ckpt=vae_ckpt,
        device=device,
        **kwargs,
    )
    return OasisRolloutWorker(config)

