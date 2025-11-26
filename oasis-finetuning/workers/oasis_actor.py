"""
Oasis Actor Worker for RL training.

Wraps the Oasis policy to provide RLVR-World compatible interface
for actor operations: log probability computation and policy updates.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add RLVR-World to path for protocol imports
RLVR_PATH = Path(__file__).parent.parent.parent / "RLVR-World" / "vid_wm" / "verl"
if str(RLVR_PATH) not in sys.path:
    sys.path.insert(0, str(RLVR_PATH))

try:
    from verl import DataProto
    from verl.workers.actor.base import BasePPOActor
except ImportError:
    # Fallback for standalone usage
    DataProto = None
    BasePPOActor = object

from ..models.oasis_policy import OasisPolicy


@dataclass
class OasisActorConfig:
    """Configuration for Oasis actor."""
    oasis_ckpt: str
    vae_ckpt: str
    dit_type: str = "DiT-S/2"
    device: str = "cuda"
    dtype: str = "float16"
    
    # Optimizer settings
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # PPO settings
    ppo_epochs: int = 1
    ppo_mini_batch_size: int = 4
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    
    # KL settings
    use_kl_loss: bool = False
    kl_coeff: float = 0.001


class OasisActorWorker(nn.Module):
    """
    Oasis Actor Worker compatible with RLVR-World training infrastructure.
    
    Provides:
    - compute_log_prob: Compute log probabilities for generated frames
    - update_actor: Update policy using PPO loss
    - generate_sequences: Generate frame sequences (delegated to rollout)
    """
    
    def __init__(self, config: OasisActorConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(config.dtype, torch.float16)
        
        # Load Oasis policy
        self.policy = OasisPolicy(
            oasis_ckpt=config.oasis_ckpt,
            vae_ckpt=config.vae_ckpt,
            dit_type=config.dit_type,
            device=config.device,
            dtype=self.dtype,
        )
        
        # Reference policy for KL (frozen copy)
        self.ref_policy: Optional[OasisPolicy] = None
        
        # Optimizer
        self.optimizer = AdamW(
            self.policy.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = None
        self.global_step = 0
    
    def init_model(self):
        """Initialize model (called by RLVR-World worker setup)."""
        self.policy.train_mode()
    
    def init_reference_policy(self):
        """Create frozen reference policy for KL computation."""
        if self.config.use_kl_loss and self.ref_policy is None:
            self.ref_policy = OasisPolicy(
                oasis_ckpt=self.config.oasis_ckpt,
                vae_ckpt=self.config.vae_ckpt,
                dit_type=self.config.dit_type,
                device=self.config.device,
                dtype=self.dtype,
            )
            self.ref_policy.eval_mode()
            for param in self.ref_policy.parameters():
                param.requires_grad = False
    
    def compute_log_prob(self, data: 'DataProto') -> 'DataProto':
        """
        Compute log probabilities for generated frames.
        
        Args:
            data: DataProto containing:
                - frames: (B, T, C, H, W) frame sequence
                - actions: (B, T-1, action_dim) actions
                
        Returns:
            DataProto with added 'old_log_probs' field
        """
        frames = data.batch['frames']
        actions = data.batch['actions']
        
        B, T = frames.shape[:2]
        
        # Encode all frames to latent space
        latents = self.policy.encode_frames(frames)
        
        # Compute log probs for each generated frame
        log_probs = []
        for t in range(1, T):
            context = latents[:, :t]
            target = latents[:, t:t+1]
            action = actions[:, t-1:t]
            
            log_prob = self.policy.compute_log_prob(
                context, action, target
            )
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)  # (B, T-1)
        
        # Create response with proper padding
        response_length = data.batch.get('response_length', T - 1)
        if isinstance(response_length, int):
            old_log_probs = torch.zeros(B, response_length, device=self.device)
            old_log_probs[:, :log_probs.shape[1]] = log_probs
        else:
            old_log_probs = log_probs
        
        if DataProto is not None:
            result = DataProto.from_dict({
                'old_log_probs': old_log_probs,
            })
        else:
            result = {'old_log_probs': old_log_probs}
        
        return result
    
    def compute_ref_log_prob(self, data: 'DataProto') -> 'DataProto':
        """
        Compute reference policy log probabilities for KL penalty.
        
        Args:
            data: DataProto with frames and actions
            
        Returns:
            DataProto with 'ref_log_prob' field
        """
        if self.ref_policy is None:
            self.init_reference_policy()
        
        frames = data.batch['frames']
        actions = data.batch['actions']
        
        B, T = frames.shape[:2]
        
        with torch.no_grad():
            latents = self.ref_policy.encode_frames(frames)
            
            log_probs = []
            for t in range(1, T):
                context = latents[:, :t]
                target = latents[:, t:t+1]
                action = actions[:, t-1:t]
                
                log_prob = self.ref_policy.compute_log_prob(
                    context, action, target
                )
                log_probs.append(log_prob)
            
            log_probs = torch.stack(log_probs, dim=1)
        
        response_length = data.batch.get('response_length', T - 1)
        if isinstance(response_length, int):
            ref_log_prob = torch.zeros(B, response_length, device=self.device)
            ref_log_prob[:, :log_probs.shape[1]] = log_probs
        else:
            ref_log_prob = log_probs
        
        if DataProto is not None:
            result = DataProto.from_dict({
                'ref_log_prob': ref_log_prob,
            })
        else:
            result = {'ref_log_prob': ref_log_prob}
        
        return result
    
    def update_actor(self, data: 'DataProto') -> 'DataProto':
        """
        Update actor policy using PPO loss.
        
        Args:
            data: DataProto containing:
                - frames, actions
                - old_log_probs: Log probs from rollout
                - advantages: Computed advantages
                - response_mask: Mask for valid responses
                
        Returns:
            DataProto with training metrics
        """
        self.policy.train_mode()
        
        frames = data.batch['frames']
        actions = data.batch['actions']
        old_log_probs = data.batch['old_log_probs']
        advantages = data.batch['advantages']
        response_mask = data.batch.get('response_mask', None)
        
        B, T = frames.shape[:2]
        metrics = {}
        
        # PPO update loop
        for epoch in range(self.config.ppo_epochs):
            # Compute current log probs
            latents = self.policy.encode_frames(frames)
            
            log_probs = []
            for t in range(1, T):
                context = latents[:, :t]
                target = latents[:, t:t+1]
                action = actions[:, t-1:t]
                
                log_prob = self.policy.compute_log_prob(
                    context, action, target
                )
                log_probs.append(log_prob)
            
            log_probs = torch.stack(log_probs, dim=1)  # (B, T-1)
            
            # Ensure shapes match
            if log_probs.shape[1] < old_log_probs.shape[1]:
                log_probs = F.pad(log_probs, (0, old_log_probs.shape[1] - log_probs.shape[1]))
            elif log_probs.shape[1] > old_log_probs.shape[1]:
                log_probs = log_probs[:, :old_log_probs.shape[1]]
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clip advantages to match log_probs shape
            if advantages.shape[1] != log_probs.shape[1]:
                if advantages.shape[1] > log_probs.shape[1]:
                    advantages = advantages[:, :log_probs.shape[1]]
                else:
                    advantages = F.pad(advantages, (0, log_probs.shape[1] - advantages.shape[1]))
            
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(
                ratio,
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio,
            )
            pg_loss = torch.max(pg_loss1, pg_loss2)
            
            # Apply mask if available
            if response_mask is not None:
                if response_mask.shape[1] != pg_loss.shape[1]:
                    if response_mask.shape[1] > pg_loss.shape[1]:
                        response_mask = response_mask[:, :pg_loss.shape[1]]
                    else:
                        response_mask = F.pad(response_mask, (0, pg_loss.shape[1] - response_mask.shape[1]))
                pg_loss = (pg_loss * response_mask).sum() / (response_mask.sum() + 1e-8)
            else:
                pg_loss = pg_loss.mean()
            
            # KL loss if enabled
            kl_loss = torch.tensor(0.0, device=self.device)
            if self.config.use_kl_loss and 'ref_log_prob' in data.batch:
                ref_log_prob = data.batch['ref_log_prob']
                if ref_log_prob.shape[1] != log_probs.shape[1]:
                    if ref_log_prob.shape[1] > log_probs.shape[1]:
                        ref_log_prob = ref_log_prob[:, :log_probs.shape[1]]
                    else:
                        ref_log_prob = F.pad(ref_log_prob, (0, log_probs.shape[1] - ref_log_prob.shape[1]))
                kl_loss = (log_probs - ref_log_prob).mean()
            
            # Total loss
            loss = pg_loss + self.config.kl_coeff * kl_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.get_trainable_parameters(),
                    self.config.grad_clip,
                )
            else:
                grad_norm = torch.tensor(0.0)
            
            self.optimizer.step()
            
            # Update scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.global_step += 1
        
        metrics = {
            'actor/pg_loss': pg_loss.item(),
            'actor/kl_loss': kl_loss.item(),
            'actor/total_loss': loss.item(),
            'actor/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'actor/learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        
        if DataProto is not None:
            result = DataProto.from_dict({})
            result.meta_info = {'metrics': metrics}
        else:
            result = {'metrics': metrics}
        
        return result
    
    def save_checkpoint(
        self,
        local_path: str,
        remote_path: Optional[str] = None,
        global_step: int = 0,
        **kwargs,
    ):
        """Save actor checkpoint."""
        import os
        os.makedirs(local_path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.policy.dit.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': global_step,
            'config': self.config.__dict__,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(local_path, 'actor.pt'))
    
    def load_checkpoint(self, local_path: str, **kwargs):
        """Load actor checkpoint."""
        import os
        checkpoint_path = os.path.join(local_path, 'actor.pt')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.policy.dit.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint.get('global_step', 0)
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def create_oasis_actor(
    oasis_ckpt: str,
    vae_ckpt: str,
    device: str = "cuda",
    **kwargs,
) -> OasisActorWorker:
    """
    Create Oasis actor worker.
    
    Args:
        oasis_ckpt: Path to Oasis DiT checkpoint
        vae_ckpt: Path to VAE checkpoint
        device: Device to load on
        **kwargs: Additional config options
        
    Returns:
        OasisActorWorker instance
    """
    config = OasisActorConfig(
        oasis_ckpt=oasis_ckpt,
        vae_ckpt=vae_ckpt,
        device=device,
        **kwargs,
    )
    return OasisActorWorker(config)

