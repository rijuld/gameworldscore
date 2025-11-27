"""
Oasis PPO Trainer for RL finetuning.

Integrates Oasis world model with RLVR-World's PPO training infrastructure
to enable ground-truth-free reinforcement learning using GameWorldScore reward.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pprint import pprint
import uuid

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add RLVR-World to path for PPO utilities
RLVR_PATH = Path(__file__).parent.parent.parent / "RLVR-World" / "vid_wm" / "verl"
if str(RLVR_PATH) not in sys.path:
    sys.path.insert(0, str(RLVR_PATH))

try:
    from verl import DataProto
    from verl.trainer.ppo import core_algos
    from verl.utils.torch_functional import masked_mean
    RLVR_AVAILABLE = True
except ImportError:
    RLVR_AVAILABLE = False
    DataProto = None

from models.oasis_policy import OasisPolicy
from rewards.game_world_score import GameWorldScoreReward, create_game_world_score_reward
from data.minecraft_dataset import (
    MinecraftDataset,
    MiDaSMinecraftDataset,
    create_minecraft_dataloader,
    create_midas_dataloaders,
)
from workers.oasis_actor import OasisActorWorker, OasisActorConfig
from workers.oasis_rollout import OasisRolloutWorker, OasisRolloutConfig


@dataclass
class OasisPPOConfig:
    """Configuration for Oasis PPO training."""
    # Model paths
    oasis_ckpt: str = "/home/rd3629/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/oasis500m.safetensors"
    vae_ckpt: str = "/home/rd3629/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/vit-l-20.safetensors"
    reward_models_dir: str = "models_for_rl_finetuning"
    
    # Data settings
    data_dir: str = "../Dataset/MiDaS-60_small"
    dataset_type: str = "auto"  # "auto", "video", "midas"
    frame_size: Tuple[int, int] = (360, 640)
    
    # Training settings
    total_epochs: int = 10
    total_training_steps: int = 10000
    train_batch_size: int = 4
    ppo_mini_batch_size: int = 4
    ppo_epochs: int = 1
    
    # Rollout settings
    n_prompt_frames: int = 1
    max_gen_frames: int = 31
    n_rollouts: int = 1
    
    # PPO hyperparameters
    learning_rate: float = 1e-5
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    grad_clip: float = 1.0
    
    # KL settings
    use_kl_in_reward: bool = True
    kl_coeff: float = 0.001
    kl_target: float = 0.1
    
    # Reward weights (GameWorldScore)
    rik_weight: float = 1.0
    rtc_weight: float = 1.0
    raq_weight: float = 1.0
    
    # Advantage estimation
    adv_estimator: str = "grpo"  # "gae", "grpo", "reinforce_plus_plus"
    
    # Checkpointing
    save_freq: int = 100
    test_freq: int = 50
    checkpoint_dir: str = "checkpoints/oasis_ppo"
    
    # Logging
    project_name: str = "oasis_rl_finetuning"
    experiment_name: str = "gameworldscore"
    use_wandb: bool = True
    
    # Device
    device: str = "cuda"


class OasisPPOTrainer:
    """
    PPO Trainer for Oasis world model.
    
    This trainer:
    1. Runs long-horizon rollouts using Oasis policy
    2. Computes ground-truth-free rewards using GameWorldScore
    3. Updates the policy using PPO with KL regularization
    
    The training loop follows RLVR-World's structure while using
    Oasis-specific components for world modeling.
    """
    
    def __init__(self, config: OasisPPOConfig):
        self.config = config
        self.device = config.device
        self.global_step = 0
        
        # Initialize components
        self._init_policy()
        self._init_reward()
        self._init_dataloader()
        self._init_optimizer()
        
        # KL controller
        if config.use_kl_in_reward:
            self._init_kl_controller()
        
        # Logging
        self.logger = None
        if config.use_wandb:
            self._init_wandb()
    
    def _init_policy(self):
        """Initialize Oasis policy and reference model."""
        print("Initializing Oasis policy...")
        
        self.policy = OasisPolicy(
            oasis_ckpt=self.config.oasis_ckpt,
            vae_ckpt=self.config.vae_ckpt,
            device=self.config.device,
        )
        
        # Reference policy for KL computation (frozen)
        if self.config.use_kl_in_reward:
            self.ref_policy = OasisPolicy(
                oasis_ckpt=self.config.oasis_ckpt,
                vae_ckpt=self.config.vae_ckpt,
                device=self.config.device,
            )
            self.ref_policy.eval_mode()
            for param in self.ref_policy.parameters():
                param.requires_grad = False
        else:
            self.ref_policy = None
    
    def _init_reward(self):
        """Initialize GameWorldScore reward function."""
        print("Initializing GameWorldScore reward...")
        
        self.reward_fn = create_game_world_score_reward(
            models_dir=self.config.reward_models_dir,
            device=self.config.device,
            rik_weight=self.config.rik_weight,
            rtc_weight=self.config.rtc_weight,
            raq_weight=self.config.raq_weight,
        )
    
    def _init_dataloader(self):
        """Initialize data loader."""
        # Support both absolute and relative paths
        data_dir = Path(self.config.data_dir)
        if not data_dir.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / self.config.data_dir
        
        if data_dir.exists():
            print(f"Loading dataset from: {data_dir}")
            
            # Create dataloader based on dataset type
            clip_length = self.config.n_prompt_frames + self.config.max_gen_frames
            
            self.train_dataloader = create_minecraft_dataloader(
                data_dir=str(data_dir),
                batch_size=self.config.train_batch_size,
                clip_length=clip_length,
                dataset_type=self.config.dataset_type,
                frame_size=self.config.frame_size,
                split="train",
            )
            
            # Try to create test dataloader
            try:
                self.test_dataloader = create_minecraft_dataloader(
                    data_dir=str(data_dir),
                    batch_size=self.config.train_batch_size,
                    clip_length=clip_length,
                    dataset_type=self.config.dataset_type,
                    frame_size=self.config.frame_size,
                    split="test",
                    shuffle=False,
                )
            except Exception:
                self.test_dataloader = None
        else:
            print(f"Warning: Data directory {data_dir} not found.")
            print("Please set --data-dir to your dataset path.")
            self.train_dataloader = None
            self.test_dataloader = None
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.policy.get_trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        if self.config.total_training_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_training_steps,
                eta_min=self.config.learning_rate * 0.1,
            )
        else:
            self.scheduler = None
    
    def _init_kl_controller(self):
        """Initialize adaptive KL controller."""
        if RLVR_AVAILABLE:
            self.kl_controller = core_algos.AdaptiveKLController(
                init_kl_coef=self.config.kl_coeff,
                target_kl=self.config.kl_target,
                horizon=1000,
            )
        else:
            # Simple fixed KL coefficient
            self.kl_controller = type('KLCtrl', (), {'value': self.config.kl_coeff})()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.__dict__,
            )
            self.logger = wandb
        except ImportError:
            print("wandb not available, using console logging only")
            self.logger = None
    
    def _generate_rollouts(
        self,
        initial_frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate video rollouts using Oasis world model.
        
        This is the core of ground-truth-free RL training:
        - Input: ONLY the first frame(s) + action sequence
        - Output: Oasis generates the ENTIRE subsequent video
        - No ground-truth future frames are used
        
        Args:
            initial_frames: (B, T_prompt, C, H, W) initial frame(s) - typically just 1 frame
            actions: (B, num_gen, action_dim) action sequence for generation
            
        Returns:
            Dict containing:
                - generated_frames: (B, num_gen, C, H, W) frames generated by Oasis
                - all_frames: (B, T_prompt + num_gen, C, H, W) initial + generated
                - log_probs: (B, num_gen) log probabilities for PPO
                - ref_log_probs: (B, num_gen) reference log probs for KL (optional)
                - actions: (B, num_gen, action_dim) the action sequence used
        """
        self.policy.eval_mode()
        
        B = initial_frames.shape[0]
        num_gen = min(actions.shape[1], self.config.max_gen_frames)
        
        with torch.no_grad():
            # Generate frames
            generated_frames = self.policy.generate_sequence(
                initial_frames=initial_frames,
                actions=actions[:, :num_gen],
                num_frames=num_gen,
            )
            
            # Compute log probabilities
            all_frames = torch.cat([initial_frames, generated_frames], dim=1)
            latents = self.policy.encode_frames(all_frames)
            
            log_probs = []
            T_prompt = initial_frames.shape[1]
            
            for t in range(T_prompt, T_prompt + num_gen):
                context = latents[:, :t]
                target = latents[:, t:t+1]
                action_idx = t - T_prompt
                action = actions[:, action_idx:action_idx+1] if action_idx < actions.shape[1] else torch.zeros(
                    B, 1, actions.shape[-1], device=self.device
                )
                
                log_prob = self.policy.compute_log_prob(context, action, target)
                log_probs.append(log_prob)
            
            log_probs = torch.stack(log_probs, dim=1)
            
            # Compute reference log probs for KL if needed
            if self.ref_policy is not None:
                ref_latents = self.ref_policy.encode_frames(all_frames)
                ref_log_probs = []
                
                for t in range(T_prompt, T_prompt + num_gen):
                    context = ref_latents[:, :t]
                    target = ref_latents[:, t:t+1]
                    action_idx = t - T_prompt
                    action = actions[:, action_idx:action_idx+1] if action_idx < actions.shape[1] else torch.zeros(
                        B, 1, actions.shape[-1], device=self.device
                    )
                    
                    ref_log_prob = self.ref_policy.compute_log_prob(context, action, target)
                    ref_log_probs.append(ref_log_prob)
                
                ref_log_probs = torch.stack(ref_log_probs, dim=1)
            else:
                ref_log_probs = None
        
        return {
            'generated_frames': generated_frames,
            'all_frames': all_frames,
            'log_probs': log_probs,
            'ref_log_probs': ref_log_probs,
            'actions': actions[:, :num_gen],
        }
    
    def _compute_rewards(
        self,
        all_frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GameWorldScore rewards for generated frames.
        
        Args:
            all_frames: (B, T, C, H, W) all frames (prompt + generated)
            actions: (B, num_gen, action_dim) actions
            
        Returns:
            rewards: (B, num_gen) rewards for each generated frame
        """
        # Use GameWorldScore reward
        rewards, info = self.reward_fn.compute_sequence_reward(
            all_frames,
            actions,
            return_per_frame=True,
        )
        
        return rewards, info
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor],
        response_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute advantages using the configured estimator.
        
        Args:
            rewards: (B, T) rewards
            log_probs: (B, T) log probabilities
            ref_log_probs: (B, T) reference log probs for KL
            response_mask: (B, T) mask for valid positions
            
        Returns:
            advantages: (B, T) computed advantages
        """
        B, T = rewards.shape
        
        # Apply KL penalty if using KL in reward
        if self.config.use_kl_in_reward and ref_log_probs is not None:
            kl = log_probs - ref_log_probs
            rewards = rewards - self.kl_controller.value * kl
        
        # Compute advantages based on estimator type
        if self.config.adv_estimator == "grpo":
            # GRPO: Normalize rewards within batch
            scores = rewards.sum(dim=-1)  # (B,)
            mean_score = scores.mean()
            std_score = scores.std() + 1e-8
            normalized_scores = (scores - mean_score) / std_score
            advantages = normalized_scores.unsqueeze(-1).expand(-1, T)
            if response_mask is not None:
                advantages = advantages * response_mask
        
        elif self.config.adv_estimator == "reinforce_plus_plus":
            # REINFORCE++: Discounted returns with whitening
            returns = torch.zeros_like(rewards)
            running_return = 0
            
            for t in reversed(range(T)):
                running_return = rewards[:, t] + self.config.gamma * running_return
                returns[:, t] = running_return
            
            # Whiten returns
            mean_return = returns.mean()
            std_return = returns.std() + 1e-8
            advantages = (returns - mean_return) / std_return
        
        else:  # GAE
            # Simple version without value function
            # In full implementation, would use critic
            returns = torch.zeros_like(rewards)
            running_return = 0
            
            for t in reversed(range(T)):
                running_return = rewards[:, t] + self.config.gamma * running_return
                returns[:, t] = running_return
            
            advantages = returns
        
        return advantages
    
    def _ppo_update(
        self,
        all_frames: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform PPO policy update.
        
        Args:
            all_frames: (B, T, C, H, W) all frames
            actions: (B, T-1, action_dim) actions
            old_log_probs: (B, T-1) log probs from rollout
            advantages: (B, T-1) computed advantages
            response_mask: (B, T-1) mask for valid positions
            
        Returns:
            Dict with training metrics
        """
        self.policy.train_mode()
        
        B = all_frames.shape[0]
        T_prompt = self.config.n_prompt_frames
        T_gen = all_frames.shape[1] - T_prompt
        
        metrics = defaultdict(list)
        
        for epoch in range(self.config.ppo_epochs):
            # Compute current log probs
            latents = self.policy.encode_frames(all_frames)
            
            log_probs = []
            for t in range(T_prompt, T_prompt + T_gen):
                context = latents[:, :t]
                target = latents[:, t:t+1]
                action_idx = t - T_prompt
                
                if action_idx < actions.shape[1]:
                    action = actions[:, action_idx:action_idx+1]
                else:
                    action = torch.zeros(B, 1, actions.shape[-1], device=self.device)
                
                log_prob = self.policy.compute_log_prob(context, action, target)
                log_probs.append(log_prob)
            
            log_probs = torch.stack(log_probs, dim=1)
            
            # Ensure shape compatibility
            min_len = min(log_probs.shape[1], old_log_probs.shape[1], advantages.shape[1])
            log_probs = log_probs[:, :min_len]
            old_log_probs_batch = old_log_probs[:, :min_len]
            advantages_batch = advantages[:, :min_len]
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs_batch)
            
            pg_loss1 = -advantages_batch * ratio
            pg_loss2 = -advantages_batch * torch.clamp(
                ratio,
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio,
            )
            pg_loss = torch.max(pg_loss1, pg_loss2)
            
            # Apply mask if available
            if response_mask is not None:
                mask = response_mask[:, :min_len]
                pg_loss = (pg_loss * mask).sum() / (mask.sum() + 1e-8)
            else:
                pg_loss = pg_loss.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            pg_loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.get_trainable_parameters(),
                    self.config.grad_clip,
                )
            else:
                grad_norm = 0.0
            
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record metrics
            metrics['pg_loss'].append(pg_loss.item())
            metrics['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            
            # Compute clip fraction
            clip_frac = ((ratio - 1).abs() > self.config.clip_ratio).float().mean().item()
            metrics['clip_fraction'].append(clip_frac)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.
        
        RL Training Flow (Ground-Truth Free):
        1. INPUT: First frame + action sequence (from dataset)
        2. GENERATE: Oasis generates entire video from first frame + actions
        3. REWARD: GameWorldScore computed on GENERATED frames only
        4. UPDATE: PPO update based on rewards
        
        No ground-truth future frames are used!
        
        Args:
            batch: Dict with 'initial_frame' (or 'frames') and 'actions'
            
        Returns:
            Dict with training metrics
        """
        # Handle both new format (initial_frame) and old format (frames)
        if 'initial_frame' in batch:
            # New format: only first frame provided
            initial_frames = batch['initial_frame'].to(self.device)  # (B, 1, C, H, W)
        else:
            # Old format: extract first frame from sequence
            frames = batch['frames'].to(self.device)
            initial_frames = frames[:, :self.config.n_prompt_frames]  # (B, 1, C, H, W)
        
        actions = batch['actions'].to(self.device)
        
        B = initial_frames.shape[0]
        
        # Action sequence for generation
        target_actions = actions[:, :self.config.max_gen_frames]
        
        # ============================================================
        # STEP 1: Generate entire video from first frame + actions
        # ============================================================
        rollout_data = self._generate_rollouts(initial_frames, target_actions)
        # rollout_data contains:
        #   - generated_frames: (B, T, C, H, W) - ALL generated by Oasis
        #   - all_frames: (B, T+1, C, H, W) - initial + generated
        #   - log_probs: (B, T) - for PPO update
        
        # ============================================================
        # STEP 2: Compute rewards on GENERATED frames (ground-truth free)
        # ============================================================
        rewards, reward_info = self._compute_rewards(
            rollout_data['all_frames'],  # includes initial + generated
            rollout_data['actions'],
        )
        # GameWorldScore evaluates:
        #   - RIK: Does the transition match the action? (IDM)
        #   - RTC: Are frames temporally consistent? (CLIP)
        #   - RAQ: Do frames look good? (Aesthetic)
        
        # ============================================================
        # STEP 3: Compute advantages for PPO
        # ============================================================
        advantages = self._compute_advantages(
            rewards,
            rollout_data['log_probs'],
            rollout_data['ref_log_probs'],
        )
        
        # ============================================================
        # STEP 4: PPO policy update
        # ============================================================
        update_metrics = self._ppo_update(
            rollout_data['all_frames'],
            rollout_data['actions'],
            rollout_data['log_probs'],
            advantages,
        )
        
        # Aggregate metrics
        metrics = {
            'reward/mean': rewards.mean().item(),
            'reward/std': rewards.std().item(),
            'reward/total': rewards.sum(dim=-1).mean().item(),
            **{f'reward/{k}': v for k, v in reward_info.items()},
            **{f'train/{k}': v for k, v in update_metrics.items()},
            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
            'train/num_generated_frames': rollout_data['generated_frames'].shape[1],
        }
        
        # Update KL controller
        if self.config.use_kl_in_reward and rollout_data['ref_log_probs'] is not None:
            kl = (rollout_data['log_probs'] - rollout_data['ref_log_probs']).mean().item()
            if hasattr(self.kl_controller, 'update'):
                self.kl_controller.update(kl, B)
            metrics['train/kl'] = kl
            metrics['train/kl_coeff'] = self.kl_controller.value
        
        return metrics
    
    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting Oasis PPO training for {self.config.total_training_steps} steps...")
        
        if self.train_dataloader is None:
            print("No training data available. Exiting.")
            return
        
        progress_bar = tqdm(total=self.config.total_training_steps, desc="Training")
        
        self.global_step = 0
        
        for epoch in range(self.config.total_epochs):
            for batch in self.train_dataloader:
                if self.global_step >= self.config.total_training_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                
                # Logging
                if self.logger is not None:
                    self.logger.log(metrics, step=self.global_step)
                
                # Console logging (every 10 steps)
                if self.global_step % 10 == 0:
                    print(f"Step {self.global_step}: reward={metrics.get('reward/mean', 0):.4f}, "
                          f"pg_loss={metrics.get('train/pg_loss', 0):.4f}")
                
                # Checkpointing
                if self.config.save_freq > 0 and self.global_step % self.config.save_freq == 0:
                    self.save_checkpoint()
                
                progress_bar.update(1)
                self.global_step += 1
            
            if self.global_step >= self.config.total_training_steps:
                break
        
        # Final checkpoint
        self.save_checkpoint()
        progress_bar.close()
        
        print("Training complete!")
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        if path is None:
            path = os.path.join(self.config.checkpoint_dir, f"step_{self.global_step}")
        
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.policy.dit.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(path, 'checkpoint.pt'))
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint_file = os.path.join(path, 'checkpoint.pt')
        
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, weights_only=False)
            self.policy.dit.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint.get('global_step', 0)
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"Loaded checkpoint from {path} at step {self.global_step}")
        else:
            print(f"No checkpoint found at {path}")


def create_oasis_ppo_trainer(
    oasis_ckpt: str,
    vae_ckpt: str,
    reward_models_dir: str = "models_for_rl_finetuning",
    **kwargs,
) -> OasisPPOTrainer:
    """
    Create Oasis PPO trainer.
    
    Args:
        oasis_ckpt: Path to Oasis DiT checkpoint
        vae_ckpt: Path to VAE checkpoint
        reward_models_dir: Directory with reward model checkpoints
        **kwargs: Additional config options
        
    Returns:
        OasisPPOTrainer instance
    """
    config = OasisPPOConfig(
        oasis_ckpt=oasis_ckpt,
        vae_ckpt=vae_ckpt,
        reward_models_dir=reward_models_dir,
        **kwargs,
    )
    return OasisPPOTrainer(config)

