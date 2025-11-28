"""
Oasis GRPO Trainer for RL finetuning.

Integrates Oasis world model with RLVR-World's GRPO training infrastructure
to enable ground-truth-free reinforcement learning using GameWorldScore reward.

GRPO (Group Relative Policy Optimization) is more stable than vanilla PPO
for diffusion model finetuning.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any, Callable, Tuple, List
from dataclasses import dataclass, field
from collections import defaultdict
from pprint import pprint
import uuid

# Configure PyTorch memory allocator to reduce fragmentation
# Updated to use PYTORCH_ALLOC_CONF instead of deprecated PYTORCH_CUDA_ALLOC_CONF
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add RLVR-World to path for GRPO utilities
# Robust path resolution:
# 1. Check RLVR_WORLD_PATH environment variable
# 2. Check relative path from project root
# 3. Fail with helpful error message

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

# 1. Environment variable
env_path = os.environ.get("RLVR_WORLD_PATH")
if env_path:
    RLVR_PATH = Path(env_path)
else:
    # 2. Relative path - point to verl repository directory
    RLVR_PATH = project_root / "RLVR-World" / "vid_wm" / "verl"

if str(RLVR_PATH) not in sys.path:
    sys.path.insert(0, str(RLVR_PATH))

try:
    if not RLVR_PATH.exists():
        raise ImportError(f"Path {RLVR_PATH} does not exist")
    
    # Check if the verl Python package exists inside the verl repository
    if not (RLVR_PATH / "verl" / "__init__.py").exists():
        raise ImportError(f"Python package verl/__init__.py not found in {RLVR_PATH}")

    from verl import DataProto
    from verl.trainer.ppo import core_algos
    from verl.utils.torch_functional import masked_mean
    RLVR_AVAILABLE = True
except ImportError as e:
    print(f"\n{'!'*80}")
    print(f"WARNING: RLVR-World integration failed.")
    print(f"Path checked: {RLVR_PATH}")
    print(f"Error: {e}")
    print(f"{'-'*80}")
    print("Troubleshooting:")
    print("1. If using git submodules, run: git submodule update --init --recursive")
    print("2. If RLVR-World is in a different location, set RLVR_WORLD_PATH env var")
    print(f"{'!'*80}\n")
    
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
class OasisGRPOConfig:
    """Configuration for Oasis GRPO training."""
    # Model paths
    oasis_ckpt: str = "/home/rd3629/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/oasis500m.safetensors"
    vae_ckpt: str = "/home/rd3629/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/vit-l-20.safetensors"
    reward_models_dir: str = "models_for_rl_finetuning"
    
    # Data settings
    data_dir: str = "Dataset/MiDaS-60_small"
    dataset_type: str = "auto"  # "auto", "video", "midas"
    frame_size: Tuple[int, int] = (360, 640)
    
    # Training settings
    total_epochs: int = 10
    total_training_steps: int = 10000
    train_batch_size: int = 1  # Number of unique prompts per step
    group_size: int = 2  # Number of rollouts per prompt - optimized for memory-speed tradeoff
    grpo_epochs: int = 2  # Number of update epochs per step - optimized for speed
    
    # Rollout settings
    n_prompt_frames: int = 1
    max_gen_frames: int = 2  # Reduced from 4 to 2 for memory optimization
    
    # GRPO hyperparameters (following RLVR-World)
    learning_rate: float = 5e-7
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    log_ratio_clip: float = 2.0  # Added back for compatibility
    entropy_coeff: float = 0.001
    grad_clip: float = 1.0
    reward_scale: float = 1.0
    
    # Memory optimization settings
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing to save memory
    use_mixed_precision: bool = True  # Enable mixed precision training (FP16)
    offload_reward_to_cpu: bool = True  # Offload reward models to CPU to save GPU memory
    offload_ref_policy_to_cpu: bool = True  # Offload reference policy to CPU (only load to GPU when needed)
    cache_encoded_frames: bool = False  # DISABLED: Caching prevents gradient flow (causes clip=0)
    kl_compute_freq: int = 5  # Compute KL divergence every N steps (1 = every step)
    
    # KL settings
    use_kl_in_reward: bool = True  # Enable KL divergence penalty (ref policy on CPU to save memory)
    kl_coeff: float = 0.01
    kl_target: float = 0.1
    
    # Reward weights (GameWorldScore)
    rik_weight: float = 1.0
    rtc_weight: float = 1.0
    raq_weight: float = 1.0
    require_vpt: bool = True
    
    # Advantage estimation
    adv_estimator: str = "grpo"  # "grpo" (recommended)
    
    # Checkpointing
    save_freq: int = 100
    test_freq: int = 50
    checkpoint_dir: str = "checkpoints/oasis_grpo"
    
    # Video saving
    video_save_freq: int = 50
    video_save_dir: str = "samples"
    
    # Logging
    project_name: str = "oasis_rl_finetuning"
    experiment_name: str = "grpo_gameworldscore"
    use_wandb: bool = True
    
    # Device
    device: str = "cuda"


class OasisGRPOTrainer:
    """
    GRPO Trainer for Oasis world model using RLVR-World's implementation.
    """
    
    def __init__(self, config: OasisGRPOConfig):
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
            
        if not RLVR_AVAILABLE:
            raise RuntimeError("RLVR-World libraries not found. Please install RLVR-World or check paths.")
    
    def _init_policy(self):
        """Initialize Oasis policy and reference model."""
        print("Initializing Oasis policy...")
        
        self.policy = OasisPolicy(
            oasis_ckpt=self.config.oasis_ckpt,
            vae_ckpt=self.config.vae_ckpt,
            device=self.config.device,
        )
        
        # Clear cache after loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable gradient checkpointing if requested
        if self.config.use_gradient_checkpointing:
            if hasattr(self.policy.dit, 'enable_gradient_checkpointing'):
                self.policy.dit.enable_gradient_checkpointing()
                print("  Gradient checkpointing enabled")
            elif hasattr(self.policy.dit, 'gradient_checkpointing_enable'):
                self.policy.dit.gradient_checkpointing_enable()
                print("  Gradient checkpointing enabled")
        
        # Reference policy for KL computation (frozen)
        if self.config.use_kl_in_reward:
            if self.config.offload_ref_policy_to_cpu:
                print("  Loading reference policy on CPU to save GPU memory")
                ref_device = "cpu"
            else:
                print("  WARNING: Loading reference policy on GPU will use significant memory")
                ref_device = self.config.device
            
            self.ref_policy = OasisPolicy(
                oasis_ckpt=self.config.oasis_ckpt,
                vae_ckpt=self.config.vae_ckpt,
                device=ref_device,
            )
            self.ref_policy.eval_mode()
            for param in self.ref_policy.parameters():
                param.requires_grad = False
            
            self.ref_policy_device = ref_device
            print(f"  Reference policy on: {ref_device}")
        else:
            self.ref_policy = None
            self.ref_policy_device = None
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory after policy init: {allocated:.2f} GB")
    
    def _init_reward(self):
        """Initialize GameWorldScore reward function."""
        print("Initializing GameWorldScore reward...")
        
        # Determine device for reward models
        reward_device = "cpu" if self.config.offload_reward_to_cpu else self.config.device
        if self.config.offload_reward_to_cpu:
            print("  Offloading reward models to CPU to save GPU memory")
        
        self.reward_fn = create_game_world_score_reward(
            models_dir=self.config.reward_models_dir,
            device=reward_device,
            rik_weight=self.config.rik_weight,
            rtc_weight=self.config.rtc_weight,
            raq_weight=self.config.raq_weight,
            require_vpt=self.config.require_vpt,
        )
        
        # Clear cache after loading reward models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory after reward init: {allocated:.2f} GB")
    
    def _init_dataloader(self):
        """Initialize data loader."""
        data_dir = Path(self.config.data_dir)
        if not data_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / self.config.data_dir
        
        if data_dir.exists():
            print(f"Loading dataset from: {data_dir}")
            
            clip_length = self.config.n_prompt_frames + self.config.max_gen_frames
            
            self.train_dataloader = create_minecraft_dataloader(
                data_dir=str(data_dir),
                batch_size=self.config.train_batch_size,
                clip_length=clip_length,
                dataset_type=self.config.dataset_type,
                frame_size=self.config.frame_size,
                split="train",
            )
        else:
            print(f"Warning: Data directory {data_dir} not found.")
            self.train_dataloader = None
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.policy.get_trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        # Initialize gradient scaler for mixed precision training
        if self.device == "cuda" and self.config.use_mixed_precision:
            self.grad_scaler = torch.cuda.amp.GradScaler()
            print("  Mixed precision training enabled (FP16)")
        else:
            self.grad_scaler = None
        
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
        self.kl_controller = core_algos.AdaptiveKLController(
            init_kl_coef=self.config.kl_coeff,
            target_kl=self.config.kl_target,
            horizon=1000,
        )
    
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
        Repeats inputs by group_size to form GRPO groups.
        Memory-optimized version: processes in chunks and cleans up aggressively.
        """
        self.policy.eval_mode()
        
        B = initial_frames.shape[0]
        G = self.config.group_size
        
        # Repeat inputs for group generation
        # (B, ...) -> (B*G, ...)
        initial_frames_repeated = initial_frames.repeat_interleave(G, dim=0)
        actions_repeated = actions.repeat_interleave(G, dim=0)
        
        num_gen = min(actions.shape[1], self.config.max_gen_frames)
        
        with torch.no_grad():
            # Generate frames
            generated_frames = self.policy.generate_sequence(
                initial_frames=initial_frames_repeated,
                actions=actions_repeated[:, :num_gen],
                num_frames=num_gen,
            )
            
            # Compute log probabilities
            all_frames = torch.cat([initial_frames_repeated, generated_frames], dim=1)
            
            # Delete intermediate tensors
            del generated_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            latents = self.policy.encode_frames(all_frames)
            
            # Delete initial_frames_repeated to free memory
            del initial_frames_repeated
            
            log_probs = []
            T_prompt = initial_frames.shape[1]
            
            for t in range(T_prompt, T_prompt + num_gen):
                context = latents[:, :t]
                target = latents[:, t:t+1]
                action_idx = t - T_prompt
                action = actions_repeated[:, action_idx:action_idx+1]
                
                log_prob = self.policy.compute_log_prob(context, action, target)
                log_probs.append(log_prob.detach())  # Detach to save memory
                
                # Clean up context and target
                del context, target, action
            
            log_probs = torch.stack(log_probs, dim=1)
            
            # Compute reference log probs for KL if needed
            ref_log_probs = None
            if self.ref_policy is not None:
                # Only compute ref log probs if KL is needed this step
                # Check if we should compute KL this step
                compute_kl_this_step = (self.global_step % self.config.kl_compute_freq == 0)
                
                if compute_kl_this_step:
                    # Delete latents before computing ref_latents
                    del latents
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # If ref policy is on CPU, move frames to CPU for inference
                    if self.ref_policy_device == "cpu":
                        all_frames_for_ref = all_frames.cpu()
                    else:
                        all_frames_for_ref = all_frames
                    
                    ref_latents = self.ref_policy.encode_frames(all_frames_for_ref)
                    ref_log_probs_list = []
                    
                    for t in range(T_prompt, T_prompt + num_gen):
                        context = ref_latents[:, :t]
                        target = ref_latents[:, t:t+1]
                        action_idx = t - T_prompt
                        
                        # Actions need to be on same device as ref policy
                        if self.ref_policy_device == "cpu":
                            action = actions_repeated[:, action_idx:action_idx+1].cpu()
                        else:
                            action = actions_repeated[:, action_idx:action_idx+1]
                        
                        ref_log_prob = self.ref_policy.compute_log_prob(context, action, target)
                        ref_log_probs_list.append(ref_log_prob.detach())
                        
                        del context, target, action
                    
                    ref_log_probs = torch.stack(ref_log_probs_list, dim=1)
                    
                    # Move ref_log_probs back to GPU if needed
                    if self.ref_policy_device == "cpu":
                        ref_log_probs = ref_log_probs.to(self.device)
                        del all_frames_for_ref
                    
                    del ref_latents, ref_log_probs_list
                else:
                    # Skip KL computation this step
                    del latents
                    ref_log_probs = None
            else:
                del latents
        
        # Create group indices
        # [0, 0, 0, 0, 1, 1, 1, 1, ...]
        indices = torch.arange(B, device=self.device).repeat_interleave(G)
        
        return {
            'generated_frames': None,  # Don't keep generated_frames in memory
            'all_frames': all_frames,
            'log_probs': log_probs,
            'ref_log_probs': ref_log_probs,
            'actions': actions_repeated[:, :num_gen],
            'indices': indices,
        }
    
    def _compute_rewards(
        self,
        all_frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GameWorldScore rewards."""
        with torch.no_grad():
            # Move frames to reward device if needed (CPU offloading)
            if self.config.offload_reward_to_cpu:
                all_frames_cpu = all_frames.cpu()
                actions_cpu = actions.cpu()
                rewards, info = self.reward_fn.compute_sequence_reward(
                    all_frames_cpu,
                    actions_cpu,
                    return_per_frame=True,
                )
                # Move rewards back to GPU
                rewards = rewards.to(self.device)
                del all_frames_cpu, actions_cpu
            else:
                rewards, info = self.reward_fn.compute_sequence_reward(
                    all_frames,
                    actions,
                    return_per_frame=True,
                )
            # Detach rewards to prevent gradient tracking
            rewards = rewards.detach()
        return rewards, info
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        indices: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute advantages using verl's implementation.
        """
        # Scale rewards
        scaled_rewards = rewards * self.config.reward_scale
        
        # Create response mask if not provided (all ones)
        if response_mask is None:
            response_mask = torch.ones_like(rewards)
            
        # Use verl's GRPO implementation
        # Convert indices to numpy for verl
        indices_np = indices.cpu().numpy()
        
        # Debug logging
        if self.global_step % 10 == 0:
            print(f"\n  [DEBUG] Advantage Computation:")
            print(f"    Rewards shape: {rewards.shape}")
            print(f"    Rewards range: [{rewards.min():.4f}, {rewards.max():.4f}]")
            print(f"    Scaled rewards range: [{scaled_rewards.min():.4f}, {scaled_rewards.max():.4f}]")
            print(f"    Indices: {indices_np}")
            print(f"    Unique indices: {np.unique(indices_np)}")
        
        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=scaled_rewards,
            response_mask=response_mask,
            index=indices_np,
        )
        
        # Debug logging
        if self.global_step % 10 == 0:
            print(f"    Advantages range: [{advantages.min():.4f}, {advantages.max():.4f}]")
            print(f"    Advantages mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")
        
        info = {
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'reward_mean': rewards.mean().item(),
        }
        
        return advantages, info
    
    def _grpo_update(
        self,
        all_frames: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform GRPO policy update using verl's core_algos.
        Uses gradient accumulation and frame caching for optimal memory-speed tradeoff.
        """
        self.policy.train_mode()
        
        B = all_frames.shape[0]
        T_prompt = self.config.n_prompt_frames
        T_gen = all_frames.shape[1] - T_prompt
        
        # OPTIMIZATION: Encode frames ONCE for reference (speeds up reading)
        # BUT: We must re-encode during update to allow gradients to flow!
        # The VAE is frozen, but we need gradients through the DiT
        if self.config.cache_encoded_frames:
            with torch.no_grad():
                cached_latents_for_reference = self.policy.encode_frames(all_frames)
                cached_latents_for_reference = cached_latents_for_reference.detach()
        
        # CRITICAL: Process in micro-batches if batch size > 1
        micro_batch_size = 1  # Process one sample at a time
        num_micro_batches = max(1, B // micro_batch_size)
        
        metrics = defaultdict(list)
        successful_updates = 0
        
        old_log_probs = old_log_probs.detach()
        advantages = advantages.detach()
        
        if response_mask is None:
            response_mask = torch.ones_like(old_log_probs)
        
        # Preallocate buffers for efficiency
        log_probs_buffer = torch.zeros(B, T_gen, device=self.device)
        
        for epoch in range(self.config.grpo_epochs):
            self.optimizer.zero_grad()
            
            epoch_pg_loss = 0
            epoch_pg_clipfrac = 0
            epoch_ppo_kl = 0
            
            # Process in micro-batches
            for mb_idx in range(num_micro_batches):
                start_idx = mb_idx * micro_batch_size
                end_idx = min((mb_idx + 1) * micro_batch_size, B)
                
                # Get micro-batch
                mb_frames = all_frames[start_idx:end_idx]
                mb_actions = actions[start_idx:end_idx]
                mb_old_log_probs = old_log_probs[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]
                mb_response_mask = response_mask[start_idx:end_idx]
                
                # CRITICAL FIX: Always re-encode to allow gradients to flow
                # The VAE encoder is frozen, but gradients need to flow through DiT
                # We encode WITHOUT no_grad() so gradients can backprop
                latents = self.policy.encode_frames(mb_frames)
                
                # Compute log probs with mixed precision
                log_probs = []
                for t in range(T_prompt, T_prompt + T_gen):
                    context = latents[:, :t].detach()
                    target = latents[:, t:t+1].detach()
                    action_idx = t - T_prompt
                    action = mb_actions[:, action_idx:action_idx+1]
                    
                    # Use mixed precision for log prob computation
                    if self.config.use_mixed_precision and self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            log_prob = self.policy.compute_log_prob(context, action, target)
                    else:
                        log_prob = self.policy.compute_log_prob(context, action, target)
                    
                    log_probs.append(log_prob)
                    
                    # Aggressive cleanup
                    del context, target
                
                # Stack log probs
                log_probs = torch.stack(log_probs, dim=1)
                
                # Store in buffer for reuse if needed
                log_probs_buffer[start_idx:end_idx] = log_probs.detach()
                
                # Cleanup
                if not self.config.cache_encoded_frames:
                    del latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Use verl's compute_policy_loss
                pg_loss, pg_clipfrac, ppo_kl, _ = core_algos.compute_policy_loss(
                    old_log_prob=mb_old_log_probs,
                    log_prob=log_probs,
                    advantages=mb_advantages,
                    response_mask=mb_response_mask,
                    cliprange=self.config.clip_ratio,
                    loss_agg_mode="token-mean"
                )
                
                # Entropy loss (simple proxy)
                entropy_proxy = -log_probs.mean()
                entropy_loss = -self.config.entropy_coeff * entropy_proxy
                
                total_loss = pg_loss + entropy_loss
                
                # Scale loss by number of micro-batches for gradient accumulation
                total_loss = total_loss / num_micro_batches
                
                # Backward pass
                if hasattr(self, 'grad_scaler') and self.grad_scaler is not None:
                    self.grad_scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Accumulate metrics
                epoch_pg_loss += pg_loss.item()
                epoch_pg_clipfrac += pg_clipfrac.item()
                epoch_ppo_kl += ppo_kl.item()
                
                # Cleanup
                del log_probs, pg_loss, entropy_loss, total_loss
                del mb_frames, mb_actions, mb_old_log_probs, mb_advantages, mb_response_mask
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Optimization step (after accumulating gradients from all micro-batches)
            if hasattr(self, 'grad_scaler') and self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.get_trainable_parameters(),
                    self.config.grad_clip,
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.get_trainable_parameters(),
                    self.config.grad_clip,
                )
                self.optimizer.step()
            
            successful_updates += 1
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Average metrics over micro-batches
            metrics['pg_loss'].append(epoch_pg_loss / num_micro_batches)
            metrics['total_loss'].append(epoch_pg_loss / num_micro_batches)
            metrics['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            metrics['clip_fraction'].append(epoch_pg_clipfrac / num_micro_batches)
            metrics['kl'].append(epoch_ppo_kl / num_micro_batches)
        
        # Clear cached latents to free memory (if they were created)
        if self.config.cache_encoded_frames:
            del cached_latents_for_reference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        result = {k: np.mean(v) if v else float('nan') for k, v in metrics.items()}
        result['update_success_rate'] = successful_updates / self.config.grpo_epochs
        
        return result
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        import time
        import gc
        
        # Aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Prepare inputs
        if 'initial_frame' in batch:
            initial_frames = batch['initial_frame'].to(self.device)
        else:
            frames = batch['frames'].to(self.device)
            initial_frames = frames[:, :self.config.n_prompt_frames]
            del frames  # Free memory immediately
        
        actions = batch['actions'].to(self.device)
        target_actions = actions[:, :self.config.max_gen_frames]
        
        # 1. Generate rollouts (with group repetition)
        print(f"  [Profile] Starting Rollout Generation...")
        gen_start = time.perf_counter()
        rollout_data = self._generate_rollouts(initial_frames, target_actions)
        gen_time = time.perf_counter() - gen_start
        print(f"  [Profile] Generation finished in {gen_time:.2f}s")
        
        # Delete initial_frames after rollout generation
        del initial_frames, target_actions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. Compute rewards
        print(f"  [Profile] Starting Reward Computation...")
        reward_start = time.perf_counter()
        rewards, reward_info = self._compute_rewards(
            rollout_data['all_frames'],
            rollout_data['actions'],
        )
        reward_time = time.perf_counter() - reward_start
        print(f"  [Profile] Reward computation finished in {reward_time:.2f}s")
        
        # Print detailed reward breakdown immediately
        print(f"  [Rewards] Mean Total: {rewards.mean().item():.4f}")
        if 'reward/rik' in reward_info:
            print(f"    RIK (Inverse Kinematics): {reward_info.get('reward/rik', 0):.4f}")
        if 'reward/rtc' in reward_info:
            print(f"    RTC (Temporal Consistency): {reward_info.get('reward/rtc', 0):.4f}")
        if 'reward/raq' in reward_info:
            print(f"    RAQ (Aesthetic Quality): {reward_info.get('reward/raq', 0):.4f}")
        
        # 3. Compute advantages (GRPO)
        print(f"  [Profile] Starting Advantage Computation...")
        adv_start = time.perf_counter()
        advantages, adv_info = self._compute_advantages(
            rewards,
            rollout_data['indices'],
        )
        adv_time = time.perf_counter() - adv_start
        print(f"  [Profile] Advantage computation finished in {adv_time:.2f}s")
        
        # Save reward statistics before deleting
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        
        # Delete rewards after computing advantages
        del rewards
        
        # 4. Update policy
        print(f"  [Profile] Starting GRPO Update...")
        grpo_start = time.perf_counter()
        update_metrics = self._grpo_update(
            rollout_data['all_frames'],
            rollout_data['actions'],
            rollout_data['log_probs'],
            advantages,
        )
        grpo_time = time.perf_counter() - grpo_start
        print(f"  [Profile] GRPO Update finished in {grpo_time:.2f}s")
        
        # Clean up rollout data
        del rollout_data, advantages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Metrics
        metrics = {
            'reward/mean': reward_mean,
            'reward/std': reward_std,
            **{f'reward/{k}': v for k, v in reward_info.items() if not k.startswith('time_')},
            **{f'advantage/{k}': v for k, v in adv_info.items()},
            **{f'train/{k}': v for k, v in update_metrics.items()},
            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
            'time/generation_sec': gen_time,
            'time/reward_total_sec': reward_time,
            'time/grpo_update_sec': grpo_time,
        }
        
        print(f"  Step {self.global_step}: R={metrics['reward/mean']:.3f}, "
              f"L={metrics['train/total_loss']:.4f}, "
              f"Clip={metrics['train/clip_fraction']:.2f}")
        
        # Detailed logging every 10 steps
        if self.global_step % 10 == 0:
            print(f"\n  === Detailed Metrics (Step {self.global_step}) ===")
            print(f"  Rewards: mean={reward_mean:.4f}, std={reward_std:.4f}")
            print(f"  Advantages: mean={adv_info.get('advantage_mean', 0):.4f}, "
                  f"std={adv_info.get('advantage_std', 0):.4f}")
            print(f"  Policy Loss: {metrics['train/pg_loss']:.4f}")
            print(f"  Total Loss: {metrics['train/total_loss']:.4f}")
            print(f"  Grad Norm: {metrics['train/grad_norm']:.4f}")
            print(f"  Clip Fraction: {metrics['train/clip_fraction']:.4f}")
            print(f"  KL Divergence: {metrics['train/kl']:.4f}")
            print(f"  Learning Rate: {metrics['train/learning_rate']:.2e}")
            
            # Reward breakdown if available
            if 'reward/rik' in metrics:
                print(f"  Reward Components:")
                print(f"    RIK: {metrics.get('reward/rik', 0):.4f}")
                print(f"    RTC: {metrics.get('reward/rtc', 0):.4f}")
                print(f"    RAQ: {metrics.get('reward/raq', 0):.4f}")
            
            print(f"  Timing: gen={gen_time:.2f}s, reward={reward_time:.2f}s, "
                  f"update={grpo_time:.2f}s")
            print(f"  =====================================\n")
        
        return metrics
    
    def fit(self):
        """Main training loop."""
        print(f"Starting Oasis GRPO Training (Group Size: {self.config.group_size})")
        
        if self.train_dataloader is None:
            return
        
        self.global_step = 0
        
        for epoch in range(self.config.total_epochs):
            epoch_pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch in epoch_pbar:
                if self.global_step >= self.config.total_training_steps:
                    break
                
                metrics = self.train_step(batch)
                
                if self.logger is not None:
                    self.logger.log(metrics, step=self.global_step)
                
                if self.config.save_freq > 0 and self.global_step % self.config.save_freq == 0:
                    self.save_checkpoint()
                
                if self.config.video_save_freq > 0 and self.global_step % self.config.video_save_freq == 0:
                    self.save_sample_video(batch)
                
                self.global_step += 1
            
            self.save_checkpoint()

    def save_sample_video(self, batch, suffix=""):
        """Save sample video."""
        # Implementation similar to original, adapted for new structure
        try:
            import sys
            from pathlib import Path
            utils_path = Path(__file__).parent.parent / "utils"
            if str(utils_path) not in sys.path:
                sys.path.insert(0, str(utils_path))
            from video_utils import frames_to_video
            
            if 'initial_frame' in batch:
                initial_frames = batch['initial_frame'].to(self.device)
            else:
                frames = batch['frames'].to(self.device)
                initial_frames = frames[:, :self.config.n_prompt_frames]
            
            actions = batch['actions'].to(self.device)
            
            # Take first sample
            initial_frames = initial_frames[:1]
            actions = actions[:1, :self.config.max_gen_frames]
            
            self.policy.eval_mode()
            with torch.no_grad():
                generated_frames = self.policy.generate_sequence(
                    initial_frames=initial_frames,
                    actions=actions,
                    num_frames=self.config.max_gen_frames,
                )
            
            all_frames = torch.cat([initial_frames, generated_frames], dim=1)
            video_frames = all_frames[0]
            
            video_dir = os.path.join(self.config.video_save_dir, self.config.experiment_name)
            os.makedirs(video_dir, exist_ok=True)
            
            filename = f"step_{self.global_step}{suffix}.mp4"
            video_path = os.path.join(video_dir, filename)
            
            frames_to_video(video_frames, video_path, fps=10)
            print(f"  Saved video: {video_path}")
            
        except Exception as e:
            print(f"  Warning: Failed to save video: {e}")

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.config.checkpoint_dir, f"step_{self.global_step}")
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model_state_dict': self.policy.dit.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__,
        }, os.path.join(path, 'checkpoint.pt'))
        print(f"Saved checkpoint to {path}")

def create_oasis_grpo_trainer(oasis_ckpt, vae_ckpt, reward_models_dir="models_for_rl_finetuning", **kwargs):
    config = OasisGRPOConfig(oasis_ckpt=oasis_ckpt, vae_ckpt=vae_ckpt, reward_models_dir=reward_models_dir, **kwargs)
    return OasisGRPOTrainer(config)
