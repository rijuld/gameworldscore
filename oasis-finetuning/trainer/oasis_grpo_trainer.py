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
    group_size: int = 4  # Number of rollouts per prompt - balanced for memory and variance
    grpo_epochs: int = 1  # Number of update epochs per step - optimized for speed
    
    # Rollout settings
    n_prompt_frames: int = 1
    max_gen_frames: int = 2  # Reduced from 4 to 2 for memory optimization
    
    # GRPO hyperparameters (following RLVR-World)
    learning_rate: float = 1e-4  # Increased for visible learning (diffusion models need higher LR)
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    log_ratio_clip: float = 2.0  # Added back for compatibility
    entropy_coeff: float = 0.001
    grad_clip: float = 1.0
    reward_scale: float = 100.0  # Provides strong signal for GRPO advantages
    
    # Memory optimization settings
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing to save memory
    use_mixed_precision: bool = True  # Enable mixed precision training (FP16)
    offload_reward_to_cpu: bool = False  # Offload reward models to CPU to save GPU memory
    offload_ref_policy_to_cpu: bool = True  # Offload reference policy to CPU (saves ~14GB GPU memory)
    cache_encoded_frames: bool = False  # DISABLED: Caching prevents gradient flow (causes clip=0)
    kl_compute_freq: int = 5  # Compute KL divergence every N steps (1 = every step)
    
    # Performance optimizations
    dataloader_num_workers: int = 4  # Number of workers for data loading
    update_micro_batch_size: int = 4  # Micro-batch size for GRPO updates - MUST match group_size for balanced advantages
    use_torch_compile: bool = False  # Enable torch.compile for potential speedup
    enable_tf32: bool = True  # Enable TensorFloat-32 on Ampere+ GPUs
    
    # KL settings
    use_kl_in_reward: bool = True  # Enable KL divergence penalty (ref policy on CPU to save memory)
    kl_coeff: float = 0.01
    kl_target: float = 0.1
    
    # Reward noise settings
    add_reward_noise: bool = True  # Add small noise to break reward ties for GRPO
    
    # Reward weights (GameWorldScore)
    rik_weight: float = 1.0
    rtc_weight: float = 1.0
    raq_weight: float = 1.0
    require_vpt: bool = True
    use_motion_smoothness: bool = True
    
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
    
    GPU/CPU Offloading Strategy:
    - Policy (DiT + VAE): Always on GPU during training and inference
    - Reward models: On GPU if offload_reward_to_cpu=False, else on CPU
      * If on CPU: frames temporarily moved to CPU for reward computation, then moved back to GPU
      * If on GPU: everything stays on GPU (fastest path)
    - Reference policy: On CPU if offload_ref_policy_to_cpu=True (saves ~14GB GPU memory)
      * Frames temporarily moved to CPU only when computing KL divergence
      * Results immediately moved back to GPU and CPU copies deleted
    - All intermediate tensors: Aggressively deleted and GPU cache cleared after each major operation
    - Training data: Always on GPU during training steps
    """
    
    def __init__(self, config: OasisGRPOConfig):
        self.config = config
        self.device = config.device
        self.global_step = 0
        
        # CRITICAL: Print and verify config values
        print("=" * 60)
        print("CRITICAL CONFIG VALUES:")
        print(f"  group_size: {self.config.group_size}")
        print(f"  update_micro_batch_size: {self.config.update_micro_batch_size}")
        print(f"  grpo_epochs: {self.config.grpo_epochs}")
        print(f"  reward_scale: {self.config.reward_scale}")
        print(f"  learning_rate: {self.config.learning_rate}")
        print("=" * 60)
        
        # FORCE OVERRIDE: Ensure micro_batch_size equals group_size for balanced gradients
        if self.config.update_micro_batch_size != self.config.group_size:
            print(f"⚠️  OVERRIDE: Setting update_micro_batch_size from {self.config.update_micro_batch_size} to {self.config.group_size}")
            self.config.update_micro_batch_size = self.config.group_size
        
        # FORCE OVERRIDE: Ensure grpo_epochs is 1 for efficiency
        if self.config.grpo_epochs != 1:
            print(f"⚠️  OVERRIDE: Setting grpo_epochs from {self.config.grpo_epochs} to 1")
            self.config.grpo_epochs = 1
        
        # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
        if config.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  Enabled TensorFloat-32 (TF32) for faster training")
        
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
        
        # Compile DiT model if requested
        if self.config.use_torch_compile:
            try:
                if hasattr(torch, 'compile'):
                    print("  Compiling policy model for faster inference...")
                    self.policy.dit = torch.compile(self.policy.dit, mode='reduce-overhead')
                    print("  ✓ Model compiled successfully")
            except Exception as e:
                print(f"  ⚠️  Model compilation failed (using uncompiled): {e}")
        else:
            print("  Using uncompiled DiT model (torch.compile disabled)")
        
        # Enable gradient checkpointing only during training (not inference)
        # We'll toggle this in train_step
        self.use_gradient_checkpointing = self.config.use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            print("  Gradient checkpointing will be enabled during training only")
        
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
        
        # Disable torch.compile for reward models due to potential compatibility issues
        # Uncomment below to enable compilation (may cause errors):
        # if reward_device == "cuda":
        #     try:
        #         if hasattr(torch, 'compile'):
        #             print("  Compiling reward models for faster inference...")
        #             if hasattr(self.reward_fn.rtc, 'clip_model'):
        #                 self.reward_fn.rtc.clip_model = torch.compile(
        #                     self.reward_fn.rtc.clip_model, mode='reduce-overhead'
        #                 )
        #             if hasattr(self.reward_fn.raq, 'aesthetic_predictor'):
        #                 self.reward_fn.raq.aesthetic_predictor = torch.compile(
        #                     self.reward_fn.raq.aesthetic_predictor, mode='reduce-overhead'
        #                 )
        #             print("  ✓ Reward models compiled successfully")
        #     except Exception as e:
        #         print(f"  ⚠️  Reward model compilation failed (using uncompiled): {e}")
        if reward_device == "cuda":
            print("  Using uncompiled reward models (torch.compile disabled for stability)")
        
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
                num_workers=self.config.dataloader_num_workers,
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
        
        # Disable gradient checkpointing during inference for speed
        if self.use_gradient_checkpointing:
            if hasattr(self.policy.dit, 'gradient_checkpointing_disable'):
                self.policy.dit.gradient_checkpointing_disable()
            elif hasattr(self.policy.dit, 'disable_gradient_checkpointing'):
                self.policy.dit.disable_gradient_checkpointing()
        
        B = initial_frames.shape[0]
        G = self.config.group_size
        
        # Ensure inputs are on GPU
        if initial_frames.device != self.device:
            initial_frames = initial_frames.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
        
        # Repeat inputs for group generation
        # (B, ...) -> (B*G, ...)
        initial_frames_repeated = initial_frames.repeat_interleave(G, dim=0)
        actions_repeated = actions.repeat_interleave(G, dim=0)
        
        num_gen = min(actions.shape[1], self.config.max_gen_frames)
        
        with torch.no_grad():
            # Generate frames (everything on GPU)
            generated_frames = self.policy.generate_sequence(
                initial_frames=initial_frames_repeated,
                actions=actions_repeated[:, :num_gen],
                num_frames=num_gen,
            )
            
            # Ensure generated frames are on GPU
            if generated_frames.device != self.device:
                generated_frames = generated_frames.to(self.device)
            
            # Compute log probabilities (all on GPU)
            all_frames = torch.cat([initial_frames_repeated, generated_frames], dim=1)
            
            # Ensure all_frames is on GPU
            if all_frames.device != self.device:
                all_frames = all_frames.to(self.device)
            
            # Delete intermediate tensors to free memory
            del generated_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Encode frames (VAE is on GPU, so latents will be on GPU)
            latents = self.policy.encode_frames(all_frames)
            
            # Ensure latents are on GPU
            if latents.device != self.device:
                latents = latents.to(self.device)
            
            # Delete initial_frames_repeated to free memory
            del initial_frames_repeated
            
            # Pre-generate noise for all steps to ensure consistency
            # Latent shape: (B*G, T, C, H, W)
            # We need noise for (B*G, num_gen, C, H, W)
            latent_shape = latents.shape
            noise_shape = (latent_shape[0], num_gen, latent_shape[2], latent_shape[3], latent_shape[4])
            all_noise = torch.randn(noise_shape, device=self.device, dtype=latents.dtype)
            
            log_probs = []
            T_prompt = initial_frames.shape[1]
            
            for t in range(T_prompt, T_prompt + num_gen):
                context = latents[:, :t]
                target = latents[:, t:t+1]
                action_idx = t - T_prompt
                action = actions_repeated[:, action_idx:action_idx+1]
                
                # Get noise for this step
                step_noise = all_noise[:, action_idx:action_idx+1]
                
                log_prob = self.policy.compute_log_prob(context, action, target, noise=step_noise)
                log_probs.append(log_prob.detach())  # Detach to save memory
                
                # Clean up context and target
                del context, target, action
            
            log_probs = torch.stack(log_probs, dim=1)
            
            # Compute reference log probs for KL if needed
            ref_log_probs = None
            if self.ref_policy is not None:
                # Always compute ref log probs if KL is needed in rewards
                # Otherwise, compute based on kl_compute_freq (for logging only)
                if self.config.use_kl_in_reward:
                    compute_kl_this_step = True  # Always compute if used in rewards
                else:
                    compute_kl_this_step = (self.global_step % self.config.kl_compute_freq == 0)
                
                if compute_kl_this_step:
                    # Delete latents before computing ref_latents to free GPU memory
                    del latents
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # If ref policy is on CPU, temporarily move frames to CPU
                    if self.ref_policy_device == "cpu":
                        all_frames_for_ref = all_frames.cpu()
                        actions_for_ref = actions_repeated.cpu()
                    else:
                        all_frames_for_ref = all_frames
                        actions_for_ref = actions_repeated
                    
                    # Compute reference log probs
                    ref_latents = self.ref_policy.encode_frames(all_frames_for_ref)
                    ref_log_probs_list = []
                    
                    for t in range(T_prompt, T_prompt + num_gen):
                        context = ref_latents[:, :t]
                        target = ref_latents[:, t:t+1]
                        action_idx = t - T_prompt
                        action = actions_for_ref[:, action_idx:action_idx+1]
                        
                        ref_log_prob = self.ref_policy.compute_log_prob(context, action, target)
                        ref_log_probs_list.append(ref_log_prob.detach())
                        
                        del context, target, action
                    
                    ref_log_probs = torch.stack(ref_log_probs_list, dim=1)
                    
                    # Immediately move ref_log_probs back to GPU and clean up CPU copies
                    if self.ref_policy_device == "cpu":
                        ref_log_probs = ref_log_probs.to(self.device)
                        del all_frames_for_ref, actions_for_ref, ref_latents
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        del ref_latents
                    
                    del ref_log_probs_list
                else:
                    # Skip KL computation this step
                    del latents
                    ref_log_probs = None
            else:
                del latents
        
        # Create group indices (on GPU)
        # [0, 0, 0, 0, 1, 1, 1, 1, ...]
        indices = torch.arange(B, device=self.device).repeat_interleave(G)
        
        # Ensure all return values are on GPU
        if all_frames.device != self.device:
            all_frames = all_frames.to(self.device)
        if log_probs.device != self.device:
            log_probs = log_probs.to(self.device)
        if ref_log_probs is not None and ref_log_probs.device != self.device:
            ref_log_probs = ref_log_probs.to(self.device)
        actions_return = actions_repeated[:, :num_gen]
        if actions_return.device != self.device:
            actions_return = actions_return.to(self.device)
            
        # Generate noise for future updates (to ensure consistency)
        # We need noise for each generated frame
        # Shape: (B*G, num_gen, C, H/patch, W/patch)
        # We can just generate it now and store it
        # Note: We need to know the latent shape. 
        # We can infer it from the log_probs computation or just generate it matching latents shape
        # Since we deleted latents, we'll recreate the shape info
        latent_h = all_frames.shape[-2] // self.policy.vae.patch_size
        latent_w = all_frames.shape[-1] // self.policy.vae.patch_size
        latent_c = 4 # Standard for Oasis VAE, but better to get from config if possible. 
                     # However, we can just use torch.randn like in compute_log_prob
        
        # Actually, we should have captured the noise used during compute_log_prob above!
        # But compute_log_prob generated it internally.
        # To fix this properly without changing the flow too much:
        # We will generate the noise HERE, and we should have passed it to compute_log_prob above.
        # Since we didn't (in the loop above), the log_probs calculated above used random noise.
        # This is fine for the "old_log_probs" as long as we save THAT noise and use it for "new_log_probs".
        # WAIT: If we didn't pass noise above, compute_log_prob generated random noise and threw it away.
        # We CANNOT recover that noise.
        # FIX: We must generate noise BEFORE the loop above and pass it in.
        
        # RE-IMPLEMENTING THE LOOP WITH PRE-GENERATED NOISE
        # (This replaces the loop logic in the original code, but since I'm editing the end of the function,
        # I need to be careful. I will rewrite the loop part in a separate chunk or assume I can't change it easily here?
        # No, I should rewrite the whole _generate_rollouts method or a large chunk of it.
        # Let's rewrite the loop part in _generate_rollouts using a larger chunk.)
        
        return {
            'generated_frames': None,  # Don't keep generated_frames in memory
            'all_frames': all_frames,  # On GPU
            'log_probs': log_probs,  # On GPU
            'ref_log_probs': ref_log_probs,  # On GPU (if computed)
            'actions': actions_return,  # On GPU
            'indices': indices,  # On GPU
            'noise': all_noise,  # Store noise for updates!
        }
    
    def _compute_rewards(
        self,
        all_frames: torch.Tensor,
        actions: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute GameWorldScore rewards.
        
        Optimized GPU/CPU offloading:
        - If reward models are on GPU: keep everything on GPU
        - If reward models are on CPU: temporarily move frames to CPU, compute, move results back to GPU
        """
        with torch.no_grad():
            # Determine reward device
            reward_device = "cpu" if self.config.offload_reward_to_cpu else self.device
            
            if reward_device == "cpu":
                # Temporarily move to CPU for reward computation
                all_frames_cpu = all_frames.cpu()
                actions_cpu = actions.cpu()
                
                # Compute rewards on CPU
                rewards, info = self.reward_fn.compute_sequence_reward(
                    all_frames_cpu,
                    actions_cpu,
                    return_per_frame=True,
                )
                
                # Immediately move rewards back to GPU and delete CPU copies
                rewards = rewards.to(self.device)
                del all_frames_cpu, actions_cpu
                
                # Clear CPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Everything stays on GPU - fastest path
                # All reward computations (RIK, RTC, RAQ) will keep tensors on GPU
                # Only VPT IDM preprocessing converts to numpy at the last step
                rewards, info = self.reward_fn.compute_sequence_reward(
                    all_frames,
                    actions,
                    return_per_frame=True,
                )
            
            # Detach rewards to prevent gradient tracking
            rewards = rewards.detach()
            
            # NOTE: KL penalty is now applied in _grpo_update as a separate loss term
            # This is more stable than subtracting from rewards before advantage computation
            
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
        
        # Add tiny noise to break ties and ensure variance (Fix 3)
        # This helps GRPO when rewards are very similar
        if self.config.add_reward_noise:
            reward_std = scaled_rewards.std()
            if reward_std > 0:
                noise = torch.randn_like(scaled_rewards) * 0.01 * reward_std
                scaled_rewards = scaled_rewards + noise
        
        # Create response mask if not provided (all ones)
        if response_mask is None:
            response_mask = torch.ones_like(rewards)
            
        # Use verl's GRPO implementation
        # Convert indices to numpy for verl
        indices_np = indices.cpu().numpy()
        
        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=scaled_rewards,
            response_mask=response_mask,
            index=indices_np,
        )
        
        # Debug logging for advantages (Fix 4)
        pos_adv = (advantages > 0).float().mean().item() * 100
        neg_adv = (advantages < 0).float().mean().item() * 100
        zero_adv = (advantages == 0).float().mean().item() * 100
        
        if self.global_step % 10 == 0:  # Log every 10 steps to avoid spam
            print(f"  [DEBUG Advantages] "
                  f"Positive: {pos_adv:.1f}% | Negative: {neg_adv:.1f}% | Zero: {zero_adv:.1f}% | "
                  f"Range: [{advantages.min().item():.3f}, {advantages.max().item():.3f}] | "
                  f"Mean: {advantages.mean().item():.3f} ± {advantages.std().item():.3f}")
            print(f"  [DEBUG Rewards] "
                  f"Range: [{rewards.min().item():.3f}, {rewards.max().item():.3f}] | "
                  f"Mean: {rewards.mean().item():.3f} ± {rewards.std().item():.3f}")
        
        info = {
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'advantage_pos_pct': pos_adv,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
        }
        
        return advantages, info
    
    def _grpo_update(
        self,
        all_frames: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        noise: torch.Tensor,  # Added noise argument
        ref_log_probs: Optional[torch.Tensor] = None,  # For KL loss computation
        response_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform GRPO policy update using verl's core_algos.
        Uses gradient accumulation and frame caching for optimal memory-speed tradeoff.
        
        CRITICAL: noise must be provided and must be the same noise used during rollout
        generation to ensure consistent log probability computation.
        """
        # CRITICAL: Assert noise is provided for consistent log prob computation
        assert noise is not None, "Noise must be provided for consistent log prob computation in GRPO update!"
        
        self.policy.train_mode()
        
        # Enable gradient checkpointing during training for memory efficiency
        if self.use_gradient_checkpointing:
            if hasattr(self.policy.dit, 'gradient_checkpointing_enable'):
                self.policy.dit.gradient_checkpointing_enable()
            elif hasattr(self.policy.dit, 'enable_gradient_checkpointing'):
                self.policy.dit.enable_gradient_checkpointing()
        
        B = all_frames.shape[0]
        T_prompt = self.config.n_prompt_frames
        T_gen = all_frames.shape[1] - T_prompt
        
        # OPTIMIZATION: Encode frames ONCE.
        # Since VAE is frozen, we don't need to backprop through it.
        # We just need the latents as input to the DiT.
        with torch.no_grad():
            all_latents = self.policy.encode_frames(all_frames)
            if all_latents.device != self.device:
                all_latents = all_latents.to(self.device)
        
        # CRITICAL: Process in micro-batches if batch size > 1
        micro_batch_size = self.config.update_micro_batch_size
        num_micro_batches = max(1, (B + micro_batch_size - 1) // micro_batch_size)
        
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
            epoch_kl_loss = 0
            
            # Process in micro-batches
            for mb_idx in range(num_micro_batches):
                start_idx = mb_idx * micro_batch_size
                end_idx = min((mb_idx + 1) * micro_batch_size, B)
                
                # Get micro-batch (ensure on GPU)
                mb_frames = all_frames[start_idx:end_idx]
                if mb_frames.device != self.device:
                    mb_frames = mb_frames.to(self.device)
                
                mb_actions = actions[start_idx:end_idx]
                if mb_actions.device != self.device:
                    mb_actions = mb_actions.to(self.device)
                
                mb_old_log_probs = old_log_probs[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]
                mb_response_mask = response_mask[start_idx:end_idx]
                mb_noise = noise[start_idx:end_idx]  # Get noise for this micro-batch
                
                # Use cached latents
                latents = all_latents[start_idx:end_idx]
                
                # Compute log probs with mixed precision
                log_probs = []
                for t in range(T_prompt, T_prompt + T_gen):
                    context = latents[:, :t].detach()
                    target = latents[:, t:t+1].detach()
                    action_idx = t - T_prompt
                    action = mb_actions[:, action_idx:action_idx+1]
                    
                    # Get noise for this step
                    step_noise = mb_noise[:, action_idx:action_idx+1]

                    # DEBUG: Verify noise is being used correctly (only on first step of first epoch/mb)
                    if epoch == 0 and mb_idx == 0 and t == T_prompt and self.global_step % 10 == 0:
                        print(f"  [NOISE CHECK] step_noise shape: {step_noise.shape}, sum: {step_noise.sum().item():.6f}")
                        print(f"  [NOISE CHECK] old_log_prob for this sample: {mb_old_log_probs[0, action_idx].item():.6f}")
                        
                        # Compute with provided noise
                        with torch.no_grad():
                            log_prob_with_noise = self.policy.compute_log_prob(context, action, target, noise=step_noise)
                            log_prob_without_noise = self.policy.compute_log_prob(context, action, target, noise=None)
                        
                        print(f"  [NOISE CHECK] log_prob WITH noise: {log_prob_with_noise[0].item():.6f}")
                        print(f"  [NOISE CHECK] log_prob WITHOUT noise: {log_prob_without_noise[0].item():.6f}")
                        print(f"  [NOISE CHECK] Difference (with vs without): {abs(log_prob_with_noise[0].item() - log_prob_without_noise[0].item()):.6f}")
                        
                        # Critical: Does new log_prob match old log_prob when using same noise?
                        diff_from_old = abs(log_prob_with_noise[0].item() - mb_old_log_probs[0, action_idx].item())
                        print(f"  [NOISE CHECK] Difference (new vs old): {diff_from_old:.6f}")
                        
                        if abs(log_prob_with_noise[0].item() - log_prob_without_noise[0].item()) < 1e-6:
                            print("  [NOISE CHECK] ❌ NOISE IS NOT BEING USED!")
                        else:
                            print("  [NOISE CHECK] ✅ Noise is being used correctly")
                        
                        if diff_from_old < 1e-4:
                            print("  [NOISE CHECK] ⚠️ new_log_prob ≈ old_log_prob (policy unchanged OR consistent noise)")
                        else:
                            print(f"  [NOISE CHECK] 🔄 Policy has changed: diff={diff_from_old:.6f}")

                    # Use mixed precision for log prob computation
                    if self.config.use_mixed_precision and self.device == "cuda":
                        with torch.amp.autocast('cuda'):
                            log_prob = self.policy.compute_log_prob(context, action, target, noise=step_noise)
                    else:
                        log_prob = self.policy.compute_log_prob(context, action, target, noise=step_noise)
                    
                    log_probs.append(log_prob)
                    
                    # Aggressive cleanup
                    del context, target
                
                # Stack log probs
                log_probs = torch.stack(log_probs, dim=1)
                
                # Priority 2: Validate log probs for NaN/Inf
                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    print(f"  WARNING: NaN/Inf in log_probs detected, skipping this micro-batch")
                    del log_probs
                    continue
                
                # Store in buffer for reuse if needed
                log_probs_buffer[start_idx:end_idx] = log_probs.detach()
                
                # Cleanup
                # if not self.config.cache_encoded_frames:
                #     del latents
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
                
                # Debug logging for loss components (Fix 5)
                if self.global_step % 10 == 0 and mb_idx == 0:  # Log every 10 steps, first micro-batch only
                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    print(f"  [DEBUG Loss] PG: {pg_loss.item():.4f} | Clip%: {pg_clipfrac:.2%} | KL: {ppo_kl:.4f}")
                    print(f"  [DEBUG Ratio] Range: [{ratio.min().item():.3f}, {ratio.max().item():.3f}] | "
                          f"Mean: {ratio.mean().item():.3f}")
                    print(f"  [DEBUG Adv Stats] Range: [{mb_advantages.min().item():.3f}, {mb_advantages.max().item():.3f}] | "
                          f"Mean: {mb_advantages.mean().item():.3f}")
                
                # Entropy loss (simple proxy)
                entropy_proxy = -log_probs.mean()
                entropy_loss = -self.config.entropy_coeff * entropy_proxy
                
                # Priority 1: Compute KL divergence loss (moved from rewards)
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.config.use_kl_in_reward and ref_log_probs is not None:
                    # Get ref_log_probs for this micro-batch
                    mb_ref_log_probs = ref_log_probs[start_idx:end_idx].detach()
                    # KL divergence: log_pi - log_ref (approximation)
                    kl_div = (log_probs - mb_ref_log_probs)
                    kl_loss = self.config.kl_coeff * kl_div.mean()
                
                # Total loss includes KL
                total_loss = pg_loss + entropy_loss + kl_loss
                
                # Priority 2: Validate loss before backward
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  WARNING: NaN/Inf loss detected (pg={pg_loss.item():.4f}, "
                          f"entropy={entropy_loss.item():.4f}, kl={kl_loss.item():.4f}), skipping backward")
                    self.optimizer.zero_grad()
                    continue
                
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
                epoch_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                
                # Cleanup
                del log_probs, pg_loss, entropy_loss, total_loss, kl_loss
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
                
                # Check for NaN gradients before optimizer step
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"  WARNING: NaN/Inf gradient detected (norm={grad_norm}), skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.get_trainable_parameters(),
                    self.config.grad_clip,
                )
                
                # Check for NaN gradients before optimizer step
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"  WARNING: NaN/Inf gradient detected (norm={grad_norm}), skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
            
            # DEBUG: Track weight changes after optimizer step
            if self.global_step % 10 == 0 and epoch == 0:
                with torch.no_grad():
                    param_sample = next(iter(self.policy.get_trainable_parameters()))
                    print(f"  [WEIGHT CHECK] After optimizer.step() - sample param mean: {param_sample.mean().item():.8f}, "
                          f"std: {param_sample.std().item():.6f}, lr: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            successful_updates += 1
            
            # Average metrics over micro-batches
            metrics['pg_loss'].append(epoch_pg_loss / num_micro_batches)
            metrics['total_loss'].append((epoch_pg_loss + epoch_kl_loss) / num_micro_batches)
            metrics['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            metrics['clip_fraction'].append(epoch_pg_clipfrac / num_micro_batches)
            metrics['kl'].append(epoch_ppo_kl / num_micro_batches)
            metrics['kl_loss'].append(epoch_kl_loss / num_micro_batches)
        
        # Clear cached latents to free memory (if they were created)
        if self.config.cache_encoded_frames:
            del cached_latents_for_reference
        
        # Aggressive GPU memory cleanup after training step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        result = {k: np.mean(v) if v else float('nan') for k, v in metrics.items()}
        result['update_success_rate'] = successful_updates / self.config.grpo_epochs
        
        return result
    
    def train_step(self, batch: Dict[str, torch.Tensor], pbar=None) -> Dict[str, float]:
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
        gen_start = time.perf_counter()
        rollout_data = self._generate_rollouts(initial_frames, target_actions)
        gen_time = time.perf_counter() - gen_start
        
        # Delete initial_frames after rollout generation
        del initial_frames, target_actions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. Compute rewards (frames stay on GPU if reward models are on GPU)
        reward_start = time.perf_counter()
        rewards, reward_info = self._compute_rewards(
            rollout_data['all_frames'],
            rollout_data['actions'],
            log_probs=rollout_data.get('log_probs'),
            ref_log_probs=rollout_data.get('ref_log_probs'),
        )
        reward_time = time.perf_counter() - reward_start
        
        # Ensure rewards are on GPU
        if rewards.device != self.device:
            rewards = rewards.to(self.device)
        


        
        # 3. Compute advantages (GRPO)
        adv_start = time.perf_counter()
        advantages, adv_info = self._compute_advantages(
            rewards,
            rollout_data['indices'],
        )
        adv_time = time.perf_counter() - adv_start
        
        # Save reward statistics before deleting
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        
        # Priority 5: Add reward scaling check
        if reward_mean < 0.01:
            print(f"  WARNING: Very small rewards detected (mean={reward_mean:.6f}). "
                  f"Consider increasing reward_scale (current={self.config.reward_scale})")
        
        if reward_std < 1e-6:
            print(f"  WARNING: Near-zero reward variance (std={reward_std:.6f}). "
                  f"Policy may not learn effectively.")
        
        # Delete rewards after computing advantages
        del rewards
        
        # 4. Update policy (ensure all data is on GPU)
        grpo_start = time.perf_counter()
        
        # Ensure all inputs are on GPU
        rollout_frames = rollout_data['all_frames']
        rollout_actions = rollout_data['actions']
        rollout_log_probs = rollout_data['log_probs']
        
        if rollout_frames.device != self.device:
            rollout_frames = rollout_frames.to(self.device)
        if rollout_actions.device != self.device:
            rollout_actions = rollout_actions.to(self.device)
        if rollout_log_probs.device != self.device:
            rollout_log_probs = rollout_log_probs.to(self.device)
        if advantages.device != self.device:
            advantages = advantages.to(self.device)
        
        update_metrics = self._grpo_update(
            rollout_frames,
            rollout_actions,
            rollout_log_probs,
            advantages,
            rollout_data['noise'],  # Pass the noise
            ref_log_probs=rollout_data.get('ref_log_probs'),  # Pass ref_log_probs for KL loss
        )
        grpo_time = time.perf_counter() - grpo_start
        
        # Clean up rollout data and free GPU memory
        del rollout_data, advantages, rollout_frames, rollout_actions, rollout_log_probs
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
        
        # Concise logging: times, sub-rewards, loss, grad norm
        rik_val = reward_info.get('rik_reward', reward_info.get('reward/rik', 0.0))
        rtc_val = reward_info.get('rtc_reward', reward_info.get('reward/rtc', 0.0))
        raq_val = reward_info.get('raq_reward', reward_info.get('reward/raq', 0.0))
        
        # Format progress bar info if available (matches tqdm format: "8/32400 [03:50<198:35:32, 22.07s/it]")
        if pbar is not None:
            n = pbar.n
            total = pbar.total if pbar.total else 0
            fmt_dict = pbar.format_dict
            
            # Get elapsed time in seconds
            elapsed = fmt_dict.get('elapsed', 0)
            elapsed_h = int(elapsed // 3600)
            elapsed_m = int((elapsed % 3600) // 60)
            elapsed_s = int(elapsed % 60)
            elapsed_str = f"{elapsed_h*60 + elapsed_m:02d}:{elapsed_s:02d}" if elapsed_h == 0 else f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"
            
            # Calculate remaining time
            if total > 0 and n > 0 and elapsed > 0:
                rate = elapsed / n
                remaining = rate * (total - n)
                rem_h = int(remaining // 3600)
                rem_m = int((remaining % 3600) // 60)
                rem_s = int(remaining % 60)
                if rem_h > 0:
                    remaining_str = f"{rem_h:02d}:{rem_m:02d}:{rem_s:02d}"
                else:
                    remaining_str = f"{rem_m:02d}:{rem_s:02d}"
            else:
                remaining_str = "??:??:??"
            
            # Get rate
            rate = fmt_dict.get('rate')
            if rate is None or rate <= 0:
                rate = elapsed / n if n > 0 and elapsed > 0 else 0
            rate_str = f"{rate:.2f}s/it" if rate > 0 else "?s/it"
            
            progress_info = f"{n}/{total} [{elapsed_str}<{remaining_str}, {rate_str}]"
        else:
            progress_info = f"Step {self.global_step}"
        
        print(f"{progress_info} | "
              f"gen={gen_time:.2f}s, reward={reward_time:.2f}s, update={grpo_time:.2f}s | "
              f"Reward={reward_mean:.3f}±{reward_std:.3f} (RIK={rik_val:.3f}, RTC={rtc_val:.3f}, RAQ={raq_val:.3f}) | "
              f"loss={metrics['train/total_loss']:.4f}, grad={metrics['train/grad_norm']:.4f}, "
              f"clip={metrics.get('train/clip_fraction', 0.0):.2%}, adv+={adv_info.get('advantage_pos_pct', 50.0):.0f}%",
              flush=True)
        
        # CRITICAL FIX: Step scheduler once per training step (not per epoch)
        # This ensures the learning rate schedule progresses correctly
        if self.scheduler is not None:
            self.scheduler.step()
            
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
                
                metrics = self.train_step(batch, pbar=epoch_pbar)
                
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
            
            # Save reference policy video if available
            if self.ref_policy is not None:
                # Prepare inputs for reference policy
                if self.ref_policy_device == "cpu":
                    ref_initial_frames = initial_frames.cpu()
                    ref_actions = actions.cpu()
                else:
                    ref_initial_frames = initial_frames
                    ref_actions = actions
                
                with torch.no_grad():
                    ref_generated_frames = self.ref_policy.generate_sequence(
                        initial_frames=ref_initial_frames,
                        actions=ref_actions,
                        num_frames=self.config.max_gen_frames,
                    )
                
                # Move back to CPU/GPU for saving (frames_to_video expects CPU usually, but let's keep consistent)
                if ref_generated_frames.device != initial_frames.device:
                    ref_generated_frames = ref_generated_frames.to(initial_frames.device)
                
                ref_all_frames = torch.cat([initial_frames, ref_generated_frames], dim=1)
                ref_video_frames = ref_all_frames[0]
                
                ref_filename = f"step_{self.global_step}{suffix}_ref.mp4"
                ref_video_path = os.path.join(video_dir, ref_filename)
                
                frames_to_video(ref_video_frames, ref_video_path, fps=10)
                print(f"  Saved reference video: {ref_video_path}")
            
        except Exception as e:
            print(f"  Warning: Failed to save video: {e}")

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.config.checkpoint_dir, f"step_{self.global_step}")
        os.makedirs(path, exist_ok=True)
        
        # Save checkpoint (this can be slow for large models)
        checkpoint_path = os.path.join(path, 'checkpoint.pt')
        torch.save({
            'model_state_dict': self.policy.dit.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__,
        }, checkpoint_path)
        print(f"\nSaved checkpoint to {path}", flush=True)

def create_oasis_grpo_trainer(oasis_ckpt, vae_ckpt, reward_models_dir="models_for_rl_finetuning", **kwargs):
    config = OasisGRPOConfig(oasis_ckpt=oasis_ckpt, vae_ckpt=vae_ckpt, reward_models_dir=reward_models_dir, **kwargs)
    return OasisGRPOTrainer(config)
