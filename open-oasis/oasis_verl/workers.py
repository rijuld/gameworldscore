import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from einops import rearrange
from omegaconf import DictConfig
import numpy as np
import os
import sys
import socket
from dataclasses import dataclass

# from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch, Execute
from verl import DataProto

# Import Oasis models
from dit import DiT_models
from vae import VAE_models
from utils import sigmoid_beta_schedule

# Import local models
from oasis_verl.models import ValueNetwork, RewardModel, MidasDataset

@dataclass
class WorkerMeta:
    keys = [
        "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"
    ]

    def __init__(self, store) -> None:
        self._store = store

    def to_dict(self):
        return {f"_{key.lower()}": self._store.get(f"_{key.lower()}", None) for key in WorkerMeta.keys}

class WorkerHelper:
    def _get_node_ip(self):
        def get_node_ip_by_sdk():
            if os.getenv("WG_BACKEND", None) == "ray":
                import ray
                return ray._private.services.get_node_ip_address()
            else:
                raise NotImplementedError("WG_BACKEND now just support ray mode.")

        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = get_node_ip_by_sdk()

        host_ip = host_ip_by_env or host_ip_by_sdk
        return host_ip

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(('', 0))
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        return self._get_node_ip(), str(self._get_free_port())

class BaseWorker(WorkerHelper):
    """A custom Worker class that bypasses CUDA checks."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        disable_worker_init = int(os.environ.get('DISABLE_WORKER_INIT', 0))
        if disable_worker_init:
            return instance

        rank = os.environ.get("RANK", None)
        worker_group_prefix = os.environ.get("WG_PREFIX", None)

        if None not in [rank, worker_group_prefix] and 'ActorClass(' not in cls.__name__:
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))

        return instance

    def _configure_before_init(self, register_center_name: str, rank: int):
        assert isinstance(rank, int), f"rank must be int, instead of {type(rank)}"

        if rank == 0:
            master_addr, master_port = self.get_availale_master_addr_port()
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }

            if os.getenv("WG_BACKEND", None) == "ray":
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center
                self.register_center = create_worker_group_register_center(name=register_center_name,
                                                                           info=rank_zero_info)

            os.environ.update(rank_zero_info)

    def __init__(self, cuda_visible_devices=None) -> None:
        import os
        import torch

        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "12345")

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            '_world_size': world_size,
            '_rank': rank,
            '_local_world_size': local_world_size,
            '_local_rank': local_rank,
            '_master_addr': master_addr,
            '_master_port': master_port
        }
        if cuda_visible_devices is not None:
            store['_cuda_visible_devices'] = cuda_visible_devices

        meta = WorkerMeta(store=store)
        self._configure_with_meta(meta=meta)

    def _configure_with_meta(self, meta: WorkerMeta):
        assert isinstance(meta, WorkerMeta)
        self.__dict__.update(meta.to_dict())
        for key in WorkerMeta.keys:
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = str(self._master_addr).replace("[", "").replace(
            "]", "") if self._master_addr else ""

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def execute_func_rank_zero(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        return result

class OasisWorker(BaseWorker):
    def __init__(self, config: DictConfig, role: str):
        print(f"Initializing OasisWorker with role {role}...")
        super().__init__()
        self.config = config
        self.role = role
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Oasis DiT
        self.model = DiT_models["DiT-S/2"]().to(self.device)
        if config.model.oasis_ckpt and os.path.exists(config.model.oasis_ckpt):
            # Load checkpoint logic here
            pass
            
        # Load VAE
        self.vae = VAE_models["vit-l-20-shallow-encoder"]().to(self.device)
        if config.model.vae_ckpt and os.path.exists(config.model.vae_ckpt):
            # Load checkpoint logic here
            pass
            
        # Reward Model - Moved to OasisRewardWorker
        # self.reward_model = RewardModel(self.device).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.actor.optim.lr)
        
        # Dataset (for initial frames)
        transform = transforms.Compose([
            transforms.Resize((360, 640)),
            transforms.ToTensor(),
        ])
        self.dataset = MidasDataset(root_dir=config.data.dataset_path, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.data_iter = iter(self.dataloader)
        
        # Scheduler params
        self.max_noise_level = 1000
        self.ddim_steps = config.rollout.ddim_steps
        self.noise_range = torch.linspace(-1, self.max_noise_level - 1, self.ddim_steps + 1)
        self.betas = sigmoid_beta_schedule(self.max_noise_level).float().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")
        
        self.vae_scaling_factor = 0.07843137255
        self.stabilization_level = 15

    def get_initial_frame(self, batch_size=1):
        try:
            frames = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            frames = next(self.data_iter)
        
        frames = frames.to(self.device)
        
        # Encode with VAE
        B = frames.shape[0]
        H, W = frames.shape[-2:]
        frames_flat = rearrange(frames, "b c h w -> (b 1) c h w")
        
        with torch.no_grad():
            # VAE encode expects inputs in [-1, 1]
            latents = self.vae.encode(frames_flat * 2 - 1).mean * self.vae_scaling_factor
        
        latents = rearrange(latents, "(b t) (h w) c -> b t c h w", t=1, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        return latents

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # prompts might contain initial states or we sample from dataset
        # For now, let's ignore prompts and sample from dataset as per train_rl.py logic
        # But in a real RLHF loop, prompts usually come from the dataset.
        
        batch_size = prompts.batch.batch_size[0] if prompts.batch is not None else 1
        n_frames = self.config.rollout.n_frames
        
        self.model.eval()
        # self.reward_model.reset()
        
        x = self.get_initial_frame(batch_size)
        actions = torch.randn(batch_size, n_frames + 1, 25).to(self.device) # Random actions for now
        
        # Storage for trajectory
        states = []
        log_probs = []
        rewards = []
        values = [] # We don't compute values here, Critic does. But we need to store states for Critic.
        
        # DDIM Loop
        for i in range(1, n_frames + 1):
            chunk = torch.randn((batch_size, 1, *x.shape[-3:]), device=self.device)
            chunk = torch.clamp(chunk, -20, +20)
            x_curr = torch.cat([x, chunk], dim=1)
            
            start_frame = max(0, i - self.model.max_frames + 1)
            
            # Inner diffusion loop
            for noise_idx in reversed(range(1, self.ddim_steps + 1)):
                t_ctx = torch.full((batch_size, i), self.stabilization_level - 1, dtype=torch.long, device=self.device)
                t = torch.full((batch_size, 1), self.noise_range[noise_idx], dtype=torch.long, device=self.device)
                t_next = torch.full((batch_size, 1), self.noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
                t_next = torch.where(t_next < 0, t, t_next)
                
                t_full = torch.cat([t_ctx, t], dim=1)
                
                x_window = x_curr[:, start_frame:]
                t_window = t_full[:, start_frame:]
                
                with torch.no_grad():
                    v = self.model(x_window, t_window, actions[:, start_frame : i + 1])
                
                # ... (DDIM update logic same as train_rl.py) ...
                alpha = self.alphas_cumprod[t_window]
                alpha_next = self.alphas_cumprod[torch.cat([t_ctx, t_next], dim=1)[:, start_frame:]]
                
                alpha_last = alpha[:, -1:]
                alpha_next_last = alpha_next[:, -1:]
                
                x_curr_last = x_window[:, -1:]
                v_last = v[:, -1:]
                
                x_start = alpha_last.sqrt() * x_curr_last - (1 - alpha_last).sqrt() * v_last
                x_noise = ((1 / alpha_last).sqrt() * x_curr_last - x_start) / (1 / alpha_last - 1).sqrt()
                
                if noise_idx == 1:
                    alpha_next_last = torch.ones_like(alpha_next_last)
                
                x_pred = alpha_next_last.sqrt() * x_start + x_noise * (1 - alpha_next_last).sqrt()
                x_curr[:, -1:] = x_pred
            
            # Compute Reward (Moved to OasisRewardWorker)
            # frames_to_decode = x_curr[:, -2:] if x_curr.shape[1] >= 2 else x_curr
            # ...
            # r = self.reward_model.compute_reward(...)
            
            states.append(x_curr[:, -1].clone()) # Store state
            # rewards.append(r)
            x = x_curr

        # Construct DataProto
        # We need to return a DataProto that contains the trajectory data
        # Batch size is B. Sequence length is n_frames.
        
        states = torch.stack(states, dim=1) # (B, T, C, H, W)
        # rewards = torch.stack(rewards, dim=1) # (B, T)
        
        # We need to return 'batch' tensordict with keys expected by PPO
        # 'input_ids' (states), 'action_ids' (actions), 'rewards', etc.
        # Since we are continuous, we might need to adapt naming or use custom keys.
        
        batch_dict = {
            'states': states,
            'actions': actions[:, 1:], # (B, T, 25) - skipping initial dummy action?
            # 'rewards': rewards, # Computed by RewardWorker
            # 'log_probs': ... # We need log probs of actions. For now placeholder.
            'old_log_probs': torch.zeros_like(states[:, :, 0, 0, 0]) # Placeholder (B, T)
        }
        
        return DataProto.from_dict(batch_dict)

class OasisRewardWorker(BaseWorker):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reward Model
        self.reward_model = RewardModel(self.device).to(self.device)
        
        # VAE for decoding (needed for reward computation)
        self.vae = VAE_models["vit-l-20-shallow-encoder"]().to(self.device)
        if config.model.vae_ckpt and os.path.exists(config.model.vae_ckpt):
            # Load checkpoint logic here
            pass
        self.vae_scaling_factor = 0.07843137255

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        states = data.batch['states'] # (B, T, C, H, W)
        actions = data.batch['actions'] # (B, T, 25)
        
        B, T = states.shape[:2]
        rewards = []
        
        # We need to reconstruct the sequence of frames to compute rewards
        # The states are latents. We need to decode them.
        # Ideally, we should process in batches, but for now let's iterate or batch process.
        
        # Flatten for VAE decoding
        states_flat = rearrange(states, "b t c h w -> (b t) (h w) c")
        
        with torch.no_grad():
            decoded = self.vae.decode(states_flat / self.vae_scaling_factor)
            decoded = (decoded + 1) / 2
            decoded = torch.clamp(decoded, 0, 1)
            
        decoded_frames = rearrange(decoded, "(b t) c h w -> b t c h w", b=B, t=T)
        
        # Compute rewards per step
        # Note: RewardModel expects (B, T, C, H, W) and returns scalar or (B,).
        # We need (B, T).
        # Our RewardModel implementation in models.py:
        # compute_reward(frames, actions, t) -> returns total_reward for time t
        
        # We need to call it for each t.
        # Also, RewardModel expects 'frames' to be the full history up to t?
        # Let's check models.py. It takes 'frames' and slices it.
        
        reward_tensor = torch.zeros(B, T).to(self.device)
        
        for t in range(T):
            # We need at least 2 frames for some rewards (RTC).
            # If t=0, we might need the initial frame?
            # The 'states' in data.batch likely start from t=1 (first generated frame).
            # We might be missing the initial frame context.
            # For now, let's assume we compute what we can.
            
            # Construct frames input for RewardModel
            # It expects (B, T_current, C, H, W)
            curr_frames = decoded_frames[:, :t+1] 
            
            # Actions: (B, T, 25).
            # RewardModel uses actions[:, t].
            
            # We need to be careful about indices.
            # If states[0] corresponds to t=1.
            
            r = self.reward_model.compute_reward(curr_frames, actions, t)
            reward_tensor[:, t] = r
            # Wait, RewardModel.compute_reward(frames, actions, t):
            #   curr = frames[:, t]
            #   prev = frames[:, t-1]
            # So if we pass curr_frames which has length t+1, we should ask for index t.
            
            reward_tensor[:, t] = r
            
        return DataProto.from_dict({'rewards': reward_tensor})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # PPO Update
        # data.batch contains 'states', 'actions', 'advantages', 'returns', 'old_log_probs'
        
        states = data.batch['states']
        actions = data.batch['actions']
        advantages = data.batch['advantages']
        old_log_probs = data.batch['old_log_probs']
        
        # Re-compute log probs and values (if shared)
        # For diffusion, "action" is the noise prediction v.
        # We need to run the model to get v_pred.
        
        # This is tricky because we need the full context (history) to predict v at each step.
        # But we stored 'states' as single frames.
        # We might need to store full history or reconstruct it.
        # For simplicity, let's assume we just need the current window.
        
        # ... Implementation details for PPO update on Diffusion ...
        # For now, just a placeholder step
        
        self.optimizer.zero_grad()
        # loss = ...
        # loss.backward()
        self.optimizer.step()
        
        return DataProto.from_dict({'metrics': {'actor_loss': 0.0}})

class OasisCriticWorker(BaseWorker):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.value_net = ValueNetwork().to(self.device)
        self.optimizer = optim.AdamW(self.value_net.parameters(), lr=config.critic.optim.lr)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        states = data.batch['states'] # (B, T, C, H, W)
        B, T = states.shape[:2]
        
        states_flat = rearrange(states, "b t c h w -> (b t) c h w")
        t_dummy = torch.zeros(B*T, 1).to(self.device) # Dummy t for now
        
        with torch.no_grad():
            values = self.value_net(states_flat, t_dummy)
            
        values = rearrange(values, "(b t) 1 -> b t", b=B, t=T)
        return DataProto.from_dict({'values': values})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        states = data.batch['states']
        returns = data.batch['returns']
        
        B, T = states.shape[:2]
        states_flat = rearrange(states, "b t c h w -> (b t) c h w")
        returns_flat = rearrange(returns, "b t -> (b t) 1")
        t_dummy = torch.zeros(B*T, 1).to(self.device)
        
        values = self.value_net(states_flat, t_dummy)
        loss = nn.MSELoss()(values, returns_flat)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return DataProto.from_dict({'metrics': {'critic_loss': loss.item()}})
