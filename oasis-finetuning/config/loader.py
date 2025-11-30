"""
Config loader for Oasis GRPO training.
All config values are loaded from YAML - no hardcoded defaults in Python.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, Dict

# Default config path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default.yaml")


def load_yaml_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested YAML config into flat dictionary."""
    flat = {}
    
    # Model
    if 'model' in config:
        flat['oasis_ckpt'] = config['model'].get('oasis_ckpt')
        flat['vae_ckpt'] = config['model'].get('vae_ckpt')
    
    # Reward
    if 'reward' in config:
        r = config['reward']
        flat['reward_models_dir'] = r.get('models_dir')
        flat['rik_weight'] = r.get('rik_weight')
        flat['rtc_weight'] = r.get('rtc_weight')
        flat['raq_weight'] = r.get('raq_weight')
        flat['require_vpt'] = r.get('require_vpt')
        flat['use_motion_smoothness'] = r.get('use_motion_smoothness')
        flat['add_reward_noise'] = r.get('add_reward_noise')
    
    # Data
    if 'data' in config:
        d = config['data']
        flat['data_dir'] = d.get('data_dir')
        flat['dataset_type'] = d.get('dataset_type')
        flat['train_batch_size'] = d.get('train_batch_size')
        flat['n_prompt_frames'] = d.get('n_prompt_frames')
        flat['max_gen_frames'] = d.get('max_gen_frames')
        flat['frame_size'] = (d.get('frame_height'), d.get('frame_width'))
        flat['dataloader_num_workers'] = d.get('num_workers')
    
    # Training
    if 'training' in config:
        t = config['training']
        flat['total_epochs'] = t.get('total_epochs')
        flat['total_training_steps'] = t.get('total_training_steps')
        flat['learning_rate'] = t.get('learning_rate')
        flat['grad_clip'] = t.get('grad_clip')
        flat['device'] = t.get('device')
    
    # GRPO
    if 'grpo' in config:
        g = config['grpo']
        flat['group_size'] = g.get('group_size')
        flat['grpo_epochs'] = g.get('grpo_epochs')
        flat['update_micro_batch_size'] = g.get('update_micro_batch_size')
        flat['gamma'] = g.get('gamma')
        flat['lam'] = g.get('lam')
        flat['clip_ratio'] = g.get('clip_ratio')
        flat['log_ratio_clip'] = g.get('log_ratio_clip')
        flat['entropy_coeff'] = g.get('entropy_coeff')
        flat['reward_scale'] = g.get('reward_scale')
        flat['adv_estimator'] = g.get('adv_estimator')
    
    # KL
    if 'kl' in config:
        k = config['kl']
        flat['use_kl_in_reward'] = k.get('use_kl_in_reward')
        flat['kl_coeff'] = k.get('kl_coeff')
        flat['kl_target'] = k.get('kl_target')
        flat['kl_compute_freq'] = k.get('kl_compute_freq')
    
    # Memory
    if 'memory' in config:
        m = config['memory']
        flat['use_gradient_checkpointing'] = m.get('use_gradient_checkpointing')
        flat['use_mixed_precision'] = m.get('use_mixed_precision')
        flat['offload_reward_to_cpu'] = m.get('offload_reward_to_cpu')
        flat['offload_ref_policy_to_cpu'] = m.get('offload_ref_policy_to_cpu')
        flat['cache_encoded_frames'] = m.get('cache_encoded_frames')
        flat['use_torch_compile'] = m.get('use_torch_compile')
        flat['enable_tf32'] = m.get('enable_tf32')
    
    # Checkpoint
    if 'checkpoint' in config:
        c = config['checkpoint']
        flat['save_freq'] = c.get('save_freq')
        flat['test_freq'] = c.get('test_freq')
        flat['checkpoint_dir'] = c.get('checkpoint_dir')
    
    # Video
    if 'video' in config:
        v = config['video']
        flat['video_save_freq'] = v.get('save_freq')
        flat['video_save_dir'] = v.get('save_dir')
    
    # Logging
    if 'logging' in config:
        l = config['logging']
        flat['project_name'] = l.get('project_name')
        flat['experiment_name'] = l.get('experiment_name')
        flat['use_wandb'] = l.get('use_wandb')
    
    return flat


@dataclass
class OasisGRPOConfig:
    """
    Configuration for Oasis GRPO training.
    All values loaded from YAML - no hardcoded defaults.
    """
    # Model paths
    oasis_ckpt: str
    vae_ckpt: str
    reward_models_dir: str
    
    # Data settings
    data_dir: str
    dataset_type: str
    frame_size: Tuple[int, int]
    
    # Training settings
    total_epochs: int
    total_training_steps: int
    train_batch_size: int
    group_size: int
    grpo_epochs: int
    
    # Rollout settings
    n_prompt_frames: int
    max_gen_frames: int
    
    # GRPO hyperparameters
    learning_rate: float
    gamma: float
    lam: float
    clip_ratio: float
    log_ratio_clip: float
    entropy_coeff: float
    grad_clip: float
    reward_scale: float
    
    # Memory optimization
    use_gradient_checkpointing: bool
    use_mixed_precision: bool
    offload_reward_to_cpu: bool
    offload_ref_policy_to_cpu: bool
    cache_encoded_frames: bool
    
    # Performance
    dataloader_num_workers: int
    update_micro_batch_size: int
    use_torch_compile: bool
    enable_tf32: bool
    
    # KL settings
    use_kl_in_reward: bool
    kl_coeff: float
    kl_target: float
    kl_compute_freq: int
    
    # Reward settings
    add_reward_noise: bool
    rik_weight: float
    rtc_weight: float
    raq_weight: float
    require_vpt: bool
    use_motion_smoothness: bool
    
    # Advantage estimation
    adv_estimator: str
    
    # Checkpointing
    save_freq: int
    test_freq: int
    checkpoint_dir: str
    
    # Video saving
    video_save_freq: int
    video_save_dir: str
    
    # Logging
    project_name: str
    experiment_name: str
    use_wandb: bool
    
    # Device
    device: str


def load_config(config_path: str = DEFAULT_CONFIG_PATH, **overrides) -> OasisGRPOConfig:
    """
    Load config from YAML and apply any overrides.
    
    Args:
        config_path: Path to YAML config file
        **overrides: Key-value pairs to override config values
    
    Returns:
        OasisGRPOConfig dataclass instance
    """
    # Load and flatten YAML
    yaml_config = load_yaml_config(config_path)
    flat_config = flatten_config(yaml_config)
    
    # Apply overrides (from command line args)
    for key, value in overrides.items():
        if value is not None:  # Only override if explicitly set
            flat_config[key] = value
    
    # Create config object
    return OasisGRPOConfig(**flat_config)


def print_config(config: OasisGRPOConfig):
    """Print config values for debugging."""
    print("=" * 60)
    print("CONFIGURATION (from YAML)")
    print("=" * 60)
    print(f"  Model:")
    print(f"    oasis_ckpt: {config.oasis_ckpt}")
    print(f"    vae_ckpt: {config.vae_ckpt}")
    print(f"  Training:")
    print(f"    learning_rate: {config.learning_rate:.2e}")
    print(f"    group_size: {config.group_size}")
    print(f"    grpo_epochs: {config.grpo_epochs}")
    print(f"    update_micro_batch_size: {config.update_micro_batch_size}")
    print(f"    reward_scale: {config.reward_scale}")
    print(f"    grad_clip: {config.grad_clip}")
    print(f"  Memory:")
    print(f"    use_mixed_precision: {config.use_mixed_precision}")
    print(f"    offload_ref_policy_to_cpu: {config.offload_ref_policy_to_cpu}")
    print(f"  Reward:")
    print(f"    rik_weight: {config.rik_weight}")
    print(f"    rtc_weight: {config.rtc_weight}")
    print(f"    raq_weight: {config.raq_weight}")
    print("=" * 60)
