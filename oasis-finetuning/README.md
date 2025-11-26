# Oasis RL Finetuning Pipeline

A unified RL finetuning pipeline for the Oasis world model, integrating code from both the Oasis repository (model, tokenizer/decoder, diffusion transformer) and the RLVR-World repository (PPO/GRPO trainer, rollout workers, KL-regularized update loop).

## Overview

This pipeline enables **ground-truth-free reinforcement learning** for world model finetuning using the **GameWorldScore** reward function.

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ground-Truth-Free RL Training                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT (from dataset):                                               │
│    • First frame (single image)                                      │
│    • Action sequence (random or sampled)                             │
│                                                                      │
│  GENERATION (Oasis World Model):                                     │
│    • Generates ENTIRE video from first frame + actions               │
│    • No ground-truth future frames used                              │
│                                                                      │
│  REWARD (GameWorldScore - computed on GENERATED frames only):        │
│    • RIK: Does the transition match the action? (IDM)                │
│    • RTC: Are frames temporally consistent? (CLIP)                   │
│    • RAQ: Do frames look visually good? (Aesthetic)                  │
│                                                                      │
│  UPDATE (PPO with KL regularization):                                │
│    • Policy gradient based on rewards                                │
│    • KL penalty to prevent divergence from pretrained model          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### GameWorldScore Components

- **RIK (Inverse Kinematics Score)**: Measures action fidelity using a pre-trained IDM
- **RTC (Temporal Consistency Score)**: Uses CLIP feature similarity for temporal smoothness
- **RAQ (Aesthetic Quality Score)**: Combines MUSIQ and LAION aesthetic predictors

Unlike traditional approaches that compare against ground-truth frames, this enables optimization on long-horizon rollouts without future frame access.

## Installation

```bash
cd oasis-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Download Models

### 1. Oasis Model
Download the Oasis DiT and VAE checkpoints from the [Oasis repository](https://github.com/etched-ai/open-oasis):

```bash
# Create a directory for model checkpoints (optional, but recommended)
mkdir -p checkpoints

# Download Oasis-500M to a specific location
# Option 1: Using -O to specify exact output path
wget https://huggingface.co/Etched/oasis-500m/resolve/main/oasis500m.safetensors -O checkpoints/oasis500m.safetensors

# Option 2: Using -P to specify output directory
# wget https://huggingface.co/Etched/oasis-500m/resolve/main/oasis500m.safetensors -P checkpoints/

# Download VAE to a specific location
wget https://huggingface.co/Etched/oasis-500m/resolve/main/vit-l-20.safetensors -O checkpoints/vit-l-20.safetensors
# Or: wget https://huggingface.co/Etched/oasis-500m/resolve/main/vit-l-20.safetensors -P checkpoints/
```

**Note:** The default paths in `train.py` are `checkpoints/oasis500m.safetensors` and `checkpoints/vit-l-20.safetensors`. If you save to a different location, specify the paths when running training:
```bash
python train.py \
    --oasis-ckpt /path/to/oasis500m.safetensors \
    --vae-ckpt /path/to/vit-l-20.safetensors
```

### 2. Reward Models
Download the pre-trained models for GameWorldScore:

```bash
python download_reward_models.py
```

This downloads:
- CLIP (temporal consistency)
- Aesthetic Predictor (visual quality)
- IDM (action fidelity)

## Quick Start

```bash
# Basic training (uses default paths: checkpoints/oasis500m.safetensors and checkpoints/vit-l-20.safetensors)
python train.py --data-dir ../open-oasis/sample_data

# With custom reward weights
python train.py \
    --rik-weight 2.0 \
    --rtc-weight 1.0 \
    --raq-weight 0.5

# Using GRPO advantage estimation
python train.py \
    --adv-estimator grpo \
    --use-wandb

# With custom model paths (if saved elsewhere)
python train.py \
    --oasis-ckpt /path/to/oasis500m.safetensors \
    --vae-ckpt /path/to/vit-l-20.safetensors
```

## Project Structure

```
oasis-finetuning/
├── __init__.py                 # Package initialization
├── train.py                    # Main training entry point
├── download_reward_models.py   # Download reward model checkpoints
├── requirements.txt            # Dependencies
│
├── config/
│   └── default.yaml           # Default configuration
│
├── models/
│   ├── __init__.py
│   ├── oasis_policy.py        # Oasis DiT wrapped as RL policy
│   └── oasis_vae.py           # VAE encoder/decoder wrapper
│
├── rewards/
│   ├── __init__.py
│   ├── game_world_score.py    # Unified GameWorldScore reward
│   ├── inverse_kinematics.py  # RIK component (action fidelity)
│   ├── temporal_consistency.py # RTC component (smoothness)
│   └── aesthetic_quality.py   # RAQ component (visual quality)
│
├── trainer/
│   ├── __init__.py
│   └── oasis_ppo_trainer.py   # PPO trainer for Oasis
│
├── workers/
│   ├── __init__.py
│   ├── oasis_actor.py         # Actor worker for policy updates
│   └── oasis_rollout.py       # Rollout worker for generation
│
├── data/
│   ├── __init__.py
│   ├── minecraft_dataset.py   # Minecraft gameplay dataset
│   └── action_utils.py        # Action encoding/decoding
│
└── utils/
    ├── __init__.py
    ├── diffusion.py           # Diffusion utilities
    └── video_utils.py         # Video I/O utilities
```

## Architecture

### OasisPolicy
Wraps the Oasis DiT to provide:
- `generate_sequence()`: Autoregressive frame generation
- `compute_log_prob()`: Log probability computation for PPO
- VAE encoding/decoding for latent space operations

### GameWorldScoreReward
Ground-truth-free reward combining:
```
R_total = w1 * RIK + w2 * RTC + w3 * RAQ
```

Each component can be computed purely from generated frames without reference to ground truth.

### OasisPPOTrainer
Training loop that:
1. Generates long-horizon rollouts using Oasis policy
2. Computes GameWorldScore rewards
3. Estimates advantages (supports GAE, GRPO, REINFORCE++)
4. Updates policy with PPO and KL regularization

## Configuration

Key configuration options:

```yaml
# Reward weights
reward:
  rik_weight: 1.0  # Action fidelity
  rtc_weight: 1.0  # Temporal consistency
  raq_weight: 1.0  # Aesthetic quality

# PPO settings
ppo:
  clip_ratio: 0.2
  gamma: 0.99
  lam: 0.95

# KL regularization
kl:
  use_kl_in_reward: true
  kl_coeff: 0.001

# Advantage estimation
advantage:
  estimator: "grpo"  # or "gae", "reinforce_plus_plus"
```

## Integration with RLVR-World

This pipeline is designed to be compatible with RLVR-World's training infrastructure:

```python
# Using RLVR-World's core algorithms
from verl.trainer.ppo import core_algos

# Adaptive KL controller
kl_controller = core_algos.AdaptiveKLController(
    init_kl_coef=0.001,
    target_kl=0.1,
    horizon=1000,
)

# Advantage estimation
advantages = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rewards,
    response_mask=mask,
    index=batch_indices,
)
```

## Extending the Pipeline

### Custom Reward Functions
Add new reward components by implementing:

```python
class CustomReward(nn.Module):
    def compute_reward(self, frame_t, frame_t1, action):
        # Your reward logic
        return reward, info
```

### Custom Advantage Estimators
Integrate with RLVR-World's estimators or add your own in `trainer/oasis_ppo_trainer.py`.

## References

- [Oasis: Open Agent World Model](https://oasis-model.github.io/)
- [RLVR: RL for Autoregressive World Models](https://arxiv.org/abs/2505.13934)
- [Matrix-Game: GameWorldScore Benchmark](https://arxiv.org/abs/2506.18701)
- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

## License

This project builds upon:
- Oasis (MIT License)
- RLVR-World (Apache 2.0 License)

