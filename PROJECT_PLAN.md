# RL-Enhanced World Model Generation

**Ground-Truth-Free Optimization for Long-Horizon Interactive Gameplay**

**Authors:** Georgy Savva (gs4288@nyu.edu), Rijul Dahiya (rd3629@nyu.edu)

---

## 1. Introduction

### Problem Statement

Self-supervised pretraining achieves strong results on short video clips but eventually drifts during longer rollouts due to test-train distribution mismatch from compounding errors. Recent work like RLVR [1] attempts to mitigate this by fine-tuning models that use generated frames as conditioning, but their reward function still compares against ground-truth frames.

### Our Approach

We argue this approach is fundamentally limited: **asking models to predict ground-truth frames doesn't account for stochastic elements inherent in interactive worlds**. Instead, we propose using RL with rewards based purely on generated output quality, similar to successful approaches in traditional video generation [2,3].

The Oasis model [4] demonstrates excellent performance for short clips through self-supervised pretraining on interactive gameplay. We extend this by introducing **RL-GameWorldScore**, a ground-truth-free reward function adapted from Matrix-Game's GameWorldScore benchmark [5]. This enables optimization for action fidelity, temporal consistency, and visual quality over extended sequences without requiring ground-truth frame comparisons.

### Training Pipeline Overview

1. Start with pre-trained Oasis model
2. Generate rollouts (H=16 to 128 frames) using random policy
3. Compute RL-GameWorldScore for each frame
4. Estimate advantages with GAE (λ=0.95)
5. Update parameters via PPO
6. Gradually increase rollout horizon H (curriculum learning)

---

## 2. Methodology

### 2.1 Base Architecture

We build on **Oasis** [4], an open-source diffusion transformer for interactive world modeling. The model generates frames conditioned on previous frames and player actions through latent diffusion.

**Key Components:**
- **DiT (Diffusion Transformer)**: Generates latent representations
- **VAE**: Encodes/decodes frames to/from latent space
- **Action Conditioning**: Integrates player actions into generation process

### 2.2 RL-GameWorldScore Reward Function

Total reward combines three components from GameWorldScore:

```
R_total = w₁·R_IK + w₂·R_TC + w₃·R_AQ
```

#### Inverse Kinematics Score (R_IK)

Measures action fidelity using the pre-trained IDM from GameWorldScore. Reward is negative cross-entropy between intended action aₜ and IDM prediction from generated transition:

```
R_IK = -CrossEntropy(IDM(sₜ, sₜ₊₁), aₜ)
```

**Purpose:** Ensures generated frames accurately reflect the intended actions.

#### Temporal Consistency (R_TC)

Uses Temporal Consistency and Motion Smoothness metrics from GameWorldScore:
- CLIP feature similarity between consecutive frames
- Frame interpolation network quality

**Purpose:** Maintains smooth, coherent transitions between frames.

#### Aesthetic Quality (R_AQ)

Uses Image Quality (MUSIQ) and Aesthetic (LAION) predictors from GameWorldScore.

**Purpose:** Ensures generated frames are visually appealing and high-quality.

### 2.3 RL Training Framework

We use **PPO** [6] with KL divergence regularization against the original supervised policy:

```
L_total = L_PPO + λ_KL · KL(π_θ || π_supervised)
```

**Why PPO?**
- Stable training with clipped objective
- Efficient sample reuse
- Well-suited for continuous action spaces (diffusion noise)

**KL Regularization:**
- Prevents catastrophic forgetting of pre-trained knowledge
- Maintains distribution close to supervised baseline
- Adjustable via λ_KL hyperparameter

---

## 3. Experimental Design

### 3.1 Evaluation Metrics

#### Primary Metric
**GameWorldScore Benchmark** [5] covering:
- Visual quality
- Temporal quality
- Controllability
- Physical rule understanding

#### Supplemental Metrics
- **Long-horizon stability**: Frames until significant visual/logical degradation (target: 200+ frames)
- **Human studies**: User ratings of gameplay coherence and distinguishability from real gameplay

### 3.2 Baselines

1. **Supervised Baseline**: Original pre-trained Oasis without RL fine-tuning
2. **MSE Reward Baseline**: Oasis fine-tuned with RL using pixel-level MSE reconstruction reward

### 3.3 Ablation Studies

Remove each reward component individually:
- **No R_IK**: Test importance of action fidelity
- **No R_TC**: Test importance of temporal consistency
- **No R_AQ**: Test importance of aesthetic quality

### 3.4 Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| RL training instability | KL penalty + curriculum learning |
| Reward hacking | Multi-component reward design |
| Computational costs | Start from pre-trained model |
| Metric-reality gap | Human studies validation |

---

## 4. Implementation Details

### 4.1 Model Configuration
- **Base Model**: Oasis DiT-S/2
- **VAE**: vit-l-20-shallow-encoder
- **Input Resolution**: 360×640
- **Latent Dimensions**: Variable based on VAE patch size

### 4.2 Training Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 (actor), 1e-3 (critic)
- **PPO Epochs**: 4
- **Batch Size**: 8-16 (depending on GPU memory)
- **GAE λ**: 0.95
- **Discount γ**: 0.99
- **KL Penalty**: λ_KL = 0.1

### 4.3 Curriculum Learning
- **Initial Horizon**: 16 frames
- **Target Horizon**: 128 frames
- **Increase Schedule**: Double every 10k steps when stability threshold met

### 4.4 Reward Weights
Initial configuration:
- w₁ (R_IK) = 0.4
- w₂ (R_TC) = 0.3
- w₃ (R_AQ) = 0.3

---

## 5. Expected Outcomes

### 5.1 Quantitative Goals
- **GameWorldScore**: >10% improvement over supervised baseline
- **Long-horizon stability**: Generate coherent sequences of 200+ frames
- **Action fidelity**: >90% accuracy on IDM prediction

### 5.2 Qualitative Goals
- Visually indistinguishable from real gameplay (human evaluation)
- Smooth, natural transitions between frames
- Consistent physics and game logic

---

## 6. Project Timeline

| Week | Milestone |
|------|-----------|
| 1-2 | Setup infrastructure, integrate reward models |
| 3-4 | Implement PPO training loop, initial experiments |
| 5-6 | Curriculum learning, hyperparameter tuning |
| 7-8 | Ablation studies, baseline comparisons |
| 9-10 | Human studies, final evaluation, paper writing |

---

## 7. References

[1] Wu et al., "RL for Autoregressive World Models," arXiv:2505.13934, 2025.

[2] SkyReels Team, "SKYREELS-V2: INFINITE-LENGTH FILM GENERATIVE MODEL," arXiv:2504.13074, 2025.

[3] J. Liu et al., "Improving Video Generation with Human Feedback," arXiv:2501.13918, 2025.

[4] Oasis Team, "Oasis: Open Agent World Model," https://oasis-model.github.io/, 2024.

[5] Y. Zhang et al., "Matrix-Game: Interactive World Foundation Model," arXiv:2506.18701, 2025.

[6] J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

---

## 8. Repository Structure

```
RLFS/
├── open-oasis/              # Main codebase
│   ├── oasis_verl/          # RL training framework
│   │   ├── models.py        # ValueNetwork, RewardModel
│   │   ├── workers.py       # OasisWorker, OasisRewardWorker, OasisCriticWorker
│   │   ├── trainer.py       # OasisRayPPOTrainer
│   │   └── config.yaml      # Training configuration
│   ├── verl/                # Local verl library (from RLVR-World)
│   ├── models_for_rl_finetuning/  # Reward model components
│   ├── train_oasis_verl.py  # Training entry point
│   └── tests/               # Unit tests
├── Dataset/                 # Training data (gitignored)
└── .venv/                   # Python environment (gitignored)
```
