# Oasis Finetuning with GameWorldScore

This repository contains code for finetuning the Oasis world model using reinforcement learning with the GameWorldScore reward system. The system enables ground-truth-free finetuning of diffusion-based world models for Minecraft video generation.

## Overview

The Oasis finetuning system uses:
- **Oasis**: A diffusion transformer (DiT) world model for Minecraft
- **GameWorldScore**: A unified reward system with 5 components:
  - **RIK** (Inverse Kinematics Score): Action fidelity using VPT IDM
  - **RTC** (Temporal Consistency Score): Motion smoothness using RAFT optical flow
  - **RAQ** (Aesthetic Quality Score): Visual quality using CLIP + aesthetic predictor
  - **RRG** (Reality Grounding Score): Domain anchoring to prevent drift
  - **AD** (Anti-Drift Reward): Sharpness, motion, texture, and anti-grid components
- **GRPO** (Group Relative Policy Optimization): Stable RL algorithm for diffusion model finetuning

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: A100 or similar with 40GB+ VRAM)
- PyTorch 2.0+

## Installation

### 1. Clone Required Repositories

The code expects the following repositories to be cloned in the parent directory:

```bash
# Navigate to the repository root (RL-Project)
cd /path/to/RL-Project

# Clone open-oasis (Oasis model implementation)
git clone https://github.com/etched-ai/open-oasis.git

# Clone RLVR-World (GRPO training infrastructure)
git clone https://github.com/thuml/RLVR-World.git

# Clone VPT (Video Pre-Training - for Inverse Dynamics Model)
git clone https://github.com/openai/Video-Pre-Training.git VPT
```

**Directory Structure:**
```
RL-Project/
├── oasis-finetuning/          # This code
├── open-oasis/                # Oasis model code
├── RLVR-World/                # GRPO training infrastructure
└── VPT/                       # Video Pre-Training (for IDM)
```

### 2. Set Up Python Environment

```bash
cd oasis-finetuning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Model Checkpoints

#### Oasis Model Checkpoints

The Oasis model checkpoints will be automatically downloaded from HuggingFace on first use, or you can download them manually:

```bash
# The model checkpoints are stored in:
# ~/.cache/huggingface/hub/models--Etched--oasis-500m/

# Or download manually and update paths in config/default.yaml:
# - oasis_ckpt: path to oasis500m.safetensors
# - vae_ckpt: path to vit-l-20.safetensors
```

#### VPT IDM Checkpoints (for RIK reward)

Download the VPT Inverse Dynamics Model checkpoints:

```bash
# Create directory for reward models
mkdir -p models_for_rl_finetuning

# Download VPT IDM (you may need to find the official download link)
# Place the following files in models_for_rl_finetuning/:
# - 4x_idm.model
# - 4x_idm.weights
```

**Note:** The VPT IDM files can be obtained from the [VPT repository](https://github.com/openai/Video-Pre-Training) or the official VPT release.

#### CLIP and Aesthetic Predictor (for RAQ reward)

The CLIP model will be automatically downloaded from HuggingFace. For the aesthetic predictor:

```bash
# Optional: Download aesthetic predictor checkpoint
# Place in models_for_rl_finetuning/aesthetic_predictor.pth
# (If not provided, a default MLP will be used)
```

### 4. Configure Environment Variables (Optional)

If repositories are in non-standard locations, set environment variables:

```bash
export RLVR_WORLD_PATH=/path/to/RLVR-World/vid_wm/verl
export VPT_PATH=/path/to/VPT
```

## Configuration

All configuration is managed through `config/default.yaml`. Key settings:

- **Model paths**: Oasis and VAE checkpoint paths
- **Reward weights**: Adjust weights for RIK, RTC, RAQ, RRG, and AD components
- **Training hyperparameters**: Learning rate, batch size, GRPO settings
- **Data paths**: Dataset directory and frame settings

Edit `config/default.yaml` or create a custom config file.

## Usage

### Basic Training

```bash
cd oasis-finetuning

# Train with default configuration
python train.py

# Train with custom config file
python train.py --config config/custom.yaml

# Override specific parameters
python train.py --learning-rate 5e-5 --group-size 4 --no-wandb
```

### Command-Line Arguments

Common overrides:

```bash
python train.py \
    --oasis-ckpt /path/to/oasis500m.safetensors \
    --vae-ckpt /path/to/vit-l-20.safetensors \
    --data-dir /path/to/dataset \
    --learning-rate 1e-5 \
    --total-steps 10000 \
    --group-size 4 \
    --reward-scale 10.0 \
    --device cuda \
    --no-wandb  # Disable Weights & Biases logging
```

### Disable Specific Reward Components

```bash
python train.py --no-rik    # Disable RIK reward
python train.py --no-rtc    # Disable RTC reward
python train.py --no-raq    # Disable RAQ reward
```

### Resume from Checkpoint

```bash
python train.py --resume-from checkpoints/step_1000/checkpoint.pt
```

## Dataset Format

The system expects Minecraft gameplay frames. Supported formats:

1. **Screenshots**: Directory of images
   ```
   Dataset/screenshots/
   ├── category1/
   │   ├── image1.png
   │   └── image2.png
   └── category2/
       └── ...
   ```

2. **Video files**: MP4 files (will be split into frames)

3. **MiDaS dataset**: Preprocessed MiDaS format

Configure in `config/default.yaml`:
```yaml
data:
  data_dir: "Dataset/screenshots"
  dataset_type: "screenshots"  # or "video", "midas", "auto"
  frame_height: 360
  frame_width: 640
```

## Training on SLURM

A SLURM script is provided for cluster training:

```bash
sbatch train_a100.slurm
```

Edit `train_a100.slurm` to configure:
- GPU allocation
- Job name and output paths
- Environment setup

## Monitoring Training

### Weights & Biases (Optional)

Enable W&B logging in config:
```yaml
logging:
  use_wandb: true
  project_name: "oasis-finetuning"
  experiment_name: "experiment-1"
```

Or disable via command line:
```bash
python train.py --no-wandb
```

### Checkpoints

Checkpoints are saved to `checkpoints/` directory:
- Saved every `save_freq` steps (default: 100)
- Includes model state, optimizer state, and training step

### Sample Videos

Sample videos are saved to `videos/` directory:
- Generated every `video_save_freq` steps (default: 50)
- Shows generated sequences for visual inspection

## Testing Checkpoints

### Test Most Recent Checkpoint

Test the most recent checkpoint on test data:

```bash
cd oasis-finetuning

# Test most recent checkpoint
python test_checkpoint.py

# Test with specific number of samples
python test_checkpoint.py --max-samples 50

# Save generated videos
python test_checkpoint.py --save-videos

# Test specific checkpoint
python test_checkpoint.py --checkpoint checkpoints/oasis_grpo/step_1000

# Custom output directory
python test_checkpoint.py --output-dir my_test_results
```

### Test Script Features

The test script (`test_checkpoint.py`) provides:

- **Automatic checkpoint detection**: Finds the most recent checkpoint automatically
- **Comprehensive evaluation**: Computes GameWorldScore rewards and component metrics
- **Per-frame analysis**: Shows reward degradation over time
- **Video generation**: Optionally saves sample videos for visual inspection
- **JSON results**: Saves detailed metrics to JSON file

### Test Output

The test script generates:

1. **Console output**: Summary statistics and metrics
2. **JSON results file**: `test_results/test_results_step_*.json` with detailed metrics
3. **Sample videos** (if `--save-videos`): `test_results/videos/sample_*.mp4`

Example output:
```
Evaluation Results
==================
Checkpoint: Step 1000
Number of samples evaluated: 50

Overall Reward Statistics:
  Mean: 0.7234 ± 0.1234
  Range: [0.4567, 0.9123]

Per-Frame Reward (showing degradation over time):
  Frame 1: 0.7456 ± 0.1123
  Frame 2: 0.7234 ± 0.1234
  ...

Component Rewards:
  rik_reward: 0.8123 ± 0.0987
  rtc_reward: 0.7234 ± 0.1123
  raq_reward: 0.6789 ± 0.1345
  ...
```

## Troubleshooting

### Import Errors

**RLVR-World not found:**
```bash
# Ensure RLVR-World is cloned in parent directory
# Or set environment variable:
export RLVR_WORLD_PATH=/path/to/RLVR-World/vid_wm/verl
```

**VPT not found:**
```bash
# Ensure VPT is cloned in parent directory
# Or set environment variable:
export VPT_PATH=/path/to/VPT
```

**open-oasis not found:**
```bash
# Ensure open-oasis is cloned in parent directory
git clone https://github.com/etched-ai/open-oasis.git ../open-oasis
```

### Memory Issues

If running out of GPU memory:

1. Reduce batch size in config:
   ```yaml
   data:
     train_batch_size: 1
     max_gen_frames: 2  # Reduce number of generated frames
   ```

2. Enable CPU offloading:
   ```yaml
   memory:
     offload_reward_to_cpu: true
     offload_ref_policy_to_cpu: true
   ```

3. Enable gradient checkpointing:
   ```yaml
   memory:
     use_gradient_checkpointing: true
   ```

### VPT IDM Loading Issues

If VPT IDM fails to load:

1. Ensure VPT repository is cloned and dependencies installed
2. Check that `4x_idm.model` and `4x_idm.weights` exist in `models_for_rl_finetuning/`
3. Install minerl mock (handled automatically, but check `utils/minerl_mock.py`)

## Project Structure

```
oasis-finetuning/
├── config/              # Configuration files
│   ├── default.yaml     # Main config file
│   └── loader.py        # Config loader
├── data/                # Data loading
│   ├── minecraft_dataset.py
│   └── action_utils.py
├── models/              # Model definitions
│   ├── oasis_policy.py  # Oasis policy wrapper
│   └── oasis_vae.py     # VAE wrapper
├── rewards/             # Reward components
│   ├── game_world_score.py      # Unified reward
│   ├── inverse_kinematics.py    # RIK
│   ├── temporal_consistency.py  # RTC
│   ├── aesthetic_quality.py     # RAQ
│   ├── reality_grounding.py     # RRG
│   └── anti_drift.py            # AD
├── trainer/             # Training infrastructure
│   └── oasis_grpo_trainer.py
├── workers/             # Worker implementations
│   ├── oasis_actor.py
│   └── oasis_rollout.py
├── utils/               # Utilities
│   ├── diffusion.py
│   ├── minerl_mock.py
│   └── video_utils.py
├── train.py             # Main training script
└── requirements.txt     # Python dependencies
```

## Citation

If you use this code, please cite:

```bibtex
@misc{oasis-finetuning,
  title={Oasis Finetuning with GameWorldScore},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/oasis-finetuning}
}
```

## License

[Specify your license here]

## Acknowledgments

- [Oasis](https://github.com/etched-ai/open-oasis) - Diffusion transformer world model
- [RLVR-World](https://github.com/thuml/RLVR-World) - GRPO training infrastructure
- [VPT](https://github.com/openai/Video-Pre-Training) - Video Pre-Training and Inverse Dynamics Model

