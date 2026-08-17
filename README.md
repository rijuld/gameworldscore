# RL-Project — Oasis world-model finetuning with GRPO

Reinforcement-learning finetuning of the [Oasis](https://github.com/etched-ai/open-oasis) Minecraft
world model using **GRPO** (Group Relative Policy Optimization) and **GameWorldScore**, a
ground-truth-free reward built from five components.

There are no reference videos to imitate. Instead, the model generates short rollouts, five reward
models score them along different axes, and GRPO pushes the policy toward the higher-scoring
rollouts in each group.

> **Research code.** This is a course/research project, not a maintained library. Expect rough
> edges — see [Known limitations](#known-limitations) before you rely on any of it.

---

## The reward: GameWorldScore

| Component | Name | What it measures | Backed by |
|-----------|------|------------------|-----------|
| **RIK** | Inverse Kinematics | Does the generated video actually reflect the action that was taken? | VPT Inverse Dynamics Model |
| **RTC** | Temporal Consistency | Is motion between frames smooth and coherent? | Optical flow / frame similarity |
| **RAQ** | Aesthetic Quality | Does it look like a good frame? | CLIP + aesthetic predictor |
| **RRG** | Reality Grounding | Does it still look like Minecraft, or has it drifted off-domain? | Domain anchoring |
| **AD** | Anti-Drift | Sharpness, motion, texture, and anti-grid penalties | Hand-built image statistics |

Each component has a weight in `oasis-finetuning/config/default.yaml`; the weighted sum is the
per-frame reward fed to GRPO. Components can be zeroed from the command line (`--no-rik`,
`--no-rtc`, `--no-raq`).

## Repository layout

```
RL-Project/
├── oasis-finetuning/     # ← the actual work in this repo (training, rewards, trainer)
├── open-oasis/           # vendored Oasis model code (DiT + VAE), MIT, Etched & Decart
├── RLVR-World/           # submodule — GRPO / verl training infrastructure
└── VPT/                  # submodule — OpenAI Video-Pre-Training, for the IDM used by RIK
```

Inside `oasis-finetuning/`:

```
├── config/               # default.yaml (single source of truth) + loader
├── data/                 # Minecraft frame datasets and action encoding
├── models/               # OasisPolicy (DiT wrapper) and OasisVAE
├── rewards/              # one module per GameWorldScore component
├── trainer/              # oasis_grpo_trainer.py — the GRPO training loop
├── workers/              # actor and rollout workers
├── utils/                # diffusion helpers, video I/O, minerl mock
├── train.py              # main training entry point
├── test_checkpoint.py    # evaluate a saved checkpoint
└── eval_ref_policy_rewards.py  # baseline rewards for the un-finetuned policy
```

## Requirements

- **Python 3.8+**
- **CUDA GPU with 40GB+ VRAM** (developed on an A100). The pipeline runs a diffusion policy, a
  frozen reference policy, and several reward models at once.
- PyTorch 2.0+

## Installation

### 1. Clone with the sibling repositories

`oasis-finetuning` resolves `open-oasis`, `RLVR-World`, and `VPT` as siblings at the repository
root, so all four directories must be present.

```bash
git clone https://github.com/rijuld/RL-Project.git
cd RL-Project

# RLVR-World and VPT are recorded as submodules but this repo has no .gitmodules file,
# so `git submodule update --init` will not work. Clone them by hand into place:
git clone https://github.com/thuml/RLVR-World.git RLVR-World
git clone https://github.com/openai/Video-Pre-Training.git VPT
```

If you keep them somewhere else, point the code at them with environment variables instead:

```bash
export RLVR_WORLD_PATH=/path/to/RLVR-World/vid_wm/verl
export VPT_PATH=/path/to/VPT
```

### 2. Python environment

```bash
cd oasis-finetuning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pyyaml          # imported by config/loader.py but missing from requirements.txt
```

### 3. Model checkpoints

| Checkpoint | Used by | Where to get it |
|------------|---------|-----------------|
| `oasis500m.safetensors` | the policy | [Etched/oasis-500m](https://huggingface.co/Etched/oasis-500m) on Hugging Face |
| `vit-l-20.safetensors` | the VAE | same Hugging Face repo |
| `4x_idm.model` + `4x_idm.weights` | RIK reward | [VPT](https://github.com/openai/Video-Pre-Training) releases — place in `models_for_rl_finetuning/` |
| CLIP + aesthetic predictor | RAQ reward | CLIP downloads automatically; the aesthetic predictor is optional (a default MLP is used if absent) |

> [!IMPORTANT]
> `config/default.yaml` ships with **absolute paths from the original author's machine**
> (`/home/rd3629/.cache/huggingface/...`). Change `model.oasis_ckpt` and `model.vae_ckpt` to your
> own paths, or override them per-run with `--oasis-ckpt` / `--vae-ckpt`.

### 4. Dataset

Point `data.data_dir` at Minecraft gameplay frames. Three layouts are supported via
`data.dataset_type`:

- `screenshots` — a directory of images, optionally grouped into subdirectories
- `video` — MP4 files, split into frames on load
- `midas` — preprocessed MiDaS format
- `auto` — detect from the directory contents

## Usage

### Train

```bash
cd oasis-finetuning

python train.py                                  # use config/default.yaml
python train.py --config config/custom.yaml      # a different config
python train.py --learning-rate 5e-5 --no-wandb  # override individual values
python train.py --resume-from checkpoints/oasis_grpo/step_1000
```

Every flag defaults to `None` and falls back to the YAML value, so the config file stays the single
source of truth.

| Flag | Overrides |
|------|-----------|
| `--config` | which YAML file to load |
| `--oasis-ckpt`, `--vae-ckpt` | model checkpoint paths |
| `--data-dir` | dataset directory |
| `--learning-rate`, `--total-steps` | training schedule |
| `--group-size`, `--reward-scale` | GRPO settings |
| `--checkpoint-dir`, `--resume-from` | checkpointing |
| `--device`, `--seed` | runtime (`--seed` defaults to 42) |
| `--no-wandb` | disable Weights & Biases logging |
| `--no-kl` | disable the KL penalty |
| `--no-rik`, `--no-rtc`, `--no-raq` | zero out that reward component |

Checkpoints land in `checkpoints/oasis_grpo/` every 100 steps and sample videos in `samples/` every
50 steps — both configurable.

### Evaluate a checkpoint

```bash
python test_checkpoint.py                        # most recent checkpoint, auto-detected
python test_checkpoint.py --max-samples 50 --save-videos
python test_checkpoint.py --checkpoint checkpoints/oasis_grpo/step_1000
```

Writes summary statistics to the console, detailed metrics to
`test_results/test_results_step_*.json`, and (with `--save-videos`) sample rollouts as MP4.

### Baseline rewards

To see what the reward looks like *before* any finetuning:

```bash
python eval_ref_policy_rewards.py --max-batches 10
```

### Run on SLURM

```bash
sbatch train_a100.slurm
```

The script is written for a specific cluster account and scratch path — edit `--account`, the `cd`
target, and the conda environment name before using it.

### Tests

```bash
python -m unittest discover tests
```

## Monitoring

Weights & Biases logging is on by default (`logging.use_wandb: true`), under project
`oasis_rl_finetuning`. Run `wandb login` first, or pass `--no-wandb`.

## Known limitations

Documented honestly so nobody rediscovers them the hard way:

- **`RLVR-World` and `VPT` are broken submodule references.** They are recorded as gitlinks but
  there is no `.gitmodules`, so a plain clone leaves two empty directories. Clone them manually as
  shown above.
- **KL regularization is incomplete.** See
  [`oasis-finetuning/GRPO_ANALYSIS.md`](oasis-finetuning/GRPO_ANALYSIS.md) for the full write-up.
  The core GRPO algorithm (group-normalized advantages, clipped policy loss, no value function) is
  correct; the KL term is applied inside the policy loss rather than subtracted from rewards, and
  the adaptive KL controller is initialized but never updated — which is why
  `kl.use_adaptive_kl` is `false` in the default config.
- **`config/default.yaml` contains machine-specific absolute paths** that must be edited before a
  first run.
- **`requirements.txt` is missing `pyyaml`**, which the config loader imports.
- **`train_a100.slurm` has a stray trailing `"`** on its last line and hardcodes one cluster's
  account and scratch directory.
- Default settings are tuned small for memory (`train_batch_size: 1`, `max_gen_frames: 2`,
  `group_size: 2`, `ddim_steps: 4`) — they fit on one GPU, not a serious training run.

## Contributing

Issues and pull requests are welcome. If you are looking for somewhere to start, the items under
[Known limitations](#known-limitations) are all real and self-contained.

1. Fork and branch: `git checkout -b my-change`
2. Keep configuration in `config/default.yaml` rather than hardcoding values in Python — the config
   file is deliberately the single source of truth
3. Run `python -m unittest discover tests` before opening a PR
4. Describe what you changed and, for anything touching the reward or trainer, what you observed in
   training

## Acknowledgments

- [Oasis / open-oasis](https://github.com/etched-ai/open-oasis) — Etched & Decart, the diffusion
  transformer world model this project finetunes (vendored here under its MIT license)
- [RLVR-World](https://github.com/thuml/RLVR-World) — GRPO / verl training infrastructure
- [VPT (Video Pre-Training)](https://github.com/openai/Video-Pre-Training) — OpenAI, the inverse
  dynamics model behind the RIK reward

## License

This repository has no top-level license file, so the code under `oasis-finetuning/` is
"all rights reserved" by default and cannot be legally reused by others. If it is meant to be open
source, add a `LICENSE` file — MIT would match the vendored `open-oasis/` code — and update this
section.

The vendored `open-oasis/` directory keeps its own MIT license (see
[`open-oasis/LICENSE`](open-oasis/LICENSE)), which must be preserved.

---

For the long-form setup guide, including troubleshooting and memory-tuning recipes, see
[`README_OASIS_FINETUNING.md`](README_OASIS_FINETUNING.md).
