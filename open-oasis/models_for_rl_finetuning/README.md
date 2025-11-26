# Models for RL Fine-tuning

This directory contains the reward models used for the RL-Enhanced World Model Generation.

## Automated Download

Run the following script to download the available models (CLIP, Aesthetic Predictor):

```bash
python3 download_models.py
```

## Manual Setup

### 1. CLIP
The script downloads `openai/clip-vit-base-patch32`. If you prefer a different version, update `download_models.py`.

### 2. Aesthetic Predictor
The script downloads the LAION aesthetic predictor linear probe (`sac+logos+ava1-l14-linearMSE.pth`).

### 3. Inverse Dynamics Model (IDM)
The "Matrix-Game" IDM is required for the Action Fidelity reward.
Please place the pre-trained IDM weights in this directory as `idm.pt`.

If you do not have the specific IDM, you can train one using the `train_idm.py` script (if provided) or use a compatible Minecraft IDM.
