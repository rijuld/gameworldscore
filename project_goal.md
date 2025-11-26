RL-Enhanced World Model Generation: Ground-Truth-Free
Optimization for Long-Horizon Interactive Gameplay
Georgy Savva: gs4288@nyu.edu Rijul Dahiya: rd3629@nyu.edu
1 Introduction
Self-supervised pretraining achieves strong results on short video
clips but eventually drifts during longer rollouts due to test-train
distribution mismatch from compounding errors. Recent work like
RLVR [1] attempts to mitigate this by fine-tuning models that
use generated frames as conditioning, but their reward function
still compares against ground-truth frames. We argue this ap-
proach is fundamentally limited: asking models to predict ground-
truth frames doesn’t account for stochastic elements inherent in
interactive worlds. Instead, we propose using RL with rewards
based purely on generated output quality, similar to successful
approaches in traditional video generation [2,3].
The Oasis model [4] demonstrates excellent performance
for short clips through self-supervised pretraining on in-
teractive gameplay. We extend this by introducing RL-
GameWorldScore, a ground-truth-free reward function adapted
from Matrix-Game’s GameWorldScore benchmark [5]. This en-
ables optimization for action fidelity, temporal consistency, and
visual quality over extended sequences without requiring ground-
truth frame comparisons.


2 Methodology
2.1 Base Architecture
We build on Oasis [4], an open-source diffusion transformer for
interactive world modeling. The model generates frames condi-
tioned on previous frames and player actions through latent dif-
fusion.
3 Experimental Design
3.1 Evaluation Metrics
Primary: GameWorldScore benchmark [5] covering visual qual-
ity, temporal quality, controllability, and physical rule under-
standing.
Supplemental:
• Long-horizon stability: Frames until significant visual/logical
degradation (target: 200+ frames).
• Human studies: User ratings of gameplay coherence and distin-
guishability from real gameplay.
Baselines:
• Supervised Baseline: Original pre-trained Oasis without RL
fine-tuning.
• MSE Reward Baseline: Oasis fine-tuned with RL using
pixel-level MSE reconstruction reward.
Ablations: Remove each reward component (RIK, RTC, RAQ)
individually.

2.2 RL-GameWorldScore Reward Function
Total reward combines three components from GameWorldScore:
Rtotal = w1RIK + w2RTC + w3RAQ (1)
Inverse Kinematics Score (RIK). Measures action fidelity us-
ing the pre-trained IDM from GameWorldScore. Reward is nega-
tive cross-entropy between intended action at and IDM prediction
from generated transition:
RIK =−CrossEntropy(IDM(st,st+1),at) (2)
Temporal Consistency (RTC). Uses Temporal Consistency
and Motion Smoothness metrics from GameWorldScore: CLIP
feature similarity and frame interpolation network.
Aesthetic Quality (RAQ). Uses Image Quality (MUSIQ) and
Aesthetic (LAION) predictors from GameWorldScore.

Training Procedure:
References
[1] Wu et al., “RL for Autoregressive World Models,”
arXiv:2505.13934, 2025.
[2] SkyReels Team, “SKYREELS-V2: INFINITE-LENGTH
FILM GENERATIVE MODEL,” arXiv:2504.13074, 2025.
[3] J. Liu et al., “Improving Video Generation with Human Feed-
back,” arXiv:2501.13918, 2025.
[4] Oasis Team, “Oasis: Open Agent World Model,”
https://oasis-model.github.io/, 2024.
[5] Y. Zhang et al., “Matrix-Game: Interactive World Foundation
Model,” arXiv:2506.18701, 2025.
[6] J. Schulman et al., “Proximal Policy Optimization Algo-
rithms,” arXiv:1707.06347, 2017.