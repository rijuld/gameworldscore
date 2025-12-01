"""
TemporalConsistencyRewardV4
Optimized for batch size = 1.

Features:
- Stable reward formula (1 / (1 + loss)) for batch size 1
- Automatic disabling of normalization when B=1
- Multiscale L1 + SSIM photometric warping loss
- Forward-backward flow consistency (small default weight)
- Temporal TV flicker loss
- Proper RAFT input normalization and flow resizing
- No forced upscaling (safe for small images)
- Batched (B,T,C,H,W) with B=1 supported
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import RAFT
try:
    import torchvision.models.optical_flow as optical_flow_models
except Exception:
    optical_flow_models = None


# ---------------------------- Lightweight SSIM ---------------------------- #
def _gaussian_kernel(window_size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_map(img1, img2, window_size=11, K=(0.01, 0.03), L=1.0):
    B, C, H, W = img1.shape
    device = img1.device
    window = _gaussian_kernel(window_size, 1.5, device).unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(img1.view(B * C, 1, H, W), window, padding=window_size // 2)
    mu2 = F.conv2d(img2.view(B * C, 1, H, W), window, padding=window_size // 2)

    mu1 = mu1.view(B, C, H, W)
    mu2 = mu2.view(B, C, H, W)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d((img1 * img1).view(B * C, 1, H, W), window, padding=window_size // 2)
        .view(B, C, H, W) - mu1_sq
    )

    sigma2_sq = (
        F.conv2d((img2 * img2).view(B * C, 1, H, W), window, padding=window_size // 2)
        .view(B, C, H, W) - mu2_sq
    )

    sigma12 = (
        F.conv2d((img1 * img2).view(B * C, 1, H, W), window, padding=window_size // 2)
        .view(B, C, H, W) - mu1_mu2
    )

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.view(B, C, -1).mean(dim=2).mean(dim=1)  # (B,)


# ---------------------------- TemporalConsistencyReward ---------------------------- #
class TemporalConsistencyRewardV2(nn.Module):
    def __init__(
        self,
        device='cuda',
        raft_model_name='raft_large',
        reward_type='reciprocal',   # safe for batch size = 1
        alpha=1.0,                  # only used if reward_type='exp'
        use_ssim=True,
        ssim_weight=0.4,            # FIXED: Balanced to prevent L1 grid artifacts
        l1_weight=0.6,              # FIXED: Reduced slightly
        tv_weight=0.01,             # FIXED: Re-enabled small TV to kill grid artifacts
        fb_weight=0.02,             # Keep low for fast motion
        multiscale=[1.0, 0.5],      # FIXED: Reverted to powers of 2 to avoid aliasing
        normalize_rewards=False,    # batch size 1 → keep False!
        debug=False
    ):
        super().__init__()

        self.device = torch.device(device)
        self.reward_type = reward_type
        self.alpha = alpha
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.tv_weight = tv_weight
        self.fb_weight = fb_weight
        self.multiscale = multiscale
        self.normalize_rewards = normalize_rewards
        self.debug = debug

        if optical_flow_models is None:
            raise ImportError("torchvision optical_flow not available.")

        # Load RAFT
        if hasattr(optical_flow_models, raft_model_name):
            self.flow_model = getattr(optical_flow_models, raft_model_name)(pretrained=True).to(self.device).eval()
        else:
            self.flow_model = optical_flow_models.raft_large(pretrained=True).to(self.device).eval()

        for p in self.flow_model.parameters():
            p.requires_grad = False

        self.eps = 1e-6

    # ------------------------------------- Utils ------------------------------------- #

    def _normalize_for_raft(self, x):
        return x.float() * 255.0

    def _resize(self, x, scale):
        if scale == 1.0:
            return x
        B, C, H, W = x.shape
        newH = max(2, int(round(H * scale)))
        newW = max(2, int(round(W * scale)))
        newH = ((newH + 7) // 8) * 8
        newW = ((newW + 7) // 8) * 8
        if newH == H and newW == W:
            return x
        return F.interpolate(x, size=(newH, newW), mode='bilinear', align_corners=False)

    def _resize_flow_to(self, flow, target_h, target_w):
        B, _, h, w = flow.shape
        if h == target_h and w == target_w:
            return flow
        flow_resized = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=False)
        flow_resized[:, 0] *= (target_w / w)
        flow_resized[:, 1] *= (target_h / h)
        return flow_resized

    def _make_grid(self, B, H, W, device):
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        return torch.stack((gx, gy), 0).unsqueeze(0).expand(B, -1, -1, -1)

    def _warp(self, img, flow):
        B, C, H, W = img.shape
        grid = self._make_grid(B, H, W, img.device)
        vgrid = grid + flow
        vgrid = torch.stack([
            2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0,
            2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
        ], dim=-1)
        return F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=False)

    # -------------------------------- Loss Components --------------------------------- #

    @torch.no_grad()
    def _estimate_flow(self, x1, x2):
        out = self.flow_model(x1, x2)
        return out[-1] if isinstance(out, (list, tuple)) else out

    def _photometric_multiscale(self, t, t1):
        losses, info = [], {}

        for scale in self.multiscale:
            ft = self._resize(t, scale)
            ft1 = self._resize(t1, scale)

            raft_in1 = self._normalize_for_raft(ft)
            raft_in2 = self._normalize_for_raft(ft1)

            flow = self._estimate_flow(raft_in1, raft_in2)

            if flow.shape[2:] != ft.shape[2:]:
                flow = self._resize_flow_to(flow, ft.shape[2], ft.shape[3])

            warped = self._warp(ft, flow)

            l1 = (warped - ft1).abs().mean(dim=[1, 2, 3])

            if self.use_ssim:
                ssim_loss = 1 - ssim_map(warped.clamp(0, 1), ft1.clamp(0, 1))
            else:
                ssim_loss = torch.zeros_like(l1)

            loss = self.l1_weight * l1 + self.ssim_weight * ssim_loss
            losses.append(loss)

        total = torch.stack(losses).mean(dim=0)
        info["photometric_loss"] = total.mean().item()

        return total, info

    def _fb_consistency(self, t, t1):
        raft1 = self._normalize_for_raft(t)
        raft2 = self._normalize_for_raft(t1)

        fwd = self._estimate_flow(raft1, raft2)
        bwd = self._estimate_flow(raft2, raft1)

        if fwd.shape[2:] != t.shape[2:]:
            fwd = self._resize_flow_to(fwd, t.shape[2], t.shape[3])
        if bwd.shape[2:] != t.shape[2:]:
            bwd = self._resize_flow_to(bwd, t.shape[2], t.shape[3])

        warped_bwd = self._warp(bwd, fwd)
        c = (fwd + warped_bwd).abs().mean(dim=[1, 2, 3])

        return c, {"fb_consistency": c.mean().item()}

    def _tv(self, t, t1):
        tv = (t1 - t).abs().mean(dim=[1, 2, 3])
        return tv

    # -------------------------------- Public API -------------------------------------- #

    @torch.no_grad()
    def compute_reward(self, t, t1, action_magnitude=None):
        t = t.to(self.device)
        t1 = t1.to(self.device)

        p_loss, pinfo = self._photometric_multiscale(t, t1)
        fb_loss, fbinfo = self._fb_consistency(t, t1)
        tv_loss = self._tv(t, t1)

        total = p_loss + self.fb_weight * fb_loss + self.tv_weight * tv_loss
        
        # Action-Aware Masking: Reduce penalty when large action is taken
        if action_magnitude is not None:
            if action_magnitude.device != self.device:
                action_magnitude = action_magnitude.to(self.device)
            
            # Scale loss: 1.0 (no action) -> 0.5 (max action)
            # This allows the model to make large changes when requested
            expected_motion_scale = 1.0 - 0.5 * action_magnitude
            total = total * expected_motion_scale

        if self.debug:
            print(f"[RTC] Photo={p_loss.item():.6f}, FB={fb_loss.item():.6f}, TV={tv_loss.item():.6f}, Total={total.item():.6f}")

        if self.reward_type == "reciprocal":
            reward = 1.0 / (1.0 + total)
        else:
            reward = torch.exp(-self.alpha * total)

        # -------- Auto-disable normalization for batch size = 1 --------
        if reward.shape[0] == 1:
            return reward, {
                **pinfo,
                **fbinfo,
                "tv": tv_loss.item(),
                "total_loss": total.item(),
                "reward": reward.item()
            }

        # (Only used if batch size > 1)
        if self.normalize_rewards:
            rmin = reward.min()
            rmax = reward.max()
            reward = (reward - rmin) / (rmax - rmin + self.eps)

        return reward, {
            **pinfo,
            **fbinfo,
            "tv": tv_loss.mean().item(),
            "total_loss": total.mean().item(),
            "reward": reward.mean().item()
        }

    @torch.no_grad()
    def compute_sequence_reward(self, frames, action_magnitude=None):
        B, T = frames.shape[:2]

        if T < 2:
            return torch.zeros(B, 0), {}

        t = frames[:, :-1].reshape(-1, *frames.shape[2:])
        t1 = frames[:, 1:].reshape(-1, *frames.shape[2:])
        
        # Reshape action magnitude if provided
        flat_action_magnitude = None
        if action_magnitude is not None:
            # action_magnitude is (B, T-1) -> (B*(T-1),)
            flat_action_magnitude = action_magnitude.reshape(-1)

        rewards, info = self.compute_reward(t, t1, action_magnitude=flat_action_magnitude)

        rewards = rewards.reshape(B, T - 1)

        return rewards, info


def load_temporal_consistency_reward_v2(device='cuda', **kwargs):
    return TemporalConsistencyRewardV2(device=device, **kwargs)
