"""
TemporalConsistencyRewardV2

Fully rewritten temporal consistency reward module implementing:
- Multiscale L1 + SSIM photometric warping loss
- Forward-backward flow consistency
- Total Variation (TV) flicker detector
- Exponential reward scaling with normalization option
- Vectorized batched sequence processing (B, T, C, H, W)
- Robust handling of RAFT API/torchvision versions

Usage:
    model = TemporalConsistencyRewardV2(device='cuda', alpha=30.0)
    rewards, info = model.compute_sequence_reward(frames)

Returns:
    rewards: (B, T-1) reward per transition in [0,1]
    info:   dict with components and aggregated statistics

Notes:
- The class ships a lightweight SSIM implementation so it is self-contained.
- RAFT expects inputs in roughly [0,255]; this implementation scales inputs by 255
  before forwarding to the optical flow model. Adjust normalization if you use a
  different flow backbone.
- If torchvision's optical_flow models are not available, the class raises an
  informative error.

"""

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import RAFT optical flow from torchvision (newer versions) or fallback
try:
    import torchvision.models.optical_flow as optical_flow_models
except Exception:
    optical_flow_models = None


# ---------------------------- Lightweight SSIM ---------------------------- #
# Self-contained SSIM implementation adapted for clarity and stability.
# Returns average SSIM in [0, 1] across channels.

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device='cpu') -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, K=(0.01, 0.03), L: float = 1.0) -> torch.Tensor:
    # img1,img2: (B, C, H, W), values in [0,1]
    # returns mean SSIM per batch element
    B, C, H, W = img1.shape
    device = img1.device
    window = _gaussian_kernel(window_size, sigma=1.5, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)

    mu1 = F.conv2d(img1.view(B * C, 1, H, W), window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2.view(B * C, 1, H, W), window, padding=window_size // 2, groups=1)

    mu1 = mu1.view(B, C, H, W)
    mu2 = mu2.view(B, C, H, W)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d((img1 * img1).view(B * C, 1, H, W), window, padding=window_size // 2).view(B, C, H, W)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d((img2 * img2).view(B * C, 1, H, W), window, padding=window_size // 2).view(B, C, H, W)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d((img1 * img2).view(B * C, 1, H, W), window, padding=window_size // 2).view(B, C, H, W)
        - mu1_mu2
    )

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # spatial average
    ssim_per_channel = ssim_map.view(B, C, -1).mean(dim=-1)  # (B, C)
    # average across channels
    ssim_per_batch = ssim_per_channel.mean(dim=1)  # (B,)
    return ssim_per_batch


# ---------------------------- Core Module ---------------------------- #
class TemporalConsistencyRewardV2(nn.Module):
    def __init__(
        self,
        device: str = 'cuda',
        raft_model_name: str = 'raft_large',
        alpha: float = 30.0,  # exponent scale for reward: reward = exp(-alpha * loss)
        use_ssim: bool = True,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5,
        tv_weight: float = 0.02,
        fb_weight: float = 1.0,
        multiscale: List[float] = [1.0, 0.5, 0.25],
        normalize_rewards: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.alpha = float(alpha)
        self.use_ssim = use_ssim
        self.ssim_weight = float(ssim_weight)
        self.l1_weight = float(l1_weight)
        self.tv_weight = float(tv_weight)
        self.fb_weight = float(fb_weight)
        self.multiscale = multiscale
        self.normalize_rewards = normalize_rewards

        if optical_flow_models is None:
            raise ImportError(
                'torchvision.models.optical_flow not found. Please install a recent torchvision `pip install torchvision`.'
            )

        # Load RAFT (or chosen) optical flow model
        if raft_model_name == 'raft_large' and getattr(optical_flow_models, 'raft_large', None) is not None:
            self.flow_model = optical_flow_models.raft_large(pretrained=True).to(self.device).eval()
        elif raft_model_name == 'raft_small' and getattr(optical_flow_models, 'raft_small', None) is not None:
            self.flow_model = optical_flow_models.raft_small(pretrained=True).to(self.device).eval()
        else:
            # fallback: try to use any available attribute
            candidates = [n for n in dir(optical_flow_models) if n.startswith('raft')]
            if not candidates:
                raise RuntimeError('No RAFT-like model found in torchvision.models.optical_flow')
            chosen = candidates[0]
            self.flow_model = getattr(optical_flow_models, chosen)(pretrained=True).to(self.device).eval()

        for p in self.flow_model.parameters():
            p.requires_grad = False

        # Small epsilon used for numerical stability
        self.eps = 1e-6

    # ---------------------- utilities ----------------------
    def _normalize_for_raft(self, img: torch.Tensor) -> torch.Tensor:
        """Scale image to RAFT-friendly input. RAFT is typically trained on 0-255 images.
        We accept inputs in [0,1] and scale to [0,255]."""
        if img.dtype != torch.float32:
            img = img.float()
        return img * 255.0

    def _resize(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return x
        B, C, H, W = x.shape
        newH = max(2, int(round(H * scale)))
        newW = max(2, int(round(W * scale)))
        return F.interpolate(x, size=(newH, newW), mode='bilinear', align_corners=False)

    def _make_grid(self, B: int, H: int, W: int, device: torch.device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        grid = torch.stack((grid_x, grid_y), dim=0)  # (2, H, W)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
        return grid

    def _warp_frame(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image (B,C,H,W) by flow (B,2,H,W). Flow is (dx, dy) in pixels.
        Uses align_corners=False for grid_sample."""
        B, C, H, W = image.shape
        device = image.device

        grid = self._make_grid(B, H, W, device=device)
        vgrid = grid + flow

        # Normalize to [-1,1]
        vgrid_x = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = torch.stack((vgrid_x, vgrid_y), dim=-1)  # (B, H, W, 2)

        warped = F.grid_sample(image, vgrid, mode='bilinear', padding_mode='border', align_corners=False)
        return warped

    # ---------------------- loss components ----------------------
    @torch.no_grad()
    def _estimate_flow(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Estimate flow img1->img2 with RAFT. Returns (B,2,H,W). Handles variable RAFT outputs."""
        # RAFT accepts (B, C, H, W) floats; we already scaled to [0,255]
        # Some RAFT wrappers return a list of flows; take the last one.
        out = self.flow_model(img1, img2)
        if isinstance(out, (list, tuple)):
            flow = out[-1]
        else:
            flow = out
        # Ensure shape (B,2,H,W)
        return flow

    def _photometric_multiscale_loss(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute multiscale photometric loss (L1 + SSIM) between warped frame_t and frame_t1.
        Returns per-batch loss (B,) and diagnostic dictionary."""
        device = frame_t.device
        B, C, H, W = frame_t.shape

        total_loss_per_scale = []
        details = {
            'l1_per_scale': [],
            'ssim_per_scale': [],
            'warped_mse_per_scale': [],
        }

        for scale in self.multiscale:
            ft = self._resize(frame_t, scale)
            ft1 = self._resize(frame_t1, scale)
            # Estimate flow at this scale (we scale inputs for RAFT consistently but RAFT may expect original size)
            # Simpler: estimate flow at the scaled resolution by up/downscaling the original flow - but we will compute directly on scaled images.
            raft_in_1 = self._normalize_for_raft(ft)
            raft_in_2 = self._normalize_for_raft(ft1)
            flow = self._estimate_flow(raft_in_1, raft_in_2)  # (B,2,h,w)

            # If raft returns flow at original resolution, we should resize flow to current resolution - assume returned flow matches input resolution.
            warped = self._warp_frame(ft, flow)

            # L1
            l1_per_pixel = (warped - ft1).abs()
            l1 = l1_per_pixel.view(B, -1).mean(dim=-1)  # (B,)

            # SSIM
            if self.use_ssim:
                ssim_val = ssim_map(warped, ft1)  # (B,)
                ssim_loss = 1.0 - ssim_val
            else:
                ssim_loss = torch.zeros_like(l1)

            # Weighted combination
            scale_loss = self.l1_weight * l1 + self.ssim_weight * ssim_loss

            total_loss_per_scale.append(scale_loss)
            details['l1_per_scale'].append(l1.mean().item())
            details['ssim_per_scale'].append((1.0 - ssim_loss).mean().item())

        # Average across scales
        total_loss = torch.stack(total_loss_per_scale, dim=0).mean(dim=0)  # (B,)
        details['photometric_loss_mean'] = total_loss.mean().item()
        return total_loss, details

    def _forward_backward_consistency(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute forward-backward flow consistency metric (B,)
        Consistency: || fwd + warp(back, fwd) ||_1 averaged"""
        raft_in_1 = self._normalize_for_raft(frame_t)
        raft_in_2 = self._normalize_for_raft(frame_t1)
        flow_fwd = self._estimate_flow(raft_in_1, raft_in_2)  # (B,2,H,W)
        flow_bwd = self._estimate_flow(raft_in_2, raft_in_1)  # (B,2,H,W)

        # Warp backward flow into forward frame coordinate: W(flow_bwd, flow_fwd)
        warped_bwd = self._warp_frame(flow_bwd, flow_fwd)

        consistency_map = (flow_fwd + warped_bwd).abs()  # (B,2,H,W)
        consistency = consistency_map.view(consistency_map.shape[0], -1).mean(dim=-1)  # (B,)

        return consistency, {'fb_consistency_mean': consistency.mean().item()}

    def _temporal_tv(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        # L1 of pixel differences (flicker detector)
        diff = (frame_t1 - frame_t).abs()
        return diff.view(diff.shape[0], -1).mean(dim=-1)

    # ---------------------- public API ----------------------
    @torch.no_grad()
    def compute_reward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute reward for a single transition (B,)"""
        if frame_t.device != self.device:
            frame_t = frame_t.to(self.device)
            frame_t1 = frame_t1.to(self.device)

        photometric_loss, photometric_details = self._photometric_multiscale_loss(frame_t, frame_t1)
        fb_consistency, fb_details = self._forward_backward_consistency(frame_t, frame_t1)
        tv = self._temporal_tv(frame_t, frame_t1)

        # Combine
        total_loss = photometric_loss + self.fb_weight * fb_consistency + self.tv_weight * tv

        # Exponential scaling to produce reward in (0,1]
        reward = torch.exp(-self.alpha * total_loss)

        # Optional normalization (per-batch): map rewards to [0,1] using min/max for better contrast in same-batch comparisons
        if self.normalize_rewards:
            rmin = reward.min(dim=0, keepdim=True).values
            rmax = reward.max(dim=0, keepdim=True).values
            reward = (reward - rmin) / (rmax - rmin + self.eps)

        info = {
            'photometric_loss_mean': photometric_details['photometric_loss_mean'],
            'fb_consistency_mean': fb_details['fb_consistency_mean'],
            'tv_mean': tv.mean().item(),
            'total_loss_mean': total_loss.mean().item(),
            'reward_mean': reward.mean().item(),
        }
        return reward, info

    @torch.no_grad()
    def compute_sequence_reward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute rewards for a sequence of frames: frames (B, T, C, H, W)
        Returns rewards (B, T-1) and aggregated info.
        """
        if frames.device != self.device:
            frames = frames.to(self.device)

        B, T = frames.shape[:2]
        if T < 2:
            return torch.zeros(B, 0, device=self.device), {
                'photometric_loss_mean': 0.0,
                'fb_consistency_mean': 0.0,
                'tv_mean': 0.0,
                'total_loss_mean': 0.0,
                'reward_mean': 0.0,
            }

        # Batch all transitions
        frame_t = frames[:, :-1].contiguous()  # (B, T-1, C, H, W)
        frame_t1 = frames[:, 1:].contiguous()
        B_seq = B * (T - 1)
        frame_t_flat = frame_t.view(B_seq, *frame_t.shape[2:])
        frame_t1_flat = frame_t1.view(B_seq, *frame_t1.shape[2:])

        rewards_flat, info_flat = self.compute_reward(frame_t_flat, frame_t1_flat)

        rewards = rewards_flat.view(B, T - 1)

        # Aggregate info: values already mean per-batch; compute means across flat sequences
        info = {
            'photometric_loss_mean': info_flat['photometric_loss_mean'],
            'fb_consistency_mean': info_flat['fb_consistency_mean'],
            'tv_mean': info_flat['tv_mean'],
            'total_loss_mean': info_flat['total_loss_mean'],
            'reward_mean': rewards.mean().item(),
        }
        return rewards, info


def load_temporal_consistency_reward_v2(device: str = 'cuda', **kwargs) -> TemporalConsistencyRewardV2:
    return TemporalConsistencyRewardV2(device=device, **kwargs)
