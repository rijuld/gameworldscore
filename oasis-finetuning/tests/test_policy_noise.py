
import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.oasis_policy import OasisPolicy

class MockDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10) # Dummy
        self.max_frames = 32 # Required by OasisPolicy
        
    def forward(self, x, t, actions):
        # Return a deterministic output based on input
        # x shape: (B, T*C, H, W) or similar
        # We just return a tensor of shape (B, 1, C, H, W) matching target
        # For simplicity, let's just return x * 0.1
        # The actual output shape depends on the DiT implementation
        # Oasis DiT usually outputs velocity of same shape as input x (or target part)
        # x is concatenation of context and noisy_target
        # We assume x has shape (B, T+1, C, H, W) effectively
        # Let's just return a tensor of ones with correct shape
        B = x.shape[0]
        # We need to know the output shape expected by compute_log_prob
        # It expects v_pred of shape (B, 1, C, H, W) roughly
        # In compute_log_prob: v_pred = self.dit(x, t, actions)
        # v_pred[:, -1:] is used.
        # So we return something that has at least 1 frame in dim 1.
        # Let's return x itself, it should have enough dimensions.
        return x

class TestPolicyNoise(unittest.TestCase):
    def setUp(self):
        # Mock the VAE and DiT loading
        with patch('models.oasis_policy.OasisVAE') as mock_vae_cls, \
             patch('models.oasis_policy.DiT_models') as mock_dit_models:
            
            # Setup mock VAE instance
            mock_vae = MagicMock()
            mock_vae.patch_size = 2
            mock_vae.encode.return_value = torch.randn(1, 4, 16, 16)
            mock_vae_cls.return_value = mock_vae
            
            # Setup mock DiT
            mock_dit = MockDiT()
            mock_dit_models.__getitem__.return_value = lambda **kwargs: mock_dit
            
            # Initialize policy
            self.policy = OasisPolicy(
                oasis_ckpt="dummy",
                vae_ckpt="dummy",
                device="cpu",
                dtype=torch.float32
            )
            
            # Replace DiT with our mock
            self.policy.dit = mock_dit
            
    def test_compute_log_prob_determinism(self):
        B, T, C, H, W = 1, 2, 4, 16, 16
        latents = torch.randn(B, T, C, H, W)
        actions = torch.randn(B, T+1, 10)
        target = torch.randn(B, 1, C, H, W)
        
        # 1. Test with explicit noise (should be deterministic)
        noise = torch.randn_like(target)
        
        log_prob1 = self.policy.compute_log_prob(latents, actions, target, noise=noise)
        log_prob2 = self.policy.compute_log_prob(latents, actions, target, noise=noise)
        
        self.assertTrue(torch.allclose(log_prob1, log_prob2), "Log probs should be identical with same noise")
        
        # 2. Test with different noise (should be different)
        noise2 = torch.randn_like(target)
        log_prob3 = self.policy.compute_log_prob(latents, actions, target, noise=noise2)
        
        self.assertFalse(torch.allclose(log_prob1, log_prob3), "Log probs should differ with different noise")
        
        # 3. Test without explicit noise (should be random/different each time)
        # Note: In our implementation, if noise is None, it generates random noise
        log_prob4 = self.policy.compute_log_prob(latents, actions, target, noise=None)
        log_prob5 = self.policy.compute_log_prob(latents, actions, target, noise=None)
        
        # There is a tiny chance they are same, but extremely unlikely
        self.assertFalse(torch.allclose(log_prob4, log_prob5), "Log probs should differ when noise is None (random)")

if __name__ == '__main__':
    unittest.main()
