# GRPO Implementation Analysis

## Overview
This document analyzes whether `oasis_grpo_trainer.py` correctly implements the Group Relative Policy Optimization (GRPO) algorithm according to its definition.

## GRPO Algorithm Definition

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm that:
1. **Samples multiple responses per prompt** (group_size > 1)
2. **Computes rewards for each response**
3. **Groups responses by prompt index**
4. **Normalizes advantages within groups** (subtract group mean, divide by group std)
5. **Uses PPO-style clipped policy loss** (no value function needed)
6. **Optionally applies KL penalty** to prevent policy drift

## Implementation Analysis

### ✅ Correctly Implemented

1. **Group Generation** (`_generate_rollouts`):
   - ✅ Correctly repeats inputs by `group_size` to create groups
   - ✅ Creates proper group indices for grouping responses
   - ✅ Shape: `(B*G, ...)` where B=batch_size, G=group_size

2. **Advantage Computation** (`_compute_advantages`):
   - ✅ Uses `core_algos.compute_grpo_outcome_advantage` from RLVR-World
   - ✅ Correctly passes token-level rewards, response mask, and indices
   - ✅ Implements group-based normalization (mean subtraction + std division)

3. **Policy Loss** (`_grpo_update`):
   - ✅ Uses `core_algos.compute_policy_loss` with PPO clipping
   - ✅ Correctly computes log probabilities during training
   - ✅ Applies entropy regularization

4. **No Value Function**:
   - ✅ Correctly omits value function (GRPO doesn't need it)

### ❌ Issues Found

#### Issue 1: Missing KL Penalty in Rewards

**Problem**: When `use_kl_in_reward=True`, the KL penalty should be subtracted from rewards **before** computing advantages, but this is not implemented.

**Expected Behavior** (from RLVR-World's `apply_kl_penalty`):
```python
# Compute KL divergence
kld = kl_penalty(old_log_prob, ref_log_prob, kl_penalty='kl')
beta = kl_controller.value
# Subtract KL penalty from rewards
token_level_rewards = token_level_scores - beta * kld
```

**Current Implementation**:
- `ref_log_probs` are computed in `_generate_rollouts` but never used
- `_compute_rewards` only computes GameWorldScore rewards, no KL penalty applied
- KL controller is initialized but never updated

**Location**: 
- `_compute_rewards` method (lines 542-588)
- `train_step` method (lines 847-864)

**Impact**: 
- If `use_kl_in_reward=True`, the KL penalty is not being applied, which can lead to policy drift
- The KL controller is never updated, so adaptive KL control doesn't work

#### Issue 2: KL Controller Never Updated

**Problem**: The adaptive KL controller is initialized but never updated with current KL values.

**Expected Behavior**:
```python
current_kl = masked_mean(kld, mask=response_mask)
kl_controller.update(current_kl=current_kl, n_steps=batch_size)
```

**Current Implementation**:
- KL controller initialized in `_init_kl_controller` (line 357)
- Never updated anywhere in the code

**Impact**: 
- Adaptive KL control doesn't work
- KL coefficient remains fixed at initial value

#### Issue 3: ref_log_probs Computed But Unused

**Problem**: `ref_log_probs` are computed during rollout generation but never used for KL penalty computation.

**Current Implementation**:
- `ref_log_probs` computed in `_generate_rollouts` (lines 464-514)
- Returned in rollout_data but never used in `train_step`

**Impact**: 
- Wasted computation (especially when `kl_compute_freq > 1`)
- KL penalty cannot be applied even if desired

## Recommendations

### Fix 1: Apply KL Penalty to Rewards

Modify `_compute_rewards` or create a new method to apply KL penalty:

```python
def _apply_kl_penalty_to_rewards(
    self,
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply KL penalty to rewards before advantage computation."""
    if not self.config.use_kl_in_reward or ref_log_probs is None:
        return rewards
    
    # Compute KL divergence
    kld = core_algos.kl_penalty(
        logprob=old_log_probs,
        ref_logprob=ref_log_probs,
        kl_penalty='kl'
    )
    kld = kld * response_mask
    
    # Get current KL coefficient
    beta = self.kl_controller.value
    
    # Apply KL penalty: rewards = scores - beta * kl
    token_level_rewards = rewards - beta * kld
    
    # Update KL controller
    current_kl = verl_F.masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    batch_size = rewards.shape[0]
    self.kl_controller.update(current_kl=current_kl, n_steps=batch_size)
    
    return token_level_rewards
```

Then call this in `train_step` before computing advantages:

```python
# Apply KL penalty to rewards if enabled
if self.config.use_kl_in_reward and rollout_data['ref_log_probs'] is not None:
    rewards = self._apply_kl_penalty_to_rewards(
        rewards=rewards,
        old_log_probs=rollout_data['log_probs'],
        ref_log_probs=rollout_data['ref_log_probs'],
        response_mask=torch.ones_like(rewards),  # or proper mask
    )
```

### Fix 2: Always Compute ref_log_probs When Needed

If `use_kl_in_reward=True`, always compute `ref_log_probs` (don't skip based on `kl_compute_freq`):

```python
# In _generate_rollouts, change:
compute_kl_this_step = (self.global_step % self.config.kl_compute_freq == 0)

# To:
compute_kl_this_step = True  # Always compute if use_kl_in_reward
```

Or better: separate `kl_compute_freq` for reward vs. logging purposes.

## Summary

**Core GRPO Algorithm**: ✅ **Correctly Implemented**
- Group-based advantage normalization: ✅
- PPO-style policy loss: ✅
- No value function: ✅

**KL Penalty Feature**: ❌ **Not Fully Implemented**
- KL penalty in rewards: ❌ Missing
- KL controller update: ❌ Missing
- ref_log_probs usage: ❌ Computed but unused

**Conclusion**: The core GRPO algorithm is correctly implemented, but the KL penalty feature (when `use_kl_in_reward=True`) is incomplete. The implementation will work correctly for GRPO without KL penalty, but if KL penalty is desired, it needs to be fixed as described above.


