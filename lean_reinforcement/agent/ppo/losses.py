"""PPO helper losses shared by Euclidean and Hyperbolic variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lean_reinforcement.agent.ppo.constants import NUM_BINS, PPO_CLIP_EPS


def returns_to_bin_targets(
    returns: torch.Tensor,
    num_bins: int = NUM_BINS,
) -> torch.Tensor:
    """Map scalar returns in [0, 1] to nearest categorical value bin index."""
    clamped = returns.clamp(0.0, 1.0)
    return (clamped * (num_bins - 1)).round().long()


def compute_gae(
    returns: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Single-step advantage estimate used in current PPO training flow."""
    return returns - values.detach()


def compute_ppo_actor_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = PPO_CLIP_EPS,
) -> torch.Tensor:
    """Compute clipped PPO surrogate objective (returned as minimization loss)."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    return -surrogate.mean()


def compute_critic_loss(
    bin_logits: torch.Tensor,
    target_bins: torch.Tensor,
) -> torch.Tensor:
    """Categorical value loss shared by both critics for fair comparison."""
    return F.cross_entropy(bin_logits, target_bins)
