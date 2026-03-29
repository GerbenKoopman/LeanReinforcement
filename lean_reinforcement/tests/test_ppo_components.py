"""Unit tests for PPO critics and losses."""

import unittest
from unittest.mock import patch

import torch

from lean_reinforcement.agent.ppo.constants import ENCODER_HIDDEN_DIM, NUM_BINS
from lean_reinforcement.agent.ppo.critics import EuclideanCritic, HyperbolicCritic
from lean_reinforcement.agent.ppo.losses import (
    compute_critic_loss,
    compute_gae,
    compute_ppo_actor_loss,
    returns_to_bin_targets,
)


class TestPPOCritics(unittest.TestCase):
    def test_euclidean_critic_outputs(self) -> None:
        critic = EuclideanCritic()
        x = torch.randn(6, ENCODER_HIDDEN_DIM)

        values, logits, probs = critic(x)

        self.assertEqual(values.shape, (6,))
        self.assertEqual(logits.shape, (6, NUM_BINS))
        self.assertEqual(probs.shape, (6, NUM_BINS))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(6), atol=1e-5))

    def test_hyperbolic_critic_outputs(self) -> None:
        critic = HyperbolicCritic()
        x = torch.randn(4, ENCODER_HIDDEN_DIM)

        values, logits, probs = critic(x)

        self.assertEqual(values.shape, (4,))
        self.assertEqual(logits.shape, (4, NUM_BINS))
        self.assertEqual(probs.shape, (4, NUM_BINS))
        self.assertFalse(torch.isnan(logits).any())
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))


class TestPPOLosses(unittest.TestCase):
    def test_returns_to_bin_targets_bounds(self) -> None:
        returns = torch.tensor([-0.4, 0.0, 0.53, 1.3])
        bins = returns_to_bin_targets(returns)

        self.assertEqual(int(bins.min().item()), 0)
        self.assertEqual(int(bins.max().item()), NUM_BINS - 1)

    def test_gae_matches_detached_difference(self) -> None:
        returns = torch.tensor([0.2, 0.8])
        values = torch.tensor([0.1, 0.5], requires_grad=True)
        adv = compute_gae(returns, values)

        expected = returns - values.detach()
        self.assertTrue(torch.allclose(adv, expected))

    def test_actor_and_critic_losses_are_finite(self) -> None:
        new_log_probs = torch.tensor([-2.1, -1.8, -0.4])
        old_log_probs = torch.tensor([-2.3, -1.6, -0.5])
        advantages = torch.tensor([0.2, -0.1, 0.5])

        actor_loss = compute_ppo_actor_loss(new_log_probs, old_log_probs, advantages)

        logits = torch.randn(3, NUM_BINS)
        targets = torch.tensor([0, NUM_BINS // 2, NUM_BINS - 1])
        critic_loss = compute_critic_loss(logits, targets)

        self.assertTrue(torch.isfinite(actor_loss))
        self.assertTrue(torch.isfinite(critic_loss))


class TestPPOAgentConfig(unittest.TestCase):
    @patch("lean_reinforcement.agent.ppo_agent.HyperbolicPPO")
    def test_hyperbolic_agent_passes_curvature(self, mock_hyperbolic_ppo) -> None:
        from lean_reinforcement.agent.ppo_agent import PPOAgent

        PPOAgent(model_name="dummy/model", use_hyperbolic=True, curvature=0.42)

        self.assertTrue(mock_hyperbolic_ppo.called)
        _, kwargs = mock_hyperbolic_ppo.call_args
        config = kwargs["config"]
        self.assertAlmostEqual(config.curvature, 0.42)

    @patch("lean_reinforcement.agent.ppo_agent.EuclideanPPO")
    def test_euclidean_agent_passes_latent_dim(self, mock_euclidean_ppo) -> None:
        from lean_reinforcement.agent.ppo_agent import PPOAgent

        PPOAgent(
            model_name="dummy/model",
            use_hyperbolic=False,
            value_head_latent_dim=256,
        )

        self.assertTrue(mock_euclidean_ppo.called)
        _, kwargs = mock_euclidean_ppo.call_args
        config = kwargs["config"]
        self.assertEqual(config.value_head_latent_dim, 256)


if __name__ == "__main__":
    unittest.main()
