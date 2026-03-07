"""Tests for the hyperbolic adapter and hyperbolic value head."""

import unittest
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from lean_reinforcement.agent.hyperbolic_adapter import (
    HyperbolicAdapter,
    HyperbolicValueHead,
    _HyperbolicHead,
    ENCODER_OUTPUT_DIM,
)
from lean_reinforcement.agent.transformer import Transformer


class TestHyperbolicAdapter(unittest.TestCase):
    """Unit tests for the standalone HyperbolicAdapter module."""

    def test_output_shape(self) -> None:
        adapter = HyperbolicAdapter(input_dim=ENCODER_OUTPUT_DIM, latent_dim=64)
        x = torch.randn(4, ENCODER_OUTPUT_DIM)
        out = adapter(x)
        self.assertEqual(out.shape, (4, 64))

    def test_output_inside_unit_ball(self) -> None:
        """All embeddings must have norm < 1 (Poincaré ball constraint)."""
        adapter = HyperbolicAdapter(input_dim=ENCODER_OUTPUT_DIM, latent_dim=64)
        x = torch.randn(8, ENCODER_OUTPUT_DIM)
        out = adapter(x)
        norms = torch.norm(out, p=2, dim=-1)
        self.assertTrue((norms < 1.0).all(), "Embeddings escaped the Poincaré ball")

    def test_gradients_flow(self) -> None:
        adapter = HyperbolicAdapter(input_dim=32, latent_dim=16)
        x = torch.randn(2, 32, requires_grad=True)
        out = adapter(x)
        loss = out.sum()
        loss.backward()

        linear_grad = adapter.linear.weight.grad
        xi_grad = adapter.xi.grad
        self.assertIsNotNone(linear_grad)
        self.assertIsNotNone(xi_grad)
        assert linear_grad is not None  # Type narrowing
        assert xi_grad is not None  # Type narrowing
        self.assertFalse(torch.isnan(linear_grad).any())
        self.assertFalse(torch.isnan(xi_grad).any())

    def test_exp_map_origin_zero_input(self) -> None:
        """exp_map at origin should handle near-zero vectors gracefully."""
        x = torch.zeros(2, 16)
        out = HyperbolicAdapter.exp_map_origin(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_custom_rho_max(self) -> None:
        adapter = HyperbolicAdapter(input_dim=32, latent_dim=16, rho_max=0.5)
        self.assertEqual(adapter.rho_max, 0.5)
        x = torch.randn(4, 32)
        out = adapter(x)
        norms = torch.norm(out, p=2, dim=-1)
        self.assertTrue((norms < 1.0).all())


class TestHyperbolicHead(unittest.TestCase):
    """Tests for the _HyperbolicHead wrapper."""

    def test_forward_shape(self) -> None:
        adapter = HyperbolicAdapter(input_dim=ENCODER_OUTPUT_DIM, latent_dim=64)
        linear = nn.Linear(64, 1)
        head = _HyperbolicHead(adapter, linear)

        x = torch.randn(4, ENCODER_OUTPUT_DIM)
        out = head(x)
        self.assertEqual(out.shape, (4, 1))

    def test_trainable_parameters(self) -> None:
        adapter = HyperbolicAdapter(input_dim=32, latent_dim=16)
        linear = nn.Linear(16, 1)
        head = _HyperbolicHead(adapter, linear)

        for param in head.parameters():
            self.assertTrue(param.requires_grad)


class TestHyperbolicValueHead(unittest.TestCase):
    """Integration tests for HyperbolicValueHead as a drop-in for ValueHead."""

    @patch("lean_reinforcement.agent.transformer.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("lean_reinforcement.agent.transformer.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained) -> None:
        self.mock_tokenizer = MagicMock()
        self.mock_transformer_model = MagicMock()
        self.mock_encoder = MagicMock()

        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_transformer_model

        self.mock_transformer_model.to.return_value = self.mock_transformer_model
        self.mock_transformer_model.get_encoder.return_value = self.mock_encoder
        self.mock_encoder.parameters.return_value = [nn.Parameter(torch.randn(2, 2))]

        self.mock_transformer = MagicMock(spec=Transformer)
        self.mock_transformer.tokenizer = self.mock_tokenizer
        self.mock_transformer.model = self.mock_transformer_model

        self.value_head = HyperbolicValueHead(self.mock_transformer)

    def test_encoder_is_frozen(self) -> None:
        for param in self.value_head.encoder.parameters():
            self.assertFalse(param.requires_grad)

    def test_value_head_attr_is_trainable(self) -> None:
        """The .value_head attribute must be a trainable nn.Module."""
        self.assertIsInstance(self.value_head.value_head, nn.Module)
        for param in self.value_head.value_head.parameters():
            self.assertTrue(param.requires_grad)

    def test_value_head_forward(self) -> None:
        """value_head.value_head(features) should return raw logits (B, 1)."""
        device = next(self.value_head.value_head.parameters()).device
        features = torch.randn(2, ENCODER_OUTPUT_DIM, device=device)
        out = self.value_head.value_head(features)
        self.assertEqual(out.shape, (2, 1))

    def test_predict_returns_tanh_range(self) -> None:
        """predict() should return values in [-1, 1]."""
        device = next(self.value_head.value_head.parameters()).device
        encoded_features = torch.randn(1, ENCODER_OUTPUT_DIM, device=device)
        with patch.object(
            self.value_head, "encode_states", return_value=encoded_features
        ):
            value = self.value_head.predict("⊢ True")
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_predict_batch(self) -> None:
        device = next(self.value_head.value_head.parameters()).device
        encoded_features = torch.randn(3, ENCODER_OUTPUT_DIM, device=device)
        with patch.object(
            self.value_head, "encode_states", return_value=encoded_features
        ):
            values = self.value_head.predict_batch(["a", "b", "c"])
            self.assertEqual(len(values), 3)
            for v in values:
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)

    def test_predict_from_features(self) -> None:
        device = next(self.value_head.value_head.parameters()).device
        features = torch.randn(1, ENCODER_OUTPUT_DIM, device=device)
        value = self.value_head.predict_from_features(features)
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_predict_from_features_batch(self) -> None:
        device = next(self.value_head.value_head.parameters()).device
        features = torch.randn(4, ENCODER_OUTPUT_DIM, device=device)
        values = self.value_head.predict_from_features_batch(features)
        self.assertEqual(len(values), 4)
        for v in values:
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_checkpoint_round_trip(self) -> None:
        """Save + load should preserve trainable weights."""
        import tempfile
        import os

        # Move to CPU for checkpoint test to avoid device issues with fresh instance
        self.value_head.cpu()
        # Give the mock tokenizer a real name_or_path for pickling
        self.mock_tokenizer.name_or_path = "test-model"

        with tempfile.TemporaryDirectory() as tmpdir:
            self.value_head.save_checkpoint(tmpdir, "test.pth")
            filepath = os.path.join(tmpdir, "test.pth")
            self.assertTrue(os.path.exists(filepath))

            ckpt = torch.load(filepath, map_location="cpu")
            self.assertIn("value_head_state_dict", ckpt)
            self.assertEqual(ckpt["type"], "hyperbolic")

            # Load into the same instance (avoids re-creating mocks)
            # Perturb weights first to confirm they're actually restored
            orig_state = {
                k: v.clone() for k, v in self.value_head.value_head.state_dict().items()
            }
            with torch.no_grad():
                for param in self.value_head.value_head.parameters():
                    param.add_(1.0)

            self.value_head.load_checkpoint(tmpdir, "test.pth")

            for k, v_orig in orig_state.items():
                v_loaded = self.value_head.value_head.state_dict()[k]
                self.assertTrue(torch.equal(v_orig, v_loaded))

    def test_trainer_compatible_optimizer(self) -> None:
        """Trainer creates an optimizer over value_head.value_head.parameters()."""
        optimizer = torch.optim.AdamW(self.value_head.value_head.parameters(), lr=1e-4)
        self.assertGreater(len(list(optimizer.param_groups[0]["params"])), 0)

    def test_trainer_compatible_training_loop(self) -> None:
        """Simulate one step of the trainer's training loop."""
        device = next(self.value_head.value_head.parameters()).device
        optimizer = torch.optim.AdamW(self.value_head.value_head.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()

        features = torch.randn(4, ENCODER_OUTPUT_DIM, device=device)
        targets = torch.randn(4, device=device).clamp(-0.99, 0.99)

        self.value_head.train()
        preds = self.value_head.value_head(features).squeeze()
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients flowed through the adapter
        adapter = self.value_head.value_head.adapter
        linear_grad = adapter.linear.weight.grad
        self.assertIsNotNone(linear_grad)
        assert linear_grad is not None  # Type narrowing
        self.assertFalse(torch.isnan(linear_grad).any())

    def test_state_dict_save_restore(self) -> None:
        """Trainer uses value_head.value_head.state_dict() for early stopping."""
        import copy

        state = copy.deepcopy(self.value_head.value_head.state_dict())
        self.assertIsInstance(state, dict)
        self.assertGreater(len(state), 0)

        # Modify weights
        with torch.no_grad():
            for param in self.value_head.value_head.parameters():
                param.add_(1.0)

        # Restore
        self.value_head.value_head.load_state_dict(state)

        # Verify restored
        for (k, v_orig), (_, v_restored) in zip(
            state.items(),
            self.value_head.value_head.state_dict().items(),
        ):
            self.assertTrue(torch.equal(v_orig, v_restored))


if __name__ == "__main__":
    unittest.main()
