"""Tests for the streamlined hyperbolic value head."""

import unittest
from unittest.mock import patch, MagicMock
from typing import Any, cast

import torch
import torch.nn as nn
from hypll.tensors import TangentTensor

from lean_reinforcement.agent.value_head import HyperbolicValueHead, ENCODER_OUTPUT_DIM
from lean_reinforcement.agent.transformer import Transformer


class TestInlineHyperbolicRegressor(unittest.TestCase):
    """Unit tests for the inline hyperbolic regressor inside HyperbolicValueHead."""

    @patch("lean_reinforcement.agent.transformer.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("lean_reinforcement.agent.transformer.AutoTokenizer.from_pretrained")
    def test_projection_stays_inside_unit_ball(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ) -> None:
        mock_tokenizer = MagicMock()
        mock_transformer_model = MagicMock()
        mock_encoder = MagicMock()

        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_transformer_model
        mock_transformer_model.to.return_value = mock_transformer_model
        mock_transformer_model.get_encoder.return_value = mock_encoder
        mock_encoder.parameters.return_value = [nn.Parameter(torch.randn(2, 2))]

        mock_transformer = MagicMock(spec=Transformer)
        mock_transformer.tokenizer = mock_tokenizer
        mock_transformer.model = mock_transformer_model

        value_head = HyperbolicValueHead(mock_transformer, latent_dim=16)
        reg: Any = value_head.value_head

        rho_max = cast(torch.Tensor, reg.rho_max)
        xi = cast(torch.Tensor, reg.xi)
        x = torch.randn(8, ENCODER_OUTPUT_DIM, device=rho_max.device)
        projected = rho_max * torch.sigmoid(xi) * x
        tangent = TangentTensor(data=projected, man_dim=1, manifold=reg.manifold)
        x_h = reg.manifold.expmap(tangent)

        norms = torch.norm(x_h.tensor, p=2, dim=-1)
        self.assertTrue((norms < 1.0).all(), "Embeddings escaped the Poincare ball")


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

    @staticmethod
    def _as_plain_tensor(value: Any) -> torch.Tensor:
        if hasattr(value, "tensor"):
            return cast(torch.Tensor, value.tensor)
        return cast(torch.Tensor, value)

    @staticmethod
    def _add_in_place(param: Any, amount: float) -> None:
        if hasattr(param, "tensor"):
            cast(torch.Tensor, param.tensor).add_(amount)
        else:
            cast(torch.Tensor, param).add_(amount)

    def test_value_head_attr_is_trainable(self) -> None:
        """The .value_head attribute must be a trainable nn.Module."""
        self.assertIsInstance(self.value_head.value_head, nn.Module)
        trainable_count = sum(
            1
            for param in self.value_head.value_head.parameters()
            if param.requires_grad
        )
        self.assertGreater(trainable_count, 0)

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

            ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
            self.assertIn("value_head_state_dict", ckpt)
            self.assertEqual(ckpt["type"], "hyperbolic")

            # Load into the same instance (avoids re-creating mocks)
            # Perturb weights first to confirm they're actually restored
            orig_state = {
                k: self._as_plain_tensor(v).clone()
                for k, v in self.value_head.value_head.state_dict().items()
            }
            with torch.no_grad():
                for param in self.value_head.value_head.parameters():
                    self._add_in_place(param, 1.0)

            self.value_head.load_checkpoint(tmpdir, "test.pth")

            for k, v_orig in orig_state.items():
                v_loaded = self.value_head.value_head.state_dict()[k]
                self.assertTrue(
                    torch.equal(
                        self._as_plain_tensor(v_orig),
                        self._as_plain_tensor(v_loaded),
                    )
                )

    def test_value_head_has_parameters(self) -> None:
        """The value head exposes parameters for optimization logic."""
        params = list(self.value_head.value_head.parameters())
        self.assertGreater(len(params), 0)

    def test_trainer_compatible_training_loop(self) -> None:
        """Simulate one training step up to backward pass."""
        device = next(self.value_head.value_head.parameters()).device
        loss_fn = torch.nn.MSELoss()

        features = torch.randn(4, ENCODER_OUTPUT_DIM, device=device)
        targets = torch.randn(4, device=device).clamp(-0.99, 0.99)

        self.value_head.train()
        preds = self.value_head.value_head(features).squeeze()
        loss = loss_fn(preds, targets)

        loss.backward()

        # Verify gradients flowed through the inline hyperbolic projection.
        reg = self.value_head.value_head
        xi_grad = cast(torch.Tensor, reg.xi.grad)
        self.assertIsNotNone(xi_grad)
        self.assertFalse(torch.isnan(xi_grad).any())

    def test_state_dict_save_restore(self) -> None:
        """Trainer uses value_head.value_head.state_dict() for early stopping."""
        import copy

        state = copy.deepcopy(self.value_head.value_head.state_dict())
        self.assertIsInstance(state, dict)
        self.assertGreater(len(state), 0)

        # Modify weights
        with torch.no_grad():
            for param in self.value_head.value_head.parameters():
                self._add_in_place(param, 1.0)

        # Restore
        self.value_head.value_head.load_state_dict(state)

        # Verify restored
        for (k, v_orig), (_, v_restored) in zip(
            state.items(),
            self.value_head.value_head.state_dict().items(),
        ):
            self.assertTrue(
                torch.equal(
                    self._as_plain_tensor(v_orig),
                    self._as_plain_tensor(v_restored),
                )
            )


if __name__ == "__main__":
    unittest.main()
