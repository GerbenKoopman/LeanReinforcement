"""
Hyperbolic Adapter for ReProver encoder embeddings.

Provides :class:`HyperbolicAdapter` (Euclidean → Poincaré ball projection)
and :class:`HyperbolicValueHead`, a drop-in replacement for
:class:`~lean_reinforcement.agent.value_head.ValueHead` that uses
hyperbolic geometry for the value estimate.

Enable via ``--use-hyperbolic`` (disabled by default).
"""

from __future__ import annotations

import os
from typing import List, cast

import torch
import torch.nn as nn
from typing_extensions import Self
from loguru import logger

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.utilities.memory import periodic_cache_cleanup

# ByT5-small encoder hidden size (concatenation of forward + backward = 1472)
ENCODER_OUTPUT_DIM = 1472


# ---------------------------------------------------------------------------
# Hyperbolic Adapter
# ---------------------------------------------------------------------------
class HyperbolicAdapter(nn.Module):
    """
    Trainable adapter that maps a Euclidean feature vector into the
    Poincaré ball:

        x_E  = Linear(encoder_out)          [input_dim → latent_dim]
        x_E  = RMSNorm(x_E)
        x_E  = ρ_max · σ(ξ) · x_E          (learned radius bound)
        x_H  = Exp_0(x_E)                   (origin-centred exp-map)
    """

    def __init__(
        self,
        input_dim: int = ENCODER_OUTPUT_DIM,
        latent_dim: int = 64,
        rho_max: float = 0.95,
        xi_init: float = 0.01,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rho_max = rho_max

        self.linear = nn.Linear(input_dim, latent_dim)
        self.rms_norm = nn.RMSNorm(latent_dim)

        # Learnable scalar controlling effective radius inside the ball.
        # rho_max * sigmoid(xi_init) ≈ rho_max * 0.5025 ≈ 0.477
        self.xi = nn.Parameter(torch.tensor(xi_init))

    @staticmethod
    def exp_map_origin(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Origin-centred exponential map: Exp_0(v) = tanh(‖v‖) · v / ‖v‖."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        direction = x / (norm + eps)
        result: torch.Tensor = torch.tanh(norm) * direction
        return result

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Map ``(B, input_dim)`` Euclidean features to ``(B, latent_dim)`` on the Poincaré ball."""
        x = self.linear(encoder_out)
        x = self.rms_norm(x)
        x = self.rho_max * torch.sigmoid(self.xi) * x
        return self.exp_map_origin(x)


# ---------------------------------------------------------------------------
# Internal wrapper so the trainer can uniformly access `value_head.value_head`
# ---------------------------------------------------------------------------
class _HyperbolicHead(nn.Module):
    """Adapter → Linear, matching the ``nn.Sequential`` interface of ``ValueHead.value_head``."""

    def __init__(self, adapter: HyperbolicAdapter, linear: nn.Linear) -> None:
        super().__init__()
        self.adapter = adapter
        self.linear = linear

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        x_h = self.adapter(encoder_out)  # (B, latent_dim)
        result: torch.Tensor = self.linear(x_h)  # (B, 1)  — raw logit, no activation
        return result


# ---------------------------------------------------------------------------
# Hyperbolic Value Head
# ---------------------------------------------------------------------------
class HyperbolicValueHead(nn.Module):
    """
    Drop-in replacement for :class:`ValueHead` that routes frozen
    ReProver encoder output through a :class:`HyperbolicAdapter` and a
    lightweight linear critic.

    The public interface is identical to ``ValueHead``:

    * ``value_head.value_head``  — the trainable ``nn.Module`` (for the
      optimizer and the trainer's training loop).
    * ``encode_states`` / ``predict`` / ``predict_batch`` etc.
    * ``save_checkpoint`` / ``load_checkpoint``

    Output range: **[-1, 1]** (tanh applied at prediction time), consistent
    with ``ValueHead``.
    """

    def __init__(
        self,
        transformer: Transformer,
        latent_dim: int = 64,
        rho_max: float = 0.95,
        xi_init: float = 0.01,
        input_dim: int = ENCODER_OUTPUT_DIM,
    ) -> None:
        super().__init__()

        # Borrow encoder + tokenizer from the shared Transformer
        self.tokenizer = transformer.tokenizer
        self.encoder = transformer.model.get_encoder()

        # Store dimensions for serialization
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rho_max = rho_max

        # Freeze the pre-trained encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable head: adapter + linear — exposed as ``self.value_head``
        # so the trainer can do ``value_head.value_head.parameters()`` etc.
        adapter = HyperbolicAdapter(
            input_dim=input_dim,
            latent_dim=latent_dim,
            rho_max=rho_max,
            xi_init=xi_init,
        )
        value_linear = nn.Linear(latent_dim, 1)
        self.value_head = _HyperbolicHead(adapter, value_linear)

        if torch.cuda.is_available():
            self.to("cuda")

        self._predict_call_count = 0

    # -- Encoder (frozen) ------------------------------------------------
    def encode_states(self, s: List[str]) -> torch.Tensor:
        """Encode proof-state strings into mean-pooled Euclidean features."""
        tokenized_s = self.tokenizer(
            s,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2300,
        )
        if torch.cuda.is_available():
            tokenized_s = tokenized_s.to("cuda")

        hidden_state = self.encoder(tokenized_s.input_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        attn_mask = tokenized_s.attention_mask.unsqueeze(2)
        features = (hidden_state * attn_mask).sum(dim=1) / lens.unsqueeze(1)

        return cast(torch.Tensor, features.detach())

    # -- Predictions (tanh → [-1, 1], same as ValueHead) -----------------
    @torch.no_grad()
    def predict(self, state_str: str) -> float:
        """Predict value of single state. Returns ∈ [-1, 1]."""
        self.eval()
        features = self.encode_states([state_str])
        value = self.value_head(features).squeeze()
        result: float = torch.tanh(value).item()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return result

    @torch.no_grad()
    def predict_batch(self, state_strs: List[str]) -> List[float]:
        """Predict batch of states. Returns ∈ [-1, 1]."""
        self.eval()
        features = self.encode_states(state_strs)
        values = self.value_head(features).squeeze()
        if values.ndim == 0:
            values = values.unsqueeze(0)
        results: List[float] = torch.tanh(values).tolist()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return results

    @torch.no_grad()
    def predict_from_features(self, features: torch.Tensor) -> float:
        """Predict from pre-computed encoder features. Returns ∈ [-1, 1]."""
        self.eval()
        value = self.value_head(features).squeeze()
        result: float = torch.tanh(value).item()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return result

    @torch.no_grad()
    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        """Predict from pre-computed encoder features (batch). Returns ∈ [-1, 1]."""
        self.eval()
        values = self.value_head(features).squeeze()
        if values.ndim == 0:
            values = values.unsqueeze(0)
        results: List[float] = torch.tanh(values).tolist()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return results

    # -- Checkpoint I/O (trainable weights only) -------------------------
    def save_checkpoint(self, folder: str, filename: str) -> None:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save(
            {
                "value_head_state_dict": self.value_head.state_dict(),
                "transformer_name": self.tokenizer.name_or_path,
                "latent_dim": self.latent_dim,
                "rho_max": self.rho_max,
                "input_dim": self.input_dim,
                "type": "hyperbolic",
            },
            filepath,
        )
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, folder: str, filename: str) -> None:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")
        ckpt = torch.load(
            filepath, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.value_head.load_state_dict(ckpt["value_head_state_dict"])
        logger.info(f"Checkpoint loaded from {filepath}")

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        return self
