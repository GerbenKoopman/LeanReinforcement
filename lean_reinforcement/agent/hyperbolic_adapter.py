"""
HYPER++ Hyperbolic Adapter for ReProver encoder embeddings.

Projects Euclidean encoder outputs into the Poincaré ball via a learned
linear projection, RMSNorm, bounded scaling, and the origin-centred
exponential map.  A lightweight value head on top produces a scalar
heuristic estimate suitable for MCTS node evaluation.

Reference architecture: HYPER++ (Desai et al., 2023).
"""

from __future__ import annotations

from typing import List, cast

import torch
import torch.nn as nn

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
    Poincaré ball following the HYPER++ recipe:

        x_E  = Linear(encoder_out)          [1472 → latent_dim]
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

        # Learned linear projection
        self.linear = nn.Linear(input_dim, latent_dim)

        # RMSNorm to center features before scaling
        self.rms_norm = nn.RMSNorm(latent_dim)

        # Learnable scalar controlling effective radius inside the ball.
        # Initialised to a small positive value so that
        #   rho_max * sigmoid(xi_init) ≈ rho_max * 0.5025 ≈ 0.477
        # which sits well inside the ball boundary.
        self.xi = nn.Parameter(torch.tensor(xi_init))

    # -- Poincaré exponential map at the origin --------------------------
    @staticmethod
    def exp_map_origin(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Origin-centred exponential map for the Poincaré ball model:

            Exp_0(v) = tanh(||v||) · v / ||v||

        Numerically stabilised with *eps* to avoid division by zero.
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        direction = x / (norm + eps)
        result: torch.Tensor = torch.tanh(norm) * direction
        return result

    # -- Forward ---------------------------------------------------------
    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        encoder_out : (batch, input_dim)
            Mean-pooled Euclidean encoder features.

        Returns
        -------
        x_H : (batch, latent_dim)
            Embeddings on the Poincaré ball.
        """
        # Linear projection into latent space
        x = self.linear(encoder_out)  # (B, latent_dim)

        # RMSNorm to stabilise feature magnitudes
        x = self.rms_norm(x)  # (B, latent_dim)

        # Learned radius bound: ρ_max · σ(ξ) ∈ (0, ρ_max)
        scale = self.rho_max * torch.sigmoid(self.xi)
        x = scale * x  # (B, latent_dim)

        # Project into the Poincaré ball
        x_h = self.exp_map_origin(x)  # (B, latent_dim)

        return x_h


# ---------------------------------------------------------------------------
# Hyperbolic Value Head (adapter + linear critic)
# ---------------------------------------------------------------------------
class HyperbolicValueHead(nn.Module):
    """
    Drop-in replacement for :class:`ValueHead` that routes the frozen
    ReProver encoder output through a :class:`HyperbolicAdapter` and a
    lightweight linear critic.

    Output range: [0, 1]  (sigmoid activation).
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

        # Freeze the pre-trained encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable adapter
        self.adapter = HyperbolicAdapter(
            input_dim=input_dim,
            latent_dim=latent_dim,
            rho_max=rho_max,
            xi_init=xi_init,
        )

        # Lightweight value head: one linear layer + sigmoid
        self.value_linear = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

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

        inp_ids = tokenized_s.input_ids
        hidden_state = self.encoder(inp_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        attn_mask = tokenized_s.attention_mask.unsqueeze(2)
        features = (hidden_state * attn_mask).sum(dim=1) / lens.unsqueeze(1)

        return cast(torch.Tensor, features.detach())

    # -- Full forward (encoder → adapter → critic) -----------------------
    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Forward from pre-computed encoder features to scalar values."""
        x_h = self.adapter(encoder_out)  # (B, latent_dim)
        value = self.value_linear(x_h)  # (B, 1)
        result: torch.Tensor = self.sigmoid(value).squeeze(-1)  # (B,)
        return result

    # -- Convenience predictions -----------------------------------------
    @torch.no_grad()
    def predict(self, state_str: str) -> float:
        """Predict value of single state. Returns ∈ [0, 1]."""
        self.eval()
        features = self.encode_states([state_str])
        result: float = self.forward(features).item()
        cnt = periodic_cache_cleanup(self._predict_call_count)
        self._predict_call_count = cnt
        return result

    @torch.no_grad()
    def predict_batch(self, state_strs: List[str]) -> List[float]:
        """Predict batch of states. Returns ∈ [0, 1]."""
        self.eval()
        features = self.encode_states(state_strs)
        results: List[float] = self.forward(features).tolist()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return results

    @torch.no_grad()
    def predict_from_features(self, features: torch.Tensor) -> float:
        """Predict from pre-computed features."""
        self.eval()
        result: float = self.forward(features).item()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return result

    @torch.no_grad()
    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        """Predict from pre-computed features (batch)."""
        self.eval()
        results: List[float] = self.forward(features).tolist()
        cnt = periodic_cache_cleanup(self._predict_call_count)
        self._predict_call_count = cnt
        return results

    # -- Checkpoint I/O (trainable weights only) -------------------------
    def save_checkpoint(self, folder: str, filename: str) -> None:
        import os

        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save(
            {
                "adapter": self.adapter.state_dict(),
                "value_linear": self.value_linear.state_dict(),
                "latent_dim": self.adapter.latent_dim,
                "rho_max": self.adapter.rho_max,
                "input_dim": self.adapter.input_dim,
            },
            filepath,
        )

    def load_checkpoint(self, folder: str, filename: str) -> None:
        import os

        filepath = os.path.join(folder, filename)
        ckpt = torch.load(filepath, map_location="cpu", weights_only=True)
        self.adapter.load_state_dict(ckpt["adapter"])
        self.value_linear.load_state_dict(ckpt["value_linear"])
        if torch.cuda.is_available():
            self.to("cuda")
