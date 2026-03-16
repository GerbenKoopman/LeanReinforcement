"""Standard Euclidean value head."""

from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.value_head.base import BaseValueHead
from lean_reinforcement.agent.value_head.constants import ENCODER_OUTPUT_DIM


class _EuclideanRegressor(nn.Module):
    """Simple MLP: input_dim -> latent_dim -> 1 with ReLU."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.in_linear = nn.Linear(input_dim, latent_dim)
        self.activation = nn.ReLU()
        self.out_linear = nn.Linear(latent_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.in_linear(features)
        hidden = self.activation(hidden)
        hidden = self.out_linear(hidden)

        return cast(torch.Tensor, hidden)


class ValueHead(BaseValueHead):
    """Euclidean value head with latent projection on frozen encoder features."""

    def __init__(
        self,
        transformer: Transformer,
        latent_dim: int = 1024,
        input_dim: int = ENCODER_OUTPUT_DIM,
    ) -> None:
        super().__init__(transformer)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.value_head = self._build_value_head(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
        )

        if torch.cuda.is_available():
            self.to("cuda")

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }

    def _build_value_head(self, input_dim: int, latent_dim: int) -> nn.Module:
        return _EuclideanRegressor(input_dim=input_dim, latent_dim=latent_dim)
