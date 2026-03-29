"""Hyperbolic Poincare-ball value head."""

from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from hypll import nn as hnn
from hypll.manifolds.poincare_ball import (
    Curvature,
    PoincareBall,
)
from hypll.tensors import TangentTensor

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.value_head.base import BaseValueHead
from lean_reinforcement.agent.value_head.constants import ENCODER_OUTPUT_DIM


class _HyperbolicRegressor(nn.Module):
    """Hyperbolic projection head: input_dim -> latent_dim -> 1."""

    def __init__(
        self,
        manifold: PoincareBall,
        input_dim: int,
        latent_dim: int,
        rho_max: float,
        xi_init: float,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.xi = nn.Parameter(torch.tensor(xi_init))

        self.in_linear = hnn.HLinear(input_dim, latent_dim, self.manifold)
        self.activation = hnn.HReLU(self.manifold)
        self.out_linear = hnn.HLinear(latent_dim, 1, self.manifold)

        self.register_buffer("rho_max", torch.tensor(rho_max))

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        rho_max = cast(torch.Tensor, self.rho_max)
        # Map Euclidean encoder features to a bounded tangent vector by
        # normalizing direction and controlling magnitude explicitly.
        # Per-coordinate scaling alone is not enough in high dimensions,
        # because vector norms can still grow large and push expmap outputs
        # onto/near the Poincare boundary.
        max_tangent_norm = rho_max * torch.sigmoid(self.xi)
        direction = encoder_out / encoder_out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x = max_tangent_norm * direction

        tangent = TangentTensor(data=x, man_dim=1, manifold=self.manifold)
        hidden = self.manifold.expmap(tangent)

        hidden = self.in_linear(hidden)
        hidden = self.activation(hidden)
        hidden = self.out_linear(hidden)

        tangent = self.manifold.logmap(x=None, y=hidden)

        return cast(torch.Tensor, tangent.tensor)


class HyperbolicValueHead(BaseValueHead):
    """Poincare-ball value head on top of frozen encoder features."""

    def __init__(
        self,
        transformer: Transformer,
        latent_dim: int = 64,
        rho_max: float = 0.95,
        xi_init: float = 0.01,
        input_dim: int = ENCODER_OUTPUT_DIM,
        curvature: float = 1.0,
    ) -> None:
        super().__init__(transformer)

        self.manifold = PoincareBall(Curvature(curvature))

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rho_max = rho_max

        self.value_head = self._build_value_head(
            input_dim=input_dim,
            latent_dim=latent_dim,
            rho_max=rho_max,
            xi_init=xi_init,
        )

        if torch.cuda.is_available():
            self.to("cuda")

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "rho_max": self.rho_max,
            "input_dim": self.input_dim,
            "type": "hyperbolic",
        }

    def _build_value_head(
        self,
        input_dim: int,
        latent_dim: int,
        rho_max: float,
        xi_init: float,
    ) -> nn.Module:
        return _HyperbolicRegressor(
            manifold=self.manifold,
            input_dim=input_dim,
            latent_dim=latent_dim,
            rho_max=rho_max,
            xi_init=xi_init,
        )
