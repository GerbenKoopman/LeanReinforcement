"""Euclidean and hyperbolic categorical critics for PPO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypll import nn as hnn
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor

from lean_reinforcement.agent.ppo.constants import (
    ENCODER_OUTPUT_DIM,
    LATENT_DIM,
    NUM_BINS,
    RHO_MAX,
    XI_INIT,
)


class BaseCategoricalCritic(nn.Module, ABC):
    """Common critic scaffold to keep Euclidean and hyperbolic variants aligned."""

    def __init__(
        self,
        input_dim: int = ENCODER_OUTPUT_DIM,
        latent_dim: int = LATENT_DIM,
        num_bins: int = NUM_BINS,
        rho_max: float = RHO_MAX,
        xi_init: float = XI_INIT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_bins = num_bins

        self.linear = nn.Linear(input_dim, latent_dim)
        self.rms_norm = nn.RMSNorm(latent_dim)
        self.xi = nn.Parameter(torch.tensor(xi_init))
        self.register_buffer("rho_max", torch.tensor(rho_max))
        self.register_buffer("support", torch.linspace(0.0, 1.0, num_bins))

    def _adapter(self, encoder_out: torch.Tensor) -> torch.Tensor:
        rho_max = cast(torch.Tensor, self.rho_max)
        x = self.linear(encoder_out)
        x = self.rms_norm(x)
        x = rho_max * torch.sigmoid(self.xi) * x
        return cast(torch.Tensor, x)

    @abstractmethod
    def _bin_logits(self, latent_features: torch.Tensor) -> torch.Tensor:
        pass

    def forward(
        self, encoder_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._adapter(encoder_out)
        bin_logits = self._bin_logits(latent)
        bin_probs = F.softmax(bin_logits, dim=-1)

        support = cast(torch.Tensor, self.support)
        value = (bin_probs * support).sum(dim=-1)
        return value, bin_logits, bin_probs


class EuclideanCritic(BaseCategoricalCritic):
    """Euclidean baseline critic with categorical value support."""

    def __init__(
        self,
        input_dim: int = ENCODER_OUTPUT_DIM,
        latent_dim: int = LATENT_DIM,
        num_bins: int = NUM_BINS,
        rho_max: float = RHO_MAX,
        xi_init: float = XI_INIT,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_bins=num_bins,
            rho_max=rho_max,
            xi_init=xi_init,
        )
        self.hidden = nn.Linear(latent_dim, latent_dim)
        self.activation = nn.ReLU()
        self.out_linear = nn.Linear(latent_dim, num_bins)

    def _bin_logits(self, latent_features: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(latent_features)
        hidden = self.activation(hidden)
        return cast(torch.Tensor, self.out_linear(hidden))


class HyperbolicCritic(BaseCategoricalCritic):
    """Poincare-ball critic using manifold operations for value prediction."""

    def __init__(
        self,
        input_dim: int = ENCODER_OUTPUT_DIM,
        latent_dim: int = LATENT_DIM,
        num_bins: int = NUM_BINS,
        rho_max: float = RHO_MAX,
        xi_init: float = XI_INIT,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_bins=num_bins,
            rho_max=rho_max,
            xi_init=xi_init,
        )
        self.manifold = PoincareBall(Curvature(1.0))
        self.hidden = hnn.HLinear(latent_dim, latent_dim, self.manifold)
        self.activation = hnn.HReLU(self.manifold)
        self.out_linear = hnn.HLinear(latent_dim, num_bins, self.manifold)

    def _bin_logits(self, latent_features: torch.Tensor) -> torch.Tensor:
        tangent = TangentTensor(data=latent_features, man_dim=1, manifold=self.manifold)
        hidden = self.manifold.expmap(tangent)

        hidden = self.hidden(hidden)
        hidden = self.activation(hidden)
        hidden = self.out_linear(hidden)

        logits = self.manifold.logmap(x=None, y=hidden).tensor
        return cast(torch.Tensor, logits)
