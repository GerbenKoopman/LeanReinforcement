"""Tests for optimizer compatibility utilities."""

import torch

from lean_reinforcement.utilities.optimizer import unwrap_optimizer_params


class _MockManifoldParameter:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


def test_unwrap_optimizer_params_handles_plain_and_manifold_params() -> None:
    plain = torch.nn.Parameter(torch.randn(3))
    manifold = _MockManifoldParameter(torch.nn.Parameter(torch.randn(4)))
    frozen = torch.nn.Parameter(torch.randn(2), requires_grad=False)

    params = unwrap_optimizer_params([plain, manifold, frozen])

    assert len(params) == 2
    assert params[0] is plain
    assert params[1] is manifold.tensor
