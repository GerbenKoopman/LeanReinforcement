"""Optimizer compatibility helpers."""

from __future__ import annotations

from typing import Any, Iterable, List

import torch


def unwrap_optimizer_params(params: Iterable[Any]) -> List[torch.Tensor]:
    """Return optimizer-safe tensors, unwrapping manifold-backed parameters.

    Some hyperbolic libraries expose trainable objects that proxy tensor ops and
    raise on generic torch function calls (e.g. ``torch.is_complex``). For
    optimizers, we pass their underlying tensor via ``.tensor`` when available.
    """

    optimizer_params: List[torch.Tensor] = []
    for param in params:
        raw = getattr(param, "tensor", param)
        if isinstance(raw, torch.Tensor) and raw.requires_grad:
            optimizer_params.append(raw)
    return optimizer_params
