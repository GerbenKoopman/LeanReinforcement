#!/usr/bin/env python3
"""
Hyperbolic MCTS Value Head — v0 Feasibility Script
====================================================

End-to-end smoke test that verifies:

1.  A frozen ByT5-small encoder (mock ReProver backbone) produces
    mean-pooled Euclidean embeddings of dimension 1472.
2.  The trainable :class:`HyperbolicAdapter` projects those embeddings
    into the Poincaré ball (dimension 64).
3.  A lightweight linear value head maps the hyperbolic embedding to a
    scalar heuristic in [0, 1].
4.  Gradients flow through the adapter and value head only — the frozen
    encoder's parameters remain untouched.

Run
---
    mamba activate lean-reinforcement
    python hyperbolic_feasibility.py

Expected output: gradient norms for all trainable components, an
assertion that the encoder is truly frozen, and a short training loop
showing the MSE loss decreasing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
BACKBONE_NAME = "google/byt5-small"
ENCODER_OUTPUT_DIM = 1472  # ByT5-small hidden size
LATENT_DIM = 64
RHO_MAX = 0.95
XI_INIT = 0.01
BATCH_SIZE = 4
LR = 1e-3
N_STEPS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 1. Frozen ReProver backbone (mock)
# ---------------------------------------------------------------------------
class FrozenReProverEncoder(nn.Module):
    """
    Wraps a ByT5-small encoder with all parameters frozen.
    Simulates the ReProver backbone for this feasibility test.
    """

    def __init__(self, model_name: str = BACKBONE_NAME) -> None:
        super().__init__()
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.encoder = full_model.get_encoder()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode(self, text_list: List[str]) -> torch.Tensor:
        """
        Tokenize → frozen encoder → masked mean-pool → detach.

        Returns
        -------
        features : (batch, 1472)
            Euclidean mean-pooled hidden states.
        """
        tok = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2300,
        )
        tok = tok.to(DEVICE)

        with torch.no_grad():
            # (B, T, 1472)
            hidden = self.encoder(tok.input_ids).last_hidden_state

        mask = tok.attention_mask.unsqueeze(2)  # (B, T, 1)
        lengths = tok.attention_mask.sum(dim=1, keepdim=True)  # (B, 1)
        # (B, 1472)
        pooled: torch.Tensor = (hidden * mask).sum(dim=1) / lengths

        return pooled.detach()


# ---------------------------------------------------------------------------
# 2. HYPER++ Adapter  (trainable)
# ---------------------------------------------------------------------------
class HyperbolicAdapter(nn.Module):
    """
    Linear → RMSNorm → learned radius bound → Poincaré exp-map.
    """

    def __init__(
        self,
        input_dim: int = ENCODER_OUTPUT_DIM,
        latent_dim: int = LATENT_DIM,
        rho_max: float = RHO_MAX,
        xi_init: float = XI_INIT,
    ) -> None:
        super().__init__()
        self.rho_max = rho_max

        self.linear = nn.Linear(input_dim, latent_dim)
        self.rms_norm = nn.RMSNorm(latent_dim)
        self.xi = nn.Parameter(torch.tensor(xi_init))

    @staticmethod
    def exp_map_origin(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Exp_0(v) = tanh(||v||) · v / ||v||   (Poincaré ball)."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        result: torch.Tensor = torch.tanh(norm) * (x / (norm + eps))
        return result

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        x = self.linear(encoder_out)  # (B, D)
        x = self.rms_norm(x)  # (B, D)
        scale = self.rho_max * torch.sigmoid(self.xi)
        x = scale * x  # bounded Euclidean
        return self.exp_map_origin(x)  # → Poincaré ball


# ---------------------------------------------------------------------------
# 3. Value Head  (trainable)
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    """Linear(latent_dim, 1) → Sigmoid → scalar ∈ [0, 1]."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.sigmoid(self.linear(x_h)).squeeze(-1)  # (B,)
        return result


# ---------------------------------------------------------------------------
# 4. Feasibility training loop
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Device: {DEVICE}\n")

    # -- Mock Lean proof states ------------------------------------------
    dummy_states: List[str] = [
        "⊢ ∀ (n : ℕ), 0 ≤ n",
        "n : ℕ\nh : n > 0\n⊢ n ≠ 0",
        "α : Type\nl : List α\n⊢ l.length ≥ 0",
        "x y : ℝ\nhxy : x < y\n⊢ x ≤ y",
    ]
    # Dummy targets (geodesic potentials ∈ [0, 1])
    dummy_targets = torch.rand(BATCH_SIZE, device=DEVICE)

    # -- Instantiate components ------------------------------------------
    print("Loading frozen ByT5-small encoder (mock ReProver backbone)…")
    backbone = FrozenReProverEncoder(BACKBONE_NAME)
    backbone.encoder.to(DEVICE)

    adapter = HyperbolicAdapter().to(DEVICE)
    value_head = ValueHead().to(DEVICE)

    # -- Optimiser: adapter + value head only ----------------------------
    trainable_params = list(adapter.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    criterion = nn.MSELoss()

    # -- Encode once (frozen, so we can reuse) ---------------------------
    encoder_features = backbone.encode(dummy_states)  # (4, 1472)
    print(f"Encoder output shape : {encoder_features.shape}")
    print(f"Encoder output dtype : {encoder_features.dtype}")
    print(f"Encoder output device: {encoder_features.device}\n")

    assert encoder_features.shape == (
        BATCH_SIZE,
        ENCODER_OUTPUT_DIM,
    ), (
        f"Expected ({BATCH_SIZE}, {ENCODER_OUTPUT_DIM}), "
        f"got {encoder_features.shape}"
    )

    # -- Training loop ---------------------------------------------------
    header = (
        f"{'Step':>4}  {'Loss':>10}  {'‖W_lin‖':>10}  " f"{'‖ξ‖':>10}  {'‖W_val‖':>10}"
    )
    print(header)
    print("-" * 56)

    for step in range(1, N_STEPS + 1):
        adapter.train()
        value_head.train()
        optimizer.zero_grad()

        x_h = adapter(encoder_features)  # (4, 64)
        preds = value_head(x_h)  # (4,)
        loss = criterion(preds, dummy_targets)

        loss.backward()

        # Gradient norms
        grad_linear = adapter.linear.weight.grad
        grad_xi = adapter.xi.grad
        grad_value = value_head.linear.weight.grad

        assert grad_linear is not None, "adapter.linear gradient None!"
        assert grad_xi is not None, "adapter.xi gradient is None!"
        msg_vh = "value_head.linear gradient is None!"
        assert grad_value is not None, msg_vh

        norm_linear = grad_linear.norm().item()
        norm_xi = grad_xi.norm().item()
        norm_value = grad_value.norm().item()

        # Sanity: no NaN / Inf
        msg_nan_lin = "NaN in adapter.linear grad"
        assert not torch.isnan(grad_linear).any(), msg_nan_lin
        assert not torch.isnan(grad_xi).any(), "NaN in adapter.xi grad"
        msg_nan_vh = "NaN in value_head.linear grad"
        assert not torch.isnan(grad_value).any(), msg_nan_vh
        msg_inf = "Inf in adapter.linear grad"
        assert not torch.isinf(grad_linear).any(), msg_inf

        if step <= 5 or step % 5 == 0:
            output_line = (
                f"{step:4d}  {loss.item():10.6f}  {norm_linear:10.6f}  "
                f"{norm_xi:10.6f}  {norm_value:10.6f}"
            )
            print(output_line)

        optimizer.step()

    # -- Verify encoder is truly frozen ----------------------------------
    print("\n— Frozen encoder assertions —")
    frozen_count = 0
    for name, param in backbone.encoder.named_parameters():
        assert not param.requires_grad, f"Encoder param {name} is NOT frozen!"
        assert param.grad is None, f"Encoder param {name} has a gradient!"
        frozen_count += 1
    print(
        f"  ✓ All {frozen_count} encoder parameters are frozen "
        f"(requires_grad=False, grad=None)."
    )

    # -- Quick property checks on hyperbolic embeddings ------------------
    with torch.no_grad():
        adapter.eval()
        x_h = adapter(encoder_features)
        norms = torch.norm(x_h, p=2, dim=-1)
        print("\n— Poincaré ball norms —")
        print(f"  min  = {norms.min().item():.6f}")
        print(f"  max  = {norms.max().item():.6f}")
        print(f"  mean = {norms.mean().item():.6f}")
        escape_msg = "Hyperbolic embedding escaped the Poincaré ball!"
        assert (norms < 1.0).all(), escape_msg
        print("  ✓ All embeddings are strictly inside unit ball.")

    # -- Summary ---------------------------------------------------------
    total_trainable = sum(p.numel() for p in trainable_params)
    total_frozen = sum(p.numel() for p in backbone.encoder.parameters())
    print("\n— Parameter counts —")
    print(f"  Trainable (adapter + value) : {total_trainable:>10,}")
    frozen_label = "Frozen    (ByT5 encoder) "
    print(f"  {frozen_label}       : {total_frozen:>10,}")
    ratio = total_trainable / total_frozen
    print(f"  Ratio                      : {ratio:.6%}")

    success_msg = (
        "\n✓ Feasibility test PASSED — gradients flow, encoder "
        "frozen, embeddings in Poincaré ball."
    )
    print(success_msg)


if __name__ == "__main__":
    main()
