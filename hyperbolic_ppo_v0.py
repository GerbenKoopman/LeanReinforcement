"""
Hyperbolic PPO v0 Feasibility Script
=====================================

Standalone benchmark for a Hyperbolic PPO update step on top of a frozen
ByT5-small backbone.  The design follows the HYPER++ protocol:

    * **Actor** – Frozen ByT5 with LoRA adapters injected into the decoder's
        self-attention and cross-attention layers (via HuggingFace ``peft``).
    * **Critic** – HYPER++ adapter (Linear → RMSNorm → Learned Scaling)
        projecting frozen encoder features onto the **Poincare ball** manifold,
        followed by a 51-bin Categorical Value Head.
  * **Loss** – PPO clipped surrogate for the actor; cross-entropy
    categorical loss for the critic (no MSE).

Run::

    python hyperbolic_ppo_v0.py
"""

from __future__ import annotations

import math
from typing import Any, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from hypll.manifolds.poincare_ball import (
    Curvature,
    PoincareBall,
)
from hypll.tensors import TangentTensor
from hypll import nn as hnn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENCODER_HIDDEN_DIM = 1472  # ByT5-small encoder output (fwd + bwd)
LATENT_DIM = 64  # Hyperbolic embedding size
NUM_BINS = 51  # Categorical value support
RHO_MAX = 0.95  # Maximum radius scaling factor
XI_INIT = 0.01  # Initial learnable radius scalar
GAMMA = 0.99  # Discount factor for GAE
GAE_LAMBDA = 0.95  # Lambda for GAE
PPO_CLIP_EPS = 0.2  # PPO clipping epsilon
BATCH_SIZE = 4  # Dummy batch size
MAX_SEQ_LEN = 128  # Dummy sequence length for inputs
NUM_STEPS = 5  # Number of PPO update iterations
LR = 3e-4  # Learning rate


# ===================================================================
# Phase 1: LoRA Actor
# ===================================================================


def build_lora_actor(
    model_name: str = "google/byt5-small",
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load a frozen ByT5 and inject LoRA adapters into decoder attention.

    Returns the PEFT-wrapped model and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Freeze every parameter in the base model
    for param in base_model.parameters():
        param.requires_grad = False

    # LoRA config – target decoder self-attention and cross-attention
    # ByT5 uses 'q', 'k', 'v', 'o' naming inside T5Attention
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o"],
        modules_to_save=None,
        # Only adapt decoder layers (match "decoder.block.*.layer.*.*.q" etc.)
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer


def get_action_log_probs(
    model: nn.Module,
    encoder_input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities of decoder actions.

    Args:
        model: PEFT-wrapped ByT5.
        encoder_input_ids: (B, S_enc) token ids for the encoder.
        decoder_input_ids: (B, S_dec) token ids treated as the 'action'.

    Returns:
        (B,) summed log-probs over the action tokens.
    """
    outputs = model(
        input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
    )
    # outputs.logits: (B, S_dec, vocab_size)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log-probs for the actual tokens (shifted by 1 for teacher forcing)
    # Target tokens are decoder_input_ids[:, 1:]
    target_ids = decoder_input_ids[:, 1:]  # (B, S_dec - 1)
    log_probs = log_probs[:, :-1, :]  # (B, S_dec - 1, V)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (B, S_dec - 1)

    # Sum over tokens to get sequence-level log-prob
    return token_log_probs.sum(dim=-1)  # (B,)


# ===================================================================
# Phase 2 & 3: Poincare Critic with Categorical Value Head
# ===================================================================


class PoincareCritic(nn.Module):
    """HYPER++ critic: Euclidean -> Poincare ball -> Categorical value.

    Architecture:
        1. Linear(1472 → 64) → RMSNorm → Learned scaling
        2. Poincare expmap at origin
        3. Hyperbolic Linear(64 -> 51) -> logmap -> softmax -> categorical value
    """

    def __init__(
        self,
        input_dim: int = ENCODER_HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_bins: int = NUM_BINS,
        rho_max: float = RHO_MAX,
        xi_init: float = XI_INIT,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.rho_max = rho_max
        self.manifold = PoincareBall(Curvature(1.0))

        # --- HYPER++ adapter ---
        self.linear = nn.Linear(input_dim, latent_dim)
        self.rms_norm = nn.RMSNorm(latent_dim)
        self.xi = nn.Parameter(torch.tensor(xi_init))

        # --- Categorical value head in hyperbolic space ---
        self.value_linear = hnn.HLinear(latent_dim, num_bins, self.manifold)

        # Support vector: evenly spaced bins [0, 1]
        self.register_buffer("support", torch.linspace(0.0, 1.0, num_bins))

    def project_to_poincare_ball(self, x_e: torch.Tensor) -> torch.Tensor:
        """Map Euclidean tangent vectors at the origin to Poincare-ball coordinates."""
        tangent = TangentTensor(data=x_e, man_dim=1, manifold=self.manifold)
        x_h = self.manifold.expmap(tangent)
        return cast(torch.Tensor, x_h.tensor)

    def forward(
        self,
        encoder_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning value estimate, bin logits, and bin probs.

        Args:
            encoder_out: (B, input_dim) mean-pooled frozen encoder features.

        Returns:
            value:      (B,) expected scalar value (prob-weighted bin centres).
            bin_logits: (B, num_bins) raw logits before softmax.
            bin_probs:  (B, num_bins) probability distribution over bins.
        """
        # HYPER++ adapter
        x = self.linear(encoder_out)
        x = self.rms_norm(x)
        x = self.rho_max * torch.sigmoid(self.xi) * x

        # Poincare projection via expmap at origin
        tangent = TangentTensor(data=x, man_dim=1, manifold=self.manifold)
        x_h = self.manifold.expmap(tangent)

        # Hyperbolic linear map, then logmap back to Euclidean logits.
        bin_logits_h = self.value_linear(x_h)  # (B, num_bins) manifold points
        bin_logits = self.manifold.logmap(x=None, y=bin_logits_h).tensor
        bin_probs = F.softmax(bin_logits, dim=-1)  # (B, num_bins)

        # Expected value = Σ p_i * z_i
        support = cast(torch.Tensor, self.support)
        value = (bin_probs * support).sum(dim=-1)  # (B,)

        return value, bin_logits, bin_probs


class HyperboloidCritic(PoincareCritic):
    """Backward-compatible alias; implementation now uses Poincare-ball geometry."""


# ===================================================================
# Phase 4 – Utility helpers
# ===================================================================


def compute_gae(
    returns: torch.Tensor,
    values: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> torch.Tensor:
    """Single-step GAE (no sequential trajectory needed for this demo).

    For the v0 feasibility test we treat each sample as an independent
    terminal transition:  advantage = return - V(s).

    In a full implementation this would iterate over timesteps.
    """
    advantages = returns - values.detach()
    return advantages


def returns_to_bin_targets(
    returns: torch.Tensor,
    num_bins: int = NUM_BINS,
) -> torch.Tensor:
    """Map scalar returns ∈ [0, 1] to the index of the nearest bin.

    Args:
        returns: (B,) empirical returns in [0, 1].
        num_bins: Number of discrete bins.

    Returns:
        (B,) LongTensor of bin indices.
    """
    # Clamp to [0, 1] for safety, then quantise
    clamped = returns.clamp(0.0, 1.0)
    indices = (clamped * (num_bins - 1)).round().long()
    return indices


def compute_ppo_actor_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = PPO_CLIP_EPS,
) -> torch.Tensor:
    """PPO clipped surrogate objective (negated for minimization)."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    return -surrogate.mean()


def compute_critic_loss(
    bin_logits: torch.Tensor,
    target_bins: torch.Tensor,
) -> torch.Tensor:
    """Categorical cross-entropy loss for the critic (NOT MSE)."""
    return F.cross_entropy(bin_logits, target_bins)


# ===================================================================
# Phase 5 – Gradient diagnostics
# ===================================================================


def report_gradient_norms(
    peft_model: nn.Module,
    critic: PoincareCritic,
) -> None:
    """Print gradient norms for key parameter groups."""
    # LoRA parameters
    lora_norm = 0.0
    lora_count = 0
    for name, param in peft_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            lora_norm += param.grad.data.norm(2).item() ** 2
            lora_count += 1
    lora_norm = math.sqrt(lora_norm)
    print(f"  LoRA grad norm          : {lora_norm:.6f}  ({lora_count} tensors)")

    # xi scalar
    if critic.xi.grad is not None:
        print(f"  xi grad                 : {critic.xi.grad.item():.6f}")
    else:
        print("  xi grad                 : None (no grad)")

    # Critic layer grad norms
    for layer_name in ("linear", "value_linear"):
        layer = getattr(critic, layer_name)
        layer_sq = 0.0
        layer_count = 0
        for param in layer.parameters():
            if param.grad is not None:
                layer_sq += param.grad.data.norm(2).item() ** 2
                layer_count += 1
        layer_norm = math.sqrt(layer_sq) if layer_count > 0 else float("nan")
        print(f"  {layer_name} grad norm    : {layer_norm:.6f} ({layer_count} tensors)")


def assert_base_frozen(peft_model: nn.Module) -> None:
    """Verify that all non-LoRA base parameters remain frozen."""
    base_model = cast(Any, peft_model).base_model.model
    for name, param in cast(Any, base_model).named_parameters():
        if "lora_" not in name:
            assert (
                not param.requires_grad
            ), f"Base parameter {name} has requires_grad=True — should be frozen!"
    print("  [OK] All base ByT5 parameters are frozen (requires_grad=False).")


# ===================================================================
# Main – Feasibility Training Loop
# ===================================================================


def main() -> None:  # noqa: C901
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 1. Build actor (frozen ByT5 + LoRA) and move to device
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Building LoRA Actor")
    print("=" * 60)
    peft_model, tokenizer = build_lora_actor()
    peft_model.to(device)

    # ------------------------------------------------------------------
    # 2. Build Poincare Critic
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2-3: Building Poincare Critic (Categorical Value Head)")
    print("=" * 60)
    critic = PoincareCritic().to(device)
    print(f"  Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

    # ------------------------------------------------------------------
    # 3. Optimiser – LoRA params + Critic params
    # ------------------------------------------------------------------
    trainable_actor_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_actor_params, "lr": LR},
            {"params": critic.parameters(), "lr": LR},
        ],
    )
    print(
        f"  Trainable actor params : {sum(p.numel() for p in trainable_actor_params):,}"
    )
    print(f"  Trainable critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # ------------------------------------------------------------------
    # 4. Dummy data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 4–5: PPO Feasibility Loop")
    print("=" * 60)

    # Dummy encoder inputs (byte-level token ids for ByT5)
    enc_input_ids = torch.randint(
        3,
        259,
        (BATCH_SIZE, MAX_SEQ_LEN),
        device=device,
    )
    # Dummy decoder inputs (action tokens)
    dec_input_ids = torch.randint(
        3,
        259,
        (BATCH_SIZE, MAX_SEQ_LEN // 4),
        device=device,
    )

    # Simulate 'old' log-probs from a previous policy rollout
    with torch.no_grad():
        old_log_probs = get_action_log_probs(
            peft_model,
            enc_input_ids,
            dec_input_ids,
        ).detach()

    # Dummy empirical returns ∈ [0, 1]
    dummy_returns = torch.rand(BATCH_SIZE, device=device)

    # ------------------------------------------------------------------
    # 5. Training iterations
    # ------------------------------------------------------------------
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()

        # ----- Actor forward -----
        new_log_probs = get_action_log_probs(
            peft_model,
            enc_input_ids,
            dec_input_ids,
        )

        # ----- Critic forward -----
        # Get frozen encoder features (mean-pooled)
        with torch.no_grad():
            encoder = cast(nn.Module, cast(Any, peft_model).get_encoder())
            enc_out = cast(Any, encoder(enc_input_ids)).last_hidden_state  # (B, S, H)
            # Mean pooling
            enc_features = enc_out.mean(dim=1)  # (B, H)

        values, bin_logits, bin_probs = critic(enc_features)

        # ----- GAE -----
        advantages = compute_gae(dummy_returns, values)

        # ----- PPO actor loss -----
        actor_loss = compute_ppo_actor_loss(
            new_log_probs,
            old_log_probs,
            advantages,
        )

        # ----- Categorical critic loss -----
        target_bins = returns_to_bin_targets(dummy_returns)
        critic_loss = compute_critic_loss(bin_logits, target_bins)

        # ----- Combined loss -----
        total_loss = actor_loss + 0.5 * critic_loss

        total_loss.backward()

        # ----- Diagnostics -----
        print(f"\n--- Step {step}/{NUM_STEPS} ---")
        print(f"  actor_loss  = {actor_loss.item():.6f}")
        print(f"  critic_loss = {critic_loss.item():.6f}")
        print(f"  total_loss  = {total_loss.item():.6f}")
        print(f"  V(s) mean   = {values.mean().item():.6f}")

        # Check for NaNs
        has_nan = (
            torch.isnan(total_loss)
            or any(
                torch.isnan(p.grad).any()
                for p in trainable_actor_params
                if p.grad is not None
            )
            or any(
                torch.isnan(p.grad).any()
                for p in critic.parameters()
                if p.grad is not None
            )
        )
        print(f"  NaN detected: {has_nan}")
        assert not has_nan, "NaN detected in loss or gradients!"

        report_gradient_norms(peft_model, critic)

        optimizer.step()

        # Update old_log_probs for next step (simulate rollout refresh)
        with torch.no_grad():
            old_log_probs = get_action_log_probs(
                peft_model,
                enc_input_ids,
                dec_input_ids,
            ).detach()

    # ------------------------------------------------------------------
    # 6. Final assertions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Final Assertions")
    print("=" * 60)
    assert_base_frozen(peft_model)
    print("\n✓ Hyperbolic PPO v0 feasibility test PASSED.\n")


if __name__ == "__main__":
    main()
