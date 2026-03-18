"""Shared PPO training engine with Euclidean and hyperbolic critic variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypedDict, cast

import torch
import torch.nn as nn

from peft import PeftModel

from lean_reinforcement.agent.ppo.actor import build_lora_actor, get_action_log_probs
from lean_reinforcement.agent.ppo.constants import (
    ACTOR_LR,
    CRITIC_LR,
    MAX_ACTION_TOKENS,
    MAX_STATE_TOKENS,
    NUM_PPO_EPOCHS,
)
from lean_reinforcement.agent.ppo.critics import EuclideanCritic, HyperbolicCritic
from lean_reinforcement.agent.ppo.losses import (
    compute_critic_loss,
    compute_gae,
    compute_ppo_actor_loss,
    returns_to_bin_targets,
)
from lean_reinforcement.utilities.optimizer import unwrap_optimizer_params


@dataclass
class PPOConfig:
    """Hyperparameters kept identical across PPO variants for fair comparison."""

    actor_lr: float = ACTOR_LR
    critic_lr: float = CRITIC_LR
    num_ppo_epochs: int = NUM_PPO_EPOCHS
    max_state_tokens: int = MAX_STATE_TOKENS
    max_action_tokens: int = MAX_ACTION_TOKENS
    # Keep this conservative because PPO runs alongside other GPU-heavy components.
    minibatch_size: int = 1


class PPOBatch(TypedDict):
    """Cached PPO minibatch tensors kept on CPU between optimization steps."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor


class _BasePPO:
    """Base PPO implementation with critic-specific subclasses."""

    def __init__(self, model_name: str, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        actor_model, tokenizer = build_lora_actor(model_name=model_name)
        self.model: Any = actor_model
        self.tokenizer: Any = tokenizer
        self.model.to(self.device)
        self.critic = self._build_critic().to(self.device)

        trainable_actor = unwrap_optimizer_params(self.model.parameters())
        critic_params = unwrap_optimizer_params(self.critic.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": trainable_actor, "lr": self.config.actor_lr},
                {"params": critic_params, "lr": self.config.critic_lr},
            ]
        )

    def _rebuild_optimizer(self) -> None:
        """Rebuild optimizer after device changes (e.g. OOM fallback)."""
        trainable_actor = unwrap_optimizer_params(self.model.parameters())
        critic_params = unwrap_optimizer_params(self.critic.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": trainable_actor, "lr": self.config.actor_lr},
                {"params": critic_params, "lr": self.config.critic_lr},
            ]
        )

    def _fallback_to_cpu(self) -> None:
        """Move PPO modules to CPU and reinitialize optimizer for safety."""
        if self.device.type == "cpu":
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.critic.to(self.device)
        self._rebuild_optimizer()

    def _build_critic(self) -> nn.Module:
        raise NotImplementedError

    def _prepare_samples(
        self,
        training_data: Sequence[Dict[str, Any]],
    ) -> Tuple[List[str], torch.Tensor]:
        samples = [
            d for d in training_data if d.get("type") == "value" and d.get("state")
        ]
        if len(samples) < 2:
            return [], torch.empty(0, device=self.device)

        states = [str(s["state"]) for s in samples]
        raw_targets = [float(s.get("value_target", 0.0)) for s in samples]
        returns = torch.tensor(
            [(t + 1.0) / 2.0 for t in raw_targets],
            device="cpu",
        ).clamp(0.0, 1.0)
        return states, returns

    def _build_cached_minibatches(
        self,
        states: Sequence[str],
        returns: torch.Tensor,
    ) -> List[PPOBatch]:
        """Build behavior-policy caches in small chunks to avoid GPU OOM."""
        minibatch_size = max(1, int(self.config.minibatch_size))
        batches: List[PPOBatch] = []

        for start in range(0, len(states), minibatch_size):
            end = min(start + minibatch_size, len(states))
            batch_states = list(states[start:end])
            batch_returns = returns[start:end]

            enc_tok = self.tokenizer(
                batch_states,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_state_tokens,
            )
            input_ids = cast(torch.Tensor, enc_tok.input_ids).to(self.device)
            attention_mask = cast(torch.Tensor, enc_tok.attention_mask).to(self.device)

            with torch.no_grad():
                dec_ids = cast(
                    torch.Tensor,
                    self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_action_tokens,
                        do_sample=False,
                    ),
                )
                old_log_probs = get_action_log_probs(
                    self.model,
                    input_ids,
                    dec_ids,
                    attention_mask=attention_mask,
                ).detach()

            batches.append(
                {
                    "input_ids": input_ids.cpu(),
                    "attention_mask": attention_mask.cpu(),
                    "decoder_input_ids": dec_ids.cpu(),
                    "old_log_probs": old_log_probs.cpu(),
                    "returns": batch_returns.detach().cpu(),
                }
            )

            del input_ids, attention_mask, dec_ids, old_log_probs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return batches

    def update_from_training_data(
        self,
        training_data: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Run PPO updates from MCTS-generated value samples."""
        try:
            return self._update_from_training_data_impl(training_data)
        except torch.cuda.OutOfMemoryError:
            # GPU pressure can spike because PPO coexists with inference/value-head models.
            # Retry once on CPU to keep long benchmark runs progressing.
            self._fallback_to_cpu()
            return self._update_from_training_data_impl(training_data)

    def _update_from_training_data_impl(
        self,
        training_data: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Internal PPO update implementation used by normal + fallback flows."""
        states, returns = self._prepare_samples(training_data)
        if not states:
            return {
                "ppo_actor_loss": 0.0,
                "ppo_critic_loss": 0.0,
                "ppo_value_mean": 0.0,
            }

        cached_batches = self._build_cached_minibatches(states, returns)
        if not cached_batches:
            return {
                "ppo_actor_loss": 0.0,
                "ppo_critic_loss": 0.0,
                "ppo_value_mean": 0.0,
            }

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_steps = 0
        values = torch.zeros(1, device=self.device)

        for _ in range(self.config.num_ppo_epochs):
            for batch in cached_batches:
                self.optimizer.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                dec_ids = batch["decoder_input_ids"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                batch_returns = batch["returns"].to(self.device)

                new_log_probs = get_action_log_probs(
                    self.model,
                    input_ids,
                    dec_ids,
                    attention_mask=attention_mask,
                )

                with torch.no_grad():
                    encoder = cast(Any, self.model).get_encoder()
                    enc_out = cast(
                        Any,
                        encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        ),
                    ).last_hidden_state
                    enc_features = enc_out.mean(dim=1)

                values, bin_logits, _ = cast(Any, self.critic)(enc_features)
                advantages = compute_gae(batch_returns, values)

                actor_loss = compute_ppo_actor_loss(
                    new_log_probs,
                    old_log_probs,
                    advantages,
                )
                target_bins = returns_to_bin_targets(batch_returns)
                critic_loss = compute_critic_loss(bin_logits, target_bins)

                loss = actor_loss + 0.5 * critic_loss
                loss.backward()
                self.optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_steps += 1

        avg_actor = total_actor_loss / max(1, total_steps)
        avg_critic = total_critic_loss / max(1, total_steps)
        value_mean = float(values.mean().item()) if values.numel() > 0 else 0.0

        return {
            "ppo_actor_loss": avg_actor,
            "ppo_critic_loss": avg_critic,
            "ppo_value_mean": value_mean,
        }

    def save_checkpoint(
        self, checkpoint_dir: Path, epoch: int, prefix: str = "ppo"
    ) -> None:
        """Save LoRA adapter, critic state, and optimizer state."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        actor_path = checkpoint_dir / f"{prefix}_actor_epoch_{epoch}.pth"
        cast(Any, self.model).save_pretrained(str(actor_path))

        critic_path = checkpoint_dir / f"{prefix}_critic_epoch_{epoch}.pth"
        torch.save(
            {
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            str(critic_path),
        )

    def load_latest_checkpoint(self, checkpoint_dir: Path, prefix: str = "ppo") -> int:
        """Load the latest checkpoint and return restored epoch, or 0 if absent."""
        max_epoch = 0
        for f in checkpoint_dir.glob(f"{prefix}_critic_epoch_*.pth"):
            try:
                epoch = int(f.stem.split("_")[-1])
                max_epoch = max(max_epoch, epoch)
            except ValueError:
                continue

        if max_epoch == 0:
            return 0

        critic_path = checkpoint_dir / f"{prefix}_critic_epoch_{max_epoch}.pth"
        ckpt = torch.load(str(critic_path), map_location=self.device)
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        actor_path = checkpoint_dir / f"{prefix}_actor_epoch_{max_epoch}.pth"
        if actor_path.exists():
            model = PeftModel.from_pretrained(
                cast(Any, self.model).base_model.model, str(actor_path)
            )
            self.model = cast(nn.Module, model.to(self.device))

        return max_epoch


class EuclideanPPO(_BasePPO):
    """PPO with Euclidean categorical critic."""

    def _build_critic(self) -> nn.Module:
        return EuclideanCritic()


class HyperbolicPPO(_BasePPO):
    """PPO with Poincare-ball categorical critic."""

    def _build_critic(self) -> nn.Module:
        return HyperbolicCritic()
