"""Shared PPO training engine with Euclidean and hyperbolic critic variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

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


@dataclass
class PPOConfig:
    """Hyperparameters kept identical across PPO variants for fair comparison."""

    actor_lr: float = ACTOR_LR
    critic_lr: float = CRITIC_LR
    num_ppo_epochs: int = NUM_PPO_EPOCHS
    max_state_tokens: int = MAX_STATE_TOKENS
    max_action_tokens: int = MAX_ACTION_TOKENS


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

        trainable_actor = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": trainable_actor, "lr": self.config.actor_lr},
                {"params": self.critic.parameters(), "lr": self.config.critic_lr},
            ]
        )

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
            device=self.device,
        ).clamp(0.0, 1.0)
        return states, returns

    def update_from_training_data(
        self,
        training_data: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Run PPO updates from MCTS-generated value samples."""
        states, returns = self._prepare_samples(training_data)
        if not states:
            return {
                "ppo_actor_loss": 0.0,
                "ppo_critic_loss": 0.0,
                "ppo_value_mean": 0.0,
            }

        enc_tok = self.tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_state_tokens,
        ).to(self.device)

        with torch.no_grad():
            dec_ids = cast(
                torch.Tensor,
                self.model.generate(
                    enc_tok.input_ids,
                    attention_mask=enc_tok.attention_mask,
                    max_new_tokens=self.config.max_action_tokens,
                    do_sample=False,
                ),
            )
            old_log_probs = get_action_log_probs(
                self.model,
                enc_tok.input_ids,
                dec_ids,
                attention_mask=enc_tok.attention_mask,
            ).detach()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        values = torch.zeros(1, device=self.device)

        for _ in range(self.config.num_ppo_epochs):
            self.optimizer.zero_grad()

            new_log_probs = get_action_log_probs(
                self.model,
                enc_tok.input_ids,
                dec_ids,
                attention_mask=enc_tok.attention_mask,
            )

            with torch.no_grad():
                encoder = cast(Any, self.model).get_encoder()
                enc_out = cast(
                    Any,
                    encoder(
                        input_ids=enc_tok.input_ids,
                        attention_mask=enc_tok.attention_mask,
                    ),
                ).last_hidden_state
                enc_features = enc_out.mean(dim=1)

            values, bin_logits, _ = cast(Any, self.critic)(enc_features)
            advantages = compute_gae(returns, values)

            actor_loss = compute_ppo_actor_loss(
                new_log_probs,
                old_log_probs,
                advantages,
            )
            target_bins = returns_to_bin_targets(returns)
            critic_loss = compute_critic_loss(bin_logits, target_bins)

            loss = actor_loss + 0.5 * critic_loss
            loss.backward()
            self.optimizer.step()

            total_actor_loss += float(actor_loss.item())
            total_critic_loss += float(critic_loss.item())

        avg_actor = total_actor_loss / self.config.num_ppo_epochs
        avg_critic = total_critic_loss / self.config.num_ppo_epochs
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
