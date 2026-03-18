from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from lean_reinforcement.agent.ppo.ppo import (
    EuclideanPPO,
    HyperbolicPPO,
    PPOConfig,
    _BasePPO,
)


class PPOAgent:
    """A wrapper for PPO training to integrate with the main training loop."""

    ppo: _BasePPO

    def __init__(self, model_name: str, use_hyperbolic: bool = False) -> None:
        """Initialize either a Euclidean or Hyperbolic PPO agent."""
        config = PPOConfig()
        if use_hyperbolic:
            self.ppo = HyperbolicPPO(model_name=model_name, config=config)
        else:
            self.ppo = EuclideanPPO(model_name=model_name, config=config)

    def update(
        self,
        training_data: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Update the PPO agent from a batch of training data."""
        return self.ppo.update_from_training_data(training_data)

    def save_checkpoint(self, checkpoint_dir: Path, epoch: int) -> None:
        """Save the PPO agent's state."""
        prefix = (
            "ppo_hyperbolic" if isinstance(self.ppo, HyperbolicPPO) else "ppo_euclidean"
        )
        self.ppo.save_checkpoint(checkpoint_dir, epoch, prefix=prefix)

    def load_latest_checkpoint(self, checkpoint_dir: Path) -> int:
        """Load the latest PPO agent's state."""
        prefix = (
            "ppo_hyperbolic" if isinstance(self.ppo, HyperbolicPPO) else "ppo_euclidean"
        )
        return self.ppo.load_latest_checkpoint(checkpoint_dir, prefix=prefix)
