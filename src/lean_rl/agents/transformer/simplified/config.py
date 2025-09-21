"""
Simplified configuration for the transformer agent.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleConfig:
    """Simple, clean configuration for LeanDojo RL training."""

    # Model architecture
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_episodes: int = 200  # Reduced for Mathlib4 training

    # Environment (LeanDojo specific)
    max_steps_per_episode: int = 30  # Theorem proving steps
    timeout: int = 600  # LeanDojo timeout in seconds
    reward_scheme: str = "dense"  # sparse, dense, or shaped

    # Device
    device: Optional[str] = None

    # Paths
    model_save_path: str = "leandojo_simplified_model.pt"
    cache_dir: Optional[str] = None  # LeanDojo cache directory

    # Mathlib4 specific
    repo_url: str = "https://github.com/leanprover-community/mathlib4"
    repo_commit: str = "29dcec074de168ac2bf835a77ef68bbe069194c5"
    max_theorems: int = 100  # Limit number of theorems for training
    theorem_difficulty: str = "basic"  # basic, intermediate, advanced

    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        # Set cache directory from environment if not specified
        if self.cache_dir is None:
            import os

            self.cache_dir = os.getenv("CACHE_DIR")
