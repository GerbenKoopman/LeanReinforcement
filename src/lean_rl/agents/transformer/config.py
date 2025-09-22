"""
Configuration for the Simplified Transformer Agent.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for the simplified Transformer Agent."""

    # Repository settings
    repo_url: str = "https://github.com/leanprover-community/mathlib4"
    repo_commit: str = "29dcec074de168ac2bf835a77ef68bbe069194c5"

    # Model hyperparameters
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    max_length: int = 256  # Max length for tactic generation

    # Training settings
    num_epochs: int = 10
    lr: float = 1e-4
    device: Optional[str] = None  # e.g., "cuda" or "cpu"

    # Environment settings
    max_steps_per_episode: int = 64
    timeout: int = 600
