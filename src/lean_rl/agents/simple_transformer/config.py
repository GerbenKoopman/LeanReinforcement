"""
Configuration for the Simple Transformer Agent.
"""

from dataclasses import dataclass

@dataclass
class SimpleTransformerConfig:
    """Hyperparameters for the SimpleTransformerAgent."""
    # Repository settings
    repo_url: str = "https://github.com/leanprover-community/mathlib4"
    repo_commit: str = "5424a3b568f65093b302da057a233327b9845424" # A recent commit
    build_deps: bool = False

    # Model hyperparameters
    vocab_size: int = 10000
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024

    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 32
    num_epochs: int = 10
    max_steps_per_episode: int = 30
    
    # Environment settings
    timeout: int = 300
