"""
Simplified transformer agent module.

This module provides a clean, efficient alternative to the complex
hierarchical transformer architecture in the main transformer folder.

Key simplifications:
- Single-level policy instead of 3-level hierarchy
- Simple tokenization instead of complex mathematical parsing
- Built-in PyTorch transformer instead of custom RoPE implementation
- Unified action representation instead of hierarchical actions
- Streamlined training loop instead of complex search trees

Usage:
    from lean_rl.agents.transformer.simplified import SimplifiedTransformerAgent, SimpleConfig

    config = SimpleConfig()
    agent = SimplifiedTransformerAgent(**config.__dict__)
    action = agent.select_action(state)
"""

from .core import SimplifiedTransformerAgent, SimpleTokenizer
from .hpc_config import SimpleHPCConfig, HPC_A100_CONFIG, HPC_MIG_CONFIG
from .trainer import LeanDojoTrainer, create_leandojo_trainer

__all__ = [
    "SimplifiedTransformerAgent",
    "SimpleTokenizer",
    "SimpleHPCConfig",
    "HPC_A100_CONFIG",
    "HPC_MIG_CONFIG",
    "LeanDojoTrainer",
    "create_leandojo_trainer",
]
