"""
Transformer-based Hierarchical Reinforcement Learning for Lean Theorem Proving.

This module implements a three-level hierarchical RL architecture using transformer
attention mechanisms for automated theorem proving in Lean 4.

Architecture:
    Strategic Level: High-level proof planning and goal decomposition
    Tactical Level: Tactic family selection and sequence planning
    Execution Level: Parameter generation and tactic application

Components:
    - MathematicalAttentionEncoder: Specialized transformer for proof states
    - HierarchicalPolicyNetwork: Three-level policy hierarchy
    - TacticPointerNetwork: Attention-based tactic selection
    - ParameterGenerator: Execution-level parameter generation
    - HierarchicalAgent: Main agent combining all components
"""

from .model.utils import (
    ProofStateTokenizer,
    TacticEncoder,
)

__all__ = [
    # Utilities
    "ProofStateTokenizer",
    "TacticEncoder",
]
