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

from .attention import (
    MathematicalAttentionEncoder,
    MultiHeadAttention,
)

from .hierarchy import (
    HierarchicalPolicyNetwork,
    StrategicPolicy,
    TacticalPolicy,
    ExecutionPolicy,
    HierarchyLevel,
)

from .pointer_network import (
    TacticPointerNetwork,
    ParameterPointerNetwork,
)

from .parameter_generator import (
    TacticParameterGenerator,
    AutoregressiveTermGenerator,
    PremiseRetriever,
)

from .agent import (
    HierarchicalTransformerAgent,
    HierarchicalSearchTree,
)

from .utils import (
    ProofStateTokenizer,
    TacticEncoder,
)

__all__ = [
    # Attention mechanisms
    "MathematicalAttentionEncoder",
    "MultiHeadAttention",
    # Hierarchical policies
    "HierarchicalPolicyNetwork",
    "StrategicPolicy",
    "TacticalPolicy",
    "ExecutionPolicy",
    # Pointer networks
    "TacticPointerNetwork",
    "ParameterPointerNetwork",
    # Parameter generation
    "TacticParameterGenerator",
    "AutoregressiveTermGenerator",
    "PremiseRetriever",
    # Main agent
    "HierarchicalTransformerAgent",
    "HierarchicalSearchTree",
    # Utilities
    "ProofStateTokenizer",
    "TacticEncoder",
    "HierarchyLevel",
]
