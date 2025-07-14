"""
Hierarchical policy networks for three-level theorem proving.

This module implements the three-level hierarchical RL architecture:
1. Strategic Level: High-level proof planning and goal decomposition
2. Tactical Level: Tactic family selection and sequence planning
3. Execution Level: Parameter generation and tactic application
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from .attention import MathematicalAttentionEncoder
from .utils import ProofStateTokenizer


class HierarchyLevel(Enum):
    """Enumeration of hierarchy levels."""

    STRATEGIC = 1
    TACTICAL = 2
    EXECUTION = 3


class StrategicActions:
    """Strategic-level actions for high-level proof planning."""

    # Basic strategic actions
    INDUCTION = "induction"
    CONTRADICTION = "contradiction"
    CASE_ANALYSIS = "case_analysis"
    DIRECT_PROOF = "direct_proof"
    REWRITE_SIMPLIFY = "rewrite_simplify"
    APPLY_LEMMA = "apply_lemma"
    UNFOLD_DEFINITION = "unfold_definition"

    # Advanced strategic actions
    STRONG_INDUCTION = "strong_induction"
    STRUCTURAL_INDUCTION = "structural_induction"
    WELL_FOUNDED_INDUCTION = "well_founded_induction"
    CONTRAPOSITIVE = "contrapositive"
    PROOF_BY_COUNTEREXAMPLE = "proof_by_counterexample"
    CONSTRUCTIVE_PROOF = "constructive_proof"
    EXISTENTIAL_INSTANTIATION = "existential_instantiation"
    UNIVERSAL_GENERALIZATION = "universal_generalization"
    DISJUNCTION_ELIMINATION = "disjunction_elimination"
    PROOF_BY_EXHAUSTION = "proof_by_exhaustion"
    INTERMEDIATE_VALUE_THEOREM = "intermediate_value_theorem"
    PIGEONHOLE_PRINCIPLE = "pigeonhole_principle"
    DOUBLE_COUNTING = "double_counting"
    PROBABILISTIC_METHOD = "probabilistic_method"
    COMPACTNESS_ARGUMENT = "compactness_argument"
    CATEGORY_THEORY_APPROACH = "category_theory_approach"
    ALGEBRAIC_MANIPULATION = "algebraic_manipulation"
    GEOMETRIC_INTUITION = "geometric_intuition"
    TOPOLOGICAL_ARGUMENT = "topological_argument"
    FUNCTIONAL_ANALYSIS = "functional_analysis"

    ALL_ACTIONS = [
        INDUCTION,
        CONTRADICTION,
        CASE_ANALYSIS,
        DIRECT_PROOF,
        REWRITE_SIMPLIFY,
        APPLY_LEMMA,
        UNFOLD_DEFINITION,
        STRONG_INDUCTION,
        STRUCTURAL_INDUCTION,
        WELL_FOUNDED_INDUCTION,
        CONTRAPOSITIVE,
        PROOF_BY_COUNTEREXAMPLE,
        CONSTRUCTIVE_PROOF,
        EXISTENTIAL_INSTANTIATION,
        UNIVERSAL_GENERALIZATION,
        DISJUNCTION_ELIMINATION,
        PROOF_BY_EXHAUSTION,
        INTERMEDIATE_VALUE_THEOREM,
        PIGEONHOLE_PRINCIPLE,
        DOUBLE_COUNTING,
        PROBABILISTIC_METHOD,
        COMPACTNESS_ARGUMENT,
        CATEGORY_THEORY_APPROACH,
        ALGEBRAIC_MANIPULATION,
        GEOMETRIC_INTUITION,
        TOPOLOGICAL_ARGUMENT,
        FUNCTIONAL_ANALYSIS,
        UNFOLD_DEFINITION,
    ]


class TacticalFamilies:
    """
    Tactical-level tactic families aligned with parameter_generator.py.

    These families correspond to the 16 tactic families defined in the
    TacticParameterGenerator's tactic_families dictionary.
    """

    # Core tactic families
    APPLY_FAMILY = "apply_family"  # apply, exact, refine, use
    REWRITE_FAMILY = "rewrite_family"  # rw, simp, conv, rwa
    INTRO_FAMILY = "intro_family"  # intro, intros, rintro
    CASE_FAMILY = "case_family"  # cases, rcases, induction, split
    CALC_FAMILY = "calc_family"  # calc, trans, symm
    FINISH_FAMILY = "finish_family"  # sorry, done, rfl, trivial

    # Extended tactic families
    AUTOMATION_FAMILY = (
        "automation_family"  # aesop, tauto, ring, norm_num, linarith, etc.
    )
    PROOF_FAMILY = "proof_family"  # have, show, suffices, assert
    STRUCTURAL_FAMILY = (
        "structural_family"  # constructor, left, right, ext, exfalso, etc.
    )
    ASSUMPTION_FAMILY = "assumption_family"  # assumption, simp_all, hint
    ADVANCED_REWRITE_FAMILY = (
        "advanced_rewrite_family"  # simp_rw, rw_mod_cast, field_simp, etc.
    )
    INDUCTION_FAMILY = "induction_family"  # induction', cases', rcases, obtain, choose
    QUANTIFIER_FAMILY = "quantifier_family"  # exists, use!, existsi, forall_intro
    CONVERSION_FAMILY = "conversion_family"  # change, convert, congr, show_term
    GOAL_MANAGEMENT_FAMILY = (
        "goal_management_family"  # swap, rotate_left, clear, rename, etc.
    )
    SPECIALIZED_FAMILY = "specialized_family"  # interval_cases, fin_cases, lift, etc.

    ALL_FAMILIES = [
        APPLY_FAMILY,
        REWRITE_FAMILY,
        INTRO_FAMILY,
        CASE_FAMILY,
        CALC_FAMILY,
        FINISH_FAMILY,
        AUTOMATION_FAMILY,
        PROOF_FAMILY,
        STRUCTURAL_FAMILY,
        ASSUMPTION_FAMILY,
        ADVANCED_REWRITE_FAMILY,
        INDUCTION_FAMILY,
        QUANTIFIER_FAMILY,
        CONVERSION_FAMILY,
        GOAL_MANAGEMENT_FAMILY,
        SPECIALIZED_FAMILY,
    ]

    @staticmethod
    def get_family_description(family: str) -> str:
        """Get a description of what tactics belong to each family."""
        descriptions = {
            "apply_family": "Application tactics: apply, exact, refine, use",
            "rewrite_family": "Basic rewrite tactics: rw, simp, conv, rwa",
            "intro_family": "Introduction tactics: intro, intros, rintro",
            "case_family": "Case analysis tactics: cases, rcases, induction, split",
            "calc_family": "Calculation tactics: calc, trans, symm",
            "finish_family": "Finishing tactics: sorry, done, rfl, trivial",
            "automation_family": "Automation tactics: aesop, tauto, ring, norm_num, linarith, nlinarith, omega, abel, polyrith, decide, norm_cast",
            "proof_family": "Proof construction tactics: have, show, suffices, assert",
            "structural_family": "Structural tactics: constructor, left, right, ext, exfalso, by_contra, contradiction, by_cases",
            "assumption_family": "Assumption-based tactics: assumption, simp_all, hint",
            "advanced_rewrite_family": "Advanced rewrite tactics: simp_rw, rw_mod_cast, simp_intro, field_simp, conv_lhs, conv_rhs",
            "induction_family": "Advanced induction tactics: induction', cases', rcases, obtain, choose",
            "quantifier_family": "Quantifier tactics: exists, use!, existsi, forall_intro",
            "conversion_family": "Conversion tactics: change, convert, congr, show_term",
            "goal_management_family": "Goal management tactics: swap, rotate_left, rotate_right, clear, rename, set",
            "specialized_family": "Specialized tactics: interval_cases, fin_cases, mod_cases, lift, push_neg",
        }
        return descriptions.get(family, f"Unknown family: {family}")

    @staticmethod
    def get_family_metadata() -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for each tactic family matching parameter_generator.py structure.

        This provides the same information as the tactic_families dictionary
        in TacticParameterGenerator for consistency.
        """
        return {
            "apply_family": {
                "tactics": ["apply", "exact", "refine", "use"],
                "parameter_types": ["hypothesis", "theorem", "lemma"],
                "max_params": 3,
            },
            "rewrite_family": {
                "tactics": ["rw", "simp", "conv", "rwa"],
                "parameter_types": ["equation", "lemma", "simp_lemma"],
                "max_params": 5,
            },
            "intro_family": {
                "tactics": ["intro", "intros", "rintro"],
                "parameter_types": ["variable_name", "pattern"],
                "max_params": 2,
            },
            "case_family": {
                "tactics": ["cases", "rcases", "induction", "split"],
                "parameter_types": ["variable", "pattern", "induction_principle"],
                "max_params": 3,
            },
            "calc_family": {
                "tactics": ["calc", "trans", "symm"],
                "parameter_types": ["expression", "equality"],
                "max_params": 4,
            },
            "finish_family": {
                "tactics": ["sorry", "done", "rfl", "trivial"],
                "parameter_types": [],
                "max_params": 0,
            },
            "automation_family": {
                "tactics": [
                    "aesop",
                    "tauto",
                    "ring",
                    "norm_num",
                    "linarith",
                    "nlinarith",
                    "omega",
                    "abel",
                    "polyrith",
                    "decide",
                    "norm_cast",
                ],
                "parameter_types": ["none"],
                "max_params": 0,
            },
            "proof_family": {
                "tactics": ["have", "show", "suffices", "assert"],
                "parameter_types": ["term", "proof"],
                "max_params": 2,
            },
            "structural_family": {
                "tactics": [
                    "constructor",
                    "left",
                    "right",
                    "ext",
                    "exfalso",
                    "by_contra",
                    "contradiction",
                    "by_cases",
                ],
                "parameter_types": ["hypothesis", "variable"],
                "max_params": 2,
            },
            "assumption_family": {
                "tactics": ["assumption", "simp_all", "hint"],
                "parameter_types": ["none"],
                "max_params": 0,
            },
            "advanced_rewrite_family": {
                "tactics": [
                    "simp_rw",
                    "rw_mod_cast",
                    "simp_intro",
                    "field_simp",
                    "conv_lhs",
                    "conv_rhs",
                ],
                "parameter_types": ["equation", "lemma", "pattern"],
                "max_params": 5,
            },
            "induction_family": {
                "tactics": ["induction'", "cases'", "rcases", "obtain", "choose"],
                "parameter_types": ["variable", "pattern", "hypothesis"],
                "max_params": 3,
            },
            "quantifier_family": {
                "tactics": ["exists", "use!", "existsi", "forall_intro"],
                "parameter_types": ["term", "witness"],
                "max_params": 4,
            },
            "conversion_family": {
                "tactics": ["change", "convert", "congr", "show_term"],
                "parameter_types": ["term", "equality"],
                "max_params": 2,
            },
            "goal_management_family": {
                "tactics": [
                    "swap",
                    "rotate_left",
                    "rotate_right",
                    "clear",
                    "rename",
                    "set",
                ],
                "parameter_types": ["variable", "name"],
                "max_params": 2,
            },
            "specialized_family": {
                "tactics": [
                    "interval_cases",
                    "fin_cases",
                    "mod_cases",
                    "lift",
                    "push_neg",
                ],
                "parameter_types": ["hypothesis", "bound"],
                "max_params": 3,
            },
        }


class HierarchicalPolicyNetwork(nn.Module):
    """
    Main hierarchical policy network coordinating all three levels.

    This network manages the hierarchical decision-making process and
    coordinates between strategic, tactical, and execution policies.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        tokenizer: Optional[ProofStateTokenizer] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.tokenizer = tokenizer or ProofStateTokenizer(vocab_size)

        # Shared proof state encoder
        self.proof_encoder = MathematicalAttentionEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Level-specific policies
        self.strategic_policy = StrategicPolicy(
            d_model, len(StrategicActions.ALL_ACTIONS)
        )
        self.tactical_policy = TacticalPolicy(
            d_model, len(TacticalFamilies.ALL_FAMILIES)
        )
        self.execution_policy = ExecutionPolicy(
            d_model, vocab_size, tokenizer=self.tokenizer
        )

        # Value networks for each level
        self.strategic_value = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )

        self.tactical_value = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )

        self.execution_value = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        level: HierarchyLevel,
        goal_mask: Optional[torch.Tensor] = None,
        hypothesis_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical policy.

        Args:
            input_ids: Tokenized proof state
            level: Which hierarchy level to use
            goal_mask: Mask for goal positions
            hypothesis_mask: Mask for hypothesis positions
            attention_mask: General attention mask
            **kwargs: Additional arguments for specific levels

        Returns:
            Dictionary with policy logits and value estimates
        """
        # Encode proof state
        encodings = self.proof_encoder(
            input_ids=input_ids,
            goal_mask=goal_mask,
            hypothesis_mask=hypothesis_mask,
            attention_mask=attention_mask,
        )

        if level == HierarchyLevel.STRATEGIC:
            return self._strategic_forward(encodings, **kwargs)
        elif level == HierarchyLevel.TACTICAL:
            return self._tactical_forward(encodings, **kwargs)
        elif level == HierarchyLevel.EXECUTION:
            return self._execution_forward(encodings, **kwargs)
        else:
            raise ValueError(f"Unknown hierarchy level: {level}")

    def _strategic_forward(
        self, encodings: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Strategic level forward pass."""
        # Use goal and context encodings for strategic decisions
        strategic_repr = encodings["goal_encoding"] + encodings["context_encoding"]

        # Pool to get single representation
        pooled_repr = strategic_repr.mean(dim=1)  # Simple mean pooling

        # Get policy logits and value
        policy_logits = self.strategic_policy(pooled_repr)
        value = self.strategic_value(pooled_repr)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "representation": pooled_repr,
        }

    def _tactical_forward(
        self,
        encodings: Dict[str, torch.Tensor],
        strategic_action: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Tactical level forward pass."""
        # Combine all encodings for tactical decisions
        tactical_repr = (
            encodings["full_encoding"]
            + encodings["hypothesis_encoding"]
            + encodings["context_encoding"]
        )

        # Pool representation
        pooled_repr = tactical_repr.mean(dim=1)

        # Condition on strategic action if provided
        if strategic_action is not None:
            strategic_embedding = self.tactical_policy.get_strategic_embedding(
                strategic_action
            )
            pooled_repr = pooled_repr + strategic_embedding

        # Get policy logits and value
        policy_logits = self.tactical_policy(pooled_repr)
        value = self.tactical_value(pooled_repr)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "representation": pooled_repr,
        }

    def _execution_forward(
        self,
        encodings: Dict[str, torch.Tensor],
        tactic_family: Optional[str] = None,
        available_terms: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Execution level forward pass."""
        # Use full encoding for parameter generation
        execution_repr = encodings["full_encoding"]

        # Get policy logits and value
        result = self.execution_policy(
            execution_repr, tactic_family=tactic_family, available_terms=available_terms
        )

        # Add value estimate
        pooled_repr = execution_repr.mean(dim=1)
        result["value"] = self.execution_value(pooled_repr)

        return result


class StrategicPolicy(nn.Module):
    """Strategic-level policy for high-level proof planning."""

    def __init__(self, d_model: int, num_actions: int):
        super().__init__()

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate strategic action logits."""
        return self.policy_head(x)


class TacticalPolicy(nn.Module):
    """Tactical-level policy for tactic family selection."""

    def __init__(self, d_model: int, num_families: int):
        super().__init__()

        self.d_model = d_model

        # Strategic action embeddings for conditioning
        self.strategic_embeddings = nn.Embedding(
            len(StrategicActions.ALL_ACTIONS), d_model
        )

        # Create mapping from action strings to indices
        self.strategic_action_to_idx = {
            action: idx for idx, action in enumerate(StrategicActions.ALL_ACTIONS)
        }

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_families),
        )

    def get_strategic_embedding(self, strategic_action: str) -> torch.Tensor:
        """Get embedding for strategic action."""
        if strategic_action not in self.strategic_action_to_idx:
            # Default to first action if unknown
            idx = 0
        else:
            idx = self.strategic_action_to_idx[strategic_action]

        idx_tensor = torch.tensor(
            [idx], dtype=torch.long, device=next(self.parameters()).device
        )
        return self.strategic_embeddings(idx_tensor).squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate tactical family logits."""
        return self.policy_head(x)


class ExecutionPolicy(nn.Module):
    """Execution-level policy for parameter generation."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_params: int = 10,
        tokenizer: Optional[ProofStateTokenizer] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_params = max_params
        self.tokenizer = tokenizer or ProofStateTokenizer(vocab_size)

        # Tactic family embeddings
        self.family_embeddings = nn.Embedding(
            len(TacticalFamilies.ALL_FAMILIES), d_model
        )

        # Create mapping from family strings to indices
        self.family_to_idx = {
            family: idx for idx, family in enumerate(TacticalFamilies.ALL_FAMILIES)
        }

        # Parameter generation heads
        self.parameter_type_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),  # term, hypothesis, none, auto
        )

        # For generating specific parameters
        self.term_generator = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, vocab_size)
        )

        # Attention for selecting existing terms/hypotheses
        self.selection_attention = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=0.1, batch_first=True
        )

        # Term embedding layer for encoding available terms
        self.term_embedding = nn.Embedding(vocab_size, d_model)

        # Linear layer to generate selection scores for available terms
        self.term_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # concat term and context
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        tactic_family: Optional[str] = None,
        available_terms: Optional[List[str]] = None,
    ) -> Dict[str, Union[torch.Tensor, None]]:
        """
        Generate execution parameters.

        Args:
            x: Encoded proof state [batch_size, seq_len, d_model]
            tactic_family: Target tactic family
            available_terms: Available terms for selection

        Returns:
            Dictionary with parameter generation outputs
        """
        batch_size, seq_len, _ = x.shape

        # Pool representation for decision making
        pooled_repr = x.mean(dim=1)  # [batch_size, d_model]

        # Condition on tactic family if provided
        if tactic_family is not None:
            family_embedding = self.get_family_embedding(tactic_family)
            pooled_repr = pooled_repr + family_embedding

        # Determine parameter type
        param_type_logits = self.parameter_type_head(pooled_repr)

        # Generate term if needed
        term_logits = self.term_generator(pooled_repr)

        # Handle available terms for selection
        if available_terms is not None and available_terms:
            # Use the tokenizer to encode term strings
            term_embeddings = self.encode_available_terms(available_terms, x.device)

            # Expand for batch dimension: [num_terms, d_model] -> [batch_size, num_terms, d_model]
            term_embeddings = term_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

            # Compute selection scores for each available term
            context_expanded = pooled_repr.unsqueeze(1).expand(
                -1, len(available_terms), -1
            )
            combined_repr = torch.cat([term_embeddings, context_expanded], dim=-1)
            selection_scores = self.term_scorer(combined_repr).squeeze(
                -1
            )  # [batch_size, num_terms]

            # Also compute attention weights for interpretability
            attended_repr, attention_weights = self.selection_attention(
                pooled_repr.unsqueeze(1), term_embeddings, term_embeddings
            )
            selection_logits = attended_repr.squeeze(1)
        else:
            # No specific terms available, use general representation
            selection_logits = pooled_repr
            selection_scores = None
            attention_weights = None

        return {
            "parameter_type_logits": param_type_logits,
            "term_generation_logits": term_logits,
            "selection_logits": selection_logits,
            "term_selection_scores": selection_scores,  # New: scores for specific available terms
            "attention_weights": attention_weights,
            "representation": pooled_repr,
        }

    def get_family_embedding(self, tactic_family: str) -> torch.Tensor:
        """Get embedding for tactic family."""
        if tactic_family not in self.family_to_idx:
            # Default to first family if unknown
            idx = 0
        else:
            idx = self.family_to_idx[tactic_family]

        idx_tensor = torch.tensor(
            [idx], dtype=torch.long, device=next(self.parameters()).device
        )
        return self.family_embeddings(idx_tensor).squeeze(0)

    def encode_available_terms(
        self,
        available_terms: Union[List[str], List[int], List[Union[str, int]]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode available terms into embeddings using the tokenizer.

        Args:
            available_terms: List of term strings or token IDs
            device: Device to place tensors on

        Returns:
            Term embeddings tensor [num_terms, d_model]
        """
        if not available_terms:
            return torch.empty(0, self.d_model, device=device)

        term_embeddings = []
        for term in available_terms:
            if isinstance(term, str):
                # Use tokenizer to encode the term string
                term_token_ids = self.tokenizer.encode(term)
                if term_token_ids:
                    # For multi-token terms, we average the embeddings
                    token_id = sum(term_token_ids) // len(term_token_ids)
                else:
                    # Fallback to unknown token
                    token_id = self.tokenizer.token_to_id.get("<unk>", 1)
            else:
                # Assume term is already a token ID
                token_id = min(
                    term, len(self.tokenizer.id_to_token) - 1
                )  # Clamp to vocab size

            # Get embedding for this token
            token_tensor = torch.tensor([token_id], device=device, dtype=torch.long)
            term_embedding = self.term_embedding(token_tensor).squeeze(0)
            term_embeddings.append(term_embedding)

        return torch.stack(term_embeddings)
