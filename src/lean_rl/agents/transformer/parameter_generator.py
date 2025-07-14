"""
Tactic parameter generation networks.

This module implements specialized networks for generating parameters for
different Lean tactics, including retrieval-augmented generation and
autoregressive term generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Union
from enum import Enum

from .attention import MultiHeadAttention, AttentionPooling


class ParameterType(Enum):
    """Types of parameters that can be generated."""

    TERM = "term"
    HYPOTHESIS = "hypothesis"
    NONE = "none"
    AUTO = "auto"
    LEMMA = "lemma"
    THEOREM = "theorem"
    EQUATION = "equation"
    PATTERN = "pattern"
    VARIABLE = "variable"


class TacticParameterGenerator(nn.Module):
    """
    Attention-based parameter generation for Lean tactics.

    This network generates appropriate parameters for different tactic families
    using retrieval, attention, and autoregressive generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        max_premise_length: int = 100,
        max_term_length: int = 50,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_premise_length = max_premise_length
        self.max_term_length = max_term_length

        # Tactic family embeddings with structured metadata
        self.tactic_families = {
            "apply_family": {
                "tactics": ["apply", "exact", "refine", "use"],
                "parameter_types": ["hypothesis", "theorem", "lemma"],
                "max_params": 3,
                "embedding_id": 0,
            },
            "rewrite_family": {
                "tactics": ["rw", "simp", "conv", "rwa"],
                "parameter_types": ["equation", "lemma", "simp_lemma"],
                "max_params": 5,
                "embedding_id": 1,
            },
            "intro_family": {
                "tactics": ["intro", "intros", "rintro"],
                "parameter_types": ["variable_name", "pattern"],
                "max_params": 2,
                "embedding_id": 2,
            },
            "case_family": {
                "tactics": ["cases", "rcases", "induction", "split"],
                "parameter_types": ["variable", "pattern", "induction_principle"],
                "max_params": 3,
                "embedding_id": 3,
            },
            "calc_family": {
                "tactics": ["calc", "trans", "symm"],
                "parameter_types": ["expression", "equality"],
                "max_params": 4,
                "embedding_id": 4,
            },
            "finish_family": {
                "tactics": ["sorry", "done", "rfl", "trivial"],
                "parameter_types": [],
                "max_params": 0,
                "embedding_id": 5,
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
                "embedding_id": 6,
            },
            "proof_family": {
                "tactics": ["have", "show", "suffices", "assert"],
                "parameter_types": ["term", "proof"],
                "max_params": 2,
                "embedding_id": 7,
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
                "embedding_id": 8,
            },
            "assumption_family": {
                "tactics": ["assumption", "simp_all", "hint"],
                "parameter_types": ["none"],
                "max_params": 0,
                "embedding_id": 9,
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
                "embedding_id": 10,
            },
            "induction_family": {
                "tactics": ["induction'", "cases'", "rcases", "obtain", "choose"],
                "parameter_types": ["variable", "pattern", "hypothesis"],
                "max_params": 3,
                "embedding_id": 11,
            },
            "quantifier_family": {
                "tactics": ["exists", "use!", "existsi", "forall_intro"],
                "parameter_types": ["term", "witness"],
                "max_params": 4,
                "embedding_id": 12,
            },
            "conversion_family": {
                "tactics": ["change", "convert", "congr", "show_term"],
                "parameter_types": ["term", "equality"],
                "max_params": 2,
                "embedding_id": 13,
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
                "embedding_id": 14,
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
                "embedding_id": 15,
            },
        }

        self.family_names = list(self.tactic_families.keys())
        self.family_embedding = nn.Embedding(len(self.tactic_families), d_model)
        self.family_to_idx = {
            family: idx for idx, family in enumerate(self.tactic_families)
        }

        # Parameter type classifier
        self.param_type_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, len(ParameterType)),
        )

        # Premise retriever
        self.premise_retriever = PremiseRetriever(d_model, n_heads)

        # Autoregressive term generator
        self.term_generator = AutoregressiveTermGenerator(vocab_size, d_model)

        # Hypothesis selector
        self.hypothesis_selector = HypothesisSelector(d_model, n_heads)

        # Rewrite sequence generator
        self.rewrite_generator = RewriteSequenceGenerator(
            d_model, n_heads, max_term_length
        )

        # Use selector for existential instantiation
        self.use_generator = UseTermGenerator(d_model, vocab_size)

        # Simp lemma selector
        self.simp_selector = SimpLemmaSelector(d_model, n_heads)

        # Loss computation weights
        self.param_type_weight = nn.Parameter(torch.tensor(1.0))
        self.generation_weight = nn.Parameter(torch.tensor(1.0))
        self.selection_weight = nn.Parameter(torch.tensor(1.0))

    def get_tactic_family(self, tactic_name: str) -> Optional[str]:
        """Get the family for a specific tactic name."""
        for family, info in self.tactic_families.items():
            if tactic_name in info["tactics"]:
                return family
        return None

    def get_supported_tactics(self) -> List[str]:
        """Get list of all supported tactics."""
        tactics = []
        for family_info in self.tactic_families.values():
            tactics.extend(family_info["tactics"])
        return tactics

    def compute_loss(
        self, outputs: Dict[str, Any], targets: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute training loss for parameter generation.

        Args:
            outputs: Model outputs from forward pass
            targets: Target parameters and types

        Returns:
            Combined loss tensor
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        # Parameter type classification loss
        if "parameter_type_logits" in outputs and "target_param_type" in targets:
            param_type_loss = F.cross_entropy(
                outputs["parameter_type_logits"], targets["target_param_type"]
            )
            total_loss += self.param_type_weight * param_type_loss

        # Generation losses (if applicable)
        if "generated_term" in outputs and "target_term" in targets:
            if "logits" in outputs["generated_term"]:
                gen_loss = F.cross_entropy(
                    outputs["generated_term"]["logits"].view(-1, self.vocab_size),
                    targets["target_term"].view(-1),
                    ignore_index=0,  # Assuming 0 is padding token
                )
                total_loss += self.generation_weight * gen_loss

        # Selection losses
        if "hypothesis_selection" in outputs and "target_hypothesis" in targets:
            if "selection_logits" in outputs["hypothesis_selection"]:
                sel_loss = F.cross_entropy(
                    outputs["hypothesis_selection"]["selection_logits"],
                    targets["target_hypothesis"],
                )
                total_loss += self.selection_weight * sel_loss

        return total_loss

    def forward(
        self,
        proof_state: torch.Tensor,
        tactic_family: str,
        available_terms: Optional[torch.Tensor] = None,
        available_hypotheses: Optional[torch.Tensor] = None,
        knowledge_base: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Generate parameters for a specific tactic family.

        Args:
            proof_state: Encoded proof state [batch_size, seq_len, d_model]
            tactic_family: Target tactic family name
            available_terms: Available term embeddings [batch_size, n_terms, d_model]
            available_hypotheses: Available hypothesis embeddings [batch_size, n_hyps, d_model]
            knowledge_base: Knowledge base embeddings [batch_size, n_premises, d_model]

        Returns:
            Dictionary with generated parameters for the tactic family
        """
        batch_size, seq_len, _ = proof_state.size()

        # Get tactic family embedding
        family_idx = self.family_to_idx.get(tactic_family, 0)
        family_emb = self.family_embedding(
            torch.tensor([family_idx], device=proof_state.device)
        )

        # Pool proof state representation
        pooled_state = proof_state.mean(dim=1)  # [batch_size, d_model]

        # Combine with tactic family embedding
        combined_repr = pooled_state + family_emb.squeeze(0)

        # Classify parameter type needed
        param_type_logits = self.param_type_classifier(combined_repr)

        # Generate parameters using structured family metadata
        if tactic_family in self.tactic_families:

            # Use family-specific generation based on parameter types
            if tactic_family == "apply_family":
                return self._generate_apply_params(
                    proof_state,
                    combined_repr,
                    available_terms,
                    available_hypotheses,
                    knowledge_base,
                    param_type_logits,
                )
            elif tactic_family == "rewrite_family":
                return self._generate_rewrite_params(
                    proof_state,
                    combined_repr,
                    available_terms,
                    knowledge_base,
                    param_type_logits,
                )
            elif tactic_family == "intro_family":
                return self._generate_intro_params(
                    proof_state, combined_repr, param_type_logits
                )
            elif tactic_family == "case_family":
                return self._generate_case_params(
                    proof_state, combined_repr, available_hypotheses, param_type_logits
                )
            elif tactic_family == "calc_family":
                return self._generate_calc_params(
                    proof_state, combined_repr, available_terms, param_type_logits
                )
            elif tactic_family == "automation_family":
                return self._generate_automation_params(
                    proof_state, combined_repr, param_type_logits
                )
            elif tactic_family == "proof_family":
                return self._generate_proof_params(
                    proof_state, combined_repr, available_terms, param_type_logits
                )
            elif tactic_family == "structural_family":
                return self._generate_structural_params(
                    proof_state, combined_repr, available_hypotheses, param_type_logits
                )
            elif tactic_family == "assumption_family":
                return self._generate_assumption_params(
                    proof_state, combined_repr, param_type_logits
                )
            elif tactic_family == "advanced_rewrite_family":
                return self._generate_advanced_rewrite_params(
                    proof_state,
                    combined_repr,
                    available_terms,
                    knowledge_base,
                    param_type_logits,
                )
            elif tactic_family == "induction_family":
                return self._generate_induction_params(
                    proof_state, combined_repr, available_hypotheses, param_type_logits
                )
            elif tactic_family == "quantifier_family":
                return self._generate_quantifier_params(
                    proof_state, combined_repr, available_terms, param_type_logits
                )
            elif tactic_family == "conversion_family":
                return self._generate_conversion_params(
                    proof_state, combined_repr, available_terms, param_type_logits
                )
            elif tactic_family == "goal_management_family":
                return self._generate_goal_management_params(
                    proof_state, combined_repr, available_hypotheses, param_type_logits
                )
            elif tactic_family == "specialized_family":
                return self._generate_specialized_params(
                    proof_state, combined_repr, available_hypotheses, param_type_logits
                )
            elif tactic_family == "finish_family":
                return self._generate_finish_params(
                    proof_state, combined_repr, param_type_logits
                )
            else:  # Unknown family
                return {
                    "parameter_type_logits": param_type_logits,
                    "generated_parameters": [],
                    "confidence_score": torch.tensor(0.5, device=proof_state.device),
                }
        else:
            # Fallback for unknown families
            return {
                "parameter_type_logits": param_type_logits,
                "generated_parameters": [],
                "confidence_score": torch.tensor(0.5, device=proof_state.device),
            }

    def _generate_no_params(
        self,
        proof_state: torch.Tensor,
        repr: torch.Tensor,
        param_type_logits: torch.Tensor,
        tactic_type: str,
    ) -> Dict[str, Any]:
        """Generate parameters for tactics that typically need no parameters."""
        return {
            "parameter_type_logits": param_type_logits,
            "parameter_type": "none",
            "tactic_type": tactic_type,
            "confidence_score": torch.tensor(1.0, device=proof_state.device),
        }

    def _generate_hypothesis_selection_params(
        self,
        proof_state: torch.Tensor,
        repr: torch.Tensor,
        available_hypotheses: Optional[torch.Tensor],
        param_type_logits: torch.Tensor,
        tactic_type: str,
        additional_term_max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate parameters for tactics that primarily need hypothesis selection."""
        result = {
            "parameter_type_logits": param_type_logits,
            "tactic_type": tactic_type,
        }

        # Select hypothesis if available
        if available_hypotheses is not None:
            hyp_selection = self.hypothesis_selector(
                repr.unsqueeze(1), available_hypotheses
            )
            result["hypothesis_selection"] = hyp_selection

        # Generate additional term if requested
        if additional_term_max_length is not None:
            additional_term = self.term_generator(
                repr, max_length=additional_term_max_length
            )
            if tactic_type == "induction":
                result["pattern_term"] = additional_term
            elif tactic_type == "specialized":
                result["bounds_term"] = additional_term
            elif tactic_type == "goal_management":
                result["new_name"] = additional_term

        return result

    def _generate_term_generation_params(
        self,
        proof_state: torch.Tensor,
        repr: torch.Tensor,
        param_type_logits: torch.Tensor,
        tactic_type: str,
        max_length: int = 20,
        use_specialized_generator: bool = False,
    ) -> Dict[str, Any]:
        """Generate parameters for tactics that primarily need term generation."""
        result = {
            "parameter_type_logits": param_type_logits,
            "tactic_type": tactic_type,
        }

        # Use specialized generator for quantifier tactics
        if use_specialized_generator and tactic_type == "quantifier":
            witness_term = self.use_generator(repr)
            result["witness_term"] = witness_term
        else:
            # Use standard term generator
            generated_term = self.term_generator(repr, max_length=max_length)

            # Set appropriate key based on tactic type
            if tactic_type == "calc":
                result["calculation_steps"] = generated_term
            elif tactic_type == "conversion":
                result["target_expression"] = generated_term
            else:  # proof, etc.
                result["generated_term"] = generated_term

        return result

    def _generate_rewrite_params(
        self,
        proof_state: torch.Tensor,
        repr: torch.Tensor,
        available_terms: Optional[torch.Tensor],
        knowledge_base: Optional[torch.Tensor],
        param_type_logits: torch.Tensor,
        include_simp_lemmas: bool = False,
    ) -> Dict[str, Any]:
        """Generate rewrite sequences using attention over available equations."""
        result = {
            "parameter_type_logits": param_type_logits,
            "tactic_type": "advanced_rewrite" if include_simp_lemmas else "rewrite",
        }

        # Generate rewrite sequence
        rewrite_sequence = self.rewrite_generator(
            repr.unsqueeze(1), proof_state, available_terms
        )
        result["rewrite_sequence"] = rewrite_sequence

        # Add simp lemmas for advanced rewrite tactics
        if include_simp_lemmas:
            simp_lemmas = self.simp_selector(repr.unsqueeze(1), knowledge_base)
            result["simp_lemmas"] = simp_lemmas

        return result

    def _generate_apply_params(
        self,
        proof_state: torch.Tensor,
        repr: torch.Tensor,
        available_terms: Optional[torch.Tensor],
        available_hypotheses: Optional[torch.Tensor],
        knowledge_base: Optional[torch.Tensor],
        param_type_logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """Generate parameters for apply tactic using retrieval and attention."""
        result = {
            "parameter_type_logits": param_type_logits,
            "tactic_type": "apply",
        }

        # Retrieve relevant premises
        if knowledge_base is not None:
            result["retrieved_premises"] = self.premise_retriever(repr, knowledge_base)

        # Select from available terms/hypotheses
        if available_terms is not None:
            result["term_selection"] = self.hypothesis_selector(
                repr.unsqueeze(1), available_terms
            )

        if available_hypotheses is not None:
            result["hypothesis_selection"] = self.hypothesis_selector(
                repr.unsqueeze(1), available_hypotheses
            )

        # Generate new term if needed
        result["generated_term"] = self.term_generator(repr, max_length=20)

        return result

    # Simplified individual functions that call generic functions
    def _generate_intro_params(self, proof_state, repr, param_type_logits):
        return self._generate_no_params(proof_state, repr, param_type_logits, "intro")

    def _generate_automation_params(self, proof_state, repr, param_type_logits):
        return self._generate_no_params(
            proof_state, repr, param_type_logits, "automation"
        )

    def _generate_assumption_params(self, proof_state, repr, param_type_logits):
        return self._generate_no_params(
            proof_state, repr, param_type_logits, "assumption"
        )

    def _generate_finish_params(self, proof_state, repr, param_type_logits):
        return self._generate_no_params(proof_state, repr, param_type_logits, "finish")

    def _generate_case_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        return self._generate_hypothesis_selection_params(
            proof_state, repr, available_hypotheses, param_type_logits, "cases"
        )

    def _generate_structural_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        return self._generate_hypothesis_selection_params(
            proof_state, repr, available_hypotheses, param_type_logits, "structural"
        )

    def _generate_induction_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        return self._generate_hypothesis_selection_params(
            proof_state,
            repr,
            available_hypotheses,
            param_type_logits,
            "induction",
            additional_term_max_length=15,
        )

    def _generate_goal_management_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        return self._generate_hypothesis_selection_params(
            proof_state,
            repr,
            available_hypotheses,
            param_type_logits,
            "goal_management",
            additional_term_max_length=5,
        )

    def _generate_specialized_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        return self._generate_hypothesis_selection_params(
            proof_state,
            repr,
            available_hypotheses,
            param_type_logits,
            "specialized",
            additional_term_max_length=10,
        )

    def _generate_calc_params(
        self, proof_state, repr, available_terms, param_type_logits
    ):
        return self._generate_term_generation_params(
            proof_state, repr, param_type_logits, "calc", max_length=30
        )

    def _generate_proof_params(
        self, proof_state, repr, available_terms, param_type_logits
    ):
        return self._generate_term_generation_params(
            proof_state, repr, param_type_logits, "proof", max_length=25
        )

    def _generate_conversion_params(
        self, proof_state, repr, available_terms, param_type_logits
    ):
        return self._generate_term_generation_params(
            proof_state, repr, param_type_logits, "conversion", max_length=20
        )

    def _generate_quantifier_params(
        self, proof_state, repr, available_terms, param_type_logits
    ):
        return self._generate_term_generation_params(
            proof_state,
            repr,
            param_type_logits,
            "quantifier",
            max_length=10,
            use_specialized_generator=True,
        )

    def _generate_advanced_rewrite_params(
        self, proof_state, repr, available_terms, knowledge_base, param_type_logits
    ):
        return self._generate_rewrite_params(
            proof_state,
            repr,
            available_terms,
            knowledge_base,
            param_type_logits,
            include_simp_lemmas=True,
        )


class PremiseRetriever(nn.Module):
    """Retrieve relevant premises from mathematical knowledge base."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()

        self.d_model = d_model
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.pooling = AttentionPooling(d_model)

        # Relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(
        self, query: torch.Tensor, knowledge_base: torch.Tensor, top_k: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve top-k relevant premises.

        Args:
            query: Query representation [batch_size, d_model]
            knowledge_base: Knowledge base [batch_size, n_premises, d_model]
            top_k: Number of premises to retrieve

        Returns:
            Dictionary with retrieved premises and scores
        """
        batch_size, n_premises, _ = knowledge_base.size()

        # Expand query to match knowledge base
        query_expanded = query.unsqueeze(1).expand(-1, n_premises, -1)

        # Compute relevance scores
        combined = torch.cat([query_expanded, knowledge_base], dim=-1)
        relevance_scores = self.relevance_scorer(combined).squeeze(-1)

        # Get top-k premises
        top_scores, top_indices = torch.topk(
            relevance_scores, min(top_k, n_premises), dim=-1
        )

        # Retrieve top premises
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, top_k)
        retrieved_premises = knowledge_base[batch_indices, top_indices]

        return {
            "premises": retrieved_premises,
            "scores": top_scores,
            "indices": top_indices,
        }


class AutoregressiveTermGenerator(nn.Module):
    """Generate new mathematical terms autoregressively."""

    def __init__(self, vocab_size: int, d_model: int = 512):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(100, d_model))

        # LSTM decoder
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        context: torch.Tensor,
        target_sequence: Optional[torch.Tensor] = None,
        max_length: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate term sequence.

        Args:
            context: Context representation [batch_size, d_model]
            target_sequence: Target sequence for training [batch_size, seq_len]
            max_length: Maximum generation length

        Returns:
            Dictionary with generation outputs
        """
        batch_size = context.size(0)
        device = context.device

        if target_sequence is not None:
            # Training mode with teacher forcing
            seq_len = target_sequence.size(1)

            # Embed target sequence
            embedded = self.token_embedding(target_sequence)
            embedded += self.positional_encoding[:seq_len].unsqueeze(0)

            # Initialize decoder with context
            hidden = context.unsqueeze(0)  # [1, batch_size, d_model]
            cell = torch.zeros_like(hidden)

            # Decode sequence
            outputs, _ = self.decoder(embedded, (hidden, cell))

            # Project to vocabulary
            logits = self.output_projection(outputs)

            return {"logits": logits}
        else:
            # Generation mode
            generated_tokens = []
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            hidden = context.unsqueeze(0)  # [1, batch_size, d_model]
            cell = torch.zeros_like(hidden)

            for step in range(max_length):
                # Embed current token
                embedded = self.token_embedding(current_token)
                embedded += self.positional_encoding[step : step + 1].unsqueeze(0)

                # Decoder step
                output, (hidden, cell) = self.decoder(embedded, (hidden, cell))

                # Get next token
                logits = self.output_projection(output)
                next_token = torch.argmax(logits, dim=-1)

                generated_tokens.append(next_token.squeeze(1))
                current_token = next_token

                # Stop if end token
                if (next_token == 0).all():
                    break

            return {"generated_tokens": torch.stack(generated_tokens, dim=1)}


class HypothesisSelector(nn.Module):
    """Select appropriate hypothesis using attention."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.pooling = AttentionPooling(d_model)

    def forward(
        self, query: torch.Tensor, hypotheses: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Select hypothesis using attention.

        Args:
            query: Query representation [batch_size, 1, d_model]
            hypotheses: Available hypotheses [batch_size, n_hyps, d_model]

        Returns:
            Dictionary with selection results
        """
        # Apply attention
        attended = self.attention(query, hypotheses, hypotheses)

        # Pool to get selection scores
        pooled = self.pooling(attended)

        # Compute selection logits
        scores = torch.matmul(
            pooled.unsqueeze(1), hypotheses.transpose(-2, -1)
        ).squeeze(1)

        return {"selection_logits": scores, "attended_representation": attended}


class RewriteSequenceGenerator(nn.Module):
    """Generate sequence of rewrite rules."""

    def __init__(self, d_model: int, n_heads: int = 8, max_length: int = 10):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.sequence_generator = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        proof_state: torch.Tensor,
        available_terms: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate rewrite sequence."""

        # Use attention to focus on relevant parts
        focused = self.attention(query, proof_state, proof_state)

        # Generate sequence (simplified)
        sequence_repr = focused.mean(dim=1, keepdim=True)

        return {
            "rewrite_representation": sequence_repr,
            "sequence_length": torch.tensor([3]),  # Placeholder
        }


class UseTermGenerator(nn.Module):
    """Generate witness terms for existential goals."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.term_generator = AutoregressiveTermGenerator(vocab_size, d_model)

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate witness term."""
        return self.term_generator(context, max_length=10)


class SimpLemmaSelector(nn.Module):
    """Select appropriate simp lemmas."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)

    def forward(
        self, query: torch.Tensor, available_lemmas: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Select simp lemmas."""

        if available_lemmas is not None:
            attended = self.attention(query, available_lemmas, available_lemmas)
            return {"selected_lemmas": attended}
        else:
            return {"selected_lemmas": query}


class ParameterValidation:
    """Utility class for validating generated parameters."""

    @staticmethod
    def validate_term_syntax(term_tokens: List[str]) -> bool:
        """Check if generated term has valid Lean syntax."""
        # Basic syntax validation
        if not term_tokens:
            return False

        # Check for balanced parentheses
        paren_count = 0
        for token in term_tokens:
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                if paren_count < 0:
                    return False

        return paren_count == 0

    @staticmethod
    def validate_hypothesis_reference(
        hyp_name: str, available_hypotheses: List[str]
    ) -> bool:
        """Check if hypothesis reference is valid."""
        return hyp_name in available_hypotheses

    @staticmethod
    def estimate_parameter_confidence(
        attention_scores: torch.Tensor, generation_scores: torch.Tensor
    ) -> float:
        """Estimate confidence in generated parameters."""
        # Combine attention entropy and generation probability
        attention_entropy = -torch.sum(
            attention_scores * torch.log(attention_scores + 1e-9)
        )
        max_gen_prob = torch.max(generation_scores)

        # Lower entropy and higher max probability = higher confidence
        confidence = float(max_gen_prob / (1 + attention_entropy))
        return min(confidence, 1.0)


class ParameterPostProcessor:
    """Post-process generated parameters for Lean compatibility."""

    def __init__(self, vocab_to_token: Dict[int, str]):
        self.vocab_to_token = vocab_to_token

    def tokens_to_string(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs to string."""
        tokens = [self.vocab_to_token.get(int(tid), "<UNK>") for tid in token_ids]
        return " ".join(tokens).strip()

    def clean_generated_term(self, raw_term: str) -> str:
        """Clean and format generated term."""
        # Remove special tokens and clean up spacing
        cleaned = raw_term.replace("<UNK>", "").replace("<PAD>", "")
        cleaned = " ".join(cleaned.split())  # Normalize whitespace
        return cleaned

    def format_tactic_with_params(self, tactic_name: str, parameters: List[str]) -> str:
        """Format complete tactic with parameters."""
        if not parameters:
            return tactic_name

        # Different tactics have different parameter formats
        if tactic_name in ["apply", "exact", "refine"]:
            return f"{tactic_name} {parameters[0]}"
        elif tactic_name in ["rw", "simp"]:
            param_list = ", ".join(parameters)
            return f"{tactic_name} [{param_list}]"
        elif tactic_name == "cases":
            if len(parameters) >= 2:
                return f"{tactic_name} {parameters[0]} with {parameters[1]}"
            else:
                return f"{tactic_name} {parameters[0]}"
        else:
            return f"{tactic_name} {' '.join(parameters)}"


class BeamSearchGenerator:
    """Beam search for better parameter generation."""

    def __init__(self, beam_size: int = 5):
        self.beam_size = beam_size

    def beam_search_decode(
        self,
        model: nn.Module,
        initial_context: torch.Tensor,
        max_length: int = 20,
        vocab_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple parameter candidates using beam search.

        Args:
            model: The autoregressive generator
            initial_context: Initial context tensor
            max_length: Maximum sequence length
            vocab_size: Vocabulary size

        Returns:
            List of candidate sequences with scores
        """
        batch_size = initial_context.size(0)
        device = initial_context.device

        # Initialize beam
        sequences = torch.zeros(
            batch_size, self.beam_size, 1, dtype=torch.long, device=device
        )
        scores = torch.zeros(batch_size, self.beam_size, device=device)

        # Beam search decoding
        for step in range(max_length):
            if step == 0:
                # First step: use initial context
                with torch.no_grad():
                    output = model(initial_context)
                    if "logits" in output:
                        logits = output["logits"][:, -1, :]  # Last token logits
                        log_probs = F.log_softmax(logits, dim=-1)

                        # Get top-k tokens for first step
                        top_scores, top_indices = torch.topk(
                            log_probs, self.beam_size, dim=-1
                        )

                        for i in range(batch_size):
                            for j in range(self.beam_size):
                                sequences[i, j, 0] = top_indices[i, j]
                                scores[i, j] = top_scores[i, j]
            else:
                # Subsequent steps: expand each beam
                all_candidates = []

                for beam_idx in range(self.beam_size):
                    current_seq = sequences[:, beam_idx, : step + 1]
                    current_score = scores[:, beam_idx]

                    # Generate next token probabilities
                    with torch.no_grad():
                        # This is simplified - in practice, you'd need to
                        # properly format the input for the model
                        dummy_output = torch.randn(
                            batch_size, vocab_size, device=device
                        )
                        log_probs = F.log_softmax(dummy_output, dim=-1)

                        # Get top-k next tokens
                        top_scores, top_indices = torch.topk(
                            log_probs, self.beam_size, dim=-1
                        )

                        for i in range(batch_size):
                            for k in range(self.beam_size):
                                new_seq = torch.cat(
                                    [
                                        current_seq[i : i + 1],
                                        top_indices[i : i + 1, k : k + 1],
                                    ],
                                    dim=-1,
                                )
                                new_score = current_score[i] + top_scores[i, k]
                                all_candidates.append((new_seq, new_score, i))

                # Select top beam_size candidates for each batch element
                for batch_idx in range(batch_size):
                    batch_candidates = [
                        (seq, score)
                        for seq, score, b_idx in all_candidates
                        if b_idx == batch_idx
                    ]
                    batch_candidates.sort(key=lambda x: x[1], reverse=True)

                    for beam_idx in range(min(self.beam_size, len(batch_candidates))):
                        seq, score = batch_candidates[beam_idx]
                        sequences[batch_idx, beam_idx, : step + 2] = seq
                        scores[batch_idx, beam_idx] = score

        # Convert to list of dictionaries
        results = []
        for batch_idx in range(batch_size):
            batch_results = []
            for beam_idx in range(self.beam_size):
                batch_results.append(
                    {
                        "sequence": sequences[batch_idx, beam_idx],
                        "score": float(scores[batch_idx, beam_idx]),
                    }
                )
            results.append(batch_results)

        return results


def create_parameter_generator(
    vocab_size: int, d_model: int = 512, n_heads: int = 8, **kwargs
) -> TacticParameterGenerator:
    """Factory function to create parameter generator with standard configuration."""
    return TacticParameterGenerator(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, **kwargs
    )
