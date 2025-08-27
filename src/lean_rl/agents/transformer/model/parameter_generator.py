"""
Tactic parameter generation networks.

This module implements specialized networks for generating parameters for
different Lean tactics, including retrieval-augmented generation and
autoregressive term generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List
from enum import Enum

import torch.nn.functional as F
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
        Compute a comprehensive training loss for parameter generation.
        This function handles various types of outputs: classification,
        generation (autoregressive), and selection (attention-based).

        Args:
            outputs: A dictionary of model outputs from the forward pass.
            targets: A dictionary of target labels and values.

        Returns:
            A combined, weighted loss tensor.
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        loss_fns = {
            "param_type": F.cross_entropy,
            "generation": F.cross_entropy,
            "selection": F.cross_entropy,
            "rewrite": lambda log_probs, indices: F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)), indices.reshape(-1)
            ),
        }

        # 1. Parameter Type Classification Loss
        if "parameter_type_logits" in outputs and "target_param_type" in targets:
            param_type_loss = loss_fns["param_type"](
                outputs["parameter_type_logits"], targets["target_param_type"]
            )
            total_loss += self.param_type_weight * param_type_loss

        # 2. Autoregressive Term Generation Loss
        if "generated_term" in outputs and "target_term" in targets:
            gen_output = outputs["generated_term"]
            if "logits" in gen_output:
                logits = gen_output["logits"]
                target = targets["target_term"]
                # Align dimensions: [batch, seq_len, vocab] vs [batch, seq_len]
                generation_loss = loss_fns["generation"](
                    logits.view(-1, self.vocab_size), target.view(-1)
                )
                total_loss += self.generation_weight * generation_loss

        # 3. Hypothesis Selection Loss
        if "hypothesis_selection" in outputs and "target_hypothesis" in targets:
            hyp_output = outputs["hypothesis_selection"]
            if "selection_logits" in hyp_output:
                selection_loss = loss_fns["selection"](
                    hyp_output["selection_logits"], targets["target_hypothesis"]
                )
                total_loss += self.selection_weight * selection_loss

        # 4. Rewrite Sequence Loss
        if "rewrite_sequence" in outputs and "target_rewrite_indices" in targets:
            rewrite_output = outputs["rewrite_sequence"]
            if "rewrite_log_probs" in rewrite_output:
                log_probs = rewrite_output["rewrite_log_probs"]
                target_indices = targets["target_rewrite_indices"]
                rewrite_loss = loss_fns["rewrite"](log_probs, target_indices)
                total_loss += (
                    self.selection_weight * rewrite_loss
                )  # Reuse selection weight

        # 5. Premise Retrieval Loss (if applicable, e.g., using contrastive loss)
        if "retrieved_premises" in outputs and "target_premises" in targets:
            # This would require a more complex loss like contrastive loss,
            # which is beyond a simple replacement here. For now, we skip this.
            pass

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
        if proof_state.dim() == 2:
            # Add a sequence length dimension for single states
            proof_state = proof_state.unsqueeze(1)

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
        rewrite_output = self.rewrite_generator(
            repr.unsqueeze(1), proof_state, available_terms
        )
        result["rewrite_sequence"] = rewrite_output

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

    def _generate_intro_params(self, proof_state, repr, param_type_logits):
        """Handles tactics like 'intro', which typically require no complex parameters."""
        return self._generate_no_params(proof_state, repr, param_type_logits, "intro")

    def _generate_automation_params(self, proof_state, repr, param_type_logits):
        """Handles automation tactics that require no parameters."""
        return self._generate_no_params(
            proof_state, repr, param_type_logits, "automation"
        )

    def _generate_assumption_params(self, proof_state, repr, param_type_logits):
        """Handles assumption tactics that require no parameters."""
        return self._generate_no_params(
            proof_state, repr, param_type_logits, "assumption"
        )

    def _generate_finish_params(self, proof_state, repr, param_type_logits):
        """Handles finishing tactics that require no parameters."""
        return self._generate_no_params(proof_state, repr, param_type_logits, "finish")

    def _generate_case_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        """Handles case-analysis tactics by selecting a hypothesis."""
        return self._generate_hypothesis_selection_params(
            proof_state, repr, available_hypotheses, param_type_logits, "cases"
        )

    def _generate_structural_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        """Handles structural tactics by selecting a hypothesis."""
        return self._generate_hypothesis_selection_params(
            proof_state, repr, available_hypotheses, param_type_logits, "structural"
        )

    def _generate_induction_params(
        self, proof_state, repr, available_hypotheses, param_type_logits
    ):
        """Handles induction by selecting a hypothesis and generating a pattern."""
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
        """Handles goal management by selecting a hypothesis and generating a new name."""
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
        """Handles specialized tactics by selecting a hypothesis and generating bounds."""
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
        """Handles calculation tactics by generating calculation steps."""
        return self._generate_term_generation_params(
            proof_state, repr, param_type_logits, "calc", max_length=30
        )

    def _generate_proof_params(
        self, proof_state, repr, available_terms, param_type_logits
    ):
        """Handles proof construction tactics by generating a proof term."""
        return self._generate_term_generation_params(
            proof_state, repr, param_type_logits, "proof", max_length=25
        )

    def _generate_conversion_params(
        self, proof_state, repr, available_terms, param_type_logits
    ):
        """Handles conversion tactics by generating a target expression."""
        return self._generate_term_generation_params(
            proof_state, repr, param_type_logits, "conversion", max_length=20
        )

    def _generate_quantifier_params(
        self, proof_state, repr, available_terms, param_type_logits
    ):
        """Handles quantifier tactics by generating a witness term."""
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
        """Handles advanced rewrite tactics by generating a rewrite sequence and selecting simp lemmas."""
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
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        proof_state: torch.Tensor,
        available_terms: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate rewrite sequence."""
        batch_size = query.size(0)
        device = query.device

        # Use attention to focus on relevant parts of the proof state
        focused_context = self.attention(query, proof_state, proof_state).mean(dim=1)

        # Initialize LSTM state
        hidden = focused_context.unsqueeze(0)
        cell = torch.zeros_like(hidden)

        # Start token for generation (e.g., a zero vector)
        current_input = torch.zeros(batch_size, 1, self.d_model, device=device)

        generated_sequence_indices = []
        generated_sequence_log_probs = []

        if available_terms is None or available_terms.shape[1] == 0:
            return {
                "rewrite_indices": torch.tensor([], dtype=torch.long, device=device),
                "rewrite_log_probs": torch.tensor([], device=device),
                "sequence_length": torch.tensor([0] * batch_size, device=device),
            }

        for _ in range(self.max_length):
            output, (hidden, cell) = self.sequence_generator(
                current_input, (hidden, cell)
            )

            # Project output to match term embedding dimension
            projected_output = self.output_projection(output.squeeze(1))

            # Compute similarity with available terms to get logits
            logits = torch.bmm(available_terms, projected_output.unsqueeze(-1)).squeeze(
                -1
            )
            log_probs = F.log_softmax(logits, dim=-1)

            # Sample the next term
            next_term_index = torch.multinomial(log_probs.exp(), 1)

            # Get the log probability of the chosen term
            chosen_log_prob = log_probs.gather(1, next_term_index)

            generated_sequence_indices.append(next_term_index)
            generated_sequence_log_probs.append(chosen_log_prob)

            # Prepare next input: the embedding of the chosen term
            current_input = torch.gather(
                available_terms,
                1,
                next_term_index.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        indices = torch.cat(generated_sequence_indices, dim=1)
        log_probs = torch.cat(generated_sequence_log_probs, dim=1)

        return {
            "rewrite_indices": indices,
            "rewrite_log_probs": log_probs,
            "sequence_length": torch.tensor(
                [self.max_length] * batch_size, device=device
            ),
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
        model: "AutoregressiveTermGenerator",
        initial_context: torch.Tensor,
        max_length: int = 20,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate multiple parameter candidates using beam search.
        Args:
            model: The autoregressive generator model.
            initial_context: Initial context tensor for the decoder.
            max_length: Maximum generation length.
        Returns:
            A list of lists of candidate sequences and their scores for each item in the batch.
        """
        batch_size = initial_context.size(0)
        device = initial_context.device

        # Initialize beams for each item in the batch
        beams = [
            [([], 0.0)] for _ in range(batch_size)
        ]  # Each beam is a list of (sequence, score)
        hidden = initial_context.unsqueeze(0)
        cell = torch.zeros_like(hidden)

        for _ in range(max_length):
            all_candidates = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for seq, score in beams[i]:
                    if not seq or seq[-1] != 0:  # 0 is EOS token
                        current_token = (
                            torch.tensor([seq[-1]], device=device)
                            if seq
                            else torch.zeros(1, dtype=torch.long, device=device)
                        )
                        embedded = (
                            model.token_embedding(current_token.unsqueeze(0))
                            + model.positional_encoding[len(seq) : len(seq) + 1]
                        )
                        output, (h, c) = model.decoder(
                            embedded,
                            (
                                hidden[:, i : i + 1, :].contiguous(),
                                cell[:, i : i + 1, :].contiguous(),
                            ),
                        )
                        logits = model.output_projection(output.squeeze(0))
                        log_probs = F.log_softmax(logits, dim=-1)
                        top_log_probs, top_indices = log_probs.topk(
                            self.beam_size, dim=-1
                        )

                        for j in range(self.beam_size):
                            new_seq = seq + [top_indices[0, j].item()]
                            new_score = score + top_log_probs[0, j].item()
                            all_candidates[i].append((new_seq, new_score))

            # Update beams with the best candidates
            for i in range(batch_size):
                if all_candidates[i]:
                    sorted_candidates = sorted(
                        all_candidates[i], key=lambda x: x[1], reverse=True
                    )
                    beams[i] = sorted_candidates[: self.beam_size]

        # Prepare results
        results = []
        for i in range(batch_size):
            batch_results = [
                {"sequence": torch.tensor(seq), "score": score}
                for seq, score in beams[i]
            ]
            results.append(batch_results)

        return results


def create_parameter_generator(
    vocab_size: int, d_model: int = 512, n_heads: int = 8, **kwargs
) -> TacticParameterGenerator:
    """Factory function to create parameter generator with standard configuration."""
    return TacticParameterGenerator(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, **kwargs
    )
