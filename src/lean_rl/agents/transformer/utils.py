"""
Utility functions and classes for transformer-based theorem proving.

This module provides tokenization, encoding, and other utility functions
for processing Lean proof states and tactics.
"""

import torch
from typing import Dict, List, Tuple
import re
from dataclasses import dataclass


@dataclass
class ProofState:
    """Structured representation of a Lean proof state."""

    goals: List[str]
    hypotheses: List[str]
    context: str
    raw_text: str
    goal_positions: List[Tuple[int, int]]  # Start, end positions in tokenized text
    hypothesis_positions: List[Tuple[int, int]]


@dataclass
class TacticInfo:
    """Information about a tactic and its parameters."""

    name: str
    family: str
    parameters: List[str]
    parameter_types: List[str]
    confidence: float


class ProofStateTokenizer:
    """
    Tokenizer specialized for Lean proof states.

    This tokenizer handles mathematical notation, goal structure,
    and hypothesis formatting in Lean proof states.
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size

        # Special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<goal>": 2,
            "<hyp>": 3,
            "<type>": 4,
            "<proof>": 5,
            "<end>": 6,
            "⊢": 7,  # Goal turnstile
            "∀": 8,  # Forall quantifier
            "∃": 9,  # Exists quantifier
            "→": 10,  # Implication
            "∧": 11,  # And
            "∨": 12,  # Or
            "¬": 13,  # Not
            "=": 14,  # Equality
        }

        # Mathematical operators and symbols
        self.math_symbols = {
            "+",
            "-",
            "*",
            "/",
            "^",
            "_",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "∈",
            "∉",
            "⊆",
            "⊇",
            "∩",
            "∪",
            "∅",
            "ℕ",
            "ℤ",
            "ℚ",
            "ℝ",
            "ℂ",
            "≤",
            "≥",
            "<",
            ">",
            "≠",
            "≈",
            "≡",
            "∼",
            "∝",
            "∞",
            "α",
            "β",
            "γ",
            "δ",
            "ε",
            "ζ",
            "η",
            "θ",
            "ι",
            "κ",
            "λ",
            "μ",
            "ν",
            "ξ",
            "ο",
            "π",
            "ρ",
            "σ",
            "τ",
            "υ",
            "φ",
            "χ",
            "ψ",
            "ω",
        }

        # Common Lean tactics and keywords
        self.lean_keywords = {
            "apply",
            "exact",
            "rw",
            "simp",
            "intro",
            "intros",
            "cases",
            "induction",
            "sorry",
            "done",
            "have",
            "show",
            "calc",
            "conv",
            "unfold",
            "fold",
            "change",
            "convert",
            "congr",
            "ext",
            "funext",
            "split",
            "left",
            "right",
            "constructor",
            "use",
            "refine",
            "rintro",
            "rcases",
            "obtain",
            "by_contra",
            "exfalso",
            "trivial",
            "rwa",
            "rfl",
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
            "suffices",
            "assert",
            "assumption",
            "simp_all",
            "hint",
            "simp_rw",
            "rw_mod_cast",
            "simp_intro",
            "field_simp",
            "conv_lhs",
            "conv_rhs",
            "induction'",
            "cases'",
            "choose",
            "exists",
            "use!",
            "existsi",
            "forall_intro",
            "show_term",
            "swap",
            "rotate_left",
            "rotate_right",
            "clear",
            "rename",
            "set",
            "interval_cases",
            "fin_cases",
            "mod_cases",
            "lift",
            "push_neg",
            "by_cases",
            "contradiction",
            "trans",
            "symm",
        }

        # Build vocabulary
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        # Add common tokens
        current_id = len(self.special_tokens)
        for symbol in self.math_symbols:
            if symbol not in self.token_to_id:
                self.token_to_id[symbol] = current_id
                self.id_to_token[current_id] = symbol
                current_id += 1

        for keyword in self.lean_keywords:
            if keyword not in self.token_to_id:
                self.token_to_id[keyword] = current_id
                self.id_to_token[current_id] = keyword
                current_id += 1

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Lean proof state text.

        Args:
            text: Raw proof state text

        Returns:
            List of tokens
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Split on special mathematical symbols and operators
        # This is a simplified tokenization - a real implementation would be more sophisticated
        tokens = []
        current_token = ""

        for char in text:
            if char in self.math_symbols or char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if not char.isspace():
                    tokens.append(char)
            else:
                current_token += char

        if current_token:
            tokens.append(current_token)

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        token_ids = []

        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id["<unk>"])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append("<unk>")

        return " ".join(tokens)

    def parse_proof_state(self, text: str) -> ProofState:
        """
        Parse proof state text into structured format.

        Args:
            text: Raw proof state text

        Returns:
            Structured ProofState object
        """
        # This is a simplified parser - real implementation would be more robust
        lines = text.strip().split("\n")

        goals = []
        hypotheses = []
        context = ""
        goal_positions = []
        hypothesis_positions = []

        current_position = 0
        in_goal = False
        in_hypothesis = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for goal marker
            if "⊢" in line:
                goals.append(line)
                goal_start = current_position
                goal_end = current_position + len(self.encode(line))
                goal_positions.append((goal_start, goal_end))
                current_position = goal_end
                in_goal = True
                in_hypothesis = False

            # Check for hypothesis (simple heuristic)
            elif ":" in line and not in_goal:
                hypotheses.append(line)
                hyp_start = current_position
                hyp_end = current_position + len(self.encode(line))
                hypothesis_positions.append((hyp_start, hyp_end))
                current_position = hyp_end
                in_hypothesis = True

            else:
                context += line + " "
                current_position += len(self.encode(line))

        return ProofState(
            goals=goals,
            hypotheses=hypotheses,
            context=context.strip(),
            raw_text=text,
            goal_positions=goal_positions,
            hypothesis_positions=hypothesis_positions,
        )

    def create_masks(
        self, proof_state: ProofState, token_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create attention masks for goals and hypotheses.

        Args:
            proof_state: Parsed proof state
            token_ids: Tokenized proof state

        Returns:
            Tuple of (goal_mask, hypothesis_mask)
        """
        seq_len = len(token_ids)
        goal_mask = torch.zeros(seq_len, dtype=torch.bool)
        hypothesis_mask = torch.zeros(seq_len, dtype=torch.bool)

        # Mark goal positions
        for start, end in proof_state.goal_positions:
            start = min(start, seq_len - 1)
            end = min(end, seq_len)
            goal_mask[start:end] = True

        # Mark hypothesis positions
        for start, end in proof_state.hypothesis_positions:
            start = min(start, seq_len - 1)
            end = min(end, seq_len)
            hypothesis_mask[start:end] = True

        return goal_mask, hypothesis_mask


class TacticEncoder:
    """
    Encoder for Lean tactics and their parameters.

    This class handles encoding tactics into structured representations
    that can be used by the hierarchical policy networks.
    """

    def __init__(self):
        # Tactic families mapping
        self.tactic_families = {
            # Apply family
            "apply": "apply_family",
            "exact": "apply_family",
            "refine": "apply_family",
            "use": "apply_family",
            # Rewrite family
            "rw": "rewrite_family",
            "simp": "rewrite_family",
            "conv": "rewrite_family",
            "rwa": "rewrite_family",
            # Introduction family
            "intro": "intro_family",
            "intros": "intro_family",
            "rintro": "intro_family",
            # Case analysis family
            "cases": "case_family",
            "rcases": "case_family",
            "induction": "case_family",
            "split": "case_family",
            # Calculation family
            "calc": "calc_family",
            "trans": "calc_family",
            "symm": "calc_family",
            # Finishing family
            "sorry": "finish_family",
            "done": "finish_family",
            "rfl": "finish_family",
            "trivial": "finish_family",
            # Automation family
            "aesop": "automation_family",
            "tauto": "automation_family",
            "ring": "automation_family",
            "norm_num": "automation_family",
            "linarith": "automation_family",
            "nlinarith": "automation_family",
            "omega": "automation_family",
            "abel": "automation_family",
            "polyrith": "automation_family",
            "decide": "automation_family",
            "norm_cast": "automation_family",
            # Proof family
            "have": "proof_family",
            "show": "proof_family",
            "suffices": "proof_family",
            "assert": "proof_family",
            # Structural family
            "constructor": "structural_family",
            "left": "structural_family",
            "right": "structural_family",
            "ext": "structural_family",
            "exfalso": "structural_family",
            "by_contra": "structural_family",
            "contradiction": "structural_family",
            "by_cases": "structural_family",
            # Assumption family
            "assumption": "assumption_family",
            "simp_all": "assumption_family",
            "hint": "assumption_family",
            # Advanced rewrite family
            "simp_rw": "advanced_rewrite_family",
            "rw_mod_cast": "advanced_rewrite_family",
            "simp_intro": "advanced_rewrite_family",
            "field_simp": "advanced_rewrite_family",
            "conv_lhs": "advanced_rewrite_family",
            "conv_rhs": "advanced_rewrite_family",
            # Induction family
            "induction'": "induction_family",
            "cases'": "induction_family",
            "obtain": "induction_family",
            "choose": "induction_family",
            # Quantifier family
            "exists": "quantifier_family",
            "use!": "quantifier_family",
            "existsi": "quantifier_family",
            "forall_intro": "quantifier_family",
            # Conversion family
            "change": "conversion_family",
            "convert": "conversion_family",
            "congr": "conversion_family",
            "show_term": "conversion_family",
            # Goal management family
            "swap": "goal_management_family",
            "rotate_left": "goal_management_family",
            "rotate_right": "goal_management_family",
            "clear": "goal_management_family",
            "rename": "goal_management_family",
            "set": "goal_management_family",
            # Specialized family
            "interval_cases": "specialized_family",
            "fin_cases": "specialized_family",
            "mod_cases": "specialized_family",
            "lift": "specialized_family",
            "push_neg": "specialized_family",
        }

        # Strategic actions mapping
        self.strategic_mapping = {
            "induction": ["induction", "induction'"],
            "contradiction": ["by_contra", "exfalso", "contradiction"],
            "case_analysis": ["cases", "rcases", "split", "cases'", "by_cases"],
            "direct_proof": ["apply", "exact", "show", "use"],
            "rewrite_simplify": ["rw", "simp", "conv", "rwa", "simp_rw", "field_simp"],
            "apply_lemma": ["apply", "exact", "refine", "use"],
            "unfold_definition": ["unfold", "change", "convert"],
            "automation": [
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
            "structural": ["constructor", "left", "right", "ext"],
            "quantifier_instantiation": [
                "exists",
                "use!",
                "existsi",
                "obtain",
                "choose",
            ],
            "goal_management": [
                "swap",
                "rotate_left",
                "rotate_right",
                "clear",
                "rename",
                "set",
            ],
            "specialized_tactics": [
                "interval_cases",
                "fin_cases",
                "mod_cases",
                "lift",
                "push_neg",
            ],
        }

    def parse_tactic(self, tactic_string: str) -> TacticInfo:
        """
        Parse a tactic string into structured information.

        Args:
            tactic_string: Raw tactic string (e.g., "apply h", "rw [eq1, eq2]")

        Returns:
            TacticInfo object with parsed information
        """
        # Simple parsing - real implementation would be more sophisticated
        parts = tactic_string.strip().split()
        if not parts:
            return TacticInfo("", "", [], [], 0.0)

        tactic_name = parts[0]
        parameters = []
        parameter_types = []

        # Extract parameters (simplified)
        if len(parts) > 1:
            param_text = " ".join(parts[1:])

            # Handle bracket notation for rewrite tactics
            if "[" in param_text and "]" in param_text:
                bracket_content = re.findall(r"\[(.*?)\]", param_text)
                if bracket_content:
                    params = bracket_content[0].split(",")
                    parameters = [p.strip() for p in params]
                    parameter_types = ["term"] * len(parameters)
            else:
                # Simple space-separated parameters
                parameters = parts[1:]
                parameter_types = ["term"] * len(parameters)

        # Determine tactic family
        family = self.tactic_families.get(tactic_name, "unknown_family")

        return TacticInfo(
            name=tactic_name,
            family=family,
            parameters=parameters,
            parameter_types=parameter_types,
            confidence=1.0,
        )

    def get_strategic_action(self, tactic_name: str) -> str:
        """
        Map tactic to strategic action.

        Args:
            tactic_name: Name of the tactic

        Returns:
            Strategic action string
        """
        for strategic_action, tactics in self.strategic_mapping.items():
            if tactic_name in tactics:
                return strategic_action

        return "direct_proof"  # Default


class StateEncoder:
    """Utility class for encoding various state representations."""

    @staticmethod
    def encode_proof_state_tensor(
        proof_state: ProofState, tokenizer: ProofStateTokenizer, max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Encode proof state as tensors for the transformer.

        Args:
            proof_state: Parsed proof state
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length

        Returns:
            Dictionary with encoded tensors
        """
        # Tokenize the full text
        token_ids = tokenizer.encode(proof_state.raw_text)

        # Truncate or pad to max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend(
                [tokenizer.token_to_id["<pad>"]] * (max_length - len(token_ids))
            )

        # Create masks
        goal_mask, hypothesis_mask = tokenizer.create_masks(proof_state, token_ids)

        # Pad masks to max_length
        if len(goal_mask) < max_length:
            goal_mask = torch.cat(
                [goal_mask, torch.zeros(max_length - len(goal_mask), dtype=torch.bool)]
            )
        else:
            goal_mask = goal_mask[:max_length]

        if len(hypothesis_mask) < max_length:
            hypothesis_mask = torch.cat(
                [
                    hypothesis_mask,
                    torch.zeros(max_length - len(hypothesis_mask), dtype=torch.bool),
                ]
            )
        else:
            hypothesis_mask = hypothesis_mask[:max_length]

        # Create attention mask (non-padding positions)
        attention_mask = torch.tensor(
            [
                1 if token_id != tokenizer.token_to_id["<pad>"] else 0
                for token_id in token_ids
            ],
            dtype=torch.bool,
        )

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "goal_mask": goal_mask,
            "hypothesis_mask": hypothesis_mask,
        }


# Create global instances
default_tokenizer = ProofStateTokenizer()
default_tactic_encoder = TacticEncoder()
