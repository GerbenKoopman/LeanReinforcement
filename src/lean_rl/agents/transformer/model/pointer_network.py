"""
Pointer networks for tactic and parameter selection.

This module implements pointer networks inspired by attention-learn-to-route
for selecting tactics and parameters with attention over proof states and
available options.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math

from .attention import MultiHeadAttention


class Attention(nn.Module):
    """
    Efficient attention mechanism using PyTorch's optimized scaled_dot_product_attention.
    """

    def __init__(self, hidden_dim: int, use_tanh: bool = True, C: float = 10.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.C = C

        # Project query, key, and value to same dimension
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> torch.Tensor:
        """
        Compute attention scores or full attention output.

        Args:
            query: Query tensor [batch_size, hidden_dim]
            key: Key tensor [batch_size, seq_len, hidden_dim]
            value: Value tensor [batch_size, seq_len, hidden_dim]. If None, uses key.
            return_attention: If True, returns attention weights

        Returns:
            If return_attention=True: Attention logits [batch_size, seq_len]
            If return_attention=False: Attention output [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = key.size()

        if value is None:
            value = key

        # Project query, key, and value
        q = self.q_proj(query).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        k = self.k_proj(key)  # [batch_size, seq_len, hidden_dim]
        v = self.v_proj(value)  # [batch_size, seq_len, hidden_dim]

        if return_attention:
            # For attention weights, we need to compute manually
            scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(hidden_dim)
            if self.use_tanh:
                scores = self.C * torch.tanh(scores)
            return scores.squeeze(1)  # [batch_size, seq_len]
        else:
            # Use optimized scaled dot-product attention for output
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
            return attn_output.squeeze(1)  # [batch_size, hidden_dim]

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method to get attention scores.

        Args:
            query: Query tensor [batch_size, hidden_dim]
            key: Key tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Attention logits [batch_size, seq_len]
        """
        return self.forward(query, key, return_attention=True)


class GlimpseAttention(nn.Module):
    """Glimpse attention for focusing on relevant parts of the input."""

    def __init__(self, hidden_dim: int, n_glimpses: int = 1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses

        self.attention = Attention(hidden_dim, use_tanh=False)

    def forward(self, query: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Apply glimpse attention.

        Args:
            query: Query tensor [batch_size, hidden_dim]
            ref: Reference tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Glimpsed representation [batch_size, hidden_dim]
        """
        current_query = query

        for _ in range(self.n_glimpses):
            # Compute attention scores
            scores = self.attention(current_query, ref)
            weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]

            # Apply attention weights using efficient einsum
            current_query = torch.einsum("bs,bsh->bh", weights, ref)

        return current_query


class TacticPointerNetwork(nn.Module):
    """
    Pointer network for selecting tactics with attention over proof state.

    This network uses pointer attention to select from available tactics
    based on the current proof state, implementing hierarchical selection
    at different abstraction levels.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_glimpses: int = 2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Tactic encoder for processing available tactics
        self.tactic_encoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        # Pointer attention mechanism
        self.pointer_attention = Attention(hidden_dim, use_tanh=True, C=10.0)
        self.glimpse_attention = GlimpseAttention(hidden_dim, n_glimpses)

        # Multi-head attention for context processing
        self.context_attention = MultiHeadAttention(hidden_dim, n_heads)

        # Hierarchy-specific projection heads
        self.strategic_projection = nn.Linear(hidden_dim, hidden_dim)
        self.tactical_projection = nn.Linear(hidden_dim, hidden_dim)
        self.execution_projection = nn.Linear(hidden_dim, hidden_dim)

        # Value estimation head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # For bidirectional LSTM, double the hidden dimension
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        proof_encoding: torch.Tensor,
        available_tactics: torch.Tensor,
        hierarchy_level: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Select tactic using pointer attention over available tactics.

        Args:
            proof_encoding: Encoded proof state [batch_size, seq_len, hidden_dim]
            available_tactics: Available tactic embeddings [batch_size, n_tactics, embedding_dim]
            hierarchy_level: 1 (strategic), 2 (tactical), 3 (execution)
            mask: Optional mask for available tactics [batch_size, n_tactics]

        Returns:
            Dictionary containing:
            - 'pointer_logits': Selection logits over available tactics
            - 'value': Value estimate for current state
            - 'attention_weights': Attention weights for interpretability
        """
        batch_size, seq_len, _ = proof_encoding.size()

        # Encode available tactics
        tactic_encoded, (h_n, c_n) = self.tactic_encoder(available_tactics)
        tactic_encoded = self.output_projection(tactic_encoded)

        # Get context-aware proof representation
        proof_context = self.context_attention(
            proof_encoding.mean(dim=1, keepdim=True), proof_encoding, proof_encoding
        ).squeeze(1)

        # Apply hierarchy-specific projection
        if hierarchy_level == 1:  # Strategic
            query = self.strategic_projection(proof_context)
        elif hierarchy_level == 2:  # Tactical
            query = self.tactical_projection(proof_context)
        else:  # Execution
            query = self.execution_projection(proof_context)

        # Apply glimpse mechanism for focusing
        focused_query = self.glimpse_attention(query, tactic_encoded)

        # Compute pointer attention scores
        pointer_logits = self.pointer_attention(focused_query, tactic_encoded)

        # Apply mask if provided
        if mask is not None:
            pointer_logits = pointer_logits.masked_fill(mask == 0, -1e9)

        # Compute value estimate
        value = self.value_head(focused_query)

        # Get attention weights for interpretability
        attention_weights = F.softmax(pointer_logits, dim=-1)

        return {
            "pointer_logits": pointer_logits,
            "value": value,
            "attention_weights": attention_weights,
            "focused_query": focused_query,
        }


class ParameterPointerNetwork(nn.Module):
    """
    Pointer network for parameter selection and generation.

    This network handles parameter selection for specific tactics,
    including term selection, hypothesis selection, and new term generation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        max_decode_len: int = 20,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_decode_len = max_decode_len

        # Embeddings for tokens and parameters
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.parameter_type_embedding = nn.Embedding(
            4, embedding_dim
        )  # term, hyp, none, auto

        # Encoder for parameter context
        self.parameter_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Decoder for parameter generation
        self.parameter_decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Attention mechanisms
        self.selection_attention = Attention(hidden_dim)
        self.copy_attention = Attention(hidden_dim)

        # Output heads
        self.vocab_head = nn.Linear(hidden_dim, vocab_size)
        self.copy_head = nn.Linear(hidden_dim, 1)

        # Parameter type classification
        self.param_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # term, hypothesis, none, auto
        )

    def forward(
        self,
        proof_encoding: torch.Tensor,
        tactic_family: str,
        available_terms: Optional[List[torch.Tensor]] = None,
        target_parameters: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate or select parameters for a specific tactic.

        Args:
            proof_encoding: Encoded proof state [batch_size, seq_len, hidden_dim]
            tactic_family: Family of the target tactic
            available_terms: Available terms for selection
            target_parameters: Target parameters for training [batch_size, param_len]

        Returns:
            Dictionary with parameter generation outputs
        """
        batch_size, seq_len, _ = proof_encoding.size()

        # Get proof context
        proof_context = proof_encoding.mean(dim=1)  # [batch_size, hidden_dim]

        # Determine parameter type needed for this tactic family
        param_type_logits = self.param_type_head(proof_context)

        # Initialize decoder state
        decoder_hidden = proof_context.unsqueeze(0)  # [1, batch_size, hidden_dim]
        decoder_cell = torch.zeros_like(decoder_hidden)

        outputs = []
        copy_scores = []

        # Start token (can be learnable or fixed)
        current_input = torch.zeros(
            batch_size, 1, self.embedding_dim, device=proof_encoding.device
        )

        for step in range(self.max_decode_len):
            # Decoder step
            decoder_output, (decoder_hidden, decoder_cell) = self.parameter_decoder(
                current_input, (decoder_hidden, decoder_cell)
            )

            # Vocabulary distribution
            vocab_logits = self.vocab_head(decoder_output.squeeze(1))

            # Copy mechanism
            copy_logits = None
            if available_terms is not None:
                # Attention over available terms
                copy_scores_step = []
                for term_tensor in available_terms:
                    if term_tensor.dim() == 1:
                        term_tensor = term_tensor.unsqueeze(0)
                    score = self.copy_attention(decoder_output.squeeze(1), term_tensor)
                    copy_scores_step.append(score)

                if copy_scores_step:
                    copy_logits = torch.cat(copy_scores_step, dim=-1)
                    copy_scores.append(copy_logits)

            outputs.append(vocab_logits)

            # Prepare next input (teacher forcing during training)
            if target_parameters is not None and step < target_parameters.size(1) - 1:
                next_token_id = target_parameters[:, step + 1]
                current_input = self.token_embedding(next_token_id).unsqueeze(1)
            else:
                # Use predicted token
                next_token_id = torch.argmax(vocab_logits, dim=-1)
                current_input = self.token_embedding(next_token_id).unsqueeze(1)

        # Stack outputs
        vocab_logits_seq = torch.stack(
            outputs, dim=1
        )  # [batch_size, max_len, vocab_size]

        result = {
            "parameter_type_logits": param_type_logits,
            "vocab_logits": vocab_logits_seq,
            "proof_context": proof_context,
        }

        if copy_scores:
            result["copy_logits"] = torch.stack(copy_scores, dim=1)

        return result

    def generate_parameters(
        self,
        proof_encoding: torch.Tensor,
        tactic_family: str,
        available_terms: Optional[List[torch.Tensor]] = None,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Generate parameters autoregressively during inference.

        Args:
            proof_encoding: Encoded proof state
            tactic_family: Family of the target tactic
            available_terms: Available terms for selection
            max_length: Maximum generation length

        Returns:
            List of generated token IDs
        """
        self.eval()

        if max_length is None:
            max_length = self.max_decode_len

        with torch.no_grad():
            batch_size = proof_encoding.size(0)
            device = proof_encoding.device

            # Get proof context
            proof_context = proof_encoding.mean(dim=1)

            # Initialize decoder state
            decoder_hidden = proof_context.unsqueeze(0)
            decoder_cell = torch.zeros_like(decoder_hidden)

            generated_tokens = []
            current_input = torch.zeros(
                batch_size, 1, self.embedding_dim, device=device
            )

            for step in range(max_length):
                # Decoder step
                decoder_output, (decoder_hidden, decoder_cell) = self.parameter_decoder(
                    current_input, (decoder_hidden, decoder_cell)
                )

                # Get vocabulary logits
                vocab_logits = self.vocab_head(decoder_output.squeeze(1))

                # Sample next token
                next_token_id = torch.argmax(vocab_logits, dim=-1)
                generated_tokens.append(next_token_id.item())

                # Prepare next input
                current_input = self.token_embedding(next_token_id).unsqueeze(1)

                # Stop if end token is generated (assuming token 0 is end token)
                if next_token_id.item() == 0:
                    break

            return generated_tokens
