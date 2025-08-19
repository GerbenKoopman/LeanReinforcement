"""
Attention mechanisms for mathematical proof state encoding.

This module implements attention mechanisms for processing Lean proof
states, with specialized attention for goals, hypotheses, and mathematical
context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


class RoPEPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with RoPE applied [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Create position indices
        position = torch.arange(seq_len, device=x.device, dtype=torch.float)

        # Compute the rotation angles
        inv_freq = self.get_buffer("inv_freq")
        if inv_freq is None:
            # Fallback computation if buffer is not available
            d_model = x.size(-1)
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, d_model, 2, device=x.device).float() / d_model)
            )
        freqs = torch.outer(position, inv_freq)  # [seq_len, d_model//2]

        # Create cos and sin matrices
        cos_freqs = torch.cos(freqs)  # [seq_len, d_model//2]
        sin_freqs = torch.sin(freqs)  # [seq_len, d_model//2]

        # Expand to match x dimensions
        cos_freqs = cos_freqs.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, seq_len, d_model//2]
        sin_freqs = sin_freqs.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, seq_len, d_model//2]

        # Split x into pairs for rotation
        x1 = x[..., 0::2]  # [batch, seq_len, d_model//2]
        x2 = x[..., 1::2]  # [batch, seq_len, d_model//2]

        # Apply rotation
        rotated_x1 = x1 * cos_freqs - x2 * sin_freqs
        rotated_x2 = x1 * sin_freqs + x2 * cos_freqs

        # Interleave the rotated components
        result = torch.zeros_like(x)
        result[..., 0::2] = rotated_x1
        result[..., 1::2] = rotated_x2

        return result

    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor [batch_size, seq_len, d_model] or [batch_size, n_heads, seq_len, d_k]
            k: Key tensor [batch_size, seq_len, d_model] or [batch_size, n_heads, seq_len, d_k]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Handle multi-head attention format
        if q.dim() == 4:  # [batch, n_heads, seq_len, d_k]
            batch_size, n_heads, seq_len, d_k = q.shape

            # Apply RoPE per head
            q_list = []
            k_list = []

            for head in range(n_heads):
                # Extract single head: [batch, seq_len, d_k]
                q_head = q[:, head, :, :]
                k_head = k[:, head, :, :]

                # Apply RoPE to this head
                q_head_rotated = self._apply_rope_to_head(q_head)
                k_head_rotated = self._apply_rope_to_head(k_head)

                q_list.append(q_head_rotated)
                k_list.append(k_head_rotated)

            # Stack heads back: [batch, n_heads, seq_len, d_k]
            q_rotated = torch.stack(q_list, dim=1)
            k_rotated = torch.stack(k_list, dim=1)

            return q_rotated, k_rotated
        else:  # [batch, seq_len, d_model]
            return self.forward(q), self.forward(k)

    def _apply_rope_to_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to a single attention head.

        Args:
            x: Input tensor [batch_size, seq_len, d_k]

        Returns:
            Tensor with RoPE applied [batch_size, seq_len, d_k]
        """
        batch_size, seq_len, d_k = x.shape

        # Create position indices
        position = torch.arange(seq_len, device=x.device, dtype=torch.float)

        # Compute the rotation angles - use only the first d_k//2 frequencies
        inv_freq = self.get_buffer("inv_freq")
        if inv_freq is None:
            # Fallback computation if buffer is not available
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, d_k, 2, device=x.device).float() / d_k)
            )

        # Use only the frequencies needed for this head dimension
        # Make sure we don't exceed the available frequencies
        needed_freqs = min(d_k // 2, inv_freq.size(0))
        head_inv_freq = inv_freq[:needed_freqs]
        freqs = torch.outer(position, head_inv_freq)  # [seq_len, d_k//2]

        # Create cos and sin matrices
        cos_freqs = torch.cos(freqs)  # [seq_len, d_k//2]
        sin_freqs = torch.sin(freqs)  # [seq_len, d_k//2]

        # Expand to match x dimensions
        cos_freqs = cos_freqs.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, seq_len, d_k//2]
        sin_freqs = sin_freqs.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, seq_len, d_k//2]

        # Split x into pairs for rotation
        x1 = x[..., 0::2]  # [batch, seq_len, d_k//2]
        x2 = x[..., 1::2]  # [batch, seq_len, d_k//2]

        # Apply rotation
        rotated_x1 = x1 * cos_freqs - x2 * sin_freqs
        rotated_x2 = x1 * sin_freqs + x2 * cos_freqs

        # Interleave the rotated components
        result = torch.zeros_like(x)
        result[..., 0::2] = rotated_x1
        result[..., 1::2] = rotated_x2

        return result


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with RoPE positional encoding."""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # RoPE for positional encoding
        self.rope = RoPEPositionalEncoding(self.d_k, max_len)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-head attention with RoPE.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]

        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query

        # Linear transformations and reshape for multi-head attention
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # [batch, n_heads, seq_len, d_k]

        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # [batch, n_heads, seq_len, d_k]

        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # [batch, n_heads, seq_len, d_k]

        # Apply RoPE to Q and K
        Q, K = self.rope.apply_rotary_pos_emb(Q, K)

        # Scaled dot-product attention
        attention = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and put through final linear layer
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.w_o(attention)

        # Residual connection and layer norm
        return self.layer_norm(output + residual)

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Handle different mask formats
            if mask.dim() == 2:  # [batch, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            elif mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V)


class MathematicalAttentionEncoder(nn.Module):
    """
    Transformer encoder for mathematical proof states using PyTorch built-ins.

    This encoder processes Lean proof states with specialized attention for:
    - Goals and hypotheses
    - Mathematical context and definitions
    - Type information and constraints

    Uses PyTorch's built-in TransformerEncoder for efficiency and reliability.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = RoPEPositionalEncoding(d_model, max_seq_len)

        # Transformer layers with RoPE
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, n_heads, d_ff, dropout, max_seq_len)
                for _ in range(n_layers)
            ]
        )

        # Specialized attention heads for different proof components (with RoPE)
        self.goal_attention = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        self.hypothesis_attention = MultiHeadAttention(
            d_model, n_heads, dropout, max_seq_len
        )
        self.context_attention = MultiHeadAttention(
            d_model, n_heads, dropout, max_seq_len
        )

        # Output projections
        self.goal_projection = nn.Linear(d_model, d_model)
        self.hypothesis_projection = nn.Linear(d_model, d_model)
        self.context_projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        goal_mask: Optional[torch.Tensor] = None,
        hypothesis_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode proof state with specialized attention.

        Args:
            input_ids: Tokenized proof state [batch_size, seq_len]
            goal_mask: Mask for goal positions [batch_size, seq_len]
            hypothesis_mask: Mask for hypothesis positions [batch_size, seq_len]
            attention_mask: General attention mask [batch_size, seq_len]

        Returns:
            Dictionary containing different encodings:
            - 'full_encoding': Complete sequence encoding
            - 'goal_encoding': Goal-focused encoding
            - 'hypothesis_encoding': Hypothesis-focused encoding
            - 'context_encoding': Context-focused encoding
        """
        # Token embedding (no positional encoding added here - RoPE is applied in attention)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Apply specialized attention for different proof components
        full_encoding = x

        # Goal-specific attention
        if goal_mask is not None:
            goal_encoding = self.goal_attention(
                x, x, x, goal_mask.unsqueeze(1).unsqueeze(1)
            )
            goal_encoding = self.goal_projection(goal_encoding)
        else:
            goal_encoding = self.goal_projection(x)

        # Hypothesis-specific attention
        if hypothesis_mask is not None:
            hyp_encoding = self.hypothesis_attention(
                x, x, x, hypothesis_mask.unsqueeze(1).unsqueeze(1)
            )
            hyp_encoding = self.hypothesis_projection(hyp_encoding)
        else:
            hyp_encoding = self.hypothesis_projection(x)

        # Context attention (full sequence)
        context_encoding = self.context_attention(
            x,
            x,
            x,
            (
                attention_mask.unsqueeze(1).unsqueeze(1)
                if attention_mask is not None
                else None
            ),
        )
        context_encoding = self.context_projection(context_encoding)

        return {
            "full_encoding": full_encoding,
            "goal_encoding": goal_encoding,
            "hypothesis_encoding": hyp_encoding,
            "context_encoding": context_encoding,
        }


class TransformerLayer(nn.Module):
    """Single transformer layer with RoPE-enabled multi-head attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout, max_len)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer layer."""
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + attn_output)

        return self.dropout(output)


class AttentionPooling(nn.Module):
    """Attention-based pooling for variable-length sequences."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence using attention weights.

        Args:
            x: Input sequence [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, d_model]
        """
        # Compute attention weights
        attention_weights = self.attention(x).squeeze(-1)  # [batch_size, seq_len]

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(
            attention_weights, dim=-1
        )  # [batch_size, seq_len]

        # Apply attention weights
        pooled = torch.sum(
            x * attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, d_model]

        return pooled
