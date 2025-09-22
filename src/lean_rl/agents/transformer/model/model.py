"""
A simple Transformer model for tactic generation.
"""

import torch
import torch.nn as nn

from typing import Optional


class SimpleTransformer(nn.Module):
    """
    A simple encoder-decoder transformer model for generating tactics.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the transformer.

        Args:
            src (torch.Tensor): Source sequence (encoded state).
            tgt (torch.Tensor): Target sequence (tactic being generated).
            src_mask (torch.Tensor, optional): Source mask. Defaults to None.
            tgt_mask (torch.Tensor, optional): Target mask. Defaults to None.
            src_padding_mask (torch.Tensor, optional): Source padding mask. Defaults to None.
            tgt_padding_mask (torch.Tensor, optional): Target padding mask. Defaults to None.

        Returns:
            torch.Tensor: Output logits.
        """
        src_embed = self.embedding(src) * self.d_model**0.5
        tgt_embed = self.embedding(tgt) * self.d_model**0.5

        output = self.transformer(
            src_embed,
            tgt_embed,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return self.output_layer(output)

    def generate_tactic(
        self, src: torch.Tensor, tokenizer, max_length: int = 50, device: str = "cpu"
    ):
        """
        Generate a tactic sequence given a source state.
        """
        self.eval()
        src_padding_mask = (src == tokenizer.pad_token_id).to(device)

        # Start with the beginning of sequence token
        tgt = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(
                device
            )

            with torch.no_grad():
                output_logits = self.forward(
                    src, tgt, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask
                )

            # Get the last token's logits and find the most likely next token
            next_token_logits = output_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            # Append the new token to the target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if we generate the end of sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break

        self.train()
        return tgt
