"""
A value head that uses a pre-trained encoder to predict the
value (win probability) of a given proof state.
"""

from typing_extensions import Self
import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ValueHead(nn.Module):

    def __init__(
        self, transformer_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small"
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(transformer_name)
        self.encoder = self.transformer.get_encoder()

        # Freeze the pre-trained encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # The new value head that will be trained
        self.value_head = nn.Sequential(
            nn.Linear(1472, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        if torch.cuda.is_available():
            self.to("cuda")

    def _encode(self, s: List[str]) -> torch.Tensor:
        """Encode a batch of texts into feature vectors."""
        tokenized_s = self.tokenizer(
            s, return_tensors="pt", padding=True, truncation=True, max_length=2300
        )
        if torch.cuda.is_available():
            tokenized_s = tokenized_s.to("cuda")

        hidden_state = self.encoder(tokenized_s.input_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)
        return features

    @torch.no_grad()
    def predict(self, state_str: str) -> float:
        """
        Predicts the value of a single state.
        Returns a float between -1.0 and 1.0.
        """
        self.eval()  # Set to evaluation mode
        input_str = state_str

        # Encode the input string (pass as a batch of 1)
        features = self._encode([input_str])

        # Get value prediction
        value = self.value_head(features).squeeze()

        # Apply tanh to squash the value between -1 and 1
        return torch.tanh(value).item()

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """

    def train(self, mode: bool = True) -> Self:
        return super().train(mode)

    def eval(self) -> Self:
        return super().eval()
