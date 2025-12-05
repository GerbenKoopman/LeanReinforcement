"""
A value head that uses a pre-trained encoder to predict the
value (win probability) of a given proof state.
"""

from typing_extensions import Self
import torch
import torch.nn as nn
from typing import List
import os
from loguru import logger

from lean_reinforcement.agent.transformer import Transformer


class ValueHead(nn.Module):
    def __init__(self, transformer: Transformer):
        super().__init__()
        self.tokenizer = transformer.tokenizer
        self.encoder = transformer.model.get_encoder()

        # Freeze the pre-trained encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # The new value head that will be trained
        self.value_head = nn.Sequential(
            nn.Linear(1472, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        if torch.cuda.is_available():
            self.to("cuda")

    def encode_states(self, s: List[str]) -> torch.Tensor:
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

        # Clean up intermediate tensors
        del tokenized_s
        del hidden_state
        del lens

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
        features = self.encode_states([input_str])

        # Get value prediction
        value = self.value_head(features).squeeze()

        # Apply tanh to squash the value between -1 and 1
        result = torch.tanh(value).item()

        # Clean up
        del features
        del value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def predict_batch(self, state_strs: List[str]) -> List[float]:
        """
        Predicts the value of a batch of states.
        Returns a list of floats between -1.0 and 1.0.
        """
        self.eval()
        features = self.encode_states(state_strs)
        values = self.value_head(features).squeeze()

        # Handle case where batch size is 1
        if values.ndim == 0:
            values = values.unsqueeze(0)

        results = torch.tanh(values).tolist()

        # Clean up
        del features
        del values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    @torch.no_grad()
    def predict_from_features(self, features: torch.Tensor) -> float:
        """
        Predicts the value from pre-computed encoder features.
        This is more efficient than predict() when encoder representations
        are already available.

        Args:
            features: Pre-computed encoder features (output of _encode)

        Returns:
            A float between -1.0 and 1.0.
        """
        self.eval()  # Set to evaluation mode

        # Get value prediction
        value = self.value_head(features).squeeze()

        # Apply tanh to squash the value between -1 and 1
        result = torch.tanh(value).item()

        # Clean up
        del value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        """
        Predicts the value from pre-computed encoder features for a batch.
        """
        self.eval()
        values = self.value_head(features).squeeze()

        # Handle case where batch size is 1
        if values.ndim == 0:
            values = values.unsqueeze(0)

        results = torch.tanh(values).tolist()

        # Clean up
        del values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def save_checkpoint(self, folder: str, filename: str):
        """
        Saves the current neural network (with its parameters) in
        folder/filename

        Args:
            folder: Directory to save checkpoint
            filename: Name of the checkpoint file
        """
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        checkpoint = {
            "value_head_state_dict": self.value_head.state_dict(),
            "transformer_name": self.tokenizer.name_or_path,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, folder: str, filename: str):
        """
        Loads parameters of the neural network from folder/filename

        Args:
            folder: Directory containing checkpoint
            filename: Name of the checkpoint file
        """
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        checkpoint = torch.load(
            filepath, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.value_head.load_state_dict(checkpoint["value_head_state_dict"])

        logger.info(f"Checkpoint loaded from {filepath}")

    def train(self, mode: bool = True) -> Self:
        return super().train(mode)

    def eval(self) -> Self:
        return super().eval()
