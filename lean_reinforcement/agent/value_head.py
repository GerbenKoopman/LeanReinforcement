"""
A value head that uses a pre-trained encoder to predict the
value (win probability) of a given proof state.
"""

from typing_extensions import Self
import torch
import torch.nn as nn
from typing import List, Optional, cast
import os
from loguru import logger

from lean_reinforcement.agent.transformer import Transformer


# Default encoder output dimension for the ByT5 model
ENCODER_OUTPUT_DIM = 1472


class ValueHead(nn.Module):
    def __init__(
        self,
        transformer: Transformer,
        hidden_dims: Optional[List[int]] = None,
        input_dim: int = ENCODER_OUTPUT_DIM,
    ):
        """
        Initialize the value head.

        Args:
            transformer: The transformer model to use for encoding states.
            hidden_dims: List of hidden layer dimensions. If None, uses [256].
                        Empty list means direct linear projection from input to output.
            input_dim: Input dimension from encoder (default: 1472 for ByT5).
        """
        super().__init__()
        self.tokenizer = transformer.tokenizer
        self.encoder = transformer.model.get_encoder()

        # Store dimensions for serialization
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256]

        # Freeze the pre-trained encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Build the value head network dynamically
        self.value_head = self._build_value_head(input_dim, self.hidden_dims)

        if torch.cuda.is_available():
            self.to("cuda")

        self._predict_call_count = 0

    def _build_value_head(
        self, input_dim: int, hidden_dims: List[int]
    ) -> nn.Sequential:
        """
        Build the value head MLP with configurable hidden dimensions.

        Args:
            input_dim: Input dimension (encoder output size).
            hidden_dims: List of hidden layer sizes. Empty list means
                        direct projection to scalar output.

        Returns:
            nn.Sequential module implementing the MLP.
        """
        if not hidden_dims:
            # Direct linear projection: input_dim -> 1
            return nn.Sequential(nn.Linear(input_dim, 1))

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        # Final projection to scalar output
        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def _periodic_cache_cleanup(self) -> None:
        """Clear GPU cache periodically instead of on every call."""
        self._predict_call_count += 1
        if self._predict_call_count % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        return cast(torch.Tensor, features.detach())

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
        result: float = torch.tanh(value).item()

        # Clean up
        del features
        del value
        self._periodic_cache_cleanup()

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

        results: List[float] = torch.tanh(values).tolist()

        # Clean up
        del features
        del values
        self._periodic_cache_cleanup()

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
        result: float = torch.tanh(value).item()

        # Clean up
        del value
        self._periodic_cache_cleanup()

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

        results: List[float] = torch.tanh(values).tolist()

        # Clean up
        del values
        self._periodic_cache_cleanup()

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
            "hidden_dims": self.hidden_dims,
            "input_dim": self.input_dim,
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
        super().train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        return self
