"""
A value head that uses a pre-trained encoder to predict the
value (win probability) of a given proof state.
"""

from typing_extensions import Self
import torch
import torch.nn as nn
from typing import List, cast
import os
from collections import OrderedDict
from loguru import logger

from lean_reinforcement.agent.transformer import Transformer


class ValueHead(nn.Module):
    def __init__(self, transformer: Transformer):
        super().__init__()
        self.tokenizer = transformer.tokenizer
        self.encoder = transformer.model.get_encoder()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simple LRU cache storing CPU features keyed by state string
        self._feature_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._max_cache_size = 2048

        # Freeze the pre-trained encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # The new value head that will be trained
        self.value_head = nn.Sequential(
            nn.Linear(1472, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.feature_dim = cast(int, self.value_head[0].in_features)

        if torch.cuda.is_available():
            self.to("cuda")

    def _encode_uncached(self, s: List[str]) -> torch.Tensor:
        tokenized_s = self.tokenizer(
            s, return_tensors="pt", padding=True, truncation=True, max_length=2300
        ).to(self.device)

        hidden_state = self.encoder(tokenized_s.input_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        features: torch.Tensor = (
            hidden_state * tokenized_s.attention_mask.unsqueeze(2)
        ).sum(dim=1) / lens.unsqueeze(1)

        # Clean up intermediate tensors
        del tokenized_s
        del hidden_state
        del lens

        return features

    def _cache_get(self, key: str) -> torch.Tensor | None:
        feat = self._feature_cache.get(key)
        if feat is not None:
            self._feature_cache.move_to_end(key)
        return feat

    def _cache_set(self, key: str, value: torch.Tensor) -> None:
        # Store on CPU to keep GPU memory low
        self._feature_cache[key] = value.detach().cpu()
        self._feature_cache.move_to_end(key)
        if len(self._feature_cache) > self._max_cache_size:
            self._feature_cache.popitem(last=False)

    def clear_feature_cache(self) -> None:
        self._feature_cache.clear()

    def encode_states(self, s: List[str]) -> torch.Tensor:
        """Encode a batch of texts into feature vectors."""
        if not s:
            return torch.empty(0, self.feature_dim, device=self.device)

        cached_features: List[torch.Tensor | None] = []
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, text in enumerate(s):
            feat = self._cache_get(text)
            if feat is None:
                missing_indices.append(idx)
                missing_texts.append(text)
            cached_features.append(feat)

        if missing_texts:
            new_features = self._encode_uncached(missing_texts)
            for pos, feat in zip(missing_indices, new_features):
                self._cache_set(s[pos], feat)
                cached_features[pos] = feat.detach().cpu()

        stacked = torch.stack(
            [feat.to(self.device) for feat in cached_features if feat is not None],
            dim=0,
        )
        return stacked

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

        results: List[float] = torch.tanh(values).tolist()

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
        result: float = torch.tanh(value).item()

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

        results: List[float] = torch.tanh(values).tolist()

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
        super().train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        return self
