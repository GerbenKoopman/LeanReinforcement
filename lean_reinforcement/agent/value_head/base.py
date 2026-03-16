"""Shared base logic for ValueHead variants."""

from __future__ import annotations

import os
from typing import Any, Dict, List, cast

import torch
import torch.nn as nn
from loguru import logger
from typing_extensions import Self

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.utilities.memory import periodic_cache_cleanup


class BaseValueHead(nn.Module):
    """Common encoder + prediction behavior for value heads."""

    value_head: nn.Module

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self.tokenizer = transformer.tokenizer
        self.encoder = transformer.model.get_encoder()

        # Freeze the pre-trained encoder.
        for param in self.encoder.parameters():
            param.requires_grad = False

        self._predict_call_count = 0

    def encode_states(self, s: List[str]) -> torch.Tensor:
        """Encode proof-state strings into mean-pooled Euclidean features."""
        tokenized_s = self.tokenizer(
            s,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2300,
        )
        if torch.cuda.is_available():
            tokenized_s = tokenized_s.to("cuda")

        hidden_state = self.encoder(tokenized_s.input_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        attn_mask = tokenized_s.attention_mask.unsqueeze(2)
        features = (hidden_state * attn_mask).sum(dim=1) / lens.unsqueeze(1)

        return cast(torch.Tensor, features.detach())

    def _predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        logits = cast(torch.Tensor, self.value_head(features)).squeeze()
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        return logits

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        """Subclass-specific checkpoint fields."""
        return {}

    def save_checkpoint(self, folder: str, filename: str) -> None:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        checkpoint: Dict[str, Any] = {
            "value_head_state_dict": self.value_head.state_dict(),
            "transformer_name": self.tokenizer.name_or_path,
        }
        checkpoint.update(self._checkpoint_metadata())

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, folder: str, filename: str) -> None:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        checkpoint = torch.load(
            filepath,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
        self.value_head.load_state_dict(checkpoint["value_head_state_dict"])
        logger.info(f"Checkpoint loaded from {filepath}")

    @torch.no_grad()
    def predict(self, state_str: str) -> float:
        """Predict value of a single state. Returns in [-1, 1]."""
        self.eval()
        features = self.encode_states([state_str])
        logits = self._predict_logits(features)
        result: float = torch.tanh(logits[0]).item()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return result

    @torch.no_grad()
    def predict_batch(self, state_strs: List[str]) -> List[float]:
        """Predict values of a batch of states. Returns in [-1, 1]."""
        self.eval()
        features = self.encode_states(state_strs)
        logits = self._predict_logits(features)
        results: List[float] = torch.tanh(logits).tolist()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return results

    @torch.no_grad()
    def predict_from_features(self, features: torch.Tensor) -> float:
        """Predict value from pre-computed features. Returns in [-1, 1]."""
        self.eval()
        logits = self._predict_logits(features)
        result: float = torch.tanh(logits[0]).item()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return result

    @torch.no_grad()
    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        """Predict values from pre-computed feature batch. Returns in [-1, 1]."""
        self.eval()
        logits = self._predict_logits(features)
        results: List[float] = torch.tanh(logits).tolist()
        self._predict_call_count = periodic_cache_cleanup(self._predict_call_count)
        return results

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        return self
