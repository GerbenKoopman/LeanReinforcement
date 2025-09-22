"""
A simplified Transformer Agent for Lean Theorem Proving.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from lean_dojo import TacticState

from ....environment import StepResult
from ...agents import BaseAgent
from .model import SimpleTransformer
from .utils import ProofStateTokenizer


class TransformerAgent(BaseAgent):
    """
    A simplified transformer agent for Lean theorem proving.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tokenizer = ProofStateTokenizer(vocab_size)

        self.model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        self._last_action_info = {}

    def select_action(self, state: TacticState, **kwargs) -> Union[str, None]:
        """
        Select an action by generating a tactic with the transformer model.
        """
        encoded_state = self.encode_state(state)
        src = encoded_state["input_ids"]

        generated_ids = self.model.generate_tactic(
            src, self.tokenizer, device=self.device.type
        )

        # Decode the generated tactic, skipping special tokens
        tactic_str = self.tokenizer.decode(
            generated_ids.squeeze(0).tolist(), skip_special_tokens=True
        )

        # Store information for the update step
        self._last_action_info = {
            "encoded_state": encoded_state,
            "generated_tactic_ids": generated_ids,
        }

        return tactic_str if tactic_str else "sorry"

    def update(self, step_result: StepResult) -> None:
        """
        Update the agent based on a step result using supervised learning.
        """
        if not step_result.action or step_result.reward <= 0:
            # Don't learn from failed steps or unproven paths for now
            return

        # Get the state before the action and the successful action
        before_state = step_result.before_state
        successful_action = step_result.action

        # Encode the state and the target action
        encoded_state = self.encode_state(before_state)
        tgt_ids = self._tokenize_tactic(successful_action)

        src = encoded_state["input_ids"]
        tgt_input = tgt_ids[:, :-1]  # Input for decoder (everything but last token)
        tgt_output = tgt_ids[:, 1:]  # Target for loss (everything but first token)

        src_padding_mask = (src == self.tokenizer.pad_token_id).to(self.device)
        tgt_padding_mask = (tgt_input == self.tokenizer.pad_token_id).to(self.device)
        tgt_mask = self.model.transformer.generate_square_subsequent_mask(
            tgt_input.size(1)
        ).to(self.device)

        # Forward pass
        logits = self.model(
            src,
            tgt_input,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        self.optimizer.zero_grad()

        # Calculate loss
        loss = self.criterion(logits.view(-1, self.vocab_size), tgt_output.reshape(-1))

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def encode_state(self, state: TacticState) -> Dict[str, torch.Tensor]:
        """
        Encode LeanDojo state for the neural network.
        """
        proof_state = self.tokenizer.parse_proof_state(state.pp)
        encoded = self.tokenizer.encode_proof_state(proof_state, max_length=256)

        # Move to device and add batch dimension
        for key, tensor in encoded.items():
            encoded[key] = tensor.unsqueeze(0).to(self.device)

        return encoded

    def _tokenize_tactic(self, tactic: str) -> torch.Tensor:
        """Tokenize a tactic string and prepare it for training."""
        # Add BOS and EOS tokens
        tactic_with_special_tokens = (
            f"{self.tokenizer.bos_token}{tactic}{self.tokenizer.eos_token}"
        )
        token_ids = self.tokenizer.encode(
            tactic_with_special_tokens, max_length=64, padding=True
        )
        return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(
            0
        )

    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save model state."""
        if filepath is None:
            scratch_dir = os.getenv("SCRATCH_SHARED", ".")
            models_dir = Path(scratch_dir) / "saved_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(models_dir / f"transformer_{int(time.time())}.pt")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
            },
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {filepath}")

    def reset(self) -> None:
        self._last_action_info = {}

    def end_episode(self, episode_reward: float) -> None:
        pass
