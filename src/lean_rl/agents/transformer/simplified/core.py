"""
Simplified Transformer Agent for Lean Theorem Proving.

This module provides a clean, efficient implementation that replaces the
overly complex hierarchical architecture with a simpler, more maintainable approach.
Integrates with LeanDojo for proper RL training on Mathlib4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SimpleTokenizer:
    """Lightweight tokenizer for Lean proof states."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size

        # Essential tokens only
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<goal>": 2,
            "<hyp>": 3,
            "⊢": 4,
            "∀": 5,
            "∃": 6,
            "→": 7,
            "∧": 8,
            "∨": 9,
            "¬": 10,
            "=": 11,
        }

        # Common Lean tactics (essential subset)
        self.tactics = [
            "apply",
            "exact",
            "rw",
            "simp",
            "intro",
            "cases",
            "induction",
            "sorry",
            "have",
            "show",
            "calc",
            "ring",
            "norm_num",
            "linarith",
        ]

        # Build vocabulary
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        current_id = len(self.special_tokens)
        for tactic in self.tactics:
            self.token_to_id[tactic] = current_id
            self.id_to_token[current_id] = tactic
            current_id += 1

    def encode(self, text: str) -> List[int]:
        """Simple whitespace tokenization with vocabulary lookup."""
        tokens = text.lower().split()
        return [
            self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens
        ]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(tid, "<unk>") for tid in token_ids]
        return " ".join(tokens)


class ProofStateEncoder(nn.Module):
    """Simple transformer encoder for proof states."""

    def __init__(
        self, vocab_size: int, d_model: int = 256, n_heads: int = 8, n_layers: int = 4
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1024, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode input to fixed-size representation."""
        seq_len = input_ids.size(1)

        # Embedding with positional encoding
        x = self.embedding(input_ids) + self.pos_encoding[:seq_len]

        # Transform
        x = self.transformer(x)

        # Pool to single vector
        x = self.pooler(x.transpose(1, 2)).squeeze(-1)

        return x


class TacticPolicy(nn.Module):
    """Simple policy network for tactic selection."""

    def __init__(self, d_model: int, num_tactics: int):
        super().__init__()

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_tactics),
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

    def forward(
        self, state_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and value estimate."""
        policy_logits = self.policy_head(state_encoding)
        value = self.value_head(state_encoding)
        return policy_logits, value


class ParameterGenerator(nn.Module):
    """Simple parameter generation using sequence-to-sequence."""

    def __init__(self, d_model: int, vocab_size: int, max_params: int = 3):
        super().__init__()

        self.max_params = max_params
        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size * max_params),
        )

    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """Generate parameters as token distributions."""
        batch_size = state_encoding.size(0)
        logits = self.generator(state_encoding)
        return logits.view(batch_size, self.max_params, -1)


class SimplifiedTransformerAgent(nn.Module):
    """Dramatically simplified transformer agent."""

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Core components
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.encoder = ProofStateEncoder(vocab_size, d_model, n_heads, n_layers)
        self.policy = TacticPolicy(d_model, len(self.tokenizer.tactics))
        self.param_generator = ParameterGenerator(d_model, vocab_size)

        # Training components
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        self.to(self.device)

    def select_action(self, state) -> str:
        """Select action for given state."""
        with torch.no_grad():
            # Encode state
            state_text = state.pp
            token_ids = self.tokenizer.encode(state_text)

            # Pad/truncate to fixed length
            max_len = 512
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            else:
                token_ids.extend([0] * (max_len - len(token_ids)))

            input_tensor = torch.tensor([token_ids], device=self.device)

            # Get policy and parameters
            state_encoding = self.encoder(input_tensor)
            policy_logits, value = self.policy(state_encoding)
            param_logits = self.param_generator(state_encoding)

            # Sample action
            tactic_probs = F.softmax(policy_logits, dim=-1)
            tactic_idx = int(torch.multinomial(tactic_probs, 1).item())
            tactic = self.tokenizer.tactics[tactic_idx]

            # Sample parameters
            parameters = []
            for i in range(self.param_generator.max_params):
                param_probs = F.softmax(param_logits[0, i], dim=-1)
                param_idx = int(torch.multinomial(param_probs, 1).item())
                param_token = self.tokenizer.id_to_token.get(param_idx, "")
                if param_token and param_token not in ["<pad>", "<unk>"]:
                    parameters.append(param_token)

            # Format final tactic
            if parameters:
                return f"{tactic} {' '.join(parameters)}"
            else:
                return tactic

    def update(
        self, states: List, actions: List[str], rewards: List[float]
    ) -> Dict[str, float]:
        """Simple policy gradient update."""
        if not states:
            return {}

        # Encode states
        batch_encodings = []
        batch_actions = []

        for state, action in zip(states, actions):
            # Encode state
            token_ids = self.tokenizer.encode(state.pp)
            max_len = 512
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            else:
                token_ids.extend([0] * (max_len - len(token_ids)))

            batch_encodings.append(token_ids)

            # Parse action
            action_parts = action.split()
            tactic = action_parts[0] if action_parts else "sorry"
            tactic_idx = (
                self.tokenizer.tactics.index(tactic)
                if tactic in self.tokenizer.tactics
                else 0
            )
            batch_actions.append(tactic_idx)

        # Convert to tensors
        input_tensor = torch.tensor(batch_encodings, device=self.device)
        action_tensor = torch.tensor(batch_actions, device=self.device)
        reward_tensor = torch.tensor(rewards, device=self.device)

        # Forward pass
        state_encodings = self.encoder(input_tensor)
        policy_logits, values = self.policy(state_encodings)

        # Compute loss
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # Simple REINFORCE loss
        advantages = reward_tensor - values.squeeze(1)
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(1), reward_tensor)

        total_loss = policy_loss + 0.5 * value_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def get_value(self, state) -> float:
        """Get value estimate for a state (useful for RL algorithms)."""
        with torch.no_grad():
            # Encode state
            state_text = state.pp
            token_ids = self.tokenizer.encode(state_text)

            # Pad/truncate to fixed length
            max_len = 512
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            else:
                token_ids.extend([0] * (max_len - len(token_ids)))

            input_tensor = torch.tensor([token_ids], device=self.device)

            # Get value estimate
            state_encoding = self.encoder(input_tensor)
            _, value = self.policy(state_encoding)

            return value.item()

    def select_action_with_value(self, state) -> Tuple[str, float]:
        """Select action and return value estimate (useful for actor-critic)."""
        with torch.no_grad():
            # Encode state
            state_text = state.pp
            token_ids = self.tokenizer.encode(state_text)

            # Pad/truncate to fixed length
            max_len = 512
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            else:
                token_ids.extend([0] * (max_len - len(token_ids)))

            input_tensor = torch.tensor([token_ids], device=self.device)

            # Get policy and parameters
            state_encoding = self.encoder(input_tensor)
            policy_logits, value = self.policy(state_encoding)
            param_logits = self.param_generator(state_encoding)

            # Sample action
            tactic_probs = F.softmax(policy_logits, dim=-1)
            tactic_idx = int(torch.multinomial(tactic_probs, 1).item())
            tactic = self.tokenizer.tactics[tactic_idx]

            # Sample parameters
            parameters = []
            for i in range(self.param_generator.max_params):
                param_probs = F.softmax(param_logits[0, i], dim=-1)
                param_idx = int(torch.multinomial(param_probs, 1).item())
                param_token = self.tokenizer.id_to_token.get(param_idx, "")
                if param_token and param_token not in ["<pad>", "<unk>"]:
                    parameters.append(param_token)

            # Format final tactic
            if parameters:
                action = f"{tactic} {' '.join(parameters)}"
            else:
                action = tactic

            return action, value.item()

    def save(self, path: str):
        """Save model."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "tokenizer_tactics": self.tokenizer.tactics,
                "tokenizer_token_to_id": self.tokenizer.token_to_id,
                "tokenizer_id_to_token": self.tokenizer.id_to_token,
            },
            path,
        )

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        self.tokenizer.tactics = checkpoint["tokenizer_tactics"]
        self.tokenizer.token_to_id = checkpoint["tokenizer_token_to_id"]
        self.tokenizer.id_to_token = checkpoint["tokenizer_id_to_token"]
