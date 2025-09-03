"""
Experience Replay Buffer for storing and sampling transitions.
"""

import random
from collections import deque
from typing import Dict, List, Optional, Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from lean_dojo import TacticState

from ..model.agent import HierarchicalAction


class ExperienceReplayBuffer:
    """Enhanced experience replay buffer with proper state encoding."""

    def __init__(self, capacity: int, device: torch.device, agent=None):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.agent = agent  # Reference to agent for encoding

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def push(
        self,
        state: TacticState,
        action: HierarchicalAction,
        reward: float,
        next_state: Optional[TacticState],
        done: bool,
    ):
        """Add experience to the buffer."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Optional[Dict[str, Any]]:
        """Sample a batch and return properly formatted tensors."""
        if len(self.buffer) < batch_size:
            return None

        experiences = random.sample(self.buffer, batch_size)
        return self._prepare_batch(experiences)

    def _prepare_batch(self, experiences: List[Dict]) -> Dict[str, Any]:
        """Convert experiences to batched tensors."""
        states = [exp["state"] for exp in experiences]
        actions = [exp["action"] for exp in experiences]
        rewards = [exp["reward"] for exp in experiences]
        next_states = [exp["next_state"] for exp in experiences]
        dones = [exp["done"] for exp in experiences]

        # Encode states on-the-fly
        encoded_states = self._encode_state_batch(states)
        encoded_next_states = self._encode_state_batch(
            [s for s in next_states if s is not None]
        )

        # Create batch dictionary
        batch = {
            "states": states,
            "actions": actions,
            "rewards": torch.tensor(rewards, dtype=torch.float32).to(self.device),
            "next_states": next_states,
            "dones": torch.tensor(dones, dtype=torch.bool).to(self.device),
            "encoded_states": encoded_states,
            "encoded_next_states": encoded_next_states,
        }

        return batch

    def _encode_state_batch(
        self, states: List[Optional[TacticState]]
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of states, handling None values, and collate them."""
        agent_module = self.agent.module if isinstance(self.agent, DDP) else self.agent
        if not agent_module:
            raise RuntimeError("Agent reference not set in ExperienceReplayBuffer")

        encoded_list = [agent_module.encode_state(s) for s in states if s is not None]

        if not encoded_list:
            return {}

        # Collate the list of dicts into a single dict of batched tensors
        collated_batch = {
            key: torch.cat([d[key] for d in encoded_list], dim=0)
            for key in encoded_list[0]
        }
        return collated_batch
