"""
Proxy classes for remote model inference.
"""

from typing import List
import torch
import torch.multiprocessing as mp

from lean_reinforcement.agent.transformer import TransformerProtocol


class QueueProxyTransformer(TransformerProtocol):
    def __init__(
        self, request_queue: mp.Queue, response_queue: mp.Queue, worker_id: int
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        # Mock tokenizer for AgentRunner if it accesses it directly (unlikely but safe to have)
        self.tokenizer = None

    def generate_tactics_with_probs(
        self, state: str, n: int = 1
    ) -> List[tuple[str, float]]:
        self.request_queue.put(
            (self.worker_id, "generate_tactics_with_probs", (state, n))
        )
        return self.response_queue.get()

    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        self.request_queue.put((self.worker_id, "generate_tactics", (state, n)))
        return self.response_queue.get()

    def generate_tactics_batch(self, states: List[str], n: int = 1) -> List[List[str]]:
        self.request_queue.put((self.worker_id, "generate_tactics_batch", (states, n)))
        return self.response_queue.get()

    def generate_tactics_with_probs_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[tuple[str, float]]]:
        self.request_queue.put(
            (self.worker_id, "generate_tactics_with_probs_batch", (states, n))
        )
        return self.response_queue.get()


class QueueProxyValueHead:
    def __init__(
        self, request_queue: mp.Queue, response_queue: mp.Queue, worker_id: int
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

    def predict(self, state: str) -> float:
        self.request_queue.put((self.worker_id, "predict_value", (state,)))
        return self.response_queue.get()

    def predict_batch(self, states: List[str]) -> List[float]:
        self.request_queue.put((self.worker_id, "predict_batch", (states,)))
        return self.response_queue.get()

    def encode_states(self, states: List[str]) -> torch.Tensor:
        self.request_queue.put((self.worker_id, "encode_states", (states,)))
        return self.response_queue.get()

    def predict_from_features(self, features: torch.Tensor) -> float:
        self.request_queue.put((self.worker_id, "predict_from_features", (features,)))
        return self.response_queue.get()

    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        self.request_queue.put(
            (self.worker_id, "predict_from_features_batch", (features,))
        )
        return self.response_queue.get()
