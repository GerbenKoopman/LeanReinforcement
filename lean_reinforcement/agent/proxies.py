"""
Proxy classes for remote model inference.
"""

from typing import List, Tuple
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
    ) -> List[Tuple[str, float]]:
        self.request_queue.put(
            (self.worker_id, "generate_tactics_with_probs", (state, n))
        )
        result: List[Tuple[str, float]] = self.response_queue.get()
        assert isinstance(result, list)
        return result

    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        self.request_queue.put((self.worker_id, "generate_tactics", (state, n)))
        result: List[str] = self.response_queue.get()
        assert isinstance(result, list)
        return result

    def generate_tactics_batch(self, states: List[str], n: int = 1) -> List[List[str]]:
        self.request_queue.put((self.worker_id, "generate_tactics_batch", (states, n)))
        result: List[List[str]] = self.response_queue.get()
        assert isinstance(result, list)
        return result

    def generate_tactics_with_probs_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[Tuple[str, float]]]:
        self.request_queue.put(
            (self.worker_id, "generate_tactics_with_probs_batch", (states, n))
        )
        result: List[List[Tuple[str, float]]] = self.response_queue.get()
        assert isinstance(result, list)
        return result


class QueueProxyValueHead:
    def __init__(
        self, request_queue: mp.Queue, response_queue: mp.Queue, worker_id: int
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

    def predict(self, state: str) -> float:
        self.request_queue.put((self.worker_id, "predict_value", (state,)))
        result: float = self.response_queue.get()
        assert isinstance(result, float)
        return result

    def predict_batch(self, states: List[str]) -> List[float]:
        self.request_queue.put((self.worker_id, "predict_batch", (states,)))
        result: List[float] = self.response_queue.get()
        assert isinstance(result, list)
        return result

    def encode_states(self, states: List[str]) -> torch.Tensor:
        self.request_queue.put((self.worker_id, "encode_states", (states,)))
        result: torch.Tensor = self.response_queue.get()
        assert isinstance(result, torch.Tensor)
        return result

    def predict_from_features(self, features: torch.Tensor) -> float:
        self.request_queue.put((self.worker_id, "predict_from_features", (features,)))
        result: float = self.response_queue.get()
        assert isinstance(result, float)
        return result

    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        self.request_queue.put(
            (self.worker_id, "predict_from_features_batch", (features,))
        )
        result: List[float] = self.response_queue.get()
        assert isinstance(result, list)
        return result
