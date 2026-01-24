"""
Proxy classes for remote model inference.
"""

from typing import List, Tuple, Optional
import queue
import torch
import torch.multiprocessing as mp
from loguru import logger

from lean_reinforcement.utilities.validation import (
    validate_inference_result_list_str,
    validate_inference_result_list_tuple_str_float,
    validate_inference_result_list_list_str,
    validate_inference_result_list_list_tuple_str_float,
    validate_float,
    validate_list_float,
    validate_torch_tensor,
)

from lean_reinforcement.agent.transformer import TransformerProtocol
from lean_reinforcement.utilities.types import InferenceRequest, InferenceResult


class InferenceTimeoutError(Exception):
    """Raised when waiting for inference response times out."""


class QueueProxyTransformer(TransformerProtocol):
    # Default timeout of 10 minutes for waiting on inference responses
    DEFAULT_TIMEOUT = 600.0

    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_id: int,
        timeout: Optional[float] = None,
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        # Mock tokenizer for AgentRunner if it accesses it directly (unlikely but safe to have)
        self.tokenizer = None

    def _get_response(self, request_type: str):
        """Get response from queue with timeout."""
        try:
            result = self.response_queue.get(timeout=self.timeout)
            return result
        except queue.Empty:
            logger.error(
                f"Worker {self.worker_id}: Timeout waiting for {request_type} response after {self.timeout}s"
            )
            raise InferenceTimeoutError(
                f"Timeout waiting for {request_type} response after {self.timeout}s"
            )

    def generate_tactics_with_probs(
        self, state: str, n: int = 1
    ) -> List[Tuple[str, float]]:
        self.request_queue.put(
            (self.worker_id, "generate_tactics_with_probs", (state, n))
        )
        result: List[Tuple[str, float]] = self._get_response(
            "generate_tactics_with_probs"
        )
        assert isinstance(result, list)
        return result

    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        self.request_queue.put((self.worker_id, "generate_tactics", (state, n)))
        result: List[str] = self._get_response("generate_tactics")
        assert isinstance(result, list)
        return result

    def generate_tactics_batch(self, states: List[str], n: int = 1) -> List[List[str]]:
        self.request_queue.put((self.worker_id, "generate_tactics_batch", (states, n)))
        result: List[List[str]] = self._get_response("generate_tactics_batch")
        assert isinstance(result, list)
        return result

    def generate_tactics_with_probs_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[Tuple[str, float]]]:
        self.request_queue.put(
            (self.worker_id, "generate_tactics_with_probs_batch", (states, n))
        )
        result: List[List[Tuple[str, float]]] = self._get_response(
            "generate_tactics_with_probs_batch"
        )
        assert isinstance(result, list)
        return result


class QueueProxyValueHead:
    # Default timeout of 10 minutes for waiting on inference responses
    DEFAULT_TIMEOUT = 600.0

    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_id: int,
        timeout: Optional[float] = None,
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

    def _get_response(self, request_type: str):
        """Get response from queue with timeout."""
        try:
            result = self.response_queue.get(timeout=self.timeout)
            return result
        except queue.Empty:
            logger.error(
                f"Worker {self.worker_id}: Timeout waiting for {request_type} response after {self.timeout}s"
            )
            raise InferenceTimeoutError(
                f"Timeout waiting for {request_type} response after {self.timeout}s"
            )

    def predict(self, state: str) -> float:
        self.request_queue.put((self.worker_id, "predict_value", (state,)))
        result: float = self._get_response("predict_value")
        assert isinstance(result, float)
        return result

    def predict_batch(self, states: List[str]) -> List[float]:
        self.request_queue.put((self.worker_id, "predict_batch", (states,)))
        result: List[float] = self._get_response("predict_batch")
        assert isinstance(result, list)
        return result

    def encode_states(self, states: List[str]) -> torch.Tensor:
        self.request_queue.put((self.worker_id, "encode_states", (states,)))
        result: torch.Tensor = self._get_response("encode_states")
        assert isinstance(result, torch.Tensor)
        return result

    def predict_from_features(self, features: torch.Tensor) -> float:
        self.request_queue.put((self.worker_id, "predict_from_features", (features,)))
        result: float = self._get_response("predict_from_features")
        assert isinstance(result, float)
        return result

    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        self.request_queue.put(
            (self.worker_id, "predict_from_features_batch", (features,))
        )
        result: List[float] = self._get_response("predict_from_features_batch")
        assert isinstance(result, list)
        return result
