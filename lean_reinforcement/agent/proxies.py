"""
Proxy classes for remote model inference.
"""

from typing import Any, List, Tuple, Optional
import gc
import queue
import time
import uuid
import ctypes
import ctypes.util
import torch
import torch.multiprocessing as mp
from loguru import logger

from lean_reinforcement.agent.transformer import TransformerProtocol
from lean_reinforcement.utilities.memory import (
    get_process_tree_rss_gb,
    MAX_WORKER_RSS_GB,
)

# Attempt to load malloc_trim for returning freed memory to OS
_libc_path = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_path) if _libc_path else None


def _malloc_trim():
    """Call glibc malloc_trim(0) to return freed memory to OS."""
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


class MemoryLimitExceeded(Exception):
    """Raised when worker RSS exceeds safe limits during queue polling."""


class InferenceTimeoutError(Exception):
    """Raised when waiting for inference response times out."""


class _ProxyMixin:
    """Shared helpers for request-id tracking and envelope-aware responses."""

    _next_request_id: int
    _stream_token: str
    response_queue: mp.Queue
    worker_id: int
    timeout: float

    def _init_mixin(self) -> None:
        self._next_request_id = 0

    def _new_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _is_response_envelope(self, message: Any) -> bool:
        return (
            isinstance(message, tuple)
            and len(message) == 4
            and isinstance(message[0], str)
            and isinstance(message[1], int)
            and isinstance(message[2], str)
        )

    def _get_response(self, request_type: str, request_id: int):
        """Get response with timeout, checking RSS between short polls.

        Uses an envelope-aware protocol to ignore stale responses that arrive
        after timeouts or worker restarts:
            (stream_token, request_id, request_type, payload)

        Falls back to legacy raw payload responses for backward compatibility.

        Raises ``MemoryLimitExceeded`` if RSS exceeds the hard cap.
        """
        POLL_INTERVAL = 5.0  # seconds between RSS checks
        RSS_HARD_CAP = MAX_WORKER_RSS_GB  # absolute hard limit (GB)
        RSS_WARN = MAX_WORKER_RSS_GB * 0.75  # warn threshold (GB)
        deadline = time.monotonic() + self.timeout
        polls = 0

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = min(POLL_INTERVAL, remaining)
            try:
                result = self.response_queue.get(timeout=wait)
                if self._is_response_envelope(result):
                    stream_token, rid, response_type, payload = result
                    if (
                        stream_token == self._stream_token
                        and rid == request_id
                        and response_type == request_type
                    ):
                        return payload

                    logger.debug(
                        f"Worker {self.worker_id}: Discarded stale/mismatched "
                        f"response (expected token={self._stream_token} "
                        f"id={request_id} type={request_type}, got "
                        f"token={stream_token} id={rid} type={response_type})"
                    )
                    continue

                # Legacy response protocol (raw payload).
                return result
            except queue.Empty:
                pass  # poll interval expired — check RSS then retry

            polls += 1
            rss = get_process_tree_rss_gb()

            # Periodic log so external tooling can correlate RSS with time
            if polls % 6 == 0:  # every ~30 s
                logger.debug(
                    f"Worker {self.worker_id}: waiting for {request_type} "
                    f"({polls * POLL_INTERVAL:.0f}s elapsed, RSS={rss:.2f} GB)"
                )

            if rss > RSS_WARN:
                logger.warning(
                    f"Worker {self.worker_id}: RSS={rss:.2f} GB while waiting "
                    f"for {request_type} (poll #{polls}, "
                    f"warn={RSS_WARN:.1f} GB, cap={RSS_HARD_CAP:.1f} GB)"
                )
                # Attempt cleanup
                gc.collect()
                _malloc_trim()
                rss = get_process_tree_rss_gb()
                if rss > RSS_HARD_CAP:
                    logger.error(
                        f"Worker {self.worker_id}: RSS={rss:.2f} GB EXCEEDS "
                        f"hard cap {RSS_HARD_CAP:.1f} GB after gc+malloc_trim. "
                        f"Aborting {request_type} to prevent OOM."
                    )
                    raise MemoryLimitExceeded(
                        f"Worker {self.worker_id} RSS {rss:.2f} GB > "
                        f"cap {RSS_HARD_CAP:.1f} GB"
                    )

        # Full timeout reached
        logger.error(
            f"Worker {self.worker_id}: Timeout waiting for {request_type} "
            f"response after {self.timeout}s"
        )
        raise InferenceTimeoutError(
            f"Timeout waiting for {request_type} response after {self.timeout}s"
        )


class QueueProxyTransformer(_ProxyMixin, TransformerProtocol):
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
        # Mock tokenizer for AgentRunner if it accesses it directly
        self.tokenizer = None
        self._stream_token = f"worker-{worker_id}-transformer-{uuid.uuid4().hex}"
        self._init_mixin()

    def generate_tactics_with_probs(
        self, state: str, n: int = 1
    ) -> List[Tuple[str, float]]:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "generate_tactics_with_probs",
                (state, n),
            )
        )
        result: List[Tuple[str, float]] = self._get_response(
            "generate_tactics_with_probs", request_id
        )
        assert isinstance(result, list)
        return result

    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "generate_tactics",
                (state, n),
            )
        )
        result: List[str] = self._get_response("generate_tactics", request_id)
        assert isinstance(result, list)
        return result

    def generate_tactics_batch(self, states: List[str], n: int = 1) -> List[List[str]]:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "generate_tactics_batch",
                (states, n),
            )
        )
        result: List[List[str]] = self._get_response(
            "generate_tactics_batch", request_id
        )
        assert isinstance(result, list)
        return result

    def generate_tactics_with_probs_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[Tuple[str, float]]]:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "generate_tactics_with_probs_batch",
                (states, n),
            )
        )
        result: List[List[Tuple[str, float]]] = self._get_response(
            "generate_tactics_with_probs_batch", request_id
        )
        assert isinstance(result, list)
        return result


class QueueProxyValueHead(_ProxyMixin):
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
        self._stream_token = f"worker-{worker_id}-value-head-{uuid.uuid4().hex}"
        self._init_mixin()

    def predict(self, state: str) -> float:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "predict_value",
                (state,),
            )
        )
        result: float = self._get_response("predict_value", request_id)
        assert isinstance(result, float)
        return result

    def predict_batch(self, states: List[str]) -> List[float]:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "predict_batch",
                (states,),
            )
        )
        result: List[float] = self._get_response("predict_batch", request_id)
        assert isinstance(result, list)
        return result

    def encode_states(self, states: List[str]) -> torch.Tensor:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "encode_states",
                (states,),
            )
        )
        result: torch.Tensor = self._get_response("encode_states", request_id)
        assert isinstance(result, torch.Tensor)
        return result

    def predict_from_features(self, features: torch.Tensor) -> float:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "predict_from_features",
                (features,),
            )
        )
        result: float = self._get_response("predict_from_features", request_id)
        assert isinstance(result, float)
        return result

    def predict_from_features_batch(self, features: torch.Tensor) -> List[float]:
        request_id = self._new_request_id()
        self.request_queue.put(
            (
                self.worker_id,
                self._stream_token,
                request_id,
                "predict_from_features_batch",
                (features,),
            )
        )
        result: List[float] = self._get_response(
            "predict_from_features_batch", request_id
        )
        assert isinstance(result, list)
        return result
