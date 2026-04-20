"""
Inference Server for centralized model execution.
"""

from typing import List, Any, Optional, Union, Sequence, Tuple
from loguru import logger
import torch
import torch.multiprocessing as mp
import queue

from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    empty_gpu_cache,
    get_gpu_memory_usage_percent,
    GPU_CLEANUP_THRESHOLD_PERCENT,
)

LegacyRequest = Tuple[int, str, Tuple[Any, ...]]
RequestWithId = Tuple[int, int, str, Tuple[Any, ...]]
RequestWithToken = Tuple[int, str, int, str, Tuple[Any, ...]]
NormalizedRequest = Tuple[int, Optional[str], Optional[int], str, Tuple[Any, ...]]


class InferenceServer:
    def __init__(
        self,
        transformer,
        value_head,
        request_queue: Union[mp.Queue, queue.Queue],
        response_queues: Sequence[Union[mp.Queue, queue.Queue]],
        batch_size: int,
    ):
        self.transformer = transformer
        self.value_head = value_head
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.batch_size = batch_size
        self.max_safe_batch_size = batch_size
        self._batch_count = 0
        self._batches_since_last_oom = 0  # Track batches since last OOM for recovery

    def _proactive_memory_cleanup(self):
        """Proactively clean up GPU memory before processing a batch."""
        self._batch_count += 1
        self._batches_since_last_oom += 1
        # Clean up every 5 batches to prevent memory fragmentation
        if self._batch_count % 5 == 0:
            aggressive_cleanup()
            empty_gpu_cache()

        # Periodically try to recover batch size (every 50 successful batches)
        if (
            self._batches_since_last_oom >= 50
            and self.max_safe_batch_size < self.batch_size
        ):
            new_size = min(self.max_safe_batch_size + 1, self.batch_size)
            logger.debug(
                f"Recovering max_safe_batch_size: {self.max_safe_batch_size} -> {new_size}"
            )
            self.max_safe_batch_size = new_size
            self._batches_since_last_oom = 0

    def _check_gpu_memory(self) -> bool:
        """Check if GPU memory usage is too high and clean up if needed.
        Returns True if memory is safe, False if critically low."""
        if not torch.cuda.is_available():
            return True

        usage_pct = get_gpu_memory_usage_percent()

        if usage_pct > GPU_CLEANUP_THRESHOLD_PERCENT:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            logger.warning(
                f"GPU memory high ({allocated / 1e9:.1f}GB allocated, "
                f"{reserved / 1e9:.1f}GB reserved / {total / 1e9:.1f}GB total). "
                "Forcing cleanup..."
            )
            aggressive_cleanup()
            empty_gpu_cache()

            # Reduce batch size more aggressively
            if self.max_safe_batch_size > 1:
                self.max_safe_batch_size = max(1, self.max_safe_batch_size // 2)
                self._batches_since_last_oom = 0
                logger.warning(
                    f"Reducing max_safe_batch_size to {self.max_safe_batch_size}"
                )
            return False
        return True

    def process_requests(self) -> bool:
        """
        Collects a batch of requests and processes them.
        Returns True if a batch was processed, False otherwise.
        """
        batch_requests: List[NormalizedRequest] = []

        target_size = self.batch_size
        if self.max_safe_batch_size < target_size:
            target_size = int(self.max_safe_batch_size)

        # Ensure at least 1
        target_size = max(1, target_size)

        # Try to get as many requests as possible without blocking too long
        try:
            while len(batch_requests) < target_size:
                req = self.request_queue.get_nowait()
                normalized = self._normalize_request(req)
                if normalized is None:
                    logger.warning(f"Skipping malformed request: {req!r}")
                    continue
                batch_requests.append(normalized)
        except queue.Empty:
            pass

        if not batch_requests:
            return False

        # Proactive memory cleanup before processing
        self._proactive_memory_cleanup()

        self._process_batch(batch_requests)
        return True

    def _normalize_request(self, request: Any) -> Optional[NormalizedRequest]:
        """Normalize request envelopes across protocol versions.

        Supported formats:
          1) Legacy: (worker_id, req_type, payload)
          2) Request ID: (worker_id, request_id, req_type, payload)
          3) Stream + ID: (worker_id, stream_token, request_id, req_type, payload)
        """
        if not isinstance(request, tuple):
            return None

        if len(request) == 3:
            worker_id, req_type, payload = request
            if (
                isinstance(worker_id, int)
                and isinstance(req_type, str)
                and isinstance(payload, tuple)
            ):
                return (worker_id, None, None, req_type, payload)
            return None

        if len(request) == 4:
            worker_id, request_id, req_type, payload = request
            if (
                isinstance(worker_id, int)
                and isinstance(request_id, int)
                and isinstance(req_type, str)
                and isinstance(payload, tuple)
            ):
                return (worker_id, None, request_id, req_type, payload)
            return None

        if len(request) == 5:
            worker_id, stream_token, request_id, req_type, payload = request
            if (
                isinstance(worker_id, int)
                and isinstance(stream_token, str)
                and isinstance(request_id, int)
                and isinstance(req_type, str)
                and isinstance(payload, tuple)
            ):
                return (worker_id, stream_token, request_id, req_type, payload)
            return None

        return None

    def _send_response(
        self,
        worker_id: int,
        stream_token: Optional[str],
        request_id: Optional[int],
        req_type: str,
        payload: Any,
    ) -> None:
        """Send a response using the same protocol family as the request."""
        if stream_token is not None and request_id is not None:
            self.response_queues[worker_id].put(
                (stream_token, request_id, req_type, payload)
            )
            return
        if request_id is not None:
            self.response_queues[worker_id].put((request_id, req_type, payload))
            return
        self.response_queues[worker_id].put(payload)

    def _fallback_response(self, req_type: str) -> Any:
        """Return a type-compatible fallback payload for failed requests."""
        if req_type in {"predict_value", "predict_from_features"}:
            return 0.0
        if req_type == "encode_states":
            return torch.tensor([])
        return []

    def _process_batch(self, batch_requests: List[NormalizedRequest]):
        # Helper to extract 'n' from payload safely for sorting
        def get_n(payload: Tuple[Any, ...]) -> int:
            if len(payload) >= 2 and isinstance(payload[1], int):
                return payload[1]
            return 0

        # Sort by type AND parameter n to ensure safe batching
        def sort_key(req: NormalizedRequest):
            _, _, _, req_type, payload = req
            return (req_type, get_n(payload))

        batch_requests.sort(key=sort_key)

        current_type = None
        current_n = -1
        current_batch: List[Tuple[Any, ...]] = []
        current_indices: List[int] = []
        current_stream_tokens: List[Optional[str]] = []
        current_request_ids: List[Optional[int]] = []

        def flush_current() -> None:
            if not current_batch:
                return

            assert current_type is not None
            if len(current_batch) > self.max_safe_batch_size:
                for i in range(0, len(current_batch), int(self.max_safe_batch_size)):
                    chunk_batch = current_batch[i : i + int(self.max_safe_batch_size)]
                    chunk_indices = current_indices[
                        i : i + int(self.max_safe_batch_size)
                    ]
                    chunk_stream_tokens = current_stream_tokens[
                        i : i + int(self.max_safe_batch_size)
                    ]
                    chunk_request_ids = current_request_ids[
                        i : i + int(self.max_safe_batch_size)
                    ]
                    self._execute_batch(
                        current_type,
                        chunk_batch,
                        chunk_indices,
                        chunk_stream_tokens,
                        chunk_request_ids,
                    )
            else:
                self._execute_batch(
                    current_type,
                    current_batch,
                    current_indices,
                    current_stream_tokens,
                    current_request_ids,
                )

        for worker_id, stream_token, request_id, req_type, payload in batch_requests:
            this_n = get_n(payload)

            # Flush if type changes OR if n changes
            if req_type != current_type or this_n != current_n:
                flush_current()

                current_type = req_type
                current_n = this_n
                current_batch = []
                current_indices = []
                current_stream_tokens = []
                current_request_ids = []

            current_batch.append(payload)
            current_indices.append(worker_id)
            current_stream_tokens.append(stream_token)
            current_request_ids.append(request_id)

        # Process last batch
        flush_current()

    def _run_transformer_batch(self, method, states, n, **kwargs):
        # Preemptive memory check
        self._check_gpu_memory()

        if len(states) > self.max_safe_batch_size:
            mid = len(states) // 2
            left = self._run_transformer_batch(method, states[:mid], n, **kwargs)
            right = self._run_transformer_batch(method, states[mid:], n, **kwargs)
            return left + right

        try:
            return method(states, n=n, **kwargs)
        except torch.cuda.OutOfMemoryError:
            pass

        aggressive_cleanup()
        empty_gpu_cache()

        new_limit = len(states) // 2
        if new_limit < 1:
            try:
                return method(states, n=n, **kwargs)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(f"OOM even with single sample! n={n}")

        if new_limit < self.max_safe_batch_size:
            self.max_safe_batch_size = new_limit
            self._batches_since_last_oom = 0
            logger.warning(
                f"OOM encountered. Reducing max safe batch size to {self.max_safe_batch_size}"
            )

        mid = len(states) // 2
        left = self._run_transformer_batch(method, states[:mid], n, **kwargs)
        right = self._run_transformer_batch(method, states[mid:], n, **kwargs)
        return left + right

    def _execute_batch(
        self,
        req_type: str,
        batch: List[Any],
        indices: List[int],
        stream_tokens: List[Optional[str]],
        request_ids: List[Optional[int]],
    ):
        try:
            self._execute_batch_inner(
                req_type, batch, indices, stream_tokens, request_ids
            )
        except Exception as exc:
            # CRITICAL: On any failure, send fallback responses to ALL
            # workers in this batch.  Without this, workers hang forever
            # on response_queue.get() because no response is ever sent.
            logger.error(
                f"Inference failed for {req_type} " f"(workers {indices}): {exc}"
            )
            fallback = self._fallback_response(req_type)
            for i, idx in enumerate(indices):
                try:
                    self._send_response(
                        idx,
                        stream_tokens[i],
                        request_ids[i],
                        req_type,
                        fallback,
                    )
                except Exception:
                    pass

    def _execute_batch_inner(
        self,
        req_type: str,
        batch: List[Any],
        indices: List[int],
        stream_tokens: List[Optional[str]],
        request_ids: List[Optional[int]],
    ):
        execution_results: List[Any] = []

        if req_type == "generate_tactics_with_probs":
            states_tuple, ns_tuple = zip(*batch)
            n = ns_tuple[0]
            execution_results = self._run_transformer_batch(
                self.transformer.generate_tactics_with_probs_batch,
                list(states_tuple),
                n=n,
            )
        elif req_type == "generate_tactics":
            states_tuple, ns_tuple = zip(*batch)
            n = ns_tuple[0]
            execution_results = self._run_transformer_batch(
                self.transformer.generate_tactics_batch, list(states_tuple), n=n
            )
        elif req_type == "generate_tactics_batch":
            # payload is (states, n)
            # Flatten
            all_states_list: List[str] = []
            lengths_list: List[int] = []
            ns_list: List[int] = []
            for p in batch:
                s, n = p
                all_states_list.extend(s)
                lengths_list.append(len(s))
                ns_list.append(n)

            n = ns_list[0]
            all_results = self._run_transformer_batch(
                self.transformer.generate_tactics_batch, all_states_list, n=n
            )

            # Split back
            execution_results = []
            start = 0
            for length in lengths_list:
                execution_results.append(all_results[start : start + length])
                start += length

        elif req_type == "generate_tactics_with_probs_batch":
            # payload is (states, n)
            all_states_list_probs: List[str] = []
            lengths_list_probs: List[int] = []
            ns_list_probs: List[int] = []
            for p in batch:
                s, n = p
                all_states_list_probs.extend(s)
                lengths_list_probs.append(len(s))
                ns_list_probs.append(n)

            n = ns_list_probs[0]
            all_results = self._run_transformer_batch(
                self.transformer.generate_tactics_with_probs_batch,
                all_states_list_probs,
                n=n,
            )

            execution_results = []
            start = 0
            for length in lengths_list_probs:
                execution_results.append(all_results[start : start + length])
                start += length

        elif req_type == "predict_value":
            if self.value_head is not None:
                states = [p[0] for p in batch]
                execution_results = self.value_head.predict_batch(list(states))
            else:
                logger.error("Received predict_value request but value_head is None")
                execution_results = [0.0] * len(batch)

        elif req_type == "predict_batch":
            if self.value_head is not None:
                all_states = []
                lengths = []
                for p in batch:
                    s = p[0]
                    all_states.extend(s)
                    lengths.append(len(s))

                all_results = []
                chunk_size = int(self.max_safe_batch_size)
                for i in range(0, len(all_states), chunk_size):
                    chunk = all_states[i : i + chunk_size]
                    try:
                        all_results.extend(self.value_head.predict_batch(chunk))
                    except torch.cuda.OutOfMemoryError:
                        aggressive_cleanup()
                        empty_gpu_cache()
                        new_limit = len(chunk) // 2
                        if new_limit < self.max_safe_batch_size:
                            self.max_safe_batch_size = max(1, new_limit)
                            self._batches_since_last_oom = 0
                            logger.warning(
                                f"OOM in value_head. Reducing max safe batch size to {self.max_safe_batch_size}"
                            )
                        # Retry with smaller chunks
                        for j in range(0, len(chunk), max(1, new_limit)):
                            sub_chunk = chunk[j : j + max(1, new_limit)]
                            all_results.extend(self.value_head.predict_batch(sub_chunk))

                execution_results = []
                start = 0
                for length in lengths:
                    execution_results.append(all_results[start : start + length])
                    start += length
            else:
                logger.error("Received predict_batch request but value_head is None")
                execution_results = [[] for _ in batch]

        elif req_type == "encode_states":
            if self.value_head is not None:
                all_states = []
                lengths = []
                for p in batch:
                    s = p[0]
                    all_states.extend(s)
                    lengths.append(len(s))

                # Process in chunks with OOM protection (same as predict_batch)
                all_features = []
                chunk_size = int(self.max_safe_batch_size)
                for i in range(0, len(all_states), chunk_size):
                    chunk = all_states[i : i + chunk_size]
                    try:
                        chunk_features = self.value_head.encode_states(chunk)
                        all_features.append(chunk_features.cpu())
                    except torch.cuda.OutOfMemoryError:
                        aggressive_cleanup()
                        empty_gpu_cache()
                        new_limit = len(chunk) // 2
                        if new_limit < self.max_safe_batch_size:
                            self.max_safe_batch_size = max(1, new_limit)
                            self._batches_since_last_oom = 0
                            logger.warning(
                                f"OOM in encode_states. Reducing max safe batch size to {self.max_safe_batch_size}"
                            )
                        # Retry with smaller chunks
                        for j in range(0, len(chunk), max(1, new_limit)):
                            sub_chunk = chunk[j : j + max(1, new_limit)]
                            sub_features = self.value_head.encode_states(sub_chunk)
                            all_features.append(sub_features.cpu())

                features = (
                    torch.cat(all_features, dim=0) if all_features else torch.tensor([])
                )

                execution_results = []
                start = 0
                for length in lengths:
                    res = features[start : start + length]
                    execution_results.append(res)
                    start += length
            else:
                logger.error("Received encode_states request but value_head is None")
                execution_results = [None for _ in batch]

        elif req_type == "predict_from_features":
            if self.value_head is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                features_batch = []
                for p in batch:
                    f = p[0].to(device)
                    if f.ndim == 1:
                        f = f.unsqueeze(0)
                    features_batch.append(f)

                if features_batch:
                    full_batch = torch.cat(features_batch, dim=0)
                    execution_results = self.value_head.predict_from_features_batch(
                        full_batch
                    )
                else:
                    execution_results = []
            else:
                logger.error(
                    "Received predict_from_features request but value_head is None"
                )
                execution_results = [0.0] * len(batch)

        elif req_type == "predict_from_features_batch":
            if self.value_head is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                features_list = [p[0].to(device) for p in batch]
                full_batch = torch.cat(features_list, dim=0)
                all_results = self.value_head.predict_from_features_batch(full_batch)

                execution_results = []
                start = 0
                for f in features_list:
                    length = f.shape[0]
                    execution_results.append(all_results[start : start + length])
                    start += length
            else:
                logger.error(
                    "Received predict_from_features_batch request but value_head is None"
                )
                execution_results = [[] for _ in batch]

        if len(execution_results) != len(indices):
            raise RuntimeError(
                f"Execution result size mismatch for {req_type}: "
                f"results={len(execution_results)} requests={len(indices)}"
            )

        # Send responses
        for i, res in enumerate(execution_results):
            self._send_response(
                indices[i],
                stream_tokens[i],
                request_ids[i],
                req_type,
                res,
            )
