"""
Inference Server for centralized model execution.
"""

from typing import List, Union, Sequence
from loguru import logger
import torch
import torch.multiprocessing as mp
import queue
import gc

from lean_reinforcement.utilities.types import (
    InferenceRequest,
    InferencePayload,
    InferenceResult,
)
from lean_reinforcement.utilities.validation import (
    is_tactic_gen_payload_list,
    is_tactic_gen_batch_payload_list,
    is_value_pred_payload_list,
    is_value_pred_batch_payload_list,
    is_feature_payload_list,
    is_feature_batch_payload_list,
    is_encode_states_payload_list,
)

Request = InferenceRequest


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
        # Start with a conservative max_safe_batch_size to prevent OOM
        self.max_safe_batch_size = min(batch_size, 4)
        self._batch_count = 0

    def _proactive_memory_cleanup(self):
        """Proactively clean up GPU memory before processing a batch."""
        self._batch_count += 1
        # Clean up every 5 batches to prevent memory fragmentation
        if self._batch_count % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _check_gpu_memory(self) -> bool:
        """Check if GPU memory usage is too high and clean up if needed.
        Returns True if memory is safe, False if critically low."""
        if not torch.cuda.is_available():
            return True

        allocated = torch.cuda.memory_allocated()
        torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory

        # If using more than 80% of total memory, force cleanup
        if allocated > 0.8 * total:
            logger.warning(
                f"GPU memory high ({allocated / 1e9:.1f}GB / {total / 1e9:.1f}GB). "
                "Forcing cleanup..."
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Reduce batch size more aggressively
            if self.max_safe_batch_size > 1:
                self.max_safe_batch_size = max(1, self.max_safe_batch_size // 2)
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
        batch_requests: List[Request] = []

        target_size = self.batch_size
        if self.max_safe_batch_size < target_size:
            target_size = int(self.max_safe_batch_size)

        # Ensure at least 1
        target_size = max(1, target_size)

        # Try to get as many requests as possible without blocking too long
        try:
            while len(batch_requests) < target_size:
                req = self.request_queue.get_nowait()
                batch_requests.append(req)
        except queue.Empty:
            pass

        if not batch_requests:
            return False

        # Proactive memory cleanup before processing
        self._proactive_memory_cleanup()

        self._process_batch(batch_requests)
        return True

    def _process_batch(self, batch_requests: List[Request]):
        # Helper to extract 'n' from payload safely for sorting
        def get_n(payload: InferencePayload) -> int:
            if len(payload) >= 2 and isinstance(payload[1], int):
                return payload[1]
            return 0

        # Sort by type AND parameter n to ensure safe batching
        def sort_key(req: Request):
            _, req_type, payload = req
            return (req_type, get_n(payload))

        batch_requests.sort(key=sort_key)

        current_type = None
        current_n = -1
        current_batch: List[InferencePayload] = []
        current_indices: List[int] = []

        for worker_id, req_type, payload in batch_requests:
            this_n = get_n(payload)

            # Flush if type changes OR if n changes
            if req_type != current_type or this_n != current_n:
                # Process previous batch
                if current_batch:
                    assert current_type is not None
                    if len(current_batch) > self.max_safe_batch_size:
                        # Split into smaller chunks
                        for i in range(
                            0, len(current_batch), int(self.max_safe_batch_size)
                        ):
                            chunk_batch = current_batch[
                                i : i + int(self.max_safe_batch_size)
                            ]
                            chunk_indices = current_indices[
                                i : i + int(self.max_safe_batch_size)
                            ]
                            self._execute_batch(
                                current_type, chunk_batch, chunk_indices
                            )
                    else:
                        self._execute_batch(
                            current_type, current_batch, current_indices
                        )

                current_type = req_type
                current_n = this_n
                current_batch = []
                current_indices = []

            current_batch.append(payload)
            current_indices.append(worker_id)

        # Process last batch
        if current_batch:
            assert current_type is not None
            if len(current_batch) > self.max_safe_batch_size:
                for i in range(0, len(current_batch), int(self.max_safe_batch_size)):
                    chunk_batch = current_batch[i : i + int(self.max_safe_batch_size)]
                    chunk_indices = current_indices[
                        i : i + int(self.max_safe_batch_size)
                    ]
                    self._execute_batch(current_type, chunk_batch, chunk_indices)
            else:
                self._execute_batch(current_type, current_batch, current_indices)

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

        gc.collect()
        torch.cuda.empty_cache()

        new_limit = len(states) // 2
        if new_limit < 1:
            try:
                return method(states, n=n, **kwargs)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(f"OOM even with single sample! n={n}")

        if new_limit < self.max_safe_batch_size:
            self.max_safe_batch_size = new_limit
            logger.warning(
                f"OOM encountered. Reducing max safe batch size to {self.max_safe_batch_size}"
            )

        mid = len(states) // 2
        left = self._run_transformer_batch(method, states[:mid], n, **kwargs)
        right = self._run_transformer_batch(method, states[mid:], n, **kwargs)
        return left + right

    def _execute_batch(
        self, req_type: str, batch: List[InferencePayload], indices: List[int]
    ):
        execution_results: List[InferenceResult] = []

        if req_type == "generate_tactics_with_probs":
            assert is_tactic_gen_payload_list(batch)
            batch_gen = batch
            states_tuple, ns_tuple = zip(*batch_gen)
            n = ns_tuple[0]
            execution_results = self._run_transformer_batch(
                self.transformer.generate_tactics_with_probs_batch,
                list(states_tuple),
                n=n,
            )
        elif req_type == "generate_tactics":
            assert is_tactic_gen_payload_list(batch)
            batch_gen = batch
            states_tuple, ns_tuple = zip(*batch_gen)
            n = ns_tuple[0]
            execution_results = self._run_transformer_batch(
                self.transformer.generate_tactics_batch, list(states_tuple), n=n
            )
        elif req_type == "generate_tactics_batch":
            # payload is (states, n)
            assert is_tactic_gen_batch_payload_list(batch)
            batch_batch = batch
            # Flatten
            all_states_list: List[str] = []
            lengths_list: List[int] = []
            ns_list: List[int] = []
            for gen_batch_payload in batch_batch:
                s, n = gen_batch_payload
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
            assert is_tactic_gen_batch_payload_list(batch)
            batch_batch = batch
            all_states_list_probs: List[str] = []
            lengths_list_probs: List[int] = []
            ns_list_probs: List[int] = []
            for gen_probs_batch_payload in batch_batch:
                s, n = gen_probs_batch_payload
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
                assert is_value_pred_payload_list(batch)
                batch_val = batch
                states = [val_payload[0] for val_payload in batch_val]
                execution_results = self.value_head.predict_batch(list(states))
            else:
                logger.error("Received predict_value request but value_head is None")
                execution_results = [0.0] * len(batch)

        elif req_type == "predict_batch":
            if self.value_head is not None:
                assert is_value_pred_batch_payload_list(batch)
                batch_val_batch = batch
                all_states = []
                lengths = []
                for val_batch_payload in batch_val_batch:
                    s = val_batch_payload[0]
                    all_states.extend(s)
                    lengths.append(len(s))

                all_results = self.value_head.predict_batch(all_states)

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
                assert is_encode_states_payload_list(batch)
                batch_encode = batch
                all_states = []
                lengths = []
                for encode_payload in batch_encode:
                    s = encode_payload[0]
                    all_states.extend(s)
                    lengths.append(len(s))

                features = self.value_head.encode_states(all_states)

                execution_results = []
                start = 0
                for length in lengths:
                    res = features[start : start + length]
                    execution_results.append(res.cpu())  # Move to CPU
                    start += length
            else:
                logger.error("Received encode_states request but value_head is None")
                execution_results = [None for _ in batch]

        elif req_type == "predict_from_features":
            if self.value_head is not None:
                assert is_feature_payload_list(batch)
                batch_feat = batch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                features_batch = []
                for feat_payload in batch_feat:
                    f = feat_payload[0].to(device)
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
                assert is_feature_batch_payload_list(batch)
                batch_feat_batch = batch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                features_list = [p[0].to(device) for p in batch_feat_batch]
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

        # Send responses
        for i, res in enumerate(execution_results):
            self.response_queues[indices[i]].put(res)
