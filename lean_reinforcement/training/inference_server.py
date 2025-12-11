"""
Inference Server for centralized model execution.
"""

from typing import List, Any, Union, Sequence
from loguru import logger
import torch
import torch.multiprocessing as mp
import queue
import gc


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
        self.max_safe_batch_size = float("inf")

    def process_requests(self) -> bool:
        """
        Collects a batch of requests and processes them.
        Returns True if a batch was processed, False otherwise.
        """
        batch_requests = []

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

        self._process_batch(batch_requests)
        return True

    def _process_batch(self, batch_requests: List[Any]):
        # Helper to extract 'n' from payload safely for sorting
        def get_n(payload):
            if (
                isinstance(payload, tuple)
                and len(payload) >= 2
                and isinstance(payload[1], int)
            ):
                return payload[1]
            return 0

        # Sort by type AND parameter n to ensure safe batching
        def sort_key(req):
            _, req_type, payload = req
            return (req_type, get_n(payload))

        batch_requests.sort(key=sort_key)

        current_type = None
        current_n = -1
        current_batch = []
        current_indices = []

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

    def _execute_batch(self, req_type: str, batch: List[Any], indices: List[int]):
        results = []

        if req_type == "generate_tactics_with_probs":
            states, ns = zip(*batch)
            n = ns[0]
            results = self._run_transformer_batch(
                self.transformer.generate_tactics_with_probs_batch, list(states), n=n
            )
        elif req_type == "generate_tactics":
            states, ns = zip(*batch)
            n = ns[0]
            results = self._run_transformer_batch(
                self.transformer.generate_tactics_batch, list(states), n=n
            )
        elif req_type == "generate_tactics_batch":
            # payload is (states, n)
            # Flatten
            all_states = []
            lengths = []
            ns = []
            for p in batch:
                s, n = p
                all_states.extend(s)
                lengths.append(len(s))
                ns.append(n)

            n = ns[0]
            all_results = self._run_transformer_batch(
                self.transformer.generate_tactics_batch, all_states, n=n
            )

            # Split back
            results = []
            start = 0
            for length in lengths:
                results.append(all_results[start : start + length])
                start += length

        elif req_type == "generate_tactics_with_probs_batch":
            # payload is (states, n)
            all_states = []
            lengths = []
            ns = []
            for p in batch:
                s, n = p
                all_states.extend(s)
                lengths.append(len(s))
                ns.append(n)

            n = ns[0]
            all_results = self._run_transformer_batch(
                self.transformer.generate_tactics_with_probs_batch, all_states, n=n
            )

            results = []
            start = 0
            for length in lengths:
                results.append(all_results[start : start + length])
                start += length

        elif req_type == "predict_value":
            if self.value_head is not None:
                states = [p[0] for p in batch]
                results = self.value_head.predict_batch(list(states))
            else:
                logger.error("Received predict_value request but value_head is None")
                results = [0.0] * len(batch)

        elif req_type == "predict_batch":
            if self.value_head is not None:
                all_states = []
                lengths = []
                for p in batch:
                    s = p[0]
                    all_states.extend(s)
                    lengths.append(len(s))

                all_results = self.value_head.predict_batch(all_states)

                results = []
                start = 0
                for length in lengths:
                    results.append(all_results[start : start + length])
                    start += length
            else:
                logger.error("Received predict_batch request but value_head is None")
                results = [[] for _ in batch]

        elif req_type == "encode_states":
            if self.value_head is not None:
                all_states = []
                lengths = []
                for p in batch:
                    s = p[0]
                    all_states.extend(s)
                    lengths.append(len(s))

                features = self.value_head.encode_states(all_states)

                results = []
                start = 0
                for length in lengths:
                    res = features[start : start + length]
                    results.append(res.cpu())  # Move to CPU
                    start += length
            else:
                logger.error("Received encode_states request but value_head is None")
                results = [None for _ in batch]

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
                    results = self.value_head.predict_from_features_batch(full_batch)
                else:
                    results = []
            else:
                logger.error(
                    "Received predict_from_features request but value_head is None"
                )
                results = [0.0] * len(batch)

        elif req_type == "predict_from_features_batch":
            if self.value_head is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                features_list = [p[0].to(device) for p in batch]
                full_batch = torch.cat(features_list, dim=0)
                all_results = self.value_head.predict_from_features_batch(full_batch)

                results = []
                start = 0
                for f in features_list:
                    length = f.shape[0]
                    results.append(all_results[start : start + length])
                    start += length
            else:
                logger.error(
                    "Received predict_from_features_batch request but value_head is None"
                )
                results = [[] for _ in batch]

        # Send responses
        for i, res in enumerate(results):
            self.response_queues[indices[i]].put(res)
