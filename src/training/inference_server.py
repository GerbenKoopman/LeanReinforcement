"""
TODO: Add module docstring
"""

from typing import List, Any, Union, Sequence
from loguru import logger
import torch
import torch.multiprocessing as mp
import queue


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

    def process_requests(self) -> bool:
        """
        Collects a batch of requests and processes them.
        Returns True if a batch was processed, False otherwise.
        """
        batch_requests = []

        # Try to get as many requests as possible without blocking too long
        try:
            while len(batch_requests) < self.batch_size:
                req = self.request_queue.get_nowait()
                batch_requests.append(req)
        except queue.Empty:
            pass

        if not batch_requests:
            return False

        self._process_batch(batch_requests)
        return True

    def _process_batch(self, batch_requests: List[Any]):
        # Sort by type to batch efficiently
        batch_requests.sort(key=lambda x: x[1])

        current_type = None
        current_batch = []
        current_indices = []

        for worker_id, req_type, payload in batch_requests:
            if req_type != current_type:
                # Process previous batch
                if current_batch:
                    assert current_type is not None
                    self._execute_batch(current_type, current_batch, current_indices)

                current_type = req_type
                current_batch = []
                current_indices = []

            current_batch.append(payload)
            current_indices.append(worker_id)

        # Process last batch
        if current_batch:
            assert current_type is not None
            self._execute_batch(current_type, current_batch, current_indices)

    def _execute_batch(self, req_type: str, batch: List[Any], indices: List[int]):
        results = []

        if req_type == "generate_tactics_with_probs":
            states, ns = zip(*batch)
            n = ns[0]
            results = self.transformer.generate_tactics_with_probs_batch(
                list(states), n=n
            )
        elif req_type == "generate_tactics":
            states, ns = zip(*batch)
            n = ns[0]
            results = self.transformer.generate_tactics_batch(list(states), n=n)
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
            all_results = self.transformer.generate_tactics_batch(all_states, n=n)

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
            all_results = self.transformer.generate_tactics_with_probs_batch(
                all_states, n=n
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
