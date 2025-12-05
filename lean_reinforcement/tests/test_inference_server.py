import unittest
from unittest.mock import MagicMock
import queue
import torch
from lean_reinforcement.training.inference_server import InferenceServer


class TestInferenceServer(unittest.TestCase):
    def setUp(self):
        self.transformer = MagicMock()
        self.value_head = MagicMock()
        self.request_queue = queue.Queue()
        self.response_queues = [queue.Queue() for _ in range(4)]
        self.batch_size = 4
        self.server = InferenceServer(
            self.transformer,
            self.value_head,
            self.request_queue,
            self.response_queues,
            self.batch_size,
        )

    def test_process_requests_empty(self):
        processed = self.server.process_requests()
        self.assertFalse(processed)

    def test_process_requests_single_batch(self):
        # Setup a request
        # Request format: (worker_id, req_type, payload)
        worker_id = 0
        req_type = "generate_tactics"
        state = "state1"
        n = 5
        payload = (state, n)

        self.request_queue.put((worker_id, req_type, payload))

        # Mock transformer response
        self.transformer.generate_tactics_batch.return_value = [["tactic1"]]

        processed = self.server.process_requests()
        self.assertTrue(processed)

        # Check transformer call
        self.transformer.generate_tactics_batch.assert_called_once()
        args, kwargs = self.transformer.generate_tactics_batch.call_args
        self.assertEqual(args[0], ["state1"])
        self.assertEqual(kwargs["n"], 5)

        # Check response
        response = self.response_queues[worker_id].get_nowait()
        self.assertEqual(response, ["tactic1"])

    def test_process_requests_mixed_batch(self):
        # Add requests of different types
        # 1. generate_tactics
        self.request_queue.put((0, "generate_tactics", ("s1", 1)))
        # 2. predict_value
        self.request_queue.put((1, "predict_value", ("s2",)))

        # Mock responses
        self.transformer.generate_tactics_batch.return_value = [["t1"]]
        self.value_head.predict_batch.return_value = [0.5]

        processed = self.server.process_requests()
        self.assertTrue(processed)

        # Check calls
        self.transformer.generate_tactics_batch.assert_called_once()
        self.value_head.predict_batch.assert_called_once()

        # Check responses
        self.assertEqual(self.response_queues[0].get_nowait(), ["t1"])
        self.assertEqual(self.response_queues[1].get_nowait(), 0.5)

    def test_predict_value_batch(self):
        # Test batching of predict_value
        self.request_queue.put((0, "predict_value", ("s1",)))
        self.request_queue.put((1, "predict_value", ("s2",)))

        self.value_head.predict_batch.return_value = [0.1, 0.2]

        processed = self.server.process_requests()
        self.assertTrue(processed)

        # Should be called once with both states
        self.value_head.predict_batch.assert_called_once()
        args, _ = self.value_head.predict_batch.call_args
        self.assertEqual(args[0], ["s1", "s2"])

        self.assertEqual(self.response_queues[0].get_nowait(), 0.1)
        self.assertEqual(self.response_queues[1].get_nowait(), 0.2)

    def test_encode_states(self):
        self.request_queue.put((0, "encode_states", (["s1"],)))

        # Mock return value needs to be a tensor
        mock_tensor = torch.tensor([[1.0]])
        self.value_head.encode_states.return_value = mock_tensor

        processed = self.server.process_requests()
        self.assertTrue(processed)

        self.value_head.encode_states.assert_called_once()

        res = self.response_queues[0].get_nowait()
        self.assertTrue(torch.equal(res, mock_tensor))

    def test_predict_from_features(self):
        feature = torch.tensor([1.0])
        self.request_queue.put((0, "predict_from_features", (feature,)))

        self.value_head.predict_from_features_batch.return_value = [0.9]

        processed = self.server.process_requests()
        self.assertTrue(processed)

        self.value_head.predict_from_features_batch.assert_called_once()
        res = self.response_queues[0].get_nowait()
        self.assertEqual(res, 0.9)

    def test_mixed_n_batching(self):
        # Test that requests with different 'n' are not batched together
        req_type = "generate_tactics"

        # Request 1: n=5
        self.request_queue.put((0, req_type, ("state1", 5)))
        # Request 2: n=10
        self.request_queue.put((1, req_type, ("state2", 10)))

        # Mock transformer response
        # We expect two calls.
        # The order depends on sorting. (req_type, n)
        # ("generate_tactics", 5) comes before ("generate_tactics", 10)
        self.transformer.generate_tactics_batch.side_effect = [
            [["tactic1"]],  # For n=5
            [["tactic2"]],  # For n=10
        ]

        processed = self.server.process_requests()
        self.assertTrue(processed)

        # Verify two separate calls
        self.assertEqual(self.transformer.generate_tactics_batch.call_count, 2)

        # Verify call arguments
        calls = self.transformer.generate_tactics_batch.call_args_list

        # First call should be for n=5
        args1, kwargs1 = calls[0]
        self.assertEqual(args1[0], ["state1"])
        self.assertEqual(kwargs1["n"], 5)

        # Second call should be for n=10
        args2, kwargs2 = calls[1]
        self.assertEqual(args2[0], ["state2"])
        self.assertEqual(kwargs2["n"], 10)

    def test_oom_handling(self):
        # Test that OOM is handled by splitting the batch
        req_type = "generate_tactics_batch"
        # Create a batch of 4 states
        states = ["s1", "s2", "s3", "s4"]
        n = 5
        # One request with 4 states
        self.request_queue.put((0, req_type, (states, n)))

        # Mock transformer response
        # First call (full batch) raises OOM
        # Second call (first half: s1, s2) succeeds
        # Third call (second half: s3, s4) succeeds

        def side_effect(batch_states, n=1):
            if len(batch_states) == 4:
                raise torch.cuda.OutOfMemoryError("OOM")
            return [["t"] for _ in batch_states]

        self.transformer.generate_tactics_batch.side_effect = side_effect

        processed = self.server.process_requests()
        self.assertTrue(processed)

        # Verify calls
        # 1. Call with 4 states -> OOM
        # 2. Call with 2 states (s1, s2) -> OK
        # 3. Call with 2 states (s3, s4) -> OK
        self.assertEqual(self.transformer.generate_tactics_batch.call_count, 3)

        # Verify result
        res = self.response_queues[0].get_nowait()
        self.assertEqual(len(res), 4)

    def test_oom_handling_persistent(self):
        # Test that max_safe_batch_size is remembered
        self.server.batch_size = 1  # Process one request at a time

        req_type = "generate_tactics_batch"
        states = ["s1", "s2", "s3", "s4"]
        n = 5

        # First request: triggers OOM
        self.request_queue.put((0, req_type, (states, n)))

        def side_effect(batch_states, n=1):
            if len(batch_states) == 4:
                raise torch.cuda.OutOfMemoryError("OOM")
            return [["t"] for _ in batch_states]

        self.transformer.generate_tactics_batch.side_effect = side_effect

        # Process first request
        self.server.process_requests()
        # Should have called 3 times (4->fail, 2->ok, 2->ok)
        self.assertEqual(self.transformer.generate_tactics_batch.call_count, 3)
        self.assertEqual(self.server.max_safe_batch_size, 2)

        # Reset mock call count
        self.transformer.generate_tactics_batch.reset_mock()
        self.transformer.generate_tactics_batch.side_effect = side_effect

        # Second request: should be split preemptively
        self.request_queue.put((1, req_type, (states, n)))

        # Process second request
        self.server.process_requests()
        # Should have called 2 times (2->ok, 2->ok) because 4 is > max_safe_batch_size (2)
        # It should NOT call with 4
        self.assertEqual(self.transformer.generate_tactics_batch.call_count, 2)

        # Verify no call with length 4
        for call in self.transformer.generate_tactics_batch.call_args_list:
            args, _ = call
            self.assertLessEqual(len(args[0]), 2)


if __name__ == "__main__":
    unittest.main()
