import unittest
from unittest.mock import MagicMock
import queue
import torch
from src.training.inference_server import InferenceServer


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


if __name__ == "__main__":
    unittest.main()
