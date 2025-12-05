import unittest
from unittest.mock import MagicMock
import torch
from lean_reinforcement.agent.proxies import QueueProxyTransformer, QueueProxyValueHead
from lean_reinforcement.agent.transformer import TransformerProtocol


class TestQueueProxyTransformer(unittest.TestCase):
    def setUp(self):
        self.request_queue = MagicMock()
        self.response_queue = MagicMock()
        self.worker_id = 1
        self.proxy = QueueProxyTransformer(
            self.request_queue, self.response_queue, self.worker_id
        )

    def test_is_transformer_protocol(self):
        self.assertIsInstance(self.proxy, TransformerProtocol)

    def test_generate_tactics(self):
        state = "example_state"
        n = 5
        expected_result = ["tactic1", "tactic2"]
        self.response_queue.get.return_value = expected_result

        result = self.proxy.generate_tactics(state, n)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "generate_tactics", (state, n))
        )
        self.assertEqual(result, expected_result)

    def test_generate_tactics_with_probs(self):
        state = "example_state"
        n = 5
        expected_result = [("tactic1", 0.9), ("tactic2", 0.1)]
        self.response_queue.get.return_value = expected_result

        result = self.proxy.generate_tactics_with_probs(state, n)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "generate_tactics_with_probs", (state, n))
        )
        self.assertEqual(result, expected_result)

    def test_generate_tactics_batch(self):
        states = ["state1", "state2"]
        n = 2
        expected_result = [["t1", "t2"], ["t3", "t4"]]
        self.response_queue.get.return_value = expected_result

        result = self.proxy.generate_tactics_batch(states, n)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "generate_tactics_batch", (states, n))
        )
        self.assertEqual(result, expected_result)

    def test_generate_tactics_with_probs_batch(self):
        states = ["state1"]
        n = 1
        expected_result = [[("t1", 1.0)]]
        self.response_queue.get.return_value = expected_result

        result = self.proxy.generate_tactics_with_probs_batch(states, n)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "generate_tactics_with_probs_batch", (states, n))
        )
        self.assertEqual(result, expected_result)


class TestQueueProxyValueHead(unittest.TestCase):
    def setUp(self):
        self.request_queue = MagicMock()
        self.response_queue = MagicMock()
        self.worker_id = 2
        self.proxy = QueueProxyValueHead(
            self.request_queue, self.response_queue, self.worker_id
        )

    def test_predict(self):
        state = "state"
        expected_result = 0.5
        self.response_queue.get.return_value = expected_result

        result = self.proxy.predict(state)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "predict_value", (state,))
        )
        self.assertEqual(result, expected_result)

    def test_predict_batch(self):
        states = ["s1", "s2"]
        expected_result = [0.1, 0.9]
        self.response_queue.get.return_value = expected_result

        result = self.proxy.predict_batch(states)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "predict_batch", (states,))
        )
        self.assertEqual(result, expected_result)

    def test_encode_states(self):
        states = ["s1"]
        expected_result = torch.tensor([1.0, 2.0])
        self.response_queue.get.return_value = expected_result

        result = self.proxy.encode_states(states)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "encode_states", (states,))
        )
        self.assertTrue(torch.equal(result, expected_result))

    def test_predict_from_features(self):
        features = torch.tensor([1.0])
        expected_result = 0.3
        self.response_queue.get.return_value = expected_result

        result = self.proxy.predict_from_features(features)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "predict_from_features", (features,))
        )
        self.assertEqual(result, expected_result)

    def test_predict_from_features_batch(self):
        features = torch.tensor([[1.0], [2.0]])
        expected_result = [0.3, 0.4]
        self.response_queue.get.return_value = expected_result

        result = self.proxy.predict_from_features_batch(features)

        self.request_queue.put.assert_called_with(
            (self.worker_id, "predict_from_features_batch", (features,))
        )
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
