import unittest
import torch
from unittest.mock import MagicMock, patch
from lean_reinforcement.agent.transformer import Transformer


class TestBatchTransformer(unittest.TestCase):
    def setUp(self):
        self.tokenizer_patcher = patch(
            "lean_reinforcement.agent.transformer.AutoTokenizer.from_pretrained"
        )
        self.model_patcher = patch(
            "lean_reinforcement.agent.transformer.AutoModelForSeq2SeqLM.from_pretrained"
        )

        self.mock_tokenizer = self.tokenizer_patcher.start()
        self.mock_model = self.model_patcher.start()

        # Mock tokenizer behavior
        self.mock_tokenizer_instance = MagicMock()
        self.mock_tokenizer.return_value = self.mock_tokenizer_instance

        # Mock model behavior
        self.mock_model_instance = MagicMock()
        self.mock_model.return_value = self.mock_model_instance
        self.mock_model_instance.to.return_value = self.mock_model_instance

        self.transformer = Transformer()

    def tearDown(self):
        self.tokenizer_patcher.stop()
        self.model_patcher.stop()

    def test_generate_tactics_batch(self):
        states = ["state1", "state2"]
        n = 2

        # Mock tokenizer output
        self.mock_tokenizer_instance.to.return_value = MagicMock(input_ids="input_ids")

        # Mock model generate output
        # Shape: (batch_size * n, seq_len) -> (2 * 2, 10)
        self.mock_model_instance.generate.return_value = torch.zeros((4, 10))

        # Mock batch_decode
        self.mock_tokenizer_instance.batch_decode.return_value = [
            "t1_s1",
            "t2_s1",
            "t1_s2",
            "t2_s2",
        ]

        results = self.transformer.generate_tactics_batch(states, n=n)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], ["t1_s1", "t2_s1"])
        self.assertEqual(results[1], ["t1_s2", "t2_s2"])

        # Verify tokenizer called with padding=True
        self.mock_tokenizer_instance.assert_called_with(
            states, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )

    def test_generate_tactics_with_probs_batch(self):
        states = ["state1"]
        n = 2

        # Mock tokenizer output
        self.mock_tokenizer_instance.to.return_value = MagicMock(input_ids="input_ids")

        # Mock model generate output
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.zeros((2, 10))
        # Scores shape: (batch_size * n,)
        mock_outputs.sequences_scores = torch.tensor([1.0, 0.5])
        self.mock_model_instance.generate.return_value = mock_outputs

        # Mock batch_decode
        self.mock_tokenizer_instance.batch_decode.return_value = ["t1", "t2"]

        results = self.transformer.generate_tactics_with_probs_batch(states, n=n)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)

        tactic1, prob1 = results[0][0]
        tactic2, prob2 = results[0][1]

        self.assertEqual(tactic1, "t1")
        self.assertEqual(tactic2, "t2")

        # Check softmax calculation
        expected_probs = torch.softmax(torch.tensor([1.0, 0.5]), dim=0).tolist()
        self.assertAlmostEqual(prob1, expected_probs[0])
        self.assertAlmostEqual(prob2, expected_probs[1])


if __name__ == "__main__":
    unittest.main()
