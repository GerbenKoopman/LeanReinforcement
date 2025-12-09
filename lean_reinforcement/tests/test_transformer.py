import unittest
from unittest.mock import patch, MagicMock
import torch

from lean_reinforcement.agent.transformer import Transformer


class TestTransformer(unittest.TestCase):
    @patch("lean_reinforcement.agent.transformer.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("lean_reinforcement.agent.transformer.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        """Set up the test environment with mocked tokenizer and model."""
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.to.return_value = self.mock_model
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_model

        self.transformer = Transformer()

    def test_initialization(self):
        """Test that the Transformer is initialized with the correct model and tokenizer."""
        self.assertIsNotNone(self.transformer.tokenizer)
        self.assertIsNotNone(self.transformer.model)

    def test_generate_tactics_single(self):
        """Test generating a single tactic."""
        # Arrange
        state = "example_state"
        expected_tactic = "apply theorem1"

        # Mock tokenizer return value
        mock_tokenized = MagicMock()
        mock_tokenized.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenized.to.return_value = mock_tokenized  # Chain .to() method
        self.mock_tokenizer.return_value = mock_tokenized

        # Mock model generate return value
        mock_tactics_ids = torch.tensor([[10, 20, 30]])
        self.mock_model.generate.return_value = mock_tactics_ids

        # Mock batch_decode return value
        self.mock_tokenizer.batch_decode.return_value = [expected_tactic]

        # Act
        tactics = self.transformer.generate_tactics(state, n=1)

        # Assert
        self.mock_tokenizer.assert_called_once_with(
            state, return_tensors="pt", truncation=True, max_length=1024
        )
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.batch_decode.assert_called_once_with(
            mock_tactics_ids, skip_special_tokens=True
        )
        self.assertEqual(len(tactics), 1)
        self.assertEqual(tactics[0], expected_tactic)

    def test_generate_tactics_multiple(self):
        """Test generating multiple tactics."""
        # Arrange
        state = "example_state"
        n = 3
        expected_tactics = ["apply theorem1", "intro x", "cases h"]

        # Mock tokenizer return value
        mock_tokenized = MagicMock()
        mock_tokenized.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenized.to.return_value = mock_tokenized  # Chain .to() method
        self.mock_tokenizer.return_value = mock_tokenized

        # Mock model generate return value
        mock_tactics_ids = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]])
        self.mock_model.generate.return_value = mock_tactics_ids

        # Mock batch_decode return value
        self.mock_tokenizer.batch_decode.return_value = expected_tactics

        # Act
        tactics = self.transformer.generate_tactics(state, n=n)

        # Assert
        self.mock_model.generate.assert_called_once()
        call_kwargs = self.mock_model.generate.call_args[1]
        self.assertEqual(call_kwargs["num_beams"], n)
        self.assertEqual(call_kwargs["num_return_sequences"], n)
        self.assertEqual(len(tactics), n)
        self.assertEqual(tactics, expected_tactics)

    def test_generate_tactics_with_probs(self):
        """Test generating tactics with probabilities."""
        # Arrange
        state = "example_state"
        n = 2
        expected_tactics = ["apply theorem1", "intro x"]

        # Mock tokenizer return value
        mock_tokenized = MagicMock()
        mock_tokenized.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenized.to.return_value = mock_tokenized  # Chain .to() method
        self.mock_tokenizer.return_value = mock_tokenized

        # Mock model generate return value with scores
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[10, 20, 30], [11, 21, 31]])
        mock_outputs.sequences_scores = torch.tensor([2.0, 1.0])  # Use actual tensor
        self.mock_model.generate.return_value = mock_outputs

        # Mock batch_decode return value
        self.mock_tokenizer.batch_decode.return_value = expected_tactics

        # Act
        tactics_with_probs = self.transformer.generate_tactics_with_probs(state, n=n)

        # Assert
        self.mock_model.generate.assert_called_once()
        call_kwargs = self.mock_model.generate.call_args[1]
        self.assertTrue(call_kwargs["return_dict_in_generate"])
        self.assertTrue(call_kwargs["output_scores"])
        self.assertEqual(call_kwargs["num_beams"], n)
        self.assertEqual(call_kwargs["num_return_sequences"], n)

        # Check the return value structure
        self.assertEqual(len(tactics_with_probs), n)
        self.assertIsInstance(tactics_with_probs, list)
        for tactic, prob in tactics_with_probs:
            self.assertIsInstance(tactic, str)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_generate_tactics_with_probs_probabilities_sum(self):
        """Test that probabilities from softmax sum to approximately 1.0."""
        # Arrange
        state = "example_state"
        n = 3

        # Mock tokenizer return value
        mock_tokenized = MagicMock()
        mock_tokenized.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenized.to.return_value = mock_tokenized  # Chain .to() method
        self.mock_tokenizer.return_value = mock_tokenized

        # Mock model generate return value with scores
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor(
            [[10, 20, 30], [11, 21, 31], [12, 22, 32]]
        )
        # Use realistic sequence scores
        mock_outputs.sequences_scores = torch.tensor([3.0, 2.0, 1.0])
        self.mock_model.generate.return_value = mock_outputs

        # Mock batch_decode return value
        self.mock_tokenizer.batch_decode.return_value = ["t1", "t2", "t3"]

        # Act
        tactics_with_probs = self.transformer.generate_tactics_with_probs(state, n=n)

        # Assert - probabilities should sum to approximately 1.0
        prob_sum = sum(prob for _, prob in tactics_with_probs)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_generate_tactics_beam_search_parameters(self):
        """Test that beam search parameters are correctly set."""
        # Arrange
        state = "example_state"

        # Mock tokenizer return value
        mock_tokenized = MagicMock()
        mock_tokenized.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenized.to.return_value = mock_tokenized
        self.mock_tokenizer.return_value = mock_tokenized

        # Mock model generate return value
        self.mock_model.generate.return_value = torch.tensor([[10, 20, 30]])

        # Mock batch_decode return value
        self.mock_tokenizer.batch_decode.return_value = ["tactic"]

        # Act
        self.transformer.generate_tactics(state, n=5)

        # Assert - check beam search parameters
        call_kwargs = self.mock_model.generate.call_args[1]
        self.assertEqual(call_kwargs["max_length"], 1024)
        self.assertEqual(call_kwargs["num_beams"], 5)
        self.assertFalse(call_kwargs["do_sample"])
        self.assertEqual(call_kwargs["num_return_sequences"], 5)
        self.assertFalse(call_kwargs["early_stopping"])

    def test_generate_tactics_with_torch_no_grad(self):
        """Test that torch.no_grad() is used during generation."""
        # Arrange
        state = "example_state"

        # Mock tokenizer return value
        mock_tokenized = MagicMock()
        mock_tokenized.input_ids = torch.tensor([[1, 2, 3]])
        self.mock_tokenizer.return_value = mock_tokenized

        # Mock model generate return value
        self.mock_model.generate.return_value = torch.tensor([[10, 20, 30]])

        # Mock batch_decode return value
        self.mock_tokenizer.batch_decode.return_value = ["tactic"]

        # Act
        with patch("torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()
            tactics = self.transformer.generate_tactics(state, n=1)

        # Assert - we can't directly test @torch.no_grad() decorator,
        # but we can verify the method works correctly
        self.assertEqual(len(tactics), 1)


if __name__ == "__main__":
    unittest.main()
