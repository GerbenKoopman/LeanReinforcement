import unittest
from unittest.mock import patch, MagicMock
import torch

from src.agent.tactic_generation import TacticGenerator


class TestTacticGenerator(unittest.TestCase):
    @patch("src.agent.tactic_generation.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("src.agent.tactic_generation.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_model

        self.tactic_generator = TacticGenerator()

    def test_generate_tactics(self):
        # Arrange
        state = "state"
        retrieved_premises = ["p1", "p2"]
        n = 2
        input_str = "p1\n\np2\n\nstate"
        tokenized_input = MagicMock()
        self.mock_tokenizer.return_value = tokenized_input
        generated_ids = torch.tensor([[1, 2], [3, 4]])
        self.mock_model.generate.return_value = generated_ids
        self.mock_tokenizer.batch_decode.return_value = ["tactic1", "tactic2"]

        # Act
        tactics = self.tactic_generator.generate_tactics(state, retrieved_premises, n)

        # Assert
        self.mock_tokenizer.assert_called_once_with(
            input_str, return_tensors="pt", max_length=2300, truncation=True
        )
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.batch_decode.assert_called_once_with(
            generated_ids, skip_special_tokens=True
        )
        self.assertEqual(tactics, ["tactic1", "tactic2"])

    def test_generate_tactics_with_probs(self):
        # Arrange
        state = "state"
        retrieved_premises = ["p1", "p2"]
        n = 2
        input_str = "p1\n\np2\n\nstate"
        tokenized_input = MagicMock()
        self.mock_tokenizer.return_value = tokenized_input

        # Mock the model's generate output
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2], [3, 4]])
        mock_output.sequences_scores = torch.tensor([-0.5, -1.0])
        self.mock_model.generate.return_value = mock_output

        self.mock_tokenizer.batch_decode.return_value = ["tactic1", "tactic2"]

        # Act
        tactics_with_probs = self.tactic_generator.generate_tactics_with_probs(
            state, retrieved_premises, n
        )

        # Assert
        self.mock_tokenizer.assert_called_once_with(
            input_str, return_tensors="pt", max_length=2300, truncation=True
        )
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.batch_decode.assert_called_once_with(
            mock_output.sequences, skip_special_tokens=True
        )

        # Check the probabilities
        probs = torch.softmax(torch.tensor([-0.5, -1.0]), dim=0).tolist()
        self.assertEqual(len(tactics_with_probs), 2)
        self.assertEqual(tactics_with_probs[0][0], "tactic1")
        self.assertAlmostEqual(tactics_with_probs[0][1], probs[0])
        self.assertEqual(tactics_with_probs[1][0], "tactic2")
        self.assertAlmostEqual(tactics_with_probs[1][1], probs[1])


if __name__ == "__main__":
    unittest.main()
