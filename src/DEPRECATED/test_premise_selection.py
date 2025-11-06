import unittest
from unittest.mock import patch, MagicMock
import torch

from DEPRECATED.premise_selection import PremiseSelector


class TestPremiseSelector(unittest.TestCase):
    @patch("src.agent.premise_selection.AutoModelForTextEncoding.from_pretrained")
    @patch("src.agent.premise_selection.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        # Mock the tokenizer and model
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_model

        self.premise_selector = PremiseSelector()

    def test_encode_single_string(self):
        # Arrange
        test_string = "test string"
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        hidden_state = torch.rand(1, 3, 8)

        self.mock_tokenizer.return_value = MagicMock(
            input_ids=input_ids, attention_mask=attention_mask
        )
        self.mock_model.return_value = MagicMock(last_hidden_state=hidden_state)

        # Act
        features = self.premise_selector._encode(test_string)

        # Assert
        self.mock_tokenizer.assert_called_once_with(
            [test_string], return_tensors="pt", padding=True
        )
        self.mock_model.assert_called_once_with(input_ids)
        self.assertEqual(features.shape, (8,))

    def test_encode_list_of_strings(self):
        # Arrange
        test_list = ["string1", "string2"]
        input_ids = torch.tensor([[1, 2], [3, 4]])
        attention_mask = torch.tensor([[1, 1], [1, 1]])
        hidden_state = torch.rand(2, 2, 8)

        self.mock_tokenizer.return_value = MagicMock(
            input_ids=input_ids, attention_mask=attention_mask
        )
        self.mock_model.return_value = MagicMock(last_hidden_state=hidden_state)

        # Act
        features = self.premise_selector._encode(test_list)

        # Assert
        self.mock_tokenizer.assert_called_once_with(
            test_list, return_tensors="pt", padding=True
        )
        self.mock_model.assert_called_once_with(input_ids)
        self.assertEqual(features.shape, (2, 8))

    def test_retrieve(self):
        # Arrange
        state = "current state"
        premises = ["p1", "p2", "p3"]
        k = 2

        # Mock the encoding to return predictable tensors
        state_emb = torch.tensor([1.0, 0.0, 0.0])
        premise_embs = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.5, 0.5, 0.0]])

        # We need to patch the _encode method inside the instance
        with patch.object(
            self.premise_selector, "_encode", side_effect=[state_emb, premise_embs]
        ) as mock_encode:
            # Act
            top_premises = self.premise_selector.retrieve(state, premises, k)

            # Assert
            self.assertEqual(mock_encode.call_count, 2)
            self.assertEqual(top_premises, ["p1", "p3"])


if __name__ == "__main__":
    unittest.main()
