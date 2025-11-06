import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

from src.agent.value_head import ValueHead


class TestValueHead(unittest.TestCase):
    @patch("src.agent.value_head.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("src.agent.value_head.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        self.mock_tokenizer = MagicMock()
        self.mock_transformer = MagicMock()
        self.mock_encoder = MagicMock()

        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_transformer

        # Mock get_encoder to return the encoder
        self.mock_transformer.get_encoder.return_value = self.mock_encoder

        # Mock that the encoder has parameters
        self.mock_encoder.parameters.return_value = [nn.Parameter(torch.randn(2, 2))]

        self.value_head = ValueHead()

    def test_initialization(self):
        # Check if encoder parameters are frozen
        for param in self.value_head.encoder.parameters():
            self.assertFalse(param.requires_grad)

        # Check if value head parameters require gradients
        for param in self.value_head.value_head.parameters():
            self.assertTrue(param.requires_grad)

    def test_encode(self):
        # Arrange
        test_list = ["string1", "string2"]
        input_ids = torch.tensor([[1, 2], [3, 4]])
        attention_mask = torch.tensor([[1, 1], [1, 1]])
        hidden_state = torch.rand(2, 2, 1472)  # Corrected dimension

        # Mock the tokenizer to return a mock object with .to() method
        mock_tokenized = MagicMock()
        mock_tokenized.input_ids = input_ids
        mock_tokenized.attention_mask = attention_mask
        mock_tokenized.to.return_value = mock_tokenized  # For CUDA support
        self.mock_tokenizer.return_value = mock_tokenized

        self.mock_encoder.return_value = MagicMock(last_hidden_state=hidden_state)

        # Act
        features = self.value_head._encode(test_list)

        # Assert
        self.mock_tokenizer.assert_called_once_with(
            test_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2300,
        )
        # Check encoder was called (exact call depends on CUDA availability)
        self.mock_encoder.assert_called_once()
        self.assertEqual(features.shape, (2, 1472))

    def test_predict(self):
        # Arrange
        state_str = "state"

        # Mock the _encode method to return a predictable tensor
        encoded_features = torch.randn(1, 1472)
        with patch.object(
            self.value_head, "_encode", return_value=encoded_features
        ) as mock_encode, patch.object(
            self.value_head.value_head, "forward", return_value=torch.tensor([[0.5]])
        ) as mock_forward:
            # Act
            value = self.value_head.predict(state_str)

            # Assert
            mock_encode.assert_called_once_with([state_str])
            mock_forward.assert_called_once_with(encoded_features)
            self.assertIsInstance(value, float)
            self.assertAlmostEqual(value, torch.tanh(torch.tensor(0.5)).item())


if __name__ == "__main__":
    unittest.main()
