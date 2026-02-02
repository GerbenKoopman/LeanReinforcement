import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

from lean_reinforcement.agent.value_head import ValueHead, ENCODER_OUTPUT_DIM
from lean_reinforcement.agent.transformer import Transformer


class TestValueHead(unittest.TestCase):
    @patch("lean_reinforcement.agent.transformer.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("lean_reinforcement.agent.transformer.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        self.mock_tokenizer = MagicMock()
        self.mock_transformer_model = MagicMock()
        self.mock_encoder = MagicMock()

        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_transformer_model

        self.mock_transformer_model.to.return_value = self.mock_transformer_model

        # Mock get_encoder to return the encoder
        self.mock_transformer_model.get_encoder.return_value = self.mock_encoder

        # Mock that the encoder has parameters
        self.mock_encoder.parameters.return_value = [nn.Parameter(torch.randn(2, 2))]

        # Create a mock transformer
        self.mock_transformer = MagicMock(spec=Transformer)
        self.mock_transformer.tokenizer = self.mock_tokenizer
        self.mock_transformer.model = self.mock_transformer_model

        self.value_head = ValueHead(self.mock_transformer)

    def test_initialization(self) -> None:
        # Check if encoder parameters are frozen
        for param in self.value_head.encoder.parameters():
            self.assertFalse(param.requires_grad)

        # Check if value head parameters require gradients
        for param in self.value_head.value_head.parameters():
            self.assertTrue(param.requires_grad)

    def test_encode_states(self) -> None:
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
        features = self.value_head.encode_states(test_list)

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

    def test_predict(self) -> None:
        # Arrange
        state_str = "state"

        encoded_features = torch.randn(1, 1472)
        with (
            patch.object(
                self.value_head, "encode_states", return_value=encoded_features
            ) as mock_encode,
            patch.object(
                self.value_head.value_head,
                "forward",
                return_value=torch.tensor([[0.5]]),
            ) as mock_forward,
        ):
            # Act
            value = self.value_head.predict(state_str)

            # Assert
            mock_encode.assert_called_once_with([state_str])
            mock_forward.assert_called_once_with(encoded_features)
            self.assertIsInstance(value, float)
            self.assertAlmostEqual(value, torch.tanh(torch.tensor(0.5)).item())

    def test_value_head_forward(self) -> None:
        # Arrange
        # Create pre-computed features with the expected dimension (1, 1472)
        features = torch.randn(1, 1472)

        # Mock the value_head forward pass to return a predictable value
        expected_raw_value = torch.tensor([[0.75]])

        with patch.object(
            self.value_head.value_head, "forward", return_value=expected_raw_value
        ) as mock_forward:
            # Act
            raw_value = self.value_head.value_head(features)
            value = torch.tanh(raw_value).item()

            # Assert
            mock_forward.assert_called_once_with(features)
            self.assertIsInstance(value, float)
            expected_result = torch.tanh(torch.tensor(0.75)).item()
            self.assertAlmostEqual(value, expected_result)
            # Verify the value is in the expected range [-1, 1]
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)


class TestValueHeadHiddenDims(unittest.TestCase):
    """Test suite for configurable hidden dimensions."""

    @patch("lean_reinforcement.agent.transformer.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("lean_reinforcement.agent.transformer.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        self.mock_tokenizer = MagicMock()
        self.mock_transformer_model = MagicMock()
        self.mock_encoder = MagicMock()

        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_transformer_model

        self.mock_transformer_model.to.return_value = self.mock_transformer_model
        self.mock_transformer_model.get_encoder.return_value = self.mock_encoder
        self.mock_encoder.parameters.return_value = [nn.Parameter(torch.randn(2, 2))]

        self.mock_transformer = MagicMock(spec=Transformer)
        self.mock_transformer.tokenizer = self.mock_tokenizer
        self.mock_transformer.model = self.mock_transformer_model

    def test_default_hidden_dims(self) -> None:
        """Test that default hidden dims is [256]."""
        value_head = ValueHead(self.mock_transformer)
        self.assertEqual(value_head.hidden_dims, [256])
        self.assertEqual(value_head.input_dim, ENCODER_OUTPUT_DIM)

    def test_empty_hidden_dims(self) -> None:
        """Test direct linear projection with empty hidden dims."""
        value_head = ValueHead(self.mock_transformer, hidden_dims=[])
        self.assertEqual(value_head.hidden_dims, [])

        # Check architecture: should be just one linear layer
        layers = list(value_head.value_head.children())
        self.assertEqual(len(layers), 1)
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertEqual(layers[0].in_features, ENCODER_OUTPUT_DIM)
        self.assertEqual(layers[0].out_features, 1)

    def test_single_hidden_layer(self) -> None:
        """Test single hidden layer configuration."""
        value_head = ValueHead(self.mock_transformer, hidden_dims=[512])
        self.assertEqual(value_head.hidden_dims, [512])

        # Check architecture: Linear -> ReLU -> Linear
        layers = list(value_head.value_head.children())
        self.assertEqual(len(layers), 3)
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertEqual(layers[0].in_features, ENCODER_OUTPUT_DIM)
        self.assertEqual(layers[0].out_features, 512)
        self.assertIsInstance(layers[1], nn.ReLU)
        self.assertIsInstance(layers[2], nn.Linear)
        self.assertEqual(layers[2].in_features, 512)
        self.assertEqual(layers[2].out_features, 1)

    def test_multiple_hidden_layers(self) -> None:
        """Test multiple hidden layers configuration."""
        hidden_dims = [512, 256, 128]
        value_head = ValueHead(self.mock_transformer, hidden_dims=hidden_dims)
        self.assertEqual(value_head.hidden_dims, hidden_dims)

        # Check architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear
        layers = list(value_head.value_head.children())
        self.assertEqual(len(layers), 7)  # 3 Linear + 3 ReLU + 1 final Linear

        # First hidden layer
        self.assertEqual(layers[0].in_features, ENCODER_OUTPUT_DIM)
        self.assertEqual(layers[0].out_features, 512)

        # Second hidden layer
        self.assertEqual(layers[2].in_features, 512)
        self.assertEqual(layers[2].out_features, 256)

        # Third hidden layer
        self.assertEqual(layers[4].in_features, 256)
        self.assertEqual(layers[4].out_features, 128)

        # Output layer
        self.assertEqual(layers[6].in_features, 128)
        self.assertEqual(layers[6].out_features, 1)

    def test_custom_input_dim(self) -> None:
        """Test custom input dimension."""
        value_head = ValueHead(self.mock_transformer, hidden_dims=[256], input_dim=768)
        self.assertEqual(value_head.input_dim, 768)

        layers = list(value_head.value_head.children())
        self.assertEqual(layers[0].in_features, 768)

    def test_forward_pass_default(self) -> None:
        """Test forward pass with default configuration."""
        value_head = ValueHead(self.mock_transformer)
        device = next(value_head.value_head.parameters()).device
        features = torch.randn(2, ENCODER_OUTPUT_DIM, device=device)
        output = value_head.value_head(features)
        self.assertEqual(output.shape, (2, 1))

    def test_forward_pass_empty_hidden(self) -> None:
        """Test forward pass with empty hidden dims."""
        value_head = ValueHead(self.mock_transformer, hidden_dims=[])
        device = next(value_head.value_head.parameters()).device
        features = torch.randn(2, ENCODER_OUTPUT_DIM, device=device)
        output = value_head.value_head(features)
        self.assertEqual(output.shape, (2, 1))

    def test_forward_pass_deep_network(self) -> None:
        """Test forward pass with deep network."""
        value_head = ValueHead(self.mock_transformer, hidden_dims=[1024, 512, 256, 128])
        device = next(value_head.value_head.parameters()).device
        features = torch.randn(2, ENCODER_OUTPUT_DIM, device=device)
        output = value_head.value_head(features)
        self.assertEqual(output.shape, (2, 1))

    def test_parameters_require_grad(self) -> None:
        """Test that value head parameters require gradients."""
        value_head = ValueHead(self.mock_transformer, hidden_dims=[512, 256])
        for param in value_head.value_head.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
