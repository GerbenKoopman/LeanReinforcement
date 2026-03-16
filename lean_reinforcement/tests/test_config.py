import unittest
from unittest.mock import patch
import sys
from lean_reinforcement.utilities.config import (
    get_config,
    TrainingConfig,
    OPTIMAL_DEFAULTS,
)


class TestConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        """Test that default values are set correctly."""
        with patch.object(sys, "argv", ["prog"]):
            config = get_config()
            self.assertIsInstance(config, TrainingConfig)
            self.assertEqual(config.data_type, OPTIMAL_DEFAULTS["data_type"])
            self.assertEqual(config.num_epochs, OPTIMAL_DEFAULTS["num_epochs"])
            self.assertEqual(config.mcts_type, OPTIMAL_DEFAULTS["mcts_type"])
            self.assertEqual(
                config.train_value_head,
                OPTIMAL_DEFAULTS["train_value_head"],
            )
            self.assertEqual(config.value_head_latent_dim, 1024)

    def test_custom_args(self) -> None:
        """Test that command line arguments override defaults."""
        test_args = [
            "prog",
            "--data-type",
            "random",
            "--num-epochs",
            "50",
            "--mcts-type",
            "alpha_zero",
            "--no-train-value-head",
            "--no-save-training-data",
            "--no-save-checkpoints",
        ]
        with patch.object(sys, "argv", test_args):
            config = get_config()
            self.assertEqual(config.data_type, "random")
            self.assertEqual(config.num_epochs, 50)
            self.assertEqual(config.mcts_type, "alpha_zero")
            self.assertFalse(config.train_value_head)
            self.assertFalse(config.save_training_data)
            self.assertFalse(config.save_checkpoints)

    def test_value_head_latent_dim_custom(self) -> None:
        """Test custom value head latent dimension."""
        test_args = [
            "prog",
            "--value-head-latent-dim",
            "512",
        ]
        with patch.object(sys, "argv", test_args):
            config = get_config()
            self.assertEqual(config.value_head_latent_dim, 512)

    def test_value_head_latent_dim_default(self) -> None:
        """Test default value head latent dimension."""
        test_args = [
            "prog",
        ]
        with patch.object(sys, "argv", test_args):
            config = get_config()
            self.assertEqual(config.value_head_latent_dim, 1024)

    def test_value_head_latent_dim_single(self) -> None:
        """Test single value head latent dimension."""
        test_args = [
            "prog",
            "--value-head-latent-dim",
            "1024",
        ]
        with patch.object(sys, "argv", test_args):
            config = get_config()
            self.assertEqual(config.value_head_latent_dim, 1024)
