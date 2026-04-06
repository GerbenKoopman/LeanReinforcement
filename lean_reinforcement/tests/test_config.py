import unittest
from unittest.mock import patch
import sys
from lean_reinforcement.utilities.config import (
    get_config,
    TrainingConfig,
)


DEFAULT_DATA_TYPE = "novel_premises"
DEFAULT_NUM_EPOCHS = 3
DEFAULT_MCTS_TYPE = "alpha_zero"
DEFAULT_TRAIN_VALUE_HEAD = True


class TestConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        """Test that default values are set correctly."""
        with patch.object(sys, "argv", ["prog"]):
            config = get_config()
            self.assertIsInstance(config, TrainingConfig)
            self.assertEqual(config.data_type, DEFAULT_DATA_TYPE)
            self.assertEqual(config.num_epochs, DEFAULT_NUM_EPOCHS)
            self.assertEqual(config.mcts_type, DEFAULT_MCTS_TYPE)
            self.assertEqual(config.train_value_head, DEFAULT_TRAIN_VALUE_HEAD)
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

    def test_debugging_flag_default_false(self) -> None:
        with patch.object(sys, "argv", ["prog"]):
            config = get_config()
            self.assertFalse(config.debugging)

    def test_debugging_flag_enabled(self) -> None:
        with patch.object(sys, "argv", ["prog", "--debugging"]):
            config = get_config()
            self.assertTrue(config.debugging)

    def test_experience_replay_max_epochs(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["prog", "--experience-replay-max-epochs", "2"],
        ):
            config = get_config()
            self.assertEqual(config.experience_replay_max_epochs, 2)
