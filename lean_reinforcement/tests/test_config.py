import unittest
from unittest.mock import patch
import sys
from lean_reinforcement.utilities.config import get_config, TrainingConfig


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        """Test that default values are set correctly."""
        with patch.object(sys, "argv", ["prog"]):
            config = get_config()
            self.assertIsInstance(config, TrainingConfig)
            self.assertEqual(config.data_type, "novel_premises")
            self.assertEqual(config.num_epochs, 10)
            self.assertEqual(config.mcts_type, "guided_rollout")
            self.assertTrue(config.train_value_head)

    def test_custom_args(self):
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
