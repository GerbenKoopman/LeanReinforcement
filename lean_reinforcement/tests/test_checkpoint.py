import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from lean_reinforcement.utilities.checkpoint import save_checkpoint, load_checkpoint
from lean_reinforcement.utilities.config import TrainingConfig


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.mock_value_head = MagicMock()
        self.checkpoint_dir = Path("/tmp/checkpoints")
        self.args = TrainingConfig(
            data_type="novel_premises",
            num_epochs=10,
            num_theorems=100,
            num_iterations=20,
            max_steps=30,
            batch_size=16,
            num_workers=16,
            mcts_type="guided_rollout",
            train_epochs=1,
            train_value_head=True,
            use_final_reward=False,
            save_training_data=True,
            save_checkpoints=True,
            resume=False,
            checkpoint_dir=str(self.checkpoint_dir),
            use_wandb=False,
        )

    @patch("lean_reinforcement.utilities.checkpoint.save_training_metadata")
    @patch("lean_reinforcement.utilities.checkpoint.cleanup_old_checkpoints")
    @patch("pathlib.Path.mkdir")
    def test_save_checkpoint(self, mock_mkdir, mock_cleanup, mock_save_metadata):
        """Test saving a checkpoint."""
        save_checkpoint(
            self.mock_value_head, 1, self.checkpoint_dir, self.args, prefix="test"
        )

        # Check that mkdir was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Check that save_checkpoint was called on the value head twice (latest and epoch)
        self.assertEqual(self.mock_value_head.save_checkpoint.call_count, 2)
        self.mock_value_head.save_checkpoint.assert_any_call(
            str(self.checkpoint_dir), "test_latest.pth"
        )
        self.mock_value_head.save_checkpoint.assert_any_call(
            str(self.checkpoint_dir), "test_epoch_1.pth"
        )

        # Check that metadata was saved
        mock_save_metadata.assert_called_once()

        # Check that cleanup was called
        mock_cleanup.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_load_checkpoint_success(self, mock_glob, mock_exists):
        """Test loading a checkpoint successfully."""
        mock_exists.return_value = True
        # Mock glob to return a list of paths
        mock_glob.return_value = [
            Path("/tmp/checkpoints/test_epoch_1.pth"),
            Path("/tmp/checkpoints/test_epoch_5.pth"),
        ]

        epoch = load_checkpoint(
            self.mock_value_head, self.checkpoint_dir, prefix="test"
        )

        self.assertEqual(epoch, 5)
        self.mock_value_head.load_checkpoint.assert_called_once_with(
            str(self.checkpoint_dir), "test_latest.pth"
        )

    @patch("pathlib.Path.exists")
    def test_load_checkpoint_not_found(self, mock_exists):
        """Test loading a checkpoint when it doesn't exist."""
        mock_exists.return_value = False

        epoch = load_checkpoint(
            self.mock_value_head, self.checkpoint_dir, prefix="test"
        )

        self.assertEqual(epoch, 0)
        self.mock_value_head.load_checkpoint.assert_not_called()
