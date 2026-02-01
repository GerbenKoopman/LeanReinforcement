import unittest
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from pathlib import Path
from lean_reinforcement.utilities.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_next_iteration,
    get_iteration_checkpoint_dir,
)
from lean_reinforcement.utilities.config import TrainingConfig


class TestCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
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
            indexed_corpus_path=None,
            train_epochs=1,
            value_head_batch_size=4,
            train_value_head=True,
            use_final_reward=False,
            save_training_data=True,
            save_checkpoints=True,
            resume=False,
            use_test_value_head=False,
            checkpoint_dir=str(self.checkpoint_dir),
            use_wandb=False,
            use_caching=False,
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


class TestIterationCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.base_checkpoint_dir = Path(self.temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_get_next_iteration_empty_dir(self):
        """Test get_next_iteration with no existing iterations."""
        result = get_next_iteration(self.base_checkpoint_dir, "alpha_zero")
        self.assertEqual(result, 1)

    def test_get_next_iteration_with_existing(self):
        """Test get_next_iteration with existing iteration directories."""
        # Create some iteration directories
        (self.base_checkpoint_dir / "alpha_zero-1").mkdir()
        (self.base_checkpoint_dir / "alpha_zero-3").mkdir()
        (self.base_checkpoint_dir / "alpha_zero-5").mkdir()
        (self.base_checkpoint_dir / "guided_rollout-2").mkdir()

        result = get_next_iteration(self.base_checkpoint_dir, "alpha_zero")
        self.assertEqual(result, 6)  # Should be 5 + 1

    def test_get_next_iteration_different_mcts_types(self):
        """Test that different mcts types have independent iteration counts."""
        (self.base_checkpoint_dir / "alpha_zero-3").mkdir()
        (self.base_checkpoint_dir / "guided_rollout-7").mkdir()

        alpha_result = get_next_iteration(self.base_checkpoint_dir, "alpha_zero")
        guided_result = get_next_iteration(self.base_checkpoint_dir, "guided_rollout")

        self.assertEqual(alpha_result, 4)
        self.assertEqual(guided_result, 8)

    def test_get_next_iteration_nonexistent_dir(self):
        """Test get_next_iteration when base directory doesn't exist."""
        nonexistent = Path("/tmp/nonexistent_checkpoint_dir_xyz123")
        result = get_next_iteration(nonexistent, "alpha_zero")
        self.assertEqual(result, 1)

    def test_get_iteration_checkpoint_dir_new(self):
        """Test creating a new iteration directory."""
        result = get_iteration_checkpoint_dir(
            self.base_checkpoint_dir, "alpha_zero", resume=False
        )
        self.assertEqual(result, self.base_checkpoint_dir / "alpha_zero-1")
        self.assertTrue(result.exists())

    def test_get_iteration_checkpoint_dir_new_with_existing(self):
        """Test creating a new iteration when some already exist."""
        (self.base_checkpoint_dir / "alpha_zero-1").mkdir()
        (self.base_checkpoint_dir / "alpha_zero-2").mkdir()

        result = get_iteration_checkpoint_dir(
            self.base_checkpoint_dir, "alpha_zero", resume=False
        )
        self.assertEqual(result, self.base_checkpoint_dir / "alpha_zero-3")
        self.assertTrue(result.exists())

    def test_get_iteration_checkpoint_dir_resume(self):
        """Test resuming from the latest iteration."""
        (self.base_checkpoint_dir / "alpha_zero-1").mkdir()
        (self.base_checkpoint_dir / "alpha_zero-2").mkdir()
        (self.base_checkpoint_dir / "alpha_zero-3").mkdir()

        result = get_iteration_checkpoint_dir(
            self.base_checkpoint_dir, "alpha_zero", resume=True
        )
        self.assertEqual(result, self.base_checkpoint_dir / "alpha_zero-3")

    def test_get_iteration_checkpoint_dir_resume_no_existing(self):
        """Test resume when no iterations exist (should create iteration 1)."""
        result = get_iteration_checkpoint_dir(
            self.base_checkpoint_dir, "alpha_zero", resume=True
        )
        self.assertEqual(result, self.base_checkpoint_dir / "alpha_zero-1")
        self.assertTrue(result.exists())
