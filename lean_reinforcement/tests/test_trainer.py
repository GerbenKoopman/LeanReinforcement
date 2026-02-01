import pytest
from unittest.mock import MagicMock, patch
from lean_reinforcement.training.trainer import Trainer
from lean_reinforcement.utilities.config import TrainingConfig


@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock(spec=TrainingConfig)
    config.use_wandb = False
    config.model_name = "test_model"
    config.mcts_type = "alpha_zero"
    config.train_value_head = True
    config.resume = False
    config.use_test_value_head = False
    config.data_type = "random"
    config.num_workers = 2
    config.batch_size = 4
    config.value_head_batch_size = 4
    config.num_epochs = 1
    config.num_theorems = 2
    config.save_training_data = False
    config.save_checkpoints = False
    config.train_epochs = 1
    config.indexed_corpus_path = None
    config.num_iterations = 10
    config.max_steps = 20
    config.num_tactics_to_expand = 8
    config.max_rollout_depth = 30
    config.max_time = 300.0
    config.env_timeout = 180
    config.use_final_reward = True
    config.checkpoint_dir = None
    return config


@patch("lean_reinforcement.training.trainer.wandb")
@patch("lean_reinforcement.training.trainer.get_checkpoint_dir")
@patch("lean_reinforcement.training.trainer.Transformer")
@patch("lean_reinforcement.training.trainer.ValueHead")
@patch("lean_reinforcement.training.trainer.Corpus")
@patch("lean_reinforcement.training.trainer.LeanDataLoader")
@patch("lean_reinforcement.training.trainer.mp")
@patch("lean_reinforcement.training.trainer.InferenceServer")
def test_trainer_initialization(
    mock_inference_server,
    mock_mp,
    mock_dataloader,
    mock_corpus,
    mock_value_head,
    mock_transformer,
    mock_get_checkpoint_dir,
    mock_wandb,
    mock_config,
):
    trainer = Trainer(mock_config)
    assert trainer.config == mock_config
    mock_get_checkpoint_dir.assert_called_once()
    mock_transformer.assert_called_once_with(model_name="test_model")
    mock_value_head.assert_called_once()
    mock_corpus.assert_called_once()
    mock_dataloader.assert_called_once()


@patch("lean_reinforcement.training.trainer.torch.nn.MSELoss")
@patch("lean_reinforcement.training.trainer.optim.AdamW")
@patch("lean_reinforcement.training.trainer.wandb")
@patch("lean_reinforcement.training.trainer.get_checkpoint_dir")
@patch("lean_reinforcement.training.trainer.Transformer")
@patch("lean_reinforcement.training.trainer.ValueHead")
@patch("lean_reinforcement.training.trainer.Corpus")
@patch("lean_reinforcement.training.trainer.LeanDataLoader")
@patch("lean_reinforcement.training.trainer.mp")
@patch("lean_reinforcement.training.trainer.InferenceServer")
@patch("lean_reinforcement.training.trainer.worker_loop")
def test_trainer_train_loop(
    mock_worker_loop,
    mock_inference_server,
    mock_mp,
    mock_dataloader,
    mock_corpus,
    mock_value_head,
    mock_transformer,
    mock_get_checkpoint_dir,
    mock_wandb,
    mock_adamw,
    mock_mse_loss,
    mock_config,
):
    # Setup mocks
    mock_queue = MagicMock()
    mock_mp.Queue.return_value = mock_queue
    mock_process = MagicMock()
    mock_mp.Process.return_value = mock_process

    # Mock dataloader.train_data
    mock_dataloader_instance = mock_dataloader.return_value
    mock_dataloader_instance.train_data = ["thm1", "thm2"]

    # Mock inference server process_requests to return False initially then True?
    # Actually process_requests returns a boolean indicating if it processed something.
    # We need to control the loop in _collect_data.
    # The loop condition is `completed_theorems < len(theorems_to_process)`.
    # We need result_queue.get_nowait() to return results to increment completed_theorems.

    # mock_result_queue = MagicMock()
    # First call raises Empty (simulating wait), second call returns result, third call raises Empty...
    # But get_nowait is called in a loop.

    # Let's just test that _run_epoch calls the right things.
    trainer = Trainer(mock_config)

    # Mock _collect_data to avoid complex queue mocking for now
    # Use object.__setattr__ to bypass method assignment restriction if necessary,
    # or just mock the method on the instance if it's not a frozen dataclass or similar.
    # Since Trainer is a normal class, we can just assign.
    # However, mypy complains about assigning to a method.
    # We can use a cast or type ignore, but user asked to avoid ignore.
    # A cleaner way for testing is to subclass or use `unittest.mock.patch.object`.
    with patch.object(
        trainer,
        "_collect_data",
        return_value=(
            [{"type": "value", "value_target": 1.0, "state": "some_state"}],
            {},
        ),
    ) as mock_collect:  # Mock parameters for optimizer
        mock_param = MagicMock()
        mock_value_head.return_value.value_head.parameters.return_value = [mock_param]

        # Mock loss.item()
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_mse_loss.return_value.return_value = mock_loss

        # Mock _train_value_head
        # trainer._train_value_head = MagicMock() # Don't mock this, we want to test the loop calling it

        trainer.train()

        assert mock_collect.call_count == 1
        # assert trainer._train_value_head.call_count == 1 # We are not mocking it anymore
        mock_adamw.assert_called_once()
        mock_mse_loss.assert_called_once()
