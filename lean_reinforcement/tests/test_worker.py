import unittest
from unittest.mock import MagicMock, patch
from lean_reinforcement.utilities.config import TrainingConfig

from lean_dojo import DojoInitError

from lean_reinforcement.training.worker import process_theorem
from lean_reinforcement.agent.mcts.alphazero import MCTS_AlphaZero


class TestWorker(unittest.TestCase):
    def setUp(self):
        self.args = TrainingConfig(
            data_type="novel_premises",
            num_epochs=1,
            num_theorems=1,
            num_workers=1,
            train_epochs=1,
            save_training_data=False,
            save_checkpoints=False,
            resume=False,
            checkpoint_dir=None,
            mcts_type="guided_rollout",
            batch_size=16,
            num_iterations=10,
            max_steps=20,
            train_value_head=False,
            use_final_reward=True,
            use_wandb=False,
        )
        self.corpus = MagicMock()
        self.dataloader = MagicMock()
        self.transformer = MagicMock()
        self.value_head = MagicMock()
        self.thm_data = {"start": [0, 0]}

    @patch("lean_reinforcement.training.worker.LeanDojoEnv")
    @patch("lean_reinforcement.training.worker.AgentRunner")
    @patch("lean_reinforcement.training.worker.Pos")
    def test_process_theorem_success(self, MockPos, MockAgentRunner, MockLeanDojoEnv):
        # Setup mocks
        mock_theorem = MagicMock()
        mock_theorem.full_name = "TestTheorem"
        self.dataloader.extract_theorem.return_value = mock_theorem
        MockPos.return_value = MagicMock()

        mock_env = MagicMock()
        MockLeanDojoEnv.return_value = mock_env

        mock_runner = MagicMock()
        MockAgentRunner.return_value = mock_runner
        expected_data = [{"type": "value", "value": 1.0}]
        expected_metrics = {"proof_search/success": True}
        mock_runner.run.return_value = (expected_metrics, expected_data)

        # Run function
        result = process_theorem(
            self.thm_data,
            self.dataloader,
            self.transformer,
            self.value_head,
            self.args,
        )

        # Assertions
        self.dataloader.extract_theorem.assert_called_with(self.thm_data)
        MockLeanDojoEnv.assert_called_once()
        MockAgentRunner.assert_called_once()
        mock_runner.run.assert_called_once()
        self.assertEqual(result, {"metrics": expected_metrics, "data": expected_data})

    @patch("lean_reinforcement.training.worker.LeanDojoEnv")
    def test_process_theorem_env_error(self, MockLeanDojoEnv):
        # Setup mocks
        mock_theorem = MagicMock()
        mock_theorem.full_name = "TestTheorem"
        self.dataloader.extract_theorem.return_value = mock_theorem

        # Simulate environment initialization error
        MockLeanDojoEnv.side_effect = DojoInitError("Init failed")

        # Run function
        result = process_theorem(
            self.thm_data,
            self.dataloader,
            self.transformer,
            self.value_head,
            self.args,
        )

        # Should return empty dict on error
        self.assertEqual(result, {})

    @patch("lean_reinforcement.training.worker.LeanDojoEnv")
    @patch("lean_reinforcement.training.worker.AgentRunner")
    @patch("lean_reinforcement.training.worker.Pos")
    def test_process_theorem_alpha_zero(
        self, MockPos, MockAgentRunner, MockLeanDojoEnv
    ):
        self.args.mcts_type = "alpha_zero"

        mock_theorem = MagicMock()
        self.dataloader.extract_theorem.return_value = mock_theorem
        MockPos.return_value = MagicMock()
        MockLeanDojoEnv.return_value = MagicMock()

        mock_runner = MagicMock()
        MockAgentRunner.return_value = mock_runner
        mock_runner.run.return_value = (None, [])

        process_theorem(
            self.thm_data,
            self.dataloader,
            self.transformer,
            self.value_head,
            self.args,
        )

        # Check that MCTS_AlphaZero was used (indirectly via kwargs passed to AgentRunner)
        # We can check the call args of AgentRunner
        _, kwargs = MockAgentRunner.call_args

        self.assertEqual(kwargs["mcts_class"], MCTS_AlphaZero)
        self.assertEqual(kwargs["mcts_kwargs"]["value_head"], self.value_head)


if __name__ == "__main__":
    unittest.main()
