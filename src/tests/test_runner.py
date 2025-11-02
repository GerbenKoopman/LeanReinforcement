import unittest
from unittest.mock import Mock, MagicMock, patch

from lean_dojo import TacticState, ProofFinished

from src.agent.runner import AgentRunner
from src.agent.mcts import MCTS_GuidedRollout
from src.utilities.gym import LeanDojoEnv
from src.agent.premise_selection import PremiseSelector
from src.agent.tactic_generation import TacticGenerator


class TestAgentRunner(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock(spec=LeanDojoEnv)
        self.premise_selector = Mock(spec=PremiseSelector)
        self.tactic_generator = Mock(spec=TacticGenerator)

        # Mock the environment's theorem and initial state
        self.env.theorem = Mock(full_name="test_theorem")
        self.initial_state = Mock(spec=TacticState)
        self.env.current_state = self.initial_state

        self.runner = AgentRunner(
            self.env,
            self.premise_selector,
            self.tactic_generator,
            mcts_class=MCTS_GuidedRollout,
            num_iterations=10,
            max_steps=5,
        )

    @patch("src.agent.runner.MCTS_GuidedRollout")
    def test_run_successful_proof(self, MockMCTS):
        # Arrange
        # Mock the MCTS instance and its methods
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = "best_tactic"

        # Simulate the environment's step function
        # First step: not terminated
        # Second step: terminated (proof finished)
        self.env.step.side_effect = [
            (Mock(spec=TacticState), 0, False, False, {}),
            (Mock(spec=ProofFinished), 1, True, False, {}),
        ]

        # Update current_state after each step
        def update_state(*args, **kwargs):
            if self.env.step.call_count == 1:
                self.env.current_state = Mock(spec=TacticState)
            else:
                self.env.current_state = Mock(spec=ProofFinished)
            return self.env.step.side_effect[self.env.step.call_count - 1]

        self.env.step.side_effect = update_state

        # Act
        success, trajectory = self.runner.run()

        # Assert
        self.assertTrue(success)
        self.assertEqual(len(trajectory), 2)
        self.assertEqual(self.env.step.call_count, 2)
        mock_mcts_instance.search.assert_called_with(10)
        self.assertEqual(mock_mcts_instance.search.call_count, 2)

    @patch("src.agent.runner.MCTS_GuidedRollout")
    def test_run_max_steps_reached(self, MockMCTS):
        # Arrange
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = "best_tactic"
        self.env.step.return_value = (
            Mock(spec=TacticState),
            0,
            False,
            False,
            {},
        )

        # Act
        success, trajectory = self.runner.run()

        # Assert
        self.assertFalse(success)
        self.assertEqual(len(trajectory), 5)
        self.assertEqual(self.env.step.call_count, 5)

    @patch("src.agent.runner.MCTS_GuidedRollout")
    def test_run_no_action_returned(self, MockMCTS):
        # Arrange
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = None

        # Act
        success, trajectory = self.runner.run()

        # Assert
        self.assertFalse(success)
        self.assertEqual(len(trajectory), 1)
        self.env.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
