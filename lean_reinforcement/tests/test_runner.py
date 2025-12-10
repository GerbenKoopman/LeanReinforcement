import unittest
from unittest.mock import Mock, MagicMock, patch

from lean_dojo import TacticState, ProofFinished

from lean_reinforcement.agent.runner import AgentRunner
from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import Transformer


class TestAgentRunner(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock(spec=LeanDojoEnv)
        self.transformer = Mock(spec=Transformer)

        # Mock generate_tactics to always return a list with at least one tactic
        def mock_generate_tactics(state_str, n=1):
            return ["tactic1", "tactic2", "tactic3"][:n] if n > 0 else ["tactic1"]

        self.transformer.generate_tactics.side_effect = mock_generate_tactics

        # Mock the environment's theorem and initial state
        self.env.theorem = Mock(full_name="test_theorem")
        self.env.theorem_pos = "mock_pos"
        self.env.dataloader = Mock()
        self.env.dataloader.get_premises.return_value = ["p1", "p2"]
        self.env.dojo_instance = Mock()
        # Mock run_tac to return a TacticState with pp attribute
        mock_tactic_state = Mock(spec=TacticState)
        mock_tactic_state.pp = "mock_state_pp"
        self.env.dojo_instance.run_tac.return_value = mock_tactic_state
        self.initial_state = Mock(spec=TacticState)
        self.initial_state.pp = "initial_state_pp"
        self.env.current_state = self.initial_state

        self.runner = AgentRunner(
            self.env,
            self.transformer,
            mcts_class=MCTS_GuidedRollout,
            num_iterations=10,
            max_steps=5,
        )

    @patch("lean_reinforcement.agent.runner.MCTS_GuidedRollout")
    def test_run_successful_proof(self, MockMCTS):
        # Arrange
        # Mock the MCTS instance and its methods
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = "best_tactic"

        # Create a mock child node
        mock_child = Mock()
        mock_child.visit_count = 10
        mock_child.action = "best_tactic"

        mock_root = Mock()
        mock_root.children = [mock_child]
        mock_mcts_instance.root = mock_root

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.transformer,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
        )

        # Keep track of step count
        step_count = [0]

        # Update current_state after each step
        def mock_step(*args, **kwargs):
            step_count[0] += 1
            if step_count[0] == 1:
                next_state = Mock(spec=TacticState)
                next_state.pp = "next_state_pp"
                self.env.current_state = next_state
                return (next_state, 0, False)
            else:
                proof_finished = Mock(spec=ProofFinished)
                self.env.current_state = proof_finished
                return (proof_finished, 1, True)

        self.env.step.side_effect = mock_step

        # Act
        metrics, trajectory = runner.run()

        # Assert
        self.assertTrue(metrics["proof_search/success"])
        # Training data is empty by default (collect_value_data=False)
        self.assertEqual(len(trajectory), 0)
        self.assertEqual(self.env.step.call_count, 2)
        mock_mcts_instance.search.assert_called_with(10)
        self.assertEqual(mock_mcts_instance.search.call_count, 2)

    @patch("lean_reinforcement.agent.runner.MCTS_GuidedRollout")
    def test_run_max_steps_reached(self, MockMCTS):
        # Arrange
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = "best_tactic"

        # Create a mock child node
        mock_child = Mock()
        mock_child.visit_count = 10
        mock_child.action = "best_tactic"

        mock_root = Mock()
        mock_root.children = [mock_child]
        mock_mcts_instance.root = mock_root

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.transformer,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
        )

        def mock_step(*args, **kwargs):
            next_state = Mock(spec=TacticState)
            next_state.pp = "next_state_pp"
            self.env.current_state = next_state
            return (next_state, 0, False)

        self.env.step.side_effect = mock_step

        # Act
        metrics, trajectory = runner.run()

        # Assert
        self.assertFalse(metrics["proof_search/success"])
        # Training data is empty by default (collect_value_data=False)
        self.assertEqual(len(trajectory), 0)
        self.assertEqual(self.env.step.call_count, 5)

    @patch("lean_reinforcement.agent.runner.MCTS_GuidedRollout")
    def test_run_no_action_returned(self, MockMCTS):
        # Arrange
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = None

        mock_root = Mock()
        mock_root.children = []  # No children, so no best action
        mock_mcts_instance.root = mock_root

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.transformer,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
        )

        # Define step mock even though it shouldn't be called
        self.env.step.return_value = (Mock(spec=TacticState), 0, False, False, {})

        # Act
        metrics, trajectory = runner.run()

        # Assert
        self.assertFalse(metrics["proof_search/success"])
        # Training data is empty by default (collect_value_data=False)
        self.assertEqual(len(trajectory), 0)
        self.env.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
