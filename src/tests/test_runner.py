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
        self.premise_selector.retrieve.return_value = ["premise1", "premise2"]
        self.tactic_generator = Mock(spec=TacticGenerator)

        # Mock generate_tactics to always return a list with at least one tactic
        def mock_generate_tactics(state_str, retrieved, n=1):
            return ["tactic1", "tactic2", "tactic3"][:n] if n > 0 else ["tactic1"]

        self.tactic_generator.generate_tactics.side_effect = mock_generate_tactics

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
        mock_root = Mock()
        mock_child = Mock()
        mock_child.action = "best_tactic"
        mock_child.visit_count = 10
        mock_root.children = [mock_child]
        mock_mcts_instance.root = mock_root

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.premise_selector,
            self.tactic_generator,
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
                return (next_state, 0, False, False, {})
            else:
                proof_finished = Mock(spec=ProofFinished)
                self.env.current_state = proof_finished
                return (proof_finished, 1, True, False, {})

        self.env.step.side_effect = mock_step

        # Act - pass all_premises and flags
        all_premises = ["p1", "p2"]
        success, training_data = runner.run(
            all_premises=all_premises, collect_value_data=True, collect_policy_data=True
        )

        # Assert
        self.assertTrue(success)
        # Should have 2 steps * 2 data types = 4 training samples
        self.assertEqual(len(training_data), 4)
        self.assertEqual(self.env.step.call_count, 2)
        mock_mcts_instance.search.assert_called_with(10)
        self.assertEqual(mock_mcts_instance.search.call_count, 2)

        # Check that value_target is assigned correctly
        value_data = [d for d in training_data if d.get("type") == "value"]
        for d in value_data:
            self.assertEqual(d["value_target"], 1.0)  # Success = 1.0

    @patch("src.agent.runner.MCTS_GuidedRollout")
    def test_run_max_steps_reached(self, MockMCTS):
        # Arrange
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = "best_tactic"
        mock_root = Mock()
        mock_child = Mock()
        mock_child.action = "best_tactic"
        mock_child.visit_count = 10
        mock_root.children = [mock_child]
        mock_mcts_instance.root = mock_root

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.premise_selector,
            self.tactic_generator,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
        )

        def mock_step(*args, **kwargs):
            next_state = Mock(spec=TacticState)
            next_state.pp = "next_state_pp"
            self.env.current_state = next_state
            return (next_state, 0, False, False, {})

        self.env.step.side_effect = mock_step

        # Act - pass all_premises and flags
        all_premises = ["p1", "p2"]
        success, training_data = runner.run(
            all_premises=all_premises,
            collect_value_data=True,
            collect_policy_data=False,
        )

        # Assert
        self.assertFalse(success)
        # Should have 5 steps * 1 data type (value only) = 5 training samples
        self.assertEqual(len(training_data), 5)
        self.assertEqual(self.env.step.call_count, 5)

        # Check that value_target is assigned correctly (failed proof = -1.0)
        for d in training_data:
            self.assertEqual(d.get("type"), "value")
            self.assertEqual(d["value_target"], -1.0)

    @patch("src.agent.runner.MCTS_GuidedRollout")
    def test_run_no_action_returned(self, MockMCTS):
        # Arrange
        mock_mcts_instance = MockMCTS.return_value
        mock_mcts_instance.get_best_action.return_value = None
        mock_root = Mock()
        mock_root.children = []  # No children means no action
        mock_mcts_instance.root = mock_root

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.premise_selector,
            self.tactic_generator,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
        )

        # Define step mock even though it shouldn't be called
        self.env.step.return_value = (Mock(spec=TacticState), 0, False, False, {})

        # Act - pass all_premises and flags
        all_premises = ["p1", "p2"]
        success, training_data = runner.run(
            all_premises=all_premises, collect_value_data=True, collect_policy_data=True
        )

        # Assert
        self.assertFalse(success)
        # Only value data is collected (no policy data since no action)
        self.assertEqual(len(training_data), 1)
        value_data = [d for d in training_data if d.get("type") == "value"]
        self.assertEqual(len(value_data), 1)
        self.env.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
