import unittest
from unittest.mock import Mock, MagicMock, patch

from lean_dojo import TacticState, ProofFinished

from lean_reinforcement.agent.runner import AgentRunner
from lean_reinforcement.agent.mcts import MCTS_GuidedRollout
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import Transformer


class TestAgentRunner(unittest.TestCase):
    def setUp(self) -> None:
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

        self.config = MagicMock()
        # Default to step-by-step for existing tests
        self.config.full_search = False

        self.runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
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
        mock_mcts_instance.max_time = 60.0  # Add max_time for runner to use

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
            self.config,
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
        # search is called with num_iterations and max_time
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
        mock_mcts_instance.max_time = 60.0  # Add max_time for runner to use

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
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
        mock_mcts_instance.max_time = 60.0  # Add max_time for runner to use

        mock_root = Mock()
        mock_root.children = []  # No children, so no best action
        mock_mcts_instance.root = mock_root

        # Create a new runner with the mocked MCTS class
        runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
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


class TestFullSearchMode(unittest.TestCase):
    """Tests for full_search mode (flaw #4 fix: backtracking)."""

    def setUp(self) -> None:
        self.env = MagicMock(spec=LeanDojoEnv)
        self.transformer = Mock(spec=Transformer)

        def mock_generate_tactics(state_str, n=1):
            return ["tactic1", "tactic2", "tactic3"][:n] if n > 0 else ["tactic1"]

        self.transformer.generate_tactics.side_effect = mock_generate_tactics

        self.env.theorem = Mock(full_name="test_theorem")
        self.env.theorem_pos = "mock_pos"
        self.initial_state = Mock(spec=TacticState)
        self.initial_state.pp = "initial_state_pp"
        self.env.current_state = self.initial_state

        self.config = MagicMock()
        self.config.full_search = True

    @patch("lean_reinforcement.agent.runner.MCTS_GuidedRollout")
    def test_full_search_finds_proof(self, MockMCTS):
        """Full search should run one big search and apply proof path."""
        mock_mcts = MockMCTS.return_value
        mock_root = Mock()
        mock_root.max_value = 1.0
        mock_mcts.root = mock_root
        mock_mcts.extract_proof_path.return_value = ["intro h", "exact h"]

        runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
            proof_timeout=600.0,
        )

        # Mock step: second call finishes proof
        step_count = [0]

        def mock_step(*args, **kwargs):
            step_count[0] += 1
            if step_count[0] < 2:
                next_state = Mock(spec=TacticState)
                next_state.pp = "next_state"
                self.env.current_state = next_state
                return ("next_state", 0, False)
            else:
                pf = Mock(spec=ProofFinished)
                self.env.current_state = pf
                return ("done", 1, True)

        self.env.step.side_effect = mock_step

        metrics, data = runner.run()

        self.assertTrue(metrics["proof_search/success"])
        # search is called exactly once (full budget)
        mock_mcts.search.assert_called_once()
        # All tactics in proof path are applied
        self.assertEqual(self.env.step.call_count, 2)
        # move_root is never called (no step-by-step commitment)
        mock_mcts.move_root.assert_not_called()

    @patch("lean_reinforcement.agent.runner.MCTS_GuidedRollout")
    def test_full_search_no_proof(self, MockMCTS):
        """Full search with no proof found should return failure."""
        mock_mcts = MockMCTS.return_value
        mock_root = Mock()
        mock_root.max_value = -0.5  # No proof found
        mock_mcts.root = mock_root

        runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
            proof_timeout=600.0,
        )

        metrics, data = runner.run()

        self.assertFalse(metrics["proof_search/success"])
        mock_mcts.search.assert_called_once()
        # No tactics applied
        self.env.step.assert_not_called()
        mock_mcts.move_root.assert_not_called()

    @patch("lean_reinforcement.agent.runner.MCTS_GuidedRollout")
    def test_full_search_uses_full_budget(self, MockMCTS):
        """Full search should use num_iterations * max_steps iterations."""
        mock_mcts = MockMCTS.return_value
        mock_root = Mock()
        mock_root.max_value = -1.0
        mock_mcts.root = mock_root

        num_iterations = 100
        max_steps = 10

        runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
            mcts_class=MockMCTS,
            num_iterations=num_iterations,
            max_steps=max_steps,
            proof_timeout=600.0,
        )

        runner.run()

        # Verify search was called with total_iterations = 100 * 10 = 1000
        call_args = mock_mcts.search.call_args
        self.assertEqual(call_args[0][0], num_iterations * max_steps)

    @patch("lean_reinforcement.agent.runner.MCTS_GuidedRollout")
    def test_full_search_collects_tree_data(self, MockMCTS):
        """Full search should collect training data from the tree."""
        mock_mcts = MockMCTS.return_value

        # Build a small mock tree
        root_state = Mock(spec=TacticState)
        root_state.pp = "root_state"

        child1_state = Mock(spec=TacticState)
        child1_state.pp = "child1_state"

        child2_state = Mock(spec=TacticState)
        child2_state.pp = "child2_state"

        # Mock Node-like objects for the tree
        mock_child1 = Mock()
        mock_child1.state = child1_state
        mock_child1.visit_count = 5
        mock_child1.action = "tactic_a"
        mock_child1.children = []
        mock_child1.max_value = 0.3

        mock_child2 = Mock()
        mock_child2.state = child2_state
        mock_child2.visit_count = 3
        mock_child2.action = "tactic_b"
        mock_child2.children = []
        mock_child2.max_value = -0.2

        mock_root = Mock()
        mock_root.state = root_state
        mock_root.visit_count = 8
        mock_root.children = [mock_child1, mock_child2]
        mock_root.max_value = 0.3

        # value() should return max_value
        def root_value():
            return mock_root.max_value

        mock_root.value = root_value

        mock_mcts.root = mock_root

        runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
            mcts_class=MockMCTS,
            num_iterations=10,
            max_steps=5,
            proof_timeout=600.0,
        )

        _, data = runner.run(collect_value_data=True)

        # Should collect data from root (expanded with children)
        # child1 and child2 have no children, so they are not collected
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["state"], "root_state")
        self.assertEqual(data[0]["step"], 0)
        self.assertIn("visit_distribution", data[0])

    def test_full_search_mode_dispatch(self):
        """run() should dispatch based on config.full_search."""
        self.config.full_search = True
        runner = AgentRunner(
            self.env,
            self.transformer,
            self.config,
            mcts_class=MagicMock(),
            num_iterations=10,
            max_steps=5,
        )

        # Patch both methods to track which is called
        with patch.object(runner, "_run_full_search") as mock_full, \
             patch.object(runner, "_run_step_by_step") as mock_step:
            mock_full.return_value = ({}, [])
            runner.run()
            mock_full.assert_called_once()
            mock_step.assert_not_called()

        self.config.full_search = False
        with patch.object(runner, "_run_full_search") as mock_full, \
             patch.object(runner, "_run_step_by_step") as mock_step:
            mock_step.return_value = ({}, [])
            runner.run()
            mock_step.assert_called_once()
            mock_full.assert_not_called()


if __name__ == "__main__":
    unittest.main()
