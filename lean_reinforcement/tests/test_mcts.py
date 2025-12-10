import math
import unittest
from unittest.mock import Mock, MagicMock

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.agent.mcts.base_mcts import Node
from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_reinforcement.agent.mcts.alphazero import MCTS_AlphaZero
from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.value_head import ValueHead


class TestNode(unittest.TestCase):
    def test_node_initialization(self):
        state = Mock(spec=TacticState)
        node = Node(state)
        self.assertEqual(node.state, state)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.action)
        self.assertEqual(node.prior_p, 0.0)
        self.assertEqual(node.visit_count, 0)
        self.assertFalse(node.is_terminal)
        self.assertIsNone(node.untried_actions)

    def test_node_value(self):
        node = Node(Mock(spec=TacticState))
        self.assertEqual(node.value(), 0.0)
        node.visit_count = 10
        node.max_value = 0.8
        self.assertEqual(node.value(), 0.8)

    def test_is_fully_expanded(self):
        node = Node(Mock(spec=TacticState))
        self.assertFalse(node.is_fully_expanded())
        node.untried_actions = ["tactic1", "tactic2"]
        self.assertFalse(node.is_fully_expanded())
        node.untried_actions = []
        self.assertTrue(node.is_fully_expanded())

    def test_is_terminal(self):
        self.assertFalse(Node(Mock(spec=TacticState)).is_terminal)
        self.assertTrue(Node(Mock(spec=ProofFinished)).is_terminal)
        self.assertTrue(Node(Mock(spec=LeanError)).is_terminal)
        self.assertTrue(Node(Mock(spec=ProofGivenUp)).is_terminal)


class MockLeanDojoEnv(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_state = Mock(spec=TacticState)
        self.theorem = "mock_theorem"
        self.theorem_pos = "mock_pos"
        self.dataloader = Mock()
        self.dataloader.get_premises.return_value = ["p1", "p2"]
        self.dojo = Mock()  # Changed from dojo_instance to dojo


class TestBaseMCTS(unittest.TestCase):
    def setUp(self):
        self.env = MockLeanDojoEnv()
        self.transformer = Mock(spec=Transformer)

    def test_base_mcts_initialization(self):
        mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)
        self.assertIsInstance(mcts.root, Node)
        self.assertEqual(mcts.root.state, self.env.current_state)

    def test_backpropagate(self):
        mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)
        node1 = Node(Mock(spec=TacticState))
        node2 = Node(Mock(spec=TacticState), parent=node1)
        node3 = Node(Mock(spec=TacticState), parent=node2)

        mcts._backpropagate(node3, 0.5)

        self.assertEqual(node3.visit_count, 1)
        self.assertEqual(node3.max_value, 0.5)
        self.assertEqual(node2.visit_count, 1)
        self.assertEqual(node2.max_value, 0.5)
        self.assertEqual(node1.visit_count, 1)
        self.assertEqual(node1.max_value, 0.5)

    def test_move_root(self):
        mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)
        root = mcts.root

        # Create children manually
        child1 = Node(Mock(spec=TacticState), parent=root, action="tactic1")
        child2 = Node(Mock(spec=TacticState), parent=root, action="tactic2")
        root.children = [child1, child2]

        # Add some grandchildren to test node counting
        grandchild = Node(Mock(spec=TacticState), parent=child1, action="tactic1_1")
        child1.children = [grandchild]

        # Test moving to an existing child
        mcts.move_root("tactic1")
        self.assertIs(mcts.root, child1)
        self.assertIsNone(mcts.root.parent)
        self.assertEqual(mcts.node_count, 2)  # child1 + grandchild

        # Test moving to a non-existent child (should reset)
        # First, update env.current_state to match what we expect for a reset
        new_state = Mock(spec=TacticState)
        self.env.current_state = new_state

        mcts.move_root("non_existent_tactic")
        self.assertIsNot(mcts.root, child1)
        self.assertEqual(mcts.root.state, new_state)
        self.assertEqual(mcts.node_count, 1)


class TestMCTSGuidedRollout(unittest.TestCase):
    def setUp(self):
        self.env = MockLeanDojoEnv()
        self.transformer = Mock(spec=Transformer)
        self.mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)

    def test_puct_score(self):
        parent = Node(Mock(spec=TacticState))
        parent.visit_count = 10
        child = Node(Mock(spec=TacticState), parent=parent)
        child.visit_count = 1
        child.max_value = 0.8
        child.prior_p = 0.5

        score = self.mcts._puct_score(child)
        expected_score = 0.8 + self.mcts.exploration_weight * 0.5 * (
            math.sqrt(10.0) / (1 + 1)
        )
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_expand(self):
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.transformer.generate_tactics_with_probs.return_value = [("tactic1", 0.5)]
        self.env.dojo.run_tac.return_value = Mock(spec=TacticState)

        child = self.mcts._expand(node)

        self.assertEqual(len(node.children), 1)
        self.assertIs(child.parent, node)
        self.assertEqual(child.action, "tactic1")
        self.assertEqual(child.prior_p, 0.5)
        self.transformer.generate_tactics_with_probs.assert_called_once()
        self.env.dojo.run_tac.assert_called_once_with(state, "tactic1")

    def test_simulate_proof_finished(self):
        node = Node(Mock(spec=ProofFinished))
        reward = self.mcts._simulate(node)
        self.assertEqual(reward, 1.0)

    def test_simulate_rollout(self):
        initial_state = Mock(spec=TacticState)
        initial_state.pp = "initial_state"
        node = Node(initial_state)

        self.transformer.generate_tactics.side_effect = [["tactic1"], ["tactic2"]]

        # First step in rollout leads to another tactic state
        intermediate_state = Mock(spec=TacticState)
        intermediate_state.pp = "intermediate_state"
        self.env.dojo.run_tac.side_effect = [
            intermediate_state,
            Mock(spec=ProofFinished),
        ]

        reward = self.mcts._simulate(node)
        self.assertEqual(reward, 1.0)
        self.assertEqual(self.env.dojo.run_tac.call_count, 2)


class TestMCTSAlphaZero(unittest.TestCase):
    def setUp(self):
        self.env = MockLeanDojoEnv()
        self.transformer = Mock(spec=Transformer)
        self.value_head = Mock(spec=ValueHead)
        self.mcts = MCTS_AlphaZero(
            value_head=self.value_head, env=self.env, transformer=self.transformer
        )

    def test_puct_score(self):
        parent = Node(Mock(spec=TacticState))
        parent.visit_count = 10

        child = Node(Mock(spec=TacticState), parent=parent)
        child.visit_count = 1
        child.max_value = 0.8
        child.prior_p = 0.8

        score = self.mcts._puct_score(child)
        # Q-value is now max_value (0.8) instead of mean (0.5)
        expected_score = 0.8 + self.mcts.exploration_weight * 0.8 * ((10) ** 0.5 / 2)
        self.assertAlmostEqual(score, expected_score, places=5)

        child_unvisited = Node(Mock(spec=TacticState), parent=parent)
        child_unvisited.visit_count = 0
        child_unvisited.max_value = float("-inf")
        child_unvisited.prior_p = 0.5

        score_unvisited = self.mcts._puct_score(child_unvisited)
        expected_score_unvisited = 0.0 + self.mcts.exploration_weight * 0.5 * (
            (10) ** 0.5 / 1
        )
        self.assertAlmostEqual(score_unvisited, expected_score_unvisited, places=5)

    def test_expand_alphazero(self):
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.transformer.generate_tactics_with_probs.return_value = [
            ("tactic1", 0.6),
            ("tactic2", 0.4),
        ]
        next_state_mock = Mock(spec=TacticState)
        next_state_mock.pp = "next_state_pp"
        self.env.dojo.run_tac.return_value = next_state_mock

        expanded_node = self.mcts._expand(node)
        self.assertIs(expanded_node, node)
        self.assertEqual(len(node.children), 2)
        self.assertEqual(node.children[0].action, "tactic1")
        self.assertEqual(node.children[0].prior_p, 0.6)
        self.assertEqual(node.children[1].action, "tactic2")
        self.assertEqual(node.children[1].prior_p, 0.4)
        self.assertTrue(node.is_fully_expanded())

    def test_simulate_alphazero(self):
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.value_head.predict.return_value = 0.75

        value = self.mcts._simulate(node)

        self.assertEqual(value, 0.75)
        self.value_head.predict.assert_called_once_with("state_pp")


if __name__ == "__main__":
    unittest.main()
