import math
import unittest
from unittest.mock import Mock, MagicMock

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.agent.mcts.base_mcts import Node, Edge
from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_reinforcement.agent.mcts.alphazero import MCTS_AlphaZero
from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.value_head import ValueHead


class TestNode(unittest.TestCase):
    def test_node_initialization(self) -> None:
        state = Mock(spec=TacticState)
        node = Node(state)
        self.assertEqual(node.state, state)
        self.assertEqual(node.children, [])
        self.assertEqual(node.visit_count, 0)
        self.assertFalse(node.is_terminal)
        self.assertIsNone(node.untried_actions)

    def test_node_value(self) -> None:
        node = Node(Mock(spec=TacticState))
        self.assertEqual(node.value(), 0.0)
        node.visit_count = 10
        node.max_value = 0.8
        self.assertEqual(node.value(), 0.8)

    def test_is_fully_expanded(self) -> None:
        node = Node(Mock(spec=TacticState))
        self.assertFalse(node.is_fully_expanded())
        node.untried_actions = ["tactic1", "tactic2"]
        self.assertFalse(node.is_fully_expanded())
        node.untried_actions = []
        self.assertTrue(node.is_fully_expanded())

    def test_is_terminal(self) -> None:
        self.assertFalse(Node(Mock(spec=TacticState)).is_terminal)
        self.assertTrue(Node(Mock(spec=ProofFinished)).is_terminal)
        self.assertTrue(Node(Mock(spec=LeanError)).is_terminal)
        self.assertTrue(Node(Mock(spec=ProofGivenUp)).is_terminal)


class MockLeanDojoEnv(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_state = Mock(spec=TacticState)
        self.current_state.pp = "mock_initial_state"  # Add pp attribute for seen_states
        self.theorem = "mock_theorem"
        self.theorem_pos = "mock_pos"
        self.dataloader = Mock()
        self.dataloader.get_premises.return_value = ["p1", "p2"]
        self.dojo = Mock()  # Changed from dojo_instance to dojo


class TestBaseMCTS(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MockLeanDojoEnv()
        self.transformer = Mock(spec=Transformer)

    def test_base_mcts_initialization(self) -> None:
        mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)
        self.assertIsInstance(mcts.root, Node)
        self.assertIsNotNone(mcts.root)
        assert mcts.root is not None  # Type narrowing for mypy
        self.assertEqual(mcts.root.state, self.env.current_state)

    def test_backpropagate(self) -> None:
        mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)
        node1 = Node(Mock(spec=TacticState))
        node2 = Node(Mock(spec=TacticState))
        node3 = Node(Mock(spec=TacticState))

        edge1 = Edge(action="t1", prior=0.5, child=node2)
        edge2 = Edge(action="t2", prior=0.5, child=node3)

        node1.children.append(edge1)
        node2.children.append(edge2)

        # Path is list of (Node, Edge)
        path = [(node1, edge1), (node2, edge2), (node3, None)]

        mcts._backpropagate(path, 0.5)

        self.assertEqual(node3.visit_count, 1)
        self.assertEqual(node3.max_value, 0.5)

        self.assertEqual(node2.visit_count, 1)
        self.assertEqual(node2.max_value, 0.5)
        self.assertEqual(edge2.visit_count, 1)

        self.assertEqual(node1.visit_count, 1)
        self.assertEqual(node1.max_value, 0.5)
        self.assertEqual(edge1.visit_count, 1)

    def test_move_root(self) -> None:
        mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)
        root = mcts.root
        self.assertIsNotNone(root)
        assert root is not None  # Type narrowing for mypy

        # Create children manually with pp attributes for seen_states
        child1_state = Mock(spec=TacticState)
        child1_state.pp = "child1_state_pp"
        child1 = Node(child1_state, parent=root, action="tactic1")

        child2_state = Mock(spec=TacticState)
        child2_state.pp = "child2_state_pp"
        child2 = Node(child2_state, parent=root, action="tactic2")
        root.children = [child1, child2]

        # Add some grandchildren to test node counting
        grandchild_state = Mock(spec=TacticState)
        grandchild_state.pp = "grandchild_state_pp"
        grandchild = Node(grandchild_state, parent=child1, action="tactic1_1")
        child1.children = [grandchild]

        # Test moving to an existing child
        mcts.move_root("tactic1")
        self.assertIs(mcts.root, child1)
        # After move_root, unreachable nodes (root, child2) are pruned
        # Only child1 and grandchild remain reachable
        self.assertEqual(mcts.node_count, 2)

        # Test moving to a non-existent child (should reset)
        # First, update env.current_state to match what we expect for a reset
        new_state = Mock(spec=TacticState)
        new_state.pp = "new_mock_state"  # Add pp attribute for seen_states
        self.env.current_state = new_state

        mcts.move_root("non_existent_tactic")
        self.assertIsNot(mcts.root, child1)
        self.assertIsNotNone(mcts.root)
        assert mcts.root is not None  # Type narrowing for mypy
        self.assertEqual(mcts.root.state, new_state)
        # Reset clears nodes
        self.assertEqual(mcts.node_count, 1)


class TestMCTSGuidedRollout(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MockLeanDojoEnv()
        self.transformer = Mock(spec=Transformer)
        self.mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)

    def test_puct_score(self) -> None:
        parent = Node(Mock(spec=TacticState))
        parent.visit_count = 10

        child = Node(Mock(spec=TacticState))
        child.max_value = 0.8

        edge = Edge(action="t1", prior=0.5, child=child)
        edge.visit_count = 1

        score = self.mcts._puct_score(parent, edge)
        expected_score = 0.8 + self.mcts.exploration_weight * 0.5 * (
            math.sqrt(10.0) / (1 + 1)
        )
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_expand(self) -> None:
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.transformer.generate_tactics_with_probs.return_value = [("tactic1", 0.5)]
        next_state = Mock(spec=TacticState)
        next_state.pp = (
            "next_state_pp"  # Different from state.pp to avoid no-op filtering
        )
        self.env.run_tactic_stateless = Mock(return_value=next_state)

        node_res, edge_res = self.mcts._expand(node)

        self.assertEqual(len(node.children), 1)
        edge = node.children[0]
        self.assertEqual(edge.action, "tactic1")
        self.assertEqual(edge.prior, 0.5)
        self.assertIsInstance(edge.child, Node)

        self.assertIs(node_res, node)
        self.assertIs(edge_res, edge)  # Best edge is the only edge

        self.transformer.generate_tactics_with_probs.assert_called_once()
        self.env.run_tactic_stateless.assert_called_once_with(state, "tactic1")

    def test_simulate_proof_finished(self) -> None:
        node = Node(Mock(spec=ProofFinished))
        reward = self.mcts._simulate(node)
        self.assertEqual(reward, 1.0)

    def test_simulate_rollout(self) -> None:
        initial_state = Mock(spec=TacticState)
        initial_state.pp = "initial_state"
        node = Node(initial_state)

        self.transformer.generate_tactics.side_effect = [["tactic1"], ["tactic2"]]

        # First step in rollout leads to another tactic state
        intermediate_state = Mock(spec=TacticState)
        intermediate_state.pp = "intermediate_state"
        self.env.run_tactic_stateless = Mock(
            side_effect=[
                intermediate_state,
                Mock(spec=ProofFinished),
            ]
        )

        reward = self.mcts._simulate(node)
        self.assertAlmostEqual(reward, 0.98)
        self.assertEqual(self.env.run_tactic_stateless.call_count, 2)


class TestMCTSAlphaZero(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MockLeanDojoEnv()
        self.transformer = Mock(spec=Transformer)
        self.value_head = Mock(spec=ValueHead)
        self.mcts = MCTS_AlphaZero(
            value_head=self.value_head, env=self.env, transformer=self.transformer
        )

    def test_puct_score(self) -> None:
        parent = Node(Mock(spec=TacticState))
        parent.visit_count = 10

        child = Node(Mock(spec=TacticState))
        child.max_value = 0.8

        edge = Edge(action="t1", prior=0.8, child=child)
        edge.visit_count = 1

        score = self.mcts._puct_score(parent, edge)
        # Q-value is now max_value (0.8) instead of mean (0.5)
        expected_score = 0.8 + self.mcts.exploration_weight * 0.8 * ((10) ** 0.5 / 2)
        self.assertAlmostEqual(score, expected_score, places=5)

        child_unvisited = Node(Mock(spec=TacticState))
        child_unvisited.max_value = float("-inf")

        edge_unvisited = Edge(action="t2", prior=0.5, child=child_unvisited)
        edge_unvisited.visit_count = 0

        score_unvisited = self.mcts._puct_score(parent, edge_unvisited)
        expected_score_unvisited = 0.0 + self.mcts.exploration_weight * 0.5 * (
            (10) ** 0.5 / 1
        )
        self.assertAlmostEqual(score_unvisited, expected_score_unvisited, places=5)

    def test_expand_alphazero(self) -> None:
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.transformer.generate_tactics_with_probs.return_value = [
            ("tactic1", 0.6),
            ("tactic2", 0.4),
        ]
        # Create unique next states to avoid duplicate filtering
        next_state_1 = Mock(spec=TacticState)
        next_state_1.pp = "next_state_pp_1"
        next_state_2 = Mock(spec=TacticState)
        next_state_2.pp = "next_state_pp_2"
        self.env.run_tactic_stateless = Mock(side_effect=[next_state_1, next_state_2])
        # Mock encode_states to return a tensor with proper shape for batch slicing
        import torch

        self.value_head.encode_states = Mock(
            return_value=torch.zeros(2, 1472)  # 2 children, 1472 features
        )

        expanded_node, best_edge = self.mcts._expand(node)
        self.assertIs(expanded_node, node)
        self.assertEqual(len(node.children), 2)

        edge1 = node.children[0]
        self.assertEqual(edge1.action, "tactic1")
        self.assertEqual(edge1.prior, 0.6)

        edge2 = node.children[1]
        self.assertEqual(edge2.action, "tactic2")
        self.assertEqual(edge2.prior, 0.4)

        self.assertTrue(node.is_fully_expanded())

    def test_simulate_alphazero(self) -> None:
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.value_head.predict.return_value = 0.75

        value = self.mcts._simulate(node)

        self.assertEqual(value, 0.75)
        self.value_head.predict.assert_called_once_with("state_pp")


if __name__ == "__main__":
    unittest.main()
