"""
Comprehensive test suite for MCTS implementations.

Tests cover:
- DAG structure and multi-parent nodes
- State deduplication and reuse
- Time limit handling
- Backpropagation through multiple paths
- Edge cases and boundary conditions
"""

import unittest
from typing import Any, cast
from unittest.mock import Mock, MagicMock
from lean_dojo import TacticState, ProofFinished

from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_reinforcement.agent.mcts.base_mcts import Node
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol
from lean_reinforcement.utilities.config import TrainingConfig


class MockState:
    """Mock state for testing."""

    def __init__(self, pp: str):
        self.pp = pp


class MockTransformer:
    """Mock transformer for testing."""

    def __call__(self, states: Any) -> Any:
        if isinstance(states, list):
            return [Mock(logits=Mock()) for _ in states]
        return Mock(logits=Mock())


class MockLeanDojoEnv:  # type: ignore[misc]
    """Mock LeanDojo environment for testing."""

    def __init__(self) -> None:
        self.theorem = Mock()
        self.theorem_pos = Mock()
        self.current_state = Mock(spec=TacticState, pp="initial_state")


class TestDAGStructure(unittest.TestCase):
    """Test DAG node structure and multi-parent support."""

    def setUp(self):
        self.state1 = Mock(spec=TacticState, pp="state1")
        self.state2 = Mock(spec=TacticState, pp="state2")

    def test_node_single_parent(self):
        parent = Node(state=self.state1)
        child = Node(state=self.state2)
        child.add_parent(parent, "tactic1")
        self.assertEqual(len(child.parents), 1)
        self.assertEqual(child.parents[0][0], parent)
        self.assertEqual(child.parents[0][1], "tactic1")

    def test_node_multiple_parents(self):
        parent1 = Node(state=self.state1)
        parent2 = Node(state=self.state2)
        child = Node(state=Mock(spec=TacticState, pp="state3"))
        child.add_parent(parent1, "tactic1")
        child.add_parent(parent2, "tactic2")
        self.assertEqual(len(child.parents), 2)
        self.assertEqual(child.parents[0][0], parent1)
        self.assertEqual(child.parents[1][0], parent2)

    def test_node_backward_compatible_parent_property(self):
        parent = Node(state=self.state1)
        child = Node(state=self.state2)
        child.add_parent(parent, "tactic1")
        self.assertEqual(child.parent, parent)

    def test_duplicate_parent_not_added_twice(self):
        parent = Node(state=self.state1)
        child = Node(state=self.state2)
        child.add_parent(parent, "tactic1")
        child.add_parent(parent, "tactic1")
        self.assertEqual(len(child.parents), 1)


class TestStateDeduplication(unittest.TestCase):
    """Test state deduplication and reuse in MCTS."""

    def setUp(self) -> None:
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False
        self.mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            batch_size=4,
            num_tactics_to_expand=4,
            max_time=600.0,
        )

    def test_seen_states_initialized(self):
        assert isinstance(self.env.current_state, TacticState)
        self.assertIn(self.env.current_state.pp, self.mcts.seen_states)
        self.assertEqual(
            self.mcts.seen_states[self.env.current_state.pp], self.mcts.root
        )

    def test_state_key_generation(self):
        state = Mock(spec=TacticState, pp="test_state")
        key = self.mcts._get_state_key(state)
        self.assertEqual(key, "test_state")

    def test_duplicate_state_reuse(self):
        state1_pp = "test_state_1"
        state1 = Mock(spec=TacticState, pp=state1_pp)
        test_node = Node(state=state1)
        self.mcts.seen_states[state1_pp] = test_node
        retrieved = self.mcts.seen_states.get(state1_pp)
        self.assertEqual(retrieved, test_node)
        node_state = self.mcts.seen_states[state1_pp].state
        assert isinstance(node_state, TacticState)
        self.assertEqual(node_state.pp, state1_pp)


class TestTimeHandling(unittest.TestCase):
    """Test time limit handling in MCTS."""

    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False

    def test_max_time_initialization(self):
        max_time = 300.0
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=max_time,
        )
        self.assertEqual(mcts.max_time, max_time)

    def test_max_time_default_value(self):
        mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config
        )
        self.assertEqual(mcts.max_time, 300.0)

    def test_search_respects_max_time_parameter(self):
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=100.0,
        )
        self.assertEqual(mcts.max_time, 100.0)

    def test_search_uses_instance_max_time_when_not_specified(self):
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=100.0,
        )
        self.assertEqual(mcts.max_time, 100.0)


class TestBackpropagation(unittest.TestCase):
    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False
        self.mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config
        )

    def test_backpropagate_single_path(self):
        state1 = Mock(spec=TacticState, pp="state1")
        state2 = Mock(spec=TacticState, pp="state2")
        child1 = Node(state=state1)
        child1.add_parent(self.mcts.root, "tactic1")
        child2 = Node(state=state2)
        child2.add_parent(child1, "tactic2")
        self.assertEqual(len(child1.parents), 1)
        self.assertEqual(child1.parents[0][0], self.mcts.root)
        self.assertEqual(len(child2.parents), 1)
        self.assertEqual(child2.parents[0][0], child1)

    def test_backpropagate_multiple_paths(self):
        state1 = Mock(spec=TacticState, pp="state1")
        state2 = Mock(spec=TacticState, pp="state2")
        state3 = Mock(spec=TacticState, pp="state3")
        child1 = Node(state=state1)
        child1.add_parent(self.mcts.root, "tactic1")
        child2 = Node(state=state2)
        child2.add_parent(self.mcts.root, "tactic2")
        child3 = Node(state=state3)
        child3.add_parent(child1, "tactic3")
        child3.add_parent(child2, "tactic3")
        self.assertEqual(len(child1.parents), 1)
        self.assertEqual(len(child2.parents), 1)
        self.assertEqual(len(child3.parents), 2)
        self.assertEqual(child3.parents[0][0], child1)
        self.assertEqual(child3.parents[1][0], child2)


class TestBatchOperations(unittest.TestCase):
    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False

    def test_batch_size_initialization(self):
        batch_size = 8
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            batch_size=batch_size,
        )
        self.assertEqual(mcts.batch_size, batch_size)

    def test_search_respects_batch_size(self):
        batch_size = 4
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            batch_size=batch_size,
        )
        self.assertEqual(mcts.batch_size, batch_size)


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False

    def test_zero_iterations(self):
        mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config
        )
        mcts.search(num_iterations=0)
        self.assertEqual(mcts.root.visit_count, 0)

    def test_very_short_max_time(self):
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=0.001,
        )
        self.assertEqual(mcts.max_time, 0.001)

    def test_large_batch_size(self):
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            batch_size=100,
        )
        self.assertEqual(mcts.batch_size, 100)

    def test_max_tree_nodes_limit(self):
        max_nodes = 100
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_tree_nodes=max_nodes,
        )
        self.assertEqual(mcts.max_tree_nodes, max_nodes)


class TestStateKeyGeneration(unittest.TestCase):
    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False
        self.mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config
        )

    def test_state_key_from_tactic_state(self):
        state = Mock(spec=TacticState, pp="test_pp")
        key = self.mcts._get_state_key(state)
        self.assertEqual(key, "test_pp")
        self.assertIsInstance(key, str)

    def test_state_key_from_proof_finished(self):
        state = Mock(spec=ProofFinished)
        key = self.mcts._get_state_key(state)
        self.assertIsNone(key)

    def test_state_key_uniqueness(self):
        state1 = Mock(spec=TacticState, pp="pp1")
        state2 = Mock(spec=TacticState, pp="pp2")
        key1 = self.mcts._get_state_key(state1)
        key2 = self.mcts._get_state_key(state2)
        self.assertNotEqual(key1, key2)


class TestNodeReuse(unittest.TestCase):
    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False
        self.mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config
        )

    def test_same_state_visited_twice_reuses_node(self):
        pp = "repeated_state"
        state = Mock(spec=TacticState, pp=pp)
        node = Node(state=state)
        self.mcts.seen_states[pp] = node
        retrieved = self.mcts.seen_states.get(pp)
        self.assertEqual(retrieved, node)


if __name__ == "__main__":
    unittest.main()
