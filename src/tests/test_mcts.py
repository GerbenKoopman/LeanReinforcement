import math
import unittest
from unittest.mock import Mock, MagicMock

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from src.agent.mcts import Node, MCTS_GuidedRollout, MCTS_AlphaZero
from src.agent.premise_selection import PremiseSelector
from src.agent.tactic_generation import TacticGenerator
from src.agent.value_head import ValueHead


class TestNode(unittest.TestCase):
    def test_node_initialization(self):
        state = Mock(spec=TacticState)
        node = Node(state)
        self.assertEqual(node.state, state)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.action)
        self.assertEqual(node.prior_p, 0.0)
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertFalse(node.is_terminal)
        self.assertIsNone(node.untried_actions)

    def test_node_value(self):
        node = Node(Mock(spec=TacticState))
        self.assertEqual(node.value(), 0.0)
        node.visit_count = 10
        node.value_sum = 5.0
        self.assertEqual(node.value(), 0.5)

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
        self.dojo_instance = Mock()


class TestBaseMCTS(unittest.TestCase):
    def setUp(self):
        self.env = MockLeanDojoEnv()
        self.premise_selector = Mock(spec=PremiseSelector)
        self.tactic_generator = Mock(spec=TacticGenerator)

    def test_base_mcts_initialization(self):
        all_premises = ["p1", "p2"]
        mcts = MCTS_GuidedRollout(
            self.env, self.premise_selector, self.tactic_generator, all_premises
        )
        self.assertIsInstance(mcts.root, Node)
        self.assertEqual(mcts.root.state, self.env.current_state)
        self.assertEqual(mcts.all_premises, ["p1", "p2"])

    def test_backpropagate(self):
        all_premises = ["p1", "p2"]
        mcts = MCTS_GuidedRollout(
            self.env, self.premise_selector, self.tactic_generator, all_premises
        )
        node1 = Node(Mock(spec=TacticState))
        node2 = Node(Mock(spec=TacticState), parent=node1)
        node3 = Node(Mock(spec=TacticState), parent=node2)

        mcts._backpropagate(node3, 0.5)

        self.assertEqual(node3.visit_count, 1)
        self.assertEqual(node3.value_sum, 0.5)
        self.assertEqual(node2.visit_count, 1)
        self.assertEqual(node2.value_sum, 0.5)
        self.assertEqual(node1.visit_count, 1)
        self.assertEqual(node1.value_sum, 0.5)


class TestMCTSGuidedRollout(unittest.TestCase):
    def setUp(self):
        self.env = MockLeanDojoEnv()
        self.premise_selector = Mock(spec=PremiseSelector)
        self.tactic_generator = Mock(spec=TacticGenerator)
        self.all_premises = ["p1", "p2"]
        self.mcts = MCTS_GuidedRollout(
            self.env, self.premise_selector, self.tactic_generator, self.all_premises
        )

    def test_ucb1(self):
        parent = Node(Mock(spec=TacticState))
        parent.visit_count = 10
        child = Node(Mock(spec=TacticState), parent=parent)
        child.visit_count = 1
        child.value_sum = 0.5
        score = self.mcts._ucb1(child)
        expected_score = 0.5 + self.mcts.exploration_weight * (math.log(10.0) ** 0.5)
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_expand(self):
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.premise_selector.retrieve.return_value = ["retrieved_premise"]
        self.tactic_generator.generate_tactics.return_value = ["tactic1"]
        self.env.dojo_instance.run_tac.return_value = Mock(spec=TacticState)

        child = self.mcts._expand(node)

        self.assertEqual(len(node.children), 1)
        self.assertIs(child.parent, node)
        self.assertEqual(child.action, "tactic1")
        self.premise_selector.retrieve.assert_called_once()
        self.tactic_generator.generate_tactics.assert_called_once()
        self.env.dojo_instance.run_tac.assert_called_once_with(state, "tactic1")

    def test_simulate_proof_finished(self):
        node = Node(Mock(spec=ProofFinished))
        reward = self.mcts._simulate(node)
        self.assertEqual(reward, 1.0)

    def test_simulate_rollout(self):
        initial_state = Mock(spec=TacticState)
        initial_state.pp = "initial_state"
        node = Node(initial_state)

        self.premise_selector.retrieve.return_value = ["retrieved_premise"]
        self.tactic_generator.generate_tactics.side_effect = [["tactic1"], ["tactic2"]]

        # First step in rollout leads to another tactic state
        intermediate_state = Mock(spec=TacticState)
        intermediate_state.pp = "intermediate_state"
        self.env.dojo_instance.run_tac.side_effect = [
            intermediate_state,
            Mock(spec=ProofFinished),
        ]

        reward = self.mcts._simulate(node)
        self.assertEqual(reward, 1.0)
        self.assertEqual(self.env.dojo_instance.run_tac.call_count, 2)


class TestMCTSAlphaZero(unittest.TestCase):
    def setUp(self):
        self.env = MockLeanDojoEnv()
        self.premise_selector = Mock(spec=PremiseSelector)
        self.tactic_generator = Mock(spec=TacticGenerator)
        self.value_head = Mock(spec=ValueHead)
        self.all_premises = ["p1", "p2"]
        self.mcts = MCTS_AlphaZero(
            self.value_head,
            self.env,
            self.premise_selector,
            self.tactic_generator,
            self.all_premises,
        )

    def test_puct_score(self):
        parent = Node(Mock(spec=TacticState))
        parent.visit_count = 10
        child = Node(Mock(spec=TacticState), parent=parent)
        child.visit_count = 1
        child.value_sum = 0.5
        child.prior_p = 0.8
        score = self.mcts._puct_score(child)
        expected_score = 0.5 + self.mcts.exploration_weight * 0.8 * ((10) ** 0.5 / 2)
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_expand_alphazero(self):
        state = Mock(spec=TacticState)
        state.pp = "state_pp"
        node = Node(state)
        self.premise_selector.retrieve.return_value = ["retrieved_premise"]
        self.tactic_generator.generate_tactics_with_probs.return_value = [
            ("tactic1", 0.6),
            ("tactic2", 0.4),
        ]
        self.env.dojo_instance.run_tac.return_value = Mock(spec=TacticState)

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
        self.premise_selector.retrieve.return_value = ["retrieved_premise"]
        self.value_head.predict.return_value = 0.75

        value = self.mcts._simulate(node)

        self.assertEqual(value, 0.75)
        self.premise_selector.retrieve.assert_called_once_with(
            "state_pp", self.mcts.all_premises, k=10
        )
        self.value_head.predict.assert_called_once_with(
            "state_pp", ["retrieved_premise"]
        )
