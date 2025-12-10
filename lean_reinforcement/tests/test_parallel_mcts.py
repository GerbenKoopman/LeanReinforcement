import unittest
from unittest.mock import MagicMock, patch
from lean_reinforcement.agent.mcts.base_mcts import Node
from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_dojo import TacticState


class TestParallelMCTS(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock()
        self.env.current_state = TacticState("pp", 0, "id")
        self.env.theorem = MagicMock()
        self.env.theorem_pos = MagicMock()

        self.transformer = MagicMock()

        self.mcts = MCTS_GuidedRollout(env=self.env, transformer=self.transformer)

    def test_virtual_loss(self):
        node = Node(TacticState("pp", 0, "id"))

        self.assertEqual(self.mcts._get_virtual_loss(node), 0)

        self.mcts._add_virtual_loss(node)
        self.assertEqual(self.mcts._get_virtual_loss(node), 1)

        self.mcts._add_virtual_loss(node)
        self.assertEqual(self.mcts._get_virtual_loss(node), 2)

        self.mcts._remove_virtual_loss(node)
        self.assertEqual(self.mcts._get_virtual_loss(node), 1)

        self.mcts._remove_virtual_loss(node)
        self.assertEqual(self.mcts._get_virtual_loss(node), 0)

    def test_expand_batch(self):
        # Setup nodes
        node1 = Node(TacticState("pp1", 0, "id1"))
        node2 = Node(TacticState("pp2", 0, "id2"))
        nodes = [node1, node2]

        # Mock transformer batch generation
        self.transformer.generate_tactics_with_probs_batch.return_value = [
            [("t1", 0.5)],
            [("t2", 0.6)],
        ]

        # Mock run_tac on the single env
        self.env.dojo.run_tac.side_effect = [
            TacticState("pp_next1", 0, "id_next1"),
            TacticState("pp_next2", 0, "id_next2"),
        ]

        children = self.mcts._expand_batch(nodes)

        self.assertEqual(len(children), 2)
        self.assertEqual(children[0].action, "t1")
        self.assertEqual(children[1].action, "t2")
        self.assertEqual(len(node1.children), 1)
        self.assertEqual(len(node2.children), 1)
        self.assertEqual(node1.children[0].prior_p, 0.5)
        self.assertEqual(node2.children[0].prior_p, 0.6)

        # Verify batch generation called
        self.transformer.generate_tactics_with_probs_batch.assert_called_once()

    def test_simulate_batch(self):
        node1 = Node(TacticState("pp1", 0, "id1"))
        node2 = Node(TacticState("pp2", 0, "id2"))
        nodes = [node1, node2]

        # Mock _simulate behavior
        with patch.object(
            self.mcts, "_simulate", side_effect=[1.0, -1.0]
        ) as mock_simulate:
            rewards = self.mcts._simulate_batch(nodes)

            self.assertEqual(rewards, [1.0, -1.0])
            self.assertEqual(mock_simulate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
