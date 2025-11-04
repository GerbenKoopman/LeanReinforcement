import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import gymnasium as gym

from lean_dojo import TacticState, ProofFinished, LeanError, Theorem
from ReProver.common import Pos

from src.utilities.gym import LeanDojoEnv


class TestLeanDojoEnv(unittest.TestCase):
    @patch("src.utilities.gym.Dojo")
    @patch("src.utilities.gym.DataLoader")
    def setUp(self, MockDataLoader, MockDojo):
        # Mock dependencies
        self.mock_dataloader = MockDataLoader.return_value
        self.mock_dojo_context = MockDojo.return_value
        self.mock_dojo_instance = MagicMock()

        # Mock theorem and position
        self.theorem = MagicMock(spec=Theorem)
        self.theorem_pos = Pos(1, 0)

        # Mock initial state from Dojo context manager
        self.initial_state = MagicMock(spec=TacticState)
        self.initial_state.pp = "initial_state_pp"
        self.mock_dojo_context.__enter__.return_value = (
            self.mock_dojo_instance,
            self.initial_state,
        )

        # Mock dataloader to return a fixed number of premises
        self.mock_dataloader.get_premises.return_value = ["p1", "p2", "p3"]

        # Instantiate the environment
        self.env = LeanDojoEnv(self.theorem, self.theorem_pos, k=2)

    def test_initialization(self):
        # Assert that dependencies were called correctly
        self.mock_dataloader.get_premises.assert_called_once_with(
            self.theorem, self.theorem_pos
        )
        self.assertIsInstance(self.env.observation_space, gym.spaces.Text)
        self.assertIsInstance(self.env.action_space, gym.spaces.MultiDiscrete)
        assert isinstance(self.env.action_space, gym.spaces.MultiDiscrete)
        self.assertEqual(self.env.action_space.nvec.tolist(), [3, 3])
        self.assertEqual(self.env.current_state, self.initial_state)

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs, "initial_state_pp")
        self.assertEqual(info, {})
        self.mock_dojo_context.__enter__.assert_called()

    def test_step_tactic_state(self):
        # Arrange
        action = "test_tactic"
        next_tactic_state = MagicMock(spec=TacticState)
        next_tactic_state.pp = "next_state_pp"
        self.mock_dojo_instance.run_tac.return_value = next_tactic_state

        # Act
        obs, reward, done, _, _ = self.env.step(action)

        # Assert
        self.mock_dojo_instance.run_tac.assert_called_once_with(
            self.initial_state, action
        )
        self.assertEqual(self.env.current_state, next_tactic_state)
        self.assertEqual(obs, "next_state_pp")
        self.assertEqual(reward, 0.1)
        self.assertFalse(done)

    def test_step_proof_finished(self):
        # Arrange
        action = "finish_proof_tactic"
        proof_finished_state = MagicMock(spec=ProofFinished)
        type(proof_finished_state).pp = PropertyMock(return_value="proof_finished")
        self.mock_dojo_instance.run_tac.return_value = proof_finished_state

        # Act
        obs, reward, done, _, _ = self.env.step(action)

        # Assert
        self.assertEqual(obs, str(proof_finished_state))
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)

    def test_step_lean_error(self):
        # Arrange
        action = "error_tactic"
        lean_error_state = MagicMock(spec=LeanError)
        type(lean_error_state).pp = PropertyMock(return_value="lean_error")
        self.mock_dojo_instance.run_tac.return_value = lean_error_state

        # Act
        obs, reward, done, _, _ = self.env.step(action)

        # Assert
        self.assertEqual(obs, str(lean_error_state))
        self.assertEqual(reward, -0.1)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
