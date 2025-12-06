import unittest
from unittest.mock import patch, MagicMock, PropertyMock

from lean_dojo import TacticState, ProofFinished, LeanError, Theorem
from ReProver.common import Pos

from lean_reinforcement.utilities.gym import LeanDojoEnv


class TestLeanDojoEnv(unittest.TestCase):
    @patch("lean_reinforcement.utilities.gym.Dojo")
    def setUp(self, MockDojo):
        # Mock theorem and position
        self.theorem = MagicMock(spec=Theorem)
        self.theorem.file_path = "path/to/file.lean"
        self.theorem_pos = Pos(1, 0)

        # Mock initial state from Dojo context manager
        self.initial_state = MagicMock(spec=TacticState)
        self.initial_state.pp = "initial_state_pp"

        self.mock_dojo = MockDojo.return_value
        self.mock_dojo.__enter__.return_value = (self.mock_dojo, self.initial_state)
        self.mock_dojo.run_tac = MagicMock()

        # Instantiate the environment
        self.env = LeanDojoEnv(self.theorem, self.theorem_pos)

    def test_initialization(self):
        # Assert that dependencies were called correctly
        self.assertEqual(self.env.current_state, self.initial_state)

    def test_reset(self):
        self.env.reset()
        self.mock_dojo.__enter__.assert_called()

    def test_step_tactic_state(self):
        # Arrange
        action = "test_tactic"
        next_tactic_state = MagicMock(spec=TacticState)
        next_tactic_state.pp = "next_state_pp"
        self.mock_dojo.run_tac.return_value = next_tactic_state

        # Act
        obs, reward, done = self.env.step(action)

        # Assert
        self.mock_dojo.run_tac.assert_called_once_with(self.initial_state, action)
        self.assertEqual(self.env.current_state, next_tactic_state)
        self.assertEqual(obs, "next_state_pp")
        self.assertEqual(reward, 0.1)
        self.assertFalse(done)

    def test_step_proof_finished(self):
        # Arrange
        action = "finish_proof_tactic"
        proof_finished_state = MagicMock(spec=ProofFinished)
        type(proof_finished_state).pp = PropertyMock(return_value="proof_finished")
        self.mock_dojo.run_tac.return_value = proof_finished_state

        # Act
        obs, reward, done = self.env.step(action)

        # Assert
        self.assertEqual(obs, str(proof_finished_state))
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)

    def test_step_lean_error(self):
        # Arrange
        action = "error_tactic"
        lean_error_state = MagicMock(spec=LeanError)
        type(lean_error_state).pp = PropertyMock(return_value="lean_error")
        self.mock_dojo.run_tac.return_value = lean_error_state

        # Act
        obs, reward, done = self.env.step(action)

        # Assert
        self.assertEqual(obs, str(lean_error_state))
        self.assertEqual(reward, -0.1)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
