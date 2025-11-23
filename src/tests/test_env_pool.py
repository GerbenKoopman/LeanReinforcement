import unittest
from unittest.mock import MagicMock, patch
from src.utilities.gym import LeanDojoEnvPool


class TestEnvPool(unittest.TestCase):
    def setUp(self):
        self.mock_env_patcher = patch("src.utilities.gym.LeanDojoEnv")
        self.mock_env_class = self.mock_env_patcher.start()

    def tearDown(self):
        self.mock_env_patcher.stop()

    def test_pool_initialization(self):
        corpus = MagicMock()
        theorem = MagicMock()
        theorem_pos = MagicMock()

        pool = LeanDojoEnvPool(corpus, theorem, theorem_pos, num_workers=3)

        self.assertEqual(len(pool.envs), 3)
        self.assertEqual(pool.pool.qsize(), 3)
        self.assertEqual(self.mock_env_class.call_count, 3)

    def test_get_env_context_manager(self):
        corpus = MagicMock()
        theorem = MagicMock()
        theorem_pos = MagicMock()

        pool = LeanDojoEnvPool(corpus, theorem, theorem_pos, num_workers=1)

        self.assertEqual(pool.pool.qsize(), 1)

        with pool.get_env() as env:
            self.assertIsNotNone(env)
            self.assertEqual(pool.pool.qsize(), 0)

        self.assertEqual(pool.pool.qsize(), 1)

    def test_pool_close(self):
        corpus = MagicMock()
        theorem = MagicMock()
        theorem_pos = MagicMock()

        # Ensure LeanDojoEnv returns distinct instances
        self.mock_env_class.side_effect = lambda *args, **kwargs: MagicMock()

        pool = LeanDojoEnvPool(corpus, theorem, theorem_pos, num_workers=2)
        pool.close()

        for env in pool.envs:
            env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
