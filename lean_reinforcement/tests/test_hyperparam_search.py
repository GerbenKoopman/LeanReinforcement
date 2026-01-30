"""
Tests for hyperparameter search module.

These tests verify the hyperparameter search functionality without running
actual proof attempts (which would be too slow for unit tests).
"""

import unittest
import tempfile

from lean_reinforcement.training.hyperparam_search import (
    HyperparameterConfig,
    TrialResult,
    HyperparameterSearcher,
    LAPTOP_DEFAULTS,
    HPC_DEFAULTS,
    LAPTOP_SEARCH_SPACE,
    HPC_SEARCH_SPACE,
)


class TestHyperparameterConfig(unittest.TestCase):
    """Tests for HyperparameterConfig dataclass."""

    def test_defaults(self):
        """Test that default values are set correctly."""
        config = HyperparameterConfig()

        self.assertEqual(config.num_workers, 10)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.num_tactics_to_expand, 12)
        self.assertEqual(config.num_iterations, 100)
        self.assertEqual(config.mcts_type, "guided_rollout")

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = HyperparameterConfig(
            num_workers=8,
            batch_size=32,
            num_iterations=200,
        )

        self.assertEqual(config.num_workers, 8)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_iterations, 200)

    def test_to_args_dict(self):
        """Test conversion to args dictionary."""
        config = HyperparameterConfig(num_workers=12)
        args_dict = config.to_args_dict()

        self.assertIsInstance(args_dict, dict)
        self.assertEqual(args_dict["num_workers"], 12)
        self.assertIn("inference_timeout", args_dict)
        self.assertIn("indexed_corpus_path", args_dict)

    def test_to_args_dict_complete(self):
        """Test that to_args_dict includes all required fields."""
        config = HyperparameterConfig()
        args_dict = config.to_args_dict()

        required_fields = [
            "num_workers",
            "batch_size",
            "num_tactics_to_expand",
            "num_iterations",
            "max_time",
            "max_steps",
            "proof_timeout",
            "env_timeout",
            "max_rollout_depth",
            "mcts_type",
            "model_name",
            "data_type",
            "num_epochs",
            "num_theorems",
            "train_epochs",
            "train_value_head",
            "use_final_reward",
            "save_training_data",
            "save_checkpoints",
            "use_wandb",
            "indexed_corpus_path",
            "resume",
            "use_test_value_head",
            "checkpoint_dir",
            "inference_timeout",
        ]

        for field in required_fields:
            self.assertIn(field, args_dict, f"Missing field: {field}")


class TestTrialResult(unittest.TestCase):
    """Tests for TrialResult dataclass."""

    def test_score_calculation(self):
        """Test that score is calculated correctly."""
        config = HyperparameterConfig()
        result = TrialResult(
            config=config,
            total_time=100.0,
            num_proofs_attempted=10,
            num_proofs_succeeded=8,
            proofs_per_second=0.08,
            success_rate=0.8,
            avg_proof_time=12.5,
            avg_steps_per_proof=5.0,
        )

        # Score is now proofs_per_hour, which is proofs_per_second * 3600
        expected_score = 0.08 * 3600.0
        self.assertAlmostEqual(result.score, expected_score, places=6)

    def test_score_with_zero_success(self):
        """Test score calculation with zero success rate."""
        config = HyperparameterConfig()
        result = TrialResult(
            config=config,
            total_time=100.0,
            num_proofs_attempted=10,
            num_proofs_succeeded=0,
            proofs_per_second=0.0,
            success_rate=0.0,
            avg_proof_time=10.0,
            avg_steps_per_proof=0.0,
        )

        self.assertEqual(result.score, 0.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = HyperparameterConfig()
        result = TrialResult(
            config=config,
            total_time=100.0,
            num_proofs_attempted=10,
            num_proofs_succeeded=8,
            proofs_per_second=0.08,
            success_rate=0.8,
            avg_proof_time=12.5,
            avg_steps_per_proof=5.0,
        )

        data = result.to_dict()

        self.assertIsInstance(data, dict)
        self.assertIn("config", data)
        self.assertIn("score", data)
        self.assertEqual(data["num_proofs_succeeded"], 8)


class TestHardwareProfiles(unittest.TestCase):
    """Tests for hardware profile configurations."""

    def test_laptop_defaults(self):
        """Test laptop profile has reasonable defaults."""
        self.assertEqual(LAPTOP_DEFAULTS.num_workers, 10)
        self.assertEqual(LAPTOP_DEFAULTS.batch_size, 16)
        self.assertLessEqual(LAPTOP_DEFAULTS.num_workers, 16)  # Laptop constraint

    def test_hpc_defaults(self):
        """Test HPC profile has reasonable defaults."""
        self.assertEqual(HPC_DEFAULTS.num_workers, 32)
        self.assertEqual(HPC_DEFAULTS.batch_size, 32)
        self.assertGreater(HPC_DEFAULTS.num_iterations, LAPTOP_DEFAULTS.num_iterations)

    def test_hpc_higher_than_laptop(self):
        """Test that HPC profile uses more resources than laptop."""
        self.assertGreater(HPC_DEFAULTS.num_workers, LAPTOP_DEFAULTS.num_workers)
        self.assertGreaterEqual(HPC_DEFAULTS.batch_size, LAPTOP_DEFAULTS.batch_size)
        self.assertGreater(
            HPC_DEFAULTS.num_tactics_to_expand,
            LAPTOP_DEFAULTS.num_tactics_to_expand,
        )

    def test_search_spaces_valid(self):
        """Test that search spaces contain valid values."""
        for space in [LAPTOP_SEARCH_SPACE, HPC_SEARCH_SPACE]:
            self.assertIn("batch_size", space)

            # All values should be positive
            for key, values in space.items():
                self.assertTrue(all(v > 0 for v in values))


class TestHyperparameterSearcher(unittest.TestCase):
    """Tests for HyperparameterSearcher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_laptop(self):
        """Test searcher initialization with laptop profile."""
        searcher = HyperparameterSearcher(
            hardware_profile="laptop",
            output_dir=self.temp_dir,
            use_wandb=False,
        )

        self.assertEqual(searcher.hardware_profile, "laptop")
        self.assertEqual(searcher.default_config, LAPTOP_DEFAULTS)
        self.assertEqual(searcher.search_space, LAPTOP_SEARCH_SPACE)

    def test_initialization_hpc(self):
        """Test searcher initialization with HPC profile."""
        searcher = HyperparameterSearcher(
            hardware_profile="hpc",
            output_dir=self.temp_dir,
            use_wandb=False,
        )

        self.assertEqual(searcher.hardware_profile, "hpc")
        self.assertEqual(searcher.default_config, HPC_DEFAULTS)
        self.assertEqual(searcher.search_space, HPC_SEARCH_SPACE)

    def test_save_and_load_results(self):
        """Test saving and loading results."""
        searcher = HyperparameterSearcher(
            hardware_profile="laptop",
            output_dir=self.temp_dir,
            use_wandb=False,
        )

        # Create mock results
        config = HyperparameterConfig()
        result = TrialResult(
            config=config,
            total_time=100.0,
            num_proofs_attempted=10,
            num_proofs_succeeded=8,
            proofs_per_second=0.08,
            success_rate=0.8,
            avg_proof_time=12.5,
            avg_steps_per_proof=5.0,
        )

        results = [result]
        searcher._save_results(results, "test_results.json")

        # Load results back
        loaded = searcher.load_results("test_results.json")

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].num_proofs_succeeded, 8)
        self.assertEqual(loaded[0].config.num_workers, config.num_workers)

    def test_generate_config_for_hpc(self):
        """Test HPC config generation from laptop config."""
        searcher = HyperparameterSearcher(
            hardware_profile="laptop",
            output_dir=self.temp_dir,
            use_wandb=False,
        )

        laptop_config = HyperparameterConfig(
            num_workers=10,
            batch_size=16,
            num_iterations=100,
        )

        hpc_config = searcher.generate_config_for_hpc(laptop_config)

        # HPC should scale up
        self.assertGreater(hpc_config["num_workers"], laptop_config.num_workers)
        self.assertGreater(hpc_config["batch_size"], laptop_config.batch_size)
        self.assertGreater(hpc_config["num_iterations"], laptop_config.num_iterations)


class TestGridSearchCombinations(unittest.TestCase):
    """Tests for grid search combination generation."""

    def test_grid_combinations_count(self):
        """Test that grid search generates correct number of combinations."""
        import itertools

        space = {
            "a": [1, 2],
            "b": [10, 20, 30],
        }

        combinations = list(itertools.product(*space.values()))
        expected_count = 2 * 3  # |a| * |b|

        self.assertEqual(len(combinations), expected_count)

    def test_search_space_combinations(self):
        """Test laptop search space combination count."""
        import itertools

        param_values = list(LAPTOP_SEARCH_SPACE.values())
        all_combinations = list(itertools.product(*param_values))

        # Should have reasonable number of combinations for grid search
        self.assertGreater(len(all_combinations), 0)
        self.assertLess(len(all_combinations), 3000)  # Not too many


class TestBinarySearchLogic(unittest.TestCase):
    """Tests for binary search logic."""

    def test_midpoint_calculation(self):
        """Test midpoint calculation in binary search."""
        low, high = 4, 16
        mid = (low + high) / 2

        self.assertEqual(mid, 10)

    def test_convergence_detection(self):
        """Test convergence detection based on tolerance."""
        tolerance = 0.1
        max_val = 16

        # Should converge when range is small enough
        low, high = 9.5, 10.5
        range_size = high - low

        should_converge = (range_size / max_val) < tolerance
        self.assertTrue(should_converge)


if __name__ == "__main__":
    unittest.main()
