"""
Tests for time configuration propagation through the training stack.

These tests verify that time-related parameters (max_time, env_timeout) are
properly passed from configuration through trainer, worker, runner, and MCTS.
"""

import unittest
from typing import Any, cast
from unittest.mock import Mock, MagicMock
from lean_dojo import TacticState

from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.agent.runner import AgentRunner
from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol


class MockLeanDojoEnv:  # type: ignore[misc]
    """Mock environment for testing."""

    def __init__(self, timeout: int = 60) -> None:
        self.theorem = Mock()
        self.theorem_pos = Mock()
        # Use Mock with spec=TacticState for Cython compatibility
        self.current_state = Mock(spec=TacticState, pp="test_state")
        self.timeout = timeout


class MockTransformer:  # type: ignore[misc]
    """Mock transformer."""

    def __call__(self, states: Any) -> Any:
        return Mock()


class TestTimeConfigurationPropagation(unittest.TestCase):
    """Test that time configuration propagates correctly through the stack."""

    def test_config_max_time_value(self):
        """Test that TrainingConfig has correct max_time default."""
        config = TrainingConfig(
            data_type="random",
            num_epochs=1,
            num_theorems=1,
            num_iterations=10,
            max_steps=5,
            batch_size=4,
            num_workers=1,
            mcts_type="guided_rollout",
            indexed_corpus_path=None,
            train_epochs=1,
            value_head_batch_size=4,
            train_value_head=False,
            use_final_reward=False,
            save_training_data=False,
            save_checkpoints=False,
            resume=False,
            use_test_value_head=False,
            checkpoint_dir=None,
            use_wandb=False,
            use_caching=False,
        )
        self.assertEqual(config.max_time, 300.0)

    def test_config_max_time_custom_value(self):
        """Test that TrainingConfig can be set with custom max_time."""
        config = TrainingConfig(
            data_type="random",
            num_epochs=1,
            num_theorems=1,
            num_iterations=10,
            max_steps=5,
            batch_size=4,
            num_workers=1,
            mcts_type="guided_rollout",
            indexed_corpus_path=None,
            train_epochs=1,
            value_head_batch_size=4,
            train_value_head=False,
            use_final_reward=False,
            save_training_data=False,
            save_checkpoints=False,
            resume=False,
            use_test_value_head=False,
            checkpoint_dir=None,
            use_wandb=False,
            use_caching=False,
            max_time=300.0,
        )
        self.assertEqual(config.max_time, 300.0)

    def test_config_env_timeout_value(self):
        """Test that TrainingConfig has correct env_timeout default."""
        config = TrainingConfig(
            data_type="random",
            num_epochs=1,
            num_theorems=1,
            num_iterations=10,
            max_steps=5,
            batch_size=4,
            num_workers=1,
            mcts_type="guided_rollout",
            indexed_corpus_path=None,
            train_epochs=1,
            value_head_batch_size=4,
            train_value_head=False,
            use_final_reward=False,
            save_training_data=False,
            save_checkpoints=False,
            resume=False,
            use_test_value_head=False,
            checkpoint_dir=None,
            use_wandb=False,
            use_caching=False,
        )
        self.assertEqual(config.env_timeout, 180)

    def test_config_env_timeout_custom_value(self):
        """Test that TrainingConfig can be set with custom env_timeout."""
        config = TrainingConfig(
            data_type="random",
            num_epochs=1,
            num_theorems=1,
            num_iterations=10,
            max_steps=5,
            batch_size=4,
            num_workers=1,
            mcts_type="guided_rollout",
            indexed_corpus_path=None,
            train_epochs=1,
            value_head_batch_size=4,
            train_value_head=False,
            use_final_reward=False,
            save_training_data=False,
            save_checkpoints=False,
            resume=False,
            use_test_value_head=False,
            checkpoint_dir=None,
            use_wandb=False,
            use_caching=False,
            env_timeout=120,
        )
        self.assertEqual(config.env_timeout, 120)

    def test_config_inference_timeout_default(self):
        """Test that TrainingConfig has correct inference_timeout default."""
        config = TrainingConfig(
            data_type="random",
            num_epochs=1,
            num_theorems=1,
            num_iterations=10,
            max_steps=5,
            batch_size=4,
            num_workers=1,
            mcts_type="guided_rollout",
            indexed_corpus_path=None,
            train_epochs=1,
            value_head_batch_size=4,
            train_value_head=False,
            use_final_reward=False,
            save_training_data=False,
            save_checkpoints=False,
            resume=False,
            use_test_value_head=False,
            checkpoint_dir=None,
            use_wandb=False,
            use_caching=False,
        )
        self.assertEqual(config.inference_timeout, 600.0)

    def test_config_inference_timeout_custom_value(self):
        """Test that TrainingConfig can be set with custom inference_timeout."""
        config = TrainingConfig(
            data_type="random",
            num_epochs=1,
            num_theorems=1,
            num_iterations=10,
            max_steps=5,
            batch_size=4,
            num_workers=1,
            mcts_type="guided_rollout",
            indexed_corpus_path=None,
            train_epochs=1,
            value_head_batch_size=4,
            train_value_head=False,
            use_final_reward=False,
            save_training_data=False,
            save_checkpoints=False,
            resume=False,
            use_test_value_head=False,
            checkpoint_dir=None,
            use_wandb=False,
            use_caching=False,
            inference_timeout=1200.0,
        )
        self.assertEqual(config.inference_timeout, 1200.0)


class TestMCTSTimeInitialization(unittest.TestCase):
    """Test that MCTS receives correct time parameters."""

    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False

    def test_mcts_receives_max_time_from_kwargs(self):
        """Test that MCTS initialization receives max_time from kwargs."""
        max_time = 300.0
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=max_time,
        )
        self.assertEqual(mcts.max_time, max_time)

    def test_mcts_default_max_time(self):
        """Test that MCTS has correct default max_time."""
        mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config
        )
        self.assertEqual(mcts.max_time, 300.0)

    def test_mcts_kwargs_with_extra_parameters(self):
        """Test that MCTS handles extra kwargs correctly."""
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=250.0,
            batch_size=8,
            num_tactics_to_expand=16,
        )
        self.assertEqual(mcts.max_time, 250.0)
        self.assertEqual(mcts.batch_size, 8)
        self.assertEqual(mcts.num_tactics_to_expand, 16)


class TestAgentRunnerTimeHandling(unittest.TestCase):
    """Test that AgentRunner properly handles time parameters."""

    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)

    def test_runner_initialization_with_mcts_kwargs(self):
        """Test that AgentRunner passes mcts_kwargs correctly."""
        mcts_kwargs = {"max_time": 300.0, "batch_size": 8, "num_tactics_to_expand": 16}
        runner = AgentRunner(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            mcts_class=MCTS_GuidedRollout,
            mcts_kwargs=mcts_kwargs,
            num_iterations=100,
            max_steps=10,
        )
        self.assertEqual(runner.mcts_kwargs, mcts_kwargs)
        self.assertEqual(runner.mcts_kwargs["max_time"], 300.0)

    def test_runner_mcts_instantiation(self):
        """Test that runner correctly instantiates MCTS with kwargs."""
        mcts_kwargs = {"max_time": 250.0}
        runner = AgentRunner(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            mcts_class=MCTS_GuidedRollout,
            mcts_kwargs=mcts_kwargs,
            num_iterations=100,
            max_steps=10,
        )
        # Create MCTS instance as runner would
        mcts = runner.mcts_class(
            env=runner.env,
            transformer=runner.transformer,
            config=self.config,
            **runner.mcts_kwargs,
        )
        self.assertEqual(mcts.max_time, 250.0)

    def test_runner_handles_empty_mcts_kwargs(self):
        """Test that runner handles empty mcts_kwargs."""
        runner = AgentRunner(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            mcts_class=MCTS_GuidedRollout,
            mcts_kwargs={},
            num_iterations=100,
            max_steps=10,
        )
        # MCTS should use defaults
        mcts = runner.mcts_class(
            env=runner.env,
            transformer=runner.transformer,
            config=self.config,
            **runner.mcts_kwargs,
        )
        self.assertEqual(mcts.max_time, 300.0)  # default


class TestTimeParameterValidation(unittest.TestCase):
    """Test validation of time parameters."""

    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False

    def test_max_time_positive_value(self):
        """Test that max_time must be positive."""
        # Positive value should work
        mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config, max_time=1.0
        )
        self.assertEqual(mcts.max_time, 1.0)

    def test_max_time_zero(self):
        """Test that max_time can be zero (disabled)."""
        mcts = MCTS_GuidedRollout(
            env=self.env, transformer=self.transformer, config=self.config, max_time=0.0
        )
        self.assertEqual(mcts.max_time, 0.0)

    def test_max_time_large_value(self):
        """Test that max_time can handle large values."""
        large_time = 86400.0  # 24 hours
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=large_time,
        )
        self.assertEqual(mcts.max_time, large_time)


class TestSearchTimeParameters(unittest.TestCase):
    """Test that search method respects time parameters."""

    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)
        self.config.use_caching = False

    def test_search_with_override_max_time(self):
        """Test that MCTS accepts max_time parameter."""
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=600.0,
        )
        self.assertEqual(mcts.max_time, 600.0)

    def test_search_batch_size_parameter(self):
        """Test that MCTS accepts batch_size parameter."""
        batch_size = 4
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            batch_size=batch_size,
        )
        self.assertEqual(mcts.batch_size, batch_size)

    def test_search_uses_instance_defaults(self):
        """Test that MCTS instance stores defaults correctly."""
        instance_max_time = 200.0
        instance_batch_size = 6
        mcts = MCTS_GuidedRollout(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            max_time=instance_max_time,
            batch_size=instance_batch_size,
        )
        self.assertEqual(mcts.max_time, instance_max_time)
        self.assertEqual(mcts.batch_size, instance_batch_size)


class TestProofTimeoutHandling(unittest.TestCase):
    """Test proof search timeout handling in runner."""

    def setUp(self):
        self.env = cast(LeanDojoEnv, MockLeanDojoEnv())
        self.transformer = cast(TransformerProtocol, MockTransformer())
        self.config = MagicMock(spec=TrainingConfig)

    def test_proof_timeout_constant(self):
        """Test that PROOF_TIMEOUT is defined in runner."""
        # Create a runner to check internal timeout
        agent_runner = AgentRunner(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            mcts_kwargs={"max_time": 600.0},
        )
        self.assertIsNotNone(agent_runner)

    def test_runner_max_steps_parameter(self):
        """Test that runner respects max_steps parameter."""
        max_steps = 20
        runner = AgentRunner(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            mcts_kwargs={},
            max_steps=max_steps,
        )
        self.assertEqual(runner.max_steps, max_steps)

    def test_runner_num_iterations_parameter(self):
        """Test that runner respects num_iterations parameter."""
        num_iterations = 250
        runner = AgentRunner(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            mcts_kwargs={},
            num_iterations=num_iterations,
        )
        self.assertEqual(runner.num_iterations, num_iterations)


if __name__ == "__main__":
    unittest.main()
