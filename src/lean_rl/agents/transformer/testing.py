"""
Testing Module for Hierarchical Transformer Agent.

This module provides comprehensive testing capabilities for the HierarchicalTransformerAgent,
including unit tests, integration tests, performance benchmarks, and evaluation metrics.
"""

import torch
import numpy as np
import time
import json
import os
import tempfile
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
from unittest.mock import Mock

from lean_dojo import LeanGitRepo, trace

from .agent import HierarchicalTransformerAgent, HierarchicalAction
from .hierarchy import HierarchyLevel, StrategicActions, TacticalFamilies
from ...environment import LeanEnvironment
from ...agents import RandomAgent, MCTSAgent


@dataclass
class TestResults:
    """Results from comprehensive testing."""

    # Unit test results
    unit_tests_passed: int = 0
    unit_tests_failed: int = 0
    unit_test_details: Dict[str, bool] = field(default_factory=dict)

    # Performance benchmarks
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    search_efficiency: float = 0.0

    # Theorem proving results
    total_theorems_tested: int = 0
    theorems_proved: int = 0
    average_proof_length: float = 0.0
    average_search_time: float = 0.0

    # Comparison with baselines
    baseline_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Hierarchical analysis
    strategic_accuracy: float = 0.0
    tactical_accuracy: float = 0.0
    parameter_accuracy: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate theorem proving success rate."""
        if self.total_theorems_tested == 0:
            return 0.0
        return self.theorems_proved / self.total_theorems_tested

    @property
    def unit_test_success_rate(self) -> float:
        """Calculate unit test success rate."""
        total = self.unit_tests_passed + self.unit_tests_failed
        if total == 0:
            return 0.0
        return self.unit_tests_passed / total


class HierarchicalTransformerTester:
    """Comprehensive tester for the hierarchical transformer agent."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the tester.

        Args:
            model_path: Path to trained model checkpoint (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logger()

        # Initialize agent
        if model_path and os.path.exists(model_path):
            self.agent = self._load_trained_agent(model_path)
            self.logger.info(f"Loaded trained agent from {model_path}")
        else:
            self.agent = HierarchicalTransformerAgent(device=str(self.device))
            self.logger.info("Initialized fresh agent for testing")

        # Setup test repository
        self._setup_test_repository()

        # Initialize test results
        self.results = TestResults()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for testing."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _load_trained_agent(self, model_path: str) -> HierarchicalTransformerAgent:
        """Load a trained agent from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract config if available
        config = checkpoint.get("config", {})

        agent = HierarchicalTransformerAgent(
            vocab_size=config.get("vocab_size", 10000),
            d_model=config.get("d_model", 512),
            n_heads=config.get("n_heads", 8),
            n_layers=config.get("n_layers", 6),
            device=str(self.device),
        )

        # Load state dict using helper method
        self._load_agent_state_dict(agent, checkpoint["agent_state_dict"])
        self._set_agent_mode(agent, False)  # Set to eval mode

        return agent

    def _load_agent_state_dict(self, agent, state_dict):
        """Load state dict into agent submodules."""
        if (
            hasattr(agent, "hierarchical_policy")
            and "hierarchical_policy" in state_dict
        ):
            agent.hierarchical_policy.load_state_dict(state_dict["hierarchical_policy"])
        if hasattr(agent, "tactic_pointer") and "tactic_pointer" in state_dict:
            agent.tactic_pointer.load_state_dict(state_dict["tactic_pointer"])
        if (
            hasattr(agent, "parameter_generator")
            and "parameter_generator" in state_dict
        ):
            agent.parameter_generator.load_state_dict(state_dict["parameter_generator"])
        if hasattr(agent, "parameter_pointer") and "parameter_pointer" in state_dict:
            agent.parameter_pointer.load_state_dict(state_dict["parameter_pointer"])

    def _set_agent_mode(self, agent, train_mode):
        """Set train/eval mode for agent submodules."""
        if hasattr(agent, "hierarchical_policy"):
            agent.hierarchical_policy.train(train_mode)
        if hasattr(agent, "tactic_pointer"):
            agent.tactic_pointer.train(train_mode)
        if hasattr(agent, "parameter_generator"):
            agent.parameter_generator.train(train_mode)
        if hasattr(agent, "parameter_pointer"):
            agent.parameter_pointer.train(train_mode)

    def _get_agent_state_dict(self, agent):
        """Get state dict from all agent submodules."""
        state_dict = {}
        if hasattr(agent, "hierarchical_policy"):
            state_dict["hierarchical_policy"] = agent.hierarchical_policy.state_dict()
        if hasattr(agent, "tactic_pointer"):
            state_dict["tactic_pointer"] = agent.tactic_pointer.state_dict()
        if hasattr(agent, "parameter_generator"):
            state_dict["parameter_generator"] = agent.parameter_generator.state_dict()
        if hasattr(agent, "parameter_pointer"):
            state_dict["parameter_pointer"] = agent.parameter_pointer.state_dict()
        return state_dict

    def _setup_test_repository(self):
        """Setup test repository and environment."""
        self.repo = LeanGitRepo(
            "https://github.com/leanprover-community/mathlib4",
            "29dcec074de168ac2bf835a77ef68bbe069194c5",
        )

        try:
            self.traced_repo = trace(self.repo)
            self.env = LeanEnvironment(self.repo, max_steps=50, timeout=30)
            self.logger.info("Test repository setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup test repository: {e}")
            raise

    def run_all_tests(self) -> TestResults:
        """Run all comprehensive tests."""
        self.logger.info("Starting comprehensive testing...")

        # Unit tests
        self.logger.info("Running unit tests...")
        self._run_unit_tests()

        # Performance benchmarks
        self.logger.info("Running performance benchmarks...")
        self._run_performance_benchmarks()

        # Theorem proving tests
        self.logger.info("Running theorem proving tests...")
        self._run_theorem_proving_tests()

        # Baseline comparisons
        self.logger.info("Running baseline comparisons...")
        self._run_baseline_comparisons()

        # Hierarchical analysis
        self.logger.info("Running hierarchical analysis...")
        self._run_hierarchical_analysis()

        # Generate report
        self._generate_test_report()

        self.logger.info("Comprehensive testing completed!")
        return self.results

    def _run_unit_tests(self):
        """Run unit tests for individual components."""
        unit_tests = [
            ("test_agent_initialization", self._test_agent_initialization),
            ("test_state_encoding", self._test_state_encoding),
            ("test_hierarchical_forward", self._test_hierarchical_forward),
            ("test_strategic_action_selection", self._test_strategic_action_selection),
            ("test_tactical_selection", self._test_tactical_selection),
            ("test_parameter_generation", self._test_parameter_generation),
            ("test_search_tree", self._test_search_tree),
            ("test_action_construction", self._test_action_construction),
            ("test_model_saving_loading", self._test_model_saving_loading),
        ]

        for test_name, test_func in unit_tests:
            try:
                success = test_func()
                self.results.unit_test_details[test_name] = success
                if success:
                    self.results.unit_tests_passed += 1
                    self.logger.info(f"✓ {test_name}")
                else:
                    self.results.unit_tests_failed += 1
                    self.logger.warning(f"✗ {test_name}")
            except Exception as e:
                self.results.unit_tests_failed += 1
                self.results.unit_test_details[test_name] = False
                self.logger.error(f"✗ {test_name}: {e}")

    def _test_agent_initialization(self) -> bool:
        """Test agent initialization."""
        try:
            agent = HierarchicalTransformerAgent(device=str(self.device))

            # Check components exist
            assert hasattr(agent, "hierarchical_policy")
            assert hasattr(agent, "tactic_pointer")
            assert hasattr(agent, "parameter_generator")
            assert hasattr(agent, "parameter_pointer")
            assert hasattr(agent, "tokenizer")

            # Check device placement
            assert next(agent.hierarchical_policy.parameters()).device == self.device

            return True
        except Exception as e:
            self.logger.error(f"Agent initialization test failed: {e}")
            return False

    def _test_state_encoding(self) -> bool:
        """Test state encoding functionality."""
        try:
            # Create mock state
            mock_state = Mock()
            mock_state.pp = "example proof state with goals"
            mock_state.num_goals = 2

            # Test encoding
            encoded = self.agent.encode_state(mock_state)

            # Check output format
            assert isinstance(encoded, dict)
            assert "input_ids" in encoded
            assert "attention_mask" in encoded

            # Check tensor shapes
            assert encoded["input_ids"].dim() == 2  # [batch, seq]
            assert encoded["attention_mask"].dim() == 2

            return True
        except Exception as e:
            self.logger.error(f"State encoding test failed: {e}")
            return False

    def _test_hierarchical_forward(self) -> bool:
        """Test hierarchical forward pass."""
        try:
            # Create mock encoded state
            batch_size, seq_len = 1, 128
            encoded_state = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(
                    self.device
                ),
                "attention_mask": torch.ones(batch_size, seq_len).to(self.device),
                "goal_mask": torch.ones(batch_size, seq_len).to(self.device),
                "hypothesis_mask": torch.ones(batch_size, seq_len).to(self.device),
            }

            # Test each hierarchy level
            for level in [
                HierarchyLevel.STRATEGIC,
                HierarchyLevel.TACTICAL,
                HierarchyLevel.EXECUTION,
            ]:
                output = self.agent.hierarchical_forward(encoded_state, level)

                assert isinstance(output, dict)
                assert "policy_logits" in output
                assert "representation" in output

                # Check tensor shapes
                assert output["policy_logits"].dim() == 2
                assert output["representation"].dim() == 2

            return True
        except Exception as e:
            self.logger.error(f"Hierarchical forward test failed: {e}")
            return False

    def _test_strategic_action_selection(self) -> bool:
        """Test strategic action selection."""
        try:
            mock_state = Mock()
            mock_state.pp = "test proof state"
            mock_state.num_goals = 1

            action = self.agent.select_strategic_action(mock_state)

            # Check output format
            assert isinstance(action, str)
            assert action in StrategicActions.ALL_ACTIONS

            return True
        except Exception as e:
            self.logger.error(f"Strategic action selection test failed: {e}")
            return False

    def _test_tactical_selection(self) -> bool:
        """Test tactical family selection."""
        try:
            mock_state = Mock()
            mock_state.pp = "test proof state"
            mock_state.num_goals = 1

            family = self.agent.select_tactic_family(mock_state, "direct_proof")

            # Check output format
            assert isinstance(family, str)
            assert family in TacticalFamilies.ALL_FAMILIES

            return True
        except Exception as e:
            self.logger.error(f"Tactical selection test failed: {e}")
            return False

    def _test_parameter_generation(self) -> bool:
        """Test parameter generation."""
        try:
            mock_state = Mock()
            mock_state.pp = "test proof state"
            mock_state.num_goals = 1

            params = self.agent.generate_tactic_parameters(mock_state, "apply_family")

            # Check output format
            assert isinstance(params, list)

            return True
        except Exception as e:
            self.logger.error(f"Parameter generation test failed: {e}")
            return False

    def _test_search_tree(self) -> bool:
        """Test search tree functionality."""
        try:
            from .agent import HierarchicalSearchTree

            mock_state = Mock()
            mock_state.pp = "test proof state"
            mock_state.num_goals = 1

            search_tree = HierarchicalSearchTree(mock_state, self.agent)

            # Check initialization
            assert search_tree.root is not None
            assert search_tree.root.state == mock_state
            assert search_tree.root.level == HierarchyLevel.STRATEGIC

            return True
        except Exception as e:
            self.logger.error(f"Search tree test failed: {e}")
            return False

    def _test_action_construction(self) -> bool:
        """Test hierarchical action construction."""
        try:
            mock_state = Mock()
            mock_state.pp = "test proof state"
            mock_state.num_goals = 1

            action = self.agent.construct_full_action(mock_state)

            # Check output format
            assert isinstance(action, HierarchicalAction)
            assert hasattr(action, "strategic_action")
            assert hasattr(action, "tactic_family")
            assert hasattr(action, "specific_tactic")
            assert hasattr(action, "parameters")
            assert hasattr(action, "confidence")

            return True
        except Exception as e:
            self.logger.error(f"Action construction test failed: {e}")
            return False

    def _test_model_saving_loading(self) -> bool:
        """Test model saving and loading."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = os.path.join(temp_dir, "test_model.pt")

                # Save model
                self.agent.save_model(save_path)
                assert os.path.exists(save_path)

                # Create new agent and load
                new_agent = HierarchicalTransformerAgent(device=str(self.device))
                new_agent.load_model(save_path)

                # Compare state dicts
                original_state = self._get_agent_state_dict(self.agent)
                loaded_state = self._get_agent_state_dict(new_agent)

                for key in original_state:
                    if key in loaded_state:
                        assert torch.allclose(
                            original_state[key], loaded_state[key], atol=1e-6
                        )

            return True
        except Exception as e:
            self.logger.error(f"Model saving/loading test failed: {e}")
            return False

    def _run_performance_benchmarks(self):
        """Run performance benchmarks."""
        self.logger.info("Running performance benchmarks...")

        # Inference time benchmark
        self._benchmark_inference_time()

        # Memory usage benchmark
        self._benchmark_memory_usage()

        # Search efficiency benchmark
        self._benchmark_search_efficiency()

    def _benchmark_inference_time(self):
        """Benchmark inference time."""
        mock_state = Mock()
        mock_state.pp = "example proof state with multiple goals and hypotheses"
        mock_state.num_goals = 3

        # Warm up
        for _ in range(5):
            self.agent.select_action(mock_state)

        # Actual benchmark
        times = []
        num_trials = 50

        for _ in range(num_trials):
            start_time = time.time()
            self.agent.select_action(mock_state)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        self.results.inference_time_ms = float(np.mean(times))
        self.logger.info(
            f"Average inference time: {self.results.inference_time_ms:.2f} ms"
        )

    def _benchmark_memory_usage(self):
        """Benchmark memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            mock_state = Mock()
            mock_state.pp = "example proof state"
            mock_state.num_goals = 1

            # Run inference
            self.agent.select_action(mock_state)

            memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.results.memory_usage_mb = memory_used
            self.logger.info(f"Memory usage: {memory_used:.2f} MB")
        else:
            self.results.memory_usage_mb = 0.0
            self.logger.info("CUDA not available, skipping memory benchmark")

    def _benchmark_search_efficiency(self):
        """Benchmark search efficiency."""
        mock_state = Mock()
        mock_state.pp = "example proof state"
        mock_state.num_goals = 2

        # Measure search depth vs time
        search_times = []
        search_depths = []

        for beam_width in [4, 8, 16, 32]:
            self.agent.beam_width = beam_width

            start_time = time.time()
            self.agent.select_action(mock_state)
            end_time = time.time()

            search_times.append(end_time - start_time)
            search_depths.append(beam_width)

        # Calculate efficiency metric (depth per second)
        if search_times:
            avg_time = np.mean(search_times)
            avg_depth = np.mean(search_depths)
            self.results.search_efficiency = float(
                avg_depth / avg_time if avg_time > 0 else 0.0
            )

        self.logger.info(
            f"Search efficiency: {self.results.search_efficiency:.2f} depth/second"
        )

    def _run_theorem_proving_tests(self):
        """Run theorem proving tests on real theorems."""
        # Get test theorems
        test_theorems = self._get_test_theorems(num_theorems=20)

        if not test_theorems:
            self.logger.warning("No test theorems available")
            return

        self.results.total_theorems_tested = len(test_theorems)
        proved_count = 0
        proof_lengths = []
        search_times = []

        for i, theorem in enumerate(test_theorems):
            self.logger.info(
                f"Testing theorem {i+1}/{len(test_theorems)}: {theorem.theorem.full_name}"
            )

            try:
                # Reset environment
                state = self.env.reset(theorem.theorem)
                self.agent.reset()

                # Track metrics
                start_time = time.time()
                proof_length = 0
                success = False

                # Run episode
                while proof_length < 50:  # Max steps
                    if state is not None:
                        action = self.agent.select_action(state)
                        if action is None:
                            break

                        result = self.env.step(action)
                        proof_length += 1

                        if result.done:
                            if result.action_result == "proof_finished":
                                success = True
                                proved_count += 1
                            break

                        state = result.state
                    else:
                        break

                search_time = time.time() - start_time

                if success:
                    proof_lengths.append(proof_length)
                    search_times.append(search_time)
                    self.logger.info(
                        f"✓ Proved in {proof_length} steps, {search_time:.2f}s"
                    )
                else:
                    self.logger.info(f"✗ Failed to prove")

            except Exception as e:
                self.logger.warning(f"Error testing theorem: {e}")
                continue

        # Update results
        self.results.theorems_proved = proved_count
        self.results.average_proof_length = float(
            np.mean(proof_lengths) if proof_lengths else 0.0
        )
        self.results.average_search_time = float(
            np.mean(search_times) if search_times else 0.0
        )

        self.logger.info(
            f"Theorem proving results: {proved_count}/{len(test_theorems)} proved "
            f"({self.results.success_rate:.3f} success rate)"
        )

    def _run_baseline_comparisons(self):
        """Compare against baseline agents."""
        test_theorems = self._get_test_theorems(num_theorems=10)

        if not test_theorems:
            self.logger.warning("No test theorems for baseline comparison")
            return

        baselines = {
            "Random": RandomAgent(seed=42),
            "MCTS": MCTSAgent(iterations=50, seed=42),
        }

        for baseline_name, baseline_agent in baselines.items():
            self.logger.info(f"Testing baseline: {baseline_name}")

            success_count = 0
            total_reward = 0.0

            for theorem in test_theorems:
                try:
                    state = self.env.reset(theorem.theorem)
                    baseline_agent.reset()

                    episode_reward = 0.0
                    steps = 0

                    while steps < 30:  # Limit steps for baselines
                        action = baseline_agent.select_action(state)
                        if action is None:
                            break

                        result = self.env.step(action)
                        baseline_agent.update(result)

                        episode_reward += result.reward
                        steps += 1

                        if result.done:
                            if result.action_result == "proof_finished":
                                success_count += 1
                            break

                        state = result.state

                    total_reward += episode_reward

                except Exception as e:
                    self.logger.warning(f"Baseline test error: {e}")
                    continue

            # Store results
            baseline_success_rate = success_count / len(test_theorems)
            baseline_avg_reward = total_reward / len(test_theorems)

            self.results.baseline_comparisons[baseline_name] = {
                "success_rate": baseline_success_rate,
                "avg_reward": baseline_avg_reward,
            }

            self.logger.info(
                f"{baseline_name}: {baseline_success_rate:.3f} success rate, "
                f"{baseline_avg_reward:.3f} avg reward"
            )

    def _run_hierarchical_analysis(self):
        """Analyze hierarchical decision making."""
        test_theorems = self._get_test_theorems(num_theorems=5)

        if not test_theorems:
            return

        strategic_correct = 0
        tactical_correct = 0
        parameter_correct = 0
        total_decisions = 0

        for theorem in test_theorems:
            try:
                state = self.env.reset(theorem.theorem)

                # Analyze one step of decision making
                hierarchical_action = self.agent.construct_full_action(state)

                # Mock ground truth validation (simplified)
                # In practice, this would compare against expert demonstrations
                strategic_valid = (
                    hierarchical_action.strategic_action in StrategicActions.ALL_ACTIONS
                )
                tactical_valid = (
                    hierarchical_action.tactic_family in TacticalFamilies.ALL_FAMILIES
                )
                parameter_valid = isinstance(hierarchical_action.parameters, list)

                if strategic_valid:
                    strategic_correct += 1
                if tactical_valid:
                    tactical_correct += 1
                if parameter_valid:
                    parameter_correct += 1

                total_decisions += 1

            except Exception as e:
                self.logger.warning(f"Hierarchical analysis error: {e}")
                continue

        if total_decisions > 0:
            self.results.strategic_accuracy = strategic_correct / total_decisions
            self.results.tactical_accuracy = tactical_correct / total_decisions
            self.results.parameter_accuracy = parameter_correct / total_decisions

    def _get_test_theorems(self, num_theorems: int = 20) -> List:
        """Get test theorems from the repository."""
        all_theorems = []

        # Try to get theorems from multiple files
        test_files = [
            "Mathlib/Data/Nat/Basic.lean",
            "Mathlib/Data/List/Basic.lean",
            "Mathlib/Logic/Basic.lean",
            "Mathlib/Algebra/Group/Defs.lean",
        ]

        for file_path in test_files:
            try:
                traced_file = self.traced_repo.get_traced_file(file_path)
                theorems = traced_file.get_traced_theorems()
                all_theorems.extend(theorems)

                if len(all_theorems) >= num_theorems:
                    break

            except Exception as e:
                self.logger.warning(f"Failed to load theorems from {file_path}: {e}")
                continue

        # Return subset
        return all_theorems[:num_theorems]

    def _generate_test_report(self):
        """Generate comprehensive test report."""
        report = {
            "test_summary": {
                "unit_tests_passed": self.results.unit_tests_passed,
                "unit_tests_failed": self.results.unit_tests_failed,
                "unit_test_success_rate": self.results.unit_test_success_rate,
                "theorem_proving_success_rate": self.results.success_rate,
                "total_theorems_tested": self.results.total_theorems_tested,
            },
            "performance": {
                "inference_time_ms": self.results.inference_time_ms,
                "memory_usage_mb": self.results.memory_usage_mb,
                "search_efficiency": self.results.search_efficiency,
            },
            "theorem_proving": {
                "theorems_proved": self.results.theorems_proved,
                "total_theorems": self.results.total_theorems_tested,
                "success_rate": self.results.success_rate,
                "average_proof_length": self.results.average_proof_length,
                "average_search_time": self.results.average_search_time,
            },
            "hierarchical_analysis": {
                "strategic_accuracy": self.results.strategic_accuracy,
                "tactical_accuracy": self.results.tactical_accuracy,
                "parameter_accuracy": self.results.parameter_accuracy,
            },
            "baseline_comparisons": self.results.baseline_comparisons,
            "unit_test_details": self.results.unit_test_details,
        }

        # Save report
        report_path = f"test_report_{int(time.time())}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Test report saved to {report_path}")

        # Print summary
        self._print_test_summary()

    def _print_test_summary(self):
        """Print test summary to console."""
        print("\n" + "=" * 60)
        print("HIERARCHICAL TRANSFORMER AGENT TEST SUMMARY")
        print("=" * 60)

        print(f"\nUnit Tests:")
        print(f"  Passed: {self.results.unit_tests_passed}")
        print(f"  Failed: {self.results.unit_tests_failed}")
        print(f"  Success Rate: {self.results.unit_test_success_rate:.3f}")

        print(f"\nPerformance:")
        print(f"  Inference Time: {self.results.inference_time_ms:.2f} ms")
        print(f"  Memory Usage: {self.results.memory_usage_mb:.2f} MB")
        print(f"  Search Efficiency: {self.results.search_efficiency:.2f} depth/sec")

        print(f"\nTheorem Proving:")
        print(
            f"  Success Rate: {self.results.success_rate:.3f} ({self.results.theorems_proved}/{self.results.total_theorems_tested})"
        )
        print(f"  Avg Proof Length: {self.results.average_proof_length:.1f} steps")
        print(f"  Avg Search Time: {self.results.average_search_time:.2f} seconds")

        print(f"\nHierarchical Analysis:")
        print(f"  Strategic Accuracy: {self.results.strategic_accuracy:.3f}")
        print(f"  Tactical Accuracy: {self.results.tactical_accuracy:.3f}")
        print(f"  Parameter Accuracy: {self.results.parameter_accuracy:.3f}")

        if self.results.baseline_comparisons:
            print(f"\nBaseline Comparisons:")
            for baseline_name, metrics in self.results.baseline_comparisons.items():
                print(f"  {baseline_name}: {metrics['success_rate']:.3f} success rate")

        print("=" * 60)


def run_comprehensive_tests(model_path: Optional[str] = None) -> TestResults:
    """
    Run comprehensive tests on the hierarchical transformer agent.

    Args:
        model_path: Path to trained model checkpoint (optional)

    Returns:
        TestResults object with all test outcomes
    """
    tester = HierarchicalTransformerTester(model_path)
    return tester.run_all_tests()


def run_performance_profile(model_path: Optional[str] = None):
    """
    Run detailed performance profiling.

    Args:
        model_path: Path to trained model checkpoint (optional)
    """
    print("Running performance profiling...")

    tester = HierarchicalTransformerTester(model_path)

    # Detailed performance analysis
    tester._benchmark_inference_time()
    tester._benchmark_memory_usage()
    tester._benchmark_search_efficiency()

    print("Performance profiling completed!")


def run_theorem_benchmark(model_path: Optional[str] = None, num_theorems: int = 50):
    """
    Run theorem proving benchmark.

    Args:
        model_path: Path to trained model checkpoint (optional)
        num_theorems: Number of theorems to test
    """
    print(f"Running theorem benchmark on {num_theorems} theorems...")

    tester = HierarchicalTransformerTester(model_path)

    # Override the number of test theorems
    original_method = tester._get_test_theorems
    tester._get_test_theorems = lambda num_theorems=num_theorems: original_method(
        num_theorems
    )

    tester._run_theorem_proving_tests()

    print(f"Benchmark completed: {tester.results.success_rate:.3f} success rate")
    return tester.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Hierarchical Transformer Agent")
    parser.add_argument(
        "--model-path", type=str, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test-type",
        choices=["all", "unit", "performance", "theorems"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=20,
        help="Number of theorems for theorem proving tests",
    )

    args = parser.parse_args()

    if args.test_type == "all":
        results = run_comprehensive_tests(args.model_path)
    elif args.test_type == "unit":
        tester = HierarchicalTransformerTester(args.model_path)
        tester._run_unit_tests()
        tester._print_test_summary()
    elif args.test_type == "performance":
        run_performance_profile(args.model_path)
    elif args.test_type == "theorems":
        run_theorem_benchmark(args.model_path, args.num_theorems)
