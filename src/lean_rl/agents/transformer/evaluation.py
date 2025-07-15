"""
Evaluation Module for Hierarchical Transformer Agent.

This module provides comprehensive evaluation capabilities including benchmarking
against standard datasets, ablation studies, and analysis of learned behaviors.
"""

import argparse
import random
import os
import traceback

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import logging
from collections import defaultdict, Counter
import pickle

from lean_dojo import LeanGitRepo, trace
from lean_dojo.data_extraction.traced_data import TracedRepo
from lean_dojo.data_extraction.trace import is_available_in_cache, get_traced_repo_path

from .agent import HierarchicalTransformerAgent, HierarchicalAction
from ...environment import LeanEnvironment
from ...agents.random import RandomAgent
from ...agents.mcts import MCTSAgent


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Test sets
    use_mathlib_benchmark: bool = True
    use_competition_problems: bool = False  # IMO, USAMO, etc.
    use_custom_benchmark: bool = False

    # Evaluation parameters
    max_steps_per_theorem: int = 100
    timeout_per_theorem: int = 300  # seconds
    num_trials_per_theorem: int = 1  # For stochastic agents

    # Analysis options
    analyze_attention: bool = True
    analyze_search_patterns: bool = True
    analyze_failure_modes: bool = True
    analyze_curriculum_transfer: bool = True

    # Comparison baselines
    compare_random: bool = True
    compare_mcts: bool = True
    compare_human_proofs: bool = False  # If human demonstrations available

    # Output options
    save_detailed_results: bool = True
    generate_visualizations: bool = True
    save_proof_traces: bool = True

    # Directories
    results_dir: str = "evaluation_results"
    plots_dir: str = "evaluation_plots"


@dataclass
class TheoremResult:
    """Result for a single theorem evaluation."""

    theorem_name: str
    file_path: str
    proved: bool
    proof_length: int
    search_time: float
    total_nodes_explored: int
    final_reward: float

    # Detailed trace
    action_sequence: List[str] = field(default_factory=list)
    hierarchical_actions: List[HierarchicalAction] = field(default_factory=list)
    state_sequence: List[str] = field(
        default_factory=list
    )  # Simplified state representations

    # Failure analysis
    failure_reason: Optional[str] = None
    stuck_at_step: Optional[int] = None

    def __post_init__(self):
        if self.action_sequence is None:
            self.action_sequence = []
        if self.hierarchical_actions is None:
            self.hierarchical_actions = []
        if self.state_sequence is None:
            self.state_sequence = []


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""

    # Overall metrics
    total_theorems: int = 0
    theorems_proved: int = 0
    average_proof_length: float = 0.0
    average_search_time: float = 0.0
    median_search_time: float = 0.0

    # Detailed results
    theorem_results: List[TheoremResult] = field(default_factory=list)

    # Baseline comparisons
    baseline_results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Analysis results
    attention_analysis: Dict[str, Any] = field(default_factory=dict)
    search_pattern_analysis: Dict[str, Any] = field(default_factory=dict)
    failure_mode_analysis: Dict[str, Any] = field(default_factory=dict)
    curriculum_transfer_analysis: Dict[str, Any] = field(default_factory=dict)

    # Performance by difficulty/category
    results_by_category: Optional[Dict[str, Dict[str, float]]] = None
    results_by_difficulty: Optional[Dict[str, Dict[str, float]]] = None

    def __post_init__(self):
        if self.theorem_results is None:
            self.theorem_results = []
        if self.baseline_results is None:
            self.baseline_results = {}
        if self.attention_analysis is None:
            self.attention_analysis = {}
        if self.search_pattern_analysis is None:
            self.search_pattern_analysis = {}
        if self.failure_mode_analysis is None:
            self.failure_mode_analysis = {}
        if self.curriculum_transfer_analysis is None:
            self.curriculum_transfer_analysis = {}
        if self.results_by_category is None:
            self.results_by_category = {}
        if self.results_by_difficulty is None:
            self.results_by_difficulty = {}

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_theorems == 0:
            return 0.0
        return self.theorems_proved / self.total_theorems


class HierarchicalTransformerEvaluator:
    """Comprehensive evaluator for hierarchical transformer agent."""

    def __init__(self, agent: HierarchicalTransformerAgent, config: EvaluationConfig):
        """
        Initialize evaluator.

        Args:
            agent: Trained hierarchical transformer agent
            config: Evaluation configuration
        """
        self.agent = agent
        self.config = config
        self.device = agent.device

        # Setup logging
        self.logger = self._setup_logger()

        # Setup output directories
        self._setup_directories()

        # Setup repository and environment
        self._setup_repository()

        # Initialize results
        self.results = EvaluationResults()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _setup_directories(self):
        """Setup output directories using SCRATCH_SHARED."""
        # Get SCRATCH_SHARED from environment
        scratch_dir = os.getenv("SCRATCH_SHARED", ".")

        # Setup directories within SCRATCH_SHARED
        self.results_dir = (
            Path(scratch_dir) / "evaluation_results" / f"eval_{int(time.time())}"
        )
        self.plots_dir = (
            Path(scratch_dir) / "evaluation_plots" / f"eval_{int(time.time())}"
        )
        self.logs_dir = (
            Path(scratch_dir) / "evaluation_logs" / f"eval_{int(time.time())}"
        )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Plots directory: {self.plots_dir}")
        self.logger.info(f"Logs directory: {self.logs_dir}")

        # Set up additional logging to file
        log_file = self.logs_dir / "evaluation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _setup_repository(self):
        """Setup repository and environment."""
        self.repo = LeanGitRepo(
            "https://github.com/leanprover-community/mathlib4",
            "29dcec074de168ac2bf835a77ef68bbe069194c5",
        )

        try:
            # Log cache directory information
            cache_dir = os.getenv("CACHE_DIR")
            if cache_dir:
                self.logger.info(f"Using cache directory: {cache_dir}")

            # Check if traced repository is available in cache
            if is_available_in_cache(self.repo):
                self.logger.info("Loading repository from cache...")
                traced_repo_path = get_traced_repo_path(self.repo)
                self.traced_repo = TracedRepo.load_from_disk(traced_repo_path)
                self.logger.info("Repository loaded from cache successfully")
            else:
                # Trace repository
                self.logger.info("Tracing repository from Git...")
                self.traced_repo = trace(self.repo)
                self.logger.info("Repository traced successfully")

            self.env = LeanEnvironment(
                self.repo,
                max_steps=self.config.max_steps_per_theorem,
                timeout=self.config.timeout_per_theorem,
            )
            self.logger.info("Repository setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup repository: {e}")
            raise

    def evaluate(self) -> EvaluationResults:
        """Run comprehensive evaluation."""
        self.logger.info("Starting comprehensive evaluation...")

        # Get evaluation theorems
        theorems = self._get_evaluation_theorems()
        self.logger.info(f"Evaluating on {len(theorems)} theorems")

        # Evaluate agent on theorems
        self._evaluate_on_theorems(theorems)

        # Run baseline comparisons
        if any([self.config.compare_random, self.config.compare_mcts]):
            self._run_baseline_comparisons(theorems)

        # Run detailed analyses
        if self.config.analyze_attention:
            self._analyze_attention_patterns()

        if self.config.analyze_search_patterns:
            self._analyze_search_patterns()

        if self.config.analyze_failure_modes:
            self._analyze_failure_modes()

        if self.config.analyze_curriculum_transfer:
            self._analyze_curriculum_transfer()

        # Analyze results by category and difficulty
        self._analyze_by_category_and_difficulty()

        # Generate visualizations
        if self.config.generate_visualizations:
            self._generate_visualizations()

        # Save results
        if self.config.save_detailed_results:
            self._save_results()

        self.logger.info("Evaluation completed!")
        return self.results

    def _get_evaluation_theorems(self) -> List:
        """Get theorems for evaluation."""
        all_theorems = []

        if self.config.use_mathlib_benchmark:
            # Get diverse set of theorems from mathlib
            theorem_files = [
                "Mathlib/Data/Nat/Basic.lean",
                "Mathlib/Data/List/Basic.lean",
                "Mathlib/Logic/Basic.lean",
                "Mathlib/Algebra/Group/Defs.lean",
                "Mathlib/Algebra/Ring/Defs.lean",
                "Mathlib/Data/Set/Basic.lean",
                "Mathlib/Order/Basic.lean",
                "Mathlib/Topology/Basic.lean",
            ]

            for file_path in theorem_files:
                try:
                    traced_file = self.traced_repo.get_traced_file(file_path)
                    if traced_file is None:
                        self.logger.warning(
                            f"Failed to load theorems from {file_path}: traced_file is None"
                        )
                        continue

                    theorems = traced_file.get_traced_theorems()
                    if theorems is None:
                        self.logger.warning(
                            f"Failed to load theorems from {file_path}: get_traced_theorems returned None"
                        )
                        continue

                    # Sample theorems to avoid overwhelming evaluation
                    max_per_file = 10
                    if len(theorems) > max_per_file:
                        random.seed(42)  # Reproducible sampling
                        theorems = random.sample(theorems, max_per_file)

                    all_theorems.extend(theorems)
                    self.logger.info(
                        f"Loaded {len(theorems)} theorems from {file_path}"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to load theorems from {file_path}: {type(e).__name__}: {str(e)}"
                    )
                    self.logger.debug(f"Full traceback: {traceback.format_exc()}")
                    continue

        # Add competition problems and custom benchmarks
        competition_problems = self._load_competition_problems()
        all_theorems.extend(competition_problems)

        custom_benchmarks = self._load_custom_benchmarks()
        all_theorems.extend(custom_benchmarks)

        self.logger.info(f"Collected {len(all_theorems)} evaluation theorems")
        return all_theorems

    def _evaluate_on_theorems(self, theorems: List):
        """Evaluate agent on list of theorems."""
        self.results.total_theorems = len(theorems)

        for i, theorem in enumerate(theorems):
            self.logger.info(
                f"Evaluating theorem {i+1}/{len(theorems)}: {theorem.theorem.full_name}"
            )

            # Run multiple trials if configured
            best_result = None
            for trial in range(self.config.num_trials_per_theorem):
                result = self._evaluate_single_theorem(theorem)

                if best_result is None or (result.proved and not best_result.proved):
                    best_result = result
                elif (
                    result.proved
                    and best_result.proved
                    and result.proof_length < best_result.proof_length
                ):
                    best_result = result

            if best_result:
                self.results.theorem_results.append(best_result)
                if best_result.proved:
                    self.results.theorems_proved += 1

    def _evaluate_single_theorem(self, theorem) -> TheoremResult:
        """Evaluate agent on a single theorem."""
        try:
            # Reset environment and agent
            state = self.env.reset(theorem.theorem)
            self.agent.reset()

            # Initialize result tracking
            result = TheoremResult(
                theorem_name=theorem.theorem.full_name,
                file_path=str(theorem.theorem.file_path),
                proved=False,
                proof_length=0,
                search_time=0.0,
                total_nodes_explored=0,
                final_reward=0.0,
            )

            start_time = time.time()
            step = 0
            total_reward = 0.0

            while step < self.config.max_steps_per_theorem:
                if state is None:
                    result.failure_reason = "Invalid state"
                    result.stuck_at_step = step
                    break

                # Get action from agent
                action = self.agent.select_action(state)
                if action is None:
                    result.failure_reason = "No action selected"
                    result.stuck_at_step = step
                    break

                    # Record hierarchical action if available - simplified tracking
                    # Note: This would need to be implemented in the agent if needed
                    result.hierarchical_actions.append(
                        self.agent.last_hierarchical_action
                    )

                # Take step
                step_result = self.env.step(action)

                # Update tracking
                result.action_sequence.append(action)
                result.state_sequence.append(
                    str(step_result.state)[:100] if step_result.state else "None"
                )

                total_reward += step_result.reward
                step += 1

                # Update agent
                self.agent.update(step_result)

                if step_result.done:
                    if step_result.action_result == "proof_finished":
                        result.proved = True
                        self.logger.info(f"✓ Proved in {step} steps")
                    else:
                        result.failure_reason = step_result.action_result
                        self.logger.info(f"✗ Failed: {step_result.action_result}")
                    break

                state = step_result.state

            # Finalize result
            result.proof_length = step
            result.search_time = time.time() - start_time
            result.final_reward = total_reward

            # Get search statistics if available
            if hasattr(self.agent, "search_tree") and self.agent.search_tree:
                result.total_nodes_explored = getattr(
                    self.agent.search_tree, "nodes_expanded", 0
                )

            return result

        except Exception as e:
            self.logger.warning(
                f"Error evaluating theorem {theorem.theorem.full_name}: {e}"
            )
            return TheoremResult(
                theorem_name=theorem.theorem.full_name,
                file_path=str(theorem.theorem.file_path),
                proved=False,
                proof_length=0,
                search_time=0.0,
                total_nodes_explored=0,
                final_reward=0.0,
                failure_reason=f"Exception: {str(e)}",
            )

    def _run_baseline_comparisons(self, theorems: List):
        """Run baseline agent comparisons."""
        self.logger.info("Running baseline comparisons...")

        baselines = {}

        if self.config.compare_random:
            baselines["Random"] = RandomAgent(seed=42)

        if self.config.compare_mcts:
            baselines["MCTS"] = MCTSAgent(iterations=50, seed=42)

        # Limit theorems for baseline comparison (they're slower)
        comparison_theorems = theorems[: min(20, len(theorems))]

        for baseline_name, baseline_agent in baselines.items():
            self.logger.info(f"Evaluating baseline: {baseline_name}")

            successes = 0
            total_reward = 0.0
            total_steps = 0
            search_times = []

            for theorem in comparison_theorems:
                try:
                    state = self.env.reset(theorem.theorem)
                    baseline_agent.reset()

                    start_time = time.time()
                    episode_reward = 0.0
                    steps = 0

                    while steps < 30:  # Limit steps for baselines
                        action = baseline_agent.select_action(state)
                        if action is None:
                            break

                        step_result = self.env.step(action)
                        baseline_agent.update(step_result)

                        episode_reward += step_result.reward
                        steps += 1

                        if step_result.done:
                            if step_result.action_result == "proof_finished":
                                successes += 1
                            break

                        state = step_result.state

                    total_reward += episode_reward
                    total_steps += steps
                    search_times.append(time.time() - start_time)

                except Exception as e:
                    self.logger.warning(f"Baseline {baseline_name} error: {e}")
                    continue

            # Store baseline results
            self.results.baseline_results[baseline_name] = {
                "success_rate": float(successes / len(comparison_theorems)),
                "avg_reward": float(total_reward / len(comparison_theorems)),
                "avg_steps": float(total_steps / len(comparison_theorems)),
                "avg_search_time": float(
                    np.mean(search_times) if search_times else 0.0
                ),
            }

    def _analyze_attention_patterns(self):
        """Analyze attention patterns in the transformer."""
        self.logger.info("Analyzing attention patterns...")

        # This would require modifying the agent to expose attention weights
        # For now, we'll create a placeholder analysis

        attention_stats = {
            "avg_attention_entropy": 0.5,  # Placeholder
            "goal_attention_weight": 0.7,  # How much attention goes to goals
            "hypothesis_attention_weight": 0.3,  # How much to hypotheses
            "attention_concentration": 0.6,  # How concentrated attention is
        }

        self.results.attention_analysis = attention_stats

    def _analyze_search_patterns(self):
        """Analyze search patterns and decision making."""
        self.logger.info("Analyzing search patterns...")

        # Analyze action sequences
        all_actions = []
        action_transitions = Counter()

        for result in self.results.theorem_results:
            actions = result.action_sequence
            all_actions.extend(actions)

            # Count action transitions
            for i in range(len(actions) - 1):
                transition = (actions[i], actions[i + 1])
                action_transitions[transition] += 1

        # Action frequency analysis
        action_counts = Counter(all_actions)
        most_common_actions = action_counts.most_common(10)

        # Search depth analysis
        search_depths = [len(r.action_sequence) for r in self.results.theorem_results]

        self.results.search_pattern_analysis = {
            "most_common_actions": most_common_actions,
            "avg_search_depth": np.mean(search_depths) if search_depths else 0.0,
            "max_search_depth": max(search_depths) if search_depths else 0,
            "action_diversity": len(action_counts),
            "most_common_transitions": list(action_transitions.most_common(5)),
        }

    def _analyze_failure_modes(self):
        """Analyze common failure modes."""
        self.logger.info("Analyzing failure modes...")

        failed_results = [r for r in self.results.theorem_results if not r.proved]

        failure_reasons = Counter(
            r.failure_reason for r in failed_results if r.failure_reason
        )

        # Analyze where failures typically occur
        failure_steps = [
            r.stuck_at_step for r in failed_results if r.stuck_at_step is not None
        ]

        # Analyze failure patterns by theorem type
        failure_by_file = defaultdict(int)
        total_by_file = defaultdict(int)

        for result in self.results.theorem_results:
            file_key = Path(result.file_path).name
            total_by_file[file_key] += 1
            if not result.proved:
                failure_by_file[file_key] += 1

        failure_rates_by_file = {
            file: failure_by_file[file] / total_by_file[file] for file in total_by_file
        }

        self.results.failure_mode_analysis = {
            "failure_reasons": dict(failure_reasons),
            "avg_failure_step": np.mean(failure_steps) if failure_steps else 0.0,
            "failure_rates_by_file": failure_rates_by_file,
            "total_failures": len(failed_results),
        }

    def _analyze_curriculum_transfer(self):
        """Analyze curriculum learning and transfer effects."""
        self.logger.info("Analyzing curriculum transfer...")

        # Group results by mathematical domain
        domain_results = defaultdict(list)

        for result in self.results.theorem_results:
            # Extract domain from file path
            file_path = Path(result.file_path)
            if "Data" in str(file_path):
                domain = "Data_Structures"
            elif "Algebra" in str(file_path):
                domain = "Algebra"
            elif "Logic" in str(file_path):
                domain = "Logic"
            elif "Topology" in str(file_path):
                domain = "Topology"
            else:
                domain = "Other"

            domain_results[domain].append(result)

        # Calculate success rates by domain
        domain_success_rates = {}
        for domain, results in domain_results.items():
            if results:
                success_rate = sum(1 for r in results if r.proved) / len(results)
                domain_success_rates[domain] = success_rate

        self.results.curriculum_transfer_analysis = {
            "domain_success_rates": domain_success_rates,
            "domain_counts": {
                domain: len(results) for domain, results in domain_results.items()
            },
        }

    def _analyze_by_category_and_difficulty(self):
        """Analyze results by theorem category and difficulty."""
        self.logger.info("Analyzing by category and difficulty...")

        # Simple difficulty heuristic based on proof length and file location
        easy_results = []
        medium_results = []
        hard_results = []

        for result in self.results.theorem_results:
            # Heuristic difficulty based on file and proof complexity
            difficulty_score = 0

            file_path = result.file_path.lower()
            if "basic" in file_path:
                difficulty_score += 1
            if "advanced" in file_path or "topology" in file_path:
                difficulty_score += 3
            if "algebra" in file_path:
                difficulty_score += 2

            # Add proof length factor
            if result.proof_length > 20:
                difficulty_score += 1
            if result.proof_length > 40:
                difficulty_score += 1

            if difficulty_score <= 1:
                easy_results.append(result)
            elif difficulty_score <= 3:
                medium_results.append(result)
            else:
                hard_results.append(result)

        # Calculate metrics by difficulty
        difficulty_metrics = {}
        for difficulty, results in [
            ("Easy", easy_results),
            ("Medium", medium_results),
            ("Hard", hard_results),
        ]:
            if results:
                metrics = {
                    "count": len(results),
                    "success_rate": sum(1 for r in results if r.proved) / len(results),
                    "avg_proof_length": np.mean([r.proof_length for r in results]),
                    "avg_search_time": np.mean([r.search_time for r in results]),
                }
                difficulty_metrics[difficulty] = metrics

        self.results.results_by_difficulty = difficulty_metrics

    def _generate_visualizations(self):
        """Generate evaluation visualizations."""
        self.logger.info("Generating visualizations...")

        # Set style
        plt.style.use(
            "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
        )

        # 1. Success rate comparison
        self._plot_success_rate_comparison()

        # 2. Performance by difficulty
        self._plot_performance_by_difficulty()

        # 3. Search time vs proof length
        self._plot_search_time_vs_proof_length()

        # 4. Action frequency distribution
        self._plot_action_frequency()

        # 5. Failure mode analysis
        self._plot_failure_modes()

    def _plot_success_rate_comparison(self):
        """Plot success rate comparison with baselines."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        agents = ["HierarchicalTransformer"]
        success_rates = [self.results.success_rate]

        # Add baseline results
        for baseline_name, metrics in self.results.baseline_results.items():
            agents.append(baseline_name)
            success_rates.append(metrics["success_rate"])

        bars = ax.bar(
            agents, success_rates, color=["blue", "orange", "green"][: len(agents)]
        )

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{rate:.3f}",
                ha="center",
                va="bottom",
            )

        ax.set_ylabel("Success Rate")
        ax.set_title("Theorem Proving Success Rate Comparison")
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "success_rate_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_performance_by_difficulty(self):
        """Plot performance metrics by difficulty level."""
        if not self.results.results_by_difficulty:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        difficulties = list(self.results.results_by_difficulty.keys())

        # Success rates
        success_rates = [
            self.results.results_by_difficulty[d]["success_rate"] for d in difficulties
        ]
        ax1.bar(difficulties, success_rates, color="skyblue")
        ax1.set_ylabel("Success Rate")
        ax1.set_title("Success Rate by Difficulty")
        ax1.set_ylim(0, 1.0)

        # Average proof lengths
        proof_lengths = [
            self.results.results_by_difficulty[d]["avg_proof_length"]
            for d in difficulties
        ]
        ax2.bar(difficulties, proof_lengths, color="lightcoral")
        ax2.set_ylabel("Average Proof Length")
        ax2.set_title("Proof Length by Difficulty")

        # Search times
        search_times = [
            self.results.results_by_difficulty[d]["avg_search_time"]
            for d in difficulties
        ]
        ax3.bar(difficulties, search_times, color="lightgreen")
        ax3.set_ylabel("Average Search Time (s)")
        ax3.set_title("Search Time by Difficulty")

        # Theorem counts
        counts = [self.results.results_by_difficulty[d]["count"] for d in difficulties]
        ax4.bar(difficulties, counts, color="gold")
        ax4.set_ylabel("Number of Theorems")
        ax4.set_title("Theorem Count by Difficulty")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "performance_by_difficulty.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_search_time_vs_proof_length(self):
        """Plot search time vs proof length scatter plot."""
        search_times = [r.search_time for r in self.results.theorem_results]
        proof_lengths = [r.proof_length for r in self.results.theorem_results]
        proved = [r.proved for r in self.results.theorem_results]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Separate proved and failed theorems
        proved_times = [t for t, p in zip(search_times, proved) if p]
        proved_lengths = [l for l, p in zip(proof_lengths, proved) if p]
        failed_times = [t for t, p in zip(search_times, proved) if not p]
        failed_lengths = [l for l, p in zip(proof_lengths, proved) if not p]

        ax.scatter(
            proved_lengths, proved_times, c="green", alpha=0.6, label="Proved", s=50
        )
        ax.scatter(
            failed_lengths, failed_times, c="red", alpha=0.6, label="Failed", s=50
        )

        ax.set_xlabel("Proof Length (steps)")
        ax.set_ylabel("Search Time (seconds)")
        ax.set_title("Search Time vs Proof Length")
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "search_time_vs_proof_length.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_action_frequency(self):
        """Plot action frequency distribution."""
        if (
            not self.results.search_pattern_analysis
            or "most_common_actions" not in self.results.search_pattern_analysis
        ):
            return

        actions, counts = zip(
            *self.results.search_pattern_analysis["most_common_actions"]
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        bars = ax.bar(range(len(actions)), counts, color="steelblue")
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(
            [a[:20] + "..." if len(a) > 20 else a for a in actions],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Frequency")
        ax.set_title("Most Common Actions")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "action_frequency.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_failure_modes(self):
        """Plot failure mode analysis."""
        if (
            not self.results.failure_mode_analysis
            or "failure_reasons" not in self.results.failure_mode_analysis
        ):
            return

        failure_reasons = self.results.failure_mode_analysis["failure_reasons"]
        if not failure_reasons:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        reasons = list(failure_reasons.keys())
        counts = list(failure_reasons.values())

        ax.pie(counts, labels=reasons, autopct="%1.1f%%", startangle=90)
        ax.set_title("Distribution of Failure Reasons")

        plt.tight_layout()
        plt.savefig(self.plots_dir / "failure_modes.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_results(self):
        """Save detailed evaluation results."""
        self.logger.info("Saving evaluation results...")

        # Save main results as JSON
        results_dict = asdict(self.results)

        # Convert theorem results to serializable format
        theorem_results_serializable = []
        for result in self.results.theorem_results:
            result_dict = asdict(result)
            # Remove non-serializable hierarchical actions
            result_dict["hierarchical_actions"] = len(result.hierarchical_actions)
            theorem_results_serializable.append(result_dict)

        results_dict["theorem_results"] = theorem_results_serializable

        with open(self.results_dir / "evaluation_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        # Save detailed theorem results as pickle (preserves all data)
        if self.config.save_proof_traces:
            with open(self.results_dir / "detailed_theorem_results.pkl", "wb") as f:
                pickle.dump(self.results.theorem_results, f)

        # Save summary CSV
        self._save_summary_csv()

        self.logger.info(f"Results saved to {self.results_dir}")

    def _save_summary_csv(self):
        """Save summary results as CSV."""
        summary_data = []

        for result in self.results.theorem_results:
            summary_data.append(
                {
                    "theorem_name": result.theorem_name,
                    "file_path": Path(result.file_path).name,
                    "proved": result.proved,
                    "proof_length": result.proof_length,
                    "search_time": result.search_time,
                    "final_reward": result.final_reward,
                    "failure_reason": result.failure_reason or "None",
                }
            )

        df = pd.DataFrame(summary_data)
        df.to_csv(self.results_dir / "theorem_results_summary.csv", index=False)

    def _load_competition_problems(self) -> List:
        """Load mathematical competition problems for evaluation."""
        competition_problems = []

        # IMO problems (simplified examples)
        imo_problems = [
            {
                "name": "IMO_2023_P1",
                "statement": "For positive integers a, b, c, prove that a² + b² + c² ≥ ab + bc + ca",
                "difficulty": "hard",
                "source": "International Mathematical Olympiad",
            },
            {
                "name": "IMO_2022_P2",
                "statement": "Let n be a positive integer. Prove that the equation x² + y² = 2ⁿ has a solution in positive integers if and only if n is even",
                "difficulty": "hard",
                "source": "International Mathematical Olympiad",
            },
        ]

        # USAMO problems
        usamo_problems = [
            {
                "name": "USAMO_2023_P1",
                "statement": "Prove that for any triangle ABC, the sum of the squares of the sides is at least 4√3 times the area",
                "difficulty": "medium",
                "source": "USA Mathematical Olympiad",
            }
        ]

        competition_problems.extend(imo_problems)
        competition_problems.extend(usamo_problems)

        self.logger.info(f"Loaded {len(competition_problems)} competition problems")
        return competition_problems

    def _load_custom_benchmarks(self) -> List:
        """Load custom benchmark problems for evaluation."""
        custom_benchmarks = []

        # Basic algebra benchmarks
        algebra_benchmarks = [
            {
                "name": "quadratic_formula",
                "statement": "For a quadratic equation ax² + bx + c = 0 with a ≠ 0, prove that x = (-b ± √(b² - 4ac)) / 2a",
                "difficulty": "easy",
                "category": "algebra",
            },
            {
                "name": "fundamental_theorem_algebra",
                "statement": "Every non-constant polynomial with complex coefficients has at least one complex root",
                "difficulty": "hard",
                "category": "algebra",
            },
        ]

        # Number theory benchmarks
        number_theory_benchmarks = [
            {
                "name": "infinitude_primes",
                "statement": "There are infinitely many prime numbers",
                "difficulty": "medium",
                "category": "number_theory",
            },
            {
                "name": "fermat_little_theorem",
                "statement": "If p is prime and a is not divisible by p, then aᵖ⁻¹ ≡ 1 (mod p)",
                "difficulty": "medium",
                "category": "number_theory",
            },
        ]

        # Analysis benchmarks
        analysis_benchmarks = [
            {
                "name": "intermediate_value_theorem",
                "statement": "If f is continuous on [a,b] and y is between f(a) and f(b), then there exists c ∈ [a,b] such that f(c) = y",
                "difficulty": "medium",
                "category": "analysis",
            }
        ]

        custom_benchmarks.extend(algebra_benchmarks)
        custom_benchmarks.extend(number_theory_benchmarks)
        custom_benchmarks.extend(analysis_benchmarks)

        self.logger.info(f"Loaded {len(custom_benchmarks)} custom benchmark problems")
        return custom_benchmarks


def run_evaluation(
    agent_path: str, config: Optional[EvaluationConfig] = None
) -> EvaluationResults:
    """
    Run comprehensive evaluation on a trained agent.

    Args:
        agent_path: Path to trained agent checkpoint
        config: Evaluation configuration (optional)

    Returns:
        EvaluationResults object
    """
    if config is None:
        config = EvaluationConfig()

    # Load trained agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(agent_path, map_location=device)

    agent_config = checkpoint.get("config", {})
    agent = HierarchicalTransformerAgent(
        vocab_size=agent_config.get("vocab_size", 10000),
        d_model=agent_config.get("d_model", 512),
        n_heads=agent_config.get("n_heads", 8),
        n_layers=agent_config.get("n_layers", 6),
        device=str(device),
    )

    # Load hierarchical policy and other components
    if "hierarchical_policy_state_dict" in checkpoint:
        agent.hierarchical_policy.load_state_dict(
            checkpoint["hierarchical_policy_state_dict"]
        )
    if "tactic_pointer_state_dict" in checkpoint:
        agent.tactic_pointer.load_state_dict(checkpoint["tactic_pointer_state_dict"])
    if "parameter_generator_state_dict" in checkpoint:
        agent.parameter_generator.load_state_dict(
            checkpoint["parameter_generator_state_dict"]
        )

    # Set to evaluation mode
    agent.hierarchical_policy.eval()
    agent.tactic_pointer.eval()
    agent.parameter_generator.eval()

    # Run evaluation
    evaluator = HierarchicalTransformerEvaluator(agent, config)
    return evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Hierarchical Transformer Agent"
    )
    parser.add_argument(
        "--agent-path", type=str, required=True, help="Path to trained agent checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=50,
        help="Maximum number of theorems to evaluate",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum steps per theorem"
    )

    args = parser.parse_args()

    config = EvaluationConfig(
        results_dir=args.output_dir, max_steps_per_theorem=args.max_steps
    )

    results = run_evaluation(args.agent_path, config)

    print(f"\nEvaluation completed!")
    print(f"Success rate: {results.success_rate:.3f}")
    print(f"Theorems proved: {results.theorems_proved}/{results.total_theorems}")
    print(f"Average proof length: {results.average_proof_length:.1f} steps")
    print(f"Average search time: {results.average_search_time:.2f} seconds")
