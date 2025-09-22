"""
Distributed Environment for High-Performance Theorem Proving

This module provides a Ray-based distributed environment that can run multiple
Lean processes in parallel, following the pattern used by ReProver and LeanAgent.
"""

import ray
import time
from typing import List, cast, Tuple
from dataclasses import dataclass
from ray.util.actor_pool import ActorPool

from lean_dojo import Theorem, TracedRepo
from .environment import LeanEnvironment


@dataclass
class EpisodeResult:
    """Result of a complete episode."""

    theorem: Theorem
    success: bool
    reward: float
    steps_taken: int
    environment_time: float
    total_time: float
    episode_history: List[tuple]


@ray.remote
class LeanEnvironmentActor:
    """Ray actor wrapping LeanEnvironment for distributed processing."""

    def __init__(
        self,
        traced_repo: TracedRepo,
        timeout: int = 600,
        max_steps: int = 100,
        reward_scheme: str = "sparse",
    ):
        self.env = LeanEnvironment(traced_repo, timeout, max_steps, reward_scheme)

    def run_episode(self, theorem: Theorem, actions: List[str]) -> EpisodeResult:
        """Run a complete episode with given actions."""
        start_time = time.time()

        state = self.env.reset(theorem)
        total_reward = 0.0
        result = None

        for i, action in enumerate(actions):
            result = self.env.step(action)
            total_reward += result.reward

            if result.done:
                break

        episode_time = time.time() - start_time
        summary = self.env.get_episode_summary()

        # Determine if episode was successful (proof finished)
        success = False
        if result is not None:
            success = result.done and result.action_result == "proof_finished"

        return EpisodeResult(
            theorem=theorem,
            success=success,
            reward=total_reward,
            steps_taken=self.env.step_count,
            environment_time=summary["environment_time"],
            total_time=episode_time,
            episode_history=summary["history"],
        )


class DistributedLeanEnvironment:
    """
    Distributed environment manager that can run multiple Lean environments in parallel.

    This follows the pattern used by ReProver and LeanAgent for high-throughput theorem proving.
    """

    def __init__(self, traced_repo: TracedRepo, num_workers: int = 4, **env_kwargs):
        """
        Initialize distributed environment.

        Args:
            traced_repo: TracedRepo instance
            num_workers: Number of Ray worker processes
            **env_kwargs: Arguments passed to each LeanEnvironment
        """
        self.traced_repo = traced_repo
        self.num_workers = num_workers
        self.env_kwargs = env_kwargs

        # Initialize Ray actors
        self.actors = [
            LeanEnvironmentActor.remote(traced_repo, **env_kwargs)
            for _ in range(num_workers)
        ]
        self.actor_pool = ActorPool(self.actors)

    def run_episodes_parallel(
        self, theorem_action_pairs: List[Tuple[Theorem, List[str]]]
    ) -> List[EpisodeResult]:
        """
        Run multiple episodes in parallel.

        Args:
            theorem_action_pairs: List of (theorem, actions) tuples

        Returns:
            List of EpisodeResult objects
        """
        try:
            # Cast to proper type since ActorPool.map_unordered returns the correct type
            results = cast(
                List[EpisodeResult],
                list(
                    self.actor_pool.map_unordered(
                        lambda actor, pair: actor.run_episode.remote(pair[0], pair[1]),
                        theorem_action_pairs,
                    )
                ),
            )
            return results
        except Exception as ex:
            raise RuntimeError(f"Error during parallel execution: {ex}")

    def close(self):
        """Clean up Ray actors."""
        for actor in self.actors:
            ray.kill(actor)


# Example usage pattern (commented out to avoid execution)
"""
# Initialize Ray (if not already done)
ray.init()

# Create distributed environment
traced_repo = ... # your traced repo
dist_env = DistributedLeanEnvironment(traced_repo, num_workers=8, timeout=300)

# Prepare theorem-action pairs for parallel execution
theorem_action_pairs = [
    (theorem1, ["tactic1", "tactic2", "tactic3"]),
    (theorem2, ["tactic1", "tactic4"]),
    # ... more pairs
]

# Run episodes in parallel
results = dist_env.run_episodes_parallel(theorem_action_pairs)

# Analyze results
for result in results:
    print(f"Theorem: {result.theorem}")
    print(f"Success: {result.success}")
    print(f"Environment time: {result.environment_time:.2f}s")
    print(f"Reset overhead: {(result.total_time - result.environment_time) / result.total_time:.1%}")

dist_env.close()
"""
