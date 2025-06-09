"""
Demo script for LeanDojo Reinforcement Learning Interface

This script demonstrates how to use the RL environment and agents
to interact with Lean theorems through LeanDojo.
"""

import os
import sys
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MATHLIB4_DATASET_PATH = os.getenv("MATHLIB4_DATASET_PATH")

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lean_dojo import LeanGitRepo, trace, Theorem
from lean_rl import LeanEnvironment, RandomAgent, WeightedRandomAgent


def setup_repository() -> tuple:
    """Set up the Lean repository and return traced repo and theorems."""
    print("Setting up Lean repository...")

    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "29dcec074de168ac2bf835a77ef68bbe069194c5",
    )

    print("Tracing repository (this may take a while)...")
    traced_repo = trace(repo, dst_dir=MATHLIB4_DATASET_PATH, build_deps=True)

    # Get some test theorems
    traced_file = traced_repo.get_traced_file("Mathlib/Algebra/BigOperators/Pi.lean")
    traced_theorems = traced_file.get_traced_theorems()

    print(f"Found {len(traced_theorems)} theorems in the file")
    return traced_repo, traced_theorems


def test_environment_basic():
    """Test basic environment functionality."""
    print("\n" + "=" * 50)
    print("TESTING BASIC ENVIRONMENT FUNCTIONALITY")
    print("=" * 50)

    traced_repo, traced_theorems = setup_repository()

    if not traced_theorems:
        print("No theorems found to test!")
        return

    # Pick the first theorem for testing
    theorem = traced_theorems[0]
    print(f"Testing with theorem: {theorem.theorem.full_name}")

    # Initialize environment
    env = LeanEnvironment(traced_repo.repo, reward_scheme="sparse")

    # Reset environment with the theorem
    initial_state = env.reset(theorem.theorem)
    print(f"Initial state has {initial_state.num_goals} goals")
    print(f"Initial goals:\n{initial_state.goals}")

    # Try a few manual actions
    test_actions = ["rfl", "simp", "intro h", "sorry"]

    for i, action in enumerate(test_actions):
        print(f"\nStep {i+1}: Trying action '{action}'")
        result = env.step(action)

        print(f"Result: {result.info['action_result']}")
        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")

        if result.state:
            print(f"New state has {result.state.num_goals} goals")

        if result.done:
            print("Episode finished!")
            break


def test_random_agent():
    """Test the random agent."""
    print("\n" + "=" * 50)
    print("TESTING RANDOM AGENT")
    print("=" * 50)

    traced_repo, traced_theorems = setup_repository()

    if not traced_theorems:
        print("No theorems found to test!")
        return

    # Initialize environment and agent
    env = LeanEnvironment(traced_repo.repo, reward_scheme="dense", max_steps=20)
    agent = RandomAgent(seed=42)

    # Test on multiple theorems
    num_theorems_to_test = min(3, len(traced_theorems))

    for i in range(num_theorems_to_test):
        theorem = traced_theorems[i]
        print(f"\nTesting theorem {i+1}: {theorem.theorem.full_name}")

        # Reset environment
        state = env.reset(theorem.theorem)
        agent.reset()

        step_count = 0
        total_reward = 0

        print(f"Initial state: {state.num_goals} goals")

        while True:
            # Agent selects action
            action = agent.select_action(state)
            print(f"Step {step_count + 1}: Action '{action}'")

            # Take step in environment
            result = env.step(action)

            # Update agent
            agent.update(result)

            # Track progress
            step_count += 1
            total_reward += result.reward

            print(f"  Result: {result.info['action_result']}")
            print(f"  Reward: {result.reward}")

            if result.state:
                print(f"  Goals remaining: {result.state.num_goals}")

            # Check if done
            if result.done:
                print(f"Episode finished after {step_count} steps")
                print(f"Total reward: {total_reward}")
                print(f"Reason: {result.info.get('termination_reason', 'Unknown')}")
                break

        print("-" * 30)


def test_weighted_random_agent():
    """Test the weighted random agent that learns from success rates."""
    print("\n" + "=" * 50)
    print("TESTING WEIGHTED RANDOM AGENT")
    print("=" * 50)

    traced_repo, traced_theorems = setup_repository()

    if not traced_theorems:
        print("No theorems found to test!")
        return

    # Initialize environment and agent
    env = LeanEnvironment(traced_repo.repo, reward_scheme="shaped", max_steps=15)
    agent = WeightedRandomAgent(learning_rate=0.1, seed=42)

    # Test on multiple episodes to see learning
    theorem = traced_theorems[0]  # Use same theorem to see adaptation
    print(f"Testing with theorem: {theorem.theorem.full_name}")

    num_episodes = 5

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}:")

        # Reset environment
        state = env.reset(theorem.theorem)
        agent.reset()

        step_count = 0
        total_reward = 0

        while True:
            # Agent selects action
            action = agent.select_action(state)

            # Take step in environment
            result = env.step(action)

            # Update agent
            agent.update(result)

            step_count += 1
            total_reward += result.reward

            if result.done:
                print(f"  Finished in {step_count} steps, reward: {total_reward:.2f}")
                break

        # Show agent's learned tactic weights (top 5)
        stats = agent.get_statistics()
        if "tactic_weights" in stats:
            print("  Top tactics by weight:")
            for tactic_info in stats["tactic_weights"][:5]:
                if tactic_info["attempts"] > 0:
                    print(
                        f"    {tactic_info['tactic']}: {tactic_info['success_rate']:.2f} "
                        f"({tactic_info['successes']}/{tactic_info['attempts']})"
                    )


def run_benchmark():
    """Run a simple benchmark comparing different agents."""
    print("\n" + "=" * 50)
    print("RUNNING AGENT BENCHMARK")
    print("=" * 50)

    traced_repo, traced_theorems = setup_repository()

    if len(traced_theorems) < 2:
        print("Need at least 2 theorems for benchmark!")
        return

    # Test different agents
    agents = {
        "Random": RandomAgent(seed=42),
        "WeightedRandom": WeightedRandomAgent(learning_rate=0.2, seed=42),
    }

    results = {}
    num_theorems = min(3, len(traced_theorems))

    for agent_name, agent in agents.items():
        print(f"\nTesting {agent_name} agent:")

        episode_results = []

        for i in range(num_theorems):
            theorem = traced_theorems[i]
            env = LeanEnvironment(traced_repo.repo, reward_scheme="dense", max_steps=10)

            state = env.reset(theorem.theorem)
            agent.reset()

            steps = 0
            total_reward = 0
            success = False

            while True:
                action = agent.select_action(state)
                result = env.step(action)
                agent.update(result)

                steps += 1
                total_reward += result.reward

                if result.done:
                    success = result.info["action_result"] == "proof_finished"
                    break

            episode_results.append(
                {
                    "theorem": theorem.theorem.full_name,
                    "steps": steps,
                    "reward": total_reward,
                    "success": success,
                }
            )

            print(
                f"  {theorem.theorem.full_name}: {steps} steps, "
                f"reward {total_reward:.2f}, "
                f"{'SUCCESS' if success else 'FAILED'}"
            )

        results[agent_name] = episode_results

    # Summary
    print("\nBenchmark Summary:")
    for agent_name, episode_results in results.items():
        avg_steps = sum(r["steps"] for r in episode_results) / len(episode_results)
        avg_reward = sum(r["reward"] for r in episode_results) / len(episode_results)
        success_rate = sum(r["success"] for r in episode_results) / len(episode_results)

        print(f"{agent_name}:")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Success rate: {success_rate:.1%}")


def main():
    """Run all tests and demos."""
    print("LeanDojo Reinforcement Learning Interface Demo")
    print("=" * 60)

    try:
        # Run basic environment test
        test_environment_basic()

        # Test random agent
        test_random_agent()

        # Test weighted random agent
        test_weighted_random_agent()

        # Run benchmark
        run_benchmark()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
