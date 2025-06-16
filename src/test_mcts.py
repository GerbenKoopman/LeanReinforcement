"""
Comprehensive test script for MCTS Agent

This script provides comprehensive testing of the Monte Carlo Tree Search agent
for theorem proving, including basic functionality, neural heuristics, and performance analysis.
"""

import sys
import time

from lean_dojo import LeanGitRepo, trace
from lean_rl import (
    LeanEnvironment,
    MCTSAgent,
    RandomAgent,
)
from lean_dojo.data_extraction.trace import is_available_in_cache


def setup_repository() -> tuple:
    """Set up the Lean repository and return traced repo and theorems."""
    print("Setting up Lean repository...")

    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "29dcec074de168ac2bf835a77ef68bbe069194c5",
    )

    # Check if already cached
    if is_available_in_cache(repo):
        print("2. Loading traced repository from cache...")
        traced_repo = trace(repo, dst_dir=None, build_deps=True)
    else:
        print("2. Tracing repository (this will take a while)...")
        traced_repo = trace(repo, dst_dir=None, build_deps=True)

    # Get some test theorems
    traced_file = traced_repo.get_traced_file("Mathlib/Algebra/BigOperators/Pi.lean")
    traced_theorems = traced_file.get_traced_theorems()

    print(f"Found {len(traced_theorems)} theorems in the file")
    return traced_repo, traced_theorems


def test_mcts_basic():
    """Test basic MCTS agent functionality."""
    print("\n" + "=" * 60)
    print("TESTING BASIC MCTS AGENT FUNCTIONALITY")
    print("=" * 60)

    traced_repo, traced_theorems = setup_repository()

    if not traced_theorems:
        print("No theorems found to test!")
        return

    # Initialize environment and agent
    env = LeanEnvironment(traced_repo.repo, reward_scheme="dense", max_steps=15)
    agent = MCTSAgent(
        iterations=50,  # Reasonable number for testing
        exploration_constant=1.4,
        max_rollout_depth=8,
        rollout_policy="random",
        seed=42,
    )

    # Test on first few theorems
    num_theorems_to_test = min(3, len(traced_theorems))

    for i in range(num_theorems_to_test):
        theorem = traced_theorems[i]
        print(f"\n--- Testing theorem {i+1}: {theorem.theorem.full_name} ---")

        # Reset environment
        state = env.reset(theorem.theorem)
        agent.reset()

        step_count = 0
        total_reward = 0
        start_time = time.time()

        print(f"Initial state: {state.num_goals} goals")

        while True:
            # Agent selects action using MCTS
            step_start = time.time()
            action = agent.select_action(state)
            action_time = time.time() - step_start

            print(
                f"Step {step_count + 1}: Action '{action}' (selected in {action_time:.2f}s)"
            )

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
                elapsed_time = time.time() - start_time
                print(
                    f"Episode finished after {step_count} steps in {elapsed_time:.2f}s"
                )
                print(f"Total reward: {total_reward}")
                print(f"Reason: {result.info.get('termination_reason', 'Unknown')}")

                if result.info["action_result"] == "proof_finished":
                    print("🎉 PROOF COMPLETED!")
                break

        # Show MCTS statistics
        stats = agent.get_statistics()
        print(f"MCTS Statistics:")
        print(f"  Tree size: {stats['tree_size']} nodes")
        print(f"  Total simulations: {stats['total_simulations']}")
        print(f"  Average node visits: {stats['average_node_visits']:.1f}")
        print(f"  Cache hits: {stats['cache_hits']}")

        if stats["top_actions_by_visits"]:
            print("  Top actions by visits:")
            for action, visits in stats["top_actions_by_visits"][:5]:
                print(f"    {action}: {visits} visits")

        print("-" * 50)


def test_mcts_neural_heuristic():
    """Test MCTS agent with neural heuristic."""
    print("\n" + "=" * 60)
    print("TESTING MCTS AGENT WITH NEURAL HEURISTIC")
    print("=" * 60)

    traced_repo, traced_theorems = setup_repository()

    if not traced_theorems:
        print("No theorems found to test!")
        return

    # Initialize environment and agent with neural heuristic
    env = LeanEnvironment(traced_repo.repo, reward_scheme="dense", max_steps=12)
    agent = MCTSAgent(
        iterations=75,
        exploration_constant=1.4,
        max_rollout_depth=6,
        rollout_policy="weighted",
        use_neural_heuristic=True,
        heuristic_learning_rate=0.01,
        feature_size=20,
        hidden_size=32,
        seed=42,
    )

    # Test on a couple of theorems
    num_theorems_to_test = min(2, len(traced_theorems))

    for i in range(num_theorems_to_test):
        theorem = traced_theorems[i]
        print(
            f"\n--- Testing theorem {i+1} with neural heuristic: {theorem.theorem.full_name} ---"
        )

        # Reset environment
        state = env.reset(theorem.theorem)
        agent.reset()

        step_count = 0
        total_reward = 0

        print(f"Initial state: {state.num_goals} goals")

        while True:
            # Agent selects action using MCTS with neural heuristic
            action = agent.select_action(state)
            print(f"Step {step_count + 1}: Action '{action}'")

            # Take step in environment
            result = env.step(action)

            # Update agent (this will train the neural network)
            agent.update(result)

            # Track progress
            step_count += 1
            total_reward += result.reward

            print(f"  Result: {result.info['action_result']}, Reward: {result.reward}")

            if result.state:
                print(f"  Goals remaining: {result.state.num_goals}")

            # Check if done
            if result.done:
                print(f"Episode finished after {step_count} steps")
                print(f"Total reward: {total_reward}")

                if result.info["action_result"] == "proof_finished":
                    print("🎉 PROOF COMPLETED!")
                break

        # Show neural heuristic statistics
        stats = agent.get_statistics()
        print(f"Neural Heuristic Statistics:")
        print(f"  Experience buffer size: {len(agent.experience_buffer)}")
        print(f"  Tree size: {stats['tree_size']} nodes")
        print(f"  Total simulations: {stats['total_simulations']}")

        print("-" * 50)


def test_mcts_vs_random_comparison():
    """Compare MCTS agent performance against random agent."""
    print("\n" + "=" * 60)
    print("COMPARING MCTS AGENT VS RANDOM AGENT")
    print("=" * 60)

    traced_repo, traced_theorems = setup_repository()

    if not traced_theorems:
        print("No theorems found to test!")
        return

    # Test parameters
    max_steps = 10
    num_tests = 3

    # Initialize environment
    env = LeanEnvironment(traced_repo.repo, reward_scheme="sparse", max_steps=max_steps)

    # Test agents
    agents = {
        "MCTS": MCTSAgent(
            iterations=50, exploration_constant=1.4, max_rollout_depth=5, seed=42
        ),
        "Random": RandomAgent(seed=42),
    }

    results = {
        name: {"successes": 0, "total_reward": 0.0, "avg_steps": 0.0}
        for name in agents.keys()
    }

    # Test each agent on the same theorems
    num_theorems_to_test = min(num_tests, len(traced_theorems))

    for i in range(num_theorems_to_test):
        theorem = traced_theorems[i]
        print(f"\n--- Testing theorem {i+1}: {theorem.theorem.full_name} ---")

        for agent_name, agent in agents.items():
            print(f"\n  Testing with {agent_name} agent:")

            # Reset environment and agent
            state = env.reset(theorem.theorem)
            agent.reset()

            step_count = 0
            total_reward = 0
            start_time = time.time()

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
                    elapsed_time = time.time() - start_time
                    print(f"    Finished in {step_count} steps ({elapsed_time:.2f}s)")
                    print(f"    Total reward: {total_reward}")

                    if result.info["action_result"] == "proof_finished":
                        print("    🎉 PROOF COMPLETED!")
                        results[agent_name]["successes"] += 1

                    results[agent_name]["total_reward"] += total_reward
                    results[agent_name]["avg_steps"] += step_count
                    break

    # Show comparison results
    print("\n" + "=" * 40)
    print("COMPARISON RESULTS")
    print("=" * 40)

    for agent_name, stats in results.items():
        avg_reward = stats["total_reward"] / num_theorems_to_test
        avg_steps = stats["avg_steps"] / num_theorems_to_test
        success_rate = (stats["successes"] / num_theorems_to_test) * 100

        print(f"{agent_name} Agent:")
        print(
            f"  Success rate: {success_rate:.1f}% ({stats['successes']}/{num_theorems_to_test})"
        )
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average steps: {avg_steps:.1f}")
        print()


def test_mcts_tree_growth():
    """Test how the MCTS tree grows during search."""
    print("\n" + "=" * 60)
    print("TESTING MCTS TREE GROWTH AND CONVERGENCE")
    print("=" * 60)

    traced_repo, traced_theorems = setup_repository()

    if not traced_theorems:
        print("No theorems found to test!")
        return

    # Test with different iteration counts
    iteration_counts = [25, 50, 100, 200]
    theorem = traced_theorems[0]  # Use first theorem

    print(f"Testing with theorem: {theorem.theorem.full_name}")

    for iterations in iteration_counts:
        print(f"\n--- Testing with {iterations} MCTS iterations ---")

        # Initialize environment and agent
        env = LeanEnvironment(traced_repo.repo, reward_scheme="dense", max_steps=5)
        agent = MCTSAgent(
            iterations=iterations,
            exploration_constant=1.4,
            max_rollout_depth=5,
            seed=42,
        )

        # Reset environment
        state = env.reset(theorem.theorem)
        agent.reset()

        # Perform one action selection to build the tree
        start_time = time.time()
        action = agent.select_action(state)
        selection_time = time.time() - start_time

        # Get statistics
        stats = agent.get_statistics()

        print(f"  Selected action: '{action}' in {selection_time:.3f}s")
        print(f"  Tree size: {stats['tree_size']} nodes")
        print(f"  Total simulations: {stats['total_simulations']}")
        print(f"  Average node visits: {stats['average_node_visits']:.1f}")

        if stats["top_actions_by_visits"]:
            print("  Top 3 actions by visits:")
            for action_name, visits in stats["top_actions_by_visits"][:3]:
                print(f"    {action_name}: {visits} visits")


def run_all_tests():
    """Run all MCTS tests."""
    print("🚀 Starting comprehensive MCTS agent testing...")
    print("=" * 70)

    try:
        # Basic functionality test
        test_mcts_basic()

        # Neural heuristic test
        test_mcts_neural_heuristic()

        # Comparison test
        test_mcts_vs_random_comparison()

        # Tree growth test
        test_mcts_tree_growth()

        print("\n" + "=" * 70)
        print("✅ All MCTS tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
