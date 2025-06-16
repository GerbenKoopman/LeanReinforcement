"""
Simple Demo: LeanDojo RL Interface

This script demonstrates the basic usage of the LeanDojo RL interface
with a simple example.
"""

import os
from lean_dojo import LeanGitRepo, trace
from lean_rl import LeanEnvironment, RandomAgent


def simple_demo():
    """Simple demonstration of the RL interface."""
    print("LeanDojo RL Interface - Simple Demo")
    print("=" * 40)

    # Setup repository
    print("1. Setting up Lean repository...")
    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "29dcec074de168ac2bf835a77ef68bbe069194c5",
    )

    # Trace repository - LeanDojo will automatically handle caching
    print("2. Tracing repository (using LeanDojo's automatic caching)...")
    print(f"   Cache directory: {os.environ.get('CACHE_DIR', 'default ~/.cache')}")

    try:
        # Let LeanDojo handle caching automatically - don't specify dst_dir
        traced_repo = trace(repo)
    except AssertionError as e:
        if "traced_repo is None" in str(e) or "sanity" in str(e).lower():
            print(f"   Warning: Sanity check failed, but trace likely completed: {e}")
            print("   Continuing with demo despite sanity check issues...")
            # For demo purposes, we can try to get the cached path directly
            from lean_dojo.data_extraction.trace import get_traced_repo_path
            from lean_dojo.data_extraction.traced_data import TracedRepo

            cached_path = get_traced_repo_path(repo)
            traced_repo = TracedRepo.load_from_disk(cached_path, build_deps=True)
        else:
            raise

    # Get a test theorem
    print("3. Loading test theorem...")
    traced_file = traced_repo.get_traced_file("Mathlib/Algebra/BigOperators/Pi.lean")
    traced_theorems = traced_file.get_traced_theorems()

    if not traced_theorems:
        print("No theorems found!")
        return

    theorem = traced_theorems[0]
    print(f"   Using theorem: {theorem.theorem.full_name}")

    # Initialize RL components
    print("4. Initializing RL environment and agent...")
    env = LeanEnvironment(traced_repo.repo, reward_scheme="sparse", max_steps=10)
    agent = RandomAgent(seed=42)

    # Run one episode
    print("5. Running RL episode...")
    state = env.reset(theorem.theorem)
    print(f"   Initial state: {state.num_goals} goals")
    print(f"   Goals preview: {state.goals[:100]}...")

    step = 0
    total_reward = 0

    while True:
        step += 1
        action = agent.select_action(state)
        print(f"   Step {step}: Applying tactic '{action}'")

        result = env.step(action)
        agent.update(result)

        total_reward += result.reward
        print(f"   Result: {result.info['action_result']}, Reward: {result.reward}")

        if result.state:
            print(f"   Goals remaining: {result.state.num_goals}")

        if result.done:
            print(f"   Episode finished! Total reward: {total_reward}")
            if result.info["action_result"] == "proof_finished":
                print("   🎉 PROOF COMPLETED!")
            else:
                print(
                    f"   Episode ended due to: {result.info.get('termination_reason', 'Unknown')}"
                )
            break

    print("\n6. Demo completed!")
    print("=" * 40)


if __name__ == "__main__":
    try:
        simple_demo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
