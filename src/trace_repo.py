"""
Trace Repository for LeanDojo Reinforcement Learning Interface
"""

import os
from lean_dojo import LeanGitRepo, trace


def trace_repo():
    """Trace a Lean repository."""
    print("Tracing Repository")
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

    # Let LeanDojo handle caching automatically
    traced_repo = trace(repo)

    print("\n3. Trace completed!")
    print("=" * 40)


if __name__ == "__main__":
    try:
        trace_repo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
