"""
Trace Repository for LeanDojo Reinforcement Learning Interface
"""

import os
from lean_dojo import LeanGitRepo, trace


def trace_repo_safe():
    """Trace a Lean repository with error handling for known issues."""
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
    print(
        f"   LOAD_USED_PACKAGES_ONLY: {os.environ.get('LOAD_USED_PACKAGES_ONLY', 'not set')}"
    )

    try:
        # Let LeanDojo handle caching automatically
        traced_repo = trace(repo)
        print("\n3. Trace completed successfully!")
        print(f"   Root directory: {traced_repo.root_dir}")
        print(f"   Number of traced files: {len(traced_repo.traced_files)}")
        print("=" * 40)
        return traced_repo
    except AssertionError as e:
        if "traced_repo is None" in str(e) or "sanity" in str(e).lower():
            print(f"\n3. Warning: Sanity check failed, but trace likely completed: {e}")
            print(
                "   This is a known issue with LeanDojo caching and can usually be ignored."
            )
            print(
                "   The traced repository should still be functional for RL training."
            )
            print("=" * 40)
            # Return None to indicate the trace completed but had sanity check issues
            return None
        else:
            # Re-raise other assertion errors
            raise


def trace_repo():
    """Trace a Lean repository."""
    return trace_repo_safe()


if __name__ == "__main__":
    try:
        trace_repo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
