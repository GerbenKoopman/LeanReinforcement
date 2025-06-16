"""
Trace Repository for LeanDojo Reinforcement Learning Interface
"""

from lean_dojo import LeanGitRepo, trace
from lean_dojo.data_extraction.trace import is_available_in_cache


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

    # Check if already cached
    if is_available_in_cache(repo):
        print("2. Loading traced repository from cache...")
        traced_repo = trace(repo, dst_dir=None, build_deps=True)
    else:
        print("2. Tracing repository (this will take a while)...")
        traced_repo = trace(repo, dst_dir=None, build_deps=True)

    print("\n3. Trace completed!")
    print("=" * 40)


if __name__ == "__main__":
    try:
        trace_repo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
