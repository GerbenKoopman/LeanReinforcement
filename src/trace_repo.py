"""
Trace Repository for LeanDojo Reinforcement Learning Interface
"""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SCRATCH_SHARED = os.getenv("SCRATCH_SHARED")

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lean_dojo import LeanGitRepo, trace
from lean_dojo.data_extraction.trace import check_files


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

    scratch_dir = SCRATCH_SHARED
    if scratch_dir is not None:
        scratch_dir = scratch_dir + "/test_traced_repo"
    destination_dir = (
        Path(scratch_dir)
        if scratch_dir
        else Path("/scratch-shared/lean-reinforcement/traced_repo")
    )

    print(f"Destination directory: {destination_dir}")

    print("2. Tracing repository...")
    traced_repo = trace(repo, dst_dir=destination_dir, build_deps=True)

    print("3. Checking trace...")
    check_files(destination_dir, no_deps=False)

    print("\n6. Trace completed!")
    print("=" * 40)


if __name__ == "__main__":
    try:
        trace_repo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
