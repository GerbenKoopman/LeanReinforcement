#!/usr/bin/env python3
"""Debug script to test theorem loading."""

import os
import traceback

# Set environment variables first
os.environ["LEAN_CACHE_ONLY"] = "1"
os.environ["DISABLE_BUILD_DEPS"] = "1"
os.environ["LOAD_USED_PACKAGES_ONLY"] = "1"

from lean_dojo import LeanGitRepo
from lean_dojo.data_extraction.trace import (
    is_available_in_cache,
    get_traced_repo_path,
)
from lean_dojo.data_extraction.traced_data import TracedRepo


def debug_theorem_loading():
    """Debug the theorem loading process."""
    print("Debugging theorem loading...")

    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "29dcec074de168ac2bf835a77ef68bbe069194c5",
    )

    # Check cache availability
    print(f"Repository available in cache: {is_available_in_cache(repo)}")

    try:
        # Get cached path
        cached_path = get_traced_repo_path(repo, build_deps=False)
        print(f"Cached path: {cached_path}")

        # Load traced repo
        print("Loading traced repo with build_deps=False (cache-only mode)...")
        try:
            traced_repo = TracedRepo.load_from_disk(cached_path, build_deps=False)
            print("✓ Successfully loaded with build_deps=False")
        except Exception as e:
            print(f"✗ Failed with build_deps=False: {e}")
            print("This indicates cache corruption or missing files")
            raise

        # Test specific known good files
        test_files = [
            "Mathlib/Algebra/BigOperators/Pi.lean",  # Used in working examples
            "Mathlib/Data/Nat/Basic.lean",
            "Mathlib/Logic/Basic.lean",
        ]

        for file_path in test_files:
            print(f"\n--- Testing {file_path} ---")
            try:
                traced_file = traced_repo.get_traced_file(file_path)
                if traced_file is None:
                    print(f"✗ traced_file is None")
                    continue

                print(f"✓ Got traced file: {type(traced_file)}")

                # Try to get theorems
                theorems = traced_file.get_traced_theorems()
                if theorems is None:
                    print(f"✗ get_traced_theorems() returned None")
                elif len(theorems) == 0:
                    print(f"✗ get_traced_theorems() returned empty list")
                else:
                    print(f"✓ Found {len(theorems)} theorems")
                    # Print first theorem info
                    if len(theorems) > 0:
                        first_theorem = theorems[0]
                        print(f"  First theorem: {first_theorem.theorem.full_name}")
                        print(f"  Type: {type(first_theorem)}")

            except Exception as e:
                print(f"✗ Error with {file_path}: {e}")
                print(f"  Full traceback: {traceback.format_exc()}")

        # List some available files
        print(f"\n--- Available traced files (first 10) ---")
        try:
            traced_files = list(traced_repo.traced_files)
            print(f"Total traced files: {len(traced_files)}")
            for i, tf in enumerate(traced_files[:10]):
                try:
                    print(f"  {i}: {tf.path}")
                except Exception as e:
                    print(f"  {i}: Error getting path: {e}")
        except Exception as e:
            print(f"Error listing traced files: {e}")

    except Exception as e:
        print(f"Fatal error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    debug_theorem_loading()
