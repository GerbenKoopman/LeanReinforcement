#!/usr/bin/env python3
"""Validate that the LeanDojo cache is properly configured for HPC use."""

import os
import sys
from pathlib import Path
from lean_dojo import LeanGitRepo
from lean_dojo.data_extraction.trace import is_available_in_cache, get_traced_repo_path


def main():
    # Load environment variables
    cache_dir = os.getenv("CACHE_DIR")
    if not cache_dir:
        print("ERROR: CACHE_DIR environment variable not set")
        sys.exit(1)

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"ERROR: Cache directory does not exist: {cache_dir}")
        sys.exit(1)

    print(f"✓ Cache directory exists: {cache_dir}")

    # Check for mathlib4 repository
    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "29dcec074de168ac2bf835a77ef68bbe069194c5",
    )

    if is_available_in_cache(repo):
        traced_path = get_traced_repo_path(repo)
        print(f"✓ Mathlib4 repository found in cache: {traced_path}")

        # Check key directories and files
        repo_root = Path(traced_path)
        mathlib_dir = repo_root / "Mathlib"

        if mathlib_dir.exists():
            print(f"✓ Mathlib directory exists")
        else:
            print(f"⚠ Mathlib directory not found at: {mathlib_dir}")

        # Check for essential files in the repository root
        essential_files = ["Mathlib.lean", "lakefile.lean", "lean-toolchain"]
        for file in essential_files:
            file_path = repo_root / file
            if file_path.exists():
                print(f"✓ Essential file found: {file}")
            else:
                print(f"⚠ Missing file: {file}")

        # Check some key subdirectories
        key_subdirs = ["Mathlib", "Archive", "Counterexamples"]
        for subdir in key_subdirs:
            subdir_path = repo_root / subdir
            if subdir_path.exists():
                print(f"✓ Key directory found: {subdir}")
            else:
                print(f"⚠ Missing directory: {subdir}")

        # Check directory sizes
        try:
            repo_size = sum(
                f.stat().st_size for f in repo_root.rglob("*") if f.is_file()
            )
            print(f"✓ Repository size: {repo_size / (1024**3):.2f} GB")
        except Exception as e:
            print(f"⚠ Could not determine repository size: {e}")

    else:
        print("ERROR: Mathlib4 repository not available in cache")
        print("Run trace_repo.py to generate the cache first")
        sys.exit(1)

    # Check SCRATCH_SHARED directory structure
    scratch_dir = os.getenv("SCRATCH_SHARED")
    if scratch_dir:
        scratch_path = Path(scratch_dir)
        if scratch_path.exists():
            print(f"✓ SCRATCH_SHARED directory exists: {scratch_dir}")

            # Check required subdirectories
            required_dirs = [
                "checkpoints",
                "evaluation_results",
                "evaluation_plots",
                "tensorboard_logs",
                "training_logs",
                "saved_models",
                "experiments",
                "test_reports",
            ]

            for dir_name in required_dirs:
                dir_path = scratch_path / dir_name
                if dir_path.exists():
                    print(f"✓ Required directory exists: {dir_name}")
                else:
                    print(f"⚠ Creating missing directory: {dir_name}")
                    dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"⚠ SCRATCH_SHARED directory does not exist: {scratch_dir}")
    else:
        print("⚠ SCRATCH_SHARED environment variable not set")

    print("\n✓ Cache validation successful!")


if __name__ == "__main__":
    main()
