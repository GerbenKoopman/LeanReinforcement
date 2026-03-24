"""Diagnostics for LeanDojo cache/toolchain compatibility.

This utility prints environment and cache metadata that commonly cause
trace/runtime mismatches on shared HPC systems.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path
from typing import Optional


def _read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        return proc.stdout.strip() or "<no output>"
    except FileNotFoundError:
        return "<not found>"
    except Exception as exc:
        return f"<failed: {exc}>"


def _find_cache_repo_dir(cache_dir: Path, commit: str) -> Optional[Path]:
    direct = cache_dir / f"leanprover-community-mathlib4-{commit}"
    if direct.exists() and direct.is_dir():
        return direct

    patterns = [
        str(cache_dir / f"*mathlib4*{commit}*"),
        str(cache_dir / f"*mathlib4*{commit[:10]}*"),
    ]
    matches: list[Path] = []
    for pattern in patterns:
        for m in glob.glob(pattern):
            p = Path(m)
            if p.is_dir() and p not in matches:
                matches.append(p)

    return matches[0] if matches else None


def _find_lean4repl(cache_repo_dir: Path) -> list[Path]:
    hits: list[Path] = []
    for rel in [
        "Lean4Repl.lean",
        "mathlib4/Lean4Repl.lean",
        "mathlib4/Mathlib/Lean4Repl.lean",
    ]:
        p = cache_repo_dir / rel
        if p.exists():
            hits.append(p)

    # Fallback recursive search (still bounded by cache repo dir).
    if not hits:
        for p in cache_repo_dir.rglob("Lean4Repl.lean"):
            if p.is_file():
                hits.append(p)
                if len(hits) >= 5:
                    break
    return hits


def _pick_cache_dir(cli_cache_dir: Optional[str]) -> Optional[Path]:
    candidates = [
        cli_cache_dir,
        os.environ.get("LEANDOJO_CACHE_DIR"),
        os.environ.get("CACHE_DIR"),
    ]
    for c in candidates:
        if c:
            return Path(c).expanduser().resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print LeanDojo/Lean/cache diagnostics for mismatch debugging."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache root. Defaults: LEANDOJO_CACHE_DIR, then CACHE_DIR.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="29dcec074de168ac2bf835a77ef68bbe069194c5",
        help="mathlib commit expected in the traced cache.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root containing lean-toolchain.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    cache_dir = _pick_cache_dir(args.cache_dir)

    print("=== Lean Cache Diagnostics ===")
    print(f"project_root={project_root}")
    print(f"python={_run_cmd(['python3', '--version'])}")
    print(
        f"python_executable={_run_cmd(['python3', '-c', 'import sys; print(sys.executable)'])}"
    )

    # LeanDojo package and version details
    lean_dojo_ver = "<unavailable>"
    lean_dojo_path = "<unavailable>"
    try:
        import lean_dojo  # type: ignore

        lean_dojo_ver = getattr(lean_dojo, "__version__", "<unknown>")
        lean_dojo_path = str(Path(lean_dojo.__file__).resolve())
    except Exception as exc:
        lean_dojo_ver = f"<import failed: {exc}>"

    print(f"lean_dojo_version={lean_dojo_ver}")
    print(f"lean_dojo_module_path={lean_dojo_path}")

    # Active Lean toolchain/binaries in environment
    print(f"lean_version={_run_cmd(['lean', '--version'])}")
    print(f"lake_version={_run_cmd(['lake', '--version'])}")
    print(f"which_lean={_run_cmd(['which', 'lean'])}")
    print(f"which_lake={_run_cmd(['which', 'lake'])}")

    project_toolchain = _read_text_if_exists(project_root / "lean-toolchain")
    print(f"project_lean_toolchain={project_toolchain or '<missing>'}")

    env_keys = [
        "LEANDOJO_OFFLINE",
        "LEANDOJO_CACHE_DIR",
        "CACHE_DIR",
        "DISABLE_REMOTE_CACHE",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "LOADEDMODULES",
        "PYTHONPATH",
        "PATH",
    ]
    print("--- Environment snapshot ---")
    for key in env_keys:
        val = os.environ.get(key)
        if key == "PATH" and val:
            parts = val.split(":")
            print(f"{key}={':'.join(parts[:6])}{' ...' if len(parts) > 6 else ''}")
        elif key == "LOADEDMODULES" and val:
            mods = val.split(":")
            print(f"{key}={mods[:8]}{' ...' if len(mods) > 8 else ''}")
        else:
            print(f"{key}={val if val is not None else '<unset>'}")

    # Module + conda mix warning signal.
    loaded_modules = os.environ.get("LOADEDMODULES", "")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if loaded_modules and conda_prefix:
        print(
            "potential_clash=modules_plus_conda_active "
            "(Snellius warns this can corrupt environments)"
        )

    if cache_dir is None:
        print("cache_dir=<unset> (set LEANDOJO_CACHE_DIR or CACHE_DIR)")
        print("=== End Diagnostics ===")
        return 0

    print(f"cache_dir={cache_dir}")
    if not cache_dir.exists():
        print("cache_status=missing")
        print("=== End Diagnostics ===")
        return 0

    cache_repo_dir = _find_cache_repo_dir(cache_dir, args.commit)
    print(f"expected_mathlib_commit={args.commit}")
    print(f"cache_repo_dir={cache_repo_dir if cache_repo_dir else '<not found>'}")

    if cache_repo_dir is None:
        print("cache_status=repo_for_commit_not_found")
        print("=== End Diagnostics ===")
        return 0

    cache_toolchain = _read_text_if_exists(
        cache_repo_dir / "mathlib4" / "lean-toolchain"
    )
    print(f"cache_lean_toolchain={cache_toolchain or '<missing>'}")

    if project_toolchain and cache_toolchain and project_toolchain != cache_toolchain:
        print("potential_clash=toolchain_mismatch_between_project_and_cache")

    repl_hits = _find_lean4repl(cache_repo_dir)
    print(f"lean4repl_present={'yes' if repl_hits else 'no'}")
    for idx, p in enumerate(repl_hits[:3], start=1):
        print(f"lean4repl_path_{idx}={p}")

    # Basic artifact signal for partial traces.
    ast_count = len(list((cache_repo_dir / "mathlib4").rglob("*.ast.json")))
    print(f"ast_file_count={ast_count}")
    if ast_count < 5000:
        print("potential_clash=incomplete_trace_artifacts_low_ast_count")

    # Detect multiple cache dirs for same commit (can hide stale data issues).
    siblings = [
        p
        for p in cache_dir.glob("*mathlib4*")
        if p.is_dir() and args.commit[:10] in p.name
    ]
    if len(siblings) > 1:
        print(f"potential_clash=multiple_cache_dirs_for_commit count={len(siblings)}")
        for idx, p in enumerate(siblings[:5], start=1):
            print(f"cache_candidate_{idx}={p}")

    print("=== End Diagnostics ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
