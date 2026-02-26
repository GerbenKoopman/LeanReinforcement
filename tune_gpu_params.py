#!/usr/bin/env python3
"""
GPU memory auto-tuner for lean_reinforcement.

Determines the largest (batch_size, num_tactics_to_expand) parameters that
fit in VRAM for both guided_rollout and alpha_zero, WITHOUT requiring
LeanDojo or a Lean installation.  Saves results to gpu_params.json.

How it works
────────────
The peak GPU memory comes from the Transformer's beam-search generation:

    model.generate(..., num_beams=num_tactics, num_return_sequences=num_tactics)

called on `batch_size` states simultaneously (via _expand_batch).

For alpha_zero, the value head forward pass adds a small constant overhead.

The script:
  1. Loads the ByT5 model once.
  2. Iterates through increasing (batch_size, num_tactics_to_expand) combos.
  3. Runs the actual generate call, catches CUDA OOM, and records the
     largest surviving configuration.
  4. Saves the results as gpu_params.json for later use by training scripts.

Usage
─────
    python tune_gpu_params.py                     # auto-detect GPU
    python tune_gpu_params.py --model kaiyuy/leandojo-lean4-tacgen-byt5-small
    python tune_gpu_params.py --reserve-gb 1.5    # leave 1.5 GB headroom
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

# ── Parameter grid ──────────────────────────────────────────────────────────

# Ordered so that the product (batch_size * num_tactics) increases gradually.
# Each entry is (batch_size, num_tactics_to_expand).
PARAM_LEVELS = [
    # Minimal
    (1, 4),
    (1, 8),
    (1, 16),
    (2, 8),
    (1, 32),
    (2, 16),
    (4, 8),
    (2, 32),
    (4, 16),
    (8, 8),
    (1, 64),
    (4, 32),
    (8, 16),
    (2, 64),
    (8, 32),
    (16, 16),
    (4, 64),
    (16, 32),
    (8, 64),
    (32, 16),
    (32, 32),
    (16, 64),
    (32, 64),
]

# Representative tactic-state strings (realistic lengths from mathlib).
# Using multiple to simulate a real batch.
SAMPLE_STATES = [
    "n m : ℕ\nh : n ≤ m\n⊢ n + (m - n) = m",
    "α : Type u_1\ninst✝ : DecidableEq α\nl : List α\na : α\n⊢ a ∈ l.dedup ↔ a ∈ l",
    "R : Type u_1\ninst✝ : CommRing R\np q : R[X]\n⊢ (p * q).leadingCoeff = p.leadingCoeff * q.leadingCoeff",
    "X : Type u_1\ninst✝ : TopologicalSpace X\nf : X → ℝ\nhf : Continuous f\n⊢ IsOpen {x | f x > 0}",
    "G : Type u_1\ninst✝ : Group G\ng h : G\n⊢ (g * h)⁻¹ = h⁻¹ * g⁻¹",
    "V : Type u_1\ninst✝¹ : AddCommGroup V\ninst✝ : Module ℝ V\ns : Set V\n⊢ Convex ℝ (convexHull ℝ s)",
    "n : ℕ\n⊢ 0 < n ! ",
    "a b c : ℤ\nh1 : a ∣ b\nh2 : a ∣ c\n⊢ a ∣ b + c",
]


@dataclass
class TuneResult:
    """Result of GPU auto-tuning for one algorithm."""

    algorithm: str
    batch_size: int
    num_tactics_to_expand: int
    peak_vram_gb: float
    total_vram_gb: float
    product: int  # batch_size * num_tactics_to_expand


def get_gpu_info() -> dict:
    """Return basic GPU info."""
    if not torch.cuda.is_available():
        return {"name": "CPU-only", "total_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_gb": round(props.total_memory / 1024**3, 2),
        "compute_capability": f"{props.major}.{props.minor}",
    }


def clear_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_generate(
    model,
    tokenizer,
    device: str,
    batch_size: int,
    num_tactics: int,
) -> Optional[float]:
    """
    Run a realistic beam-search generation and return peak VRAM (GB).
    Returns None on OOM.
    """
    clear_gpu()
    states = (SAMPLE_STATES * ((batch_size // len(SAMPLE_STATES)) + 1))[:batch_size]

    try:
        torch.cuda.reset_peak_memory_stats()
        tokenized = tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        input_length = tokenized.input_ids.shape[1]
        with torch.no_grad():
            _ = model.generate(
                tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                max_length=input_length + 512,
                num_beams=num_tactics,
                do_sample=False,
                num_return_sequences=num_tactics,
                early_stopping=False,
                length_penalty=0.0,
            )

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        del tokenized
        clear_gpu()
        return round(peak_gb, 3)

    except torch.cuda.OutOfMemoryError:
        clear_gpu()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_gpu()
            return None
        raise


def test_value_head(
    model,
    tokenizer,
    device: str,
    batch_size: int,
    num_tactics: int,
    hidden_dims: list[int],
) -> Optional[float]:
    """
    Test generate + value head forward pass (alpha_zero overhead).
    Returns peak VRAM or None on OOM.
    """
    clear_gpu()

    # First do the generate (same as guided_rollout)
    gen_peak = test_generate(model, tokenizer, device, batch_size, num_tactics)
    if gen_peak is None:
        return None

    # Then simulate value head forward on encoder outputs
    clear_gpu()
    try:
        torch.cuda.reset_peak_memory_stats()
        states = (SAMPLE_STATES * ((batch_size // len(SAMPLE_STATES)) + 1))[
            :batch_size
        ]
        tokenized = tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            encoder_out = model.encoder(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
            )
            # Pool encoder features (mean over sequence dim)
            features = encoder_out.last_hidden_state.mean(dim=1)  # (batch, hidden)
            hidden_size = features.shape[-1]

            # Simulate MLP forward with the configured hidden dims
            x = features
            in_dim = hidden_size
            for h_dim in hidden_dims:
                w = torch.randn(in_dim, h_dim, device=device)
                x = torch.relu(x @ w)
                in_dim = h_dim
            w_out = torch.randn(in_dim, 1, device=device)
            _ = torch.tanh(x @ w_out)

        value_peak = torch.cuda.max_memory_allocated() / 1024**3
        del tokenized, encoder_out, features
        clear_gpu()

        # Peak is the max of generate peak and value head peak
        return round(max(gen_peak, value_peak), 3)

    except torch.cuda.OutOfMemoryError:
        clear_gpu()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_gpu()
            return None
        raise


def run_tuning(
    model_name: str,
    reserve_gb: float,
    value_head_hidden_dims: list[int],
) -> dict:
    """Run the full auto-tuning process. Returns the results dict."""
    gpu_info = get_gpu_info()
    if gpu_info["total_gb"] == 0:
        print("ERROR: No CUDA GPU detected. Cannot tune GPU parameters.")
        sys.exit(1)

    total_gb = gpu_info["total_gb"]
    usable_gb = total_gb - reserve_gb
    print(f"GPU: {gpu_info['name']} ({total_gb:.1f} GB total, {usable_gb:.1f} GB usable)")
    print(f"Model: {model_name}")
    print(f"Reserve: {reserve_gb:.1f} GB headroom")
    print()

    # Load model once
    print("Loading model... ", end="", flush=True)
    device = "cuda"
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(model_name)
    model = __import__("transformers").AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    ).to(device)
    model_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"done ({model_gb:.2f} GB)")
    print()

    results: dict = {
        "gpu": gpu_info,
        "model_name": model_name,
        "model_size_gb": round(model_gb, 3),
        "reserve_gb": reserve_gb,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "guided_rollout": None,
        "alpha_zero": None,
    }

    # ── Guided Rollout tuning ───────────────────────────────────────────
    print("=" * 60)
    print("Tuning: guided_rollout")
    print("=" * 60)
    best_gr: Optional[TuneResult] = None

    for batch_size, num_tactics in PARAM_LEVELS:
        product = batch_size * num_tactics
        label = f"  batch={batch_size:>2}, tactics={num_tactics:>2} (product={product:>4})"

        peak = test_generate(model, tokenizer, device, batch_size, num_tactics)
        if peak is None:
            print(f"{label} → OOM ✗")
            # If even the smallest config OOMs, keep trying (might be transient)
            # But if we already have a result, stop (configs are ordered by size)
            if best_gr is not None:
                print("  Stopping: next level exceeds VRAM")
                break
            continue

        if peak > usable_gb:
            print(f"{label} → {peak:.2f} GB (exceeds {usable_gb:.1f} GB limit) ✗")
            if best_gr is not None:
                break
            continue

        print(f"{label} → {peak:.2f} GB ✓")
        best_gr = TuneResult(
            algorithm="guided_rollout",
            batch_size=batch_size,
            num_tactics_to_expand=num_tactics,
            peak_vram_gb=peak,
            total_vram_gb=total_gb,
            product=product,
        )

    if best_gr:
        results["guided_rollout"] = asdict(best_gr)
        print(
            f"\n  Best: batch_size={best_gr.batch_size}, "
            f"num_tactics={best_gr.num_tactics_to_expand} "
            f"(peak {best_gr.peak_vram_gb:.2f} GB)\n"
        )
    else:
        print("\n  WARNING: No configuration fit in VRAM!\n")

    # ── AlphaZero tuning ────────────────────────────────────────────────
    print("=" * 60)
    print("Tuning: alpha_zero")
    print("=" * 60)
    best_az: Optional[TuneResult] = None

    for batch_size, num_tactics in PARAM_LEVELS:
        product = batch_size * num_tactics
        label = f"  batch={batch_size:>2}, tactics={num_tactics:>2} (product={product:>4})"

        peak = test_value_head(
            model, tokenizer, device, batch_size, num_tactics, value_head_hidden_dims
        )
        if peak is None:
            print(f"{label} → OOM ✗")
            if best_az is not None:
                print("  Stopping: next level exceeds VRAM")
                break
            continue

        if peak > usable_gb:
            print(f"{label} → {peak:.2f} GB (exceeds {usable_gb:.1f} GB limit) ✗")
            if best_az is not None:
                break
            continue

        print(f"{label} → {peak:.2f} GB ✓")
        best_az = TuneResult(
            algorithm="alpha_zero",
            batch_size=batch_size,
            num_tactics_to_expand=num_tactics,
            peak_vram_gb=peak,
            total_vram_gb=total_gb,
            product=product,
        )

    if best_az:
        results["alpha_zero"] = asdict(best_az)
        print(
            f"\n  Best: batch_size={best_az.batch_size}, "
            f"num_tactics={best_az.num_tactics_to_expand} "
            f"(peak {best_az.peak_vram_gb:.2f} GB)\n"
        )
    else:
        print("\n  WARNING: No configuration fit in VRAM!\n")

    # Cleanup
    del model, tokenizer
    clear_gpu()

    return results


def print_cli_args(results: dict) -> None:
    """Print CLI flags for easy copy-paste."""
    print("=" * 60)
    print("Recommended CLI arguments")
    print("=" * 60)

    for algo in ("guided_rollout", "alpha_zero"):
        params = results.get(algo)
        if params is None:
            print(f"\n  {algo}: no viable configuration found")
            continue

        print(f"\n  {algo}:")
        print(
            f"    --mcts-type {algo} "
            f"--batch-size {params['batch_size']} "
            f"--num-tactics-to-expand {params['num_tactics_to_expand']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune GPU parameters for lean_reinforcement"
    )
    parser.add_argument(
        "--model",
        default="kaiyuy/leandojo-lean4-tacgen-byt5-small",
        help="HuggingFace model name (default: ByT5-small)",
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=1.0,
        help="GB of VRAM to keep free as headroom (default: 1.0)",
    )
    parser.add_argument(
        "--value-head-hidden-dims",
        type=int,
        nargs="*",
        default=[1024, 512, 256, 128, 64],
        help="Hidden dims for value head MLP (matches training config)",
    )
    parser.add_argument(
        "--output",
        default="gpu_params.json",
        help="Output file for tuned parameters (default: gpu_params.json)",
    )
    args = parser.parse_args()

    results = run_tuning(
        model_name=args.model,
        reserve_gb=args.reserve_gb,
        value_head_hidden_dims=args.value_head_hidden_dims,
    )

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print_cli_args(results)

    # Print usage hint
    print()
    print("To use these params in training:")
    print(
        "  python -c \"import json; p=json.load(open('"
        + str(output_path)
        + "'))['guided_rollout']; "
        "print(f\\\"--batch-size {p['batch_size']} "
        "--num-tactics-to-expand {p['num_tactics_to_expand']}\\\")\""
    )
    print()
    print("Or just run training — gpu_params.json is loaded automatically.")


if __name__ == "__main__":
    main()
