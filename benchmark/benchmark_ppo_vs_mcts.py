#!/usr/bin/env python3
"""
Benchmark: LeanReinforcement vs Euclidean/Hyperbolic MCTS vs Euclidean/Hyperbolic PPO.

This script runs a controlled four-way comparison across:

  1. **Euclidean MCTS** — AlphaZero MCTS with standard ``ValueHead`` (MLP critic).
  2. **Hyperbolic MCTS** — AlphaZero MCTS with ``HyperbolicValueHead``
     (Poincaré ball critic).
  3. **Hyperbolic PPO**  — Frozen ByT5 + LoRA actor + ``PoincareCritic``
     (Poincare-ball manifold + categorical value head).
  4. **Euclidean PPO**   — Frozen ByT5 + LoRA actor + ``EuclideanCritic``
     (Euclidean manifold + categorical value head).

Training and Evaluation:
  - Training runs on the **train split** using self-play MCTS data collection
  - After each epoch, training data is used to train the value head
  - Optionally, with ``--run-test-eval``, the trained model is evaluated on
    the **test split** after training completes

Usage::

    # Full benchmark (all 4 methods × 2 seeds × 3 sizes)
    python -m benchmark.benchmark_ppo_vs_mcts

    # Only specific methods
    python -m benchmark.benchmark_ppo_vs_mcts --methods euclidean_mcts euclidean_ppo

    # With test evaluation after training
    python -m benchmark.benchmark_ppo_vs_mcts --run-test-eval

    # Custom test split and number of theorems
    python -m benchmark.benchmark_ppo_vs_mcts --run-test-eval --test-split val --test-num-theorems 100

    # Status check
    python -m benchmark.benchmark_ppo_vs_mcts --status

    # Plot results
    python -m benchmark.benchmark_ppo_vs_mcts --plot-only

    # Fast benchmark (all 4 methods, only seed 42 and light size)
    python -m benchmark.benchmark_ppo_vs_mcts --fast
"""

from __future__ import annotations

import os
import argparse
import gc
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, TypedDict, cast

import wandb
from dotenv import load_dotenv
from loguru import logger

import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.onnx_transformer import (
    ONNXTransformer,
    is_onnx_available,
)
from lean_reinforcement.agent.ppo import HyperbolicPPO, EuclideanPPO
from lean_reinforcement.agent.value_head import ValueHead, HyperbolicValueHead
from lean_reinforcement.utilities.checkpoint import load_checkpoint
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    empty_gpu_cache,
    log_gpu_memory,
)
from lean_reinforcement.training.trainer import Trainer
from lean_reinforcement.training.inference_server import InferenceServer
from lean_reinforcement.training.progress import (
    ProgressStats,
    make_progress_display,
)
from benchmark.run_benchmark import (
    BASE_PARAMS,
    SIZE_CONFIGS,
    NUM_EPOCHS,
)

from benchmark.evaluate_benchmark import (
    build_eval_config,
    load_or_init_corpus,
    TestEvaluator,
)

# Set allocator config early so PyTorch can pick it up before CUDA init.
_alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "")
if "expandable_segments:True" not in _alloc_conf:
    os.environ["PYTORCH_ALLOC_CONF"] = (
        f"{_alloc_conf},expandable_segments:True"
        if _alloc_conf
        else "expandable_segments:True"
    )

load_dotenv()


def _post_run_memory_cleanup(run_name: str) -> None:
    """Best-effort cleanup to avoid VRAM accumulation across benchmark runs."""
    gc.collect()
    aggressive_cleanup()
    empty_gpu_cache()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
    except Exception:
        pass
    log_gpu_memory(prefix=f"After cleanup ({run_name}) - ")


# ── Four benchmark methods ──────────────────────────────────────────────────


class MethodSettings(TypedDict):
    mcts_type: str
    use_hyperbolic: bool
    is_ppo: bool


METHODS = ["euclidean_mcts", "hyperbolic_mcts", "hyperbolic_ppo", "euclidean_ppo"]
SEEDS = [42, 43]
SIZES = ["light", "medium", "heavy"]
FAST_SEEDS = [42]
FAST_SIZES = ["light"]
FAST_NUM_EPOCHS = 2
FAST_TRAIN_EPOCHS = 5

# Method → (mcts_type, use_hyperbolic, is_ppo)
METHOD_CONFIG: Dict[str, MethodSettings] = {
    "euclidean_mcts": {
        "mcts_type": "alpha_zero",
        "use_hyperbolic": False,
        "is_ppo": False,
    },
    "hyperbolic_mcts": {
        "mcts_type": "alpha_zero",
        "use_hyperbolic": True,
        "is_ppo": False,
    },
    "hyperbolic_ppo": {
        "mcts_type": "alpha_zero",
        "use_hyperbolic": True,
        "is_ppo": True,
    },
    "euclidean_ppo": {
        "mcts_type": "alpha_zero",
        "use_hyperbolic": False,
        "is_ppo": True,
    },
}

# Display labels for plots and logs
METHOD_LABELS = {
    "euclidean_mcts": "Euclidean MCTS (ValueHead)",
    "hyperbolic_mcts": "Hyperbolic MCTS (Poincaré)",
    "hyperbolic_ppo": "Hyperbolic PPO (Poincare + LoRA)",
    "euclidean_ppo": "Euclidean PPO (MLP + LoRA)",
}

METHOD_COLORS = {
    "euclidean_mcts": "tab:blue",
    "hyperbolic_mcts": "tab:orange",
    "hyperbolic_ppo": "tab:green",
    "euclidean_ppo": "tab:red",
}


# ── Directory helpers ────────────────────────────────────────────────────────


def get_benchmark_dir() -> Path:
    """Get or create the four-way benchmark cache directory."""
    checkpoint_dir = os.getenv("CHECKPOINT_DIR")
    if checkpoint_dir:
        base = Path(checkpoint_dir) / "benchmark_4way"
    else:
        base = Path("checkpoints") / "benchmark_4way"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_run_dir(benchmark_dir: Path, method: str, seed: int, size: str) -> Path:
    return benchmark_dir / f"{method}_{seed}_{size}"


def is_run_complete(run_dir: Path, num_epochs: int) -> bool:
    marker = run_dir / "benchmark_complete.json"
    if marker.exists():
        try:
            with open(marker) as f:
                marker_data = json.load(f)
            marker_epochs = int(marker_data.get("num_epochs", -1))
            if marker_epochs == num_epochs:
                return True
            logger.warning(
                f"Ignoring stale completion marker in {run_dir.name}: "
                f"marker num_epochs={marker_epochs}, requested={num_epochs}"
            )
        except Exception:
            logger.warning(
                f"Ignoring unreadable completion marker in {run_dir.name}: {marker}"
            )
    # Check for final epoch checkpoint
    for prefix in ["value_head_alpha_zero", "ppo_actor", "ppo_critic"]:
        final_ckpt = run_dir / f"{prefix}_epoch_{num_epochs}.pth"
        if final_ckpt.exists():
            return True
    return False


def purge_run_artifacts(run_dir: Path) -> None:
    """Remove stale artifacts so a forced fresh run truly starts clean."""
    if not run_dir.exists():
        return

    patterns = [
        "benchmark_complete.json",
        "benchmark_config.json",
        "benchmark_progress.json",
        "theorem_results_epoch_*.json",
        "training_data_epoch_*.json",
        "temp_data_epoch_*.jsonl",
        "val_loss_epoch_*.json",
        "value_head_*_epoch_*.pth",
        "value_head_*_latest.pth",
        "ppo_*_epoch_*.pth",
        "ppo_*_latest.pth",
    ]

    removed = 0
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1

    eval_dir = run_dir / "eval"
    if eval_dir.exists() and eval_dir.is_dir():
        shutil.rmtree(eval_dir)

    logger.info(f"Purged {removed} stale artifacts in {run_dir}")


def get_completed_epochs(run_dir: Path) -> int:
    if not run_dir.exists():
        return 0
    max_epoch = 0
    for f in run_dir.glob("theorem_results_epoch_*.json"):
        try:
            epoch_num = int(f.stem.split("_")[-1])
            max_epoch = max(max_epoch, epoch_num)
        except ValueError:
            continue
    for f in run_dir.glob("value_head_*_epoch_*.pth"):
        try:
            epoch_num = int(f.stem.split("_")[-1])
            max_epoch = max(max_epoch, epoch_num)
        except ValueError:
            continue
    for f in run_dir.glob("ppo_*_epoch_*.pth"):
        try:
            epoch_num = int(f.stem.split("_")[-1])
            max_epoch = max(max_epoch, epoch_num)
        except ValueError:
            continue
    return max_epoch


# ── Config builders ──────────────────────────────────────────────────────────


def build_config(
    method: str,
    seed: int,
    size: str,
    run_dir: Path,
    num_epochs: int = NUM_EPOCHS,
    resume: bool = False,
    train_epochs_override: int | None = None,
) -> TrainingConfig:
    """Build a TrainingConfig for a specific benchmark run."""
    size_params = cast(Dict[str, float | int], SIZE_CONFIGS[size])
    mcfg = METHOD_CONFIG[method]

    start_from = 0
    if resume and run_dir.exists():
        start_from = get_completed_epochs(run_dir)

    remaining_epochs = max(num_epochs - start_from, 0)

    data_type_val = str(BASE_PARAMS["data_type"])
    num_theorems_val = int(cast(int, BASE_PARAMS["num_theorems"]))
    max_steps_val = int(cast(int, BASE_PARAMS["max_steps"]))
    batch_size_val = int(cast(int, BASE_PARAMS["batch_size"]))
    num_workers_val = int(cast(int, BASE_PARAMS["num_workers"]))
    # Default to 50 value-head epochs unless overridden (e.g. fast smoke mode)
    train_epochs_val = (
        train_epochs_override if train_epochs_override is not None else 50
    )
    value_head_batch_size_val = int(cast(int, BASE_PARAMS["value_head_batch_size"]))
    value_head_latent_dim_val = int(cast(int, BASE_PARAMS["value_head_latent_dim"]))
    # Always train value head for the four-way benchmark so checkpoints exist
    # for resuming and post-training evaluation.
    train_value_head_val = True
    use_final_reward_val = bool(cast(bool, BASE_PARAMS["use_final_reward"]))
    save_training_data_val = bool(cast(bool, BASE_PARAMS["save_training_data"]))
    use_caching_val = bool(cast(bool, BASE_PARAMS["use_caching"]))
    inference_timeout_val = float(cast(float, BASE_PARAMS["inference_timeout"]))
    model_name_val = str(BASE_PARAMS["model_name"])
    num_tactics_to_expand_val = int(cast(int, BASE_PARAMS["num_tactics_to_expand"]))
    max_rollout_depth_val = int(cast(int, BASE_PARAMS["max_rollout_depth"]))

    num_iterations_val = int(cast(int, size_params["num_iterations"]))
    max_time_val = float(cast(float, size_params["max_time"]))
    env_timeout_val = int(cast(int, size_params["env_timeout"]))
    proof_timeout_val = float(cast(float, size_params["proof_timeout"]))

    config = TrainingConfig(
        # Data and MCTS
        data_type=data_type_val,
        num_epochs=remaining_epochs,
        num_theorems=num_theorems_val,
        num_iterations=num_iterations_val,
        max_steps=max_steps_val,
        batch_size=batch_size_val,
        num_workers=num_workers_val,
        mcts_type=mcfg["mcts_type"],
        indexed_corpus_path=cast(
            str | None,
            BASE_PARAMS.get("indexed_corpus_path"),
        ),
        # Training
        train_epochs=train_epochs_val,
        value_head_batch_size=value_head_batch_size_val,
        value_head_latent_dim=value_head_latent_dim_val,
        train_value_head=train_value_head_val,
        use_hyperbolic=mcfg["use_hyperbolic"],
        use_final_reward=use_final_reward_val,
        save_training_data=save_training_data_val,
        use_caching=use_caching_val,
        # Reproducibility
        seed=seed,
        # Checkpoints
        save_checkpoints=True,
        resume=resume and start_from > 0,
        checkpoint_dir=str(run_dir),
        use_wandb=True,
        # Inference
        inference_timeout=inference_timeout_val,
        # Model
        model_name=model_name_val,
        num_tactics_to_expand=num_tactics_to_expand_val,
        max_rollout_depth=max_rollout_depth_val,
        # Timeouts
        max_time=max_time_val,
        env_timeout=env_timeout_val,
        proof_timeout=proof_timeout_val,
    )
    return config


def mark_run_complete(
    run_dir: Path,
    method: str,
    seed: int,
    size: str,
    elapsed: float,
    num_epochs: int = NUM_EPOCHS,
) -> None:
    marker = run_dir / "benchmark_complete.json"
    with open(marker, "w") as f:
        json.dump(
            {
                "method": method,
                "seed": seed,
                "size": size,
                "num_epochs": num_epochs,
                "elapsed_time_seconds": elapsed,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )


# ── Trainers ─────────────────────────────────────────────────────────────────


class MCTSBenchmarkTrainer(Trainer):
    """Trainer for Euclidean and Hyperbolic MCTS runs (standard pipeline)."""

    def __init__(
        self,
        config: TrainingConfig,
        run_dir: Path,
        start_epoch_override: int = 0,
        method: str = "euclidean_mcts",
    ):
        self.config = config
        self._start_epoch_override = start_epoch_override
        self._method = method

        self.progress_stats = ProgressStats(
            total_epochs=start_epoch_override + config.num_epochs,
            total_workers=config.num_workers,
            cumulative_total_theorems=config.num_epochs * config.num_theorems,
        )
        self.progress_display = make_progress_display(self.progress_stats)

        if config.seed is not None:
            self._set_seeds(config.seed)

        self.checkpoint_dir = run_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_wandb:
            wandb.init(
                entity="gerbennkoopman-university-of-amsterdam",
                project="lean-reinforcement-benchmark-4way",
                name=run_dir.name,
                config=asdict(self.config),
                tags=[method],
                resume="allow",
            )

        self._setup_models()
        self._setup_data()
        self._setup_multiprocessing()

    def _setup_models(self) -> None:
        """Set up transformer + value head (Euclidean or Poincaré)."""
        logger.info(f"[{self._method}] Checkpoint dir: {self.checkpoint_dir}")

        if self.config.use_onnx and is_onnx_available():
            self.transformer = cast(
                Transformer, ONNXTransformer(model_name=self.config.model_name)
            )
        else:
            self.transformer = Transformer(model_name=self.config.model_name)

        self.value_head = None
        self.start_epoch = self._start_epoch_override

        transformer_for_vh = cast(Transformer, self.transformer)
        if self.config.use_hyperbolic:
            logger.info("Using Hyperbolic (Poincaré ball) value head")
            self.value_head = HyperbolicValueHead(transformer_for_vh)
        else:
            logger.info("Using Euclidean (MLP) value head")
            self.value_head = ValueHead(
                transformer_for_vh,
                latent_dim=self.config.value_head_latent_dim,
            )

        if self.config.resume:
            prefix = f"value_head_{self.config.mcts_type}"
            loaded_epoch = load_checkpoint(
                self.value_head,
                self.checkpoint_dir,
                prefix=prefix,
            )
            if loaded_epoch > 0:
                self.start_epoch = loaded_epoch
                logger.info(f"Resuming from epoch {self.start_epoch}")

        log_gpu_memory(prefix="After model init - ")

    def train(self) -> List[Dict[str, Any]]:
        all_metrics: List[Dict[str, Any]] = []
        self.progress_stats.run_start_time = time.time()
        self.progress_display.start()
        try:
            inference_server = InferenceServer(
                self.transformer,
                self.value_head,
                self.request_queue,
                self.response_queues,
                self.config.batch_size,
            )
            for epoch in range(
                self.start_epoch, self.start_epoch + self.config.num_epochs
            ):
                metrics = self._run_epoch(epoch, inference_server)
                all_metrics.extend(metrics)
            return all_metrics
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
            return all_metrics
        finally:
            self.progress_display.stop()
            self._cleanup_workers()
            if self.config.use_wandb:
                wandb.finish()


class PPOBenchmarkTrainer(Trainer):
    """Trainer for the Hyperbolic and Euclidean PPO approaches.

    During data collection the PPO trainer still uses AlphaZero MCTS
    (with a critic-backed value head) to explore the proof
    tree — just like the MCTS baselines. The key difference is that
    after each epoch's data collection, the **decoder policy** is also
    updated via PPO (using LoRA weights), and the critic uses either
    a Euclidean or Poincare-ball manifold with categorical value loss.
    """

    def __init__(
        self,
        config: TrainingConfig,
        run_dir: Path,
        start_epoch_override: int = 0,
    ):
        self.config = config
        self._start_epoch_override = start_epoch_override

        self.progress_stats = ProgressStats(
            total_epochs=start_epoch_override + config.num_epochs,
            total_workers=config.num_workers,
            cumulative_total_theorems=config.num_epochs * config.num_theorems,
        )
        self.progress_display = make_progress_display(self.progress_stats)

        if config.seed is not None:
            self._set_seeds(config.seed)

        self.checkpoint_dir = run_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_wandb:
            wandb.init(
                entity="gerbennkoopman-university-of-amsterdam",
                project="lean-reinforcement-benchmark-4way",
                name=run_dir.name,
                config=asdict(self.config),
                tags=["hyperbolic_ppo" if config.use_hyperbolic else "euclidean_ppo"],
                resume="allow",
            )

        self._setup_models()
        self._setup_data()
        self._setup_multiprocessing()

    def _setup_models(self) -> None:
        """Set up LoRA-wrapped ByT5 + Critic + MCTS value proxy."""
        if self.config.use_hyperbolic:
            logger.info("[hyperbolic_ppo] Setting up LoRA actor + Poincare critic")
        else:
            logger.info("[euclidean_ppo] Setting up LoRA actor + MLP critic")

        # Load the base transformer (used for MCTS proof search by workers)
        if self.config.use_onnx and is_onnx_available():
            self.transformer = cast(
                Transformer, ONNXTransformer(model_name=self.config.model_name)
            )
        else:
            self.transformer = Transformer(model_name=self.config.model_name)

        # Use a generic Any-typed holder for the PPO model to avoid mypy
        # inferring a concrete subtype when we switch implementations.
        self.ppo_model: Any = None
        if self.config.use_hyperbolic:
            self.ppo_model = HyperbolicPPO(model_name=self.config.model_name)
        else:
            self.ppo_model = EuclideanPPO(model_name=self.config.model_name)

        # --- Value head for MCTS data collection ---
        # The MCTS workers in PPO still need a value function for rollouts.
        # This is a standard value head (not the PPO critic).
        transformer_for_vh = cast(Transformer, self.transformer)
        if self.config.use_hyperbolic:
            self.value_head = HyperbolicValueHead(transformer_for_vh)
        else:
            self.value_head = ValueHead(
                transformer_for_vh,
                latent_dim=self.config.value_head_latent_dim,
            )

        self.start_epoch = self._start_epoch_override

        if self.config.resume:
            self._load_ppo_checkpoint()

        log_gpu_memory(prefix="After PPO model init - ")

    def _run_ppo_update(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Run PPO update steps on collected MCTS data."""
        metrics = self.ppo_model.update_from_training_data(training_data)
        avg_actor = metrics["ppo_actor_loss"]
        avg_critic = metrics["ppo_critic_loss"]
        logger.info(
            f"PPO update: actor_loss={avg_actor:.4f}, critic_loss={avg_critic:.4f}"
        )

        if self.config.use_wandb:
            wandb.log(
                {
                    "ppo/actor_loss": avg_actor,
                    "ppo/critic_loss": avg_critic,
                    "ppo/value_mean": metrics.get("ppo_value_mean", 0.0),
                }
            )

        return {
            "ppo_actor_loss": avg_actor,
            "ppo_critic_loss": avg_critic,
            "ppo_value_mean": metrics.get("ppo_value_mean", 0.0),
        }

    def _save_ppo_checkpoint(self, epoch: int) -> None:
        """Save LoRA adapter and critic weights."""
        self.ppo_model.save_checkpoint(self.checkpoint_dir, epoch, prefix="ppo")
        logger.info(f"PPO checkpoint saved: epoch {epoch}")

    def _load_ppo_checkpoint(self) -> None:
        """Resume from the latest PPO checkpoint."""
        loaded_epoch = self.ppo_model.load_latest_checkpoint(
            self.checkpoint_dir,
            prefix="ppo",
        )
        if loaded_epoch > 0:
            self.start_epoch = loaded_epoch
            logger.info(f"Resumed PPO from epoch {loaded_epoch}")

    def train(self) -> List[Dict[str, Any]]:
        """Training loop: MCTS data collection → value head + PPO updates."""
        all_metrics: List[Dict[str, Any]] = []
        self.progress_stats.run_start_time = time.time()
        self.progress_display.start()
        try:
            inference_server = InferenceServer(
                self.transformer,
                self.value_head,
                self.request_queue,
                self.response_queues,
                self.config.batch_size,
            )
            for epoch in range(
                self.start_epoch, self.start_epoch + self.config.num_epochs
            ):
                # Standard MCTS data collection + value head training
                metrics = self._run_epoch(epoch, inference_server)
                all_metrics.extend(metrics)

                # Load the training data that was just saved by _run_epoch
                td_path = self.checkpoint_dir / f"training_data_epoch_{epoch + 1}.json"
                training_data: List[Dict[str, Any]] = []
                if td_path.exists():
                    with open(td_path) as fp:
                        training_data = json.load(fp)

                # PPO update on collected data
                if training_data:
                    ppo_metrics = self._run_ppo_update(training_data)
                    if all_metrics:
                        all_metrics[-1].update(ppo_metrics)

                # Save PPO checkpoint
                self._save_ppo_checkpoint(epoch + 1)

            return all_metrics
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
            return all_metrics
        finally:
            self.progress_display.stop()
            self._cleanup_workers()
            if self.config.use_wandb:
                wandb.finish()


# ── Run orchestration ────────────────────────────────────────────────────────


def run_single(
    method: str,
    seed: int,
    size: str,
    benchmark_dir: Path,
    num_epochs: int = NUM_EPOCHS,
    run_index: int = 0,
    total_runs: int = 0,
    run_test_eval: bool = False,
    test_num_theorems: int = 500,
    test_split: str = "test",
    log_search_tree: bool = False,
    train_epochs_override: int | None = None,
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    run_dir = get_run_dir(benchmark_dir, method, seed, size)
    run_name = f"{method}_{seed}_{size}"

    training_complete = is_run_complete(run_dir, num_epochs)
    force_fresh_training = False

    # Guard against stale completion markers without the expected checkpoint.
    if training_complete and not METHOD_CONFIG[method]["is_ppo"]:
        expected_ckpt = (
            run_dir
            / f"value_head_{METHOD_CONFIG[method]['mcts_type']}_epoch_{num_epochs}.pth"
        )
        if not expected_ckpt.exists():
            logger.warning(
                f"[REPAIR] {run_name} marked complete but missing "
                f"{expected_ckpt.name}; forcing fresh training"
            )
            force_fresh_training = True
            training_complete = False
            purge_run_artifacts(run_dir)

    # Check if test evaluation has already been done
    eval_results_file = run_dir / "eval" / "theorem_results_epoch_1.json"
    test_eval_complete = eval_results_file.exists() if run_test_eval else False

    if training_complete and not run_test_eval:
        logger.info(f"[SKIP] {run_name} — training already complete")
        return {"status": "skipped", "run_name": run_name}

    if training_complete and test_eval_complete:
        logger.info(f"[SKIP] {run_name} — training and test eval already complete")
        return {"status": "skipped", "run_name": run_name}

    completed = 0 if force_fresh_training else get_completed_epochs(run_dir)
    resume = completed > 0

    if resume and not training_complete:
        logger.info(f"[RESUME] {run_name} — from epoch {completed}/{num_epochs}")
    elif not training_complete:
        logger.info(f"[START] {run_name}")
    else:
        logger.info(f"[EVAL ONLY] {run_name} — training complete, running test eval")

    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(
        method,
        seed,
        size,
        run_dir,
        num_epochs=num_epochs,
        resume=resume,
        train_epochs_override=train_epochs_override,
    )
    config.log_search_tree = log_search_tree

    # Skip training if already complete
    if not training_complete:
        if config.num_epochs <= 0:
            mark_run_complete(run_dir, method, seed, size, 0.0, num_epochs=num_epochs)
            return {"status": "skipped", "run_name": run_name}

        # Save config
        config_file = run_dir / "benchmark_config.json"
        with open(config_file, "w") as f:
            json.dump(
                {
                    "method": method,
                    "seed": seed,
                    "size": size,
                    "size_params": SIZE_CONFIGS[size],
                    "method_config": METHOD_CONFIG[method],
                    "num_epochs": num_epochs,
                    "resumed_from_epoch": completed,
                    "config": asdict(config),
                },
                f,
                indent=2,
            )

        start_time = time.time()
        try:
            mcfg = METHOD_CONFIG[method]

            trainer: Trainer
            if mcfg["is_ppo"]:
                trainer = PPOBenchmarkTrainer(
                    config, run_dir, start_epoch_override=completed
                )
            else:
                trainer = MCTSBenchmarkTrainer(
                    config,
                    run_dir,
                    start_epoch_override=completed,
                    method=method,
                )

            trainer.progress_stats.benchmark_run_name = run_name
            trainer.progress_stats.benchmark_run_index = run_index
            trainer.progress_stats.benchmark_total_runs = total_runs
            trainer.train()

            elapsed = time.time() - start_time
            mark_run_complete(
                run_dir, method, seed, size, elapsed, num_epochs=num_epochs
            )
            logger.success(f"[DONE] {run_name} in {elapsed / 3600:.1f}h")

            result = {"status": "completed", "run_name": run_name, "time": elapsed}

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            logger.warning(f"[INTERRUPTED] {run_name} after {elapsed / 3600:.1f}h")
            return {"status": "interrupted", "run_name": run_name, "time": elapsed}
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[FAILED] {run_name} after {elapsed / 3600:.1f}h: {e}")
            return {
                "status": "failed",
                "run_name": run_name,
                "error": str(e),
                "time": elapsed,
            }
    else:
        # Training already complete, skip to evaluation
        result = {"status": "training_skipped", "run_name": run_name, "time": 0.0}

    # Optional: Run test evaluation after training (or if training was already complete)
    if run_test_eval:
        logger.info(f"[TEST EVAL] Starting test evaluation for {run_name}")
        try:
            # Both MCTS and PPO methods save a value head checkpoint
            # (value_head_alpha_zero_latest.pth) during training.
            # PPO and hyperbolic_mcts use HyperbolicValueHead; euclidean uses ValueHead.
            is_ppo = METHOD_CONFIG[method]["is_ppo"]
            use_hyperbolic_eval = is_ppo or METHOD_CONFIG[method]["use_hyperbolic"]

            # All methods use alpha_zero MCTS type for search
            mcts_type = METHOD_CONFIG[method]["mcts_type"]

            # Build eval config
            eval_config = build_eval_config(
                algorithm=mcts_type,
                seed=seed,
                size=size,
                run_dir=run_dir,
                num_theorems=test_num_theorems,
            )

            # Load corpus (cached to avoid GitHub API rate limits)
            corpus, dataloader = load_or_init_corpus()

            # Run evaluation
            eval_start = time.time()
            evaluator = TestEvaluator(
                eval_config,
                run_dir,
                dataset_split=test_split,
                checkpoint_prefix_override=None,
                shared_corpus=corpus,
                shared_dataloader=dataloader,
                use_hyperbolic=use_hyperbolic_eval,
            )
            evaluator.train()
            eval_elapsed = time.time() - eval_start

            # Load results
            eval_results_file = run_dir / "eval" / "theorem_results_epoch_1.json"
            if eval_results_file.exists():
                with open(eval_results_file) as f:
                    eval_data = json.load(f)
                result["test_eval"] = {
                    "status": "completed",
                    "split": test_split,
                    "num_theorems": eval_data.get("total", 0),
                    "successful": eval_data.get("proved", 0),
                    "success_rate": eval_data.get("proved", 0)
                    / eval_data.get("total", 1),
                    "time_seconds": eval_elapsed,
                }
                test_eval = cast(Dict[str, Any], result["test_eval"])
                logger.success(
                    f"[TEST EVAL DONE] {run_name}: "
                    f"{test_eval['success_rate']:.1%} success rate "
                    f"({test_eval['successful']}/{test_eval['num_theorems']})"
                )
            else:
                result["test_eval"] = {
                    "status": "failed",
                    "error": "No eval results file found",
                }

        except Exception as e:
            logger.error(f"[TEST EVAL FAILED] {run_name}: {e}")

            logger.error(traceback.format_exc())
            result["test_eval"] = {"status": "failed", "error": str(e)}

    return result


def get_all_runs(
    methods: List[str] | None = None,
    seeds: List[int] | None = None,
    sizes: List[str] | None = None,
) -> List[Dict[str, Any]]:
    methods = methods or METHODS
    seeds = seeds or SEEDS
    sizes = sizes or SIZES
    runs = []
    for method in methods:
        for seed in seeds:
            for size in sizes:
                runs.append({"method": method, "seed": seed, "size": size})
    return runs


# ── Status display ───────────────────────────────────────────────────────────


def print_status(
    benchmark_dir: Path,
    methods: List[str] | None = None,
    seeds: List[int] | None = None,
    sizes: List[str] | None = None,
    num_epochs: int = NUM_EPOCHS,
) -> None:
    runs = get_all_runs(methods=methods, seeds=seeds, sizes=sizes)
    print(f"\n{'=' * 90}")
    print(f"FOUR-WAY BENCHMARK STATUS — {benchmark_dir}")
    print(f"{'=' * 90}")
    print(f"{'Run Name':<45} {'Status':<12} {'Epochs':<10} {'Complete'}")
    print(f"{'-' * 90}")

    done = 0
    for run in runs:
        rd = get_run_dir(benchmark_dir, run["method"], run["seed"], run["size"])
        name = f"{run['method']}_{run['seed']}_{run['size']}"
        if is_run_complete(rd, num_epochs):
            status, epochs, comp = "✓ DONE", f"{num_epochs}/{num_epochs}", "Yes"
            done += 1
        elif rd.exists():
            ce = get_completed_epochs(rd)
            status = "⟳ PARTIAL" if ce > 0 else "○ STARTED"
            epochs, comp = f"{ce}/{num_epochs}", "No"
        else:
            status, epochs, comp = "— PENDING", f"0/{num_epochs}", "No"
        print(f"{name:<45} {status:<12} {epochs:<10} {comp}")

    print(f"{'-' * 90}")
    print(f"Total: {done}/{len(runs)} complete")
    print(f"{'=' * 90}\n")


# ── Plotting ─────────────────────────────────────────────────────────────────


def collect_epoch_data(run_dir: Path) -> Dict[str, Any]:
    """Collect per-epoch metrics from a single run directory."""
    data: Dict[str, Any] = {
        "epochs": [],
        "success_rates": [],
        "proved_counts": [],
        "total_counts": [],
        "best_val_losses": [],
    }
    if not run_dir.exists():
        return data

    for epoch_file in sorted(run_dir.glob("theorem_results_epoch_*.json")):
        try:
            epoch_num = int(epoch_file.stem.split("_")[-1])
            with open(epoch_file) as fp:
                result = json.load(fp)
            data["epochs"].append(epoch_num)
            data["success_rates"].append(result.get("success_rate", 0.0))
            data["proved_counts"].append(result.get("proved", 0))
            data["total_counts"].append(result.get("total", 0))
        except (ValueError, json.JSONDecodeError, KeyError):
            continue

    # Try to collect val losses
    val_losses: Dict[int, float] = {}
    for loss_file in sorted(run_dir.glob("val_loss_epoch_*.json")):
        try:
            epoch_num = int(loss_file.stem.split("_")[-1])
            with open(loss_file) as fp:
                ld = json.load(fp)
            val_losses[epoch_num] = ld.get("best_val_loss")
        except (ValueError, json.JSONDecodeError, KeyError):
            continue
    if val_losses:
        data["best_val_losses"] = [val_losses.get(e) for e in data["epochs"]]
    else:
        data["best_val_losses"] = [None] * len(data["epochs"])

    return data


def plot_results(benchmark_dir: Path, output_dir: Path) -> None:
    """Generate comparison plots for the four-way benchmark."""
    matplotlib.use("Agg")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Success rate per epoch, one subplot per size ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

    for size_idx, size in enumerate(SIZES):
        ax = axes[size_idx]
        ax.set_title(f"Proof Success Rate ({size.capitalize()})", fontsize=14)
        ax.set_xlabel("Epoch")
        if size_idx == 0:
            ax.set_ylabel("Success Rate (%)")
        ax.grid(True, alpha=0.3)

        for method in METHODS:
            all_rates = []
            max_epochs = 0

            for seed in SEEDS:
                rd = get_run_dir(benchmark_dir, method, seed, size)
                ed = collect_epoch_data(rd)
                if ed["epochs"]:
                    rates = [r * 100 for r in ed["success_rates"]]
                    all_rates.append((ed["epochs"], rates))
                    max_epochs = max(max_epochs, max(ed["epochs"]))

            if not all_rates:
                continue

            epochs_grid = np.arange(1, max_epochs + 1)
            rate_matrix = np.full((len(all_rates), len(epochs_grid)), np.nan)
            for i, (epochs, rates) in enumerate(all_rates):
                for e, r in zip(epochs, rates):
                    if 1 <= e <= max_epochs:
                        rate_matrix[i, e - 1] = r

            mean = np.nanmean(rate_matrix, axis=0)
            std = np.nanstd(rate_matrix, axis=0)

            label = METHOD_LABELS[method]
            color = METHOD_COLORS[method]
            ax.plot(
                epochs_grid,
                mean,
                "-o",
                label=label,
                color=color,
                markersize=3,
                linewidth=1.5,
            )
            ax.fill_between(epochs_grid, mean - std, mean + std, alpha=0.2, color=color)
            for i, (epochs, rates) in enumerate(all_rates):
                ax.plot(epochs, rates, "--", alpha=0.3, color=color, linewidth=0.8)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    plt.suptitle(
        "Four-Way Comparison: Euclidean vs Hyperbolic MCTS & PPO",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "4way_success_rate_per_epoch.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(output_dir / "4way_success_rate_per_epoch.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / '4way_success_rate_per_epoch.png'}")

    # ── 2. Combined comparison: all sizes per method ─────────────────────
    fig, axes = plt.subplots(1, len(METHODS), figsize=(8 * len(METHODS), 6))
    if len(METHODS) == 1:
        axes = [axes]

    size_colors = {"light": "tab:green", "medium": "tab:blue", "heavy": "tab:red"}

    for m_idx, method in enumerate(METHODS):
        ax = axes[m_idx]
        ax.set_title(f"{METHOD_LABELS[method]}", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Success Rate (%)")
        ax.grid(True, alpha=0.3)

        for size in SIZES:
            all_rates = []
            max_epochs = 0
            for seed in SEEDS:
                rd = get_run_dir(benchmark_dir, method, seed, size)
                ed = collect_epoch_data(rd)
                if ed["epochs"]:
                    rates = [r * 100 for r in ed["success_rates"]]
                    all_rates.append((ed["epochs"], rates))
                    max_epochs = max(max_epochs, max(ed["epochs"]))

            if not all_rates:
                continue

            epochs_grid = np.arange(1, max_epochs + 1)
            rate_matrix = np.full((len(all_rates), len(epochs_grid)), np.nan)
            for i, (epochs, rates) in enumerate(all_rates):
                for e, r in zip(epochs, rates):
                    if 1 <= e <= max_epochs:
                        rate_matrix[i, e - 1] = r

            mean = np.nanmean(rate_matrix, axis=0)
            std = np.nanstd(rate_matrix, axis=0)

            ax.plot(
                epochs_grid,
                mean,
                "-o",
                label=f"{size.capitalize()} ({SIZE_CONFIGS[size]['num_iterations']} iter)",
                color=size_colors[size],
                markersize=3,
                linewidth=1.5,
            )
            ax.fill_between(
                epochs_grid, mean - std, mean + std, alpha=0.15, color=size_colors[size]
            )

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="lower right")
        ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "4way_size_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "4way_size_comparison.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / '4way_size_comparison.png'}")

    # ── 3. Final epoch bar chart ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

    for size_idx, size in enumerate(SIZES):
        ax = axes[size_idx]
        ax.set_title(f"Final-Epoch Success ({size.capitalize()})", fontsize=14)
        if size_idx == 0:
            ax.set_ylabel("Success Rate (%)")

        x = np.arange(len(SEEDS))
        width = 0.20

        for m_idx, method in enumerate(METHODS):
            rates = []
            for seed in SEEDS:
                rd = get_run_dir(benchmark_dir, method, seed, size)
                ed = collect_epoch_data(rd)
                if ed["success_rates"]:
                    rates.append(ed["success_rates"][-1] * 100)
                else:
                    rates.append(0.0)

            bars = ax.bar(
                x[: len(rates)] + m_idx * width - width * 1.5,
                rates,
                width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                alpha=0.85,
            )
            for bar, rate in zip(bars, rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in SEEDS])
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Final-Epoch Proof Success: 4-Way Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "4way_final_bar.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "4way_final_bar.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / '4way_final_bar.png'}")

    # ── 4. Summary table (JSON) ──────────────────────────────────────────
    summary: Dict[str, Any] = {}
    for method in METHODS:
        summary[method] = {}
        for size in SIZES:
            final_rates = []
            for seed in SEEDS:
                rd = get_run_dir(benchmark_dir, method, seed, size)
                ed = collect_epoch_data(rd)
                if ed["success_rates"]:
                    final_rates.append(ed["success_rates"][-1])
            if final_rates:
                summary[method][size] = {
                    "mean": float(np.mean(final_rates)),
                    "std": float(np.std(final_rates)),
                    "per_seed": final_rates,
                }

    summary_file = output_dir / "4way_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_file}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four-way benchmark: Euclidean vs Hyperbolic MCTS and PPO"
    )
    parser.add_argument("--status", action="store_true", help="Print status and exit.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate plots from existing data and exit.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Which methods to benchmark.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only the fast subset: seed 42 and light size.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/benchmark_4way",
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--run-test-eval",
        action="store_true",
        help="Run test evaluation after training completes for each run.",
    )
    parser.add_argument(
        "--test-num-theorems",
        type=int,
        default=500,
        help="Number of theorems to use for test evaluation (default: 500).",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        choices=["test", "val", "train"],
        default="test",
        help="Dataset split to use for test evaluation (default: test).",
    )
    args = parser.parse_args()

    benchmark_dir = (
        Path(args.benchmark_dir) if args.benchmark_dir else get_benchmark_dir()
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods
    seeds = FAST_SEEDS if args.fast else args.seeds
    sizes = FAST_SIZES if args.fast else args.sizes
    num_epochs = FAST_NUM_EPOCHS if args.fast else NUM_EPOCHS
    train_epochs_override = FAST_TRAIN_EPOCHS if args.fast else None

    # Fast mode overrides: reduce number of theorems to make smoke runs quicker
    if args.fast:
        try:
            BASE_PARAMS["num_theorems"] = 25
            logger.info("Fast mode: setting BASE_PARAMS['num_theorems']=25")
        except Exception:
            logger.warning(
                "Failed to override BASE_PARAMS['num_theorems'] for fast mode"
            )

    if args.status:
        print_status(
            benchmark_dir,
            methods=methods,
            seeds=seeds,
            sizes=sizes,
            num_epochs=num_epochs,
        )
        return

    if args.plot_only:
        plot_results(benchmark_dir, Path(args.output_dir))
        return

    # Run benchmark
    runs = get_all_runs(methods, seeds, sizes)
    logger.info(f"Benchmark dir: {benchmark_dir}")
    logger.info(f"Total runs: {len(runs)}")
    if args.fast:
        logger.info(
            "Fast mode enabled: "
            f"seed=42, size=light, epochs={FAST_NUM_EPOCHS}, "
            f"theorems=25, value_head_train_epochs={train_epochs_override}"
        )
    print_status(
        benchmark_dir,
        methods=methods,
        seeds=seeds,
        sizes=sizes,
        num_epochs=num_epochs,
    )

    results = []
    for i, run in enumerate(runs, 1):
        run_name = f"{run['method']}_{run['seed']}_{run['size']}"
        logger.info(
            f"\n{'#' * 60}\n" f"# RUN {i}/{len(runs)}: {run_name}\n" f"{'#' * 60}"
        )
        result = run_single(
            run["method"],
            run["seed"],
            run["size"],
            benchmark_dir,
            num_epochs=num_epochs,
            run_index=i,
            total_runs=len(runs),
            run_test_eval=args.run_test_eval,
            test_num_theorems=args.test_num_theorems,
            test_split=args.test_split,
            train_epochs_override=train_epochs_override,
        )
        results.append(result)
        _post_run_memory_cleanup(run_name)

        # Incremental progress
        progress_file = benchmark_dir / "benchmark_progress.json"
        with open(progress_file, "w") as f:
            json.dump(
                {
                    "total_runs": len(runs),
                    "completed_runs": sum(
                        1 for r in results if r["status"] in ("completed", "skipped")
                    ),
                    "results": results,
                },
                f,
                indent=2,
            )

    # Final summary
    print_status(
        benchmark_dir,
        methods=methods,
        seeds=seeds,
        sizes=sizes,
        num_epochs=num_epochs,
    )

    # Generate plots
    plot_results(benchmark_dir, Path(args.output_dir))

    # Print quick comparison
    print("\n" + "=" * 60)
    print("FINAL RESULT SUMMARY")
    print("=" * 60)
    for result in results:
        status = result["status"]
        name = result["run_name"]
        t = result.get("time", 0)
        print(f"  {name:<45} {status:<12} {t / 3600:.1f}h")
    print("=" * 60)


if __name__ == "__main__":
    main()
