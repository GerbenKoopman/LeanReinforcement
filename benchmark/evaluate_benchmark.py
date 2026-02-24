#!/usr/bin/env python3
"""
Evaluate all benchmark checkpoints on the test set.

For each completed benchmark run, loads the final checkpoint and evaluates
on the test split using the same hyperparameters used during training.

Produces a JSON results file and per-run theorem-level results.
"""

import argparse
import gc
import json
import pickle
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from loguru import logger

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.value_head import ValueHead
from lean_reinforcement.training.trainer import Trainer
from lean_reinforcement.training.worker import worker_loop
from lean_reinforcement.training.progress import (
    ProgressStats,
    make_progress_display,
)
from lean_reinforcement.utilities.checkpoint import load_checkpoint
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.utilities.dataloader import LeanDataLoader
from ReProver.common import Corpus

from benchmark.run_benchmark import (
    BASE_PARAMS,
    SIZE_CONFIGS,
    ALGORITHMS,
    SEEDS,
    SIZES,
    NUM_EPOCHS,
    get_benchmark_dir,
    get_run_dir,
    is_run_complete,
    get_all_runs,
)

load_dotenv()


class TestEvaluator(Trainer):
    """Trainer subclass for test-set evaluation of benchmark checkpoints."""

    def __init__(
        self,
        config: TrainingConfig,
        run_dir: Path,
        dataset_split: str = "test",
        checkpoint_prefix_override: str | None = None,
        shared_corpus: Optional[Corpus] = None,
        shared_dataloader: Optional[LeanDataLoader] = None,
    ):
        self.config = config
        self.dataset_split = dataset_split
        self._checkpoint_prefix_override = checkpoint_prefix_override

        # Initialize progress tracking (required by parent train() and _collect_data())
        self.progress_stats = ProgressStats(
            total_epochs=config.num_epochs,
            total_workers=config.num_workers,
            cumulative_total_theorems=config.num_epochs * config.num_theorems,
        )
        self.progress_display = make_progress_display(self.progress_stats)

        # Set seeds for reproducibility during evaluation
        if config.seed is not None:
            self._set_seeds(config.seed)

        # Use a separate eval subdirectory to avoid overwriting training results
        self.checkpoint_dir = run_dir / "eval"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # No wandb during evaluation
        self._setup_models_for_eval(run_dir)

        # Use shared corpus/dataloader to avoid repeated GitHub API calls
        if shared_corpus is not None and shared_dataloader is not None:
            self.corpus = shared_corpus
            self.dataloader = shared_dataloader
        else:
            self._setup_data()

        self._setup_multiprocessing()

    def _setup_models_for_eval(self, run_dir: Path) -> None:
        """Set up models and load the best checkpoint."""

        logger.info(f"Loading models for evaluation from {run_dir}")
        self.transformer = Transformer(model_name=self.config.model_name)
        self.value_head = None
        self.start_epoch = 0

        # Allow overriding the checkpoint prefix, e.g. to load a
        # guided_rollout-trained value head for alpha_zero evaluation.
        prefix = (
            self._checkpoint_prefix_override or f"value_head_{self.config.mcts_type}"
        )
        checkpoint_path = run_dir / f"{prefix}_latest.pth"
        if checkpoint_path.exists():
            self.value_head = ValueHead(
                self.transformer, hidden_dims=self.config.value_head_hidden_dims
            )
            loaded_epoch = load_checkpoint(self.value_head, run_dir, prefix=prefix)
            logger.info(f"Loaded checkpoint from epoch {loaded_epoch}")
        elif self.config.mcts_type == "alpha_zero":
            # Alpha zero requires a value head even without checkpoint
            self.value_head = ValueHead(
                self.transformer, hidden_dims=self.config.value_head_hidden_dims
            )
            logger.warning(
                f"No checkpoint found at {checkpoint_path}, using untrained value head"
            )

        self._log_gpu_memory("After model initialization - ")

    def _run_epoch(self, epoch: int, inference_server) -> List[Dict[str, Any]]:
        """
        Rolling-queue evaluation that guarantees every theorem is attempted.

        All theorems are enqueued up front so workers stay busy continuously
        (no idle time between batches).  A wall-clock safety deadline prevents
        infinite hangs: if no result and no inference activity arrives for
        ``proof_timeout`` seconds the stuck workers are killed, their pending
        theorems recorded as failures, and fresh workers are spawned.
        Worker crashes are detected and handled individually.
        """
        # ── Select data split ────────────────────────────────────────────────
        if self.dataset_split == "val":
            split_data = self.dataloader.val_data
        elif self.dataset_split == "test":
            split_data = self.dataloader.test_data
        else:
            split_data = self.dataloader.train_data

        theorems = split_data[: self.config.num_theorems]
        total = len(theorems)

        logger.info(f"Evaluating on {self.dataset_split} split ({total} theorems)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Progress display ─────────────────────────────────────────────────
        self.progress_stats.reset_epoch(epoch=epoch + 1, total_theorems=total)
        self.progress_stats.alive_workers = self.config.num_workers
        self.progress_display.refresh()

        num_workers = self.config.num_workers
        # Stall timeout: if nothing happens (no results AND no inference
        # requests processed) for this long, assume all workers are stuck.
        stall_timeout = self.config.proof_timeout + 60
        # Hard safety ceiling for the entire evaluation.
        safety_timeout = max(
            3600.0, total / num_workers * self.config.proof_timeout * 2
        )

        all_theorem_results: List[Dict[str, Any]] = []
        results_received = 0
        logged_dead_workers: set = set()
        had_any_activity = False  # Only trigger stall timeout after first result
        startup_timeout = (
            300.0  # 5 minutes for initial worker startup to recover from rate limiting
        )

        self._drain_theorem_queue()
        self._start_workers()

        # ── Enqueue ALL theorems at once (rolling queue) ─────────────────────
        for thm in theorems:
            self.theorem_queue.put(thm)

        last_activity = time.monotonic()
        eval_start = time.monotonic()
        startup_start = time.monotonic()

        try:
            while results_received < total:
                now = time.monotonic()

                # ── Hard safety ceiling ───────────────────────────────────
                if now - eval_start > safety_timeout:
                    missing = total - results_received
                    logger.warning(
                        f"Safety timeout ({safety_timeout/3600:.1f}h) reached. "
                        f"Recording {missing} remaining theorem(s) as failures."
                    )
                    for _ in range(missing):
                        all_theorem_results.append(
                            {
                                "theorem_name": "safety_timeout",
                                "success": False,
                                "steps": 0,
                                "time": 0.0,
                            }
                        )
                        self.progress_stats.record_theorem(
                            name="safety_timeout", success=False
                        )
                    results_received += missing
                    break

                # ── Serve GPU inference ───────────────────────────────────
                processed = inference_server.process_requests()
                if processed:
                    last_activity = time.monotonic()

                # ── Drain result queue ────────────────────────────────────
                got_result = False
                try:
                    while results_received < total:
                        res = self.result_queue.get_nowait()
                        results_received += 1
                        got_result = True
                        had_any_activity = True
                        last_activity = time.monotonic()

                        theorem_name = res.get(
                            "theorem_name", f"unknown_{results_received}"
                        )
                        success = False
                        steps = 0
                        elapsed_t = 0.0
                        if res and "metrics" in res:
                            m = res["metrics"]
                            success = m.get("proof_search/success", False)
                            steps = m.get("proof_search/steps", 0)
                            elapsed_t = m.get("proof_search/time", 0.0)

                        all_theorem_results.append(
                            {
                                "theorem_name": theorem_name,
                                "success": success,
                                "steps": steps,
                                "time": elapsed_t,
                            }
                        )
                        self.progress_stats.record_theorem(
                            name=theorem_name, success=success, elapsed=elapsed_t
                        )
                        self.progress_display.refresh()
                except queue.Empty:
                    pass

                if not processed and not got_result:
                    time.sleep(0.01)

                # ── Detect crashed workers ────────────────────────────────
                for idx, p in enumerate(self.workers):
                    if (
                        not p.is_alive()
                        and p.exitcode not in (0, None)
                        and p.pid not in logged_dead_workers
                    ):
                        logged_dead_workers.add(p.pid)
                        logger.warning(
                            f"Worker {idx} (PID {p.pid}) crashed "
                            f"(exit {p.exitcode}) – recording as failure."
                        )
                        results_received += 1
                        last_activity = time.monotonic()
                        all_theorem_results.append(
                            {
                                "theorem_name": f"worker_{idx}_crash",
                                "success": False,
                                "steps": 0,
                                "time": 0.0,
                            }
                        )
                        self.progress_stats.record_theorem(
                            name=f"worker_{idx}_crash", success=False
                        )
                        self.progress_display.refresh()

                        # Restart the crashed worker
                        try:
                            p.join(timeout=1)
                        except Exception:
                            pass
                        new_p = mp.Process(
                            target=worker_loop,
                            args=(
                                idx,
                                self.request_queue,
                                self.response_queues[idx],
                                self.theorem_queue,
                                self.result_queue,
                                self.config,
                            ),
                        )
                        new_p.start()
                        self.workers[idx] = new_p

                self.progress_stats.alive_workers = sum(
                    1 for p in self.workers if p.is_alive()
                )

                # ── Stall detection: restart all workers ──────────────────
                # Trigger stall timeout if:
                # 1. Startup takes too long (> 30 min) with zero results, OR
                # 2. After first result, no activity for proof_timeout + 60s
                time_since_startup = time.monotonic() - startup_start
                no_results_yet = results_received == 0
                startup_timeout_fired = (
                    no_results_yet and time_since_startup > startup_timeout
                )
                activity_timeout_fired = (
                    had_any_activity
                    and time.monotonic() - last_activity > stall_timeout
                )

                if startup_timeout_fired or activity_timeout_fired:
                    if startup_timeout_fired:
                        msg = (
                            f"Startup timeout ({startup_timeout/60:.0f}m): "
                            f"no results after {time_since_startup/60:.1f}m. "
                            f"Workers likely stuck during initialization. "
                        )
                    else:
                        msg = (
                            f"No activity for {stall_timeout:.0f}s "
                            f"({total - results_received} theorems pending, "
                            f"{sum(1 for p in self.workers if p.is_alive())} workers alive). "
                        )
                    logger.warning(msg + "Restarting workers.")

                    pending = total - results_received
                    alive = sum(1 for p in self.workers if p.is_alive())

                    # How many theorems are currently in-flight?  At most
                    # one per alive worker.  Record those as failures.
                    in_flight = min(alive, pending)
                    self._stop_workers()
                    self._drain_queues()
                    for _ in range(in_flight):
                        results_received += 1
                        all_theorem_results.append(
                            {
                                "theorem_name": "stall_timeout",
                                "success": False,
                                "steps": 0,
                                "time": 0.0,
                            }
                        )
                        self.progress_stats.record_theorem(
                            name="stall_timeout", success=False
                        )
                    self.progress_display.refresh()
                    logged_dead_workers = set()

                    # Re-enqueue remaining theorems that haven't been picked up
                    still_remaining = total - results_received
                    if still_remaining > 0:
                        # Re-submit theorems starting from results_received index
                        re_enqueue = theorems[results_received:]
                        self._start_workers()
                        startup_start = time.monotonic()  # Reset startup timer
                        for thm in re_enqueue:
                            self.theorem_queue.put(thm)
                        last_activity = time.monotonic()
                    else:
                        break

        finally:
            self._stop_workers()
            self._drain_queues()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # ── Persist results and build metrics list ────────────────────────────
        total_attempted = len(all_theorem_results)
        proved = [t for t in all_theorem_results if t["success"]]
        failed = [t for t in all_theorem_results if not t["success"]]
        sr = len(proved) / total_attempted * 100 if total_attempted else 0

        logger.info(
            f"Evaluation complete: {len(proved)}/{total_attempted} "
            f"proved ({sr:.1f}%)."
        )

        results_file = self.checkpoint_dir / f"theorem_results_epoch_{epoch + 1}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "seed": self.config.seed,
                    "mcts_type": self.config.mcts_type,
                    "total": total_attempted,
                    "proved": len(proved),
                    "failed": len(failed),
                    "theorems": all_theorem_results,
                },
                f,
                indent=2,
            )

        return [
            {
                "proof_search/success": t["success"],
                "proof_search/steps": t["steps"],
                "proof_search/time": t["time"],
            }
            for t in all_theorem_results
        ]


def build_eval_config(
    algorithm: str, seed: int, size: str, run_dir: Path, num_theorems: int = 500
) -> TrainingConfig:
    """Build evaluation config with same hyperparameters as training."""
    size_params = SIZE_CONFIGS[size]

    # Extract values with type assertions
    data_type_val: str = BASE_PARAMS["data_type"]  # type: ignore[assignment]
    max_steps_val: int = BASE_PARAMS["max_steps"]  # type: ignore[assignment]
    indexed_corpus_val = BASE_PARAMS["indexed_corpus_path"]
    value_head_hidden_dims_val: list = BASE_PARAMS["value_head_hidden_dims"]  # type: ignore[assignment]
    use_final_reward_val: bool = BASE_PARAMS["use_final_reward"]  # type: ignore[assignment]
    inference_timeout_val: float = BASE_PARAMS["inference_timeout"]  # type: ignore[assignment]
    model_name_val: str = BASE_PARAMS["model_name"]  # type: ignore[assignment]
    num_tactics_val: int = BASE_PARAMS["num_tactics_to_expand"]  # type: ignore[assignment]
    max_rollout_val: int = BASE_PARAMS["max_rollout_depth"]  # type: ignore[assignment]
    num_iterations_val: int = size_params["num_iterations"]  # type: ignore[assignment]
    max_time_val: float = size_params["max_time"]  # type: ignore[assignment]
    env_timeout_val: int = size_params["env_timeout"]  # type: ignore[assignment]
    proof_timeout_val: float = size_params["proof_timeout"]  # type: ignore[assignment]

    return TrainingConfig(
        data_type=data_type_val,
        num_epochs=1,  # Single evaluation pass
        num_theorems=num_theorems,
        num_iterations=num_iterations_val,
        max_steps=max_steps_val,
        batch_size=8,  # Reduced for evaluation memory efficiency
        num_workers=12,
        mcts_type=algorithm,
        indexed_corpus_path=indexed_corpus_val,  # type: ignore[arg-type]
        train_epochs=0,  # No training during evaluation
        value_head_batch_size=2,
        value_head_hidden_dims=value_head_hidden_dims_val,  # type: ignore[arg-type]
        train_value_head=True,  # Load value head for both algorithms
        use_final_reward=use_final_reward_val,
        save_training_data=False,
        use_caching=False,  # Caching causes OOM with large eval sets
        seed=seed,
        save_checkpoints=False,
        resume=False,
        use_test_value_head=False,
        checkpoint_dir=str(run_dir),
        use_wandb=False,
        inference_timeout=inference_timeout_val,
        model_name=model_name_val,
        num_tactics_to_expand=num_tactics_val,
        max_rollout_depth=max_rollout_val,
        max_time=max_time_val,
        env_timeout=env_timeout_val,
        proof_timeout=proof_timeout_val,
    )


def get_corpus_cache_path() -> Path:
    """Get the path to the persistent corpus cache."""
    cache_dir = Path.home() / ".cache" / "lean-reinforcement" / "benchmark"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "corpus_cache.pkl"


def load_or_init_corpus(
    corpus_path: str = "leandojo_benchmark_4/corpus.jsonl",
) -> tuple[Corpus, LeanDataLoader]:
    """Load corpus and dataloader, using cache if available.

    This function initializes the corpus once and caches it to disk to avoid
    repeated GitHub API calls during corpus initialization (which causes
    rate limiting when running many sequential eval runs).

    Args:
        corpus_path: Path to the corpus.jsonl file.

    Returns:
        Tuple of (Corpus, LeanDataLoader) instances.
    """
    cache_path = get_corpus_cache_path()

    # Try to load from cache
    if cache_path.exists():
        try:
            logger.info(f"Loading cached corpus from {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            return cached_data["corpus"], cached_data["dataloader"]
        except Exception as e:
            logger.warning(f"Failed to load corpus cache: {e}. Reinitializing...")
            cache_path.unlink(missing_ok=True)

    # Initialize corpus (this may trigger GitHub API calls)
    logger.info(f"Initializing corpus from {corpus_path}")
    corpus = Corpus(corpus_path)

    logger.info("Initializing dataloader")
    data_type = cast(str, BASE_PARAMS["data_type"])
    dataloader = LeanDataLoader(
        corpus,
        dataset_path="leandojo_benchmark_4",
        data_type=data_type,
    )

    # Cache for future use
    try:
        logger.info(f"Caching corpus to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump({"corpus": corpus, "dataloader": dataloader}, f)
    except Exception as e:
        logger.warning(f"Failed to cache corpus: {e}. Continuing without cache...")

    return corpus, dataloader


def evaluate_single_run(
    algorithm: str,
    seed: int,
    size: str,
    benchmark_dir: Path,
    num_theorems: int = 500,
    split: str = "test",
    eval_mcts_type: str | None = None,
    shared_corpus: Optional[Corpus] = None,
    shared_dataloader: Optional[LeanDataLoader] = None,
) -> Dict[str, Any]:
    """Evaluate a single benchmark run on the test set.

    Args:
        algorithm: The algorithm used during training (for locating checkpoints).
        seed: Random seed.
        size: Size config key (light/medium/heavy).
        benchmark_dir: Path to benchmark directory.
        num_theorems: Target number of theorems to evaluate.
        split: Dataset split to evaluate on.
        eval_mcts_type: MCTS type to use during evaluation. If different from
            ``algorithm``, the value head trained by ``algorithm`` is loaded but
            MCTS search uses ``eval_mcts_type`` instead.  For example, setting
            ``algorithm="guided_rollout"`` and ``eval_mcts_type="alpha_zero"``
            evaluates the guided-rollout-trained value head with alpha-zero
            search.
        shared_corpus: Optional pre-initialized corpus to avoid repeated GitHub API calls.
        shared_dataloader: Optional pre-initialized dataloader corresponding to shared_corpus.
    """
    run_dir = get_run_dir(benchmark_dir, algorithm, seed, size)
    # Determine the MCTS type used for evaluation
    effective_mcts = eval_mcts_type or algorithm
    # When eval_mcts_type differs from training algorithm, mark the run name
    if eval_mcts_type and eval_mcts_type != algorithm:
        run_name = f"{algorithm}_{seed}_{size}_as_{eval_mcts_type}"
        eval_results_file = run_dir / f"eval_results_{split}_{eval_mcts_type}.json"
        eval_subdir = run_dir / f"eval_{eval_mcts_type}"
    else:
        run_name = f"{algorithm}_{seed}_{size}"
        eval_results_file = run_dir / f"eval_results_{split}.json"
        eval_subdir = run_dir / "eval"

    # Check if evaluation was already done AND is complete
    if eval_results_file.exists():
        with open(eval_results_file) as f:
            existing: Dict[str, Any] = json.load(f)
        existing_total = existing.get("num_theorems", 0)
        if existing_total >= num_theorems:
            logger.info(
                f"[SKIP] {run_name} — evaluation already complete "
                f"({existing_total}/{num_theorems} theorems)"
            )
            return existing
        else:
            logger.warning(
                f"[RESUME] {run_name} — previous evaluation incomplete "
                f"({existing_total}/{num_theorems} theorems), re-running"
            )
            # Remove partial results so we start fresh
            eval_results_file.unlink(missing_ok=True)
            # Also remove partial theorem results
            partial_theorem_file = eval_subdir / "theorem_results_epoch_1.json"
            if partial_theorem_file.exists():
                partial_theorem_file.unlink()

    # Check if training run is complete
    if not is_run_complete(run_dir, NUM_EPOCHS):
        logger.warning(f"[SKIP] {run_name} — training not complete")
        return {"status": "training_incomplete", "run_name": run_name}

    logger.info(f"[EVAL] {run_name} on {split} split ({num_theorems} theorems)")

    config = build_eval_config(effective_mcts, seed, size, run_dir, num_theorems)

    # When evaluating with a different MCTS type, we need to load the
    # checkpoint that was saved under the *training* algorithm's prefix
    checkpoint_prefix = None
    if eval_mcts_type and eval_mcts_type != algorithm:
        checkpoint_prefix = f"value_head_{algorithm}"

    start_time = time.time()
    try:
        evaluator = TestEvaluator(
            config,
            run_dir,
            dataset_split=split,
            checkpoint_prefix_override=checkpoint_prefix,
            shared_corpus=shared_corpus,
            shared_dataloader=shared_dataloader,
        )
        # Override eval subdir for cross-algorithm evaluation
        if eval_mcts_type and eval_mcts_type != algorithm:
            evaluator.checkpoint_dir = eval_subdir
            eval_subdir.mkdir(parents=True, exist_ok=True)

        results = evaluator.train()
        elapsed = time.time() - start_time

        # Count successes from theorem results
        theorem_results_file = eval_subdir / "theorem_results_epoch_1.json"
        successful = 0
        total = 0
        theorem_details = []

        if theorem_results_file.exists():
            with open(theorem_results_file) as f:
                epoch_results = json.load(f)
                successful = epoch_results.get("proved", 0)
                total = epoch_results.get("total", 0)
                theorem_details = epoch_results.get("theorems", [])
        else:
            # Fall back to counting from metrics
            for metric in results:
                if isinstance(metric, dict) and "proof_search/success" in metric:
                    total += 1
                    if metric["proof_search/success"]:
                        successful += 1

        success_rate = successful / total if total > 0 else 0.0

        eval_result = {
            "status": "completed",
            "run_name": run_name,
            "algorithm": algorithm,
            "eval_mcts_type": effective_mcts,
            "seed": seed,
            "size": size,
            "split": split,
            "num_theorems": total,
            "successful": successful,
            "success_rate": success_rate,
            "time_seconds": elapsed,
            "size_params": SIZE_CONFIGS[size],
            "theorem_details": theorem_details,
        }

        with open(eval_results_file, "w") as f:
            json.dump(eval_result, f, indent=2)

        logger.success(
            f"[DONE] {run_name}: {successful}/{total} = {success_rate:.1%} "
            f"in {elapsed/60:.1f}min"
        )
        return eval_result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[FAILED] {run_name}: {e}")
        return {"status": "failed", "run_name": run_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark checkpoints on test set"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=500,
        help="Number of theorems to evaluate on.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
    )
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save aggregated evaluation results.",
    )
    parser.add_argument(
        "--cross-eval",
        action="store_true",
        default=False,
        help=(
            "Also evaluate guided_rollout checkpoints using alpha_zero MCTS. "
            "This produces three comparison points: guided_rollout search, "
            "guided_rollout-trained value head with alpha_zero search, "
            "and alpha_zero-trained value head with alpha_zero search."
        ),
    )

    args = parser.parse_args()
    benchmark_dir = (
        Path(args.benchmark_dir) if args.benchmark_dir else get_benchmark_dir()
    )

    runs = [
        r
        for r in get_all_runs()
        if r["algorithm"] in args.algorithms
        and r["seed"] in args.seeds
        and r["size"] in args.sizes
    ]

    # Build evaluation schedule: (algorithm, seed, size, eval_mcts_type)
    eval_schedule: list[tuple[str, int, str, str | None]] = []
    for run in runs:
        # Standard evaluation: same MCTS type as training
        eval_schedule.append((run["algorithm"], run["seed"], run["size"], None))
    # Cross-evaluation: guided_rollout checkpoints evaluated with alpha_zero MCTS
    if args.cross_eval:
        for run in runs:
            if run["algorithm"] == "guided_rollout":
                eval_schedule.append(
                    (run["algorithm"], run["seed"], run["size"], "alpha_zero")
                )

    logger.info(f"Evaluating {len(eval_schedule)} runs on {args.split} set")

    # Initialize corpus once for all evaluation runs to avoid repeated GitHub API calls
    logger.info("Initializing shared corpus and dataloader...")
    try:
        shared_corpus, shared_dataloader = load_or_init_corpus()
    except Exception as e:
        logger.error(f"Failed to initialize corpus: {e}")
        logger.warning("Falling back to per-run corpus initialization (may be slower)")
        shared_corpus = None
        shared_dataloader = None

    all_results = []
    for i, (algorithm, seed, size, eval_mcts) in enumerate(eval_schedule, 1):
        logger.info(f"\nEvaluation {i}/{len(eval_schedule)}")
        result = evaluate_single_run(
            algorithm,
            seed,
            size,
            benchmark_dir,
            args.num_theorems,
            args.split,
            eval_mcts_type=eval_mcts,
            shared_corpus=shared_corpus,
            shared_dataloader=shared_dataloader,
        )
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"EVALUATION SUMMARY ({args.split} set)")
    print(f"{'='*90}")
    print(f"{'Run':<45} {'Status':<12} {'Success':<12} {'Rate':<10} {'Time'}")
    print(f"{'-'*90}")

    for r in all_results:
        name = r.get("run_name", "?")
        status = r.get("status", "?")
        if status == "completed":
            succ = f"{r['successful']}/{r['num_theorems']}"
            rate = f"{r['success_rate']:.1%}"
            t = f"{r['time_seconds']/60:.1f}min"
        else:
            succ = rate = t = "—"
        print(f"{name:<45} {status:<12} {succ:<12} {rate:<10} {t}")
    print(f"{'='*90}\n")

    # Save aggregated results
    output_file = args.output_file or str(
        benchmark_dir / f"eval_summary_{args.split}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "split": args.split,
                "num_theorems": args.num_theorems,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
