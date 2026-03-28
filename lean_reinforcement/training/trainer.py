import os
import sys
import time
import json
import glob
import pickle
import random
import queue
import numpy as np
from typing import List, Dict, Any, Optional, Set, cast, Union
from dataclasses import asdict
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from loguru import logger
import wandb

from ReProver.common import Corpus
from ReProver.common import Pos

from lean_reinforcement.utilities.dataloader import LeanDataLoader
from lean_reinforcement.utilities.checkpoint import (
    get_checkpoint_dir,
    get_iteration_checkpoint_dir,
    save_checkpoint,
    load_checkpoint,
)
from lean_reinforcement.utilities.analyze_training_data import (
    analyze_value_data,
    print_training_stats,
    save_training_data,
)
from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.value_head import ValueHead, HyperbolicValueHead
from lean_reinforcement.agent.ppo_agent import PPOAgent
from lean_reinforcement.training.datasets import ValueHeadDataset
from lean_reinforcement.training.inference_server import InferenceServer
from lean_reinforcement.training.progress import (
    ProgressStats,
    make_progress_display,
)
from lean_reinforcement.training.worker import worker_loop
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    configure_glibc_env_for_children,
    empty_gpu_cache,
    get_available_memory_gb,
    log_gpu_memory,
    set_oom_score_adj,
    MAX_WORKER_RSS_GB,
    RSS_WATCHDOG_EXIT_CODE,
    TRAINER_MIN_AVAILABLE_GB,
)
from lean_reinforcement.utilities.optimizer import unwrap_optimizer_params
from lean_reinforcement.utilities.gym import (
    LeanDojoEnv,
    is_outdated_traced_repo_error,
)


# In-process caches to avoid repeating expensive data setup between trials.
_CORPUS_CACHE: Dict[str, Corpus] = {}
_PREFLIGHT_DONE: Set[str] = set()


def _safe_wandb_log(data: Dict[str, Any]) -> None:
    """Log metrics to wandb, silently ignoring connection errors.

    The wandb-core process may be killed by OOM or an external monitor;
    we catch the resulting errors since local metrics are already saved.
    """
    try:
        wandb.log(data)
    except (ConnectionResetError, BrokenPipeError, OSError) as exc:
        logger.warning(f"wandb.log() failed (connection lost): {exc}")
    except Exception as exc:
        logger.warning(f"wandb.log() failed unexpectedly: {exc}")


def _state_value_to_plain_tensor(value: Any) -> torch.Tensor:
    """Unwrap state_dict values that may be manifold-backed tensors."""
    raw = getattr(value, "tensor", value)
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Expected tensor-like state value, got {type(raw)!r}")
    return raw


class Trainer:
    def __init__(self, config: TrainingConfig, *, reuse_data_cache: bool = True):
        self.config = config
        self.reuse_data_cache = reuse_data_cache

        if config.debugging:
            # Make logs maximally informative for troubleshooting runs.
            logger.remove()
            logger.add(
                sys.stderr,
                level="DEBUG",
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
                    "pid={process.id} | {name}:{function}:{line} - {message}"
                ),
                backtrace=True,
                diagnose=True,
            )
            logger.debug("Debugging mode enabled: verbose logging active")

        # Live progress display
        self.progress_stats = ProgressStats(
            total_epochs=config.num_epochs,
            total_workers=config.num_workers,
            cumulative_total_theorems=config.num_epochs * config.num_theorems,
        )
        self.progress_display = make_progress_display(
            self.progress_stats, enable_live=not config.debugging
        )

        # Set global random seeds for reproducibility
        if config.seed is not None:
            self._set_seeds(config.seed)
            logger.info(f"Global random seed set to {config.seed}")

        # Get the base checkpoint directory, then create iteration-specific subdirectory
        base_checkpoint_dir = get_checkpoint_dir()
        self.checkpoint_dir = get_iteration_checkpoint_dir(
            base_checkpoint_dir, config.mcts_type, resume=config.resume
        )

        # Setup wandb
        if self.config.use_wandb:
            wandb.init(
                entity="gerbennkoopman-university-of-amsterdam",
                project="lean-reinforcement",
                config=asdict(self.config),
            )

        self._setup_models()
        self._setup_data()
        self._setup_multiprocessing()

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_data(self) -> None:
        logger.info(f"Loading data from 'leandojo_benchmark_4/{self.config.data_type}'")

        dataset_path = "leandojo_benchmark_4"
        preflight_cache_key = (
            f"{os.path.abspath(dataset_path)}::{self.config.data_type}"
        )

        # Fast fail for stale LeanDojo traces (common on shared HPC caches).
        # We probe one theorem before constructing the heavy Corpus object.
        if (not self.reuse_data_cache) or (preflight_cache_key not in _PREFLIGHT_DONE):
            probe_loader = LeanDataLoader(
                corpus=None,
                dataset_path=dataset_path,
                data_type=self.config.data_type,
            )
            self._preflight_lean_dojo_trace(probe_loader)
            if self.reuse_data_cache:
                _PREFLIGHT_DONE.add(preflight_cache_key)
        else:
            logger.debug(
                "Skipping LeanDojo preflight (already completed for this dataset/type)."
            )

        if self.config.indexed_corpus_path and os.path.exists(
            self.config.indexed_corpus_path
        ):
            indexed_path = os.path.abspath(self.config.indexed_corpus_path)
            corpus_cache_key = f"indexed::{indexed_path}"
            if self.reuse_data_cache and corpus_cache_key in _CORPUS_CACHE:
                logger.info(f"Reusing cached indexed corpus from {indexed_path}")
                self.corpus = _CORPUS_CACHE[corpus_cache_key]
            else:
                logger.info(f"Loading indexed corpus from {indexed_path}")
                with open(indexed_path, "rb") as f:
                    indexed_corpus = pickle.load(f)
                self.corpus = indexed_corpus.corpus
                if self.reuse_data_cache:
                    _CORPUS_CACHE[corpus_cache_key] = self.corpus
        else:
            corpus_path = os.path.abspath(os.path.join(dataset_path, "corpus.jsonl"))
            corpus_cache_key = f"jsonl::{corpus_path}"
            if self.reuse_data_cache and corpus_cache_key in _CORPUS_CACHE:
                logger.info(f"Reusing cached corpus from {corpus_path}")
                self.corpus = _CORPUS_CACHE[corpus_cache_key]
            else:
                self.corpus = Corpus(corpus_path)
                if self.reuse_data_cache:
                    _CORPUS_CACHE[corpus_cache_key] = self.corpus

        self.dataloader = LeanDataLoader(
            self.corpus,
            dataset_path=dataset_path,
            data_type=self.config.data_type,
        )

    def _preflight_lean_dojo_trace(self, dataloader: LeanDataLoader) -> None:
        """Validate that LeanDojo traced repo cache is compatible before training."""
        if not dataloader.train_data:
            return

        sample = dataloader.train_data[0]
        theorem = dataloader.extract_theorem(sample)
        if theorem is None:
            logger.warning("Preflight skipped: could not extract theorem from dataset.")
            return

        try:
            theorem_pos = Pos(*sample["start"])
        except Exception:
            logger.warning(
                f"Preflight skipped: invalid theorem position for {theorem.full_name}."
            )
            return

        env: Optional[LeanDojoEnv] = None
        try:
            env = LeanDojoEnv(theorem, theorem_pos, self.config.env_timeout)
            logger.info("LeanDojo preflight check passed.")
        except Exception as exc:
            if is_outdated_traced_repo_error(exc):
                raise RuntimeError(
                    "LeanDojo traced repo cache is incompatible with the installed "
                    "LeanDojo version (missing Lean4Repl.lean). "
                    "Clear/rebuild LeanDojo traces in this environment and rerun. "
                    "For details, run: python3 -m "
                    "lean_reinforcement.utilities.lean_cache_diagnostics"
                ) from exc
            raise RuntimeError(
                f"LeanDojo preflight failed for theorem {theorem.full_name}: {exc}"
            ) from exc
        finally:
            if env is not None:
                env.close()

    def _setup_multiprocessing(self) -> None:
        mp.set_start_method("spawn", force=True)
        self.request_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.theorem_queue: mp.Queue = mp.Queue()
        self.response_queues: List[mp.Queue] = [
            mp.Queue() for _ in range(self.config.num_workers)
        ]
        self.workers: List[mp.Process] = []

    def train(self) -> List[Dict[str, Any]]:
        """
        Runs the training loop.
        Returns a list of metrics for each epoch.
        """
        # Protect the desktop: make this process the preferred OOM-kill
        # target so the Linux OOM killer terminates training instead of
        # the desktop environment (GNOME Shell / gdm).
        set_oom_score_adj(1000)

        # --- Memory preflight check ---
        try:
            import psutil

            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            estimated_gb = 6.0 + self.config.num_workers * 3.0
            if estimated_gb > total_ram_gb * 0.85:
                logger.warning(
                    f"Memory preflight: estimated workload ~{estimated_gb:.0f} GB "
                    f"exceeds 85% of system RAM ({total_ram_gb:.0f} GB). "
                    f"Consider reducing --num-workers from {self.config.num_workers} "
                    f"to {max(1, int((total_ram_gb * 0.85 - 6) / 3))}."
                )
        except ImportError:
            pass

        all_metrics = []
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
            logger.info("Training interrupted by user.")
            return all_metrics
        except Exception as e:
            logger.error(f"Training crashed: {e}")
            raise e
        finally:
            self.progress_display.stop()
            self._drain_queues()
            self._cleanup_workers()

    def _setup_models(self) -> None:
        logger.info(f"Using checkpoint directory: {self.checkpoint_dir}")

        self.transformer = Transformer(model_name=self.config.model_name)

        self.value_head: Optional[ValueHead | HyperbolicValueHead] = None
        self.ppo_agent: Optional[PPOAgent] = None
        self.start_epoch = 0

        if self.config.training_mode == "ppo":
            logger.info("Initializing PPO agent")
            self.ppo_agent = PPOAgent(
                model_name=self.config.model_name,
                use_hyperbolic=self.config.use_hyperbolic,
            )
            if self.config.resume:
                self.start_epoch = self.ppo_agent.load_latest_checkpoint(
                    self.checkpoint_dir
                )
                logger.info(f"Resuming PPO training from epoch {self.start_epoch}")

        elif self.config.mcts_type == "alpha_zero" or self.config.train_value_head:
            transformer_for_value_head = cast(Transformer, self.transformer)

            if self.config.use_hyperbolic:
                logger.info("Using hyperbolic (Poincaré ball) value head")
                self.value_head = HyperbolicValueHead(
                    transformer_for_value_head,
                )
            else:
                self.value_head = ValueHead(
                    transformer_for_value_head,
                    latent_dim=self.config.value_head_latent_dim,
                )

            if self.config.resume or self.config.use_test_value_head:
                if self.config.use_test_value_head:
                    prefix = "value_head_test"
                else:
                    prefix = f"value_head_{self.config.mcts_type}"

                loaded_epoch = load_checkpoint(
                    self.value_head, self.checkpoint_dir, prefix=prefix
                )

                if self.config.resume:
                    self.start_epoch = loaded_epoch
                    logger.info(f"Resuming training from epoch {self.start_epoch}")
                else:
                    logger.info(
                        f"Initialized value head from {prefix} (epoch {loaded_epoch})"
                    )
                    self.start_epoch = 0

        log_gpu_memory(logger, prefix="After model initialization - ", level=20)

    def _run_epoch(
        self, epoch: int, inference_server: InferenceServer
    ) -> List[Dict[str, Any]]:
        # Drain any leftover theorems from previous epochs before starting
        self._drain_theorem_queue()

        self._start_workers()
        logger.info(
            f"Starting Epoch {epoch + 1}/{self.start_epoch + self.config.num_epochs}"
        )

        empty_gpu_cache()

        # Make a defensive copy to avoid mutating the original
        # This prevents issues where shuffle or other operations affect subsequent epochs
        epoch_data = list(self.dataloader.train_data)

        # Seed the shuffle per-epoch for reproducibility
        if self.config.seed is not None:
            epoch_seed = self.config.seed + epoch
            random.seed(epoch_seed)

        random.shuffle(epoch_data)
        theorems_to_process = epoch_data[: self.config.num_theorems]

        theorems_queued = 0
        for thm in theorems_to_process:
            self.theorem_queue.put(thm)
            theorems_queued += 1

        # Update progress display for this epoch
        self.progress_stats.reset_epoch(
            epoch=epoch + 1, total_theorems=len(theorems_to_process)
        )
        self.progress_stats.total_epochs = self.start_epoch + self.config.num_epochs
        self.progress_stats.alive_workers = self.config.num_workers
        self.progress_display.refresh()

        training_data_buffer, epoch_metrics = self._collect_data(
            theorems_to_process, inference_server, epoch
        )

        self._stop_workers()
        self._drain_queues()

        empty_gpu_cache()

        if not training_data_buffer:
            logger.warning("No data collected in this epoch. Skipping training.")
            return epoch_metrics

        self.progress_stats.phase = "saving"
        self.progress_display.refresh()
        self._analyze_and_save_data(training_data_buffer, epoch)

        if self.config.training_mode == "ppo":
            self.progress_stats.phase = "training_ppo"
            self.progress_display.refresh()
            self._train_ppo_epoch(training_data_buffer, epoch)

        elif self.config.train_value_head:
            self.progress_stats.phase = "training_value_head"
            self.progress_display.refresh()
            self._train_value_head_epoch(training_data_buffer)

            # Save the best validation loss for this epoch
            best_val_loss = getattr(self, "_last_best_val_loss", None)
            if best_val_loss is not None:
                val_loss_file = self.checkpoint_dir / f"val_loss_epoch_{epoch + 1}.json"
                try:
                    with open(val_loss_file, "w") as f:
                        json.dump(
                            {"epoch": epoch + 1, "best_val_loss": best_val_loss}, f
                        )
                except Exception as e:
                    logger.error(f"Failed to save val loss: {e}")

        if self.config.save_checkpoints:
            if self.config.training_mode == "ppo" and self.ppo_agent:
                self.ppo_agent.save_checkpoint(self.checkpoint_dir, epoch + 1)
                logger.info(f"PPO checkpoint saved for epoch {epoch + 1}")
            elif self.config.train_value_head and self.value_head:
                prefix = f"value_head_{self.config.mcts_type}"
                save_checkpoint(
                    self.value_head,
                    epoch + 1,
                    self.checkpoint_dir,
                    self.config,
                    prefix=prefix,
                )
                logger.info(f"Value head checkpoint saved for epoch {epoch + 1}")

        return epoch_metrics

    def _train_ppo_epoch(self, training_data_buffer: List[Dict[str, Any]], epoch: int):
        assert self.ppo_agent is not None, "PPO agent must be initialized"
        logger.info("Updating PPO agent...")
        ppo_metrics = self.ppo_agent.update(training_data_buffer)
        logger.info(f"PPO update complete. Metrics: {ppo_metrics}")
        if self.config.use_wandb:
            _safe_wandb_log(ppo_metrics)

    def _start_workers(self) -> None:
        # Set glibc tuning env vars in the parent BEFORE spawning.
        # Children inherit the environment; glibc in the child reads
        # MALLOC_ARENA_MAX at C-library init (before Python’s main()).
        # This is the only reliable way to get the env-var path working.
        # Workers also call mallopt() directly as a belt-and-suspenders.
        configure_glibc_env_for_children()

        logger.info(f"Starting {self.config.num_workers} workers")
        self.workers = []
        for i in range(self.config.num_workers):
            p = mp.Process(
                target=worker_loop,
                args=(
                    i,
                    self.request_queue,
                    self.response_queues[i],
                    self.theorem_queue,
                    self.result_queue,
                    self.config,
                    self.checkpoint_dir,
                ),
            )
            p.start()
            self.workers.append(p)

    def _stop_workers(self) -> None:
        logger.info("Stopping workers for this epoch...")
        for _ in range(self.config.num_workers):
            self.theorem_queue.put(None)

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} did not exit gracefully. Terminating.")
                p.terminate()
                p.join()
        self.workers = []

    def _cleanup_workers(self) -> None:
        logger.info("Shutting down workers...")
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join()

    def _drain_theorem_queue(self) -> None:
        """Drain theorem queue to ensure clean state before new epoch."""
        drained = 0
        try:
            while not self.theorem_queue.empty():
                self.theorem_queue.get_nowait()
                drained += 1
        except Exception:
            pass
        if drained > 0:
            logger.warning(f"Drained {drained} leftover theorems from queue")

    def _drain_queues(self) -> None:
        logger.info("Draining queues...")

        # Drain theorem queue (important: prevents leftover theorems from previous epochs)
        try:
            while not self.theorem_queue.empty():
                self.theorem_queue.get_nowait()
        except Exception:
            pass

        try:
            while not self.request_queue.empty():
                self.request_queue.get_nowait()
        except Exception:
            pass

        for q in self.response_queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass

        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except Exception:
            pass

    def _collect_data(
        self,
        theorems_to_process: List[Any],
        inference_server: InferenceServer,
        epoch: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Collect proof attempts from workers.

        Primary approach: Count actual results from the queue until we have
        results for all theorems.

        Returns: (training_data_buffer, epoch_metrics)

        Exit conditions:
        1. Received results for all theorems
        2. All workers dead + no new results for 60 seconds (queue drained)
        3. Safety timeout (4 hours)
        4. Stall timeout (no progress for too long)
        """
        total_theorems = len(theorems_to_process)
        results_received = 0
        training_data_buffer: List[Dict[str, Any]] = []
        collected_metrics: List[Dict[str, Any]] = []
        logged_dead_workers: set = set()  # Track workers we've already logged as dead

        # Track per-theorem results for detailed logging
        theorem_results: List[Dict[str, Any]] = []

        temp_data_file = self.checkpoint_dir / f"temp_data_epoch_{epoch + 1}.jsonl"

        # Ensure checkpoint directory exists before writing temporary files
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback: try creating parent directory path via os.makedirs
            os.makedirs(str(self.checkpoint_dir), exist_ok=True)

        if os.path.exists(temp_data_file):
            os.remove(temp_data_file)

        epoch_start_time = time.time()
        last_result_time = time.time()
        last_activity_time = time.time()  # Tracks ANY progress (results or inference)

        # Dynamic safety timeout based on config:
        # Worst case = every theorem takes the full proof_timeout, processed
        # in parallel across num_workers, plus a 2x safety buffer for
        # inference overhead, worker startup, and value head training.
        num_batches = max(1, total_theorems / max(1, self.config.num_workers))
        max_epoch_time = num_batches * self.config.proof_timeout * 2
        # Clamp to a reasonable range: at least 30 min, at most 12 hours
        max_epoch_time = max(1800, min(max_epoch_time, 12 * 3600))

        # How long to wait for more results after all workers die.
        # Increased from 60s to allow time for worker restart and initialization,
        # especially on systems with slower resource allocation.
        drain_timeout = 120  # seconds

        # Hard guard against apparent hangs: if no theorem result has arrived
        # for a long timeim, break even if inference requests are still trickling.
        base_no_result_timeout = max(int(self.config.proof_timeout * 2), 600)
        endgame_no_result_timeout = max(int(self.config.proof_timeout + 120), 300)

        # Track worker crashes to detect systemic issues
        total_worker_crashes = 0

        # Stall timeout: if no activity (no results AND no inference processing)
        # for this long, assume workers are truly stuck.
        # During alpha_zero evaluation, the inference server being busy is a
        # signal that workers are still computing, even if no theorem results
        # have arrived yet. So we track inference activity separately.
        stall_timeout = getattr(self, "stall_timeout_override", None) or (
            max(self.config.proof_timeout, self.config.inference_timeout) * 3
        )

        logger.debug(
            f"Collecting results for {total_theorems} theorems "
            f"(safety timeout: {max_epoch_time/3600:.1f}h, stall timeout: {stall_timeout/60:.0f}m)"
        )

        while results_received < total_theorems:
            elapsed = time.time() - epoch_start_time

            # Safety net timeout
            if elapsed > max_epoch_time:
                logger.warning(
                    f"Safety timeout reached ({elapsed/3600:.1f}h). "
                    f"Collected {results_received}/{total_theorems} results. "
                    f"Proceeding with training."
                )
                break

            # --- Memory pressure check ---
            # When system-wide available memory is low, pause briefly to
            # let workers finish and free memory.  Uses absolute GB
            # rather than percentage for reliability across system sizes.
            avail = get_available_memory_gb()
            if avail < TRAINER_MIN_AVAILABLE_GB:
                import psutil

                own_rss_gb = psutil.Process().memory_info().rss / (1024**3)
                logger.warning(
                    f"Main process: low memory "
                    f"(avail={avail:.1f} GB, own RSS={own_rss_gb:.2f} GB). "
                    f"Pausing to let workers release memory."
                )
                aggressive_cleanup()
                empty_gpu_cache()
                # Brief pause to let workers finish and release memory
                time.sleep(2.0)
                # NOTE: Do NOT skip inference processing here!
                # Skipping causes all workers to time out simultaneously,
                # leading to response queue desync and cascading errors.

            # Process inference requests (keeps GPU busy)
            processed_batch = inference_server.process_requests()
            if processed_batch:
                last_activity_time = time.time()

            # Collect results from workers
            got_result = False
            try:
                while True:
                    res = self.result_queue.get_nowait()
                    results_received += 1
                    got_result = True
                    last_result_time = time.time()
                    last_activity_time = time.time()

                    if res:
                        if "metrics" in res:
                            # Always collect metrics
                            collected_metrics.append(res["metrics"])
                            if self.config.use_wandb:
                                _safe_wandb_log(res["metrics"])

                            # Track per-theorem success/failure with name
                            theorem_name = res.get(
                                "theorem_name", f"unknown_{results_received}"
                            )
                            success = res["metrics"].get("proof_search/success", False)
                            theorem_results.append(
                                {
                                    "theorem_name": theorem_name,
                                    "success": success,
                                    "steps": res["metrics"].get(
                                        "proof_search/steps", 0
                                    ),
                                    "time": res["metrics"].get(
                                        "proof_search/time", 0.0
                                    ),
                                }
                            )
                            # Update live progress display
                            self.progress_stats.record_theorem(
                                name=theorem_name,
                                success=success,
                                steps=res["metrics"].get("proof_search/steps", 0),
                                elapsed=res["metrics"].get("proof_search/time", 0.0),
                                metrics=res["metrics"],
                            )
                            self.progress_display.refresh()

                            if self.config.debugging:
                                steps = res["metrics"].get("proof_search/steps", 0)
                                elapsed_s = res["metrics"].get("proof_search/time", 0.0)
                                worker_id = res.get("worker_id", "?")
                                failure_reasons = [
                                    k.replace("proof_search/", "")
                                    for k, v in res["metrics"].items()
                                    if k.startswith("proof_search/")
                                    and k
                                    not in {
                                        "proof_search/success",
                                        "proof_search/steps",
                                        "proof_search/time",
                                    }
                                    and bool(v)
                                ]
                                reason_str = (
                                    f" | reasons={','.join(failure_reasons)}"
                                    if failure_reasons
                                    else ""
                                )
                                msg = (
                                    f"Theorem {results_received}/{total_theorems}: "
                                    f"{theorem_name} | worker={worker_id} | "
                                    f"steps={steps} | time={elapsed_s:.2f}s{reason_str}"
                                )
                                if success:
                                    logger.info(f"[SUCCESS] {msg}")
                                else:
                                    logger.warning(f"[FAILED] {msg}")

                        if "data" in res:
                            data = res["data"]
                            training_data_buffer.extend(data)

                            # Track positive and negative samples
                            if data and self.config.use_wandb:
                                positive_count = sum(
                                    1
                                    for item in data
                                    if item.get("value_target", 0) > 0
                                )
                                negative_count = sum(
                                    1
                                    for item in data
                                    if item.get("value_target", 0) < 0
                                )
                                _safe_wandb_log(
                                    {
                                        "training_data/positive_samples": positive_count,
                                        "training_data/negative_samples": negative_count,
                                    }
                                )
                        elif isinstance(res, list):
                            training_data_buffer.extend(res)

                        # Periodically save to disk to avoid memory issues
                        if len(training_data_buffer) > 100:
                            with open(temp_data_file, "a") as f:
                                for item in training_data_buffer:
                                    f.write(json.dumps(item) + "\n")
                            training_data_buffer = []

                    # Progress is shown in the live display (updated above)
            except queue.Empty:
                pass

            if not processed_batch and not got_result:
                time.sleep(0.01)

            # Check worker status
            alive_workers = [p for p in self.workers if p.is_alive()]
            dead_workers = [
                (i, p) for i, p in enumerate(self.workers) if not p.is_alive()
            ]
            self.progress_stats.alive_workers = len(alive_workers)
            self.progress_display.refresh()

            # Log dead workers and restart crashed/recycled ones (once per worker)
            for i, p in dead_workers:
                if p.pid not in logged_dead_workers:
                    logged_dead_workers.add(p.pid)

                    # RSS watchdog kills (exit code 42) and crashes
                    # (any non-zero) are both restartable.  Only
                    # graceful exit (code 0) means the worker was told
                    # to stop via the None sentinel.
                    is_watchdog_kill = p.exitcode == RSS_WATCHDOG_EXIT_CODE
                    is_crash = p.exitcode not in (0, None)

                    if is_crash:
                        total_worker_crashes += 1
                        if is_watchdog_kill:
                            logger.warning(
                                f"Worker {i} (PID: {p.pid}) recycled by RSS watchdog "
                                f"(exceeded {MAX_WORKER_RSS_GB:.0f} GB). "
                                f"Total recycles so far: {total_worker_crashes}. "
                                f"Restarting with clean address space."
                            )
                            # Watchdog recycling happens AFTER the worker sends its result,
                            # so we already counted this theorem. Don't double-count.
                        else:
                            logger.warning(
                                f"Worker {i} (PID: {p.pid}) crashed (exit code: {p.exitcode}). "
                                f"Total crashes so far: {total_worker_crashes}. "
                                f"Restarting worker and skipping current theorem."
                            )
                            # True crashes happen during processing, before sending result.
                            # Count as a failed completion to keep results_received reachable.
                            results_received += 1
                            last_result_time = time.time()
                            last_activity_time = time.time()
                            self.progress_stats.record_theorem(
                                name=f"worker_{i}_crash",
                                success=False,
                                failure_reason="worker crash",
                            )
                            self.progress_display.refresh()

                        # Terminate the old process object to be safe
                        p.join(timeout=1)

                        # Start a new worker with the same ID
                        new_worker = mp.Process(
                            target=worker_loop,
                            args=(
                                i,
                                self.request_queue,
                                self.response_queues[i],
                                self.theorem_queue,
                                self.result_queue,
                                self.config,
                                self.checkpoint_dir,
                            ),
                        )
                        new_worker.start()
                        self.workers[i] = new_worker
                        # Reset last_result_time to give the restarted worker time to
                        # initialize and process its next theorem without triggering drain timeout
                        last_result_time = time.time()
                        logger.info(f"Worker {i} restarted successfully.")
                    else:
                        # Graceful exit (exit code 0)
                        logger.debug(
                            f"Worker {i} (PID: {p.pid}) exited gracefully. "
                            f"{len(alive_workers)} workers still alive."
                        )

            # Check for stall - no activity (no results AND no inference)
            # for too long. Inference activity means workers are still
            # computing even if no theorem results have arrived yet.
            time_since_last_activity = time.time() - last_activity_time
            if time_since_last_activity > stall_timeout:
                logger.warning(
                    f"No activity for {time_since_last_activity/60:.1f} minutes "
                    f"(stall timeout: {stall_timeout/60:.0f}m). "
                    f"Collected {results_received}/{total_theorems} results. "
                    f"Proceeding with training."
                )
                break

            # Independent no-result guard: catches alive-but-idle states where
            # no theorem results are produced for an extended period.
            remaining_theorems = total_theorems - results_received
            no_result_timeout = base_no_result_timeout
            if remaining_theorems <= max(1, self.config.num_workers // 2):
                no_result_timeout = min(no_result_timeout, endgame_no_result_timeout)

            time_since_last_result = time.time() - last_result_time
            if time_since_last_result > no_result_timeout:
                logger.warning(
                    f"No theorem results for {time_since_last_result/60:.1f} minutes "
                    f"(limit: {no_result_timeout/60:.0f}m). "
                    f"Collected {results_received}/{total_theorems} results "
                    f"({remaining_theorems} remaining). "
                    f"Total worker crashes in this epoch: {total_worker_crashes}. "
                    f"Proceeding with training from partial data."
                )
                break

            # If all workers are dead, wait a bit for queue to drain, then exit
            if not alive_workers:
                time_since_last_result = time.time() - last_result_time
                if time_since_last_result > drain_timeout:
                    logger.warning(
                        f"All workers dead and no results for {drain_timeout}s. "
                        f"Collected {results_received}/{total_theorems} results "
                        f"({results_received*100//total_theorems}% complete). "
                        f"Total worker crashes in this epoch: {total_worker_crashes}. "
                        f"Proceeding with training from partial data."
                    )
                    break

        # Final stats
        elapsed = time.time() - epoch_start_time
        logger.info(
            f"Data collection complete: {results_received}/{total_theorems} "
            f"results in {elapsed/60:.1f} min"
        )

        # Load back temp data
        if os.path.exists(temp_data_file):
            if training_data_buffer:
                with open(temp_data_file, "a") as f:
                    for item in training_data_buffer:
                        f.write(json.dumps(item) + "\n")
                training_data_buffer = []

            logger.info("Loading training data from temporary file...")
            with open(temp_data_file, "r") as f:
                for line in f:
                    training_data_buffer.append(json.loads(line))
            os.remove(temp_data_file)

        # Log and save per-theorem results summary
        if theorem_results:
            proved = [t for t in theorem_results if t["success"]]
            failed = [t for t in theorem_results if not t["success"]]
            sr = len(proved) / len(theorem_results) * 100 if theorem_results else 0
            logger.info(
                f"Epoch {epoch + 1} summary: "
                f"{len(proved)}/{len(theorem_results)} proved ({sr:.1f}%) "
                f"in {elapsed/60:.1f} min"
            )

            # Save theorem results to checkpoint directory
            results_file = (
                self.checkpoint_dir / f"theorem_results_epoch_{epoch + 1}.json"
            )
            try:
                with open(results_file, "w") as f:
                    json.dump(
                        {
                            "epoch": epoch + 1,
                            "seed": self.config.seed,
                            "mcts_type": self.config.mcts_type,
                            "total": len(theorem_results),
                            "proved": len(proved),
                            "failed": len(failed),
                            "success_rate": (
                                len(proved) / len(theorem_results)
                                if theorem_results
                                else 0
                            ),
                            "theorems": theorem_results,
                        },
                        f,
                        indent=2,
                    )
                logger.info(f"Theorem results saved to {results_file}")
            except Exception as e:
                logger.error(f"Failed to save theorem results: {e}")

        # Warn if collection exited significantly early
        if results_received < total_theorems * 0.8:
            logger.critical(
                f"**WARNING** Epoch {epoch+1} exited with only {results_received}/{total_theorems} results ({100*results_received//total_theorems}%). "
                f"This is likely due to worker crashes or system issues. "
                f"Total crashes this epoch: {total_worker_crashes}. "
                f"Value head training will use incomplete data. "
                f"Consider investigating worker logs in logs/ directory."
            )
        elif results_received < total_theorems:
            logger.warning(
                f"Epoch {epoch+1} collected {results_received}/{total_theorems} results ({100*results_received//total_theorems}%). "
                f"Total crashes: {total_worker_crashes}."
            )

        return training_data_buffer, collected_metrics

    def _analyze_and_save_data(
        self, training_data_buffer: List[Dict[str, Any]], epoch: int
    ):
        if self.config.train_value_head:
            stats = analyze_value_data(training_data_buffer)
            print_training_stats(stats)

            if self.config.save_training_data:
                data_save_path = (
                    self.checkpoint_dir / f"training_data_epoch_{epoch + 1}.json"
                )
                save_training_data(training_data_buffer, data_save_path)

    def _load_experience_replay_data(self) -> List[Dict[str, Any]]:
        """
        Load all training data from previous epochs for experience replay.
        Returns a combined list of all value training samples.
        """
        all_data: List[Dict[str, Any]] = []
        data_files = sorted(
            glob.glob(str(self.checkpoint_dir / "training_data_epoch_*.json"))
        )

        if not data_files:
            logger.info("No previous training data found for experience replay.")
            return all_data

        logger.info(
            f"Loading experience replay data from {len(data_files)} epoch files..."
        )

        for filepath in data_files:
            try:
                with open(filepath, "r") as f:
                    epoch_data = json.load(f)
                    value_samples = [d for d in epoch_data if d.get("type") == "value"]
                    all_data.extend(value_samples)
                    logger.debug(
                        f"  Loaded {len(value_samples)} samples from {os.path.basename(filepath)}"
                    )
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")

        logger.info(f"Total experience replay samples: {len(all_data)}")
        return all_data

    def _train_value_head_epoch(self, training_data_buffer: List[Dict[str, Any]]):
        current_value_data = [
            d for d in training_data_buffer if d.get("type") == "value"
        ]

        replay_data = self._load_experience_replay_data()

        combined_data = replay_data + current_value_data

        seen_states = {}
        for item in combined_data:
            state = item.get("state", "")
            seen_states[state] = item

        unique_data = list(seen_states.values())
        logger.info(
            f"Experience replay: {len(replay_data)} replay + "
            f"{len(current_value_data)} current = "
            f"{len(unique_data)} unique samples"
        )

        assert (
            self.value_head is not None
        ), "ValueHead must be initialized before training"

        self._train_value_head_model(
            self.value_head,
            unique_data,
            epochs=self.config.train_epochs,
            batch_size=self.config.value_head_batch_size,
            use_wandb=self.config.use_wandb,
        )

    def _train_value_head_model(
        self,
        value_head: Union[ValueHead, HyperbolicValueHead],
        data_buffer: List[Dict[str, Any]],
        epochs: int = 50,
        batch_size: int = 32,
        use_wandb: bool = True,
        val_split: float = 0.1,
        patience: int = 5,
    ):
        if not data_buffer:
            logger.warning("Value Head training skipped: No data provided.")
            return

        value_targets = [item["value_target"] for item in data_buffer]
        avg_target = sum(value_targets) / len(value_targets)
        positive_samples = sum(1 for v in value_targets if v > 0)
        negative_samples = sum(1 for v in value_targets if v < 0)

        logger.info(
            f"Training Value Head: {len(data_buffer)} samples "
            f"({positive_samples}+ / {negative_samples}-), avg target {avg_target:.4f}"
        )

        avg_mcts = None
        if "mcts_value" in data_buffer[0]:
            mcts_values = [item["mcts_value"] for item in data_buffer]
            avg_mcts = sum(mcts_values) / len(mcts_values)
            logger.debug(f"  Average MCTS value estimate: {avg_mcts:.4f}")

        random.shuffle(data_buffer)
        val_size = int(len(data_buffer) * val_split)

        if val_size < 1:
            logger.warning(
                "Dataset too small for validation split. Using all data for training."
            )
            train_data = data_buffer
            val_data = None
        else:
            val_data = data_buffer[:val_size]
            train_data = data_buffer[val_size:]
            logger.debug(
                f"  Train/Val split: {len(train_data)} train, {len(val_data)} val (unbalanced)"
            )

        positive_data = [item for item in train_data if item["value_target"] > 0]
        negative_data = [item for item in train_data if item["value_target"] < 0]

        if positive_data and negative_data:
            min_count = min(len(positive_data), len(negative_data))
            logger.debug(f"  Balancing training set to {min_count} samples per class.")
            random.shuffle(positive_data)
            random.shuffle(negative_data)
            balanced_train_data = positive_data[:min_count] + negative_data[:min_count]
            random.shuffle(balanced_train_data)
        else:
            logger.warning(
                "  Cannot balance training set: One class is missing. Using full training set."
            )
            balanced_train_data = train_data

        if use_wandb:
            _safe_wandb_log(
                {
                    "value_head/avg_target": avg_target,
                    "value_head/positive_samples": positive_samples,
                    "value_head/negative_samples": negative_samples,
                    "value_head/avg_mcts_value": avg_mcts,
                    "value_head/training_samples": len(balanced_train_data),
                    "value_head/validation_samples": len(val_data) if val_data else 0,
                }
            )

        train_dataset = ValueHeadDataset(balanced_train_data)
        val_dataset = ValueHeadDataset(val_data) if val_data else None

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = (
            DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
            )
            if val_dataset
            else None
        )

        optimizer = optim.AdamW(
            unwrap_optimizer_params(value_head.value_head.parameters()),
            lr=1e-4,
        )
        loss_fn = torch.nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Update progress display for value head training phase
        ps = self.progress_stats
        ps.value_head_total_epochs = epochs
        ps.value_head_patience_limit = patience
        ps.phase = "training_value_head"
        self.progress_display.refresh()

        value_head.train()

        epoch = 0
        for epoch in range(epochs):
            value_head.train()
            total_train_loss = 0
            for batch in train_loader:
                states = batch["state"]
                batch_value_targets = batch["value_target"].to(
                    dtype=torch.float32,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                batch_value_targets = torch.clamp(
                    batch_value_targets, min=-0.99, max=0.99
                )
                features = value_head.encode_states(states)
                value_preds = value_head.value_head(features).squeeze()
                loss = loss_fn(value_preds, batch_value_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            if val_loader:
                value_head.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        states = batch["state"]
                        batch_value_targets = batch["value_target"].to(
                            dtype=torch.float32,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
                        batch_value_targets = torch.clamp(
                            batch_value_targets, min=-0.99, max=0.99
                        )
                        features = value_head.encode_states(states)
                        value_preds = value_head.value_head(features).squeeze()
                        loss = loss_fn(value_preds, batch_value_targets)
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)

                # Update progress display
                ps.value_head_epoch = epoch + 1
                ps.value_head_train_loss = avg_train_loss
                ps.value_head_val_loss = avg_val_loss
                ps.value_head_patience_counter = patience_counter
                self.progress_display.refresh()

                if use_wandb:
                    _safe_wandb_log(
                        {
                            "value_head/train_loss": avg_train_loss,
                            "value_head/val_loss": avg_val_loss,
                            "value_head/epoch": epoch + 1,
                        }
                    )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Hyperbolic layers can expose manifold-backed tensors that
                    # fail under deepcopy due to torch function interception.
                    # Snapshot as plain cloned tensors for robust restore.
                    best_model_state = {
                        k: _state_value_to_plain_tensor(v).detach().clone()
                        for k, v in value_head.value_head.state_dict().items()
                    }
                    ps.value_head_best_val_loss = best_val_loss
                else:
                    patience_counter += 1
                    ps.value_head_patience_counter = patience_counter
                    self.progress_display.refresh()

                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                        break
            else:
                ps.value_head_epoch = epoch + 1
                ps.value_head_train_loss = avg_train_loss
                self.progress_display.refresh()

                if use_wandb:
                    _safe_wandb_log(
                        {
                            "value_head/train_loss": avg_train_loss,
                            "value_head/epoch": epoch + 1,
                        }
                    )

        logger.info(
            f"Value Head training done: {epoch+1}/{epochs} epochs, "
            f"best val loss: {best_val_loss:.4f}"
        )

        if best_model_state is not None:
            logger.info("Restoring best model weights from validation.")
            value_head.value_head.load_state_dict(best_model_state)

        # Store best validation loss for later analysis / plotting
        self._last_best_val_loss = (
            best_val_loss if best_val_loss < float("inf") else None
        )

        value_head.eval()
        aggressive_cleanup()
        empty_gpu_cache()
