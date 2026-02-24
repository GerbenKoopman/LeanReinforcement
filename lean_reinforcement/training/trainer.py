import os
import time
import json
import glob
import copy
import pickle
import random
import queue
import gc
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from loguru import logger
import wandb

from ReProver.common import Corpus

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
from lean_reinforcement.agent.value_head import ValueHead
from lean_reinforcement.training.datasets import ValueHeadDataset
from lean_reinforcement.training.inference_server import InferenceServer
from lean_reinforcement.training.progress import (
    ProgressStats,
    make_progress_display,
)
from lean_reinforcement.training.worker import worker_loop
from lean_reinforcement.utilities.config import TrainingConfig


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # Live progress display
        self.progress_stats = ProgressStats(
            total_epochs=config.num_epochs,
            total_workers=config.num_workers,
            cumulative_total_theorems=config.num_epochs * config.num_theorems,
        )
        self.progress_display = make_progress_display(self.progress_stats)

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

    def _setup_models(self) -> None:
        logger.info(f"Using checkpoint directory: {self.checkpoint_dir}")
        self.transformer = Transformer(model_name=self.config.model_name)

        self.value_head: Optional[ValueHead] = None
        self.start_epoch = 0

        if self.config.mcts_type == "alpha_zero" or self.config.train_value_head:
            self.value_head = ValueHead(
                self.transformer, hidden_dims=self.config.value_head_hidden_dims
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

        self._log_gpu_memory("After model initialization - ")

    def _setup_data(self) -> None:
        logger.info(f"Loading data from 'leandojo_benchmark_4/{self.config.data_type}'")

        if self.config.indexed_corpus_path and os.path.exists(
            self.config.indexed_corpus_path
        ):
            logger.info(
                f"Loading indexed corpus from {self.config.indexed_corpus_path}"
            )
            with open(self.config.indexed_corpus_path, "rb") as f:
                indexed_corpus = pickle.load(f)
            self.corpus = indexed_corpus.corpus
        else:
            corpus_path = os.path.join("leandojo_benchmark_4/corpus.jsonl")
            self.corpus = Corpus(corpus_path)

        self.dataloader = LeanDataLoader(
            self.corpus,
            dataset_path="leandojo_benchmark_4",
            data_type=self.config.data_type,
        )

    def _setup_multiprocessing(self) -> None:
        mp.set_start_method("spawn", force=True)
        self.request_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.theorem_queue: mp.Queue = mp.Queue()
        self.response_queues: List[mp.Queue] = [
            mp.Queue() for _ in range(self.config.num_workers)
        ]
        self.workers: List[mp.Process] = []

    def _log_gpu_memory(self, prefix: str = "") -> None:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

    def train(self) -> List[Dict[str, Any]]:
        """
        Runs the training loop.
        Returns a list of metrics for each epoch.
        """
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

    def _run_epoch(
        self, epoch: int, inference_server: InferenceServer
    ) -> List[Dict[str, Any]]:
        # Drain any leftover theorems from previous epochs before starting
        self._drain_theorem_queue()

        self._start_workers()
        logger.info(
            f"Starting Epoch {epoch + 1}/{self.start_epoch + self.config.num_epochs}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Seed the shuffle per-epoch for reproducibility
        if self.config.seed is not None:
            epoch_seed = self.config.seed + epoch
            random.seed(epoch_seed)

        random.shuffle(self.dataloader.train_data)
        theorems_to_process = self.dataloader.train_data[: self.config.num_theorems]

        for thm in theorems_to_process:
            self.theorem_queue.put(thm)

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not training_data_buffer:
            logger.warning("No data collected in this epoch. Skipping training.")
            return epoch_metrics

        self.progress_stats.phase = "saving"
        self.progress_display.refresh()
        self._analyze_and_save_data(training_data_buffer, epoch)

        if self.config.train_value_head:
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

        if (
            self.config.train_value_head
            and self.value_head is not None
            and self.config.save_checkpoints
        ):
            prefix = f"value_head_{self.config.mcts_type}"
            save_checkpoint(
                self.value_head,
                epoch + 1,
                self.checkpoint_dir,
                self.config,
                prefix=prefix,
            )
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")

        return epoch_metrics

    def _start_workers(self) -> None:
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

        # How long to wait for more results after all workers die
        drain_timeout = 60  # seconds

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
                                wandb.log(res["metrics"])

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
                            )
                            self.progress_display.refresh()

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
                                wandb.log(
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

            # Log dead workers and restart crashed ones (once per worker)
            for i, p in dead_workers:
                if p.pid not in logged_dead_workers:
                    logged_dead_workers.add(p.pid)

                    # Check if worker crashed (non-zero exit code)
                    if p.exitcode not in (0, None):
                        logger.warning(
                            f"Worker {i} (PID: {p.pid}) crashed (exit code: {p.exitcode}). "
                            f"Restarting worker and skipping current theorem."
                        )

                        # Mark the lost theorem as "completed" (failed)
                        results_received += 1
                        last_result_time = time.time()
                        last_activity_time = time.time()
                        self.progress_stats.record_theorem(
                            name=f"worker_{i}_crash", success=False
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
                            ),
                        )
                        new_worker.start()
                        self.workers[i] = new_worker
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

            # If all workers are dead, wait a bit for queue to drain, then exit
            if not alive_workers:
                time_since_last_result = time.time() - last_result_time
                if time_since_last_result > drain_timeout:
                    logger.warning(
                        f"All workers dead and no results for {drain_timeout}s. "
                        f"Collected {results_received}/{total_theorems} results. "
                        f"Proceeding with training."
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
        value_head: ValueHead,
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
            wandb.log(
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if val_dataset
            else None
        )

        optimizer = optim.AdamW(value_head.value_head.parameters(), lr=1e-4)
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
                    wandb.log(
                        {
                            "value_head/train_loss": avg_train_loss,
                            "value_head/val_loss": avg_val_loss,
                            "value_head/epoch": epoch + 1,
                        }
                    )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(value_head.value_head.state_dict())
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
                    wandb.log(
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
