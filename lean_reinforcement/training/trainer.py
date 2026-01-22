import os
import time
import json
import pickle
import random
import queue
import gc
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
from lean_reinforcement.training.worker import worker_loop
from lean_reinforcement.utilities.config import TrainingConfig


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_dir = get_checkpoint_dir()

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

    def _setup_models(self) -> None:
        logger.info(f"Using checkpoint directory: {self.checkpoint_dir}")
        self.transformer = Transformer(model_name=self.config.model_name)

        self.value_head: Optional[ValueHead] = None
        self.start_epoch = 0

        if self.config.mcts_type == "alpha_zero" or self.config.train_value_head:
            self.value_head = ValueHead(self.transformer)

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
            self._cleanup_workers()

    def _run_epoch(
        self, epoch: int, inference_server: InferenceServer
    ) -> List[Dict[str, Any]]:
        # Drain any leftover theorems from previous epochs before starting
        self._drain_theorem_queue()

        self._start_workers()
        logger.info(f"Starting Epoch {epoch + 1}/{self.config.num_epochs}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        random.shuffle(self.dataloader.train_data)
        theorems_to_process = self.dataloader.train_data[: self.config.num_theorems]

        for thm in theorems_to_process:
            self.theorem_queue.put(thm)

        logger.info(
            f"Processing {len(theorems_to_process)} theorems with {self.config.num_workers} workers."
        )

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

        self._analyze_and_save_data(training_data_buffer, epoch)

        if self.config.train_value_head:
            self._train_value_head_epoch(training_data_buffer)

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

        temp_data_file = self.checkpoint_dir / f"temp_data_epoch_{epoch + 1}.jsonl"
        if os.path.exists(temp_data_file):
            os.remove(temp_data_file)

        epoch_start_time = time.time()
        last_result_time = time.time()

        # Safety net: 4 hour max per epoch
        max_epoch_time = 4 * 3600

        # How long to wait for more results after all workers die
        drain_timeout = 60  # seconds

        # Stall timeout: if no progress for this long, assume workers are stuck
        # This should be longer than the longest expected single theorem time
        # proof_timeout (default 1200s) + some buffer for slow inference
        stall_timeout = (
            self.config.proof_timeout * 1.5
        )  # 30 minutes with default settings

        logger.info(
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

            # Collect results from workers
            got_result = False
            try:
                while True:
                    res = self.result_queue.get_nowait()
                    results_received += 1
                    got_result = True
                    last_result_time = time.time()

                    if res:
                        if "metrics" in res:
                            # Always collect metrics
                            collected_metrics.append(res["metrics"])
                            if self.config.use_wandb:
                                wandb.log(res["metrics"])

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

                    # Progress logging
                    if results_received % self.config.num_workers == 0:
                        pct = 100 * results_received / total_theorems
                        logger.info(
                            f"Progress: {results_received}/{total_theorems} "
                            f"({pct:.0f}%) in {elapsed/60:.1f} min"
                        )
            except queue.Empty:
                pass

            if not processed_batch and not got_result:
                time.sleep(0.01)

            # Check worker status
            alive_workers = [p for p in self.workers if p.is_alive()]
            dead_workers = [
                (i, p) for i, p in enumerate(self.workers) if not p.is_alive()
            ]

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
                        logger.info(
                            f"Worker {i} (PID: {p.pid}) exited gracefully. "
                            f"{len([w for w in self.workers if w.is_alive()])} workers still alive."
                        )

            # Check for stall - no progress for too long even with alive workers
            time_since_last_result = time.time() - last_result_time
            if time_since_last_result > stall_timeout:
                logger.warning(
                    f"No progress for {time_since_last_result/60:.1f} minutes "
                    f"(stall timeout: {stall_timeout/60:.0f}m). "
                    f"Collected {results_received}/{total_theorems} results. "
                    f"Proceeding with training."
                )
                break

            # If all workers are dead, wait a bit for queue to drain, then exit
            if not alive_workers:
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

    def _train_value_head_epoch(self, training_data_buffer: List[Dict[str, Any]]):
        value_data = [d for d in training_data_buffer if d.get("type") == "value"]
        assert (
            self.value_head is not None
        ), "ValueHead must be initialized before training"

        self._train_value_head_model(
            self.value_head,
            value_data,
            epochs=self.config.train_epochs,
            batch_size=32,  # Could be config param
            use_wandb=self.config.use_wandb,
        )

    def _train_value_head_model(
        self,
        value_head: ValueHead,
        data_buffer: List[Dict[str, Any]],
        epochs: int = 1,
        batch_size: int = 32,
        use_wandb: bool = True,
    ):
        if not data_buffer:
            logger.warning("Value Head training skipped: No data provided.")
            return

        value_targets = [item["value_target"] for item in data_buffer]
        avg_target = sum(value_targets) / len(value_targets)
        positive_samples = sum(1 for v in value_targets if v > 0)
        negative_samples = sum(1 for v in value_targets if v < 0)

        logger.info(f"Training Value Head on {len(data_buffer)} samples...")
        logger.info(
            f"  Data distribution: {positive_samples} positive, {negative_samples} negative"
        )
        logger.info(f"  Average target value: {avg_target:.4f}")

        avg_mcts = None
        if "mcts_value" in data_buffer[0]:
            mcts_values = [item["mcts_value"] for item in data_buffer]
            avg_mcts = sum(mcts_values) / len(mcts_values)
            logger.info(f"  Average MCTS value estimate: {avg_mcts:.4f}")

        positive_data = [item for item in data_buffer if item["value_target"] > 0]
        negative_data = [item for item in data_buffer if item["value_target"] < 0]

        if positive_data and negative_data:
            min_count = min(len(positive_data), len(negative_data))
            logger.info(f"  Balancing dataset to {min_count} samples per class.")
            random.shuffle(positive_data)
            random.shuffle(negative_data)
            balanced_data = positive_data[:min_count] + negative_data[:min_count]
            random.shuffle(balanced_data)
            training_data = balanced_data
        else:
            logger.warning(
                "  Cannot balance dataset: One class is missing. Using full dataset."
            )
            training_data = data_buffer

        if use_wandb:
            wandb.log(
                {
                    "value_head/avg_target": avg_target,
                    "value_head/positive_samples": positive_samples,
                    "value_head/negative_samples": negative_samples,
                    "value_head/avg_mcts_value": avg_mcts,
                    "value_head/training_samples": len(training_data),
                }
            )

        value_head.train()

        dataset = ValueHeadDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(value_head.value_head.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                states = batch["state"]
                batch_value_targets = batch["value_target"].to(
                    dtype=torch.float32,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                batch_value_targets = torch.clamp(
                    batch_value_targets, min=-0.99, max=0.99
                )
                features = value_head.encode_states(states)
                value_preds = torch.tanh(value_head.value_head(features).squeeze())
                loss = loss_fn(value_preds, batch_value_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(
                f"Value Head Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}"
            )
            if use_wandb:
                wandb.log({"value_head/avg_loss": avg_loss})

        value_head.eval()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
