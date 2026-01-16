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

    def train(self) -> None:
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
                self._run_epoch(epoch, inference_server)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as e:
            logger.error(f"Training crashed: {e}")
            raise e
        finally:
            self._cleanup_workers()

    def _run_epoch(self, epoch: int, inference_server: InferenceServer):
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

        training_data_buffer = self._collect_data(
            theorems_to_process, inference_server, epoch
        )

        self._stop_workers()
        self._drain_queues()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not training_data_buffer:
            logger.warning("No data collected in this epoch. Skipping training.")
            return

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

    def _drain_queues(self) -> None:
        logger.info("Draining queues...")
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
    ) -> List[Dict[str, Any]]:
        completed_theorems = 0
        training_data_buffer: List[Dict[str, Any]] = []

        temp_data_file = self.checkpoint_dir / f"temp_data_epoch_{epoch + 1}.jsonl"
        if os.path.exists(temp_data_file):
            os.remove(temp_data_file)

        while completed_theorems < len(theorems_to_process):
            processed_batch = inference_server.process_requests()

            try:
                while True:
                    res = self.result_queue.get_nowait()
                    if res:
                        if "metrics" in res and self.config.use_wandb:
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

                        if len(training_data_buffer) > 100:
                            with open(temp_data_file, "a") as f:
                                for item in training_data_buffer:
                                    f.write(json.dumps(item) + "\n")
                            training_data_buffer = []

                    completed_theorems += 1
                    if completed_theorems % self.config.num_workers == 0:
                        logger.info(
                            f"Completed {completed_theorems}/{len(theorems_to_process)} proofs"
                        )
            except queue.Empty:
                pass

            if not processed_batch:
                time.sleep(0.01)

            dead_workers = [p for p in self.workers if not p.is_alive()]
            if dead_workers:
                logger.error(
                    f"Found {len(dead_workers)} dead worker(s). Stopping training."
                )
                for p in dead_workers:
                    logger.error(f"Worker exit code: {p.exitcode}")
                for p in self.workers:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError("One or more worker processes died unexpectedly.")

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

        return training_data_buffer

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
