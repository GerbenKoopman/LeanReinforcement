"""
Main script for agent training and guided-rollout mcts dataset creation. Will
implement tactic generation training in the future.
"""

from typing import List, Dict, Any
from loguru import logger
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import gc
import os
import subprocess
import shutil
import time
from dotenv import load_dotenv
import wandb
import random
import queue

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
from lean_reinforcement.utilities.config import get_config, TrainingConfig
from dataclasses import asdict


# Load environment variables from .env file
load_dotenv()


# --- Training Functions ---


def train_value_head(
    value_head: ValueHead,
    data_buffer: List[Dict[str, Any]],
    epochs: int = 1,
    batch_size: int = 32,
    use_wandb: bool = True,
):
    """
    Trains the value head on collected data.
    """
    if not data_buffer:
        logger.warning("Value Head training skipped: No data provided.")
        return

    # Log training data statistics
    value_targets = [item["value_target"] for item in data_buffer]
    avg_target = sum(value_targets) / len(value_targets)
    positive_samples = sum(1 for v in value_targets if v > 0)
    negative_samples = sum(1 for v in value_targets if v < 0)

    logger.info(f"Training Value Head on {len(data_buffer)} samples...")
    logger.info(
        f"  Data distribution: {positive_samples} positive, {negative_samples} negative"
    )
    logger.info(f"  Average target value: {avg_target:.4f}")

    # Log MCTS statistics if available
    avg_mcts = None
    if "mcts_value" in data_buffer[0]:
        mcts_values = [item["mcts_value"] for item in data_buffer]
        avg_mcts = sum(mcts_values) / len(mcts_values)
        logger.info(f"  Average MCTS value estimate: {avg_mcts:.4f}")

    if use_wandb:
        wandb.log(
            {
                "value_head/avg_target": avg_target,
                "value_head/positive_samples": positive_samples,
                "value_head/negative_samples": negative_samples,
                "value_head/avg_mcts_value": avg_mcts,
            }
        )

    value_head.train()  # Set model to training mode

    dataset = ValueHeadDataset(data_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(value_head.value_head.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            states = batch["state"]
            value_targets = batch["value_target"].to(
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Get features from the frozen encoder
            features = value_head.encode_states(states)

            # Get value prediction from the trainable head
            value_preds = value_head.value_head(features).squeeze()

            loss = loss_fn(value_preds, value_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Value Head Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"value_head/avg_loss": avg_loss})

    value_head.eval()  # Set back to evaluation mode

    # Clear GPU memory after training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Main Loop ---


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


def main(args: TrainingConfig):
    # --- Setup wandb ---
    if args.use_wandb:
        wandb.init(
            entity="gerbennkoopman-university-of-amsterdam",
            project="lean-reinforcement",
            config=asdict(args),
        )

    # --- Setup checkpoint directory ---
    checkpoint_dir = get_checkpoint_dir()
    logger.info(f"Using checkpoint directory: {checkpoint_dir}")

    # --- Models ---
    transformer = Transformer()

    value_head = None
    start_epoch = 0
    if args.mcts_type == "alpha_zero" or args.train_value_head:
        value_head = ValueHead(transformer)

        # Load checkpoint if resume flag is set
        if args.resume:
            start_epoch = load_checkpoint(value_head, checkpoint_dir)
            logger.info(f"Resuming training from epoch {start_epoch}")

    log_gpu_memory("After model initialization - ")

    # --- DataLoader ---
    logger.info(f"Loading data from 'leandojo_benchmark_4/{args.data_type}'")
    corpus_path = os.path.join("leandojo_benchmark_4/corpus.jsonl")
    corpus = Corpus(corpus_path)
    dataloader = LeanDataLoader(
        corpus, dataset_path="leandojo_benchmark_4", data_type=args.data_type
    )
    traced_repo = dataloader.trace_repo()

    # Ensure the repo is built (fix for missing .lake/lakefile.olean)
    repo_path = traced_repo.root_dir
    lakefile_path = repo_path / ".lake" / "lakefile.olean"

    # Robust check and copy for lakefile.olean
    if not lakefile_path.exists():
        logger.warning(f"{lakefile_path} missing. Checking build artifacts...")

        def find_and_copy_lakefile():
            candidates = list(repo_path.glob(".lake/**/lakefile.olean"))
            if candidates:
                src = candidates[0]
                logger.info(
                    f"Found lakefile.olean at {src}, copying to {lakefile_path}"
                )
                lakefile_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, lakefile_path)
                return True
            return False

        if not find_and_copy_lakefile():
            logger.warning(
                f"lakefile.olean not found in build artifacts. Running lake build in {repo_path}..."
            )
            try:
                subprocess.run(["lake", "build"], cwd=repo_path, check=True)
                logger.info("lake build completed successfully.")
                if not lakefile_path.exists():
                    if not find_and_copy_lakefile():
                        logger.warning(
                            "Could not find lakefile.olean even after build."
                        )
            except subprocess.CalledProcessError as e:
                logger.error(f"lake build failed: {e}")
                raise e

    # --- Multiprocessing Setup ---
    mp.set_start_method("spawn", force=True)  # Force spawn for CUDA safety

    request_queue = mp.Queue()
    result_queue = mp.Queue()
    theorem_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(args.num_workers)]

    workers = []
    try:
        for i in range(args.num_workers):
            p = mp.Process(
                target=worker_loop,
                args=(
                    i,
                    request_queue,
                    response_queues[i],
                    theorem_queue,
                    result_queue,
                    corpus,
                    args,
                ),
            )
            p.start()
            workers.append(p)

        # --- Self-Play and Training Loop ---
        inference_server = InferenceServer(
            transformer, value_head, request_queue, response_queues, args.batch_size
        )

        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{args.num_epochs}")
            training_data_buffer = []

            # Clear GPU cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Shuffle training theorems each epoch
            random.shuffle(dataloader.train_data)

            theorems_to_process = dataloader.train_data[: args.num_theorems]

            # Fill theorem queue
            for thm in theorems_to_process:
                theorem_queue.put(thm)

            logger.info(
                f"Processing {len(theorems_to_process)} theorems with {args.num_workers} workers."
            )

            # Inference Loop
            completed_theorems = 0

            while completed_theorems < len(theorems_to_process):
                # 1. Process Inference Requests
                processed_batch = inference_server.process_requests()

                # 2. Check for Results
                try:
                    while True:
                        res = result_queue.get_nowait()
                        if res:
                            training_data_buffer.extend(res)
                        completed_theorems += 1
                        if completed_theorems % 10 == 0:
                            logger.info(
                                f"Completed {completed_theorems}/{len(theorems_to_process)} proofs"
                            )
                except queue.Empty:
                    pass

                # Small sleep to prevent busy loop burning CPU
                if not processed_batch:
                    time.sleep(0.01)

                # Check if any worker has died unexpectedly
                dead_workers = [p for p in workers if not p.is_alive()]
                if dead_workers:
                    logger.error(
                        f"Found {len(dead_workers)} dead worker(s). Stopping training to prevent hang."
                    )
                    for p in dead_workers:
                        logger.error(f"Worker exit code: {p.exitcode}")

                    # Kill remaining workers and exit
                    for p in workers:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(
                        "One or more worker processes died unexpectedly."
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- MODEL TRAINING STEP ---
            if not training_data_buffer:
                logger.warning("No data collected in this epoch. Skipping training.")
                continue

            # --- Analyze collected data ---
            if args.train_value_head:
                stats = analyze_value_data(training_data_buffer)
                print_training_stats(stats)

                # Optionally save training data for offline analysis
                if args.save_training_data:
                    data_save_path = (
                        checkpoint_dir / f"training_data_epoch_{epoch + 1}.json"
                    )
                    save_training_data(training_data_buffer, data_save_path)

            # --- CONDITIONAL TRAINING ---
            if args.train_value_head:
                value_data = [
                    d for d in training_data_buffer if d.get("type") == "value"
                ]
                assert (
                    value_head is not None
                ), "ValueHead must be initialized before training"
                train_value_head(
                    value_head,
                    value_data,
                    epochs=args.train_epochs,
                    use_wandb=args.use_wandb,
                )

            # --- Save checkpoints after each epoch ---
            if (
                args.train_value_head
                and value_head is not None
                and args.save_checkpoints
            ):
                save_checkpoint(value_head, epoch + 1, checkpoint_dir, args)
                logger.info(f"Checkpoint saved for epoch {epoch + 1}")

        # Cleanup
        for _ in range(args.num_workers):
            theorem_queue.put(None)

        for p in workers:
            p.join()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        raise e
    finally:
        logger.info("Shutting down workers...")
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join()


if __name__ == "__main__":
    args = get_config()
    main(args)
