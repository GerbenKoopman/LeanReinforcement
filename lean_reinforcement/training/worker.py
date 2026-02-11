"""
Worker module for parallel theorem proving.
"""

from typing import Dict, Any, Optional, Type
from loguru import logger
import torch.multiprocessing as mp
import gc
import queue
import os
import random
import numpy as np
import torch

from lean_dojo import DojoInitError
from ReProver.common import Pos

from lean_reinforcement.utilities.dataloader import LeanDataLoader
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.agent.runner import AgentRunner
from lean_reinforcement.agent.mcts import BaseMCTS, MCTS_GuidedRollout, MCTS_AlphaZero
from lean_reinforcement.agent.proxies import (
    QueueProxyTransformer,
    QueueProxyValueHead,
    InferenceTimeoutError,
)


def process_theorem(
    thm_data: Dict[str, Any],
    dataloader: LeanDataLoader,
    transformer: QueueProxyTransformer,
    value_head: Optional[QueueProxyValueHead],
    args: TrainingConfig,
) -> Dict[str, Any]:
    """
    Process a single theorem: initialize env, run agent, collect data.
    """
    theorem = dataloader.extract_theorem(thm_data)
    if not theorem:
        return {}

    theorem_pos = Pos(*thm_data["start"])
    if not theorem_pos:
        return {}

    try:
        env = LeanDojoEnv(theorem, theorem_pos, args.env_timeout)
    except DojoInitError as e:
        logger.error(
            f"Failed to initialize environment for theorem {theorem.full_name}: {e}"
        )
        gc.collect()  # Clean up any partially created objects
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error initializing environment for theorem {theorem.full_name}: {e}"
        )
        gc.collect()  # Clean up any partially created objects
        return {}

    mcts_class: Type[BaseMCTS]
    mcts_kwargs: Dict[str, Any]

    if args.mcts_type == "alpha_zero":
        mcts_class = MCTS_AlphaZero
        mcts_kwargs = {"value_head": value_head}
    else:
        mcts_class = MCTS_GuidedRollout
        mcts_kwargs = {}

    mcts_kwargs["batch_size"] = args.batch_size
    mcts_kwargs["num_tactics_to_expand"] = args.num_tactics_to_expand
    mcts_kwargs["max_rollout_depth"] = args.max_rollout_depth
    mcts_kwargs["max_time"] = args.max_time

    runner = AgentRunner(
        env=env,
        transformer=transformer,
        config=args,
        mcts_class=mcts_class,
        mcts_kwargs=mcts_kwargs,
        num_iterations=args.num_iterations,
        max_steps=args.max_steps,
        proof_timeout=args.proof_timeout,
    )

    try:
        metrics, theorem_training_data = runner.run(
            collect_value_data=args.train_value_head,
            use_final_reward=args.use_final_reward,
            use_wandb=args.use_wandb,
        )
        logger.debug(
            f"Collected {len(theorem_training_data)} training samples for theorem: {theorem.full_name}"
        )
        return {
            "metrics": metrics,
            "data": theorem_training_data,
            "theorem_name": theorem.full_name,
        }
    except InferenceTimeoutError as e:
        logger.error(
            f"Inference timeout during proof search for theorem {theorem.full_name}: {e}"
        )
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": 0.0,
                "proof_search/inference_timeout": True,
            },
            "data": [],
            "theorem_name": theorem.full_name,
        }
    except Exception as e:
        logger.error(f"Error during proof search for theorem {theorem.full_name}: {e}")
        # Return partial metrics if possible - at minimum we want to track that this failed
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": 0.0,
            },
            "data": [],
            "theorem_name": theorem.full_name,
        }
    finally:
        if env:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")

        del runner
        del env
        gc.collect()


def worker_loop(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    theorem_queue: mp.Queue,
    result_queue: mp.Queue,
    args: TrainingConfig,
):
    """
    Worker process loop.
    """
    # Configure logging for this worker
    logger.add(f"logs/worker_{worker_id}.log", rotation="10 MB")

    # Set deterministic seed per worker for reproducibility
    if args.seed is not None:
        worker_seed = args.seed * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        logger.info(f"Worker {worker_id} seed set to {worker_seed}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    transformer_proxy = QueueProxyTransformer(
        request_queue,
        response_queue,
        worker_id,
        timeout=args.inference_timeout,
    )
    value_head_proxy = None
    if args.mcts_type == "alpha_zero":
        value_head_proxy = QueueProxyValueHead(
            request_queue,
            response_queue,
            worker_id,
            timeout=args.inference_timeout,
        )

    dataloader = LeanDataLoader(
        corpus=None,
        dataset_path="leandojo_benchmark_4",
        data_type=args.data_type,
        load_splits=False,
    )

    theorems_processed = 0

    while True:
        try:
            thm_data = theorem_queue.get(timeout=1)
        except queue.Empty:
            continue

        if thm_data is None:
            break

        # Process theorem
        data = process_theorem(
            thm_data,
            dataloader,
            transformer_proxy,
            value_head_proxy,
            args,
        )

        # Send result back
        result_queue.put(data)

        # Force garbage collection every 4 theorems to prevent memory accumulation
        theorems_processed += 1
        if theorems_processed % 4 == 0:
            del data
            gc.collect()
