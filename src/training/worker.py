"""
TODO: Add module docstring
"""

import argparse
from typing import Union, List, Dict, Any, Optional
from loguru import logger
import torch.multiprocessing as mp
import gc
import queue

from lean_dojo import DojoInitError
from ReProver.common import Corpus, Pos

from src.utilities.dataloader import LeanDataLoader
from src.utilities.gym import LeanDojoEnv
from src.agent.runner import AgentRunner
from src.agent.mcts.guidedrollout import MCTS_GuidedRollout
from src.agent.mcts.alphazero import MCTS_AlphaZero
from src.agent.proxies import QueueProxyTransformer, QueueProxyValueHead


def process_theorem(
    thm_data: Dict[str, Any],
    corpus: Corpus,
    dataloader: LeanDataLoader,
    transformer: QueueProxyTransformer,
    value_head: Optional[QueueProxyValueHead],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Process a single theorem: initialize env, run agent, collect data.
    """
    theorem = dataloader.extract_theorem(thm_data)
    if not theorem:
        return []

    theorem_pos = Pos(*thm_data["start"])
    if not theorem_pos:
        return []

    try:
        env = LeanDojoEnv(theorem, theorem_pos)
    except DojoInitError as e:
        logger.error(
            f"Failed to initialize environment for theorem {theorem.full_name}: {e}"
        )
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error initializing environment for theorem {theorem.full_name}: {e}"
        )
        return []

    if args.mcts_type == "alpha_zero":
        mcts_class = MCTS_AlphaZero
        mcts_kwargs = {"value_head": value_head}
    else:
        mcts_class = MCTS_GuidedRollout
        mcts_kwargs = {}

    mcts_kwargs["batch_size"] = args.batch_size

    runner = AgentRunner(
        env=env,
        transformer=transformer,
        mcts_class=mcts_class,
        mcts_kwargs=mcts_kwargs,
        num_iterations=args.num_iterations,
        max_steps=args.max_steps,
    )

    try:
        _, theorem_training_data = runner.run(
            collect_value_data=args.train_value_head,
            use_final_reward=args.use_final_reward,
            use_wandb=args.use_wandb,
        )
        logger.debug(
            f"Collected {len(theorem_training_data)} training samples for theorem: {theorem.full_name}"
        )
        return theorem_training_data
    except Exception as e:
        logger.error(f"Error during proof search for theorem {theorem.full_name}: {e}")
        return []
    finally:
        del runner
        del env
        gc.collect()


def worker_loop(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    theorem_queue: mp.Queue,
    result_queue: mp.Queue,
    corpus_path: Union[str, Corpus],
    args: argparse.Namespace,
):
    """
    Worker process loop.
    """
    if isinstance(corpus_path, str):
        corpus = Corpus(corpus_path)
    else:
        corpus = corpus_path

    transformer_proxy = QueueProxyTransformer(request_queue, response_queue, worker_id)
    value_head_proxy = None
    if args.mcts_type == "alpha_zero":
        value_head_proxy = QueueProxyValueHead(request_queue, response_queue, worker_id)

    dataloader = LeanDataLoader(
        corpus, dataset_path="leandojo_benchmark_4", data_type=args.data_type
    )

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
            corpus,
            dataloader,
            transformer_proxy,
            value_head_proxy,
            args,
        )

        # Send result back
        result_queue.put(data)
