"""
Worker module for parallel theorem proving.
"""

from typing import Union, Dict, Any, Optional
from loguru import logger
import torch.multiprocessing as mp
import gc
import queue
import os

from lean_dojo import DojoInitError
from ReProver.common import Corpus, Pos

from lean_reinforcement.utilities.dataloader import LeanDataLoader
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.agent.runner import AgentRunner
from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_reinforcement.agent.mcts.alphazero import MCTS_AlphaZero
from lean_reinforcement.agent.proxies import QueueProxyTransformer, QueueProxyValueHead


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
        env = LeanDojoEnv(theorem, theorem_pos)
    except DojoInitError as e:
        logger.error(
            f"Failed to initialize environment for theorem {theorem.full_name}: {e}"
        )
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error initializing environment for theorem {theorem.full_name}: {e}"
        )
        return {}

    if args.mcts_type == "alpha_zero":
        mcts_class = MCTS_AlphaZero
        mcts_kwargs: Dict[str, Union[QueueProxyValueHead, int, None]] = {
            "value_head": value_head
        }
    else:
        mcts_class = MCTS_GuidedRollout
        mcts_kwargs: Dict[str, Union[QueueProxyValueHead, int, None]] = {}

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
        metrics, theorem_training_data = runner.run(
            collect_value_data=args.train_value_head,
            use_final_reward=args.use_final_reward,
            use_wandb=args.use_wandb,
        )
        logger.debug(
            f"Collected {len(theorem_training_data)} training samples for theorem: {theorem.full_name}"
        )
        return {"metrics": metrics, "data": theorem_training_data}
    except Exception as e:
        logger.error(f"Error during proof search for theorem {theorem.full_name}: {e}")
        return {}
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
    args: TrainingConfig,
):
    """
    Worker process loop.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
            dataloader,
            transformer_proxy,
            value_head_proxy,
            args,
        )

        # Send result back
        result_queue.put(data)
