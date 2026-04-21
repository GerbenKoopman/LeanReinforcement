"""
Worker module for parallel theorem proving.
"""

from typing import Dict, Any, Optional, Type
from pathlib import Path
from loguru import logger
import torch.multiprocessing as mp
import queue
import os
import random
import sys
import time
import warnings
import numpy as np
import torch

from lean_dojo import DojoInitError
from ReProver.common import Pos

from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    configure_glibc_for_workers,
    get_available_memory_gb,
    get_process_tree_rss_gb,
    install_memory_dump_signal_handler,
    kill_child_processes,
    kill_lean_orphans,
    set_oom_score_adj,
    start_rss_watchdog,
    MAX_WORKER_RSS_GB,
    RSS_WATCHDOG_EXIT_CODE,
    WORKER_MIN_AVAILABLE_GB,
)
from lean_reinforcement.utilities.dataloader import LeanDataLoader
from lean_reinforcement.utilities.gym import (
    LeanDojoEnv,
    is_outdated_traced_repo_error,
)
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.agent.runner import AgentRunner
from lean_reinforcement.agent.mcts import BaseMCTS, MCTS_GuidedRollout, MCTS_AlphaZero
from lean_reinforcement.agent.proxies import (
    QueueProxyTransformer,
    QueueProxyValueHead,
    InferenceTimeoutError,
    MemoryLimitExceeded,
)

# Workers use this exit code when they voluntarily recycle themselves
# due to high RSS.  The trainer recognises it and restarts the worker.
_RSS_RECYCLE_EXIT_CODE = RSS_WATCHDOG_EXIT_CODE


def _summarize_error(e: Exception, max_len: int = 160) -> str:
    """Create a compact one-line error summary suitable for progress output."""
    msg = " ".join(str(e).split())
    if not msg:
        msg = e.__class__.__name__
    else:
        msg = f"{e.__class__.__name__}: {msg}"
    if len(msg) > max_len:
        return msg[: max_len - 1] + "…"
    return msg


def process_theorem(
    thm_data: Dict[str, Any],
    dataloader: LeanDataLoader,
    transformer: QueueProxyTransformer,
    value_head: Optional[QueueProxyValueHead],
    args: TrainingConfig,
    checkpoint_dir: Optional[Path] = None,
    worker_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process a single theorem: initialize env, run agent, collect data.
    """
    theorem_start = time.time()
    theorem = dataloader.extract_theorem(thm_data)
    if not theorem:
        logger.error(
            f"Worker {worker_id}: Failed to extract theorem from data; keys={list(thm_data.keys())}"
        )
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/extraction_error": True,
                "proof_search/failure_reason": "theorem extraction failed",
            },
            "data": [],
            "theorem_name": "unknown_extraction_failed",
            "worker_id": worker_id,
        }

    theorem_pos = Pos(*thm_data["start"])
    if not theorem_pos:
        logger.error(
            f"Worker {worker_id}: Failed to create position for theorem {theorem.full_name}"
        )
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/position_error": True,
                "proof_search/failure_reason": "invalid theorem position",
            },
            "data": [],
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }

    logger.debug(
        f"Worker {worker_id}: Starting theorem {theorem.full_name} "
        f"(env_timeout={args.env_timeout}s, proof_timeout={args.proof_timeout}s, "
        f"mcts={args.mcts_type}, iterations={args.num_iterations}, steps={args.max_steps})"
    )

    try:
        env = LeanDojoEnv(theorem, theorem_pos, args.env_timeout)
    except DojoInitError as e:
        logger.error(
            f"Failed to initialize environment for theorem {theorem.full_name}: {e}"
        )
        aggressive_cleanup()  # Clean up any partially created objects
        reason = _summarize_error(e)
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/env_init_error": True,
                "proof_search/failure_reason": reason,
            },
            "data": [],
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }
    except Exception as e:
        logger.exception(
            f"Unexpected error initializing environment for theorem {theorem.full_name}: {e}"
        )
        outdated_trace = is_outdated_traced_repo_error(e)
        aggressive_cleanup()  # Clean up any partially created objects
        reason = _summarize_error(e)
        if outdated_trace:
            reason = f"outdated trace cache: {reason}"
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/env_init_unexpected_error": True,
                "proof_search/outdated_trace_cache": outdated_trace,
                "proof_search/failure_reason": reason,
            },
            "data": [],
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }

    # --- Mid-theorem RSS check ---
    # Abort before MCTS search if RSS is already near the ceiling.
    tree_rss_after_env = get_process_tree_rss_gb()
    if tree_rss_after_env > MAX_WORKER_RSS_GB:
        logger.error(
            f"Process-tree RSS {tree_rss_after_env:.1f} GB exceeds hard cap "
            f"{MAX_WORKER_RSS_GB:.1f} GB after env creation for "
            f"{theorem.full_name}. Aborting theorem."
        )
        try:
            env.close()
        except Exception:
            pass
        try:
            kill_child_processes()
        except Exception:
            pass
        del env
        aggressive_cleanup()
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/rss_abort": True,
                "proof_search/failure_reason": (
                    f"rss abort: {tree_rss_after_env:.1f} GB > {MAX_WORKER_RSS_GB:.1f} GB"
                ),
            },
            "data": [],
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }

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
    mcts_kwargs["max_tree_nodes"] = getattr(args, "max_tree_nodes", 1000)
    mcts_kwargs["log_search_tree"] = getattr(args, "log_search_tree", False)

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
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        )
        logger.debug(
            f"Collected {len(theorem_training_data)} training "
            f"samples for theorem: {theorem.full_name}"
        )
        return {
            "metrics": metrics,
            "data": theorem_training_data,
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }
    except InferenceTimeoutError as e:
        logger.exception(
            f"Inference timeout during proof search for theorem {theorem.full_name}: {e}"
        )
        reason = _summarize_error(e)
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/inference_timeout": True,
                "proof_search/failure_reason": reason,
            },
            "data": [],
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }
    except MemoryLimitExceeded as e:
        logger.exception(
            f"Memory limit exceeded during proof search for theorem "
            f"{theorem.full_name}: {e}"
        )
        reason = _summarize_error(e)
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/memory_limit_exceeded": True,
                "proof_search/failure_reason": reason,
            },
            "data": [],
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }
    except Exception as e:
        logger.exception(
            f"Error during proof search for theorem {theorem.full_name}: {e}"
        )
        reason = _summarize_error(e)
        # Return partial metrics if possible - at minimum we want to track that this failed
        return {
            "metrics": {
                "proof_search/success": False,
                "proof_search/steps": 0,
                "proof_search/time": time.time() - theorem_start,
                "proof_search/unexpected_error": True,
                "proof_search/failure_reason": reason,
            },
            "data": [],
            "theorem_name": theorem.full_name,
            "worker_id": worker_id,
        }
    finally:
        if env:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")

        # Kill any orphaned lean/lake processes after env.close().
        try:
            kill_child_processes()
            kill_lean_orphans()
        except Exception:
            pass

        del runner
        del env
        aggressive_cleanup()


def worker_loop(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    theorem_queue: mp.Queue,
    result_queue: mp.Queue,
    args: TrainingConfig,
    checkpoint_dir: "Path",
):
    """
    Worker process loop.
    """
    # --- glibc malloc tuning (must be first) ---
    configure_glibc_for_workers()

    # --- SIGUSR1 memory diagnostic handler ---
    install_memory_dump_signal_handler()

    # Configure logging for this worker:
    # Remove default stderr handler so worker logs don't interfere
    # with the main process's live progress display, then add a
    # file-only handler.  Must happen BEFORE start_rss_watchdog()
    # so the watchdog's startup logger.info goes to the file, not
    # to stderr (which clobbers the Rich progress display).
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(
        f"logs/worker_{worker_id}.log",
        rotation="10 MB",
        level="DEBUG" if args.debugging else "INFO",
        backtrace=args.debugging,
        diagnose=args.debugging,
    )
    if args.debugging:
        # Mirror worker logs to stderr for live cluster debugging.
        logger.add(
            sys.stderr,
            level="DEBUG",
            backtrace=True,
            diagnose=False,
        )

    # --- RSS watchdog (daemon thread, checks every 1s) ---
    start_rss_watchdog(hard_cap_gb=MAX_WORKER_RSS_GB, check_interval=1.0)

    # Suppress NVML/accelerator warnings in worker processes
    warnings.filterwarnings("ignore", message="Can't initialize NVML")

    # Set deterministic seed per worker for reproducibility
    if args.seed is not None:
        worker_seed = args.seed * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        logger.info(f"Worker {worker_id} seed set to {worker_seed}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # --- Hard memory limit on the Lean 4 REPL subprocess ---
    # Patch lean_dojo's TACTIC_MEMORY_LIMIT at runtime since lean_dojo
    # reads it at import time.
    lean_mem_gb = getattr(args, "lean_memory_limit_gb", 2)
    os.environ["TACTIC_MEMORY_LIMIT"] = f"{lean_mem_gb}g"
    # Patch the already-imported constant so Dojo.__enter__ picks it up
    try:
        import lean_dojo.constants as _ldc

        _ldc.TACTIC_MEMORY_LIMIT = f"{lean_mem_gb}g"
        # Also patch the dojo module's local reference if it cached it
        import lean_dojo.interaction.dojo as _ldd

        _ldd.TACTIC_MEMORY_LIMIT = f"{lean_mem_gb}g"
    except Exception as e:
        logger.warning(f"Worker {worker_id}: Could not patch TACTIC_MEMORY_LIMIT: {e}")
    logger.info(
        f"Worker {worker_id}: Lean REPL memory limit set to {lean_mem_gb} GB "
        f"(--memory={lean_mem_gb * 1024} MB)"
    )

    # NOTE: RLIMIT_AS is NOT used — it caps virtual address space (not
    # RSS) and would break Python/torch/ray mmap usage.  The Lean
    # --memory flag above is the correct enforcement layer.

    # Set high OOM score so the kernel kills workers before the desktop
    set_oom_score_adj(1000)

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

    # Memory backpressure: pause when system memory is low.
    MEMORY_BACKOFF_MAX = 30.0
    MEMORY_BACKOFF_BASE = 2.0

    while True:
        try:
            thm_data = theorem_queue.get(timeout=1)
        except queue.Empty:
            continue

        if thm_data is None:
            break

        # --- Per-worker RSS check before new theorem ---
        rss_gb = get_process_tree_rss_gb()
        rss_soft_cap = MAX_WORKER_RSS_GB * 0.75
        if rss_gb > rss_soft_cap:
            logger.warning(
                f"Worker {worker_id}: process-tree RSS {rss_gb:.1f} GB exceeds "
                f"soft cap {rss_soft_cap:.1f} GB before new theorem. "
                f"Attempting cleanup."
            )
            aggressive_cleanup()
            rss_gb = get_process_tree_rss_gb()
            if rss_gb > rss_soft_cap:
                logger.error(
                    f"Worker {worker_id}: process-tree RSS still {rss_gb:.1f} GB "
                    f"after cleanup. Recycling worker."
                )
                sys.exit(_RSS_RECYCLE_EXIT_CODE)

        # --- System memory backpressure ---
        backoff = MEMORY_BACKOFF_BASE
        avail_gb = get_available_memory_gb()
        while avail_gb < WORKER_MIN_AVAILABLE_GB:
            logger.warning(
                f"Worker {worker_id}: Low system memory "
                f"({avail_gb:.1f} GB available, "
                f"need {WORKER_MIN_AVAILABLE_GB:.0f} GB). "
                f"Waiting {backoff:.0f}s before starting next theorem."
            )
            aggressive_cleanup()
            time.sleep(backoff)
            backoff = min(backoff * 1.5, MEMORY_BACKOFF_MAX)
            avail_gb = get_available_memory_gb()

        # Process theorem
        data = process_theorem(
            thm_data,
            dataloader,
            transformer_proxy,
            value_head_proxy,
            args,
            checkpoint_dir,
            worker_id=worker_id,
        )

        # Send result back
        result_queue.put(data)

        rss_soft_cap_gb = MAX_WORKER_RSS_GB * 0.9
        rss_after_theorem_gb = get_process_tree_rss_gb()
        if rss_after_theorem_gb > rss_soft_cap_gb:
            logger.warning(
                f"Worker {worker_id}: process-tree RSS {rss_after_theorem_gb:.1f} GB exceeds "
                f"soft cap {rss_soft_cap_gb:.1f} GB after theorem. "
                "Attempting cleanup and worker recycle if still high."
            )
            aggressive_cleanup()
            rss_post_cleanup_gb = get_process_tree_rss_gb()
            if rss_post_cleanup_gb > rss_soft_cap_gb:
                logger.error(
                    f"Worker {worker_id}: process-tree RSS still high at {rss_post_cleanup_gb:.1f} GB "
                    "after cleanup. Exiting worker to prevent runaway memory growth."
                )
                sys.exit(_RSS_RECYCLE_EXIT_CODE)

        # gc + malloc_trim after every theorem to prevent RSS creep.
        theorems_processed += 1
        del data
        aggressive_cleanup()

    # Cleanup when worker exits
    logger.info(
        f"Worker {worker_id} shutting down after processing {theorems_processed} theorems"
    )

    # Clean up dataloader and other resources
    del dataloader
    del transformer_proxy
    if value_head_proxy is not None:
        del value_head_proxy

    # Final aggressive cleanup before exit
    aggressive_cleanup()

    logger.info(f"Worker {worker_id} cleanup complete")
