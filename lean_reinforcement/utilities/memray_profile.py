"""
Optional ``memray`` native-memory profiling for worker processes.

Usage
-----
Set ``LEAN_RL_MEMRAY=1`` in the environment to enable profiling.
Each worker writes a ``logs/memray_worker_<id>_<pid>.bin`` file that
can be analysed *offline* with::

    memray flamegraph logs/memray_worker_0_12345.bin
    memray table    logs/memray_worker_0_12345.bin
    memray tree     logs/memray_worker_0_12345.bin

The ``--native`` flag captures C/C++ stack frames (glibc malloc,
pexpect, JSON parsing) in addition to Python frames.  This is the
key advantage over Python-only ``tracemalloc``: most of the anonymous
VMA growth we're investigating happens *below* the Python layer.

Implementation
--------------
``memray.Tracker`` is a context manager.  We start it at the very
beginning of ``worker_loop()`` and it runs for the lifetime of the
worker process.  When the worker exits (normally or via ``os._exit``
from the RSS watchdog), memray flushes the file.

Since memray has non-trivial overhead (~5-15% CPU, ~2x allocation
tracking memory), it is **off by default**.  Enable it only for
diagnostic runs.
"""

import os
import sys
from typing import Any
import memray

#: Whether memray profiling is enabled (set LEAN_RL_MEMRAY=1).
MEMRAY_ENABLED: bool = os.environ.get("LEAN_RL_MEMRAY", "0") == "1"


def start_memray_tracker(worker_id: int) -> Any:
    """Start a ``memray.Tracker`` for this worker process.

    Returns the tracker context manager (already ``__enter__``'d) so
    the caller can ``__exit__`` it on clean shutdown.  Returns ``None``
    if memray is disabled or not installed.
    """
    if not MEMRAY_ENABLED:
        return None

    pid = os.getpid()
    os.makedirs("logs", exist_ok=True)
    path = f"logs/memray_worker_{worker_id}_{pid}.bin"

    try:
        tracker = memray.Tracker(
            path,
            native_traces=True,
            follow_fork=True,
        )
        tracker.__enter__()
        sys.stderr.write(
            f"[MEMRAY] Worker {worker_id} (PID {pid}): " f"profiling to {path}\n"
        )
        sys.stderr.flush()
        return tracker
    except Exception as e:
        sys.stderr.write(f"[MEMRAY] Worker {worker_id}: Failed to start tracker: {e}\n")
        sys.stderr.flush()
        return None


def stop_memray_tracker(tracker: Any) -> None:
    """Stop a previously started memray tracker."""
    if tracker is None:
        return
    try:
        tracker.__exit__(None, None, None)
    except Exception:
        pass
