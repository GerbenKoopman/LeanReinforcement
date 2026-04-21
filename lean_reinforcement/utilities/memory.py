"""Memory management utilities for lean_reinforcement.

Provides a single API for memory cleanup, monitoring, and threshold logic:

- ``aggressive_cleanup()`` — gc.collect + malloc_trim to reclaim RSS.
- ``empty_gpu_cache()`` — release unused GPU VRAM.
- ``periodic_cache_cleanup()`` — lightweight GPU cache cleanup in hot loops.
- ``kill_child_processes()`` / ``kill_lean_orphans()`` — process cleanup.
- ``start_rss_watchdog()`` — daemon thread that hard-kills on RSS breach.
- ``configure_glibc_for_workers()`` — tune glibc malloc to reduce fragmentation.
- ``dump_memory_diagnostic()`` — SIGUSR1-triggered diagnostic snapshot.
"""

import ctypes
import ctypes.util
import gc
import logging
import os
import signal
import sys
import threading
from functools import lru_cache
from typing import Dict, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger as LoguruLogger


# ---------------------------------------------------------------------------
# Memory thresholds (single source of truth)
# ---------------------------------------------------------------------------

#: Workers pause when available RAM is below this (GiB).
#: Override via ``LEAN_RL_WORKER_MIN_AVAILABLE_GB`` env var.
WORKER_MIN_AVAILABLE_GB: float = float(
    os.environ.get("LEAN_RL_WORKER_MIN_AVAILABLE_GB", "2.0")
)

#: Trainer pauses when available RAM is below this (GiB).
#: Override via ``LEAN_RL_TRAINER_MIN_AVAILABLE_GB`` env var.
TRAINER_MIN_AVAILABLE_GB: float = float(
    os.environ.get("LEAN_RL_TRAINER_MIN_AVAILABLE_GB", "3.0")
)

#: MCTS aborts when available RAM drops below this (GiB).
#: Override via ``LEAN_RL_MIN_AVAILABLE_GB`` env var.
MCTS_MIN_AVAILABLE_GB: float = float(os.environ.get("LEAN_RL_MIN_AVAILABLE_GB", "1.5"))

#: Per-worker RSS hard cap (GiB). Triggers worker recycling.
#: Override via ``LEAN_RL_MAX_RSS_GB`` env var.
MAX_WORKER_RSS_GB: float = float(os.environ.get("LEAN_RL_MAX_RSS_GB", "7.0"))

#: InferenceServer triggers GPU cleanup when VRAM allocation exceeds
#: this percentage of total GPU memory.
GPU_CLEANUP_THRESHOLD_PERCENT: int = 80


# ---------------------------------------------------------------------------
# malloc_trim: force glibc to return freed arenas to the OS
# ---------------------------------------------------------------------------

_libc: Optional[ctypes.CDLL] = None


def _get_libc() -> Optional[ctypes.CDLL]:
    """Lazily load libc.  Returns None on non-glibc systems (macOS, musl)."""
    global _libc
    if _libc is not None:
        return _libc
    if sys.platform != "linux":
        return None
    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            _libc = ctypes.CDLL(libc_name)
        else:
            _libc = ctypes.CDLL("libc.so.6")
        return _libc
    except OSError:
        return None


def malloc_trim() -> bool:
    """Call glibc ``malloc_trim(0)`` to return freed heap pages to the OS."""
    libc = _get_libc()
    if libc is None:
        return False
    try:
        libc.malloc_trim(0)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU memory cleanup
# ---------------------------------------------------------------------------


def aggressive_cleanup() -> None:
    """Full gc + malloc_trim for maximum RSS reduction."""
    # A single gen-2 collection already collects all generations.
    gc.collect()
    malloc_trim()


# ---------------------------------------------------------------------------
# GPU memory cleanup
# ---------------------------------------------------------------------------


def empty_gpu_cache() -> None:
    """Release unused cached GPU memory. No-op if CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def periodic_cache_cleanup(call_count: int, interval: int = 100) -> int:
    """
    Lightweight GPU cache cleanup for hot loops.

    Increments call_count and triggers empty_gpu_cache() every `interval` calls.

    Args:
        call_count: Current call counter
        interval: Number of calls between cleanup triggers (default: 100)

    Returns:
        Updated call counter

    Example:
        self._generate_call_count = periodic_cache_cleanup(self._generate_call_count)
    """
    call_count += 1
    if call_count % interval == 0:
        empty_gpu_cache()
    return call_count


def get_gpu_memory_usage_percent() -> float:
    """Return current GPU memory allocation as a percentage of total VRAM.

    Returns 0.0 if CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        if total == 0:
            return 0.0
        return (allocated / total) * 100.0
    except Exception:
        return 0.0


def log_gpu_memory(
    log: "logging.Logger | LoguruLogger | None" = None,
    prefix: str = "",
    level: int = logging.DEBUG,
) -> None:
    """Log current GPU memory usage in a single line."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        msg = f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        if log is not None:
            log.log(level, msg)
        else:
            from loguru import logger as _log

            _log.debug(msg)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# RSS / system memory monitoring
# ---------------------------------------------------------------------------


# Cache page size and PID path for hot-path /proc reads.
_PAGE_SIZE_BYTES: int = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_BYTES_PER_GIB: float = 1024.0**3


def _join_cgroup_file(base: str, relative_group: str, filename: str) -> str:
    rel = (relative_group or "/").strip()
    if rel in {"", "/"}:
        return os.path.join(base, filename)
    return os.path.join(base, rel.lstrip("/"), filename)


@lru_cache(maxsize=1)
def _resolve_cgroup_memory_files() -> (
    tuple[Optional[str], Optional[str], Optional[str]]
):
    """Resolve memory.current/memory.max/memory.stat for this process cgroup.

    Uses /proc/self/cgroup so we inspect the active Slurm step cgroup rather than
    assuming root-level cgroup paths.
    """
    rel_v2: Optional[str] = None
    rel_v1_memory: Optional[str] = None

    try:
        with open("/proc/self/cgroup") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                _, controllers, relative_group = parts
                relative_group = relative_group or "/"
                if controllers == "":
                    rel_v2 = relative_group
                else:
                    if "memory" in controllers.split(","):
                        rel_v1_memory = relative_group
    except Exception:
        pass

    if rel_v2 is not None:
        current = _join_cgroup_file("/sys/fs/cgroup", rel_v2, "memory.current")
        limit = _join_cgroup_file("/sys/fs/cgroup", rel_v2, "memory.max")
        stat = _join_cgroup_file("/sys/fs/cgroup", rel_v2, "memory.stat")
        if any(os.path.exists(path) for path in (current, limit, stat)):
            return current, limit, stat

    if rel_v1_memory is not None:
        current = _join_cgroup_file(
            "/sys/fs/cgroup/memory", rel_v1_memory, "memory.usage_in_bytes"
        )
        limit = _join_cgroup_file(
            "/sys/fs/cgroup/memory", rel_v1_memory, "memory.limit_in_bytes"
        )
        stat = _join_cgroup_file("/sys/fs/cgroup/memory", rel_v1_memory, "memory.stat")
        if any(os.path.exists(path) for path in (current, limit, stat)):
            return current, limit, stat

    # Fallback for environments where /proc/self/cgroup does not map cleanly.
    fallback_candidates = (
        (
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.stat",
        ),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.stat",
        ),
    )
    for current, limit, stat in fallback_candidates:
        if any(os.path.exists(path) for path in (current, limit, stat)):
            return current, limit, stat

    return None, None, None


def _read_cgroup_bytes(path: str) -> Optional[int]:
    """Read a cgroup byte counter. Returns None for unavailable/unlimited."""
    try:
        with open(path) as f:
            raw = f.read().strip()
        if not raw or raw == "max":
            return None
        value = int(raw)
        # Some cgroup v1 setups use huge sentinel values for "no limit".
        if value <= 0 or value >= (1 << 60):
            return None
        return value
    except Exception:
        return None


def _read_cgroup_stat(path: str) -> Dict[str, int]:
    """Read cgroup memory.stat into a key->bytes mapping."""
    stats: Dict[str, int] = {}
    try:
        with open(path) as f:
            for raw_line in f:
                parts = raw_line.split()
                if len(parts) != 2:
                    continue
                key, value_raw = parts
                try:
                    stats[key] = int(value_raw)
                except ValueError:
                    continue
    except Exception:
        return {}
    return stats


def _read_effective_cgroup_v2_limit_bytes(limit_path: str) -> Optional[int]:
    """Return the effective v2 memory.max limit across ancestor cgroups.

    Some environments set memory.max="max" at the process cgroup while an
    ancestor applies the actual limit. We walk toward /sys/fs/cgroup and take
    the minimum finite limit encountered.
    """
    root = os.path.normpath("/sys/fs/cgroup")
    current_dir = os.path.normpath(os.path.dirname(limit_path))
    finite_limits: list[int] = []

    while True:
        path = os.path.join(current_dir, "memory.max")
        value = _read_cgroup_bytes(path)
        if value is not None:
            finite_limits.append(value)

        if current_dir == root:
            break

        parent = os.path.normpath(os.path.dirname(current_dir))
        if parent == current_dir or not parent.startswith(root):
            break
        current_dir = parent

    if not finite_limits:
        return None
    return min(finite_limits)


def get_cgroup_memory_current_gb() -> Optional[float]:
    """Return current cgroup memory usage in GiB, if available."""
    current_path, _, _ = _resolve_cgroup_memory_files()
    if current_path is None:
        return None
    value = _read_cgroup_bytes(current_path)
    if value is not None:
        return value / _BYTES_PER_GIB
    return None


def get_cgroup_memory_limit_gb() -> Optional[float]:
    """Return cgroup memory limit in GiB, if available and finite."""
    _, limit_path, _ = _resolve_cgroup_memory_files()
    if limit_path is None:
        return None

    if os.path.basename(limit_path) == "memory.max":
        value = _read_effective_cgroup_v2_limit_bytes(limit_path)
    else:
        value = _read_cgroup_bytes(limit_path)

    if value is not None:
        return value / _BYTES_PER_GIB
    return None


def get_cgroup_memory_stat_gb() -> Dict[str, float]:
    """Return selected cgroup memory.stat components in GiB.

    Normalized keys:
    - anon: anonymous memory (v2: anon, v1 fallback: rss)
    - file: file-backed cache (v2: file, v1 fallback: cache)
    - shmem: shared memory (if provided by kernel/cgroup version)
    """
    _, _, stat_path = _resolve_cgroup_memory_files()
    if stat_path is None:
        return {}

    stats = _read_cgroup_stat(stat_path)
    if not stats:
        return {}

    anon_bytes = stats.get("anon", stats.get("rss"))
    file_bytes = stats.get("file", stats.get("cache"))
    shmem_bytes = stats.get("shmem")

    out: Dict[str, float] = {}
    if anon_bytes is not None:
        out["anon"] = anon_bytes / _BYTES_PER_GIB
    if file_bytes is not None:
        out["file"] = file_bytes / _BYTES_PER_GIB
    if shmem_bytes is not None:
        out["shmem"] = shmem_bytes / _BYTES_PER_GIB
    return out


def get_cgroup_oom_kill_count() -> Optional[int]:
    """Return cumulative cgroup oom_kill count from memory.events, if available."""
    current_path, _, _ = _resolve_cgroup_memory_files()
    if current_path is None:
        return None

    events_path = os.path.join(os.path.dirname(current_path), "memory.events")
    events = _read_cgroup_stat(events_path)
    if not events:
        return None
    return events.get("oom_kill")


def get_cgroup_available_memory_gb() -> Optional[float]:
    """Return available cgroup memory in GiB, if cgroup limits are present."""
    current = get_cgroup_memory_current_gb()
    limit = get_cgroup_memory_limit_gb()
    if current is None or limit is None:
        return None
    return max(0.0, limit - current)


def get_rss_gb() -> float:
    """Return the current process RSS in GiB (Linux only, fast)."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            pages = int(f.read().split()[1])
        return pages * _PAGE_SIZE_BYTES / _BYTES_PER_GIB
    except Exception:
        return 0.0


def get_process_tree_rss_gb() -> float:
    """Return RSS of this process plus descendants in GiB.

    Falls back to ``get_rss_gb()`` if ``psutil`` is unavailable.
    """
    own_rss_gb = get_rss_gb()
    try:
        import psutil

        current = psutil.Process(os.getpid())
        total_bytes = current.memory_info().rss
        for child in current.children(recursive=True):
            try:
                total_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_bytes / _BYTES_PER_GIB
    except Exception:
        return own_rss_gb


def get_available_memory_gb() -> float:
    """Return effective available memory in GiB.

    Uses host MemAvailable and, when present, cgroup-available memory.
    In Slurm/cgroup-limited jobs this prevents overestimating free RAM.
    """
    system_avail_gb: Optional[float] = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    system_avail_gb = int(line.split()[1]) / (1024**2)  # kB -> GB
                    break
    except Exception:
        pass

    cgroup_avail_gb = get_cgroup_available_memory_gb()

    if system_avail_gb is None:
        return cgroup_avail_gb if cgroup_avail_gb is not None else 0.0
    if cgroup_avail_gb is None:
        return system_avail_gb
    return min(system_avail_gb, cgroup_avail_gb)


# ---------------------------------------------------------------------------
# OOM score adjustment (Linux)
# ---------------------------------------------------------------------------


def kill_child_processes() -> None:
    """SIGKILL all descendant processes of the current process."""
    try:
        import psutil
        import signal

        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
        for child in children:
            try:
                child.send_signal(signal.SIGKILL)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        # Wait briefly for processes to die
        psutil.wait_procs(children, timeout=2)
    except Exception:
        pass


def kill_lean_orphans() -> None:
    """SIGKILL orphaned ``lean``/``lake`` processes owned by the current user."""
    try:
        import psutil
        import signal

        my_uid = os.getuid()
        targets = []
        for proc in psutil.process_iter(["pid", "name", "uids", "ppid"]):
            try:
                if proc.info["uids"] and proc.info["uids"].real == my_uid:
                    name = (proc.info["name"] or "").lower()
                    ppid = int(proc.info.get("ppid", -1))
                    if name in {"lean", "lake"} and ppid == 1:
                        targets.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        for proc in targets:
            try:
                proc.send_signal(signal.SIGKILL)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if targets:
            psutil.wait_procs(targets, timeout=2)
    except Exception:
        pass


def configure_glibc_for_workers() -> None:
    """Tune glibc malloc to reduce heap fragmentation in workers.

    - ARENA_MAX=2: fewer arenas make malloc_trim more effective.
    - MMAP_THRESHOLD=4096: allocations >= 4 KB use mmap (immediately reclaimable).
    - TRIM_THRESHOLD=4096: aggressive return of freed pages.

    Uses ``mallopt()`` via ctypes (env vars are ignored after glibc init).
    Call at the start of each worker process.
    """
    # Also set env vars as a belt-and-suspenders measure: subprocesses
    # (e.g. Lean REPL) inherit the environment and will pick these up
    # at *their* glibc init.
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "4096")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "4096")

    # --- mallopt() via ctypes (the actual fix) ---
    # glibc mallopt constants (from <malloc.h>):
    #   M_TRIM_THRESHOLD  = -1
    #   M_MMAP_THRESHOLD  = -3
    #   M_ARENA_MAX       = -8
    _M_TRIM_THRESHOLD = -1
    _M_MMAP_THRESHOLD = -3
    _M_ARENA_MAX = -8

    libc = _get_libc()
    if libc is None:
        return

    try:
        libc.mallopt(_M_ARENA_MAX, 2)
        libc.mallopt(_M_MMAP_THRESHOLD, 4096)
        libc.mallopt(_M_TRIM_THRESHOLD, 4096)
    except Exception:
        # Non-glibc libc (musl, macOS) — silently ignore
        pass


def configure_glibc_env_for_children() -> None:
    """Set glibc tuning env vars before spawning workers.

    Also LD_PRELOADs jemalloc if available (set ``LEAN_RL_NO_JEMALLOC=1``
    to disable). Call once in the trainer before ``_start_workers()``.
    """
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ["MALLOC_MMAP_THRESHOLD_"] = "4096"
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "4096"

    # --- jemalloc LD_PRELOAD (opt-out via LEAN_RL_NO_JEMALLOC=1) ---
    if not os.environ.get("LEAN_RL_NO_JEMALLOC"):
        _try_enable_jemalloc()


def _try_enable_jemalloc() -> None:
    """Find and LD_PRELOAD jemalloc for child processes."""
    import sys
    from pathlib import Path

    candidates = [
        # Ray's bundled jemalloc (most reliable on conda envs)
        Path(sys.prefix)
        / "lib"
        / "python3.10"
        / "site-packages"
        / "ray"
        / "core"
        / "libjemalloc.so",
        # System-wide
        Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"),
        Path("/usr/lib/libjemalloc.so.2"),
    ]
    # Also try site-packages in the current env
    try:
        import ray

        ray_dir = Path(ray.__file__).parent
        candidates.insert(0, ray_dir / "core" / "libjemalloc.so")
    except ImportError:
        pass

    for path in candidates:
        if path.is_file():
            existing = os.environ.get("LD_PRELOAD", "")
            if str(path) not in existing:
                os.environ["LD_PRELOAD"] = (
                    f"{path}:{existing}" if existing else str(path)
                )
            import logging

            logging.getLogger(__name__).info(
                f"jemalloc enabled for worker subprocesses: {path}"
            )
            return


def set_oom_score_adj(score: int = 1000) -> None:
    """Set OOM score adjustment (1000 = kill first). Silently ignores errors."""
    try:
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
            f.write(str(score))
    except (PermissionError, OSError, FileNotFoundError):
        pass


# Exit code used when the RSS watchdog kills a worker.  The trainer
# recognises this code and restarts the worker without logging an
# unexpected-crash warning.
RSS_WATCHDOG_EXIT_CODE: int = 42


def start_rss_watchdog(
    hard_cap_gb: float = MAX_WORKER_RSS_GB,
    check_interval: float = 1.0,
) -> threading.Thread:
    """Daemon thread that ``os._exit()``s when RSS exceeds *hard_cap_gb*.

    Catches growth during blocking C calls where Python-level checks
    can't fire. Returns the started thread.
    """

    def _watchdog() -> None:
        _wd_pid = os.getpid()
        _wd_checks = 0
        logger.info(
            f"[RSS WATCHDOG] Started for PID {_wd_pid}, "
            f"hard_cap={hard_cap_gb:.1f} GB, "
            f"interval={check_interval}s"
        )
        while True:
            try:
                rss = get_rss_gb()
                _wd_checks += 1
                # Periodic heartbeat every 60 checks (~60s)
                if _wd_checks % 60 == 0:
                    logger.debug(
                        f"[RSS WATCHDOG] PID {_wd_pid} "
                        f"check #{_wd_checks} rss={rss:.2f}GB "
                        f"cap={hard_cap_gb:.1f}GB"
                    )
                if rss > hard_cap_gb:
                    pid = os.getpid()
                    try:
                        with open(f"/proc/{pid}/comm") as f:
                            comm = f.read().strip()
                    except Exception:
                        comm = "unknown"
                    msg = (
                        f"[RSS WATCHDOG] {comm} (PID {pid}): "
                        f"RSS {rss:.1f} GB exceeds hard cap "
                        f"{hard_cap_gb:.1f} GB. Exiting NOW."
                    )
                    try:
                        sys.stderr.write(msg + "\n")
                        sys.stderr.flush()
                    except Exception:
                        pass

                    # Best effort diagnostic snapshot before forced exit.
                    # Keep this resilient: any failure still proceeds to exit.
                    try:
                        os.makedirs(_DIAG_DIR, exist_ok=True)
                        dump_path = os.path.join(
                            _DIAG_DIR,
                            f"rss_watchdog_pid{pid}_{int(threading.get_native_id())}.txt",
                        )
                        written = dump_memory_diagnostic(dump_path)
                        try:
                            sys.stderr.write(
                                f"[RSS WATCHDOG] wrote memory diagnostic: {written}\n"
                            )
                            sys.stderr.flush()
                        except Exception:
                            pass
                    except Exception:
                        pass

                    os._exit(RSS_WATCHDOG_EXIT_CODE)
            except Exception:
                pass  # /proc read failure — skip this cycle
            # Use Event.wait for clean shutdown capability, but a daemon
            # thread is fine with plain sleep.
            threading.Event().wait(check_interval)

    t = threading.Thread(target=_watchdog, daemon=True, name="rss-watchdog")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Memory diagnostic dump (triggered by SIGUSR1 or explicit call)
# ---------------------------------------------------------------------------

# Directory where per-worker diagnostic files are written.
_DIAG_DIR = "logs"


def _read_proc_file(path: str) -> str:
    """Read a /proc file, returning '' on any error."""
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def _mallinfo2_dict() -> dict:
    """Call glibc ``mallinfo2()`` and return its fields as a dict."""
    libc = _get_libc()
    if libc is None:
        return {}

    # struct mallinfo2 has 10 size_t fields (LP64: 10 × 8 = 80 bytes).
    import ctypes as _ct

    class _Mallinfo2(_ct.Structure):
        _fields_ = [
            ("arena", _ct.c_size_t),  # Non-mmapped space allocated from system
            ("ordblks", _ct.c_size_t),  # Number of free chunks
            ("smblks", _ct.c_size_t),  # Number of fast-bin blocks
            ("hblks", _ct.c_size_t),  # Number of mmapped regions
            ("hblkhd", _ct.c_size_t),  # Space in mmapped regions
            ("usmblks", _ct.c_size_t),  # Unused — always 0
            ("fsmblks", _ct.c_size_t),  # Space in fast-bin blocks
            ("uordblks", _ct.c_size_t),  # Total allocated space
            ("fordblks", _ct.c_size_t),  # Total free space
            ("keepcost", _ct.c_size_t),  # Releasable (via malloc_trim) space
        ]

    try:
        libc.mallinfo2.restype = _Mallinfo2
        mi = libc.mallinfo2()
        # _fields_ can be 2-tuple (name, type) or 3-tuple (name, type, size)
        return {field[0]: getattr(mi, field[0]) for field in _Mallinfo2._fields_}
    except (AttributeError, OSError):
        # glibc < 2.33 doesn't have mallinfo2
        return {}


#: Above this RSS (GiB), ``dump_memory_diagnostic`` skips the expensive
#: ``gc.get_objects()`` passes to avoid hanging on bloated heaps.
FAST_DUMP_RSS_THRESHOLD_GB: float = 4.0


def _parse_smaps_top_regions(pid: int, top_n: int = 20) -> list[str]:
    """Parse ``/proc/<pid>/smaps`` and return the *top_n* VMAs by RSS.

    Each returned string looks like::

        1234 MB  [heap]  (rw-p)
         567 MB  /usr/lib/libc.so.6  (r-xp)

    Returns an empty list on any error.
    """
    regions: list[tuple[int, str, str]] = []  # (rss_kb, label, perms)
    try:
        with open(f"/proc/{pid}/smaps") as f:
            current_label = ""
            current_perms = ""
            current_rss = 0
            in_region = False
            for line in f:
                if line and line[0] in "0123456789abcdef":
                    # New VMA header line
                    if in_region and current_rss > 0:
                        regions.append((current_rss, current_label, current_perms))
                    parts = line.split()
                    current_perms = parts[1] if len(parts) > 1 else ""
                    current_label = parts[-1] if len(parts) > 5 else "[anon]"
                    current_rss = 0
                    in_region = True
                elif line.startswith("Rss:"):
                    current_rss = int(line.split()[1])  # kB
            if in_region and current_rss > 0:
                regions.append((current_rss, current_label, current_perms))
    except Exception:
        return []

    regions.sort(key=lambda x: -x[0])
    result = []
    for rss_kb, label, perms in regions[:top_n]:
        rss_mb = rss_kb / 1024
        result.append(f"  {rss_mb:>8.1f} MB  {label:40s} ({perms})")
    return result


def dump_memory_diagnostic(out_path: str | None = None) -> str:
    """Write a detailed memory diagnostic snapshot to *out_path*.

    Captures /proc/self/status, smaps_rollup, top smaps VMAs, glibc
    mallinfo2, and Python object counts (skipped when RSS is high).
    Returns the path of the written file.
    """
    pid = os.getpid()
    if out_path is None:
        import time as _time_mod

        os.makedirs(_DIAG_DIR, exist_ok=True)
        out_path = os.path.join(
            _DIAG_DIR,
            f"memdump_pid{pid}_{int(_time_mod.time())}.txt",
        )

    lines: list[str] = []
    # Define MB at function scope to avoid "possibly unbound" errors
    MB = 1024 * 1024

    def _section(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    import time as _time

    rss_gb = get_rss_gb()
    fast_mode = rss_gb > FAST_DUMP_RSS_THRESHOLD_GB

    lines.append(f"Memory diagnostic for PID {pid}")
    lines.append(f"Timestamp: {_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"RSS: {rss_gb:.3f} GB")
    if fast_mode:
        lines.append(
            f"MODE: FAST (RSS {rss_gb:.1f} GB > {FAST_DUMP_RSS_THRESHOLD_GB:.0f} GB "
            f"threshold — skipping gc.get_objects() to avoid OOM/hang)"
        )

    # ---- /proc/self/status (VmRSS, VmSize, VmData, VmStk, etc.) ----
    _section("/proc/self/status (selected)")
    status = _read_proc_file(f"/proc/{pid}/status")
    for line in status.splitlines():
        if line.startswith(("Vm", "Rss", "Threads")):
            lines.append(f"  {line}")

    # ---- /proc/self/smaps_rollup ----
    _section("/proc/self/smaps_rollup")
    rollup = _read_proc_file(f"/proc/{pid}/smaps_rollup")
    if rollup:
        for line in rollup.splitlines():
            lines.append(f"  {line}")
    else:
        lines.append("  (not available)")

    # ---- glibc mallinfo2 ----
    _section("glibc mallinfo2()")
    mi = _mallinfo2_dict()
    if mi:
        lines.append(f"  arena      (sbrk heap)      : {mi['arena'] / MB:>10.1f} MB")
        lines.append(
            f"  uordblks   (in-use allocs)   : {mi['uordblks'] / MB:>10.1f} MB"
        )
        lines.append(
            f"  fordblks   (free in arenas)  : {mi['fordblks'] / MB:>10.1f} MB"
        )
        lines.append(
            f"  keepcost   (trimmable top)   : {mi['keepcost'] / MB:>10.1f} MB"
        )
        lines.append(f"  hblkhd     (mmap'd regions)  : {mi['hblkhd'] / MB:>10.1f} MB")
        lines.append(f"  hblks      (# mmap regions)  : {mi['hblks']:>10d}")
        lines.append(f"  ordblks    (# free chunks)   : {mi['ordblks']:>10d}")
        lines.append(f"  fsmblks    (fastbin space)   : {mi['fsmblks'] / MB:>10.1f} MB")
        lines.append(f"  smblks     (# fastbin blocks): {mi['smblks']:>10d}")
        # Derived
        total_heap = mi["arena"] + mi["hblkhd"]
        fragmentation = mi["fordblks"] / mi["arena"] * 100 if mi["arena"] else 0
        lines.append("  ---")
        lines.append(f"  total heap (arena+mmap)      : {total_heap / MB:>10.1f} MB")
        lines.append(f"  fragmentation (free/arena)   : {fragmentation:>10.1f} %")
    else:
        lines.append("  (mallinfo2 not available — glibc < 2.33?)")

    # ---- Top smaps VMAs by RSS ----
    _section("Top 20 smaps VMAs by RSS")
    vma_lines = _parse_smaps_top_regions(pid, top_n=20)
    if vma_lines:
        lines.append(f"  {'RSS':>10s}  {'Mapping':40s}  Perms")
        lines.append(f"  {'---':>10s}  {'-------':40s}  -----")
        lines.extend(vma_lines)
    else:
        lines.append("  (could not read /proc/self/smaps)")

    # ---- Fast-mode early return (skip expensive gc iteration) ----
    if fast_mode:
        _section("Top 30 Python object types by instance count")
        lines.append(
            f"  SKIPPED — RSS {rss_gb:.1f} GB exceeds fast-dump threshold. "
            f"gc.get_objects() on a heap this large risks OOM or 60+ second hang."
        )
        _section("Top 30 Python object types by estimated size")
        lines.append("  SKIPPED (fast mode)")
        _section("Large individual objects (> 1 MB)")
        lines.append("  SKIPPED (fast mode)")
        _section("gc statistics")
        for i, stats in enumerate(gc.get_stats()):
            lines.append(f"  gen {i}: {stats}")

        # ---- Write out ----
        text = "\n".join(lines) + "\n"
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                f.write(text)
        except Exception:
            sys.stderr.write(text)
        return out_path

    # ---- Top Python objects by count ----
    _section("Top 30 Python object types by instance count")
    # Initialize dictionaries before try block to avoid "possibly unbound" errors
    type_counts: dict[str, int] = {}
    type_sizes: dict[str, int] = {}
    try:
        gc.collect()
        for obj in gc.get_objects():
            tn = type(obj).__name__
            type_counts[tn] = type_counts.get(tn, 0) + 1
            try:
                type_sizes[tn] = type_sizes.get(tn, 0) + sys.getsizeof(obj)
            except (TypeError, ValueError):
                pass

        for tn, count in sorted(type_counts.items(), key=lambda x: -x[1])[:30]:
            sz_mb = type_sizes.get(tn, 0) / MB
            lines.append(f"  {tn:40s}  count={count:>8d}  size~{sz_mb:>8.1f} MB")
    except Exception as e:
        lines.append(f"  (error: {e})")

    # ---- Top Python objects by size ----
    _section("Top 30 Python object types by estimated size")
    try:
        for tn, sz in sorted(type_sizes.items(), key=lambda x: -x[1])[:30]:
            sz_mb = sz / (1024 * 1024)
            count = type_counts.get(tn, 0)
            lines.append(f"  {tn:40s}  size~{sz_mb:>8.1f} MB  count={count:>8d}")
    except Exception as e:
        lines.append(f"  (error: {e})")

    # ---- Large individual objects ----
    _section("Large individual objects (> 1 MB)")
    try:
        large_objs: list[tuple[str, int, str]] = []
        for obj in gc.get_objects():
            try:
                sz = sys.getsizeof(obj)
                if sz > 1024 * 1024:
                    tn = type(obj).__name__
                    # Get a short repr to identify it
                    try:
                        r = repr(obj)[:120]
                    except Exception:
                        r = "<no repr>"
                    large_objs.append((tn, sz, r))
            except (TypeError, ValueError):
                pass
        large_objs.sort(key=lambda x: -x[1])
        for tn, sz, r in large_objs[:30]:
            sz_mb = sz / (1024 * 1024)
            lines.append(f"  {tn:30s} {sz_mb:>8.1f} MB  {r}")
    except Exception as e:
        lines.append(f"  (error: {e})")

    # ---- gc stats ----
    _section("gc statistics")
    for i, stats in enumerate(gc.get_stats()):
        lines.append(f"  gen {i}: {stats}")

    # ---- Write out ----
    text = "\n".join(lines) + "\n"
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
    except Exception:
        # Last resort: print to stderr
        sys.stderr.write(text)

    return out_path


def install_memory_dump_signal_handler() -> None:
    """Install a SIGUSR1 handler that writes a memory diagnostic to ``logs/``."""
    if not hasattr(signal, "SIGUSR1"):
        return

    def _handler(signum: int, frame: object) -> None:
        try:
            path = dump_memory_diagnostic()
            # Write path to stderr (loguru may be unsafe in signal handler)
            sys.stderr.write(f"[SIGUSR1] Memory diagnostic written to {path}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[SIGUSR1] Failed to dump diagnostic: {e}\n")
            sys.stderr.flush()

    signal.signal(signal.SIGUSR1, _handler)
