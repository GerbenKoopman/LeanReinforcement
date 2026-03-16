#!/usr/bin/env python3
"""
Advanced memory diagnostic wrapper for Lean RL training.
Monitors memory usage and kills the workload if system RAM exceeds 85%.
Globally scans for Ray daemons, Ray workers, and Lean binaries, bypassing standard process trees.
"""

import subprocess
import psutil
import time
import sys
import signal
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple


class MemoryMonitor:
    """Monitors memory usage of the global RL workload and kills it if threshold is exceeded."""

    MEMORY_THRESHOLD = 85.0  # Kill at 85% system memory
    SNAPSHOT_THRESHOLD = 65.0  # Proactive diagnostic snapshot at 65%
    POLL_INTERVAL = 0.05  # 50ms samples (20x per second)
    SPIKE_THRESHOLD = 5.0  # Kill if memory jumps >5% in a single 50ms sample
    LOG_INTERVAL = 5.0  # Periodic logging interval (seconds)

    def __init__(self):
        self.total_system_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.total_system_ram_bytes = psutil.virtual_memory().total
        self.start_time = datetime.now()
        self.peak_memory_percent = 0.0
        self.peak_memory_gb = 0.0
        self.samples_taken = 0
        self.last_memory_percent = 0.0
        self.spike_detected = False
        self.spike_history: List[Tuple[float, datetime]] = []
        self.snapshot_taken = False  # True after proactive 65% snapshot
        self.current_uid = os.getuid()

        self.log_path = (
            f"memory_stress_test_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.last_log_time = time.monotonic()

        with open(self.log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                "timestamp,elapsed_s,system_percent,system_used_gb,system_available_gb,"
                "total_rss_mb,main_rss_mb,ray_rss_mb,lean_rss_mb,process_count,"
                "ray_count,lean_count,samples,delta_percent,event,"
                "python_pids_rss\n"
            )

    def _append_log_line(
        self, system_percent: float, stats: Dict[str, Any], event: str = "periodic"
    ):
        system_mem = psutil.virtual_memory()
        elapsed_s = (datetime.now() - self.start_time).total_seconds()
        delta_percent = system_percent - self.last_memory_percent
        timestamp = datetime.now().isoformat(timespec="seconds")

        # Build per-PID RSS field for python/ray processes: "pid:rss_mb|pid:rss_mb"
        python_pids_rss = "|".join(
            f"{name.rsplit('_', 1)[1]}:{rss_mb:.0f}"
            for name, rss_mb in stats.get("process_details", [])
            if name.startswith("python") and "_" in name
        )

        line = (
            f"{timestamp},{elapsed_s:.2f},{system_percent:.2f},"
            f"{system_mem.used / (1024**3):.3f},{system_mem.available / (1024**3):.3f},"
            f"{stats['total_rss_mb']:.2f},{stats['main_rss_mb']:.2f},"
            f"{stats['ray_rss_mb']:.2f},{stats['lean_binary_rss_mb']:.2f},"
            f"{stats['process_count']},{stats['ray_count']},{stats['lean_count']},"
            f"{self.samples_taken},{delta_percent:.2f},{event},"
            f"{python_pids_rss}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(line)

    def maybe_log_intermediate_state(
        self, system_percent: float, stats: Dict[str, Any]
    ):
        now = time.monotonic()
        if (now - self.last_log_time) >= self.LOG_INTERVAL:
            self._append_log_line(system_percent, stats, event="periodic")
            self.last_log_time = now

    @staticmethod
    def _classify_process(name: str, cmdline_str: str) -> str:
        """Classify process as main/ray/python/lean/other with strict Lean matching."""
        name = (name or "").lower()
        cmdline_str = (cmdline_str or "").lower()

        # Python first: prevent false "lean" matches from module names like
        # lean_reinforcement in python command lines.
        if name.startswith("python") or " python" in f" {cmdline_str} ":
            return "python"

        # Ray workers/daemons
        if (
            "ray" in name
            or "ray::" in name
            or "default_worker.py" in cmdline_str
            or "ray " in cmdline_str
            or "/ray/" in cmdline_str
        ):
            return "ray"

        # Strict Lean binary detection: executable name lean/lake only.
        if name in {"lean", "lake"}:
            return "lean"

        cmd_parts = [part.strip().split("/")[-1] for part in cmdline_str.split()]
        if any(part in {"lean", "lake"} for part in cmd_parts):
            return "lean"

        return "other"

    def get_all_workload_processes(self, main_pid: int) -> List[psutil.Process]:
        """Globally scan for all processes belonging to this RL run."""
        processes = []
        target_pids = set()

        # 1. Grab Main process and its direct strict children
        try:
            main_proc = psutil.Process(main_pid)
            processes.append(main_proc)
            target_pids.add(main_pid)
            for child in main_proc.children(recursive=True):
                processes.append(child)
                target_pids.add(child.pid)
        except psutil.NoSuchProcess:
            pass

        # 2. Globally scan for detached Ray workers, Daemons, and Lean binaries
        for proc in psutil.process_iter(["pid", "name", "cmdline", "uids"]):
            try:
                pid = proc.info["pid"]
                if pid in target_pids:
                    continue

                # Only check processes owned by the current user
                if not proc.info["uids"] or proc.info["uids"].real != self.current_uid:
                    continue

                name = (proc.info["name"] or "").lower()
                cmdline = proc.info["cmdline"] or []
                cmdline_str = " ".join(str(c).lower() for c in cmdline)

                proc_type = self._classify_process(name, cmdline_str)

                if proc_type in {"ray", "lean", "python"}:
                    processes.append(psutil.Process(pid))
                    target_pids.add(pid)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return processes

    def get_workload_memory(self, pid: int) -> Dict[str, Any]:
        """Calculate memory for the globally discovered workload."""
        stats: Dict[str, Any] = {
            "main_rss_mb": 0.0,
            "ray_rss_mb": 0.0,
            "lean_binary_rss_mb": 0.0,
            "total_rss_mb": 0.0,
            "process_count": 0,
            "lean_count": 0,
            "ray_count": 0,
            "process_details": [],
        }

        processes = self.get_all_workload_processes(pid)

        for proc in processes:
            try:
                mem = proc.memory_info()
                rss_mb = mem.rss / (1024**2)

                name = proc.name().lower()
                cmdline = proc.cmdline() or []
                cmdline_str = " ".join(str(c).lower() for c in cmdline)

                stats["total_rss_mb"] += rss_mb
                stats["process_count"] += 1

                proc_type = self._classify_process(name, cmdline_str)

                if proc.pid == pid:
                    stats["main_rss_mb"] = rss_mb
                    stats["process_details"].append(("main_process", rss_mb))
                elif proc_type == "lean":
                    stats["lean_binary_rss_mb"] += rss_mb
                    stats["lean_count"] += 1
                    stats["process_details"].append((f"lean_{proc.pid}", rss_mb))
                elif proc_type in {"ray", "python"}:
                    stats["ray_rss_mb"] += rss_mb
                    stats["ray_count"] += 1
                    stats["process_details"].append((f"python_{proc.pid}", rss_mb))
                else:
                    stats["process_details"].append((f"child_{proc.pid}", rss_mb))

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Sort details by memory usage (descending) so we see the biggest hogs first
        stats["process_details"].sort(key=lambda x: x[1], reverse=True)
        return stats

    def _read_smaps_rollup(self, pid: int) -> str:
        """Read /proc/<pid>/smaps_rollup for an external process."""
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                return f.read()
        except Exception:
            return "(could not read smaps_rollup)"

    def _read_smaps_top_regions(self, pid: int, top_n: int = 20) -> str:
        """Parse /proc/<pid>/smaps and return top VMAs by RSS.

        This is the GOLD diagnostic for memory bloat: it shows exactly
        which heap, mmap, or library mapping is consuming physical RAM.
        Works externally on any process — no cooperation needed.

        Now includes virtual address ranges to distinguish heap vs mmap
        vs thread-stack VMAs.
        """
        # (rss_kb, label, perms, addr_range)
        regions: List[Tuple[int, str, str, str]] = []
        try:
            with open(f"/proc/{pid}/smaps") as f:
                current_label = ""
                current_perms = ""
                current_rss = 0
                current_addr = ""
                in_region = False
                for line in f:
                    if line and line[0] in "0123456789abcdef":
                        if in_region and current_rss > 0:
                            regions.append(
                                (
                                    current_rss,
                                    current_label,
                                    current_perms,
                                    current_addr,
                                )
                            )
                        parts = line.split()
                        current_addr = parts[0] if parts else ""
                        current_perms = parts[1] if len(parts) > 1 else ""
                        current_label = parts[-1] if len(parts) > 5 else "[anon]"
                        current_rss = 0
                        in_region = True
                    elif line.startswith("Rss:"):
                        current_rss = int(line.split()[1])
                if in_region and current_rss > 0:
                    regions.append(
                        (current_rss, current_label, current_perms, current_addr)
                    )
        except Exception:
            return "(could not read /proc/{}/smaps)".format(pid)

        regions.sort(key=lambda x: -x[0])
        lines = [f"  {'RSS':>10s}  {'Mapping':40s}  {'Perms':6s}  Address Range"]
        lines.append(f"  {'---':>10s}  {'-------':40s}  {'-----':6s}  -------------")
        for rss_kb, label, perms, addr in regions[:top_n]:
            rss_mb = rss_kb / 1024
            lines.append(f"  {rss_mb:>8.1f} MB  {label:40s} ({perms:4s})  {addr}")
        total_rss_kb = sum(r[0] for r in regions)
        lines.append(f"  {'---':>10s}")
        lines.append(
            f"  {total_rss_kb / 1024:>8.1f} MB  TOTAL across {len(regions)} VMAs"
        )
        return "\n".join(lines)

    def _read_proc_status_vm(self, pid: int) -> str:
        """Read Vm* and Rss* lines from /proc/<pid>/status."""
        try:
            lines = []
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith(("Vm", "Rss", "Threads")):
                        lines.append(line.rstrip())
            return "\n".join(lines)
        except Exception:
            return "(could not read)"

    def profile_memory_hog(self, main_pid: int, stats: Dict) -> None:
        """Profile the top memory-consuming process before killing.

        1. Identifies the biggest memory hog (excluding the main process).
        2. Reads /proc/<pid>/smaps_rollup AND full smaps VMA breakdown
           externally (no cooperation needed — works on any process).
        3. Sends SIGUSR1 to ALL Python workers above 500 MB — the
           worker's signal handler writes a full diagnostic to
           logs/memdump_pid<PID>_<timestamp>.txt.
        4. Waits proportionally (3s base + 2s per GB above 2 GB) so
           large-heap workers have time to write their dump.
        5. Verifies dump files were created and warns if any are missing.
        """
        print("\n" + "=" * 80)
        print("🔬 PROFILING TOP MEMORY HOG BEFORE KILL")
        print("=" * 80)

        if not stats.get("process_details"):
            print("  No process details available")
            return

        # Find the top hog (skip main process for profiling)
        top_name, top_rss_mb = None, 0.0
        top_pid = None
        for name, rss_mb in stats["process_details"]:
            if name == "main_process":
                continue
            if rss_mb > top_rss_mb:
                top_name, top_rss_mb = name, rss_mb
                # Extract PID from the name format "python_12345" or "lean_12345"
                parts = name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    top_pid = int(parts[1])

        if top_pid is None:
            print("  Could not identify top memory hog PID")
            return

        print(f"\n  Top hog: {top_name} — {top_rss_mb:.0f} MB (PID {top_pid})")

        # ---- External /proc inspection (works on any process) ----
        print(f"\n  --- /proc/{top_pid}/status (Vm* lines) ---")
        print(f"  {self._read_proc_status_vm(top_pid)}")

        print(f"\n  --- /proc/{top_pid}/smaps_rollup ---")
        rollup = self._read_smaps_rollup(top_pid)
        for line in rollup.splitlines():
            print(f"    {line}")

        # ---- Full smaps VMA breakdown (the GOLD diagnostic) ----
        # Shows exactly which heap/mmap/library mappings hold the RSS.
        print(f"\n  --- /proc/{top_pid}/smaps top VMAs by RSS ---")
        print(self._read_smaps_top_regions(top_pid, top_n=20))

        # ---- Send SIGUSR1 to ALL Python workers above 500 MB ----
        # Previously only signaled top 2; now signal all to guarantee
        # we capture the ballooning worker even if classification varies.
        signaled_pids: List[int] = []
        for name, rss_mb in stats["process_details"]:
            if not name.startswith("python"):
                continue
            if rss_mb < 500:
                continue
            parts = name.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            sig_pid = int(parts[1])
            print(f"\n  Sending SIGUSR1 to PID {sig_pid} ({name}, {rss_mb:.0f} MB)...")
            try:
                os.kill(sig_pid, signal.SIGUSR1)
                signaled_pids.append(sig_pid)
            except OSError as e:
                print(f"  Failed to signal PID {sig_pid}: {e}")

        if signaled_pids:
            # Scale wait time: 3s base + 2s per GB above 2 GB for the top hog.
            # A 13 GB worker needs ~25s; the fast-mode dump (skipping gc
            # iteration) should complete in <2s even at that size.
            wait_time = max(5.0, 3.0 + max(0, (top_rss_mb - 2000) / 1000) * 2)
            print(f"\n  Waiting {wait_time:.0f}s for diagnostic dumps to complete...")
            time.sleep(wait_time)

            # ---- Verify dump files were created ----
            import glob

            for sig_pid in signaled_pids:
                pattern = f"logs/memdump_pid{sig_pid}_*.txt"
                dumps = sorted(glob.glob(pattern))
                if dumps:
                    latest = dumps[-1]
                    size_kb = os.path.getsize(latest) / 1024
                    print(
                        f"  ✅ PID {sig_pid}: dump written ({latest}, {size_kb:.0f} KB)"
                    )
                else:
                    print(
                        f"  ❌ PID {sig_pid}: NO dump file found matching {pattern}. "
                        f"Worker may have been blocked in C code or killed before "
                        f"the signal handler could run."
                    )
        else:
            print("\n  No Python workers above 500 MB to signal.")

        # ---- Also profile the main process ----
        main_rss = stats.get("main_rss_mb", 0)
        if main_rss > 2000:  # Only if main is using significant memory
            print(
                f"\n  --- Main process (PID {main_pid}, {main_rss:.0f} MB) smaps_rollup ---"
            )
            rollup = self._read_smaps_rollup(main_pid)
            for line in rollup.splitlines():
                print(f"    {line}")

            # Also show the main process VMA breakdown
            print(f"\n  --- Main process (PID {main_pid}) smaps top VMAs by RSS ---")
            print(self._read_smaps_top_regions(main_pid, top_n=15))

        print("=" * 80)

    def kill_workload(self, main_pid: int):
        """Forcefully kill all globally discovered workload processes."""
        print("\n" + "=" * 80)
        print("🚨 EMERGENCY SHUTDOWN - KILLING GLOBAL WORKLOAD 🚨")
        print("=" * 80)

        processes = self.get_all_workload_processes(main_pid)

        for proc in processes:
            try:
                # Don't kill the main process until the end
                if proc.pid != main_pid:
                    print(f"  Killing PID {proc.pid} ({proc.name()})")
                    proc.send_signal(signal.SIGKILL)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        try:
            print(f"  Killing main PID {main_pid}")
            main_proc = psutil.Process(main_pid)
            main_proc.send_signal(signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Wait for processes to die
        try:
            gone, alive = psutil.wait_procs(processes, timeout=3)
            if alive:
                print(f"  Warning: {len(alive)} processes still alive")
        except Exception:
            pass

    def format_memory_bar(self, percent: float, width: int = 30) -> str:
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        if percent >= self.MEMORY_THRESHOLD:
            return f"🔴 [{bar}]"
        elif percent >= 70:
            return f"🟡 [{bar}]"
        return f"🟢 [{bar}]"

    def print_status(
        self,
        system_percent: float,
        stats: Dict,
        spike: bool = False,
        spike_value: float = 0.0,
    ):
        print("\033[H\033[J", end="", flush=True)

        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split(".")[0]

        print("=" * 80)
        print(f"🔬 LEAN RL GLOBAL MEMORY TEST - Running {elapsed_str}")
        if spike:
            print(f"⚡ SPIKE DETECTED: +{spike_value:.1f}% in one 50ms sample!")
        print("=" * 80)

        system_mem = psutil.virtual_memory()
        print(f"\n📊 SYSTEM MEMORY ({self.total_system_ram_gb:.1f} GB total):")
        print(
            f"   Used:      {system_mem.used / (1024**3):.2f} GB ({system_percent:.1f}%)"
        )
        print(f"   Available: {system_mem.available / (1024**3):.2f} GB")
        print(f"   {self.format_memory_bar(system_percent)} {system_percent:.1f}%")

        print(
            f"\n🌳 GLOBAL WORKLOAD ({stats['total_rss_mb']:.0f} MB across {stats['process_count']} processes):"
        )
        print(f"   Main:      {stats['main_rss_mb']:>8.0f} MB")
        print(
            f"   Ray:       {stats['ray_rss_mb']:>8.0f} MB ({stats['ray_count']} procs)"
        )
        print(
            f"   Lean:      {stats['lean_binary_rss_mb']:>8.0f} MB ({stats['lean_count']} procs)"
        )

        # Show top 8 memory hogs
        if stats["process_details"]:
            print("\n   Top Process Memory Hogs:")
            for name, rss_mb in stats["process_details"][:8]:
                print(f"     • {name:25s}: {rss_mb:>7.0f} MB")

        print("\n📈 MONITORING STATS:")
        print(
            f"   Samples:   {self.samples_taken:>6d} ({self.POLL_INTERVAL}s interval)"
        )
        print(
            f"   Peak:      {self.peak_memory_gb:>6.2f} GB ({self.peak_memory_percent:>5.1f}%)"
        )
        print(f"   Last Δ:    {system_percent - self.last_memory_percent:>6.1f}%")

        if self.spike_history:
            recent_spikes = [s for s, _ in self.spike_history[-5:]]
            print(f"   Spikes (↑5%): {', '.join(f'{s:.1f}%' for s in recent_spikes)}")

        print("\n⚙️  THRESHOLDS:")
        print(
            f"   Kill:  {self.MEMORY_THRESHOLD:.1f}% RAM  |  Spike: {self.SPIKE_THRESHOLD:.1f}% per {self.POLL_INTERVAL*1000:.0f}ms"
        )

        if spike:
            print(
                f"\n🔥 **SPIKE EXCEEDED THRESHOLD: {spike_value:.1f}% > {self.SPIKE_THRESHOLD:.1f}%**"
            )
        elif system_percent >= self.MEMORY_THRESHOLD:
            print(
                f"\n🚨 **SUSTAINED HIGH MEMORY: {system_percent:.1f}% >= {self.MEMORY_THRESHOLD:.1f}%**"
            )
        elif system_percent >= 70:
            print(f"\n⚠️  Memory approaching threshold ({system_percent:.1f}%)")

        print("=" * 80)
        sys.stdout.flush()

    def monitor_sample(self, pid: int) -> bool:
        self.samples_taken += 1
        system_mem = psutil.virtual_memory()
        system_percent = system_mem.percent

        stats = self.get_workload_memory(pid)

        if system_percent > self.peak_memory_percent:
            self.peak_memory_percent = system_percent
            self.peak_memory_gb = system_mem.used / (1024**3)

        spike = False
        spike_value = 0.0
        if self.samples_taken > 1:
            spike_value = system_percent - self.last_memory_percent
            if spike_value > self.SPIKE_THRESHOLD:
                spike = True
                self.spike_detected = True
                self.spike_history.append((spike_value, datetime.now()))

        self.print_status(system_percent, stats, spike=spike, spike_value=spike_value)
        self.maybe_log_intermediate_state(system_percent, stats)

        self.last_memory_percent = system_percent

        # ---- Proactive snapshot at 65% (well before 85% kill) ----
        # Two-pronged approach:
        # 1. EXTERNAL: Read /proc/<pid>/smaps directly for the top Python
        #    process — works even if the process is blocked in C code or
        #    doesn't have a SIGUSR1 handler installed.
        # 2. INTERNAL: Send SIGUSR1 to all Python workers for mallinfo2.
        if not self.snapshot_taken and system_percent >= self.SNAPSHOT_THRESHOLD:
            self.snapshot_taken = True
            self._append_log_line(system_percent, stats, event="snapshot_65pct")

            # ---- Identify all Python processes and the top hog ----
            python_procs = []
            for name, rss_mb in stats.get("process_details", []):
                if not name.startswith("python"):
                    continue
                parts = name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    python_procs.append((int(parts[1]), rss_mb, name))

            python_procs.sort(key=lambda x: -x[1])  # biggest first

            print(
                f"\n📸 PROACTIVE SNAPSHOT at {system_percent:.1f}% — "
                f"{len(python_procs)} Python processes found"
            )
            for ppid, rss_mb, pname in python_procs:
                print(f"  PID {ppid:>7d}: {rss_mb:>8.0f} MB  ({pname})")

            # ---- EXTERNAL DIAGNOSTIC: smaps VMA for top 3 Python PIDs ----
            # This is the GOLD diagnostic — no cooperation needed.
            snapshot_dir = "logs"
            os.makedirs(snapshot_dir, exist_ok=True)
            for ppid, rss_mb, pname in python_procs[:3]:
                ts = int(time.time())
                snap_path = f"{snapshot_dir}/snapshot_65pct_pid{ppid}_{ts}.txt"
                print(f"\n  📸 External smaps dump for PID {ppid} ({rss_mb:.0f} MB)...")
                try:
                    with open(snap_path, "w") as sf:
                        sf.write(f"=== 65% Snapshot — PID {ppid} ({pname}) ===\n")
                        sf.write(f"RSS at snapshot: {rss_mb:.1f} MB\n")
                        sf.write(f"System memory: {system_percent:.1f}%\n")
                        elapsed_s = (datetime.now() - self.start_time).total_seconds()
                        sf.write(f"Elapsed: {elapsed_s:.1f}s\n\n")

                        # /proc/status VM lines
                        sf.write("--- /proc/status ---\n")
                        sf.write(self._read_proc_status_vm(ppid))
                        sf.write("\n\n")

                        # smaps_rollup
                        sf.write("--- /proc/smaps_rollup ---\n")
                        sf.write(self._read_smaps_rollup(ppid))
                        sf.write("\n\n")

                        # Top VMAs by RSS (the real gold)
                        sf.write("--- Top VMAs by RSS ---\n")
                        sf.write(self._read_smaps_top_regions(ppid, top_n=30))
                        sf.write("\n\n")

                        # cmdline
                        sf.write("--- /proc/cmdline ---\n")
                        try:
                            with open(f"/proc/{ppid}/cmdline") as cf:
                                sf.write(cf.read().replace("\x00", " "))
                        except Exception:
                            sf.write("(could not read)")
                        sf.write("\n\n")

                        # fd count + any interesting fds (shm, anon_inode)
                        sf.write("--- /proc/fd summary ---\n")
                        try:
                            fds = os.listdir(f"/proc/{ppid}/fd")
                            sf.write(f"Total open fds: {len(fds)}\n")
                            # check for shm / memfd / deleted fds
                            shm_count = 0
                            memfd_count = 0
                            for fd_name in fds[:500]:  # cap to avoid huge dirs
                                try:
                                    link = os.readlink(f"/proc/{ppid}/fd/{fd_name}")
                                    if "/dev/shm" in link or "shm" in link:
                                        shm_count += 1
                                    if "memfd" in link:
                                        memfd_count += 1
                                except Exception:
                                    pass
                            sf.write(f"shm-related fds: {shm_count}\n")
                            sf.write(f"memfd fds: {memfd_count}\n")
                        except Exception as e:
                            sf.write(f"(could not read: {e})")
                        sf.write("\n")

                    print(
                        f"  ✅ Written {snap_path} ({os.path.getsize(snap_path) / 1024:.0f} KB)"
                    )
                except Exception as e:
                    print(f"  ❌ Failed to write {snap_path}: {e}")

            # ---- Also snapshot the main process if it's large ----
            main_rss = stats.get("main_rss_mb", 0)
            if main_rss > 2000:
                ts = int(time.time())
                snap_path = f"{snapshot_dir}/snapshot_65pct_main_{pid}_{ts}.txt"
                print(
                    f"\n  📸 External smaps dump for MAIN PID {pid} ({main_rss:.0f} MB)..."
                )
                try:
                    with open(snap_path, "w") as sf:
                        sf.write(f"=== 65% Snapshot — MAIN PID {pid} ===\n")
                        sf.write(f"RSS: {main_rss:.1f} MB\n\n")
                        sf.write("--- /proc/status ---\n")
                        sf.write(self._read_proc_status_vm(pid))
                        sf.write("\n\n--- /proc/smaps_rollup ---\n")
                        sf.write(self._read_smaps_rollup(pid))
                        sf.write("\n\n--- Top VMAs by RSS ---\n")
                        sf.write(self._read_smaps_top_regions(pid, top_n=30))
                        sf.write("\n")
                    print(f"  ✅ Written {snap_path}")
                except Exception as e:
                    print(f"  ❌ Failed: {e}")

            # ---- INTERNAL: SIGUSR1 for mallinfo2 from workers ----
            signaled = 0
            for ppid, rss_mb, pname in python_procs:
                try:
                    os.kill(ppid, signal.SIGUSR1)
                    signaled += 1
                except OSError:
                    pass
            print(f"\n  📸 Also sent SIGUSR1 to {signaled} workers for internal dumps.")

        if system_percent >= self.MEMORY_THRESHOLD:
            self._append_log_line(system_percent, stats, event="threshold_exceeded")
            return False
        if spike:
            self._append_log_line(system_percent, stats, event="spike_detected")
            return False

        return True

    def print_final_report(self, stats: Dict[str, Any]):
        elapsed = datetime.now() - self.start_time
        print("\n" + "=" * 80)
        print("📋 FINAL REPORT")
        print("=" * 80)
        print(f"Duration:              {str(elapsed).split('.')[0]}")
        print(f"Samples Collected:     {self.samples_taken}")
        print(
            f"Peak Memory Usage:     {self.peak_memory_gb:.2f} GB ({self.peak_memory_percent:.1f}%)"
        )

        if self.spike_history:
            print("\nTransient Spikes (>5% per 50ms):")
            for delta, ts in self.spike_history[-10:]:
                elapsed_at_spike = (ts - self.start_time).total_seconds()
                print(f"  • {elapsed_at_spike:>6.1f}s: +{delta:.1f}%")

        print("\nFinal Workload Memory:")
        print(f"  Main:        {stats['main_rss_mb']:>8.0f} MB")
        print(
            f"  Ray:         {stats['ray_rss_mb']:>8.0f} MB ({stats['ray_count']} procs)"
        )
        print(
            f"  Lean:        {stats['lean_binary_rss_mb']:>8.0f} MB ({stats['lean_count']} procs)"
        )
        print(f"  Total:       {stats['total_rss_mb']:>8.0f} MB")
        print("=" * 80)


def main():
    cmd = [
        sys.executable,
        "-m",
        "lean_reinforcement.training.train",
        "--data-type",
        "novel_premises",
        "--mcts-type",
        "guided_rollout",
        "--num-epochs",
        "3",
        "--num-theorems",
        "100",
        "--num-iterations",
        "200",
        "--max-steps",
        "10",
        "--num-workers",
        "4",
        "--value-head-latent-dim",
        "1024",
        "--train-epochs",
        "50",
        "--train-value-head",
        "--use-final-reward",
        "--save-training-data",
        "--save-checkpoints",
        "--env-timeout",
        "72",
        "--max-time",
        "60",
        "--proof-timeout",
        "300.0",
        "--full-search",
    ]

    print("=" * 80)
    print("🚀 LEAN RL TRAINING WITH GLOBAL MEMORY MONITORING")
    print("=" * 80)
    print(f"\n{' '.join(cmd[:5])} [...]\n")
    print("Mode: GLOBAL SCAN (Catching Ray Daemons & detached workers)")
    print("Starting in 3 seconds...\n")
    time.sleep(3)

    process = subprocess.Popen(cmd, stdout=None, stderr=None)
    monitor = MemoryMonitor()
    print(f"📝 Intermediate state log: {monitor.log_path}")
    print(f"✅ Subprocess PID {process.pid} started\n")
    print("=" * 80 + "\n")
    time.sleep(1)

    try:
        while True:
            if process.poll() is not None:
                print("\n\n✅ Training process completed normally")
                break

            if not monitor.monitor_sample(process.pid):
                final_stats = monitor.get_workload_memory(process.pid)
                # Profile the top memory hog BEFORE killing everything.
                # This sends SIGUSR1 to get internal diagnostics and
                # reads /proc/<pid>/smaps_rollup externally.
                monitor.profile_memory_hog(process.pid, final_stats)
                monitor.kill_workload(process.pid)
                monitor.print_final_report(final_stats)

                if monitor.spike_detected:
                    print("\n❌ TERMINATED: Transient memory spike detected")
                    return 2
                else:
                    print("\n❌ TERMINATED: Sustained high memory usage")
                    return 1

            time.sleep(MemoryMonitor.POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        try:
            final_stats = monitor.get_workload_memory(process.pid)
            monitor._append_log_line(
                psutil.virtual_memory().percent, final_stats, event="keyboard_interrupt"
            )
        except Exception:
            pass
        monitor.kill_workload(process.pid)
        return 130

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        try:
            final_stats = monitor.get_workload_memory(process.pid)
            monitor._append_log_line(
                psutil.virtual_memory().percent, final_stats, event="unexpected_error"
            )
        except Exception:
            pass
        if process.poll() is None:
            monitor.kill_workload(process.pid)
        return 1

    finally:
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        except Exception:
            pass

    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
