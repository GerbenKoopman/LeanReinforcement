"""
Live terminal progress display for training and benchmarking.

Uses the ``rich`` library to render a compact, tqdm‑style live dashboard
that replaces the wall of logger.info() lines during data collection.

Key design decisions:
  * All *file*-level logging (JSON dumps, wandb, per-theorem detail logs)
    is untouched; only the *terminal* noise is consolidated into the live
    panel.
  * The display degrades gracefully: if ``rich`` is not installed, a plain
    fallback prints a one-liner every N theorems (same as before).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ── Dataclass that holds the running stats ──────────────────────────────────


@dataclass
class ProgressStats:
    """Mutable bag of counters updated by the trainer during data collection."""

    # Benchmark-level context (optional, set by run_benchmark)
    benchmark_run_index: int = 0  # 1-based
    benchmark_total_runs: int = 0
    benchmark_run_name: str = ""

    # Epoch-level
    current_epoch: int = 0
    total_epochs: int = 0

    # Theorem-level (within current epoch)
    total_theorems: int = 0
    theorems_done: int = 0
    theorems_proved: int = 0
    theorems_failed: int = 0

    # Cumulative across all epochs in this run
    cumulative_theorems_done: int = 0
    cumulative_theorems_proved: int = 0
    cumulative_total_theorems: int = 0

    # Timing
    epoch_start_time: float = 0.0
    run_start_time: float = 0.0

    # Value head training
    value_head_epoch: int = 0
    value_head_total_epochs: int = 0
    value_head_train_loss: float = 0.0
    value_head_val_loss: float = 0.0
    value_head_best_val_loss: float = float("inf")
    value_head_patience_counter: int = 0
    value_head_patience_limit: int = 0

    # Phase indicator
    phase: str = "idle"  # "collecting", "training_value_head", "saving", "idle"

    # Last theorem result (for the scrolling line)
    last_theorem_name: str = ""
    last_theorem_success: Optional[bool] = None
    last_theorem_time: float = 0.0

    # Workers
    alive_workers: int = 0
    total_workers: int = 0

    # Recent theorem log (ring buffer for last N results)
    recent_results: List[Dict[str, Any]] = field(default_factory=list)
    _max_recent: int = 5

    def record_theorem(
        self, name: str, success: bool, steps: int = 0, elapsed: float = 0.0
    ) -> None:
        self.theorems_done += 1
        self.cumulative_theorems_done += 1
        if success:
            self.theorems_proved += 1
            self.cumulative_theorems_proved += 1
        else:
            self.theorems_failed += 1
        self.last_theorem_name = name
        self.last_theorem_success = success
        self.last_theorem_time = elapsed
        self.recent_results.append(
            {"name": name, "ok": success, "steps": steps, "t": elapsed}
        )
        if len(self.recent_results) > self._max_recent:
            self.recent_results.pop(0)

    def reset_epoch(self, epoch: int, total_theorems: int) -> None:
        self.current_epoch = epoch
        self.total_theorems = total_theorems
        self.theorems_done = 0
        self.theorems_proved = 0
        self.theorems_failed = 0
        self.epoch_start_time = time.time()
        self.recent_results = []
        self.last_theorem_name = ""
        self.last_theorem_success = None
        self.phase = "collecting"

    # ── Computed properties ─────────────────────────────────────────────

    @property
    def epoch_elapsed(self) -> float:
        if self.epoch_start_time == 0:
            return 0.0
        return time.time() - self.epoch_start_time

    @property
    def run_elapsed(self) -> float:
        if self.run_start_time == 0:
            return 0.0
        return time.time() - self.run_start_time

    @property
    def epoch_success_rate(self) -> float:
        if self.theorems_done == 0:
            return 0.0
        return self.theorems_proved / self.theorems_done

    @property
    def cumulative_success_rate(self) -> float:
        if self.cumulative_theorems_done == 0:
            return 0.0
        return self.cumulative_theorems_proved / self.cumulative_theorems_done

    @property
    def epoch_progress_frac(self) -> float:
        if self.total_theorems == 0:
            return 0.0
        return self.theorems_done / self.total_theorems

    @property
    def overall_progress_frac(self) -> float:
        if self.cumulative_total_theorems == 0:
            return 0.0
        return self.cumulative_theorems_done / self.cumulative_total_theorems

    @property
    def epoch_eta_seconds(self) -> Optional[float]:
        if self.theorems_done == 0 or self.epoch_elapsed == 0:
            return None
        per_thm = self.epoch_elapsed / self.theorems_done
        remaining = self.total_theorems - self.theorems_done
        return per_thm * remaining

    @property
    def run_eta_seconds(self) -> Optional[float]:
        if self.cumulative_theorems_done == 0 or self.run_elapsed == 0:
            return None
        per_thm = self.run_elapsed / self.cumulative_theorems_done
        remaining = self.cumulative_total_theorems - self.cumulative_theorems_done
        return per_thm * remaining


# ── Formatting helpers ──────────────────────────────────────────────────────


def _fmt_time(seconds: Optional[float]) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    if seconds is None or seconds < 0:
        return "--:--"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def _bar(frac: float, width: int = 30) -> str:
    """ASCII progress bar."""
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


# ── Rich live display ───────────────────────────────────────────────────────


class LiveProgressDisplay:
    """
    Manages a ``rich.live.Live`` context that renders a compact dashboard.

    Usage::

        display = LiveProgressDisplay(stats)
        display.start()
        # … update stats in your loop …
        display.refresh()   # call periodically or after each theorem
        display.stop()
    """

    def __init__(self, stats: ProgressStats, refresh_rate: float = 4.0):
        self._stats = stats
        self._console = (
            Console(stderr=True)  # type: ignore[possibly-undefined]
            if HAS_RICH
            else None
        )
        self._live: Optional[Any] = None
        self._refresh_rate = refresh_rate
        self._loguru_handler_id: Optional[int] = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def _redirect_loguru(self) -> None:
        """Redirect loguru to write through rich's console so log lines
        appear *above* the live panel instead of clashing with it."""
        try:
            from loguru import logger as _logger

            # Remove loguru's default stderr handler (id=0) to prevent clashes
            _logger.remove(0)

            # Add a handler that writes through the rich console
            console = self._console

            def _rich_sink(message: Any) -> None:
                """Write loguru messages through rich so they render above
                the Live panel."""
                text = str(message).rstrip("\n")
                if console is not None:
                    console.print(text, highlight=False, markup=False)

            self._loguru_handler_id = _logger.add(
                _rich_sink, format="{message}", level="DEBUG"
            )
        except Exception:
            # If anything goes wrong, leave loguru alone
            pass

    def _restore_loguru(self) -> None:
        """Restore loguru's default stderr handler."""
        try:
            from loguru import logger as _logger

            if self._loguru_handler_id is not None:
                _logger.remove(self._loguru_handler_id)
                self._loguru_handler_id = None
            # Re-add the default stderr handler
            _logger.add(sys.stderr)
        except Exception:
            pass

    def start(self) -> None:
        if not HAS_RICH or self._console is None:
            return
        self._live = Live(  # type: ignore[possibly-undefined]
            self._render(),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            transient=False,
        )
        self._live.start()
        self._redirect_loguru()

    def stop(self) -> None:
        if self._live is not None:
            self._restore_loguru()
            # Render one final frame so the summary stays visible
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def print_line(self, text: str) -> None:
        """Print a permanent line above the live display (e.g. for theorem results)."""
        if self._live is not None:
            self._console.print(text)  # type: ignore[union-attr]
        else:
            print(text, file=sys.stderr)

    # ── Rendering ───────────────────────────────────────────────────────

    def _render(self) -> Panel:
        s = self._stats

        # ── Title ───────────────────────────────────────────────────────
        if s.benchmark_run_name:
            title = (
                f" Benchmark {s.benchmark_run_index}/{s.benchmark_total_runs}: "
                f"[bold cyan]{s.benchmark_run_name}[/] "
            )
        else:
            title = f" [bold cyan]{s.phase.replace('_', ' ').title()}[/] "

        # ── Build content lines ─────────────────────────────────────────
        lines: List[str] = []

        if s.phase == "collecting":
            # Epoch info
            pct = s.epoch_progress_frac * 100
            bar = _bar(s.epoch_progress_frac)
            eta = _fmt_time(s.epoch_eta_seconds)
            elapsed = _fmt_time(s.epoch_elapsed)

            lines.append(
                f"[bold]Epoch {s.current_epoch}/{s.total_epochs}[/]  "
                f"{bar} [bold]{pct:5.1f}%[/]  "
                f"{s.theorems_done}/{s.total_theorems} theorems  "
                f"[dim]elapsed {elapsed}  eta {eta}[/]"
            )

            # Success / failure counts
            sr = s.epoch_success_rate * 100
            lines.append(
                f"  [green]✓ {s.theorems_proved}[/] proved  "
                f"[red]✗ {s.theorems_failed}[/] failed  "
                f"[yellow]success rate {sr:.1f}%[/]  "
                f"[dim]workers {s.alive_workers}/{s.total_workers}[/]"
            )

            # Cumulative / overall progress (if multi-epoch)
            if s.total_epochs > 1:
                cum_pct = s.overall_progress_frac * 100
                cum_bar = _bar(s.overall_progress_frac, width=20)
                cum_sr = s.cumulative_success_rate * 100
                run_eta = _fmt_time(s.run_eta_seconds)
                run_elapsed = _fmt_time(s.run_elapsed)
                lines.append(
                    f"  [dim]Overall {cum_bar} {cum_pct:5.1f}%  "
                    f"{s.cumulative_theorems_done}/{s.cumulative_total_theorems} thms  "
                    f"avg success {cum_sr:.1f}%  "
                    f"elapsed {run_elapsed}  eta {run_eta}[/]"
                )

            # Last few theorem results
            if s.recent_results:
                parts = []
                for r in s.recent_results:
                    icon = "[green]✓[/]" if r["ok"] else "[red]✗[/]"
                    name = r["name"]
                    if len(name) > 35:
                        name = "…" + name[-34:]
                    parts.append(f"  {icon} {name} [dim]({r['t']:.0f}s)[/]")
                lines.append("")
                lines.extend(parts)

        elif s.phase == "training_value_head":
            pct = (
                (s.value_head_epoch / s.value_head_total_epochs * 100)
                if s.value_head_total_epochs > 0
                else 0
            )
            bar = _bar(s.value_head_epoch / max(1, s.value_head_total_epochs), width=30)
            lines.append(
                f"[bold]Value Head Training[/]  "
                f"{bar} [bold]{pct:5.1f}%[/]  "
                f"epoch {s.value_head_epoch}/{s.value_head_total_epochs}"
            )
            lines.append(
                f"  train loss [bold]{s.value_head_train_loss:.4f}[/]  "
                f"val loss [bold]{s.value_head_val_loss:.4f}[/]  "
                f"best [bold green]{s.value_head_best_val_loss:.4f}[/]  "
                f"patience {s.value_head_patience_counter}/{s.value_head_patience_limit}"
            )

        elif s.phase == "saving":
            lines.append("[bold]Saving checkpoint & training data…[/]")

        else:
            lines.append("[dim]Idle[/]")

        content = "\n".join(lines)
        return Panel(  # type: ignore[possibly-undefined]
            content, title=title, border_style="blue", expand=True
        )


# ── Plain fallback (no rich) ───────────────────────────────────────────────


class PlainProgressDisplay:
    """Minimal fallback when ``rich`` is not installed."""

    def __init__(self, stats: ProgressStats, log_every: int = 5):
        self._stats = stats
        self._log_every = log_every
        self._last_logged = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        s = self._stats
        if s.phase == "collecting" and s.theorems_done > 0:
            self._print_status()

    def refresh(self) -> None:
        s = self._stats
        if s.theorems_done - self._last_logged >= self._log_every:
            self._print_status()
            self._last_logged = s.theorems_done

    def print_line(self, text: str) -> None:
        # Strip rich markup for plain output
        import re

        clean = re.sub(r"\[/?[^\]]*\]", "", text)
        print(clean, file=sys.stderr)

    def _print_status(self) -> None:
        s = self._stats
        pct = s.epoch_progress_frac * 100
        sr = s.epoch_success_rate * 100
        elapsed = _fmt_time(s.epoch_elapsed)
        eta = _fmt_time(s.epoch_eta_seconds)
        print(
            f"  Epoch {s.current_epoch}/{s.total_epochs}  "
            f"{s.theorems_done}/{s.total_theorems} ({pct:.0f}%)  "
            f"✓{s.theorems_proved} ✗{s.theorems_failed} ({sr:.0f}%)  "
            f"elapsed {elapsed}  eta {eta}",
            file=sys.stderr,
        )


# ── Factory ─────────────────────────────────────────────────────────────────


def make_progress_display(
    stats: ProgressStats,
) -> "LiveProgressDisplay | PlainProgressDisplay":
    if HAS_RICH:
        return LiveProgressDisplay(stats)
    return PlainProgressDisplay(stats)
