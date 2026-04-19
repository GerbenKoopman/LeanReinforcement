from lean_reinforcement.training import progress


class _FakeStderr:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_make_progress_display_uses_plain_when_not_tty(monkeypatch) -> None:
    monkeypatch.setattr(progress.sys, "stderr", _FakeStderr(False))
    monkeypatch.delenv("LEAN_RL_FORCE_LIVE_PROGRESS", raising=False)
    monkeypatch.delenv("LEAN_RL_DISABLE_LIVE_PROGRESS", raising=False)

    stats = progress.ProgressStats()
    display = progress.make_progress_display(stats, enable_live=True)

    assert isinstance(display, progress.PlainProgressDisplay)


def test_make_progress_display_disable_live_env(monkeypatch) -> None:
    monkeypatch.setattr(progress.sys, "stderr", _FakeStderr(True))
    monkeypatch.setenv("LEAN_RL_DISABLE_LIVE_PROGRESS", "1")

    stats = progress.ProgressStats()
    display = progress.make_progress_display(stats, enable_live=True)

    assert isinstance(display, progress.PlainProgressDisplay)


def test_make_progress_display_respects_enable_live_flag(monkeypatch) -> None:
    monkeypatch.setattr(progress.sys, "stderr", _FakeStderr(True))

    stats = progress.ProgressStats()
    display = progress.make_progress_display(stats, enable_live=False)

    assert isinstance(display, progress.NullProgressDisplay)
