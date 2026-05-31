"""The pre-pass quiescence barrier (R6).

``loadavg_1m`` is a 1-minute EWMA, so right after a pass it still carries the
harness's own tail. Rather than refuse the next pass (forcing the operator to
restart), ``_await_quiescent_host`` polls — refreshing the reading each tick —
until the host settles, or gives up at a cap and flags the pass instead."""

import sdbench.tui.run_cmd as run_cmd


class _Reporter:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(message)


class _Thermal:
    def __init__(self, throttled: bool = False, detail: str = "nominal"):
        self.throttled = throttled
        self.detail = detail


def _quiet_thermal(monkeypatch):
    monkeypatch.setattr(run_cmd, "check_thermal_state", lambda: _Thermal())


def _no_sleep(monkeypatch):
    monkeypatch.setattr(run_cmd.time, "sleep", lambda _s: None)


def test_returns_true_once_loadavg_drops(monkeypatch):
    loads = iter([3.1, 2.5, 1.2])  # decays under the ceiling on the third poll
    monkeypatch.setattr(run_cmd.os, "getloadavg", lambda: (next(loads), 0.0, 0.0))
    _quiet_thermal(monkeypatch)
    _no_sleep(monkeypatch)

    reporter = _Reporter()
    settled = run_cmd._await_quiescent_host(reporter, loadavg_max=2.0, cap_s=120, poll_s=5)

    assert settled is True
    assert any("waiting for quiet host" in line for line in reporter.lines)


def test_returns_true_immediately_on_quiet_host(monkeypatch):
    monkeypatch.setattr(run_cmd.os, "getloadavg", lambda: (0.5, 0.0, 0.0))
    _quiet_thermal(monkeypatch)
    _no_sleep(monkeypatch)

    reporter = _Reporter()
    settled = run_cmd._await_quiescent_host(reporter, loadavg_max=2.0, cap_s=120, poll_s=5)

    assert settled is True
    assert reporter.lines == []  # already quiet: no waiting noise


def test_gives_up_and_flags_on_cap(monkeypatch):
    monkeypatch.setattr(run_cmd.os, "getloadavg", lambda: (9.0, 0.0, 0.0))
    _quiet_thermal(monkeypatch)
    _no_sleep(monkeypatch)

    reporter = _Reporter()
    settled = run_cmd._await_quiescent_host(reporter, loadavg_max=2.0, cap_s=10, poll_s=5)

    assert settled is False
    assert any("not quiet" in line for line in reporter.lines)


def test_waits_on_thermal_throttle(monkeypatch):
    monkeypatch.setattr(run_cmd.os, "getloadavg", lambda: (0.5, 0.0, 0.0))
    states = iter([_Thermal(throttled=True, detail="hot"), _Thermal()])
    monkeypatch.setattr(run_cmd, "check_thermal_state", lambda: next(states))
    _no_sleep(monkeypatch)

    reporter = _Reporter()
    settled = run_cmd._await_quiescent_host(reporter, loadavg_max=2.0, cap_s=120, poll_s=5)

    assert settled is True


def test_loadavg_unavailable_does_not_block(monkeypatch):
    def _raise():
        raise OSError("getloadavg unavailable")

    monkeypatch.setattr(run_cmd.os, "getloadavg", lambda: _raise())
    _quiet_thermal(monkeypatch)
    _no_sleep(monkeypatch)

    reporter = _Reporter()
    settled = run_cmd._await_quiescent_host(reporter, loadavg_max=2.0, cap_s=120, poll_s=5)

    assert settled is True  # unknown loadavg is treated as ok, not a hang
