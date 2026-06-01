"""Gate the run before power numbers are recorded in a contaminated env.

The first published contributor run was done on battery with a runaway
``AddressBookManager`` chewing 53% CPU; latency stood, but the per-engine W
and J figures were not comparable to any other run. ``check_power_env``
exists to refuse those conditions by default."""

from sdbench.tui import preflight


class _FakePowerState:
    def __init__(self, ac_powered: bool, low_power_mode: bool):
        self.ac_powered = ac_powered
        self.battery_percent = 80
        self.low_power_mode = low_power_mode
        self.sleep_disabled = False
        self.display_sleep_min = 10
        self.caffeinate_pids: list[int] = []


def _patch_env(monkeypatch, *, ac: bool, low_power: bool, loadavg: float | None):
    monkeypatch.setattr(
        preflight.telemetry,
        "collect_host_power_state",
        lambda: _FakePowerState(ac_powered=ac, low_power_mode=low_power),
    )
    monkeypatch.setattr(preflight.os, "getloadavg", lambda: (loadavg, loadavg, loadavg) if loadavg is not None else (_ for _ in ()).throw(OSError()))


def test_check_power_env_passes_on_ac_quiet_host(monkeypatch):
    _patch_env(monkeypatch, ac=True, low_power=False, loadavg=0.6)

    check = preflight.check_power_env()

    assert check.ok
    assert check.ac_ok and check.low_power_ok and check.loadavg_ok
    assert check.issues == []


def test_check_power_env_refuses_battery(monkeypatch):
    _patch_env(monkeypatch, ac=False, low_power=False, loadavg=0.4)

    check = preflight.check_power_env()

    assert not check.ok
    assert not check.ac_ok
    assert any("AC power" in msg for msg in check.issues)


def test_check_power_env_refuses_low_power_mode(monkeypatch):
    _patch_env(monkeypatch, ac=True, low_power=True, loadavg=0.4)

    check = preflight.check_power_env()

    assert not check.ok
    assert not check.low_power_ok
    assert any("low-power" in msg for msg in check.issues)


def test_measure_idle_loadavg_takes_median(monkeypatch):
    readings = iter([3.0, 2.8, 2.9, 5.0, 2.7])  # one transient spike
    monkeypatch.setattr(preflight.os, "getloadavg", lambda: (next(readings), 0.0, 0.0))

    base = preflight.measure_idle_loadavg(samples=5, sleep=lambda _s: None)

    assert base == 2.9  # median rejects the 5.0 spike


def test_measure_idle_loadavg_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(preflight.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError()))

    assert preflight.measure_idle_loadavg(sleep=lambda _s: None) is None


def test_loadavg_ceiling_is_baseline_plus_margin():
    assert preflight.loadavg_ceiling(2.9, margin=1.0) == 3.9


def test_loadavg_ceiling_falls_back_to_absolute_without_baseline():
    assert preflight.loadavg_ceiling(None) == preflight.POWER_LOADAVG_MAX


def test_check_power_env_flags_noisy_host_without_refusing(monkeypatch):
    # 5.4 is the loadavg the bad contributor run reproduced — a useful canary.
    # A noisy host is no longer refused at env-check time: the 1-min EWMA carries
    # our own tail between passes, so loadavg is waited out before each pass
    # (run_cmd._await_quiescent_host) and only flagged here, never gating ``ok``.
    _patch_env(monkeypatch, ac=True, low_power=False, loadavg=5.4)

    check = preflight.check_power_env(loadavg_max=2.0)

    assert check.ok  # AC + not low-power: power numbers are the right *type*
    assert not check.loadavg_ok
    assert any("loadavg" in msg for msg in check.issues)
