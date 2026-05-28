"""Sleep prevention and the privileged power sampler, with root kept minimal.

The benchmark harness runs unprivileged. Only the powermetrics sampler is
elevated: it is launched as ``sudo powermetrics ...`` so the user grants root to
that one auditable command and nothing else (R6.1). Sleep is blocked with
``caffeinate`` bound to the harness PID so it never outlives the run (R6.5).

Command construction is pure and tested; process spawning is injected so tests
never need sudo or a Mac.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Spawn = Callable[[list[str]], "subprocess.Popen"]


def caffeinate_command(pid: int) -> list[str]:
    # -dimsu: keep display, idle, disk, system awake; -w waits on (binds to) our PID.
    return ["caffeinate", "-dimsu", "-w", str(pid)]


def powermetrics_command(log_path: str | Path, interval_ms: int, samplers: list[str]) -> list[str]:
    # `-n` keeps sudo non-interactive: with credentials already cached by
    # authorize_sudo, sudo runs without ever touching /dev/tty (a stray sudo
    # warning printed to the controlling terminal would corrupt the fullscreen
    # Live — that's the "C-c fixes the screen" bug).
    return [
        "sudo",
        "-n",
        "powermetrics",
        "--samplers",
        ",".join(samplers),
        "-i",
        str(interval_ms),
        "-f",
        "plist",
        "-o",
        str(log_path),
    ]


def _default_spawn(argv: list[str]) -> "subprocess.Popen":
    # stdin=DEVNULL + stdout/stderr=DEVNULL muzzles the child's normal channels.
    # We deliberately do NOT pass start_new_session=True: macOS sudo caches
    # credentials per controlling tty (default `timestamp_type=tty`), and a
    # setsid-detached child can no longer find that cache — `sudo -n` then fails
    # silently and powermetrics never starts. Staying in our session keeps the
    # tty-scoped cache reachable. With `sudo -n` and cached creds, sudo runs
    # without ever prompting and powermetrics writes its plist as expected.
    return subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def authorize_sudo(runner=subprocess.run) -> bool:
    """Pre-authorize sudo in the foreground (`sudo -v`) so the user sees the password
    prompt once, before the background sampler starts. The sampler then reuses the
    cached credentials and never blocks silently on a prompt. Returns True on success.
    """
    try:
        return runner(["sudo", "-v"]).returncode == 0
    except (FileNotFoundError, OSError):
        return False


@dataclass
class PowerSession:
    """Context manager: prevents sleep, and (if enabled) runs the sudo sampler.

    On exit it terminates the sampler and waits so powermetrics flushes the plist
    before any post-hoc alignment reads it.
    """

    log_path: Path
    interval_ms: int
    samplers: list[str]
    enabled: bool = True
    spawn: Spawn = _default_spawn
    _sampler: "subprocess.Popen | None" = None
    _caffeinate: "subprocess.Popen | None" = None

    def __enter__(self) -> "PowerSession":
        self._caffeinate = self.spawn(caffeinate_command(os.getpid()))
        if self.enabled:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._sampler = self.spawn(powermetrics_command(self.log_path, self.interval_ms, self.samplers))
        return self

    def __exit__(self, *exc) -> None:
        for proc in (self._sampler, self._caffeinate):
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=10)  # let powermetrics flush the plist before we read it
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
        self._sampler = None
        self._caffeinate = None
