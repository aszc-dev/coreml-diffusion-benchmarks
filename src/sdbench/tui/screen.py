"""Full-screen Rich primitives: keyboard input and shared chrome.

A single ``Live(screen=True)`` owns the terminal (alternate screen), so views
redraw in place instead of scrolling the history. Views update layout regions
and block on ``read_key()``. This is the flashy, full-screen surface the tui
requirements call for.
"""

import contextlib
import os
import shutil
import time

import readchar
from readchar import key as _key
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sdbench.tui.console import console, human_bytes

UP, DOWN, LEFT, RIGHT, ENTER, SPACE, ESC = "up", "down", "left", "right", "enter", "space", "esc"


def _build_keymap() -> dict:
    mapping: dict[str, str] = {}
    for name, token in (("UP", UP), ("DOWN", DOWN), ("LEFT", LEFT), ("RIGHT", RIGHT), ("ENTER", ENTER), ("ESC", ESC)):
        value = getattr(_key, name, None)
        if value:
            mapping[value] = token
    mapping["\r"] = ENTER
    mapping["\n"] = ENTER
    mapping[" "] = SPACE
    return mapping


_KEYMAP = _build_keymap()


def read_key() -> str:
    """Block for one keypress, normalized to a small token set (or the literal char)."""
    try:
        ch = readchar.readkey()
    except KeyboardInterrupt:
        return ESC
    if ch in _KEYMAP:
        return _KEYMAP[ch]
    return ch.lower() if len(ch) == 1 else ch


def _tree_size(path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def workspace_bytes(ws) -> int:
    """Bytes consumed by sdbench's own generated files (artifacts, results, cache, input)."""
    return sum(_tree_size(d) for d in (ws.artifacts_dir, ws.results_dir, ws.shared_input_dir, ws.cache_dir))


# Walking the artifact tree is heavier than shutil.disk_usage, so cache it with a
# short TTL — the dashboard refreshes many times per second but our footprint
# barely moves within a run.
_USAGE_CACHE: dict[str, tuple[float, int]] = {}


def workspace_bytes_cached(ws, ttl: float = 2.0) -> int:
    key = str(ws.root)
    now = time.monotonic()
    cached = _USAGE_CACHE.get(key)
    if cached is not None and (now - cached[0]) < ttl:
        return cached[1]
    value = workspace_bytes(ws)
    _USAGE_CACHE[key] = (now, value)
    return value


def invalidate_usage(ws) -> None:
    _USAGE_CACHE.pop(str(ws.root), None)


def usage_text(ws) -> str:
    """Corner readout: how much *our* files use, plus whole-disk fullness as a nice extra."""
    used = workspace_bytes_cached(ws)
    disk = shutil.disk_usage(ws.root)
    pct = (disk.used / disk.total * 100) if disk.total else 0.0
    return f"sdbench {human_bytes(used)}  ·  disk {pct:.0f}% used"


def state_text(state) -> Text:
    """Workspace-state chips for the header (✓ present / – missing)."""
    chips = [
        (state.checkpoint_present, "checkpoint"),
        (state.artifacts_total > 0 and state.artifacts_present == state.artifacts_total, f"artifacts {state.artifacts_present}/{state.artifacts_total}"),
        (state.has_runplan, "run plan"),
        (state.has_results, "results"),
        (getattr(state, "has_report", False), "report"),
    ]
    text = Text()
    for ok, label in chips:
        text.append("✓ " if ok else "– ", style="sdbench.ok" if ok else "sdbench.warn")
        text.append(f"{label}    ", style="sdbench.dim")
    return text


def header(title: str, chips: Text, disk_text: str) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right")
    grid.add_row(Text(title, style="sdbench.title"), Text(disk_text, style="sdbench.size"))
    grid.add_row(chips, Text(""))
    return Panel(grid, border_style="sdbench.dim", padding=(0, 1))


def footer(hint: str) -> Panel:
    return Panel(Text(hint, style="sdbench.dim"), border_style="sdbench.dim", padding=(0, 1))


class _UnbufferedFdFile:
    """A text-file-like that writes straight to a fd with no Python-side buffering.

    Why we need this: while the dashboard is running, BOTH the main thread (run
    loop hooks) and the capture-reader thread call ``live.refresh()``, which writes
    a full frame through ``console.file``. Rich serialises the API calls, but a
    line-buffered TextIOWrapper still holds the tail of each frame (Rich frames
    end on escape codes, not newlines) until the next write flushes it — at which
    point the two threads' tails can come out in the wrong order, shifting the
    cursor by a char and breaking the borders. Bypassing Python's buffer by going
    straight to ``os.write`` makes every Rich write self-flushing and atomic.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def write(self, data: str) -> int:
        encoded = data.encode("utf-8", errors="replace")
        view = memoryview(encoded)
        while view:
            written = os.write(self._fd, view)
            view = view[written:]
        return len(data)

    def flush(self) -> None:  # already flushed — provided for API compatibility
        return None

    def isatty(self) -> bool:
        try:
            return os.isatty(self._fd)
        except OSError:
            return False

    def fileno(self) -> int:
        return self._fd

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        return None  # the fd's lifetime is managed by live_screen, not by us


@contextlib.contextmanager
def live_screen():
    """Own the alternate screen for a full-screen view.

    Duplicates the real-terminal fd into a saved fd and wraps it in an
    UNBUFFERED text-file-like so every Rich write goes straight to the terminal
    without sitting in a Python buffer (see _UnbufferedFdFile for why). Pins
    that file as the Console's file for the Live's lifetime — any later fd 1/2
    redirect (run capture) does NOT divert the Live's writes because they
    travel through the saved fd.
    """
    saved_fd = os.dup(1)
    saved_file = _UnbufferedFdFile(saved_fd)
    previous = console.file
    console.file = saved_file
    try:
        # auto_refresh=True so the Live owns its own draw thread. Every other
        # thread (main run loop, capture reader thread) only mutates state via
        # live.update(); Rich's internal thread is the sole writer to the
        # terminal. Without this, concurrent refresh() calls from main+reader
        # threads desync the cursor and shift the frame by a char (the bug we
        # chased through buffering and sudo). refresh_per_second=10 keeps the
        # bars / spinners smooth without churning.
        with Live(console=console, screen=True, auto_refresh=True, refresh_per_second=10) as live:
            yield live
    finally:
        console.file = previous
        try:
            os.close(saved_fd)
        except OSError:
            pass


def frame(header_panel, body, footer_panel) -> Layout:
    root = Layout()
    root.split_column(
        Layout(header_panel, name="header", size=4),
        Layout(body, name="body", ratio=1),
        Layout(footer_panel, name="footer", size=3),
    )
    return root
