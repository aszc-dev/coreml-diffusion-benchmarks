"""Shared Rich console, theme, and byte formatting for the interactive commands."""

from rich.console import Console
from rich.theme import Theme

THEME = Theme(
    {
        "sdbench.title": "bold cyan",
        "sdbench.dim": "dim",
        "sdbench.warn": "bold yellow",
        "sdbench.danger": "bold red",
        "sdbench.ok": "bold green",
        "sdbench.size": "bold magenta",
    }
)

console = Console(theme=THEME)


def human_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string (binary units)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
