"""Checkpoint acquisition with mandatory SHA verification.

The same SD 1.5 weights must back every backend (R0.2), so the checkpoint is
pinned by SHA-256 and verified whether the user points at a local file or has it
auto-downloaded from the official Hugging Face repo. A wrong or corrupt file is
rejected, not silently benchmarked. Downloads are cached under the workspace and
re-verified on reuse.

Network and hashing are injected/streamed so the resolver is testable offline.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sdbench.provenance import sha256_file
from sdbench.tui.console import console, human_bytes


@dataclass(frozen=True)
class CheckpointSpec:
    repo: str
    filename: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


# Official SD 1.5 weights. SHA pinned to the canonical pruned-emaonly safetensors.
DEFAULT_CHECKPOINT = CheckpointSpec(
    repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned-emaonly.safetensors",
    sha256="6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa",
)


def verify_sha256(path: str | Path, expected: str) -> bool:
    return sha256_file(path).lower() == expected.lower()


def download_file(url: str, dest: str | Path, *, urlopen: Callable = urllib.request.urlopen, chunk: int = 1 << 20) -> Path:
    """Stream a URL to `dest`, showing a progress bar. Returns the destination path."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    response = urlopen(url)
    total = _content_length(response)
    with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn(), console=console) as progress:
        task = progress.add_task("download", total=total)
        with out.open("wb") as handle:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                handle.write(block)
                progress.advance(task, len(block))
    _close(response)
    return out


def resolve_checkpoint(
    ws,
    spec: CheckpointSpec = DEFAULT_CHECKPOINT,
    *,
    explicit: str | Path | None = None,
    auto_download: bool = False,
    downloader: Callable[[str, Path], Path] | None = None,
) -> Path:
    """Return a verified checkpoint path, or raise with an actionable message.

    Order: an explicit path (verified) -> the workspace cache (verified) ->
    auto-download (then verified). A SHA mismatch is always fatal.
    """
    downloader = downloader or (lambda url, dest: download_file(url, dest))

    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        _require_match(path, spec)
        return path

    cached = ws.cache_dir / spec.filename
    if cached.is_file() and verify_sha256(cached, spec.sha256):
        console.print(f"[sdbench.ok]Using cached checkpoint[/] {cached} [sdbench.dim](sha verified)[/]")
        return cached

    if not auto_download:
        raise FileNotFoundError(
            f"No verified checkpoint. Point at a local {spec.filename} or pass auto_download=True "
            f"to fetch it from {spec.repo} (sha {spec.sha256[:12]}…)."
        )

    console.print(f"[sdbench.title]Downloading[/] {spec.filename} from {spec.repo} …")
    downloader(spec.url, cached)
    _require_match(cached, spec)
    console.print(f"[sdbench.ok]Checkpoint verified[/] ({human_bytes(cached.stat().st_size)}).")
    return cached


def _require_match(path: Path, spec: CheckpointSpec) -> None:
    actual = sha256_file(path)
    if actual.lower() != spec.sha256.lower():
        raise ValueError(
            f"Checkpoint SHA mismatch for {path}.\n  expected {spec.sha256}\n  actual   {actual}\n"
            "Refusing to benchmark non-identical weights (R0.2)."
        )


def _content_length(response) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Content-Length")
    return int(value) if value else None


def _close(response) -> None:
    closer = getattr(response, "close", None)
    if callable(closer):
        closer()
