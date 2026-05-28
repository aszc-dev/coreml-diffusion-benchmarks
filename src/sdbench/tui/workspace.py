"""Canonical filesystem layout for a benchmark workspace.

One source of truth for *where things live*, shared by every interactive
command so paths are never hard-coded twice. Mirrors the layout documented in
AGENTS.md (artifacts/, results/{data,tables,raw}, assets/shared_input/).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def resolve(cls, root: Path | str | None = None) -> "Workspace":
        base = Path(root) if root else Path.cwd()
        return cls(root=base.expanduser().resolve())

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def results_data_dir(self) -> Path:
        return self.results_dir / "data"

    @property
    def results_tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def results_raw_dir(self) -> Path:
        return self.results_dir / "raw"

    @property
    def shared_input_dir(self) -> Path:
        return self.root / "assets" / "shared_input"

    @property
    def cache_dir(self) -> Path:
        # Downloaded checkpoints land here (Phase 5); declared now so cleanup can
        # reclaim it once it exists.
        return self.root / ".cache" / "sdbench"

    @property
    def state_dir(self) -> Path:
        # Small, durable interactive state (run plan); survives a `cleanup`.
        return self.root / ".sdbench"

    @property
    def runplan_path(self) -> Path:
        return self.state_dir / "runplan.json"

    @property
    def disk_footprint_path(self) -> Path:
        # Committed, measured artifact sizes (populated by `measure-disk`, Phase 5).
        return self.root / "config" / "disk_footprint.yaml"

    @property
    def matrix_path(self) -> Path:
        return self.root / "config" / "matrix.yaml"

    @property
    def provenance_path(self) -> Path:
        return self.results_data_dir / "provenance.json"
