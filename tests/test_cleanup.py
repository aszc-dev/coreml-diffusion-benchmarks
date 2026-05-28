import json
from pathlib import Path

import pytest

from sdbench.tui.cleanup import delete_target, discover_targets, run_cleanup
from sdbench.tui.console import human_bytes
from sdbench.tui.workspace import Workspace


def _make_workspace(root: Path) -> Workspace:
    # artifacts: heavy compiled blob (dir) + tracked conversion JSON that must survive
    variant = root / "artifacts" / "apple_coreml" / "split_einsum_v2_ane"
    mlmodelc = variant / "model.mlmodelc"
    mlmodelc.mkdir(parents=True)
    (mlmodelc / "weight.bin").write_bytes(b"x" * 1000)
    (mlmodelc / "model.mil").write_bytes(b"y" * 500)
    (variant / "build-conversion.json").write_text(json.dumps({"convert_s": 1.0}))

    data = root / "results" / "data"
    data.mkdir(parents=True)
    (data / "results.jsonl").write_text('{"a": 1}\n')
    tables = root / "results" / "tables"
    tables.mkdir(parents=True)
    (tables / "latency.md").write_text("# latency\n")
    raw = root / "results" / "raw"
    raw.mkdir(parents=True)
    (raw / "run.plist").write_bytes(b"z" * 200)

    shared = root / "assets" / "shared_input"
    shared.mkdir(parents=True)
    (shared / "shared_input.npz").write_bytes(b"n" * 300)
    return Workspace.resolve(root)


def test_discover_targets_finds_all_present_categories(tmp_path):
    ws = _make_workspace(tmp_path)
    targets = {t.key: t for t in discover_targets(ws)}

    assert set(targets) == {"artifacts", "power-raw", "results", "shared-input"}
    # artifacts size excludes the tracked JSON: 1000 + 500 bytes only
    assert targets["artifacts"].size_bytes == 1500
    assert all("conversion.json" not in str(p) for t in targets.values() for p in t.paths)


def test_delete_artifacts_preserves_conversion_json(tmp_path):
    ws = _make_workspace(tmp_path)
    artifacts = next(t for t in discover_targets(ws) if t.key == "artifacts")

    freed = delete_target(artifacts)

    variant = ws.artifacts_dir / "apple_coreml" / "split_einsum_v2_ane"
    assert freed == 1500
    assert not (variant / "model.mlmodelc").exists()
    assert (variant / "build-conversion.json").exists()


def test_run_cleanup_only_and_assume_yes(tmp_path):
    ws = _make_workspace(tmp_path)

    freed = run_cleanup(ws, only=["shared-input", "power-raw"], assume_yes=True)

    assert freed == 500  # 300 (npz) + 200 (plist)
    assert not (ws.shared_input_dir / "shared_input.npz").exists()
    assert not (ws.results_raw_dir / "run.plist").exists()
    # unselected targets are untouched
    assert (ws.results_data_dir / "results.jsonl").exists()
    assert (ws.artifacts_dir / "apple_coreml" / "split_einsum_v2_ane" / "model.mlmodelc").exists()


def test_run_cleanup_unknown_only_key_deletes_nothing(tmp_path):
    ws = _make_workspace(tmp_path)
    assert run_cleanup(ws, only=["nope"], assume_yes=True) == 0
    assert (ws.shared_input_dir / "shared_input.npz").exists()


def test_run_cleanup_empty_workspace(tmp_path):
    ws = Workspace.resolve(tmp_path)
    assert run_cleanup(ws, assume_yes=True) == 0


def test_assert_within_blocks_external_path(tmp_path):
    from sdbench.tui.cleanup import CleanupTarget, _assert_within

    ws = Workspace.resolve(tmp_path / "ws")
    ws.root.mkdir()
    outside = (tmp_path / "elsewhere" / "checkpoint.safetensors").resolve()
    target = CleanupTarget(key="x", label="x", description="x", paths=(outside,), size_bytes=1)

    with pytest.raises(ValueError, match="outside workspace"):
        _assert_within(ws, list(target.paths))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0 B"), (1023, "1023 B"), (1024, "1.0 KB"), (1536, "1.5 KB"), (1024**3, "1.0 GB")],
)
def test_human_bytes(value, expected):
    assert human_bytes(value) == expected
