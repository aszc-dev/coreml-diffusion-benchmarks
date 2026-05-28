from dataclasses import replace

from sdbench.provenance import (
    Fingerprint,
    Ledger,
    collect_fingerprint,
    invalidate_stale_results,
    is_stale,
    load_ledger,
    parse_lock_versions,
    record_run,
    save_ledger,
)
from sdbench.results import BenchmarkRecord, load_jsonl, upsert_jsonl, write_jsonl
from sdbench.tui.workspace import Workspace


def _fp(**overrides) -> Fingerprint:
    base = dict(
        checkpoint_sha256="abc",
        tool_version="0.1.0",
        chip="Apple M2 Pro",
        harness_deps={"torch": "2.6.0"},
        convert_ct8_deps={"coremltools": "8.3.0"},
        convert_ct9_deps={"coremltools": "9.0"},
    )
    base.update(overrides)
    return Fingerprint(**base)


# ----- fingerprint -----

def test_digest_is_stable_and_sensitive():
    a = _fp()
    assert a.digest == _fp().digest
    assert a.digest != _fp(checkpoint_sha256="def").digest
    assert a.digest != _fp(convert_ct9_deps={"coremltools": "9.1"}).digest


def test_parse_lock_versions_reads_real_locks(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "torch"\nversion = "2.6.0"\n\n'
        '[[package]]\nname = "numpy"\nversion = "2.1.3"\n',
        encoding="utf-8",
    )
    assert parse_lock_versions(lock) == {"torch": "2.6.0", "numpy": "2.1.3"}
    assert parse_lock_versions(tmp_path / "missing.lock") == {}


def test_collect_fingerprint_reads_three_locks(tmp_path):
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "torch"\nversion = "2.6.0"\n', encoding="utf-8")
    (tmp_path / "envs" / "apple-ct8").mkdir(parents=True)
    (tmp_path / "envs" / "apple-ct8" / "uv.lock").write_text('[[package]]\nname = "coremltools"\nversion = "8.3.0"\n', encoding="utf-8")
    (tmp_path / "envs" / "team-ct9").mkdir(parents=True)
    (tmp_path / "envs" / "team-ct9" / "uv.lock").write_text('[[package]]\nname = "coremltools"\nversion = "9.0"\n', encoding="utf-8")
    ws = Workspace.resolve(tmp_path)

    fp = collect_fingerprint(ws, checkpoint_sha256="abc", chip="Apple M2 Pro")
    assert fp.harness_deps == {"torch": "2.6.0"}
    assert fp.convert_ct8_deps == {"coremltools": "8.3.0"}
    assert fp.convert_ct9_deps == {"coremltools": "9.0"}


# ----- ledger / staleness -----

def test_ledger_roundtrip(tmp_path):
    ledger = Ledger(digest="d", fingerprint={"chip": "x"}, run_ids=["r1"], updated_s=1.0)
    save_ledger(ledger, tmp_path / "provenance.json")
    assert load_ledger(tmp_path / "provenance.json") == ledger
    assert load_ledger(tmp_path / "nope.json") is None


def test_is_stale():
    fp = _fp()
    assert is_stale(fp, None) is False
    assert is_stale(fp, Ledger(digest=fp.digest, fingerprint={}, run_ids=[], updated_s=0.0)) is False
    assert is_stale(fp, Ledger(digest="other", fingerprint={}, run_ids=[], updated_s=0.0)) is True


def test_record_run_appends_then_resets_on_change(tmp_path):
    ws = Workspace.resolve(tmp_path)
    fp = _fp()
    record_run(ws, fp, "r1")
    ledger = record_run(ws, fp, "r2")
    assert ledger.run_ids == ["r1", "r2"]

    changed = _fp(checkpoint_sha256="def")
    ledger2 = record_run(ws, changed, "r3")
    assert ledger2.run_ids == ["r3"]  # reset because the fingerprint changed


def test_invalidate_stale_results_clears_dependent_files(tmp_path):
    ws = Workspace.resolve(tmp_path)
    fp = _fp()
    record_run(ws, fp, "r1")  # ledger now matches fp

    (ws.results_data_dir / "results.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    ws.results_tables_dir.mkdir(parents=True, exist_ok=True)
    (ws.results_tables_dir / "latency.md").write_text("# t\n", encoding="utf-8")

    # same fingerprint -> nothing invalidated
    assert invalidate_stale_results(ws, fp) == []
    assert (ws.results_data_dir / "results.jsonl").exists()

    # changed fingerprint -> results + tables removed
    removed = invalidate_stale_results(ws, _fp(checkpoint_sha256="zzz"))
    assert any("results.jsonl" in r for r in removed)
    assert not (ws.results_data_dir / "results.jsonl").exists()
    assert not (ws.results_tables_dir / "latency.md").exists()


# ----- upsert -----

def _record(cell_id: str, latency: float, status: str = "ok") -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id="run", cell_id=cell_id, backend="mlx", requested_compute_unit="GPU",
        realized_compute_unit="GPU", attention="NATIVE", precision="fp16", resolution=512,
        status=status, latency_ms_median=latency, latency_ms_iqr=1.0,
        gpu_power_w=None, ane_power_w=None, energy_per_unet_step_j=None,
        estimated_energy_per_50_step_image_j=None, mse=None, cosine=None,
        numerically_divergent=None, on_disk_size_bytes=None, weight_only_size_bytes=None,
        effective_bits_per_parameter=None, compute_precision="fp16",
        graph_capture_s=None, convert_s=None, first_load_compile_s=None, failure_reason=None,
    )


def test_upsert_replaces_matching_cell_and_keeps_others(tmp_path):
    path = tmp_path / "results.jsonl"
    write_jsonl([_record("a", 10.0), _record("b", 20.0)], path)

    # re-run cell "a" only -> "a" updated, "b" preserved
    upsert_jsonl([_record("a", 99.0)], path)

    rows = {r.cell_id: r for r in load_jsonl(path)}
    assert rows["a"].latency_ms_median == 99.0
    assert rows["b"].latency_ms_median == 20.0
    assert len(rows) == 2


def test_upsert_appends_new_cell(tmp_path):
    path = tmp_path / "results.jsonl"
    write_jsonl([_record("a", 10.0)], path)
    upsert_jsonl([_record("c", 30.0)], path)
    assert {r.cell_id for r in load_jsonl(path)} == {"a", "c"}


def test_record_roundtrips_with_provenance_digest(tmp_path):
    path = tmp_path / "results.jsonl"
    rec = replace(_record("a", 10.0), provenance_digest="deadbeef")
    write_jsonl([rec], path)
    assert load_jsonl(path)[0].provenance_digest == "deadbeef"
