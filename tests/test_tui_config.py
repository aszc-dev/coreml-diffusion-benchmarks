import pytest

from sdbench.config import CellConfig
from sdbench.thermal import check_thermal_state
from sdbench.tui.capabilities import Capabilities, detect_capabilities, evaluate_gate
from sdbench.tui.preflight import disk_report, load_footprint
from sdbench.tui.prompts import build_cell_rows, default_ids, full_suite_ids
from sdbench.tui.runplan import RunPlan, load_runplan, save_runplan


# ----- capabilities / gates -----

def test_detect_capabilities_parses_apple_generation():
    assert detect_capabilities("Apple M2 Pro").apple_generation == 2
    assert detect_capabilities("Apple M4 Max").apple_generation == 4
    assert detect_capabilities("Intel(R) Core(TM) i7").apple_generation is None


def test_ane_activation_quant_gate_by_generation():
    m2 = detect_capabilities("Apple M2 Pro")
    m4 = detect_capabilities("Apple M4")
    assert m2.supports("ane_activation_quant") is False
    assert m4.supports("ane_activation_quant") is True


def test_evaluate_gate_states():
    m2 = Capabilities(chip="Apple M2 Pro", apple_generation=2)
    assert evaluate_gate(m2, None).state == "ok"
    assert evaluate_gate(m2, {"ane_activation_quant": True}).state == "unmet"
    unknown = Capabilities(chip="mystery", apple_generation=None)
    assert evaluate_gate(unknown, {"ane_activation_quant": True}).state == "unknown"


# ----- thermal probe -----

def test_thermal_throttled_when_speed_limited():
    state = check_thermal_state(probe=lambda: "CPU_Speed_Limit \t= 70\n")
    assert state.throttled is True
    assert "70" in state.detail


def test_thermal_ok_at_full_speed():
    state = check_thermal_state(probe=lambda: "CPU_Speed_Limit = 100\n")
    assert state.throttled is False


def test_thermal_unavailable_when_probe_raises():
    def boom() -> str:
        raise FileNotFoundError("pmset missing")

    state = check_thermal_state(probe=boom)
    assert state.throttled is False
    assert state.source == "unavailable"


# ----- cell rows / selection -----

def _cell(cid: str, enabled: bool, requires=None) -> CellConfig:
    return CellConfig(
        id=cid, backend="coreml_diffusion", compute_unit="CPU_AND_NE",
        attention="SPLIT_EINSUM_V2", precision="fp16", resolution=512,
        label=cid, enabled=enabled, requires=requires,
    )


def test_build_cell_rows_locks_unmet_gates_and_defaults():
    caps = Capabilities(chip="Apple M2 Pro", apple_generation=2)
    cells = [
        _cell("on", True),
        _cell("off", False),
        _cell("gated", True, requires={"ane_activation_quant": True}),
    ]
    rows = {r.cell.id: r for r in build_cell_rows(cells, caps)}
    assert rows["on"].default_selected is True
    assert rows["off"].default_selected is False and rows["off"].selectable is True
    assert rows["gated"].selectable is False  # locked: M2 lacks ANE activation quant
    assert default_ids(list(rows.values())) == ["on"]
    assert set(full_suite_ids(list(rows.values()))) == {"on", "off"}  # gated excluded


# ----- disk report -----

def test_disk_report_known_and_unknown():
    report = disk_report(free_bytes=10_000, footprint={"a": 3000, "b": 4000}, cell_ids=["a", "b", "c"])
    assert report.known_required_bytes == 7000
    assert report.unknown_cells == ["c"]
    assert report.fully_known is False
    assert report.fits is None  # unknown cell -> can't decide


def test_disk_report_fits_when_fully_known():
    assert disk_report(10_000, {"a": 3000}, ["a"]).fits is True
    assert disk_report(1_000, {"a": 3000}, ["a"]).fits is False


def test_load_footprint_missing_returns_empty(tmp_path):
    assert load_footprint(tmp_path / "nope.yaml") == {}


def test_load_footprint_reads_cells(tmp_path):
    fp = tmp_path / "disk_footprint.yaml"
    fp.write_text("cells:\n  apple-ane-fp16: 1722195455\n", encoding="utf-8")
    assert load_footprint(fp) == {"apple-ane-fp16": 1722195455}


# ----- run plan -----

def test_runplan_roundtrip(tmp_path):
    plan = RunPlan(cell_ids=["a", "b"], power_enabled=True, verbosity="verbose", run_conditions="quiet room")
    path = tmp_path / ".sdbench" / "runplan.json"
    save_runplan(plan, path)
    assert load_runplan(path) == plan


def test_runplan_rejects_bad_verbosity():
    with pytest.raises(ValueError, match="verbosity"):
        RunPlan(cell_ids=["a"], power_enabled=False, verbosity="loud")
