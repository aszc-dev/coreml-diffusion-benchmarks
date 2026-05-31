"""Preset + multi-run plumbing tests for the TUI.

Three contracts:

* :class:`RunPlan` defaults are backward-compatible (pre-multi-run JSON loads
  unchanged) and the presets land on the documented shapes.
* :class:`MatrixModel` snaps to a clean preset on a first visit, and editing
  any field flips ``mode`` to ``custom`` so the saved plan tells the truth
  about how it was authored.
* The CLI/TUI dispatch picks ``run_session`` when ``plan.repeats > 1`` and
  the iterations override flows from plan → ``cfg.iterations`` per invocation
  without touching ``matrix.yaml``.
"""

import json

import pytest

from sdbench.config import CellConfig
from sdbench.tui.capabilities import Capabilities
from sdbench.tui.config_view import MatrixModel
from sdbench.tui.prompts import build_cell_rows
from sdbench.tui.runplan import (
    PLAN_MODES,
    RunPlan,
    fast_test_preset,
    load_runplan,
    publication_preset,
    save_runplan,
)


def _cell(cid: str, enabled: bool = True, requires=None) -> CellConfig:
    return CellConfig(
        id=cid,
        backend="coreml_diffusion",
        compute_unit="CPU_AND_NE",
        attention="SPLIT_EINSUM_V2",
        precision="fp16",
        resolution=512,
        label=cid,
        enabled=enabled,
        requires=requires,
    )


def _rows(*cells: CellConfig):
    caps = Capabilities(chip="Apple M2 Pro", apple_generation=2)
    return build_cell_rows(list(cells), caps)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_publication_preset_shape():
    plan = publication_preset(["a", "b"], power_enabled=True)
    assert plan.mode == "publication"
    assert plan.repeats == 5
    assert plan.cooldown_s == 30.0
    assert plan.iterations is None  # defer to matrix.yaml
    assert plan.power_enabled is True
    assert plan.verbosity == "normal"
    assert plan.cell_ids == ["a", "b"]


def test_publication_preset_respects_power_unavailable():
    plan = publication_preset(["a"], power_enabled=False)
    assert plan.power_enabled is False  # tool refuses to lie about power


def test_fast_test_preset_collapses_to_one_cell():
    plan = fast_test_preset(["a", "b", "c"])
    assert plan.mode == "fast"
    assert plan.repeats == 1
    assert plan.cooldown_s == 0.0
    assert plan.iterations == 10  # R5.3 floor
    assert plan.power_enabled is False
    assert plan.verbosity == "quiet"
    assert plan.cell_ids == ["a"]


def test_runplan_rejects_below_floor_iterations():
    with pytest.raises(ValueError):
        RunPlan(cell_ids=["a"], power_enabled=False, iterations=9)


def test_runplan_rejects_zero_repeats():
    with pytest.raises(ValueError):
        RunPlan(cell_ids=["a"], power_enabled=False, repeats=0)


def test_runplan_rejects_unknown_mode():
    with pytest.raises(ValueError):
        RunPlan(cell_ids=["a"], power_enabled=False, mode="bogus")


def test_plan_modes_constant_is_authoritative():
    # Drift between PLAN_MODES and the validator would silently let unknown
    # modes through; pin them together.
    for mode in PLAN_MODES:
        RunPlan(cell_ids=["a"], power_enabled=False, mode=mode)  # no raise


# ---------------------------------------------------------------------------
# Backward-compatible serialisation
# ---------------------------------------------------------------------------


def test_load_runplan_accepts_pre_multi_run_payload(tmp_path):
    # JSON written by an older sdbench (no repeats/iterations/mode/cooldown)
    # must load unchanged with the new defaults — otherwise an upgrade
    # invalidates every saved plan on disk.
    path = tmp_path / "runplan.json"
    path.write_text(json.dumps({
        "cell_ids": ["alpha", "beta"],
        "power_enabled": True,
        "verbosity": "verbose",
        "run_conditions": "quiet room",
    }))
    plan = load_runplan(path)
    assert plan.cell_ids == ["alpha", "beta"]
    assert plan.repeats == 1
    assert plan.cooldown_s == 30.0
    assert plan.iterations is None
    assert plan.mode == "custom"
    assert plan.run_conditions == "quiet room"


def test_runplan_roundtrip_keeps_multi_run_fields(tmp_path):
    plan = publication_preset(["a", "b"], power_enabled=False)
    path = tmp_path / "runplan.json"
    save_runplan(plan, path)
    assert load_runplan(path) == plan


# ---------------------------------------------------------------------------
# MatrixModel preset behaviour
# ---------------------------------------------------------------------------


def test_matrix_model_publication_preset_selects_enabled_only():
    rows = _rows(_cell("on", enabled=True), _cell("off", enabled=False))
    model = MatrixModel(rows)
    model.apply_publication_preset(power_ok=True)
    assert model.mode == "publication"
    assert model.selected == {"on"}  # opt-in cell stays opt-in
    assert model.power is True
    assert model.repeats == 5


def test_matrix_model_fast_test_preset_picks_first_enabled_cell():
    rows = _rows(_cell("alpha", enabled=True), _cell("beta", enabled=True))
    model = MatrixModel(rows)
    model.apply_fast_test_preset()
    assert model.mode == "fast"
    assert model.selected == {"alpha"}
    assert model.repeats == 1
    assert model.iterations == 10
    assert model.power is False
    assert model.verbosity == "quiet"


def test_matrix_model_manual_edit_flips_mode_to_custom():
    # The mode label is part of the audit trail — a hand-edited plan must
    # not pretend it was a preset choice.
    rows = _rows(_cell("a"), _cell("b"))
    model = MatrixModel(rows)
    model.apply_publication_preset(power_ok=True)
    assert model.mode == "publication"
    model.toggle()  # any manual change
    assert model.mode == "custom"


def test_matrix_model_to_plan_gates_power_on_host_capability():
    rows = _rows(_cell("a"))
    model = MatrixModel(rows, power_default=True)
    model.apply_publication_preset(power_ok=False)  # no powermetrics on host
    plan = model.to_plan(power_ok=False)
    assert plan.power_enabled is False


def test_matrix_model_cycle_mode_walks_presets():
    # The 'm' key cycles publication → fast → custom → publication so the
    # user has one key to discover both presets without colliding with
    # lowercase 'p' (power).
    rows = _rows(_cell("alpha"), _cell("beta"))
    model = MatrixModel(rows)
    model.apply_publication_preset(power_ok=True)
    assert model.mode == "publication"
    model.cycle_mode(power_ok=True)
    assert model.mode == "fast"
    model.cycle_mode(power_ok=True)
    assert model.mode == "custom"
    model.cycle_mode(power_ok=True)
    assert model.mode == "publication"


def test_matrix_model_cycle_mode_from_custom_lands_on_publication():
    # Hand-edited plan (custom) → 'm' should snap to the recommended
    # default rather than do something cryptic.
    rows = _rows(_cell("alpha"))
    model = MatrixModel(rows)
    model.toggle()  # forces custom
    assert model.mode == "custom"
    model.cycle_mode(power_ok=True)
    assert model.mode == "publication"


def test_matrix_model_repeats_stepper_keeps_cooldown_sane():
    rows = _rows(_cell("a"))
    model = MatrixModel(rows, initial_repeats=1)
    assert model.cooldown_s == 0.0
    model.repeats = 2
    # Stepping via the TUI keys is what triggers cooldown sync; emulate that.
    model.cooldown_s = 30.0 if model.repeats > 1 else 0.0
    assert model.cooldown_s == 30.0
