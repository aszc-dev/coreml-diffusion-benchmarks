import io

from rich.console import Console

from sdbench.config import CellConfig
from sdbench.tui.console import THEME
from sdbench.tui.app import Menu, _menu_frame
from sdbench.tui.capabilities import Capabilities
from sdbench.tui.config_view import MatrixModel, _render as render_config
from sdbench.tui.prompts import build_cell_rows
from sdbench.tui.workspace import Workspace


def _render_to_str(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, width=120, height=40, color_system=None, theme=THEME).print(renderable)
    return buf.getvalue()


def _cell(cid, enabled=True, requires=None, backend="coreml_diffusion", precision="fp16"):
    return CellConfig(id=cid, backend=backend, compute_unit="CPU_AND_NE", attention="SPLIT_EINSUM_V2",
                      precision=precision, resolution=512, label=cid, enabled=enabled, requires=requires)


# ----- imports / cycles -----

def test_tui_modules_import_without_cycle():
    import sdbench.tui.app  # noqa: F401
    import sdbench.tui.config_view  # noqa: F401
    import sdbench.tui.dashboard  # noqa: F401
    import sdbench.tui.screen  # noqa: F401


# ----- menu model -----

def test_menu_wraps():
    menu = Menu([("a", "A"), ("b", "B"), ("c", "C")])
    assert menu.key == "a"
    menu.move(-1)
    assert menu.key == "c"
    menu.move(1)
    assert menu.key == "a"


# ----- matrix model -----

def _rows():
    caps = Capabilities(chip="Apple M2 Pro", apple_generation=2)
    cells = [
        _cell("on", enabled=True),
        _cell("off", enabled=False),
        _cell("gated", enabled=True, requires={"ane_activation_quant": True}),
    ]
    return build_cell_rows(cells, caps)


def test_matrix_toggle_and_locked():
    model = MatrixModel(_rows())
    assert set(model.chosen_ids()) == {"on"}  # default-selected
    model.index = 1  # "off"
    model.toggle()
    assert set(model.chosen_ids()) == {"on", "off"}
    model.index = 2  # "gated" — locked, toggle is a no-op
    model.toggle()
    assert "gated" not in model.chosen_ids()


def test_matrix_select_all_excludes_locked_and_clear():
    model = MatrixModel(_rows())
    model.select_all()
    assert set(model.chosen_ids()) == {"on", "off"}
    model.clear()
    assert model.chosen_ids() == []


def test_matrix_cycle_verbosity():
    model = MatrixModel(_rows())
    assert model.verbosity == "normal"
    model.cycle_verbosity()
    assert model.verbosity == "verbose"
    model.cycle_verbosity()
    assert model.verbosity == "quiet"


# ----- read_key mapping -----

def test_workspace_bytes_sums_only_our_files(tmp_path):
    from sdbench.tui import screen as scr

    ws = Workspace.resolve(tmp_path)
    ws.artifacts_dir.mkdir(parents=True)
    (ws.artifacts_dir / "blob.bin").write_bytes(b"x" * 1000)
    ws.results_data_dir.mkdir(parents=True)
    (ws.results_data_dir / "r.jsonl").write_text("y" * 500)
    (tmp_path / "unrelated.txt").write_text("z" * 9999)  # outside our dirs -> not counted

    scr.invalidate_usage(ws)
    assert scr.workspace_bytes(ws) == 1500
    assert "sdbench" in scr.usage_text(ws) and "used" in scr.usage_text(ws)


def test_read_key_maps_enter(monkeypatch):
    import sdbench.tui.screen as screen

    monkeypatch.setattr(screen.readchar, "readkey", lambda: "\r")
    assert screen.read_key() == screen.ENTER
    monkeypatch.setattr(screen.readchar, "readkey", lambda: "Q")
    assert screen.read_key() == "q"


# ----- smoke renders (must not raise, must contain expected chrome) -----

def test_menu_frame_renders(tmp_path):
    from sdbench.config import load_benchmark_config

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        f"checkpoint: {tmp_path}/x.safetensors\nseed: 0\nwarmup: 1\niterations: 10\nresolution_default: 512\n"
        "equivalence: {mse_max: 1.0e-3, cosine_min: 0.999}\npower: {interval_ms: 100, baseline_seconds: 2}\n"
        "thermal: {abort_on_throttle: false}\n"
        "cells:\n  - {id: c1, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512}\n",
        encoding="utf-8",
    )
    ws = Workspace.resolve(tmp_path)
    cfg = load_benchmark_config(matrix)
    out = _render_to_str(_menu_frame(ws, cfg, Menu([("convert", "Convert artifacts"), ("quit", "Quit")])))
    assert "Convert artifacts" in out and "disk" in out and "used" in out


def test_config_view_renders(tmp_path):
    ws = Workspace.resolve(tmp_path)
    (ws.root / "config").mkdir(parents=True, exist_ok=True)
    (ws.matrix_path).write_text(
        f"checkpoint: {tmp_path}/x.safetensors\nseed: 0\nwarmup: 1\niterations: 10\nresolution_default: 512\n"
        "equivalence: {mse_max: 1.0e-3, cosine_min: 0.999}\npower: {interval_ms: 100, baseline_seconds: 2}\n"
        "thermal: {abort_on_throttle: false}\n"
        "cells:\n  - {id: c1, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512}\n",
        encoding="utf-8",
    )
    from sdbench.config import load_benchmark_config

    model = MatrixModel(_rows())
    out = _render_to_str(render_config(ws, model, True, "", load_benchmark_config(ws.matrix_path)))
    assert "Matrix" in out and "verbosity" in out


def test_dashboard_renders(tmp_path):
    from types import SimpleNamespace

    from sdbench.tui.dashboard import DashboardReporter

    class FakeLive:
        def update(self, *_): ...
        def refresh(self): ...

    ws = Workspace.resolve(tmp_path)
    dash = DashboardReporter(FakeLive(), ws, cell_ids=["c1", "c2"])
    dash.run_start(2)
    dash.cell_start("c1", 0, 2)
    dash.log("hello")
    out = _render_to_str(dash.render())
    # whole cell list visible, not just the current one
    assert "Cells" in out and "Logs" in out and "c1" in out and "c2" in out

    rec = SimpleNamespace(cell_id="c1", status="ok", latency_ms_median=399.6, gpu_power_w=4.0,
                          ane_power_w=None, cosine=0.997, mse=1e-3, numerically_divergent=True)
    dash.show_summary([rec])
    assert "Run summary" in _render_to_str(dash.render())


def test_convert_dashboard_lists_all_builds(tmp_path):
    from types import SimpleNamespace

    from sdbench.tui.dashboard import ConvertDashboard

    class FakeLive:
        def update(self, *_): ...
        def refresh(self): ...

    def _build(name):
        d = tmp_path / name
        return SimpleNamespace(backend="apple_coreml", output_dir=d, expected_artifact=d / "m.mlmodelc")

    builds = [_build("split_einsum_v2_ane"), _build("original_gpu")]
    ws = Workspace.resolve(tmp_path)
    dash = ConvertDashboard(FakeLive(), ws, builds)
    dash.on_build(builds[0], 0, 2)
    dash.on_line("Converting UNet …")
    dash.on_done(builds[0])
    out = _render_to_str(dash.render())
    # both builds listed even though only one ran; statuses + streamed log present
    assert "Builds" in out and "Toolchain logs" in out and "Converting UNet" in out
    assert "split_einsum_v2_ane" in out and "original_gpu" in out
    assert "done" in out and "pending" in out
