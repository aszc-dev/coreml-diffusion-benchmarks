import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run.sh"


def test_run_wrapper_dry_run_builds_full_matrix_command(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--config",
            "config/test.yaml",
            "--shared-input",
            "input.npz",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "powermetrics --samplers cpu_power,gpu_power,ane_power -i 100" in result.stdout
    assert "caffeinate -dimsu uv run sdbench run --config config/test.yaml --shared-input input.npz" in result.stdout
    assert "run-cell" not in result.stdout


def test_run_wrapper_dry_run_builds_single_cell_command(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--no-power",
            "--backend",
            "diffusers_mps",
            "--compute-unit",
            "MPS",
            "--attention",
            "NATIVE",
            "--precision",
            "fp16",
            "--resolution",
            "512",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Power sampling disabled" in result.stdout
    assert "sdbench run-cell" in result.stdout
    assert "--backend diffusers_mps" in result.stdout
    assert "--compute-unit MPS" in result.stdout
    assert "--precision fp16" in result.stdout


def test_run_wrapper_dry_run_builds_single_cell_id_command(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--no-power",
            "--cell",
            "diffusers-mps-fp16",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "sdbench run-cell" in result.stdout
    assert "--cell diffusers-mps-fp16" in result.stdout
