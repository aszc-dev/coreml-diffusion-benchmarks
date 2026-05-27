import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert" / "team_ct9.py"


def test_team_ct9_conversion_script_dry_run_reports_unet_plan(tmp_path):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")

    result = subprocess.run(
        [
            "python",
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "out"),
            "--attention",
            "SPLIT_EINSUM_V2",
            "--compute-unit",
            "CPU_AND_NE",
            "--resolution",
            "512",
            "--precision",
            "w4",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    plan = json.loads(result.stdout)
    assert plan["backend"] == "coreml_diffusion"
    assert plan["coremltools_major"] == 9
    assert plan["checkpoint"] == str(checkpoint)
    assert plan["attention"] == "SPLIT_EINSUM_V2"
    assert plan["compute_unit"] == "CPU_AND_NE"
    assert plan["resolution"] == 512
    assert plan["precision"] == "w4"
    assert plan["quantize"] == "4"
    assert plan["batch_size"] == 2
