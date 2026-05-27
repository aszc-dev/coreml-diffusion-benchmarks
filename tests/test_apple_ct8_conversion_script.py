import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert" / "apple_ct8.py"


def test_apple_ct8_conversion_script_dry_run_reports_unet_plan(tmp_path):
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
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    plan = json.loads(result.stdout)
    assert plan["backend"] == "apple_coreml"
    assert plan["coremltools_major"] == 8
    assert plan["checkpoint"] == str(checkpoint)
    assert plan["attention"] == "SPLIT_EINSUM_V2"
    assert plan["compute_unit"] == "CPU_AND_NE"
    assert plan["resolution"] == 512
