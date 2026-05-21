from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_cast_baseline_cli_outputs_npz_and_reports(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    interim_dir.mkdir()
    reports_dir.mkdir()
    zc = np.full((10, 6), 10.0, dtype=np.float32)
    zc[4:7, 2] = 4.0
    np.savez_compressed(
        interim_dir / "cast_label_input_v001.npz",
        cast_depth=np.arange(10, dtype=np.float32),
        cast_azimuth_deg=np.arange(6, dtype=np.float32) * 2.0,
        cast_zc=zc,
    )
    paths_config = tmp_path / "paths.yaml"
    label_config = tmp_path / "label.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  interim: {interim_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )
    label_config.write_text(
        "\n".join(
            [
                "baseline:",
                "  method: rolling_quantile",
                "  window_m: 5.0",
                "  quantile: 0.90",
                "  min_finite_fraction: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04b_build_cast_baseline.py",
            "--paths",
            str(paths_config),
            "--label-config",
            str(label_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CAST baseline errors=0" in result.stdout
    output_npz = interim_dir / "cast_zc_baseline_v001.npz"
    output_json = reports_dir / "cast_zc_baseline_report_v001.json"
    output_md = reports_dir / "cast_zc_baseline_report_v001.md"
    assert output_npz.exists()
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["method"] == "rolling_quantile"
    assert report["window_samples"] == 5
    with np.load(output_npz) as data:
        assert data["zc_base"].shape == (10, 6)
        assert data["relative_drop"][5, 2] > 0.5
