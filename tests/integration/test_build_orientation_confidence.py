from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_orientation_confidence_cli_outputs_report(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    interim_dir.mkdir()
    reports_dir.mkdir()
    depth_only_npz = interim_dir / "depth_only_v001.npz"
    paths_config = tmp_path / "paths.yaml"
    np.savez_compressed(
        depth_only_npz,
        pose_depth=np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32),
        inc_deg=np.array([0.5, 2.0, 5.0, 8.0], dtype=np.float32),
        relbearing_deg=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
    )
    paths_config.write_text(
        "\n".join(["data:", f"  interim: {interim_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03g_build_orientation_confidence.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Orientation confidence errors=0" in result.stdout
    output_npz = interim_dir / "orientation_confidence_v001.npz"
    output_json = reports_dir / "orientation_confidence_report.json"
    output_md = reports_dir / "orientation_confidence_report.md"
    assert output_npz.exists()
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["relbearing_sign_dependency"] == "independent_of_plus_minus_convention"
    assert report["low_inclination_ratio"] == 0.25
    assert report["stable_inclination_ratio"] == 0.5
    with np.load(output_npz) as data:
        assert data["orientation_confidence"].shape == (4,)
        assert data["orientation_uncertain"].dtype == np.bool_
