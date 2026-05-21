from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    interim = tmp_path / "interim"
    reports = tmp_path / "reports"
    interim.mkdir()
    reports.mkdir()
    depth_count = 8
    side_count = 8
    cast_count = 180
    depth = np.arange(depth_count, dtype=np.float32)
    relbearing = np.linspace(0.0, 105.0, depth_count, dtype=np.float32)
    orientation = np.ones(depth_count, dtype=np.float32)
    true_theta = 90.0
    raw_theta = np.mod(true_theta - relbearing, 360.0)
    cast_axis = np.linspace(0.0, 360.0, cast_count, endpoint=False, dtype=np.float32)
    side_axis = np.arange(side_count, dtype=np.float32) * 45.0
    cast_zc = np.full((depth_count, cast_count), 5.0, dtype=np.float32)
    waveform = np.ones((depth_count, 2, side_count, 4), dtype=np.float32)
    for index, theta in enumerate(raw_theta):
        cast_index = int(np.argmin(np.abs(np.mod(cast_axis - theta + 180.0, 360.0) - 180.0)))
        side_index = int(np.argmin(np.abs(np.mod(side_axis - theta + 180.0, 360.0) - 180.0)))
        cast_zc[index, cast_index] = 1.0
        waveform[index, :, side_index, :] = 10.0

    paths = {
        "config": tmp_path / "paths.yaml",
        "preview": interim / "depth_resample_overlap_preview_v001.npz",
        "orientation": interim / "orientation_confidence_v001.npz",
        "small_slice": interim / "small_slice_overlap_v001.npz",
        "validation": reports / "relbearing_sign_validation_overlap_report.json",
        "output_json": reports / "relbearing_calibration_report.json",
        "output_md": reports / "relbearing_calibration_report.md",
        "review_dir": reports / "relbearing_manual_review",
        "output_config": tmp_path / "alignment.relbearing_calibration.example.yaml",
    }
    paths["config"].write_text(
        "\n".join(["data:", f"  interim: {interim}", f"  reports: {reports}", ""]),
        encoding="utf-8",
    )
    np.savez_compressed(
        paths["preview"],
        canonical_depth=depth,
        relbearing_deg_on_grid=relbearing,
        small_slice_preview_depth=depth,
        small_slice_cast_zc_on_preview=cast_zc,
        small_slice_xsi_waveform_on_preview=waveform,
    )
    np.savez_compressed(
        paths["orientation"],
        pose_depth=depth,
        orientation_confidence=orientation,
    )
    np.savez_compressed(paths["small_slice"], cast_azimuth_deg=cast_axis)
    paths["validation"].write_text(
        json.dumps({"decision": "insufficient_evidence", "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    return paths


def test_calibrate_relbearing_convention_cli_outputs_report_and_figures(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03i_calibrate_relbearing_convention.py",
            "--paths",
            str(paths["config"]),
            "--output-config",
            str(paths["output_config"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RelBearing calibration decision=data_supported_plus_recommendation" in result.stdout
    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["final_recommendation"] == "data_supported_plus_recommendation"
    assert report["single_sign_alignment_approved"] is False
    assert report["valid_window_count"] >= 5
    assert (paths["review_dir"] / "hypothesis_score_summary.png").exists()
    assert (paths["review_dir"] / "review_summary_template.md").exists()
    assert "production_alignment_config: not_written" in paths["output_config"].read_text(
        encoding="utf-8"
    )
