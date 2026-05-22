from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

pytest.importorskip("matplotlib")


def _write_synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    reports = tmp_path / "reports"
    receiver_dir = raw / "XSILMR"
    receiver_dir.mkdir(parents=True)
    interim.mkdir()
    reports.mkdir()
    depth_count = 16
    side_count = 8
    cast_count = 180
    depth = np.arange(100.0, 116.0, dtype=np.float32)
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

    savemat(
        raw / "CAST.mat",
        {"CAST": {"Depth": depth.astype(np.float64), "Zc": cast_zc.T}},
        do_compression=True,
    )
    savemat(
        raw / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": depth.astype(np.float64),
            "Inc": np.full(depth_count, 8.0, dtype=np.float32),
            "RelBearing": relbearing,
        },
        do_compression=True,
    )
    for receiver in range(1, 3):
        fields = {"Depth": depth.astype(np.float64), "Tad": np.array([[10.0]], dtype=np.float32)}
        for side_index, side in enumerate("ABCDEFGH"):
            fields[f"WaveRng{receiver:02d}Side{side}"] = waveform[
                :,
                receiver - 1,
                side_index,
                :,
            ].T
        savemat(
            receiver_dir / f"XSILMR{receiver:02d}.mat",
            {f"XSILMR{receiver:02d}": fields},
            do_compression=True,
        )

    paths = {
        "config": tmp_path / "paths.yaml",
        "mapping": tmp_path / "raw_variable_mapping.yaml",
        "depth_only": interim / "depth_only_v001.npz",
        "proposal": reports / "depth_grid_proposal.json",
        "preview": interim / "depth_resample_overlap_preview_v001.npz",
        "orientation": interim / "orientation_confidence_v001.npz",
        "small_slice": interim / "small_slice_overlap_v001.npz",
        "validation": reports / "relbearing_sign_validation_overlap_report.json",
        "output_json": reports / "relbearing_calibration_report.json",
        "output_md": reports / "relbearing_calibration_report.md",
        "candidate_windows": reports / "relbearing_candidate_windows.md",
        "review_dir": reports / "relbearing_manual_review",
        "output_config": tmp_path / "alignment.relbearing_calibration.example.yaml",
    }
    paths["config"].write_text(
        "\n".join(
            [
                "data:",
                f"  raw: {raw}",
                f"  interim: {interim}",
                f"  reports: {reports}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["mapping"].write_text(
        "\n".join(
            [
                "mapping_version: raw_variable_mapping_v001",
                "cast:",
                "  file: CAST.mat",
                "  zc_variable: CAST.Zc",
                "  depth_variable: CAST.Depth",
                "  azimuth_start_deg: 0.0",
                "  azimuth_step_deg: 2.0",
                "  zc_source_shape_order: [cast_azimuth, depth]",
                "  zc_canonical_shape_order: [depth, cast_azimuth]",
                "pose:",
                "  file: D2_XSI_RelBearing_Inclination.mat",
                "  depth_variable: Depth_inc",
                "  inclination_variable: Inc",
                "  relbearing_variable: RelBearing",
                "  source_shape_order: [depth]",
                "xsi:",
                "  receiver_dir: XSILMR",
                "  expected_receiver_files: 2",
                "  depth_variable_pattern: XSILMR{receiver:02d}.Depth",
                "  time_variable_pattern: XSILMR{receiver:02d}.Tad",
                "  waveform_variable_pattern: XSILMR{receiver:02d}.WaveRng{receiver:02d}Side{side}",
                "  side_labels: [A, B, C, D, E, F, G, H]",
                "  waveform_source_shape_order: [time, depth]",
                "  waveform_canonical_shape_order: [depth, time]",
                "  depth_source_shape_order: [depth]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        paths["depth_only"],
        cast_depth=depth,
        pose_depth=depth,
        xsi_depth_by_receiver=np.stack([depth, depth]),
        inc_deg=np.full(depth_count, 8.0, dtype=np.float32),
        relbearing_deg=relbearing,
    )
    paths["proposal"].write_text(
        json.dumps({"common_overlap_min": float(depth[0]), "common_overlap_max": float(depth[-1])}),
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
            "--mapping",
            str(paths["mapping"]),
            "--output-config",
            str(paths["output_config"]),
            "--depth-window-size",
            "2.0",
            "--max-depth-samples",
            "4",
            "--max-time-samples",
            "4",
            "--max-receivers",
            "2",
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
    assert report["fallback_window_counted_as_evidence"] is False
    assert paths["candidate_windows"].exists()
    assert (paths["review_dir"] / "hypothesis_score_summary.png").exists()
    assert (paths["review_dir"] / "review_summary_template.md").exists()
    assert "production_alignment_config: not_written" in paths["output_config"].read_text(
        encoding="utf-8"
    )
