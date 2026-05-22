from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat


def _write_raw(raw_dir: Path) -> None:
    depth = np.arange(6, dtype=np.float64) + 100.0
    savemat(
        raw_dir / "CAST.mat",
        {
            "CAST": {
                "Depth": depth,
                "Zc": np.arange(180 * depth.size, dtype=np.float32).reshape(
                    180,
                    depth.size,
                    order="F",
                ),
            }
        },
        do_compression=True,
    )
    savemat(
        raw_dir / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": depth.copy(),
            "Inc": np.linspace(1.0, 6.0, depth.size).astype(np.float32),
            "RelBearing": np.linspace(10.0, 20.0, depth.size).astype(np.float32),
        },
        do_compression=True,
    )


def _write_mapping(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "mapping_version: raw_variable_mapping_v001",
                "cast:",
                "  file: CAST.mat",
                "  zc_variable: CAST.Zc",
                "  depth_variable: CAST.Depth",
                "  azimuth_start_deg: 0.0",
                "  azimuth_step_deg: 2.0",
                "  azimuth_count: 180",
                "pose:",
                "  file: D2_XSI_RelBearing_Inclination.mat",
                "  depth_variable: Depth_inc",
                "  inclination_variable: Inc",
                "  relbearing_variable: RelBearing",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_label_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "label_version: cast_weak_v001",
                "azimuth:",
                "  cast_azimuth_count: 180",
                "  cast_azimuth_step_deg: 2.0",
                "  cast_azimuth_direction: normal",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_cast_label_input_cli_outputs_npz_and_reports(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir()
    interim_dir.mkdir()
    reports_dir.mkdir()
    _write_raw(raw_dir)
    mapping = tmp_path / "raw_variable_mapping.yaml"
    label_config = tmp_path / "label.yaml"
    paths_config = tmp_path / "paths.yaml"
    orientation_npz = interim_dir / "orientation_confidence_v001.npz"
    _write_mapping(mapping)
    _write_label_config(label_config)
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  raw: {raw_dir}",
                f"  interim: {interim_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        orientation_npz,
        pose_depth=np.arange(6, dtype=np.float32) + 100.0,
        orientation_confidence=np.linspace(0.0, 1.0, 6).astype(np.float32),
        low_inc_mask=np.array([True, False, False, False, False, False]),
        orientation_uncertain=np.array([True, False, False, False, False, False]),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04a_prepare_cast_label_input.py",
            "--paths",
            str(paths_config),
            "--mapping",
            str(mapping),
            "--label-config",
            str(label_config),
            "--chunk-depth-samples",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CAST label input errors=0" in result.stdout
    output_npz = interim_dir / "cast_label_input_v001.npz"
    output_json = reports_dir / "cast_label_input_summary_v001.json"
    output_md = reports_dir / "cast_label_input_summary_v001.md"
    assert output_npz.exists()
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["chunking"]["chunk_depth_samples"] == 3
    assert "XSI waveform reading" in report["not_performed"]
    with np.load(output_npz) as data:
        assert data["cast_zc"].shape == (6, 180)
        assert data["cast_azimuth_deg"][-1] == 358.0
        assert data["relbearing_deg"].shape == (6,)
