from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat


def _write_raw(raw_dir: Path) -> None:
    receiver_dir = raw_dir / "XSILMR"
    receiver_dir.mkdir(parents=True)
    depth = np.arange(100.0, 112.0, dtype=np.float64)
    zc = np.arange(180 * depth.size, dtype=np.float32).reshape(180, depth.size, order="F")
    savemat(
        raw_dir / "CAST.mat",
        {"CAST": {"Depth": depth, "Zc": zc}},
        do_compression=True,
    )
    savemat(
        raw_dir / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": depth.copy(),
            "Inc": np.linspace(2, 6, depth.size).astype(np.float32),
            "RelBearing": np.linspace(10, 40, depth.size).astype(np.float32),
        },
        do_compression=True,
    )
    for receiver in range(1, 3):
        fields = {"Depth": depth.copy(), "Tad": np.array([[10.0]], dtype=np.float32)}
        for side in "AB":
            fields[f"WaveRng{receiver:02d}Side{side}"] = np.arange(
                32 * depth.size,
                dtype=np.int32,
            ).reshape(32, depth.size, order="F")
        savemat(
            receiver_dir / f"XSILMR{receiver:02d}.mat",
            {f"XSILMR{receiver:02d}": fields},
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
                "  side_labels: [A, B]",
                "  waveform_source_shape_order: [time, depth]",
                "  waveform_canonical_shape_order: [depth, time]",
                "  depth_source_shape_order: [depth]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_overlap_targeted_small_slice_outputs_windowed_arrays(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir()
    interim_dir.mkdir()
    reports_dir.mkdir()
    _write_raw(raw_dir)
    mapping = tmp_path / "raw_variable_mapping.yaml"
    paths_config = tmp_path / "paths.yaml"
    depth_only = interim_dir / "depth_only_v001.npz"
    proposal = reports_dir / "depth_grid_proposal.json"
    _write_mapping(mapping)
    depth = np.arange(100.0, 112.0, dtype=np.float32)
    np.savez_compressed(
        depth_only,
        cast_depth=depth,
        pose_depth=depth,
        xsi_depth_by_receiver=np.stack([depth, depth]),
    )
    proposal.write_text(
        json.dumps({"common_overlap_min": 102.0, "common_overlap_max": 108.0}),
        encoding="utf-8",
    )
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/01f_read_small_slice.py",
            "--paths",
            str(paths_config),
            "--mapping",
            str(mapping),
            "--overlap-targeted",
            "--max-depth-samples",
            "4",
            "--max-time-samples",
            "5",
            "--max-receivers",
            "2",
            "--max-sides",
            "2",
            "--max-cast-azimuth",
            "6",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Small slice" in result.stdout
    summary = json.loads(
        (interim_dir / "small_slice_overlap_summary_v001.json").read_text(encoding="utf-8")
    )
    assert summary["depth_window"]["requested"]["depth_start"] == 104.0
    with np.load(interim_dir / "small_slice_overlap_v001.npz") as data:
        assert data["cast_zc"].shape == (3, 6) or data["cast_zc"].shape == (4, 6)
        assert data["xsi_waveform"].shape[1:] == (2, 2, 5)
        assert data["cast_depth"].min() >= 104.0
        assert data["cast_depth"].max() <= 106.0
