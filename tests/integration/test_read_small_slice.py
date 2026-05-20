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
    savemat(
        raw_dir / "CAST.mat",
        {
            "CAST": {
                "Depth": np.arange(20, dtype=np.float64),
                "Zc": np.arange(180 * 20, dtype=np.float32).reshape(180, 20, order="F"),
            }
        },
        do_compression=True,
    )
    savemat(
        raw_dir / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": np.arange(20, dtype=np.float64),
            "Inc": np.arange(20, dtype=np.float32),
            "RelBearing": np.arange(20, dtype=np.float32),
        },
        do_compression=True,
    )
    for receiver in range(1, 3):
        fields = {
            "Depth": np.arange(20, dtype=np.float64),
            "Tad": np.array([[10.0]], dtype=np.float32),
        }
        for side in "AB":
            fields[f"WaveRng{receiver:02d}Side{side}"] = np.arange(
                1024 * 20,
                dtype=np.int32,
            ).reshape(1024, 20, order="F")
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
                "  time_unit: unknown_to_verify",
                "  depth_unit: unknown_to_verify",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_read_small_slice_cli_outputs_json_and_npz(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir()
    interim_dir.mkdir()
    _write_raw(raw_dir)
    mapping = tmp_path / "raw_variable_mapping.yaml"
    paths_config = tmp_path / "paths.yaml"
    output_json = interim_dir / "small_slice_summary_v001.json"
    output_npz = interim_dir / "small_slice_v001.npz"
    _write_mapping(mapping)
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  raw: {raw_dir}",
                f"  interim: {interim_dir}",
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
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
            "--max-depth-samples",
            "3",
            "--max-time-samples",
            "4",
            "--max-receivers",
            "2",
            "--max-sides",
            "2",
            "--max-cast-azimuth",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Small slice" in result.stdout
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["variables"]["cast_zc"]["shape"] == [3, 5]
    assert summary["variables"]["xsi_waveform"]["shape"] == [3, 2, 2, 4]
    with np.load(output_npz) as data:
        assert data["cast_zc"].shape == (3, 5)
        assert data["xsi_waveform"].shape == (3, 2, 2, 4)
