from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from cement_channel.data.small_slice_reader import (
    MatReadRequest,
    SmallSliceLimits,
    read_mat_file_slices,
    read_small_slice,
)


def _write_sample_raw(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    receiver_dir = raw_dir / "XSILMR"
    receiver_dir.mkdir(parents=True)
    cast_zc = np.arange(180 * 20, dtype=np.float32).reshape(180, 20, order="F")
    savemat(
        raw_dir / "CAST.mat",
        {"CAST": {"Depth": np.arange(20, dtype=np.float64), "Zc": cast_zc}},
        do_compression=True,
    )
    savemat(
        raw_dir / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": np.arange(20, dtype=np.float64),
            "Inc": np.linspace(10, 20, 20).astype(np.float32),
            "RelBearing": np.linspace(100, 120, 20).astype(np.float32),
        },
        do_compression=True,
    )
    for receiver in range(1, 3):
        fields = {
            "Depth": np.arange(20, dtype=np.float64),
            "Tad": np.array([[10.0]], dtype=np.float32),
        }
        for side in "AB":
            values = np.arange(1024 * 20, dtype=np.int32).reshape(1024, 20, order="F")
            fields[f"WaveRng{receiver:02d}Side{side}"] = values + receiver
        savemat(
            receiver_dir / f"XSILMR{receiver:02d}.mat",
            {f"XSILMR{receiver:02d}": fields},
            do_compression=True,
        )
    return raw_dir


def _mapping() -> dict:
    return {
        "mapping_version": "raw_variable_mapping_v001",
        "cast": {
            "file": "CAST.mat",
            "zc_variable": "CAST.Zc",
            "depth_variable": "CAST.Depth",
            "azimuth_mode": "implicit_uniform",
            "azimuth_start_deg": 0.0,
            "azimuth_step_deg": 2.0,
            "zc_source_shape_order": ["cast_azimuth", "depth"],
            "zc_canonical_shape_order": ["depth", "cast_azimuth"],
        },
        "pose": {
            "file": "D2_XSI_RelBearing_Inclination.mat",
            "depth_variable": "Depth_inc",
            "inclination_variable": "Inc",
            "relbearing_variable": "RelBearing",
            "source_shape_order": ["depth"],
        },
        "xsi": {
            "receiver_dir": "XSILMR",
            "expected_receiver_files": 2,
            "depth_variable_pattern": "XSILMR{receiver:02d}.Depth",
            "time_variable_pattern": "XSILMR{receiver:02d}.Tad",
            "waveform_variable_pattern": "XSILMR{receiver:02d}.WaveRng{receiver:02d}Side{side}",
            "side_labels": ["A", "B"],
            "waveform_source_shape_order": ["time", "depth"],
            "waveform_canonical_shape_order": ["depth", "time"],
            "depth_source_shape_order": ["depth"],
            "time_unit": "unknown_to_verify",
            "depth_unit": "unknown_to_verify",
        },
    }


def test_mat_struct_field_pattern_and_small_slice(tmp_path: Path) -> None:
    raw_dir = _write_sample_raw(tmp_path)
    requests = [
        MatReadRequest(
            variable_path="CAST.Zc",
            role="cast_zc",
            source_orientation=["cast_azimuth", "depth"],
            canonical_orientation=["depth", "cast_azimuth"],
            max_depth_samples=3,
            max_time_samples=4,
            max_cast_azimuth=5,
        )
    ]

    result = read_mat_file_slices(raw_dir / "CAST.mat", requests)

    assert result["CAST.Zc"].shape == (3, 5)
    assert result["CAST.Zc"][0, 0] == 0
    assert result["CAST.Zc"][1, 0] == 180


def test_source_to_canonical_transpose_for_xsi_waveform(tmp_path: Path) -> None:
    raw_dir = _write_sample_raw(tmp_path)
    request = MatReadRequest(
        variable_path="XSILMR01.WaveRng01SideA",
        role="xsi_waveform",
        source_orientation=["time", "depth"],
        canonical_orientation=["depth", "time"],
        max_depth_samples=3,
        max_time_samples=4,
        max_cast_azimuth=180,
    )

    result = read_mat_file_slices(raw_dir / "XSILMR" / "XSILMR01.mat", [request])

    assert result["XSILMR01.WaveRng01SideA"].shape == (3, 4)
    assert result["XSILMR01.WaveRng01SideA"][0].tolist() == [1, 2, 3, 4]
    assert result["XSILMR01.WaveRng01SideA"][1].tolist() == [1025, 1026, 1027, 1028]


def test_read_small_slice_builds_canonical_arrays(tmp_path: Path) -> None:
    raw_dir = _write_sample_raw(tmp_path)
    paths_config = {"data": {"raw": str(raw_dir)}}

    summary, arrays = read_small_slice(
        paths_config,
        _mapping(),
        mapping_path=tmp_path / "raw_variable_mapping.yaml",
        limits=SmallSliceLimits(
            max_depth_samples=3,
            max_time_samples=4,
            max_receivers=2,
            max_sides=2,
            max_cast_azimuth=5,
        ),
    )

    assert not summary.errors
    assert arrays["cast_zc"].shape == (3, 5)
    assert arrays["xsi_waveform"].shape == (3, 2, 2, 4)
    assert arrays["xsi_depth"].shape == (2, 3)
    assert arrays["xsi_tad"].shape == (2,)
    assert summary.variables["cast_zc"].canonical_orientation_suggestion == [
        "depth",
        "cast_azimuth",
    ]


def test_missing_path_returns_structured_error(tmp_path: Path) -> None:
    paths_config = {"data": {"raw": str(tmp_path / "missing")}}

    summary, arrays = read_small_slice(
        paths_config,
        _mapping(),
        mapping_path=tmp_path / "raw_variable_mapping.yaml",
    )

    assert arrays == {}
    assert any("Raw directory does not exist" in error for error in summary.errors)
