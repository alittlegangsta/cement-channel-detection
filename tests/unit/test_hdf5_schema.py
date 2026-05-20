from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from cement_channel.data.io_hdf5 import (
    build_tiny_hdf5_prototype,
    validate_tiny_hdf5_schema,
)


def _write_small_slice(tmp_path: Path) -> tuple[Path, Path]:
    npz_path = tmp_path / "small_slice_v001.npz"
    summary_path = tmp_path / "small_slice_summary_v001.json"
    np.savez_compressed(
        npz_path,
        xsi_waveform=np.ones((3, 2, 2, 4), dtype=np.float32),
        cast_zc=np.ones((3, 5), dtype=np.float32),
        cast_depth=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        pose_depth=np.array([1.1, 2.1, 3.1], dtype=np.float32),
        pose_inc_deg=np.array([10.0, 11.0, 12.0], dtype=np.float32),
        pose_rel_bearing_deg=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        cast_azimuth_deg=np.arange(5, dtype=np.float32),
        xsi_side_azimuth_deg=np.array([0.0, 180.0], dtype=np.float32),
        receiver_index=np.array([1, 2], dtype=np.int16),
        side_index=np.array([1, 2], dtype=np.int16),
        xsi_tad=np.array([10.0, 10.0], dtype=np.float32),
        xsi_depth=np.ones((2, 3), dtype=np.float32),
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "schema_v001",
                "data_version": "data_v001",
                "source_files": {"cast": "/tmp/CAST.mat"},
                "warnings": ["XSI time unit is unknown_to_verify."],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return npz_path, summary_path


def test_build_tiny_hdf5_prototype_writes_required_schema(tmp_path: Path) -> None:
    npz_path, summary_path = _write_small_slice(tmp_path)
    output_hdf5 = tmp_path / "tiny_aligned_prototype_v001.h5"

    result = build_tiny_hdf5_prototype(
        small_slice_npz=npz_path,
        small_slice_summary=summary_path,
        output_hdf5=output_hdf5,
    )

    assert not result.errors
    validation = validate_tiny_hdf5_schema(output_hdf5)
    assert validation.is_valid
    with h5py.File(output_hdf5, "r") as h5:
        assert h5["/aligned/xsi_waveform"].shape == (3, 2, 2, 4)
        assert h5["/aligned/cast_zc"].shape == (3, 5)
        assert "/axis/time_sample_index" in h5
        assert "/metadata/schema_version" in h5


def test_validate_tiny_hdf5_schema_reports_missing_dataset(tmp_path: Path) -> None:
    broken = tmp_path / "broken.h5"
    with h5py.File(broken, "w") as h5:
        h5.create_group("aligned")

    validation = validate_tiny_hdf5_schema(broken)

    assert not validation.is_valid
    assert any("Missing required dataset" in error for error in validation.errors)
