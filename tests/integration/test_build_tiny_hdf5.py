from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def test_build_tiny_hdf5_cli_outputs_hdf5(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir()
    processed.mkdir()
    npz_path = interim / "small_slice_v001.npz"
    summary_path = interim / "small_slice_summary_v001.json"
    output_hdf5 = processed / "tiny_aligned_prototype_v001.h5"
    paths_config = tmp_path / "paths.yaml"
    np.savez_compressed(
        npz_path,
        xsi_waveform=np.ones((3, 2, 2, 4), dtype=np.float32),
        cast_zc=np.ones((3, 5), dtype=np.float32),
        cast_depth=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        pose_depth=np.array([1.0, 2.0, 3.0], dtype=np.float32),
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
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  interim: {interim}",
                f"  processed: {processed}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/01g_build_tiny_hdf5.py",
            "--paths",
            str(paths_config),
            "--small-slice-npz",
            str(npz_path),
            "--small-slice-summary",
            str(summary_path),
            "--output-hdf5",
            str(output_hdf5),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Tiny HDF5 prototype" in result.stdout
    with h5py.File(output_hdf5, "r") as h5:
        assert h5["/aligned/xsi_waveform"].shape == (3, 2, 2, 4)
        assert h5["/aligned/cast_zc"].shape == (3, 5)
        assert "/axis/time_sample_index" in h5
        assert "/metadata/source_files" in h5
