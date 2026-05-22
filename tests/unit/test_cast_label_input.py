from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from cement_channel.labels.cast_label_input import prepare_cast_label_input


def _write_cast_pose(raw_dir: Path) -> None:
    depth = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
    zc = np.arange(180 * depth.size, dtype=np.float32).reshape(180, depth.size, order="F")
    savemat(
        raw_dir / "CAST.mat",
        {"CAST": {"Depth": depth, "Zc": zc}},
        do_compression=True,
    )
    savemat(
        raw_dir / "D2_XSI_RelBearing_Inclination.mat",
        {
            "Depth_inc": np.array([100.0, 102.0, 103.0], dtype=np.float64),
            "Inc": np.array([1.0, 5.0, 7.0], dtype=np.float32),
            "RelBearing": np.array([350.0, 10.0, 20.0], dtype=np.float32),
        },
        do_compression=True,
    )


def _mapping() -> dict:
    return {
        "cast": {
            "file": "CAST.mat",
            "depth_variable": "CAST.Depth",
            "zc_variable": "CAST.Zc",
            "azimuth_count": 180,
            "azimuth_step_deg": 2.0,
        },
        "pose": {
            "file": "D2_XSI_RelBearing_Inclination.mat",
            "depth_variable": "Depth_inc",
            "inclination_variable": "Inc",
            "relbearing_variable": "RelBearing",
        },
    }


def _label_config() -> dict:
    return {
        "azimuth": {
            "cast_azimuth_count": 180,
            "cast_azimuth_step_deg": 2.0,
            "cast_azimuth_direction": "normal",
        }
    }


def test_prepare_cast_label_input_reads_only_cast_pose_and_interpolates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cast_pose(raw_dir)
    orientation_npz = tmp_path / "orientation_confidence_v001.npz"
    np.savez_compressed(
        orientation_npz,
        pose_depth=np.array([100.0, 102.0, 103.0], dtype=np.float32),
        orientation_confidence=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        low_inc_mask=np.array([True, False, False]),
        orientation_uncertain=np.array([True, False, False]),
    )

    report, arrays = prepare_cast_label_input(
        {"data": {"raw": str(raw_dir)}},
        _mapping(),
        _label_config(),
        paths_config_path=tmp_path / "paths.yaml",
        mapping_path=tmp_path / "mapping.yaml",
        label_config_path=tmp_path / "label.yaml",
        orientation_confidence_npz=orientation_npz,
        chunk_depth_samples=2,
    )

    assert report.errors == []
    assert arrays["cast_zc"].shape == (4, 180)
    assert arrays["cast_zc"][1, 0] == 180.0
    assert arrays["cast_azimuth_deg"][0] == 0.0
    assert arrays["cast_azimuth_deg"][-1] == 358.0
    assert np.isclose(arrays["inc_deg"][1], 3.0)
    assert arrays["orientation_confidence"][1] == 0.5
    assert report.chunking["chunk_depth_samples"] == 2
    assert "XSI waveform reading" in report.not_performed


def test_prepare_cast_label_input_uses_circular_relbearing_interpolation(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cast_pose(raw_dir)
    orientation_npz = tmp_path / "orientation_confidence_v001.npz"
    np.savez_compressed(
        orientation_npz,
        pose_depth=np.array([100.0, 103.0], dtype=np.float32),
        orientation_confidence=np.ones(2, dtype=np.float32),
        low_inc_mask=np.zeros(2, dtype=bool),
        orientation_uncertain=np.zeros(2, dtype=bool),
    )

    _report, arrays = prepare_cast_label_input(
        {"data": {"raw": str(raw_dir)}},
        _mapping(),
        _label_config(),
        paths_config_path=tmp_path / "paths.yaml",
        mapping_path=tmp_path / "mapping.yaml",
        label_config_path=tmp_path / "label.yaml",
        orientation_confidence_npz=orientation_npz,
        chunk_depth_samples=4,
    )

    assert arrays["relbearing_deg"][1] < 1.0 or arrays["relbearing_deg"][1] > 359.0
