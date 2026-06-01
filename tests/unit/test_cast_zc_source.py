from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cement_channel.labels.cast_zc_source import (
    CastZcSourceError,
    build_cast_zc_source_from_arrays,
    load_controlled_cast_zc_source,
)


def _valid_arrays() -> dict[str, np.ndarray]:
    depth = np.asarray([100.0, 101.0, 102.0], dtype=np.float32)
    azimuth = np.arange(4, dtype=np.float32) * 90.0
    zc = np.asarray(
        [
            [3.0, 2.4, 4.0, 5.0],
            [3.5, 2.0, 4.5, 5.5],
            [4.0, 2.8, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    return {"cast_depth": depth, "cast_azimuth_deg": azimuth, "cast_zc": zc}


def test_build_cast_zc_source_validates_raw_depth_azimuth_shape() -> None:
    source = build_cast_zc_source_from_arrays(
        _valid_arrays(),
        source_file=Path("cast_label_input_v001.npz"),
        min_finite_ratio=0.9,
    )

    assert source.cast_zc.shape == (3, 4)
    assert source.report.source_field == "cast_zc"
    assert source.report.depth_field == "cast_depth"
    assert source.report.azimuth_field == "cast_azimuth_deg"
    assert source.report.finite_ratio == 1.0
    assert source.report.no_final_labels is True
    assert "azimuth count is not 180" in source.report.warnings[0]


def test_load_controlled_cast_zc_source_prefers_existing_raw_npz(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    source_path = tmp_path / "cast_label_input_v001.npz"
    np.savez_compressed(source_path, **_valid_arrays())

    source = load_controlled_cast_zc_source([missing, source_path], min_finite_ratio=0.9)

    assert source.report.source_file == str(source_path)
    np.testing.assert_allclose(source.cast_depth, [100.0, 101.0, 102.0])


def test_cast_zc_source_refuses_zc_ratio_as_raw_zc() -> None:
    arrays = _valid_arrays()
    arrays.pop("cast_zc")
    arrays["zc_ratio"] = np.ones((3, 4), dtype=np.float32)

    with pytest.raises(CastZcSourceError, match="derived CAST fields"):
        build_cast_zc_source_from_arrays(arrays, source_file=Path("baseline.npz"))


def test_cast_zc_source_refuses_mismatched_depth_grid() -> None:
    arrays = _valid_arrays()
    arrays["cast_depth"] = np.asarray([100.0, 101.0], dtype=np.float32)

    with pytest.raises(CastZcSourceError, match="depth grid length"):
        build_cast_zc_source_from_arrays(arrays, source_file=Path("bad.npz"))
