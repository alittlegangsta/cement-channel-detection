from __future__ import annotations

import numpy as np

from cement_channel.qc.cast_qc import run_cast_zc_qc, run_pose_range_qc


def test_cast_zc_qc_basic_statistics() -> None:
    cast_zc = np.ones((3, 180), dtype=np.float32)
    cast_zc[0, 0] = np.inf

    result = run_cast_zc_qc(cast_zc)

    assert result.shape == [3, 180]
    assert result.inf_ratio and result.inf_ratio > 0
    assert any("non-finite" in warning for warning in result.warnings)


def test_cast_zc_qc_shape_warning_for_azimuth_count() -> None:
    result = run_cast_zc_qc(np.ones((3, 5), dtype=np.float32))

    assert any("azimuth count" in warning for warning in result.warnings)


def test_pose_range_qc_warns_out_of_range() -> None:
    result = run_pose_range_qc(
        np.array([10.0, 181.0], dtype=np.float32),
        np.array([0.0, 360.0], dtype=np.float32),
    )

    assert any("outside" in warning for warning in result["inc_deg"].warnings)
    assert any("outside" in warning for warning in result["rel_bearing_deg"].warnings)
