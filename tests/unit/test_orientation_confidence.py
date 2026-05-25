from __future__ import annotations

import numpy as np

from cement_channel.alignment.orientation_confidence import build_orientation_confidence


def test_build_orientation_confidence_linear_thresholds(tmp_path) -> None:
    depth_only_npz = tmp_path / "depth_only_v001.npz"
    np.savez_compressed(
        depth_only_npz,
        pose_depth=np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32),
        inc_deg=np.array([0.5, 1.0, 3.0, 5.0, np.nan], dtype=np.float32),
        relbearing_deg=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
    )

    report, arrays = build_orientation_confidence(depth_only_npz=depth_only_npz)

    assert np.allclose(
        arrays["orientation_confidence"],
        np.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32),
    )
    assert arrays["low_inc_mask"].tolist() == [True, True, False, False, True]
    assert arrays["stable_inc_mask"].tolist() == [False, False, False, True, False]
    assert report.low_inclination_ratio == 0.6
    assert report.stable_inclination_ratio == 0.2
    assert report.relbearing_sign_dependency == "independent_of_plus_minus_convention"
    assert report.errors == []


def test_build_orientation_confidence_reports_length_mismatch(tmp_path) -> None:
    depth_only_npz = tmp_path / "depth_only_v001.npz"
    np.savez_compressed(
        depth_only_npz,
        pose_depth=np.array([100.0, 101.0], dtype=np.float32),
        inc_deg=np.array([2.0, 3.0, 4.0], dtype=np.float32),
    )

    report, _ = build_orientation_confidence(depth_only_npz=depth_only_npz)

    assert report.errors
    assert "length mismatch" in report.errors[0]
