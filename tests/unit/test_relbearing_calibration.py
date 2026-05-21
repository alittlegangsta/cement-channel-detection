from __future__ import annotations

import numpy as np

from cement_channel.alignment.relbearing_calibration import (
    calibrate_relbearing_convention,
    cast_azimuth_deg,
    generate_hypotheses,
    xsi_side_azimuth_deg,
)


def _synthetic_case(sign: str, *, depth_count: int = 8) -> tuple[np.ndarray, ...]:
    side_count = 8
    cast_count = 180
    receiver_count = 2
    time_count = 4
    depth = np.arange(depth_count, dtype=np.float32)
    relbearing = np.linspace(0.0, 105.0, depth_count, dtype=np.float32)
    orientation = np.ones(depth_count, dtype=np.float32)
    true_high_side_theta = 90.0
    cast_axis = np.linspace(0.0, 360.0, cast_count, endpoint=False, dtype=np.float32)
    side_axis = np.arange(side_count, dtype=np.float32) * 45.0
    if sign == "plus":
        raw_theta = np.mod(true_high_side_theta - relbearing, 360.0)
    else:
        raw_theta = np.mod(true_high_side_theta + relbearing, 360.0)

    cast_zc = np.full((depth_count, cast_count), 5.0, dtype=np.float32)
    waveform = np.ones((depth_count, receiver_count, side_count, time_count), dtype=np.float32)
    for index, theta in enumerate(raw_theta):
        cast_index = int(np.argmin(np.abs(np.mod(cast_axis - theta + 180.0, 360.0) - 180.0)))
        side_index = int(np.argmin(np.abs(np.mod(side_axis - theta + 180.0, 360.0) - 180.0)))
        cast_zc[index, cast_index] = 1.0
        waveform[index, :, side_index, :] = 10.0
    return depth, relbearing, orientation, cast_zc, waveform


def test_generate_hypotheses_full_grid() -> None:
    hypotheses = generate_hypotheses()

    assert len(hypotheses) == 64
    assert {hypothesis.relbearing_sign for hypothesis in hypotheses} == {"plus", "minus"}
    assert {hypothesis.cast_azimuth_direction for hypothesis in hypotheses} == {
        "normal",
        "reversed",
    }


def test_side_order_clockwise_and_counterclockwise() -> None:
    clockwise = xsi_side_azimuth_deg(8, side_order="clockwise", side_a_offset_deg=0.0)
    counterclockwise = xsi_side_azimuth_deg(
        8,
        side_order="counterclockwise",
        side_a_offset_deg=0.0,
    )

    assert np.allclose(clockwise[:3], np.array([0.0, 45.0, 90.0]))
    assert np.allclose(counterclockwise[:3], np.array([0.0, 315.0, 270.0]))


def test_cast_azimuth_normal_and_reversed() -> None:
    normal = cast_azimuth_deg(4, cast_azimuth_direction="normal")
    reversed_axis = cast_azimuth_deg(4, cast_azimuth_direction="reversed")

    assert np.allclose(normal, np.array([0.0, 90.0, 180.0, 270.0]))
    assert np.allclose(reversed_axis, np.array([0.0, 270.0, 180.0, 90.0]))


def test_clear_plus_case_recommends_plus() -> None:
    depth, relbearing, orientation, cast_zc, waveform = _synthetic_case("plus")

    report, _ = calibrate_relbearing_convention(
        depth=depth,
        relbearing_deg=relbearing,
        orientation_confidence=orientation,
        cast_zc=cast_zc,
        xsi_waveform=waveform,
    )

    assert report.final_recommendation == "data_supported_plus_recommendation"
    assert report.best_hypothesis is not None
    assert report.best_hypothesis["hypothesis"]["relbearing_sign"] == "plus"
    assert report.single_sign_alignment_approved is False


def test_clear_minus_case_recommends_minus() -> None:
    depth, relbearing, orientation, cast_zc, waveform = _synthetic_case("minus")

    report, _ = calibrate_relbearing_convention(
        depth=depth,
        relbearing_deg=relbearing,
        orientation_confidence=orientation,
        cast_zc=cast_zc,
        xsi_waveform=waveform,
    )

    assert report.final_recommendation == "data_supported_minus_recommendation"
    assert report.best_hypothesis is not None
    assert report.best_hypothesis["hypothesis"]["relbearing_sign"] == "minus"
    assert report.single_sign_alignment_approved is False


def test_insufficient_evidence_stays_unresolved() -> None:
    depth_count = 8
    depth = np.arange(depth_count, dtype=np.float32)
    relbearing = np.linspace(0.0, 105.0, depth_count, dtype=np.float32)
    orientation = np.ones(depth_count, dtype=np.float32)
    cast_zc = np.ones((depth_count, 180), dtype=np.float32)
    waveform = np.ones((depth_count, 2, 8, 4), dtype=np.float32)

    report, _ = calibrate_relbearing_convention(
        depth=depth,
        relbearing_deg=relbearing,
        orientation_confidence=orientation,
        cast_zc=cast_zc,
        xsi_waveform=waveform,
    )

    assert report.final_recommendation == "unresolved_keep_plus_primary_minus_ablation"
    assert report.valid_window_count == 0
