from __future__ import annotations

import numpy as np

from cement_channel.alignment.depth_resample import (
    canonical_depth_from_proposal,
    interpolate_1d,
    interpolate_angle_deg,
    prepare_depth_series,
)


def test_prepare_depth_series_sorts_and_deduplicates() -> None:
    depth, values = prepare_depth_series(
        np.array([3.0, 1.0, 2.0, 2.0]),
        np.array([30.0, 10.0, 20.0, 21.0]),
    )

    assert depth.tolist() == [1.0, 2.0, 3.0]
    assert values.tolist() == [10.0, 20.0, 30.0]


def test_interpolate_1d_disables_extrapolation_by_default() -> None:
    result = interpolate_1d(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 10.0, 20.0]),
        np.array([-1.0, 0.5, 3.0]),
    )

    assert np.isnan(result[0])
    assert result[1] == 5.0
    assert np.isnan(result[2])


def test_interpolate_angle_deg_handles_wrap() -> None:
    result = interpolate_angle_deg(
        np.array([0.0, 1.0]),
        np.array([350.0, 10.0]),
        np.array([0.5]),
    )

    assert result[0] < 1.0 or result[0] > 359.0


def test_canonical_depth_from_proposal() -> None:
    depth = canonical_depth_from_proposal(
        {"depth_start": 100.0, "depth_step": 0.5, "sample_count": 3}
    )

    assert depth.tolist() == [100.0, 100.5, 101.0]
