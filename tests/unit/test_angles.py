from __future__ import annotations

import numpy as np

from cement_channel.utils.angles import (
    circular_distance_deg,
    circular_mean_deg,
    signed_circular_delta_deg,
    wrap_deg,
)


def test_wrap_deg_scalar_and_array() -> None:
    assert wrap_deg(-1.0) == 359.0
    assert wrap_deg(360.0) == 0.0
    assert np.allclose(wrap_deg(np.array([-10.0, 370.0])), np.array([350.0, 10.0]))


def test_circular_distance_uses_short_path() -> None:
    assert circular_distance_deg(350.0, 10.0) == 20.0
    assert circular_distance_deg(90.0, 270.0) == 180.0


def test_signed_circular_delta() -> None:
    assert signed_circular_delta_deg(10.0, 350.0) == 20.0
    assert signed_circular_delta_deg(350.0, 10.0) == -20.0


def test_circular_mean_wraps_across_zero() -> None:
    result = circular_mean_deg(np.array([350.0, 10.0]))

    assert result < 1.0 or result > 359.0


def test_weighted_circular_mean() -> None:
    result = circular_mean_deg(np.array([0.0, 90.0]), weights=np.array([3.0, 1.0]))

    assert 0.0 < result < 45.0
