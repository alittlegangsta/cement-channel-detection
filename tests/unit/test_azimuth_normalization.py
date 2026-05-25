from __future__ import annotations

import numpy as np

from cement_channel.alignment.azimuth_normalization import (
    align_azimuth_to_high_side,
    aligned_azimuth_candidates,
    default_cast_azimuth_deg,
    default_xsi_side_azimuth_deg,
    orientation_confidence_from_inclination,
    orientation_uncertain_mask,
)


def test_align_azimuth_plus_minus_candidates() -> None:
    assert align_azimuth_to_high_side(350.0, 20.0, convention="plus") == 10.0
    assert align_azimuth_to_high_side(10.0, 20.0, convention="minus") == 350.0
    assert align_azimuth_to_high_side(370.0, 20.0, convention="no_rotation") == 10.0

    candidates = aligned_azimuth_candidates(np.array([0.0, 90.0]), 30.0)
    assert np.allclose(candidates["plus"], np.array([30.0, 120.0]))
    assert np.allclose(candidates["minus"], np.array([330.0, 60.0]))


def test_orientation_confidence_linear_thresholds() -> None:
    confidence = orientation_confidence_from_inclination(np.array([0.0, 1.0, 3.0, 5.0, 6.0]))

    assert np.allclose(confidence, np.array([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert orientation_confidence_from_inclination(3.0) == 0.5


def test_orientation_uncertain_mask() -> None:
    mask = orientation_uncertain_mask(np.array([0.5, 1.0, 1.1, np.nan]))

    assert mask.tolist() == [True, True, False, True]


def test_default_azimuth_axes() -> None:
    assert np.allclose(default_xsi_side_azimuth_deg(), np.arange(8) * 45.0)
    assert default_cast_azimuth_deg().shape == (180,)
    assert default_cast_azimuth_deg()[-1] == 358.0
