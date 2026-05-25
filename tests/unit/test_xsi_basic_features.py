from __future__ import annotations

import numpy as np

from cement_channel.features.xsi_basic_features import (
    XSI_BASIC_FEATURE_NAMES,
    aggregate_receiver_features,
    extract_basic_features,
)


def test_extract_basic_features_computes_expected_time_summaries() -> None:
    waveform = np.array([[[[1.0, -1.0, 2.0, -2.0]]]], dtype=np.float32)

    features = extract_basic_features(
        waveform,
        early_fraction=(0.0, 0.5),
        late_fraction=(0.5, 1.0),
    )

    values = dict(zip(XSI_BASIC_FEATURE_NAMES, features[0, 0, 0], strict=True))
    assert np.isclose(values["rms_energy"], np.sqrt(2.5))
    assert values["peak_abs"] == 2.0
    assert values["mean_abs"] == 1.5
    assert values["early_energy"] == 1.0
    assert values["late_energy"] == 4.0
    assert values["late_over_early_ratio"] == 4.0


def test_extract_basic_features_requires_four_dimensions() -> None:
    waveform = np.ones((2, 8, 4), dtype=np.float32)

    try:
        extract_basic_features(waveform)
    except ValueError as exc:
        assert "shape [depth, receiver, side, time]" in str(exc)
    else:
        raise AssertionError("Expected ValueError for rank-3 waveform.")


def test_aggregate_receiver_features_returns_mean_and_median() -> None:
    features = np.zeros((1, 3, 1, 2), dtype=np.float32)
    features[0, :, 0, 0] = np.array([1.0, 2.0, 100.0], dtype=np.float32)
    features[0, :, 0, 1] = np.array([4.0, 6.0, 8.0], dtype=np.float32)

    mean, median = aggregate_receiver_features(features)

    np.testing.assert_allclose(mean[0, 0], np.array([103.0 / 3.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(median[0, 0], np.array([2.0, 6.0], dtype=np.float32))
