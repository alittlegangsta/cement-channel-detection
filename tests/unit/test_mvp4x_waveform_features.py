from __future__ import annotations

import numpy as np

from cement_channel.features.mvp4x_waveform_features import (
    BASE_FEATURE_NAMES,
    build_depth_waveform_features,
    compute_waveform_base_features,
    feature_correlation_summary,
)


def test_compute_waveform_base_features_has_expected_columns() -> None:
    waveform = np.tile(np.linspace(-1.0, 1.0, 16, dtype=np.float32), (3, 1))

    features = compute_waveform_base_features(waveform)

    assert features.shape == (3, len(BASE_FEATURE_NAMES))
    assert np.all(np.isfinite(features))
    assert features[0, BASE_FEATURE_NAMES.index("peak_abs")] == 1.0
    assert features[0, BASE_FEATURE_NAMES.index("zero_crossing_rate")] > 0.0


def test_build_depth_waveform_features_preserves_depth_rows() -> None:
    receiver_side = np.ones((5, 2, 3, len(BASE_FEATURE_NAMES)), dtype=np.float32)

    matrix, names, groups = build_depth_waveform_features(
        receiver_side,
        base_feature_names=list(BASE_FEATURE_NAMES),
    )

    assert matrix.shape[0] == 5
    assert matrix.shape[1] == len(names)
    assert len(groups) == len(names)
    assert np.all(np.isfinite(matrix))


def test_feature_correlation_summary_reports_pairs() -> None:
    features = np.column_stack(
        [
            np.arange(10, dtype=np.float32),
            np.arange(10, dtype=np.float32) * 2.0,
            np.sin(np.arange(10, dtype=np.float32)),
        ]
    )

    summary = feature_correlation_summary(features, ["a", "b", "c"], top_n=2)

    assert summary["status"] == "completed"
    assert summary["top_abs_correlations"][0]["feature_a"] == "a"
    assert summary["top_abs_correlations"][0]["feature_b"] == "b"

