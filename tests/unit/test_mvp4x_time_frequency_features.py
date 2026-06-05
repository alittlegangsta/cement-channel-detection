from __future__ import annotations

import numpy as np

from cement_channel.features.mvp4x_time_frequency_features import (
    TF_BASE_FEATURE_NAMES,
    build_depth_time_frequency_features,
    compute_time_frequency_base_features,
)


def test_compute_time_frequency_base_features_is_finite_and_fixed_width() -> None:
    time = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    waveform = np.vstack(
        [
            np.sin(2.0 * np.pi * 8.0 * time),
            np.sin(2.0 * np.pi * 16.0 * time),
            np.cos(2.0 * np.pi * 24.0 * time),
        ]
    ).astype(np.float32)

    features = compute_time_frequency_base_features(waveform)

    assert features.shape == (3, len(TF_BASE_FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_build_depth_time_frequency_features_preserves_depth_rows() -> None:
    receiver_side = np.ones((4, 2, 3, len(TF_BASE_FEATURE_NAMES)), dtype=np.float32)

    matrix, names, groups = build_depth_time_frequency_features(
        receiver_side,
        base_feature_names=list(TF_BASE_FEATURE_NAMES),
    )

    assert matrix.shape[0] == 4
    assert matrix.shape[1] == len(names)
    assert len(groups) == len(names)
    assert np.isfinite(matrix).all()
