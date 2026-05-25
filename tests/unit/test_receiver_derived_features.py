from __future__ import annotations

import numpy as np

from cement_channel.features.receiver_derived_features import (
    build_receiver_derived_feature_table,
    compute_receiver_derived_features,
    match_sample_depths_to_basic,
    transform_receiver_features,
)
from cement_channel.features.receiver_feature_schema import load_receiver_feature_config


def _config():
    return load_receiver_feature_config("configs/mvp4b_receiver_features.example.yaml")


def _basic_arrays() -> dict[str, np.ndarray]:
    depth = np.array([102.0, 101.0, 100.0], dtype=np.float32)
    receiver = np.arange(13, dtype=np.float32)
    features = np.zeros((3, 13, 8, 6), dtype=np.float32)
    for depth_index in range(3):
        for side in range(8):
            base = 1.0 + depth_index + side * 0.1
            features[depth_index, :, side, :] = (
                base + receiver[:, None] * np.linspace(0.01, 0.06, 6)[None, :]
            )
    return {
        "xsi_depth": depth,
        "receiver_index": np.arange(1, 14, dtype=np.int16),
        "feature_names": np.array(
            [
                "rms_energy",
                "peak_abs",
                "mean_abs",
                "early_energy",
                "late_energy",
                "late_over_early_ratio",
            ]
        ),
        "xsi_basic_features": features,
        "no_model_training": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }


def _sample_arrays() -> dict[str, np.ndarray]:
    depth = np.repeat(np.array([102.0, 101.0, 100.0], dtype=np.float32), 8)
    side = np.tile(np.arange(8, dtype=np.int16), 3)
    transformed = np.ones((depth.size, 2), dtype=np.float32)
    labels = (side == 1).astype(np.int8)
    return {
        "sample_id": np.arange(depth.size, dtype=np.int64),
        "depth": depth,
        "side_index": side,
        "label_presence_plus": labels,
        "label_presence_minus_audit": labels.copy(),
        "valid_for_azimuthal_validation": np.ones(depth.size, dtype=bool),
        "plus_minus_disagreement": np.zeros(depth.size, dtype=bool),
        "exclude_large_depth_match_error": np.zeros(depth.size, dtype=bool),
        "sample_weight": np.ones(depth.size, dtype=np.float32),
        "transformed_features": transformed,
        "transformed_feature_names": np.array(["a", "b"]),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }


def test_match_sample_depths_to_basic_handles_descending_basic_depths() -> None:
    indices, errors = match_sample_depths_to_basic(
        np.array([100.0, 101.0, 102.0], dtype=np.float32),
        np.array([102.0, 101.0, 100.0], dtype=np.float32),
    )

    assert indices.tolist() == [2, 1, 0]
    np.testing.assert_allclose(errors, 0.0)


def test_compute_receiver_derived_features_returns_expected_families() -> None:
    config = _config()
    profile = _basic_arrays()["xsi_basic_features"][:2, :, 0, :]
    names = _basic_arrays()["feature_names"]

    features, feature_names = compute_receiver_derived_features(profile, names, config=config)

    assert features.shape == (2, 90)
    assert len(feature_names) == 90
    assert "far_minus_near_late_energy" in feature_names
    assert "receiver_energy_decay_slope_rms_energy" in feature_names
    assert np.all(np.isfinite(features))
    assert np.all(features[:, feature_names.index("far_minus_near_rms_energy")] > 0.0)


def test_transform_receiver_features_is_finite_and_scaled() -> None:
    config = _config()
    raw = np.array([[1.0, 2.0, -1.0], [2.0, 4.0, 1.0], [3.0, 8.0, 3.0]], dtype=np.float32)

    transformed, names, stats = transform_receiver_features(
        raw,
        ["a", "b", "c"],
        config=config,
    )

    assert transformed.shape[0] == 3
    assert names
    assert stats["log1p_positive_feature_count"] == 2
    assert np.all(np.isfinite(transformed))


def test_build_receiver_derived_feature_table_appends_features_without_labels() -> None:
    updated, report = build_receiver_derived_feature_table(
        basic_arrays=_basic_arrays(),
        sample_arrays=_sample_arrays(),
        config=_config(),
        inputs={"basic_features_npz": "synthetic.npz"},
    )

    assert report.errors == []
    assert report.raw_receiver_feature_count == 90
    assert report.finite_ratio["transformed_receiver_features"] == 1.0
    assert report.used_label_information_for_feature_construction is False
    assert updated["receiver_features_added"].shape == (24, 90)
    assert updated["transformed_features"].shape[0] == 24
    assert updated["transformed_features"].shape[1] > 2
    assert bool(updated["no_final_labels"])
