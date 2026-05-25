from __future__ import annotations

import numpy as np

from cement_channel.training.sample_schema import parse_sample_table_config
from cement_channel.training.sample_table import (
    build_baseline_sample_table_from_arrays,
    compute_sample_weight,
    transform_features,
)
from tests.unit.test_sample_schema import _valid_raw_config


def _config():
    return parse_sample_table_config(_valid_raw_config())


def _label_arrays() -> dict[str, np.ndarray]:
    presence = np.array([[1, 0], [1, 0]], dtype=np.int8)
    return {
        "xsi_depth": np.array([100.0, 99.0], dtype=np.float32),
        "xsi_side_azimuth_deg": np.array([0.0, 45.0], dtype=np.float32),
        "label_presence_plus": presence,
        "label_severity_plus": np.where(presence == 1, 2, 0).astype(np.int8),
        "label_confidence_plus": np.array([[0.8, 0.2], [0.9, 0.3]], dtype=np.float32),
        "label_presence_minus_audit": np.array([[0, 0], [1, 0]], dtype=np.int8),
        "plus_minus_disagreement": np.array([[True, False], [False, False]]),
        "orientation_confidence": np.ones((2, 2), dtype=np.float32),
        "valid_for_azimuthal_validation": np.array([[True, True], [True, False]]),
        "valid_for_non_azimuthal_summary": np.ones((2, 2), dtype=bool),
        "cast_depth_mismatch": np.array([0.1, 0.7], dtype=np.float32),
        "no_final_labels": np.asarray(True),
    }


def _feature_arrays() -> dict[str, np.ndarray]:
    values = np.ones((2, 2, 6), dtype=np.float32)
    values[0, 0, :] = np.array([10, 100, 5, 4, 8, 2], dtype=np.float32)
    return {
        "xsi_basic_features_by_side": values,
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
        "no_model_training": np.asarray(True),
    }


def test_transform_features_outputs_log_and_robust_scaled_columns() -> None:
    config = _config()
    features = np.array([[0.0, 1.0], [3.0, 7.0], [8.0, 15.0]], dtype=np.float32)

    transformed, stats = transform_features(features, ("a", "b"), config)

    assert transformed.shape == (3, 4)
    assert set(stats) == {"a", "b"}
    assert np.all(np.isfinite(transformed))
    assert stats["a"]["finite_count"] == 3


def test_compute_sample_weight_applies_policy_flags() -> None:
    config = _config()

    weights = compute_sample_weight(
        label_confidence=np.array([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        valid_for_azimuthal=np.array([True, False, True, True]),
        plus_minus_disagreement=np.array([True, False, False, False]),
        large_depth_error=np.array([False, False, True, False]),
        feature_valid=np.array([True, True, True, False]),
        sample_config=config,
    )

    np.testing.assert_allclose(weights, np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32))


def test_build_baseline_sample_table_flattens_depth_side_samples() -> None:
    report_arrays, stats = build_baseline_sample_table_from_arrays(
        label_arrays=_label_arrays(),
        feature_arrays=_feature_arrays(),
        sample_config=_config(),
    )

    assert stats["errors"] == []
    assert report_arrays["features"].shape == (4, 6)
    assert report_arrays["transformed_features"].shape == (4, 12)
    assert report_arrays["sample_weight"][0] == np.float32(0.4)
    assert report_arrays["sample_weight"][2] == np.float32(0.0)
    assert report_arrays["exclude_large_depth_match_error"][2]
    assert report_arrays["label_presence_plus"].tolist() == [1, 0, 1, 0]
    assert stats["counts"]["candidate_count"] == 2
    assert stats["counts"]["high_confidence_candidate_count"] == 2
    assert stats["excluded_counts"]["exclude_large_depth_match_error"] == 2
    assert bool(report_arrays["no_model_training"].reshape(()))
    assert bool(report_arrays["no_final_labels"].reshape(()))


def test_build_baseline_sample_table_reports_shape_mismatch() -> None:
    features = _feature_arrays()
    features["xsi_basic_features_by_side"] = np.ones((1, 2, 6), dtype=np.float32)

    _arrays, stats = build_baseline_sample_table_from_arrays(
        label_arrays=_label_arrays(),
        feature_arrays=features,
        sample_config=_config(),
    )

    assert any("shape mismatch" in error for error in stats["errors"])
