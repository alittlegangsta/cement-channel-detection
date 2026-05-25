from __future__ import annotations

import numpy as np

from cement_channel.training.baseline_schema import parse_baseline_config
from cement_channel.training.simple_baseline import (
    binary_metrics,
    fit_logistic_regression,
    predict_scores,
    prepare_baseline_samples,
    run_simple_baseline_from_arrays,
)


def _baseline_config() -> dict:
    return {
        "config_version": "mvp4b_simple_baseline_v001",
        "input_sample_table": "baseline_sample_table_v001",
        "label": "label_presence_plus",
        "label_status": "human_reviewed_candidate_v001",
        "primary_label": "plus",
        "audit_label": "minus_ablation",
        "no_final_labels": True,
        "model_type": ["logistic_regression", "linear_probe"],
        "feature_set": ["transformed_features"],
        "sample_filter": {
            "high_confidence_only": True,
            "valid_for_azimuthal_validation": True,
            "exclude_plus_minus_disagreement": False,
            "exclude_large_depth_match_error": True,
            "min_samples_per_class": 4,
        },
        "sample_weight": {"use_sample_weight": True, "source": "sample_weight"},
        "split": {
            "method": "depth_block_group_split",
            "n_splits": 3,
            "depth_block_size_ft": 10.0,
            "min_gap_ft": 0.0,
            "min_samples_per_class_per_fold": 2,
        },
        "evaluation": {
            "metrics": [
                "weighted_accuracy",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
                "calibration_summary",
            ],
            "permutation_check": True,
            "permutation_seed": 11,
            "min_permutation_balanced_accuracy_margin": 0.0,
            "suspicious_metric_threshold": 1.0,
            "calibration_bins": 5,
        },
        "optimizer": {"max_iterations": 80, "learning_rate": 0.2, "l2_penalty": 0.0001},
        "allowed_scope": "sanity_model_only",
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "no_production_model": True,
    }


def _sample_arrays() -> dict[str, np.ndarray]:
    depth_unique = np.arange(0.0, 120.0, 2.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side_index = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    signal = np.sin(depth / 12.0) + (side_index * 0.4)
    label = side_index.astype(np.int8)
    transformed = np.column_stack(
        [
            signal,
            signal**2,
            np.cos(depth / 20.0),
            side_index.astype(np.float32),
        ]
    ).astype(np.float32)
    return {
        "sample_id": np.arange(depth.size, dtype=np.int64),
        "depth": depth,
        "side_index": side_index,
        "label_presence_plus": label,
        "label_presence_minus_audit": label.copy(),
        "valid_for_azimuthal_validation": np.ones(depth.size, dtype=bool),
        "plus_minus_disagreement": np.zeros(depth.size, dtype=bool),
        "exclude_large_depth_match_error": np.zeros(depth.size, dtype=bool),
        "sample_weight": np.ones(depth.size, dtype=np.float32),
        "transformed_features": transformed,
        "transformed_feature_names": np.array(["signal", "signal2", "depth_cos", "side"]),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }


def test_fit_logistic_regression_separates_simple_data() -> None:
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    weights = np.ones(4, dtype=np.float32)

    coef = fit_logistic_regression(
        x,
        y,
        weights,
        max_iterations=200,
        learning_rate=0.4,
        l2_penalty=0.0,
    )
    score = predict_scores(x, coef, model_type="logistic_regression")

    assert score[:2].max() < 0.5
    assert score[2:].min() > 0.5


def test_binary_metrics_reports_weighted_scores_and_calibration() -> None:
    metrics = binary_metrics(
        np.array([0, 0, 1, 1], dtype=np.int8),
        np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32),
        np.ones(4, dtype=np.float32),
        calibration_bins=4,
    )

    assert metrics["weighted_accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert len(metrics["calibration_summary"]) == 4


def test_prepare_baseline_samples_filters_large_depth_error_and_zero_weight() -> None:
    arrays = _sample_arrays()
    arrays["exclude_large_depth_match_error"][0] = True
    arrays["sample_weight"][1] = 0.0
    config = parse_baseline_config(_baseline_config())

    prepared = prepare_baseline_samples(arrays, config)

    assert prepared["selected_count"] == arrays["depth"].size - 2
    assert prepared["excluded_large_depth_match_error"] == 1
    assert prepared["excluded_zero_or_invalid_weight"] == 1


def test_run_simple_baseline_from_arrays_outputs_metrics_and_predictions() -> None:
    config = parse_baseline_config(_baseline_config())

    report, rows = run_simple_baseline_from_arrays(
        arrays=_sample_arrays(),
        baseline_config=config,
        inputs={"sample_table_npz": "synthetic.npz"},
    )

    assert report.errors == []
    assert report.sample_counts["selected_samples"] == 120
    assert "logistic_regression" in report.aggregate_metrics
    assert "linear_probe" in report.aggregate_metrics
    assert report.permutation_check
    assert rows
    assert rows[0]["csv_version"] == "simple_baseline_predictions_v001"
