from __future__ import annotations

import numpy as np

from cement_channel.training.depth_level_refinement import (
    build_feature_group_indices,
    run_depth_level_refinement,
)
from cement_channel.training.depth_level_refinement_schema import (
    DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION,
    parse_depth_level_refinement_config,
)


def _config():
    raw = {
        "schema_version": "schema_v001",
        "config_version": DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION,
        "stage": "MVP-4B-R4c",
        "task": "controlled_depth_level_feature_refinement",
        "input_labels": "depth_level_labels_v001",
        "input_features": "depth_level_xsi_features_v001",
        "input_baseline_report": "depth_level_baseline_report_v001",
        "target_variant": "high_confidence_positive_vs_clear_negative",
        "label_status": "weak_label_candidate",
        "allowed_models": ["logistic_regression"],
        "feature_groups": [
            "all_depth_features",
            "late_over_early_features",
            "energy_window_features",
            "side_contrast_features",
            "receiver_summary_features",
            "robust_top_features_from_baseline",
        ],
        "target_filters": {
            "clear_negative_min_label_confidence": 0.5,
            "clear_negative_min_orientation_confidence": 0.5,
            "max_plus_minus_disagreement_fraction": 0.25,
        },
        "review_intervals": [
            {
                "name": "review_horizontal_severe_band_5700ft",
                "depth_min_ft": 5680.0,
                "depth_max_ft": 5720.0,
                "reason": "outside synthetic sample",
            }
        ],
        "robustness_checks": {
            "exclude_5700_band": [False, True],
            "confidence_thresholds": [0.4, 0.5, 0.6],
            "depth_block_splits": [3, 5],
            "permutation_repeats": 1,
            "feature_group_ablation": True,
            "fold_stability_required": True,
        },
        "split": {
            "method": "depth_block_split",
            "depth_block_size_ft": 20.0,
            "min_gap_ft": 0.0,
            "min_samples_per_class_per_fold": 1,
        },
        "optimizer": {
            "max_iterations": 60,
            "learning_rate": 0.1,
            "l2_penalty": 0.0001,
        },
        "gate_thresholds": {
            "min_margin_mean": 0.02,
            "min_margin_permutation": 0.0,
            "max_predicted_positive_rate": 0.95,
            "min_predicted_positive_rate": 0.05,
            "min_folds_above_permutation_fraction": 0.5,
            "suspicious_high_balanced_accuracy": 1.0,
        },
        "allowed_scope": "controlled_depth_level_refinement_only",
        "no_model_training_claim": True,
        "no_production_model": True,
        "no_final_labels": True,
        "no_mvp4c": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
    }
    return parse_depth_level_refinement_config(raw)


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    depth = np.arange(120, dtype=np.float32)
    label = (depth.astype(np.int32) % 10) < 5
    signal = label.astype(np.float32) * 2.0 + np.linspace(0.0, 0.1, depth.size)
    feature_names = np.asarray(
        [
            "side_mean_late_over_early_ratio",
            "side_mean_early_energy",
            "side_contrast_mean_abs",
            "receiver_mean_peak_abs",
            "near_far_ratio_mean_rms_energy",
            "max_side_anomaly_mean_abs",
        ]
    )
    features = np.column_stack(
        [
            signal,
            signal * 0.8,
            signal * 0.6,
            signal * 0.4,
            signal * 0.2,
            signal * 0.5,
        ]
    ).astype(np.float32)
    labels = {
        "depth": depth,
        "depth_has_channel_any": label,
        "depth_clear_negative_mask": ~label,
        "depth_review_band_mask": np.zeros(depth.size, dtype=bool),
        "depth_label_confidence": np.ones(depth.size, dtype=np.float32),
        "depth_orientation_confidence": np.ones(depth.size, dtype=np.float32),
        "depth_plus_minus_disagreement_fraction": np.zeros(depth.size, dtype=np.float32),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }
    feature_arrays = {
        "depth": depth,
        "depth_level_xsi_features": features,
        "depth_level_xsi_feature_names": feature_names,
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }
    baseline = {
        "report_version": "depth_level_baseline_v001",
        "usable_target_variants": ["high_confidence_positive_vs_clear_negative"],
        "best_result": {
            "target_variant": "high_confidence_positive_vs_clear_negative",
            "model_type": "logistic_regression",
        },
        "top_features": {
            "high_confidence_positive_vs_clear_negative:logistic_regression": [
                {"feature_name": "side_mean_late_over_early_ratio"},
                {"feature_name": "side_contrast_mean_abs"},
            ]
        },
        "production_training": False,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }
    return labels, feature_arrays, baseline


def test_build_feature_group_indices_uses_name_patterns_and_baseline_top_features() -> None:
    _, feature_arrays, baseline = _arrays()
    config = _config()
    names = feature_arrays["depth_level_xsi_feature_names"].astype(str).tolist()

    groups = build_feature_group_indices(names, baseline, config)

    assert groups["all_depth_features"].size == 6
    assert groups["late_over_early_features"].size == 1
    assert groups["energy_window_features"].size == 2
    assert groups["receiver_summary_features"].size == 2
    assert groups["robust_top_features_from_baseline"].size == 2


def test_run_depth_level_refinement_reports_passing_scenarios() -> None:
    labels, features, baseline = _arrays()

    report, rows = run_depth_level_refinement(
        label_arrays=labels,
        feature_arrays=features,
        baseline_report=baseline,
        config=_config(),
        inputs={"depth_level_labels_npz": "synthetic_labels.npz"},
    )

    assert report.report_version == "depth_level_refinement_v001"
    assert report.errors == []
    assert report.passing_scenario_count > 0
    assert report.best_result is not None
    assert report.best_feature_group is not None
    assert rows
    assert rows[0]["csv_version"] == "depth_level_refinement_predictions_v001"
    assert report.no_final_labels is True
