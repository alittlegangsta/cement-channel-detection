from __future__ import annotations

import numpy as np

from cement_channel.training.baseline_schema import parse_baseline_config
from cement_channel.training.simple_baseline import (
    RemediationAblationScenario,
    run_baseline_remediation_ablation,
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


def _remediation_arrays() -> dict[str, np.ndarray]:
    depth_unique = np.arange(0.0, 120.0, 2.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side_index = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    label = side_index.astype(np.int8)
    raw = np.column_stack(
        [
            10.0 + side_index,
            20.0 + side_index,
            5.0 + side_index,
            100.0 - side_index,
            15.0 + side_index,
            np.where(label == 1, 0.35, 0.15),
        ]
    ).astype(np.float32)
    original = np.column_stack([raw[:, -1], raw[:, -1] ** 2]).astype(np.float32)
    enhanced = np.column_stack([original, side_index.astype(np.float32)]).astype(np.float32)
    weights = np.ones(depth.size, dtype=np.float32)
    return {
        "sample_id": np.arange(depth.size, dtype=np.int64),
        "depth": depth,
        "side_index": side_index,
        "features": raw,
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
        "label_presence_plus": label,
        "label_presence_minus_audit": label.copy(),
        "valid_for_azimuthal_validation": np.ones(depth.size, dtype=bool),
        "plus_minus_disagreement": np.zeros(depth.size, dtype=bool),
        "exclude_large_depth_match_error": np.zeros(depth.size, dtype=bool),
        "depth_match_error": np.zeros(depth.size, dtype=np.float32),
        "sample_weight": weights,
        "sample_weight_unweighted": weights,
        "sample_weight_confidence_only": weights,
        "sample_weight_class_balanced_confidence": weights,
        "sample_weight_capped_class_balanced_confidence": weights,
        "transformed_features_original": original,
        "transformed_feature_names_original": np.array(["ratio", "ratio2"]),
        "base_transformed_feature_count": np.asarray(2, dtype=np.int32),
        "transformed_features": enhanced,
        "transformed_feature_names": np.array(["ratio", "ratio2", "side"]),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }


def test_run_baseline_remediation_ablation_compares_weight_and_feature_sets() -> None:
    config = parse_baseline_config(_baseline_config())
    scenarios = (
        RemediationAblationScenario(
            "original_unweighted",
            "original_transformed",
            "unweighted",
            "include",
            0.5,
        ),
        RemediationAblationScenario(
            "late_ratio_balanced",
            "late_over_early_ratio_only",
            "capped_class_balanced_confidence",
            "include",
            0.5,
        ),
    )

    report = run_baseline_remediation_ablation(
        arrays=_remediation_arrays(),
        baseline_config=config,
        inputs={"sample_table_npz": "synthetic.npz"},
        scenarios=scenarios,
    )

    assert report.report_version == "baseline_remediation_ablation_v001"
    assert report.scenario_count == 2
    assert report.summary_rows
    assert {row["weight_policy"] for row in report.summary_rows} == {
        "unweighted",
        "capped_class_balanced_confidence",
    }
    assert all(row["predicted_positive_rate"] is not None for row in report.summary_rows)
    assert report.no_final_labels is True
    assert report.no_stc is True
    assert report.no_apes is True
