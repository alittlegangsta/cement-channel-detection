from __future__ import annotations

import numpy as np

from cement_channel.training.depth_level_baseline import (
    build_target_variant,
    run_depth_level_baseline,
)
from cement_channel.training.depth_level_baseline_schema import (
    DEPTH_LEVEL_BASELINE_CONFIG_VERSION,
    parse_depth_level_baseline_config,
)


def _config(min_samples: int = 4):
    raw = _valid_raw_config()
    raw["target_filters"]["min_samples_per_class"] = min_samples
    raw["target_filters"]["min_samples_per_class_per_fold"] = 1
    raw["split"]["depth_block_size_ft"] = 10.0
    raw["split"]["min_gap_ft"] = 0.0
    raw["optimizer"]["max_iterations"] = 80
    raw["evaluation"]["min_permutation_balanced_accuracy_margin"] = 0.0
    return parse_depth_level_baseline_config(raw)


def _valid_raw_config() -> dict:
    return {
        "schema_version": "schema_v001",
        "config_version": DEPTH_LEVEL_BASELINE_CONFIG_VERSION,
        "stage": "MVP-4B-R4b",
        "task": "depth_level_baseline_sanity_model",
        "input_labels": "depth_level_labels_v001",
        "input_features": "depth_level_xsi_features_v001",
        "primary_task": "depth_has_channel",
        "label_status": "weak_label_candidate",
        "model_type": ["logistic_regression", "linear_probe"],
        "feature_set": ["depth_level_xsi_features"],
        "target_variants": [
            "all_positive_vs_negative",
            "strong_positive_vs_clear_negative",
            "high_confidence_positive_vs_clear_negative",
        ],
        "target_filters": {
            "high_confidence_positive": {
                "min_label_confidence": 0.5,
                "min_orientation_confidence": 0.5,
                "max_plus_minus_disagreement_fraction": 0.25,
            },
            "clear_negative": {
                "min_label_confidence": 0.5,
                "min_orientation_confidence": 0.5,
                "max_plus_minus_disagreement_fraction": 0.25,
            },
            "exclude_review_band": True,
            "min_samples_per_class": 20,
            "min_samples_per_class_per_fold": 5,
            "warn_if_variant_too_small": True,
        },
        "split": {
            "method": "depth_block_split",
            "n_splits": 3,
            "depth_block_size_ft": 250.0,
            "min_gap_ft": 5.0,
        },
        "evaluation": {
            "metrics": [
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "permutation_margin",
            ],
            "permutation_check": True,
            "permutation_seed": 202405,
            "min_permutation_balanced_accuracy_margin": 0.03,
            "degenerate_prediction_min_positive_rate": 0.05,
            "degenerate_prediction_max_positive_rate": 0.95,
            "stable_fold_min_count": 2,
        },
        "optimizer": {
            "max_iterations": 350,
            "learning_rate": 0.1,
            "l2_penalty": 0.0001,
        },
        "allowed_scope": "depth_level_baseline_sanity_only",
        "no_model_training_claim": True,
        "no_production_model": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    depth = np.arange(60, dtype=np.float32)
    label = (depth.astype(np.int32) % 10) < 5
    strong = np.zeros(60, dtype=bool)
    strong[np.flatnonzero(label)[:8]] = True
    clear = ~label
    signal = np.linspace(0.0, 0.2, 60, dtype=np.float32)
    signal[label] += 2.0
    features = np.column_stack([signal, np.sin(depth / 10.0)]).astype(np.float32)
    label_arrays = {
        "depth": depth,
        "depth_has_channel_any": label,
        "depth_strong_positive_mask": strong,
        "depth_clear_negative_mask": clear,
        "depth_review_band_mask": np.zeros(60, dtype=bool),
        "depth_label_confidence": np.ones(60, dtype=np.float32),
        "depth_orientation_confidence": np.ones(60, dtype=np.float32),
        "depth_plus_minus_disagreement_fraction": np.zeros(60, dtype=np.float32),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }
    feature_arrays = {
        "depth": depth,
        "depth_level_xsi_features": features,
        "depth_level_xsi_feature_names": np.asarray(["side_mean_signal", "receiver_noise"]),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }
    return label_arrays, feature_arrays


def test_build_target_variant_skips_undersized_strong_positive() -> None:
    labels, features = _arrays()
    config = _config(min_samples=20)
    prepared = __import__(
        "cement_channel.training.depth_level_baseline",
        fromlist=["prepare_depth_level_baseline_inputs"],
    ).prepare_depth_level_baseline_inputs(labels, features)

    variant = build_target_variant(prepared, config, "strong_positive_vs_clear_negative")

    assert variant["runnable"] is False
    assert variant["summary"]["status"] == "skipped_too_few_samples"
    assert variant["summary"]["positive_count"] == 8


def test_run_depth_level_baseline_reports_usable_variant_and_permutation_margin() -> None:
    labels, features = _arrays()

    report, rows = run_depth_level_baseline(
        label_arrays=labels,
        feature_arrays=features,
        config=_config(min_samples=4),
        inputs={"depth_level_labels_npz": "synthetic_labels.npz"},
    )

    assert report.errors == []
    assert "all_positive_vs_negative" in report.usable_target_variants
    assert report.best_result is not None
    assert report.best_result["balanced_accuracy_margin"] >= 0.0
    assert rows
    assert rows[0]["csv_version"] == "depth_level_baseline_predictions_v001"
    assert report.no_final_labels is True
