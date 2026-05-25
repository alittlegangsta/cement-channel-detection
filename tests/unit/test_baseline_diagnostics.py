from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from cement_channel.training.baseline_diagnostics import (
    diagnose_baseline_failure,
    prediction_distribution_summary,
    standardized_difference,
)
from cement_channel.training.baseline_schema import parse_baseline_config


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
            "permutation_seed": 7,
            "min_permutation_balanced_accuracy_margin": 0.02,
            "suspicious_metric_threshold": 0.98,
            "calibration_bins": 5,
        },
        "optimizer": {"max_iterations": 10, "learning_rate": 0.1, "l2_penalty": 0.0},
        "allowed_scope": "sanity_model_only",
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "no_production_model": True,
    }


def _sample_arrays() -> dict[str, np.ndarray]:
    depth_unique = np.arange(0.0, 90.0, 3.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    label = side.astype(np.int8)
    late_ratio = np.where(label == 1, 0.25, 0.20).astype(np.float32)
    raw_features = np.column_stack(
        [
            10.0 + side,
            20.0 + side,
            5.0 + side,
            100.0 - side,
            15.0 + side,
            late_ratio,
        ]
    ).astype(np.float32)
    transformed = np.column_stack([np.log1p(raw_features), raw_features]).astype(np.float32)
    disagreement = np.zeros(depth.size, dtype=bool)
    disagreement[::5] = True
    weights = np.where(label == 1, 1.0, 0.2).astype(np.float32)
    return {
        "sample_id": np.arange(depth.size, dtype=np.int64),
        "depth": depth,
        "side_index": side,
        "label_presence_plus": label,
        "label_presence_minus_audit": label.copy(),
        "label_confidence_plus": np.linspace(0.5, 1.0, depth.size).astype(np.float32),
        "orientation_confidence": np.ones(depth.size, dtype=np.float32),
        "valid_for_azimuthal_validation": np.ones(depth.size, dtype=bool),
        "plus_minus_disagreement": disagreement,
        "depth_match_error": np.zeros(depth.size, dtype=np.float32),
        "exclude_large_depth_match_error": np.zeros(depth.size, dtype=bool),
        "sample_weight": weights,
        "features": raw_features,
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
        "transformed_features": transformed,
        "transformed_feature_names": np.array(
            [
                "log1p_rms_energy",
                "log1p_peak_abs",
                "log1p_mean_abs",
                "log1p_early_energy",
                "log1p_late_energy",
                "log1p_late_over_early_ratio",
                "robust_scaled_rms_energy",
                "robust_scaled_peak_abs",
                "robust_scaled_mean_abs",
                "robust_scaled_early_energy",
                "robust_scaled_late_energy",
                "robust_scaled_late_over_early_ratio",
            ]
        ),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }


def _simple_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "permutation_check": {
                    "logistic_regression": {
                        "passes_margin": False,
                        "real_balanced_accuracy": 0.5,
                        "permutation_balanced_accuracy": 0.5,
                    }
                },
                "split": {"method": "depth_block_group_split", "folds": [1, 2, 3]},
                "errors": [
                    "logistic_regression: permutation balanced_accuracy "
                    "is not lower than real labels."
                ],
                "warnings": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _prediction_csv(path: Path, sample_count: int) -> None:
    rows = [
        {
            "csv_version": "simple_baseline_predictions_v001",
            "model_type": "logistic_regression",
            "fold_index": index % 3,
            "sample_id": index,
            "depth": float(index),
            "side_index": index % 2,
            "label_presence_plus": index % 2,
            "label_presence_minus_audit": index % 2,
            "plus_minus_disagreement": "False",
            "sample_weight": 1.0,
            "score": 0.82,
            "prediction": 1,
        }
        for index in range(sample_count)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_standardized_difference_detects_candidate_shift() -> None:
    values = np.array([0.0, 0.1, 1.0, 1.1], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int8)

    effect = standardized_difference(values, labels)

    assert effect is not None
    assert effect > 5.0


def test_prediction_distribution_flags_single_class_degeneracy() -> None:
    rows = [{"model_type": "m", "score": "0.8", "prediction": "1"} for _ in range(5)]

    summary = prediction_distribution_summary(rows)

    assert summary["m"]["predicted_positive_rate"] == 1.0
    assert summary["m"]["degenerate_all_candidate"] is True


def test_diagnose_baseline_failure_reports_reasons_and_figures(tmp_path: Path) -> None:
    arrays = _sample_arrays()
    sample_table = tmp_path / "baseline_sample_table_v001.npz"
    np.savez_compressed(sample_table, **arrays)
    report_path = tmp_path / "simple_baseline_report_v001.json"
    csv_path = tmp_path / "simple_baseline_v001.csv"
    _simple_report(report_path)
    _prediction_csv(csv_path, arrays["depth"].size)

    report = diagnose_baseline_failure(
        sample_table_npz=sample_table,
        simple_baseline_report_json=report_path,
        simple_baseline_csv=csv_path,
        baseline_config=parse_baseline_config(_baseline_config()),
        output_dir=tmp_path / "figures",
        overwrite=True,
    )

    assert report.no_go_confirmed is True
    assert "class_weight_failure" in report.no_go_reason_classes
    assert report.answers["model_degenerated_to_single_class"] is True
    assert report.answers["sample_weight_causes_effective_class_imbalance"] is True
    assert len(report.figures) == 4
    assert (tmp_path / "figures" / "01_prediction_score_distribution.png").exists()
