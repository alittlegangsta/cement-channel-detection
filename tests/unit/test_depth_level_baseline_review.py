from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from cement_channel.visualization.depth_level_baseline_review import (  # noqa: E402
    generate_depth_level_baseline_review_figures,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _baseline_report() -> dict:
    check = {
        "real_balanced_accuracy": 0.7,
        "permutation_balanced_accuracy": 0.5,
        "balanced_accuracy_margin": 0.2,
        "required_margin": 0.03,
        "predicted_positive_rate": 0.5,
        "degenerate_prediction": False,
        "stable_fold_count": 3,
        "stable_fold_min_count": 2,
        "stable_folds_pass": True,
        "fold_checks": [],
        "permutation_lower_than_real": True,
        "passes_margin": True,
        "usable": True,
    }
    metrics = {
        "sample_count": 20,
        "weight_sum": 20.0,
        "balanced_accuracy": 0.7,
        "precision": 0.7,
        "recall": 0.7,
        "f1": 0.7,
        "predicted_positive_rate": 0.5,
        "degenerate_prediction": False,
    }
    return {
        "report_version": "depth_level_baseline_v001",
        "best_result": {
            "target_variant": "high_confidence_positive_vs_clear_negative",
            "model_type": "logistic_regression",
            **check,
        },
        "target_variant_summaries": {
            "high_confidence_positive_vs_clear_negative": {
                "status": "runnable",
                "sample_count": 20,
                "positive_count": 10,
                "negative_count": 10,
            }
        },
        "fold_metrics": [
            {
                "target_variant": "high_confidence_positive_vs_clear_negative",
                "model_type": "logistic_regression",
                "fold_index": 0,
                "permutation": False,
                "metrics": metrics,
            }
        ],
        "aggregate_metrics": {
            "high_confidence_positive_vs_clear_negative": {
                "logistic_regression": metrics,
            }
        },
        "permutation_metrics": {
            "high_confidence_positive_vs_clear_negative": {
                "logistic_regression": {**metrics, "balanced_accuracy": 0.5},
            }
        },
        "permutation_check": {
            "high_confidence_positive_vs_clear_negative": {
                "logistic_regression": check,
            }
        },
        "top_features": {
            "high_confidence_positive_vs_clear_negative:logistic_regression": [
                {"feature_name": "f0", "mean_coefficient": 0.4},
                {"feature_name": "f1", "mean_coefficient": -0.2},
            ]
        },
        "usable_target_variants": ["high_confidence_positive_vs_clear_negative"],
        "production_training": False,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }


def _write_csv(path: Path) -> None:
    rows = [
        {
            "csv_version": "depth_level_baseline_predictions_v001",
            "target_variant": "high_confidence_positive_vs_clear_negative",
            "model_type": "logistic_regression",
            "fold_index": index % 3,
            "depth": 100.0 + index,
            "label": index % 2,
            "sample_weight": 1.0,
            "score": 0.2 if index % 2 == 0 else 0.8,
            "prediction": index % 2,
        }
        for index in range(24)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(labels_path: Path, features_path: Path) -> None:
    depth = np.arange(100.0, 124.0, dtype=np.float32)
    label = (np.arange(24) % 2) == 1
    np.savez_compressed(
        labels_path,
        depth=depth,
        depth_strong_positive_mask=label & (np.arange(24) < 8),
        depth_clear_negative_mask=~label,
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    np.savez_compressed(
        features_path,
        depth=depth,
        depth_level_xsi_features=np.column_stack(
            [label.astype(np.float32), np.linspace(0.0, 1.0, 24)]
        ).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["f0", "f1"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )


def test_generate_depth_level_baseline_review_figures_writes_outputs(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "depth_level_baseline_report_v001.json"
    csv_path = tmp_path / "depth_level_baseline_report_v001.csv"
    labels_path = tmp_path / "depth_level_labels_v001.npz"
    features_path = tmp_path / "depth_level_xsi_features_v001.npz"
    output_dir = tmp_path / "review"
    _write_json(report_path, _baseline_report())
    _write_csv(csv_path)
    _write_npz(labels_path, features_path)

    review = generate_depth_level_baseline_review_figures(
        baseline_report_json=report_path,
        baseline_csv=csv_path,
        depth_level_labels_npz=labels_path,
        depth_level_features_npz=features_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert review.errors == []
    assert review.review_version == "depth_level_baseline_review_v001"
    assert review.preferred_target_variant == "high_confidence_positive_vs_clear_negative"
    assert len(review.figures) == 6
    assert (output_dir / "01_depth_label_vs_prediction_score.png").read_bytes().startswith(
        b"\x89PNG"
    )
    assert (output_dir / "depth_level_baseline_review_summary_v001.json").exists()
