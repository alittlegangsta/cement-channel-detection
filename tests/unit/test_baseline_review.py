from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from cement_channel.visualization.baseline_review import generate_baseline_review_figures


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _baseline_report() -> dict:
    metrics = {
        "sample_count": 4,
        "weight_sum": 4.0,
        "weighted_accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "precision": 0.75,
        "recall": 0.75,
        "f1": 0.75,
        "brier": 0.2,
    }
    return {
        "report_version": "simple_baseline_v001",
        "fold_metrics": [
            {
                "model_type": "logistic_regression",
                "fold_index": 0,
                "permutation": False,
                "metrics": metrics,
            },
            {
                "model_type": "logistic_regression",
                "fold_index": 0,
                "permutation": True,
                "metrics": {**metrics, "balanced_accuracy": 0.5, "f1": 0.5},
            },
        ],
        "aggregate_metrics": {"logistic_regression": metrics},
        "permutation_aggregate_metrics": {
            "logistic_regression": {**metrics, "balanced_accuracy": 0.5, "f1": 0.5}
        },
        "minus_audit_comparison": {"logistic_regression": {**metrics, "balanced_accuracy": 0.7}},
        "coefficient_summary": {
            "logistic_regression:f1": {"mean_coefficient": 0.4},
            "logistic_regression:f2": {"mean_coefficient": -0.2},
        },
        "production_training": False,
        "no_final_labels": True,
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "no_production_model": True,
    }


def _write_csv(path: Path) -> None:
    rows = [
        {
            "csv_version": "simple_baseline_predictions_v001",
            "model_type": "logistic_regression",
            "fold_index": 0,
            "sample_id": index,
            "depth": 100.0 + index,
            "side_index": index % 2,
            "label_presence_plus": index % 2,
            "label_presence_minus_audit": index % 2,
            "plus_minus_disagreement": "False",
            "sample_weight": 1.0,
            "score": 0.2 if index % 2 == 0 else 0.8,
            "prediction": index % 2,
        }
        for index in range(20)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_generate_baseline_review_figures_writes_expected_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "simple_baseline_report_v001.json"
    csv_path = tmp_path / "simple_baseline_v001.csv"
    output_dir = tmp_path / "review"
    _write_json(report_path, _baseline_report())
    _write_csv(csv_path)

    report = generate_baseline_review_figures(
        simple_baseline_report_json=report_path,
        simple_baseline_csv=csv_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert report.errors == []
    assert report.review_version == "simple_baseline_review_v001"
    assert len(report.figures) == 7
    assert (output_dir / "01_fold_metric_summary.png").read_bytes().startswith(b"\x89PNG")
    assert (output_dir / "review_summary_template.md").exists()
    assert (output_dir / "simple_baseline_review_summary_v001.json").exists()
