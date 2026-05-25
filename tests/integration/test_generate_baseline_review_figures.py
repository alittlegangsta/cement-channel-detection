from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_inputs(reports_dir: Path) -> None:
    metrics = {
        "sample_count": 10,
        "weight_sum": 10.0,
        "weighted_accuracy": 0.7,
        "balanced_accuracy": 0.7,
        "precision": 0.7,
        "recall": 0.7,
        "f1": 0.7,
        "brier": 0.2,
    }
    _write_json(
        reports_dir / "simple_baseline_report_v001.json",
        {
            "report_version": "simple_baseline_v001",
            "fold_metrics": [
                {
                    "model_type": "logistic_regression",
                    "fold_index": 0,
                    "permutation": False,
                    "metrics": metrics,
                }
            ],
            "aggregate_metrics": {"logistic_regression": metrics},
            "permutation_aggregate_metrics": {
                "logistic_regression": {**metrics, "balanced_accuracy": 0.5}
            },
            "minus_audit_comparison": {"logistic_regression": metrics},
            "coefficient_summary": {
                "logistic_regression:f1": {"mean_coefficient": 0.3},
                "logistic_regression:f2": {"mean_coefficient": -0.1},
            },
            "production_training": False,
            "no_final_labels": True,
            "no_deep_learning": True,
            "no_stc": True,
            "no_apes": True,
            "no_production_model": True,
        },
    )
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
    with (reports_dir / "simple_baseline_v001.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_generate_baseline_review_figures_cli_writes_review_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06e_generate_baseline_review_figures.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Baseline review figures errors=0" in result.stdout
    output_dir = reports_dir / "simple_baseline_review_v001"
    summary = json.loads(
        (output_dir / "simple_baseline_review_summary_v001.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["review_version"] == "simple_baseline_review_v001"
    assert len(summary["figures"]) == 7
    assert (output_dir / "01_fold_metric_summary.png").exists()
