from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_inputs(root_dir: Path) -> None:
    reports_dir = root_dir / "reports"
    interim_dir = root_dir / "interim"
    reports_dir.mkdir(parents=True)
    interim_dir.mkdir()
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
    _write_json(
        reports_dir / "depth_level_baseline_report_v001.json",
        {
            "report_version": "depth_level_baseline_v001",
            "best_result": {
                "target_variant": "high_confidence_positive_vs_clear_negative",
                "model_type": "logistic_regression",
                **check,
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
            "fold_metrics": [],
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
            "usable_target_variants": ["high_confidence_positive_vs_clear_negative"],
            "production_training": False,
            "no_final_labels": True,
            "no_stc": True,
            "no_apes": True,
            "no_deep_learning": True,
            "no_mvp4c": True,
        },
    )
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
    with (reports_dir / "depth_level_baseline_report_v001.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    depth = np.arange(100.0, 124.0, dtype=np.float32)
    label = (np.arange(24) % 2) == 1
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
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
        interim_dir / "depth_level_xsi_features_v001.npz",
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


def test_generate_depth_level_baseline_review_cli_writes_review_dir(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    _write_inputs(root_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {root_dir / 'interim'}",
                f"  reports: {root_dir / 'reports'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06w_generate_depth_level_baseline_review.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level baseline review figures errors=0" in result.stdout
    output_dir = root_dir / "reports" / "depth_level_baseline_review_v001"
    summary = json.loads(
        (output_dir / "depth_level_baseline_review_summary_v001.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["review_version"] == "depth_level_baseline_review_v001"
    assert len(summary["figures"]) == 6
    assert (output_dir / "01_depth_label_vs_prediction_score.png").exists()
