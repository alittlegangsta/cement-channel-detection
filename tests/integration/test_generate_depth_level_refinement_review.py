from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


SCENARIO_ID = "all_depth_features__exclude5700_true__conf_0p5__split_3__logistic_regression"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_inputs(root_dir: Path) -> None:
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    scenario = {
        "scenario_id": SCENARIO_ID,
        "feature_group": "all_depth_features",
        "exclude_5700_band": True,
        "confidence_threshold": 0.5,
        "n_splits": 3,
        "model_type": "logistic_regression",
        "balanced_accuracy_mean": 0.7,
        "permutation_balanced_accuracy_mean": 0.5,
        "margin_mean": 0.2,
        "predicted_positive_rate": 0.5,
        "passes_gate_thresholds": True,
    }
    _write_json(
        reports_dir / "depth_level_refinement_report_v001.json",
        {
            "report_version": "depth_level_refinement_v001",
            "best_result": scenario,
            "best_feature_group": "all_depth_features",
            "scenario_summaries": [scenario],
            "confidence_threshold_summary": {
                "0.5": {"best_margin_mean": 0.2, "passing_scenario_count": 1}
            },
            "split_summary": {"3": {"best_margin_mean": 0.2, "passing_scenario_count": 1}},
            "exclude_5700_summary": {
                "True": {"best_margin_mean": 0.2, "passing_scenario_count": 1}
            },
            "top_features": {
                SCENARIO_ID: [
                    {"feature_name": "f0", "mean_coefficient": 0.4},
                    {"feature_name": "f1", "mean_coefficient": -0.2},
                ]
            },
            "manual_confirmation_items": [],
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
            "csv_version": "depth_level_refinement_predictions_v001",
            "scenario_id": SCENARIO_ID,
            "feature_group": "all_depth_features",
            "exclude_5700_band": True,
            "confidence_threshold": 0.5,
            "n_splits": 3,
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
    with (reports_dir / "depth_level_refinement_report_v001.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    depth = np.arange(100.0, 124.0, dtype=np.float32)
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
        depth=depth,
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
            [np.arange(24), np.arange(24)[::-1]]
        ).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["f0", "f1"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )


def test_generate_depth_level_refinement_review_cli_writes_review_dir(
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
            "scripts/06z_generate_depth_level_refinement_review.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level refinement review figures errors=0" in result.stdout
    output_dir = root_dir / "reports" / "depth_level_refinement_review_v001"
    summary = json.loads(
        (output_dir / "depth_level_refinement_review_summary_v001.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["review_version"] == "depth_level_refinement_review_v001"
    assert len(summary["figures"]) == 8
    assert (output_dir / "08_exclude_5700_sensitivity.png").exists()
