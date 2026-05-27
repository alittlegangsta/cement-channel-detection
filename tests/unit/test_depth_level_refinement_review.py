from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from cement_channel.visualization.depth_level_refinement_review import (  # noqa: E402
    generate_depth_level_refinement_review_figures,
)

SCENARIO_ID = "all_depth_features__exclude5700_true__conf_0p5__split_3__logistic_regression"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _report() -> dict:
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
    return {
        "report_version": "depth_level_refinement_v001",
        "best_result": scenario,
        "best_feature_group": "all_depth_features",
        "scenario_summaries": [
            scenario,
            {
                **scenario,
                "scenario_id": "late",
                "feature_group": "late_over_early_features",
                "margin_mean": 0.04,
            },
        ],
        "confidence_threshold_summary": {
            "0.5": {"best_margin_mean": 0.2, "passing_scenario_count": 1}
        },
        "split_summary": {"3": {"best_margin_mean": 0.2, "passing_scenario_count": 1}},
        "exclude_5700_summary": {
            "True": {"best_margin_mean": 0.2, "passing_scenario_count": 1}
        },
        "top_features": {
            scenario["scenario_id"]: [
                {"feature_name": "f0", "mean_coefficient": 0.4},
                {"feature_name": "f1", "mean_coefficient": -0.2},
            ]
        },
        "manual_confirmation_items": ["Confirm feature group."],
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(labels_path: Path, features_path: Path) -> None:
    depth = np.arange(100.0, 124.0, dtype=np.float32)
    np.savez_compressed(
        labels_path,
        depth=depth,
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
            [np.arange(24), np.arange(24)[::-1]]
        ).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["f0", "f1"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )


def test_generate_depth_level_refinement_review_figures_writes_outputs(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "depth_level_refinement_report_v001.json"
    csv_path = tmp_path / "depth_level_refinement_report_v001.csv"
    labels_path = tmp_path / "depth_level_labels_v001.npz"
    features_path = tmp_path / "depth_level_xsi_features_v001.npz"
    output_dir = tmp_path / "review"
    _write_json(report_path, _report())
    _write_csv(csv_path)
    _write_npz(labels_path, features_path)

    review = generate_depth_level_refinement_review_figures(
        refinement_report_json=report_path,
        refinement_csv=csv_path,
        depth_level_labels_npz=labels_path,
        depth_level_features_npz=features_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert review.errors == []
    assert review.review_version == "depth_level_refinement_review_v001"
    assert len(review.figures) == 8
    assert review.manual_confirmation_items == ["Confirm feature group."]
    assert (output_dir / "01_depth_label_score_by_depth.png").read_bytes().startswith(
        b"\x89PNG"
    )
    assert (output_dir / "review_summary_template.md").exists()
