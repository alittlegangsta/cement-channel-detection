from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from cement_channel.evaluation.depth_level_review_pack import (
    build_depth_level_review_pack_from_config,
)

SCENARIO_ID = "robust_top_features_from_baseline__exclude5700_true__conf_0p6__split_5__linear_probe"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    depth = np.arange(100.0, 130.0, dtype=np.float32)
    label = np.zeros(depth.size, dtype=bool)
    label[[0, 1, 2, 10, 11, 12, 20, 21]] = True
    clear_negative = ~label
    review_band = (depth >= 118.0) & (depth <= 122.0)
    label_conf = np.linspace(0.45, 0.9, depth.size, dtype=np.float32)
    disagreement = np.zeros(depth.size, dtype=np.float32)
    disagreement[15:18] = 0.4
    labels_path = tmp_path / "depth_level_labels_v001.npz"
    features_path = tmp_path / "depth_level_xsi_features_v001.npz"
    np.savez_compressed(
        labels_path,
        depth=depth,
        depth_has_channel_any=label,
        depth_candidate_fraction=label.astype(np.float32),
        depth_max_severity=label.astype(np.int8) * 2,
        depth_max_confidence=label_conf,
        depth_min_zc=np.linspace(2.0, 5.0, depth.size, dtype=np.float32),
        depth_p05_zc=np.linspace(2.2, 5.2, depth.size, dtype=np.float32),
        depth_max_relative_drop=label.astype(np.float32) * 0.5,
        depth_largest_azimuth_object_width=label.astype(np.float32) * 10.0,
        depth_label_confidence=label_conf,
        depth_orientation_confidence=np.ones(depth.size, dtype=np.float32),
        depth_plus_minus_disagreement_fraction=disagreement,
        depth_clear_negative_mask=clear_negative,
        depth_review_band_mask=review_band,
        depth_valid_fraction=np.ones(depth.size, dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    features = np.column_stack(
        [
            np.linspace(0.0, 1.0, depth.size),
            np.linspace(1.0, 0.0, depth.size),
            label.astype(np.float32),
        ]
    ).astype(np.float32)
    np.savez_compressed(
        features_path,
        depth=depth,
        depth_level_xsi_features=features,
        depth_level_xsi_feature_names=np.asarray(["f0", "f1", "receiver_mean_f2"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    refinement_report = tmp_path / "depth_level_refinement_report_v001.json"
    _write_json(
        refinement_report,
        {
            "report_version": "depth_level_refinement_v001",
            "best_result": {
                "scenario_id": SCENARIO_ID,
                "feature_group": "robust_top_features_from_baseline",
                "confidence_threshold": 0.6,
            },
            "best_feature_group": "robust_top_features_from_baseline",
            "robustness_summary": {
                "depends_on_5700_band": False,
                "stable_over_permutation": True,
            },
            "top_features": {
                SCENARIO_ID: [
                    {"feature_name": "f0", "mean_coefficient": 0.8},
                    {"feature_name": "receiver_mean_f2", "mean_coefficient": 0.5},
                ]
            },
            "production_training": False,
            "no_final_labels": True,
            "no_stc": True,
            "no_apes": True,
            "no_deep_learning": True,
            "no_mvp4c": True,
        },
    )
    gate_report = tmp_path / "depth_level_refinement_gate_report.json"
    _write_json(gate_report, {"decision": "go"})
    rows = []
    for index, depth_value in enumerate(depth):
        score = 0.8 if label[index] else 0.2
        if index in {15, 16, 17}:
            score = 0.82
        if index in {20, 21}:
            score = 0.18
        rows.append(
            {
                "csv_version": "depth_level_refinement_predictions_v001",
                "scenario_id": SCENARIO_ID,
                "feature_group": "robust_top_features_from_baseline",
                "exclude_5700_band": "True",
                "confidence_threshold": "0.6",
                "n_splits": "5",
                "model_type": "linear_probe",
                "fold_index": str(index % 5),
                "depth": str(float(depth_value)),
                "label": "1" if label[index] else "0",
                "sample_weight": "1.0",
                "score": str(score),
                "prediction": "1" if score >= 0.5 else "0",
            }
        )
    prediction_csv = tmp_path / "depth_level_refinement_report_v001.csv"
    _write_csv(prediction_csv, rows)
    return {
        "labels": labels_path,
        "features": features_path,
        "refinement_report": refinement_report,
        "gate_report": gate_report,
        "prediction_csv": prediction_csv,
    }


def _write_review_config(path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/depth_level_manual_review.example.yaml").read_text(encoding="utf-8")
    )
    config["review_intervals"][0]["depth_min_ft"] = 118.0
    config["review_intervals"][0]["depth_max_ft"] = 122.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_build_depth_level_review_pack_writes_interval_outputs(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "review"
    review_config = tmp_path / "depth_level_manual_review.yaml"
    _write_review_config(review_config)

    report, intervals = build_depth_level_review_pack_from_config(
        depth_level_labels_npz=paths["labels"],
        depth_level_features_npz=paths["features"],
        refinement_report_json=paths["refinement_report"],
        refinement_gate_report_json=paths["gate_report"],
        refinement_csv=paths["prediction_csv"],
        review_config_path=review_config,
        output_dir=output_dir,
        overwrite=True,
    )

    assert report.errors == []
    assert report.source_gate_decision == "go"
    assert report.no_final_labels is True
    assert intervals
    assert {interval["interval_type"] for interval in intervals} >= {
        "true_positive_like",
        "clear_negative_like",
        "false_positive_like",
        "false_negative_like",
        "5700_band_review",
    }
    assert (output_dir / "review_intervals.csv").exists()
    data = json.loads((output_dir / "review_intervals.json").read_text(encoding="utf-8"))
    assert data["report"]["review_pack_version"] == "depth_level_manual_review_v001"
    assert "weak-label candidate" in (output_dir / "review_summary.md").read_text(
        encoding="utf-8"
    )


def test_build_depth_level_review_pack_blocks_non_go_gate(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    _write_json(paths["gate_report"], {"decision": "conditional_go"})

    report, _ = build_depth_level_review_pack_from_config(
        depth_level_labels_npz=paths["labels"],
        depth_level_features_npz=paths["features"],
        refinement_report_json=paths["refinement_report"],
        refinement_gate_report_json=paths["gate_report"],
        refinement_csv=paths["prediction_csv"],
        review_config_path="configs/depth_level_manual_review.example.yaml",
    )

    assert report.errors
    assert "must be go" in report.errors[0]
