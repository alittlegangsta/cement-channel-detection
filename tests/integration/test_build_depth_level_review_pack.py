from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCENARIO_ID = "robust_top_features_from_baseline__exclude5700_true__conf_0p6__split_5__linear_probe"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_inputs(root_dir: Path) -> None:
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    depth = np.arange(80.0, 125.0, dtype=np.float32)
    label = (depth.astype(np.int32) % 9) < 3
    clear_negative = ~label
    label_confidence = np.ones(depth.size, dtype=np.float32) * 0.8
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
        depth=depth,
        depth_has_channel_any=label,
        depth_candidate_fraction=label.astype(np.float32),
        depth_max_severity=label.astype(np.int8),
        depth_max_confidence=label_confidence,
        depth_min_zc=np.linspace(2.0, 4.0, depth.size, dtype=np.float32),
        depth_p05_zc=np.linspace(2.1, 4.1, depth.size, dtype=np.float32),
        depth_max_relative_drop=label.astype(np.float32) * 0.4,
        depth_largest_azimuth_object_width=label.astype(np.float32) * 8.0,
        depth_label_confidence=label_confidence,
        depth_orientation_confidence=np.ones(depth.size, dtype=np.float32),
        depth_plus_minus_disagreement_fraction=np.zeros(depth.size, dtype=np.float32),
        depth_clear_negative_mask=clear_negative,
        depth_review_band_mask=(depth >= 118.0) & (depth <= 122.0),
        depth_valid_fraction=np.ones(depth.size, dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.column_stack([depth, label.astype(np.float32)]).astype(
            np.float32
        ),
        depth_level_xsi_feature_names=np.asarray(["f0", "receiver_mean_f1"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    _write_json(
        reports_dir / "depth_level_refinement_report_v001.json",
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
                SCENARIO_ID: [{"feature_name": "receiver_mean_f1", "mean_coefficient": 1.0}]
            },
            "production_training": False,
            "no_final_labels": True,
            "no_stc": True,
            "no_apes": True,
            "no_deep_learning": True,
            "no_mvp4c": True,
        },
    )
    _write_json(reports_dir / "depth_level_refinement_gate_report.json", {"decision": "go"})
    rows = []
    for index, depth_value in enumerate(depth):
        score = 0.8 if label[index] else 0.2
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
    with (reports_dir / "depth_level_refinement_report_v001.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_depth_level_review_pack_cli_writes_outputs(tmp_path: Path) -> None:
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
            "scripts/06ab_build_depth_level_review_pack.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level manual review pack errors=0" in result.stdout
    output_dir = root_dir / "reports" / "depth_level_manual_review_v001"
    assert (output_dir / "review_intervals.csv").exists()
    summary = (output_dir / "review_summary.md").read_text(encoding="utf-8")
    assert "no final labels" in summary
