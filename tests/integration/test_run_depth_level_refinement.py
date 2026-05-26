from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_inputs(root_dir: Path) -> None:
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    depth = np.arange(120, dtype=np.float32)
    label = (depth.astype(np.int32) % 10) < 5
    signal = label.astype(np.float32) * 2.0 + np.linspace(0.0, 0.1, depth.size)
    feature_names = np.asarray(
        [
            "side_mean_late_over_early_ratio",
            "side_mean_early_energy",
            "side_contrast_mean_abs",
            "receiver_mean_peak_abs",
            "near_far_ratio_mean_rms_energy",
            "max_side_anomaly_mean_abs",
        ]
    )
    features = np.column_stack(
        [
            signal,
            signal * 0.8,
            signal * 0.6,
            signal * 0.4,
            signal * 0.2,
            signal * 0.5,
        ]
    ).astype(np.float32)
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
        depth=depth,
        depth_has_channel_any=label,
        depth_clear_negative_mask=~label,
        depth_review_band_mask=np.zeros(depth.size, dtype=bool),
        depth_label_confidence=np.ones(depth.size, dtype=np.float32),
        depth_orientation_confidence=np.ones(depth.size, dtype=np.float32),
        depth_plus_minus_disagreement_fraction=np.zeros(depth.size, dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=features,
        depth_level_xsi_feature_names=feature_names,
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    (reports_dir / "depth_level_baseline_report_v001.json").write_text(
        json.dumps(
            {
                "report_version": "depth_level_baseline_v001",
                "usable_target_variants": ["high_confidence_positive_vs_clear_negative"],
                "best_result": {
                    "target_variant": "high_confidence_positive_vs_clear_negative",
                    "model_type": "logistic_regression",
                },
                "top_features": {
                    "high_confidence_positive_vs_clear_negative:logistic_regression": [
                        {"feature_name": "side_mean_late_over_early_ratio"},
                        {"feature_name": "side_contrast_mean_abs"},
                    ]
                },
                "production_training": False,
                "no_final_labels": True,
                "no_stc": True,
                "no_apes": True,
                "no_deep_learning": True,
                "no_mvp4c": True,
            }
        ),
        encoding="utf-8",
    )


def _write_refinement_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/depth_level_refinement.example.yaml").read_text())
    config["allowed_models"] = ["logistic_regression"]
    config["robustness_checks"]["permutation_repeats"] = 1
    config["split"]["depth_block_size_ft"] = 20.0
    config["split"]["min_gap_ft"] = 0.0
    config["split"]["min_samples_per_class_per_fold"] = 1
    config["optimizer"]["max_iterations"] = 60
    config["gate_thresholds"]["min_margin_mean"] = 0.02
    config["gate_thresholds"]["min_margin_permutation"] = 0.0
    config["gate_thresholds"]["max_predicted_positive_rate"] = 0.95
    config["gate_thresholds"]["min_predicted_positive_rate"] = 0.05
    config["gate_thresholds"]["min_folds_above_permutation_fraction"] = 0.5
    config["gate_thresholds"]["suspicious_high_balanced_accuracy"] = 1.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_run_depth_level_refinement_cli_writes_report_and_csv(tmp_path: Path) -> None:
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
    refinement_config = tmp_path / "depth_level_refinement.yaml"
    _write_refinement_config(refinement_config)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06y_run_depth_level_refinement.py",
            "--paths",
            str(paths_config),
            "--refinement-config",
            str(refinement_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level refinement recommendation=" in result.stdout
    output_json = root_dir / "reports" / "depth_level_refinement_report_v001.json"
    output_csv = root_dir / "reports" / "depth_level_refinement_report_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_version"] == "depth_level_refinement_v001"
    assert report["passing_scenario_count"] > 0
    assert report["no_final_labels"] is True
    assert output_csv.read_text(encoding="utf-8").splitlines()[0].startswith("csv_version")
