from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_receiver_table(path: Path) -> None:
    depth_unique = np.arange(0.0, 120.0, 2.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    label = side.astype(np.int8)
    side_features = np.column_stack(
        [
            np.sin(depth / 10.0),
            np.cos(depth / 10.0),
            side.astype(np.float32) * 0.1,
        ]
    ).astype(np.float32)
    receiver_features = np.column_stack(
        [
            np.where(label == 1, 0.8, 0.2),
            np.where(label == 1, 0.5, 0.1),
            side.astype(np.float32),
            np.where(label == 1, 0.7, 0.3),
        ]
    ).astype(np.float32)
    np.savez_compressed(
        path,
        sample_id=np.arange(depth.size, dtype=np.int64),
        depth=depth,
        side_index=side,
        label_presence_plus=label,
        label_presence_minus_audit=label.copy(),
        valid_for_azimuthal_validation=np.ones(depth.size, dtype=bool),
        plus_minus_disagreement=np.zeros(depth.size, dtype=bool),
        exclude_large_depth_match_error=np.zeros(depth.size, dtype=bool),
        sample_weight=np.ones(depth.size, dtype=np.float32),
        sample_weight_capped_class_balanced_confidence=np.ones(depth.size, dtype=np.float32),
        transformed_features=np.column_stack([side_features, receiver_features]).astype(np.float32),
        transformed_feature_names=np.array(
            [
                "side_a",
                "side_b",
                "side_c",
                "robust_scaled_receiver_far_receiver_mean_late_over_early_ratio",
                "robust_scaled_receiver_far_minus_near_late_over_early_ratio",
                "robust_scaled_receiver_far_receiver_mean_late_energy",
                "robust_scaled_receiver_near_receiver_mean_late_energy",
            ]
        ),
        receiver_transformed_features_added=receiver_features,
        receiver_transformed_feature_names_added=np.array(
            [
                "robust_scaled_receiver_far_receiver_mean_late_over_early_ratio",
                "robust_scaled_receiver_far_minus_near_late_over_early_ratio",
                "robust_scaled_receiver_far_receiver_mean_late_energy",
                "robust_scaled_receiver_near_receiver_mean_late_energy",
            ]
        ),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def _write_baseline_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/mvp4b_simple_baseline.example.yaml").read_text())
    config["split"]["depth_block_size_ft"] = 10.0
    config["split"]["min_gap_ft"] = 0.0
    config["sample_filter"]["min_samples_per_class"] = 4
    config["split"]["min_samples_per_class_per_fold"] = 2
    config["optimizer"]["max_iterations"] = 40
    config["optimizer"]["learning_rate"] = 0.2
    config["evaluation"]["min_permutation_balanced_accuracy_margin"] = 0.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_run_receiver_feature_ablation_cli_writes_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_receiver_table(interim_dir / "baseline_sample_table_receiver_enhanced_v001.npz")
    baseline_config = tmp_path / "baseline.yaml"
    _write_baseline_config(baseline_config)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {interim_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06m_run_receiver_feature_ablation.py",
            "--paths",
            str(paths_config),
            "--baseline-config",
            str(baseline_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Receiver feature ablation errors=0" in result.stdout
    output_json = reports_dir / "receiver_feature_ablation_v001.json"
    output_csv = reports_dir / "receiver_feature_ablation_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_version"] == "receiver_feature_ablation_v001"
    assert report["scenario_count"] == 10
    assert "receiver_derived_only" in report["feature_sets_compared"]
    assert report["no_final_labels"] is True
    assert report["no_stc"] is True
    assert report["no_apes"] is True
    assert report["mvp4c_allowed"] is False
