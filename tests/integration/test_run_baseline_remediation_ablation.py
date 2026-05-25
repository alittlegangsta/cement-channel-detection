from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_enhanced_sample_table(path: Path) -> None:
    depth_unique = np.arange(0.0, 120.0, 2.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    label = side.astype(np.int8)
    raw = np.column_stack(
        [
            10.0 + side,
            20.0 + side,
            5.0 + side,
            100.0 - side,
            15.0 + side,
            np.where(label == 1, 0.35, 0.15),
        ]
    ).astype(np.float32)
    original = np.column_stack([raw[:, -1], raw[:, -1] ** 2]).astype(np.float32)
    enhanced = np.column_stack([original, side.astype(np.float32)]).astype(np.float32)
    weights = np.ones(depth.size, dtype=np.float32)
    np.savez_compressed(
        path,
        sample_id=np.arange(depth.size, dtype=np.int64),
        depth=depth,
        side_index=side,
        features=raw,
        feature_names=np.array(
            [
                "rms_energy",
                "peak_abs",
                "mean_abs",
                "early_energy",
                "late_energy",
                "late_over_early_ratio",
            ]
        ),
        label_presence_plus=label,
        label_presence_minus_audit=label.copy(),
        valid_for_azimuthal_validation=np.ones(depth.size, dtype=bool),
        plus_minus_disagreement=np.zeros(depth.size, dtype=bool),
        exclude_large_depth_match_error=np.zeros(depth.size, dtype=bool),
        depth_match_error=np.zeros(depth.size, dtype=np.float32),
        sample_weight=weights,
        sample_weight_unweighted=weights,
        sample_weight_confidence_only=weights,
        sample_weight_class_balanced_confidence=weights,
        sample_weight_capped_class_balanced_confidence=weights,
        transformed_features_original=original,
        transformed_feature_names_original=np.array(["ratio", "ratio2"]),
        base_transformed_feature_count=np.asarray(2, dtype=np.int32),
        transformed_features=enhanced,
        transformed_feature_names=np.array(["ratio", "ratio2", "side"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/mvp4b_simple_baseline.example.yaml").read_text())
    config["split"]["depth_block_size_ft"] = 10.0
    config["split"]["min_gap_ft"] = 0.0
    config["sample_filter"]["min_samples_per_class"] = 4
    config["split"]["min_samples_per_class_per_fold"] = 2
    config["optimizer"]["max_iterations"] = 40
    config["optimizer"]["learning_rate"] = 0.2
    config["evaluation"]["min_permutation_balanced_accuracy_margin"] = 0.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_run_baseline_remediation_ablation_cli_writes_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_enhanced_sample_table(interim_dir / "baseline_sample_table_enhanced_v001.npz")
    baseline_config = tmp_path / "baseline.yaml"
    _write_config(baseline_config)
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
            "scripts/06j_run_baseline_remediation_ablation.py",
            "--paths",
            str(paths_config),
            "--baseline-config",
            str(baseline_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Baseline remediation ablation errors=0" in result.stdout
    output_json = reports_dir / "baseline_remediation_ablation_v001.json"
    output_csv = reports_dir / "baseline_remediation_ablation_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_version"] == "baseline_remediation_ablation_v001"
    assert report["scenario_count"] == 16
    assert report["summary_rows"]
    assert report["no_final_labels"] is True
    assert report["no_stc"] is True
    assert report["no_apes"] is True
