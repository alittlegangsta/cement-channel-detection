from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("matplotlib")


def _write_sample_table(path: Path) -> None:
    depth_unique = np.arange(0.0, 90.0, 3.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    label = side.astype(np.int8)
    late_ratio = np.where(label == 1, 0.25, 0.20).astype(np.float32)
    raw = np.column_stack(
        [10.0 + side, 20.0 + side, 5.0 + side, 100.0 - side, 15.0 + side, late_ratio]
    ).astype(np.float32)
    transformed = np.column_stack([np.log1p(raw), raw]).astype(np.float32)
    disagreement = np.zeros(depth.size, dtype=bool)
    disagreement[::4] = True
    np.savez_compressed(
        path,
        sample_id=np.arange(depth.size, dtype=np.int64),
        depth=depth,
        side_index=side,
        label_presence_plus=label,
        label_presence_minus_audit=label.copy(),
        label_confidence_plus=np.linspace(0.5, 1.0, depth.size).astype(np.float32),
        orientation_confidence=np.ones(depth.size, dtype=np.float32),
        valid_for_azimuthal_validation=np.ones(depth.size, dtype=bool),
        plus_minus_disagreement=disagreement,
        depth_match_error=np.zeros(depth.size, dtype=np.float32),
        exclude_large_depth_match_error=np.zeros(depth.size, dtype=bool),
        sample_weight=np.where(label == 1, 1.0, 0.2).astype(np.float32),
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
        transformed_features=transformed,
        transformed_feature_names=np.array(
            [
                "log1p_rms_energy",
                "log1p_peak_abs",
                "log1p_mean_abs",
                "log1p_early_energy",
                "log1p_late_energy",
                "log1p_late_over_early_ratio",
                "robust_scaled_rms_energy",
                "robust_scaled_peak_abs",
                "robust_scaled_mean_abs",
                "robust_scaled_early_energy",
                "robust_scaled_late_energy",
                "robust_scaled_late_over_early_ratio",
            ]
        ),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def _write_baseline_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "permutation_check": {
                    "logistic_regression": {
                        "passes_margin": False,
                        "real_balanced_accuracy": 0.5,
                        "permutation_balanced_accuracy": 0.5,
                    }
                },
                "split": {"method": "depth_block_group_split", "folds": [1, 2, 3]},
                "errors": [
                    "logistic_regression: permutation balanced_accuracy "
                    "is not lower than real labels."
                ],
                "warnings": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_prediction_csv(path: Path, sample_count: int) -> None:
    rows = [
        {
            "csv_version": "simple_baseline_predictions_v001",
            "model_type": "logistic_regression",
            "fold_index": index % 3,
            "sample_id": index,
            "depth": float(index),
            "side_index": index % 2,
            "label_presence_plus": index % 2,
            "label_presence_minus_audit": index % 2,
            "plus_minus_disagreement": "False",
            "sample_weight": 1.0,
            "score": 0.82,
            "prediction": 1,
        }
        for index in range(sample_count)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/mvp4b_simple_baseline.example.yaml").read_text())
    config["split"]["depth_block_size_ft"] = 10.0
    config["split"]["min_gap_ft"] = 0.0
    config["sample_filter"]["min_samples_per_class"] = 4
    config["split"]["min_samples_per_class_per_fold"] = 2
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_diagnose_baseline_failure_cli_writes_reports_and_figures(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    sample_table = interim_dir / "baseline_sample_table_v001.npz"
    _write_sample_table(sample_table)
    _write_baseline_report(reports_dir / "simple_baseline_report_v001.json")
    _write_prediction_csv(reports_dir / "simple_baseline_v001.csv", sample_count=60)
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
            "scripts/06g_diagnose_baseline_failure.py",
            "--paths",
            str(paths_config),
            "--baseline-config",
            str(baseline_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Baseline failure diagnostics errors=0" in result.stdout
    report = json.loads(
        (reports_dir / "baseline_failure_diagnostics_v001.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["diagnostics_version"] == "baseline_failure_diagnostics_v001"
    assert report["no_go_confirmed"] is True
    assert report["answers"]["model_degenerated_to_single_class"] is True
    assert (reports_dir / "baseline_failure_diagnostics_v001.md").exists()
    assert (
        reports_dir
        / "baseline_failure_diagnostics_v001"
        / "01_prediction_score_distribution.png"
    ).exists()
