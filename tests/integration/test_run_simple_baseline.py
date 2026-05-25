from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_sample_table(path: Path) -> None:
    depth_unique = np.arange(0.0, 150.0, 2.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side_index = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    signal = np.sin(depth / 15.0) + (side_index * 0.35)
    label = side_index.astype(np.int8)
    transformed = np.column_stack(
        [signal, signal**2, np.cos(depth / 25.0), side_index.astype(np.float32)]
    ).astype(np.float32)
    np.savez_compressed(
        path,
        sample_id=np.arange(depth.size, dtype=np.int64),
        depth=depth,
        side_index=side_index,
        label_presence_plus=label,
        label_presence_minus_audit=label.copy(),
        valid_for_azimuthal_validation=np.ones(depth.size, dtype=bool),
        plus_minus_disagreement=np.zeros(depth.size, dtype=bool),
        exclude_large_depth_match_error=np.zeros(depth.size, dtype=bool),
        sample_weight=np.ones(depth.size, dtype=np.float32),
        transformed_features=transformed,
        transformed_feature_names=np.array(["signal", "signal2", "depth_cos", "side"]),
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
    config["optimizer"]["max_iterations"] = 80
    config["evaluation"]["min_permutation_balanced_accuracy_margin"] = 0.0
    config["evaluation"]["suspicious_metric_threshold"] = 1.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_run_simple_baseline_cli_writes_reports_and_csv(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_sample_table(interim_dir / "baseline_sample_table_v001.npz")
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
            "scripts/06d_run_simple_baseline.py",
            "--paths",
            str(paths_config),
            "--baseline-config",
            str(baseline_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Simple baseline sanity model errors=0" in result.stdout
    report = json.loads(
        (reports_dir / "simple_baseline_report_v001.json").read_text(encoding="utf-8")
    )
    assert report["report_version"] == "simple_baseline_v001"
    assert report["production_training"] is False
    assert report["no_final_labels"] is True
    assert report["aggregate_metrics"]
    csv_path = reports_dir / "simple_baseline_v001.csv"
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("csv_version")
    assert (reports_dir / "simple_baseline_report_v001.md").exists()
