from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_inputs(interim_dir: Path) -> None:
    depth = np.arange(72, dtype=np.float32)
    label = (depth.astype(np.int32) % 12) < 6
    strong = np.zeros(72, dtype=bool)
    strong[np.flatnonzero(label)[:12]] = True
    clear = ~label
    signal = np.linspace(0.0, 0.3, 72, dtype=np.float32)
    signal[label] += 2.0
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
        depth=depth,
        depth_has_channel_any=label,
        depth_strong_positive_mask=strong,
        depth_clear_negative_mask=clear,
        depth_review_band_mask=np.zeros(72, dtype=bool),
        depth_label_confidence=np.ones(72, dtype=np.float32),
        depth_orientation_confidence=np.ones(72, dtype=np.float32),
        depth_plus_minus_disagreement_fraction=np.zeros(72, dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.column_stack([signal]).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["side_mean_signal"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/depth_level_baseline.example.yaml").read_text())
    config["target_filters"]["min_samples_per_class"] = 4
    config["target_filters"]["min_samples_per_class_per_fold"] = 1
    config["split"]["depth_block_size_ft"] = 12.0
    config["split"]["min_gap_ft"] = 0.0
    config["optimizer"]["max_iterations"] = 80
    config["evaluation"]["min_permutation_balanced_accuracy_margin"] = 0.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_run_depth_level_baseline_cli_writes_report_and_csv(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_inputs(interim_dir)
    baseline_config = tmp_path / "depth_level_baseline.yaml"
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
            "scripts/06v_run_depth_level_baseline.py",
            "--paths",
            str(paths_config),
            "--baseline-config",
            str(baseline_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level baseline sanity errors=0" in result.stdout
    output_json = reports_dir / "depth_level_baseline_report_v001.json"
    output_csv = reports_dir / "depth_level_baseline_report_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_version"] == "depth_level_baseline_v001"
    assert report["no_final_labels"] is True
    assert report["production_training"] is False
    assert report["usable_target_variants"]
    assert output_csv.read_text(encoding="utf-8").splitlines()[0].startswith("csv_version")
