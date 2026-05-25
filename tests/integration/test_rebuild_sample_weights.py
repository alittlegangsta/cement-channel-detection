from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_sample_table(path: Path) -> None:
    depth_unique = np.arange(0.0, 120.0, 5.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    side = np.tile(np.array([0, 1], dtype=np.int16), depth_unique.size)
    label = side.astype(np.int8)
    confidence = np.where(label == 1, 0.95, 0.55).astype(np.float32)
    disagreement = np.zeros(depth.size, dtype=bool)
    disagreement[3::9] = True
    depth_error = np.zeros(depth.size, dtype=np.float32)
    depth_error[7] = 0.8
    exclude_large = depth_error > 0.5
    features = np.column_stack(
        [
            10.0 + side,
            20.0 + side,
            5.0 + side,
            100.0 - side,
            15.0 + side,
            np.where(label == 1, 0.3, 0.2),
        ]
    ).astype(np.float32)
    np.savez_compressed(
        path,
        depth=depth,
        side_index=side,
        label_presence_plus=label,
        label_confidence_plus=confidence,
        label_presence_minus_audit=label.copy(),
        plus_minus_disagreement=disagreement,
        valid_for_azimuthal_validation=np.ones(depth.size, dtype=bool),
        depth_match_error=depth_error,
        exclude_large_depth_match_error=exclude_large,
        sample_weight=np.where(label == 1, confidence, confidence * 0.1).astype(np.float32),
        features=features,
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
        transformed_features=features,
        transformed_feature_names=np.array(
            [
                "rms_energy",
                "peak_abs",
                "mean_abs",
                "early_energy",
                "late_energy",
                "late_over_early_ratio",
            ]
        ),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/mvp4b_sample_table.example.yaml").read_text())
    config["split"]["depth_block_size_ft"] = 20.0
    config["split"]["min_gap_ft"] = 0.0
    config["split"]["min_samples_per_class_per_fold"] = 1
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_rebuild_sample_weights_cli_writes_reweighted_table_and_report(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    sample_table = interim_dir / "baseline_sample_table_v001.npz"
    _write_sample_table(sample_table)
    weight_config = tmp_path / "mvp4b_sample_table.yaml"
    _write_config(weight_config)
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
            "scripts/06h_rebuild_sample_weights.py",
            "--paths",
            str(paths_config),
            "--sample-table-config",
            str(weight_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Sample weights rebuilt errors=0" in result.stdout
    output_npz = interim_dir / "baseline_sample_table_reweighted_v001.npz"
    output_json = reports_dir / "sample_weight_policy_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    capped = report["policy_summary"]["capped_class_balanced_confidence"]
    assert capped["candidate_effective_weight_fraction"] <= 0.6
    with np.load(output_npz, allow_pickle=False) as data:
        assert "sample_weight_confidence_only" in data.files
        assert "sample_weight_class_balanced_confidence" in data.files
        assert "sample_weight_capped_class_balanced_confidence" in data.files
        assert "sample_weight_unweighted" in data.files
        np.testing.assert_allclose(
            data["sample_weight"],
            data["sample_weight_capped_class_balanced_confidence"],
        )
        assert bool(data["no_final_labels"])
