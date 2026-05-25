from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_baseline_sample_table_cli_writes_outputs(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    features_dir = root_dir / "features"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    features_dir.mkdir()
    reports_dir.mkdir()
    presence = np.array([[1, 0], [1, 0]], dtype=np.int8)
    np.savez_compressed(
        interim_dir / "xsi_label_samples_v001.npz",
        xsi_depth=np.array([100.0, 99.0], dtype=np.float32),
        xsi_side_azimuth_deg=np.array([0.0, 45.0], dtype=np.float32),
        label_presence_plus=presence,
        label_severity_plus=np.where(presence == 1, 2, 0).astype(np.int8),
        label_confidence_plus=np.full((2, 2), 0.8, dtype=np.float32),
        label_presence_minus_audit=presence,
        plus_minus_disagreement=np.zeros((2, 2), dtype=bool),
        orientation_confidence=np.ones((2, 2), dtype=np.float32),
        valid_for_azimuthal_validation=np.ones((2, 2), dtype=bool),
        valid_for_non_azimuthal_summary=np.ones((2, 2), dtype=bool),
        cast_depth_mismatch=np.zeros(2, dtype=np.float32),
        no_final_labels=np.asarray(True),
    )
    np.savez_compressed(
        features_dir / "xsi_basic_features_v001.npz",
        xsi_basic_features_by_side=np.ones((2, 2, 6), dtype=np.float32),
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
        no_model_training=np.asarray(True),
    )
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {interim_dir}",
                f"  features: {features_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06a_build_baseline_sample_table.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Baseline sample table errors=0" in result.stdout
    output_npz = interim_dir / "baseline_sample_table_v001.npz"
    with np.load(output_npz) as data:
        assert data["features"].shape == (4, 6)
        assert data["transformed_features"].shape == (4, 12)
        assert bool(data["no_model_training"].reshape(()))
        assert bool(data["no_final_labels"].reshape(()))
    report = json.loads(
        (reports_dir / "baseline_sample_table_report_v001.json").read_text(encoding="utf-8")
    )
    assert report["sample_table_version"] == "baseline_sample_table_v001"
    assert report["shape"]["samples"] == 4
    assert (reports_dir / "baseline_sample_table_report_v001.md").exists()
