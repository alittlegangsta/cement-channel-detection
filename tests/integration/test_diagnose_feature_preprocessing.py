from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def test_diagnose_feature_preprocessing_cli_outputs_reports_and_figures(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    features = np.ones((4, 2), dtype=np.float32)
    transformed = np.column_stack([np.log1p(features), features]).astype(np.float32)
    np.savez_compressed(
        interim_dir / "baseline_sample_table_v001.npz",
        features=features,
        feature_names=np.array(["rms_energy", "late_over_early_ratio"]),
        transformed_features=transformed,
        transformed_feature_names=np.array(
            [
                "log1p_rms_energy",
                "log1p_late_over_early_ratio",
                "robust_scaled_rms_energy",
                "robust_scaled_late_over_early_ratio",
            ]
        ),
        label_presence_plus=np.array([0, 0, 1, 1], dtype=np.int8),
        valid_for_azimuthal_validation=np.ones(4, dtype=bool),
        plus_minus_disagreement=np.zeros(4, dtype=bool),
        sample_weight=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        depth_match_error=np.zeros(4, dtype=np.float32),
        transform_stats_json=np.asarray(
            '{"rms_energy":{"clipped_count":0,"clip_low":0.0,"clip_high":2.0},'
            '"late_over_early_ratio":{"clipped_count":0,"clip_low":0.0,"clip_high":2.0}}'
        ),
        no_model_training=np.asarray(True),
        no_final_labels=np.asarray(True),
    )
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
            "scripts/06b_diagnose_feature_preprocessing.py",
            "--paths",
            str(paths_config),
            "--max-samples",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Feature preprocessing diagnostics errors=0" in result.stdout
    report = json.loads(
        (reports_dir / "feature_preprocessing_diagnostics_v001.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["diagnostics_version"] == "feature_preprocessing_diagnostics_v001"
    assert len(report["figures"]) == 5
    figure_dir = reports_dir / "feature_preprocessing_diagnostics_v001"
    assert (figure_dir / "01_feature_hist_raw_vs_log.png").read_bytes().startswith(b"\x89PNG")
    assert (reports_dir / "feature_preprocessing_diagnostics_v001.md").exists()
