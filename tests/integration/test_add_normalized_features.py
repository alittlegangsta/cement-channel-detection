from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_sample_table(path: Path) -> None:
    depth_unique = np.arange(0.0, 80.0, 4.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 4)
    side = np.tile(np.arange(4, dtype=np.int16), depth_unique.size)
    raw = np.column_stack(
        [
            10.0 + side,
            20.0 + side,
            5.0 + side,
            100.0 - side,
            15.0 + side,
            0.1 + 0.05 * side,
        ]
    ).astype(np.float32)
    np.savez_compressed(
        path,
        sample_id=np.arange(depth.size, dtype=np.int64),
        depth=depth,
        side_index=side,
        label_presence_plus=(side == 1).astype(np.int8),
        label_confidence_plus=np.ones(depth.size, dtype=np.float32),
        label_presence_minus_audit=(side == 1).astype(np.int8),
        plus_minus_disagreement=np.zeros(depth.size, dtype=bool),
        valid_for_azimuthal_validation=np.ones(depth.size, dtype=bool),
        depth_match_error=np.zeros(depth.size, dtype=np.float32),
        sample_weight=np.ones(depth.size, dtype=np.float32),
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
        transformed_features=np.log1p(raw).astype(np.float32),
        transformed_feature_names=np.array(
            [
                "log1p_rms_energy",
                "log1p_peak_abs",
                "log1p_mean_abs",
                "log1p_early_energy",
                "log1p_late_energy",
                "log1p_late_over_early_ratio",
            ]
        ),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def test_add_normalized_features_cli_writes_enhanced_table_and_report(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    sample_table = interim_dir / "baseline_sample_table_reweighted_v001.npz"
    _write_sample_table(sample_table)
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
            "scripts/06i_add_normalized_features.py",
            "--paths",
            str(paths_config),
            "--rolling-window-samples",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Enhanced features generated errors=0" in result.stdout
    output_npz = interim_dir / "baseline_sample_table_enhanced_v001.npz"
    output_json = reports_dir / "enhanced_feature_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["used_label_information_for_features"] is False
    assert report["enhanced_transformed_feature_finite_ratio"] == 1.0
    with np.load(output_npz, allow_pickle=False) as data:
        assert data["transformed_features"].shape[1] == 32
        assert data["base_transformed_feature_count"] == 6
        assert "transformed_features_original" in data.files
        assert bool(data["no_final_labels"])
