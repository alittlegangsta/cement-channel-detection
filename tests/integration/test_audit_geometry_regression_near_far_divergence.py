from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_inputs(root_dir: Path) -> None:
    interim = root_dir / "interim"
    reports = root_dir / "reports"
    interim.mkdir(parents=True)
    reports.mkdir()
    depth = np.arange(12, dtype=np.float32)
    kernels = np.asarray(
        [
            "r7_reference_point",
            "midpoint_window",
            "uniform_source_receiver_interval",
            "triangular_midpoint_weighted",
        ]
    )
    target = np.zeros((4, depth.size), dtype=np.float32)
    target[:, [2, 8]] = 0.5
    target[:, 4:6] = 0.15
    simple = np.zeros((4, depth.size, 13), dtype=np.float32)
    combined = simple.copy()
    lcc = simple.copy()
    combined[:, 2, :] = 0.30
    lcc[:, 2, :] = 0.25
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=kernels,
        receiver_max=target,
        weighted_channel_fraction_zc_lt_2p5=simple,
        combined_channel_fraction=combined,
        largest_connected_component_fraction=lcc,
        no_final_labels=np.asarray(True),
    )
    feature0 = np.asarray(
        [0.00, 0.01, 20.0, 0.02, 0.03, 0.04, 0.05, 0.06, 18.0, 0.07, 0.08, 0.09],
        dtype=np.float32,
    )
    feature1 = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    feature2 = np.linspace(1.0, 0.0, depth.size, dtype=np.float32)
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.column_stack([feature0, feature1, feature2]).astype(
            np.float32
        ),
        depth_level_xsi_feature_names=np.asarray(
            [
                "near_far_ratio_mean_early_energy",
                "near_far_ratio_mean_rms_energy",
                "near_far_ratio_mean_peak_abs",
            ]
        ),
        no_final_labels=np.asarray(True),
    )
    zc = np.ones((12, 4), dtype=np.float32) * 3.0
    zc[2, 0] = -1.0
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=depth,
        cast_zc=zc,
    )


def test_geometry_regression_near_far_divergence_cli_writes_reports(
    tmp_path: Path,
) -> None:
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06ar_audit_geometry_regression_near_far_divergence.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "denominator_small_derivable=False" in result.stdout
    output_json = (
        root_dir / "reports" / "geometry_regression_near_far_divergence_audit_v001.json"
    )
    output_md = (
        root_dir / "reports" / "geometry_regression_near_far_divergence_audit_v001.md"
    )
    output_csv = (
        root_dir / "reports" / "geometry_regression_near_far_divergence_audit_v001.csv"
    )
    assert output_json.exists()
    assert output_md.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["no_ratio_preprocessing_change"] is True
    assert report["denominator_small_derivable"] is False
    assert report["outlier_depths"]
