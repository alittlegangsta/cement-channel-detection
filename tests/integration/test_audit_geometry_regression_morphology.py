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
    shape = (4, depth.size, 13)
    primary = np.zeros(shape, dtype=np.float32)
    combined = np.zeros(shape, dtype=np.float32)
    lcc = np.zeros(shape, dtype=np.float32)
    max_az = np.zeros(shape, dtype=np.float32)
    relative = np.zeros(shape, dtype=np.float32)
    confidence = np.ones(shape, dtype=np.float32)
    total = np.ones(shape, dtype=np.int32) * 180
    primary[:, 2, 0] = 0.4
    combined[:, 2, 0] = 0.4
    max_az[:, 2, :] = 0.6
    primary[:, 5:7, :] = 0.18
    combined[:, 5:7, :] = 0.30
    lcc[:, 5:7, :] = 0.20
    relative[:, 5:7, :] = 0.22
    primary[:, 8, :] = 0.05
    combined[:, 8, :] = 0.05
    lcc[:, 8, :] = 0.01
    max_az[:, 8, :] = 0.10
    confidence[:, 10, :] = 0.0
    total[:, 10, :] = 0
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=kernels,
        receiver_index=np.arange(1, 14, dtype=np.int16),
        weighted_channel_fraction_zc_lt_2p5=primary,
        largest_connected_component_fraction=lcc,
        max_azimuth_channel_fraction=max_az,
        relative_anomaly_fraction=relative,
        combined_channel_fraction=combined,
        depth_label_confidence=confidence,
        total_cell_count=total,
        no_final_labels=np.asarray(True),
    )
    base = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.column_stack([base, base**2]).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(
            ["near_far_ratio_mean_early_energy", "receiver_max_early_energy"]
        ),
        no_final_labels=np.asarray(True),
    )


def test_geometry_regression_morphology_cli_writes_reports(tmp_path: Path) -> None:
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
            "scripts/06ap_audit_geometry_regression_morphology.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "selected_intervals=" in result.stdout
    output_json = root_dir / "reports" / "geometry_regression_morphology_audit_v001.json"
    output_md = root_dir / "reports" / "geometry_regression_morphology_audit_v001.md"
    output_csv = root_dir / "reports" / "geometry_regression_morphology_audit_v001.csv"
    assert output_json.exists()
    assert output_md.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["no_morphology_target_added"] is True
    assert report["selected_intervals"]
