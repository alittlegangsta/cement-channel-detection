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
    depth = np.arange(9, dtype=np.float32)
    kernels = np.asarray(
        [
            "r7_reference_point",
            "midpoint_window",
            "uniform_source_receiver_interval",
            "triangular_midpoint_weighted",
        ]
    )
    base = np.linspace(0.0, 0.4, depth.size, dtype=np.float32)
    receiver_offsets = np.linspace(0.0, 0.12, 13, dtype=np.float32)
    primary = np.stack(
        [
            np.clip(base[:, None] + receiver_offsets[None, :] + index * 0.01, 0.0, 1.0)
            for index in range(4)
        ],
        axis=0,
    )
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=kernels,
        receiver_index=np.arange(1, 14, dtype=np.int16),
        weighted_channel_fraction_zc_lt_2p5=primary,
        receiver_mean=np.mean(primary, axis=2),
        receiver_p90=np.percentile(primary, 90.0, axis=2).astype(np.float32),
        receiver_max=np.max(primary, axis=2),
        receiver_std=np.std(primary, axis=2),
        full_360_fraction=np.clip(np.max(primary, axis=2) + 0.1, 0.0, 1.0),
        no_final_labels=np.asarray(True),
    )
    features = np.column_stack([base, base**2]).astype(np.float32)
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=features,
        depth_level_xsi_feature_names=np.asarray(
            ["near_far_ratio_mean_early_energy", "receiver_max_early_energy"]
        ),
        no_final_labels=np.asarray(True),
    )


def test_geometry_regression_target_view_cli_writes_reports(tmp_path: Path) -> None:
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
            "scripts/06ao_audit_geometry_regression_target_views.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rows=20" in result.stdout
    output_json = root_dir / "reports" / "geometry_regression_target_view_audit_v001.json"
    output_md = root_dir / "reports" / "geometry_regression_target_view_audit_v001.md"
    output_csv = root_dir / "reports" / "geometry_regression_target_view_audit_v001.csv"
    assert output_json.exists()
    assert output_md.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["primary_target_view_changed"] is False
    assert len(report["view_summaries"]) == 20
