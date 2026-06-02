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
    depth = np.linspace(2350.0, 5750.0, 30, dtype=np.float32)
    kernels = np.asarray(
        [
            "r7_reference_point",
            "midpoint_window",
            "uniform_source_receiver_interval",
            "triangular_midpoint_weighted",
        ]
    )
    base = np.r_[np.ones(10) * 0.35, np.ones(10) * 0.08, np.ones(10) * 0.06].astype(
        np.float32
    )
    targets = np.vstack([base, base * 0.9, base * 0.8, base * 0.85]).astype(np.float32)
    cube = np.repeat(targets[:, :, None], 13, axis=2)
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=kernels,
        receiver_mean=targets * 0.7,
        receiver_p90=targets * 0.9,
        receiver_max=targets,
        full_360_fraction=np.clip(targets * 1.2, 0.0, 1.0),
        receiver_std=targets * 0.2,
        weighted_channel_fraction_zc_lt_2p5=cube,
        largest_connected_component_fraction=cube,
        max_azimuth_channel_fraction=np.clip(cube * 3.0, 0.0, 1.0),
        relative_anomaly_fraction=cube * 0.8,
        combined_channel_fraction=cube,
        depth_label_confidence=np.ones_like(cube, dtype=np.float32),
        total_cell_count=np.ones_like(cube, dtype=np.int32),
        no_final_labels=np.asarray(True),
    )
    feature0 = np.linspace(0.0, 0.1, depth.size, dtype=np.float32)
    feature0[2] = 10.0
    feature1 = np.linspace(0.05, 0.2, depth.size, dtype=np.float32)
    feature2 = np.linspace(0.2, 0.05, depth.size, dtype=np.float32)
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
        feature_group_names=np.asarray(["near_far_receiver_ratio"]),
        feature_group_counts_json=np.asarray(
            json.dumps({"near_far_receiver_ratio": 3})
        ),
        no_final_labels=np.asarray(True),
    )
    cast_depth = np.linspace(2350.0, 5750.0, 90, dtype=np.float32)
    cast_zc = np.ones((90, 8), dtype=np.float32) * 3.0
    cast_zc[2, 0] = -1.0
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=cast_depth,
        cast_zc=cast_zc,
        orientation_confidence=np.linspace(1.0, 0.6, cast_depth.size).astype(np.float32),
        inc_deg=np.linspace(3.0, 8.0, cast_depth.size).astype(np.float32),
        relbearing_deg=np.linspace(0.0, 90.0, cast_depth.size).astype(np.float32),
    )


def test_generate_geometry_regression_depth_regime_review_cli(tmp_path: Path) -> None:
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
            "scripts/06at_generate_geometry_regression_depth_regime_review.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "formal_cv_protocol_changed=False" in result.stdout
    assert (
        root_dir / "reports" / "geometry_regression_depth_regime_audit_v001.json"
    ).exists()
    assert (
        root_dir / "reports" / "geometry_regression_depth_regime_bins_v001.csv"
    ).exists()
    assert (
        root_dir / "reports" / "geometry_regression_depth_regime_boundaries_v001.csv"
    ).exists()
    review_dir = root_dir / "reports" / "geometry_regression_depth_regime_review_v001"
    assert (review_dir / "review_summary.md").exists()
    assert (review_dir / "candidate_regime_boundaries.png").exists()
    decision = json.loads(
        (
            root_dir / "reports" / "geometry_regression_depth_regime_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["decision"] == "stop_request_formal_regime_policy_approval"
