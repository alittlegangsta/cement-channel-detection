from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

KERNELS = np.asarray(
    [
        "r7_reference_point",
        "midpoint_window",
        "uniform_source_receiver_interval",
        "triangular_midpoint_weighted",
    ]
)


def _write_inputs(root_dir: Path) -> None:
    interim = root_dir / "interim"
    reports = root_dir / "reports"
    interim.mkdir(parents=True)
    reports.mkdir()
    depth = np.arange(96.0, 108.5, 0.5, dtype=np.float32)
    target_depth = np.asarray([100.0, 103.0, 106.0], dtype=np.float32)
    receiver_max = np.asarray(
        [
            [0.1, 0.3, 0.0],
            [0.2, 0.35, 0.05],
            [0.15, 0.25, 0.04],
            [0.25, 0.6, 0.02],
        ],
        dtype=np.float32,
    )
    weighted = np.repeat(receiver_max[:, :, None], 13, axis=2)
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=target_depth,
        geometry_kernel=KERNELS,
        receiver_max=receiver_max,
        weighted_channel_fraction_zc_lt_2p5=weighted,
        depth_label_confidence=np.ones((4, target_depth.size, 13), dtype=np.float32),
        no_final_labels=np.asarray(True),
    )
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=target_depth,
        depth_level_xsi_features=np.asarray(
            [[0.1, 0.8], [0.9, 0.2], [0.2, 0.5]],
            dtype=np.float32,
        ),
        depth_level_xsi_feature_names=np.asarray(["energy", "attenuation"]),
        no_final_labels=np.asarray(True),
    )
    cast_zc = np.full((depth.size, 6), 4.0, dtype=np.float32)
    cast_zc[(depth >= 103.0) & (depth <= 104.0), 2:4] = 2.0
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=depth,
        cast_azimuth_deg=np.arange(6, dtype=np.float32) * 60.0,
        cast_zc=cast_zc,
    )
    np.savez_compressed(
        interim / "cast_zc_baseline_v001.npz",
        cast_depth=depth,
        cast_azimuth_deg=np.arange(6, dtype=np.float32) * 60.0,
        relative_drop=np.where(cast_zc < 2.5, 0.5, 0.0).astype(np.float32),
    )
    np.savez_compressed(
        interim / "geometry_aware_depth_labels_v001.npz",
        has_channel_any=np.asarray([False, True, False]),
    )
    reports.joinpath("geometry_regression_audit_v001.json").write_text(
        json.dumps(
            {
                "best_kernel": "triangular_midpoint_weighted",
                "no_final_labels": True,
                "kernel_summaries": [
                    {
                        "geometry_kernel": str(kernel),
                        "sample_count": 3,
                        "zero_fraction": 0.0,
                        "nonzero_fraction": 1.0,
                        "spearman_correlation": 0.8,
                        "permutation_spearman_correlation": 0.2,
                        "real_minus_permutation_margin": 0.6,
                        "cross_validated_mae": 0.1,
                        "cross_validated_r2_sanity": 0.2,
                        "fold_stability": 1.0,
                        "depends_on_5700_band": False,
                    }
                    for kernel in KERNELS
                ],
            }
        ),
        encoding="utf-8",
    )


def test_generate_geometry_regression_review_cli_writes_outputs(tmp_path: Path) -> None:
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
            "scripts/06aj_generate_geometry_regression_review.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Geometry regression review errors=0" in result.stdout
    output_dir = root_dir / "reports" / "geometry_regression_manual_review_v001"
    assert (output_dir / "review_summary.md").exists()
    assert (output_dir / "kernel_comparison.csv").exists()
    assert (output_dir / "selected_interval_review_list.json").exists()
    summary = json.loads(
        (output_dir / "geometry_regression_review_summary_v001.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["figure_count"] == 10
    assert summary["no_final_labels"] is True
