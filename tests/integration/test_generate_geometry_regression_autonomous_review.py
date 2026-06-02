from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FIGURE_NAMES = (
    "raw_zc_invalid_vs_depth.png",
    "raw_zc_invalid_vs_azimuth.png",
    "invalid_zc_counterfactual_target_delta.png",
    "target_view_comparison_vs_depth.png",
    "target_view_distribution_by_fold.png",
    "target_view_cv_summary.png",
    "morphology_metrics_vs_depth.png",
    "receiver_fraction_heatmap.png",
    "near_far_ratio_outliers_vs_depth.png",
    "pearson_spearman_divergence_summary.png",
    "fold_boundary_overview.png",
)


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
    base = np.linspace(0.05, 0.35, depth.size, dtype=np.float32)
    target = np.vstack([base, base * 0.8, base * 0.7, base * 0.9]).astype(np.float32)
    receiver_mean = target * 0.7
    receiver_p90 = target * 0.9
    receiver_std = target * 0.1
    full_360 = np.clip(target * 1.2, 0.0, 1.0)
    receiver_cube = np.repeat(target[:, :, None], 13, axis=2)
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=kernels,
        receiver_mean=receiver_mean,
        receiver_p90=receiver_p90,
        receiver_max=target,
        receiver_std=receiver_std,
        full_360_fraction=full_360,
        weighted_channel_fraction_zc_lt_2p5=receiver_cube,
        largest_connected_component_fraction=receiver_cube,
        combined_channel_fraction=receiver_cube,
        relative_anomaly_fraction=receiver_cube,
        no_final_labels=np.asarray(True),
    )
    ratio = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    ratio[3] = 10.0
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.column_stack([ratio]).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["near_far_ratio_mean_early_energy"]),
        no_final_labels=np.asarray(True),
    )
    cast_depth = np.linspace(2350.0, 5750.0, 90, dtype=np.float32)
    cast_zc = np.ones((cast_depth.size, 8), dtype=np.float32) * 3.0
    cast_zc[3, 0] = -1.0
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=cast_depth,
        cast_zc=cast_zc,
        cast_azimuth_deg=np.linspace(0.0, 315.0, 8, dtype=np.float32),
    )
    _write_reports(reports)


def _write_reports(reports: Path) -> None:
    cast_qc = {
        "negative_zc_target_influence_significant": False,
        "global_cell_counts": {
            "zc_lt_0_count": 1,
            "nonfinite_count": 0,
        },
        "depth_summary": {
            "first_negative_depth": 2360.0,
            "last_negative_depth": 2360.0,
        },
        "kernel_summaries": [
            {
                "geometry_kernel": "triangular_midpoint_weighted",
                "mean_abs_delta": 0.001,
                "max_abs_delta": 0.01,
                "no_overlap_region_fraction": 0.02,
            }
        ],
    }
    target_view = {
        "view_summaries": [
            {
                "target_view": name,
                "geometry_kernel": "triangular_midpoint_weighted",
                "cv_r2": -1.0,
                "single_receiver_extreme_sensitivity_warning": name == "receiver_max",
                "saturation_warning": name == "full_360_fraction",
            }
            for name in (
                "receiver_mean",
                "receiver_p90",
                "receiver_max",
                "full_360_fraction",
                "receiver_std",
            )
        ]
    }
    morphology = {
        "simple_low_zc_relationships": [
            {
                "geometry_kernel": "triangular_midpoint_weighted",
                "morphology_field": "largest_connected_component_fraction",
                "pearson_vs_simple_low_zc_mean": 0.9,
            }
        ],
        "selected_intervals": [
            {
                "interval_type": "morphology_sensitive",
                "geometry_kernel": "triangular_midpoint_weighted",
                "depth_min": 2400.0,
                "depth_max": 2450.0,
                "score": 0.4,
            }
        ],
    }
    depth_regime = {
        "regime_shift_flags": [
            "target_fold_regime_shift",
            "feature_fold_regime_shift",
            "morphology_fold_shift",
        ],
        "target_shift_summary": [
            {"geometry_kernel": "r7_reference_point", "shift_flag": True},
            {"geometry_kernel": "midpoint_window", "shift_flag": True},
            {"geometry_kernel": "uniform_source_receiver_interval", "shift_flag": True},
            {"geometry_kernel": "triangular_midpoint_weighted", "shift_flag": True},
        ],
        "selected_intervals": [
            {
                "interval_type": "fold0_high_target",
                "geometry_kernel": "triangular_midpoint_weighted",
                "depth_min": 2350.0,
                "depth_max": 3500.0,
            },
            {
                "interval_type": "fold1_low_target",
                "geometry_kernel": "triangular_midpoint_weighted",
                "depth_min": 3500.0,
                "depth_max": 4700.0,
            },
            {
                "interval_type": "fold2_low_target",
                "geometry_kernel": "triangular_midpoint_weighted",
                "depth_min": 4700.0,
                "depth_max": 5750.0,
            },
        ],
    }
    near_far = {
        "denominator_small_derivable": False,
        "global_correlations": [
            {
                "geometry_kernel": "r7_reference_point",
                "feature_name": "near_far_ratio_mean_early_energy",
                "pearson": 0.5,
                "spearman": 0.01,
            },
            {
                "geometry_kernel": "triangular_midpoint_weighted",
                "feature_name": "near_far_ratio_mean_early_energy",
                "pearson": 0.4,
                "spearman": 0.02,
            },
        ],
        "feature_quantiles": [
            {
                "feature_name": "near_far_ratio_mean_early_energy",
                "extreme_value_count": 1,
            }
        ],
        "outlier_depths": [
            {
                "feature_name": "near_far_ratio_mean_early_energy",
                "depth_index": 3,
                "depth": 2500.0,
                "feature_value": 10.0,
                "robust_z_score": 9.0,
            }
        ],
        "divergence_summary": [
            {
                "global_divergence_flag": True,
                "fold_divergence_flags": [False, False, False],
            }
        ],
    }
    payloads = {
        "cast_zc_physical_qc_v001.json": cast_qc,
        "geometry_regression_target_view_audit_v001.json": target_view,
        "geometry_regression_morphology_audit_v001.json": morphology,
        "geometry_regression_depth_regime_audit_v001.json": depth_regime,
        "geometry_regression_near_far_divergence_audit_v001.json": near_far,
    }
    for name, payload in payloads.items():
        (reports / name).write_text(json.dumps(payload), encoding="utf-8")


def test_generate_geometry_regression_autonomous_review_cli(tmp_path: Path) -> None:
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
            "scripts/06as_generate_geometry_regression_autonomous_review.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=stop_request_depth_regime_stratification_approval" in result.stdout
    review_dir = root_dir / "reports" / "geometry_regression_autonomous_review_v001"
    for name in FIGURE_NAMES:
        assert (review_dir / name).exists()
    assert (review_dir / "selected_diagnostic_intervals.csv").exists()
    assert (review_dir / "selected_diagnostic_intervals.json").exists()
    assert (review_dir / "review_summary.md").exists()
    decision_json = root_dir / "reports" / "geometry_regression_autonomous_decision.json"
    decision = json.loads(decision_json.read_text(encoding="utf-8"))
    assert decision["decision"] == "stop_request_depth_regime_stratification_approval"
