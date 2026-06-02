from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_regression_near_far_divergence_audit import (
    audit_geometry_regression_near_far_divergence,
)


def _label_arrays() -> dict[str, np.ndarray]:
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
    return {
        "depth": depth,
        "geometry_kernel": kernels,
        "receiver_max": target,
        "weighted_channel_fraction_zc_lt_2p5": simple,
        "combined_channel_fraction": combined,
        "largest_connected_component_fraction": lcc,
        "no_final_labels": np.asarray(True),
    }


def _feature_arrays(depth: np.ndarray) -> dict[str, np.ndarray]:
    feature0 = np.asarray(
        [0.00, 0.01, 20.0, 0.02, 0.03, 0.04, 0.05, 0.06, 18.0, 0.07, 0.08, 0.09],
        dtype=np.float32,
    )
    feature1 = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    feature2 = np.linspace(1.0, 0.0, depth.size, dtype=np.float32)
    return {
        "depth": depth,
        "depth_level_xsi_features": np.column_stack([feature0, feature1, feature2]).astype(
            np.float32
        ),
        "depth_level_xsi_feature_names": np.asarray(
            [
                "near_far_ratio_mean_early_energy",
                "near_far_ratio_mean_rms_energy",
                "near_far_ratio_mean_peak_abs",
            ]
        ),
        "no_final_labels": np.asarray(True),
    }


def _cast_arrays() -> dict[str, np.ndarray]:
    zc = np.ones((12, 4), dtype=np.float32) * 3.0
    zc[2, 0] = -1.0
    zc[8, 1] = 15.0
    return {
        "cast_depth": np.arange(12, dtype=np.float32),
        "cast_zc": zc,
    }


def test_near_far_divergence_audit_reports_outliers_and_overlaps() -> None:
    labels = _label_arrays()
    report, rows = audit_geometry_regression_near_far_divergence(
        label_arrays=labels,
        feature_arrays=_feature_arrays(labels["depth"]),
        cast_arrays=_cast_arrays(),
    )

    assert report.denominator_small_derivable is False
    assert report.no_ratio_preprocessing_change is True
    assert report.no_ratio_clip is True
    assert report.no_outlier_removal is True
    assert len(report.global_correlations) == 12
    assert len(report.per_fold_correlations) == 36
    assert any(row["extreme_value_count"] > 0 for row in report.feature_quantiles)
    assert any(row["negative_zc_overlap"] for row in report.outlier_depths)
    assert any(row["upper_zc_review_overlap"] for row in report.outlier_depths)
    assert any(row["morphology_sensitive_any_kernel"] for row in report.outlier_depths)
    assert any(row["section"] == "divergence_summary" for row in rows)
