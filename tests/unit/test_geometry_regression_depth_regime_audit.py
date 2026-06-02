from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_regression_depth_regime_audit import (
    audit_geometry_regression_depth_regimes,
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
    target[:, :4] = 0.5
    target[:, 4:8] = 0.05
    target[:, 8:] = 0.04
    morphology = np.repeat(target[:, :, None], 13, axis=2)
    confidence = np.ones_like(morphology, dtype=np.float32)
    return {
        "depth": depth,
        "geometry_kernel": kernels,
        "receiver_max": target,
        "largest_connected_component_fraction": morphology,
        "max_azimuth_channel_fraction": morphology,
        "relative_anomaly_fraction": morphology,
        "combined_channel_fraction": morphology,
        "depth_label_confidence": confidence,
        "no_final_labels": np.asarray(True),
    }


def _feature_arrays(depth: np.ndarray) -> dict[str, np.ndarray]:
    feature0 = np.r_[np.ones(4) * 10.0, np.zeros(8)].astype(np.float32)
    feature1 = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    return {
        "depth": depth,
        "depth_level_xsi_features": np.column_stack([feature0, feature1]).astype(np.float32),
        "depth_level_xsi_feature_names": np.asarray(
            ["near_far_ratio_mean_early_energy", "receiver_max_early_energy"]
        ),
        "no_final_labels": np.asarray(True),
    }


def _cast_arrays() -> dict[str, np.ndarray]:
    zc = np.ones((12, 4), dtype=np.float32) * 3.0
    zc[0, 0] = -1.0
    return {
        "cast_depth": np.arange(12, dtype=np.float32),
        "cast_zc": zc,
    }


def test_depth_regime_audit_flags_target_and_feature_shift() -> None:
    labels = _label_arrays()
    report, rows = audit_geometry_regression_depth_regimes(
        label_arrays=labels,
        feature_arrays=_feature_arrays(labels["depth"]),
        cast_arrays=_cast_arrays(),
    )

    assert report.formal_cv_protocol_changed is False
    assert report.no_cv_split_change is True
    assert "target_fold_regime_shift" in report.regime_shift_flags
    assert "feature_fold_regime_shift" in report.regime_shift_flags
    assert report.fold_invalid_zc_summaries[0]["negative_zc_count"] == 1
    assert report.selected_intervals
    assert any(row["section"] == "fold_target_summary" for row in rows)
