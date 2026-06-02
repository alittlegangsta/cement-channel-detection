from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_regression_morphology_audit import (
    MORPHOLOGY_FIELDS,
    audit_geometry_regression_morphology,
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

    return {
        "depth": depth,
        "geometry_kernel": kernels,
        "receiver_index": np.arange(1, 14, dtype=np.int16),
        "weighted_channel_fraction_zc_lt_2p5": primary,
        "largest_connected_component_fraction": lcc,
        "max_azimuth_channel_fraction": max_az,
        "relative_anomaly_fraction": relative,
        "combined_channel_fraction": combined,
        "depth_label_confidence": confidence,
        "total_cell_count": total,
        "no_final_labels": np.asarray(True),
    }


def _feature_arrays(depth: np.ndarray) -> dict[str, np.ndarray]:
    base = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    return {
        "depth": depth,
        "depth_level_xsi_features": np.column_stack([base, base**2]).astype(np.float32),
        "depth_level_xsi_feature_names": np.asarray(
            ["near_far_ratio_mean_early_energy", "receiver_max_early_energy"]
        ),
        "no_final_labels": np.asarray(True),
    }


def test_morphology_audit_reports_relationships_and_selected_intervals() -> None:
    labels = _label_arrays()
    report, rows = audit_geometry_regression_morphology(
        label_arrays=labels,
        feature_arrays=_feature_arrays(labels["depth"]),
    )

    assert report.no_morphology_target_added is True
    assert set(report.morphology_fields) == set(MORPHOLOGY_FIELDS)
    assert report.fold_summaries
    assert report.simple_low_zc_relationships
    assert report.xsi_feature_relationships
    interval_types = {row["interval_type"] for row in report.selected_intervals}
    assert "local_severe_but_sparse" in interval_types
    assert "broad_moderate_anomaly" in interval_types
    assert "scattered_low_zc_noise_like" in interval_types
    assert "boundary_affected_interval" in interval_types
    assert any(row["section"] == "selected_interval" for row in rows)
