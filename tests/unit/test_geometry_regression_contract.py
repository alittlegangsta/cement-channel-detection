from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_regression_audit import (
    GeometryRegressionAuditConfig,
    audit_geometry_regression,
)
from cement_channel.evaluation.geometry_regression_contract import (
    build_contract_invariants,
    build_contract_inventory,
)
from cement_channel.labels.geometry_aware_regression_labels import (
    build_geometry_aware_regression_label_report_from_arrays,
)


def _label_arrays() -> dict[str, np.ndarray]:
    depth = np.arange(6, dtype=np.float32)
    kernels = np.asarray(
        [
            "r7_reference_point",
            "midpoint_window",
            "uniform_source_receiver_interval",
            "triangular_midpoint_weighted",
        ]
    )
    primary = np.zeros((4, depth.size, 13), dtype=np.float32)
    for index in range(4):
        primary[index] = np.linspace(0.0, 0.2 + index * 0.02, depth.size)[:, None]
    return {
        "depth": depth,
        "geometry_kernel": kernels,
        "receiver_index": np.arange(1, 14, dtype=np.int16),
        "weighted_channel_fraction_zc_lt_2p5": primary,
        "raw_channel_fraction_zc_lt_2p5": primary,
        "relative_anomaly_fraction": primary * 0.0,
        "combined_channel_fraction": primary,
        "largest_connected_component_fraction": primary,
        "max_azimuth_channel_fraction": primary,
        "depth_label_confidence": np.ones_like(primary),
        "candidate_cell_count": np.ones((4, depth.size, 13), dtype=np.int32),
        "total_cell_count": np.ones((4, depth.size, 13), dtype=np.int32),
        "receiver_mean": np.mean(primary, axis=2),
        "receiver_max": np.max(primary, axis=2),
        "receiver_p90": np.percentile(primary, 90.0, axis=2).astype(np.float32),
        "receiver_std": np.std(primary, axis=2),
        "full_360_fraction": np.max(primary, axis=2),
        "derived_positive_at_fraction_0p01": np.max(primary, axis=2) >= 0.01,
        "derived_positive_at_fraction_0p05": np.max(primary, axis=2) >= 0.05,
        "derived_positive_at_fraction_0p10": np.max(primary, axis=2) >= 0.10,
        "primary_target": np.asarray("weighted_channel_fraction_zc_lt_2p5"),
        "raw_zc_source_file": np.asarray("cast_label_input_v001.npz"),
        "raw_zc_source_field": np.asarray("cast_zc"),
        "raw_zc_finite_ratio": np.asarray(1.0, dtype=np.float32),
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


def test_contract_inventory_and_invariants_pass_for_clarified_reports() -> None:
    labels = _label_arrays()
    features = _feature_arrays(labels["depth"])
    stage9 = build_geometry_aware_regression_label_report_from_arrays(
        arrays=labels,
        previous_report={
            "raw_zc_source": {"source_file": "cast_label_input_v001.npz"},
            "geometry_sign_status": {"depth_axis_sign": -1},
        },
    ).to_dict()
    stage10, _rows = audit_geometry_regression(
        label_arrays=labels,
        feature_arrays=features,
        config=GeometryRegressionAuditConfig(permutation_repeats=2),
    )

    inventory = build_contract_inventory(
        label_arrays=labels,
        feature_arrays=features,
        stage9_report=stage9,
        stage10_report=stage10.to_dict(),
        inputs={},
    )
    invariants = build_contract_invariants(
        label_arrays=labels,
        feature_arrays=features,
        stage9_report=stage9,
        stage10_report=stage10.to_dict(),
        inputs={},
    )

    assert inventory.issue_classification == "report_ambiguity_only"
    assert invariants.passed is True
    assert not invariants.errors
