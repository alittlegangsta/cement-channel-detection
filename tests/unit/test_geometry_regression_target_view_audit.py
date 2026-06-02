from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_regression_target_view_audit import (
    TARGET_VIEWS,
    audit_geometry_regression_target_views,
)


def _label_arrays() -> dict[str, np.ndarray]:
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
    return {
        "depth": depth,
        "geometry_kernel": kernels,
        "receiver_index": np.arange(1, 14, dtype=np.int16),
        "weighted_channel_fraction_zc_lt_2p5": primary,
        "receiver_mean": np.mean(primary, axis=2),
        "receiver_p90": np.percentile(primary, 90.0, axis=2).astype(np.float32),
        "receiver_max": np.max(primary, axis=2),
        "receiver_std": np.std(primary, axis=2),
        "full_360_fraction": np.clip(np.max(primary, axis=2) + 0.1, 0.0, 1.0),
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


def test_target_view_audit_reports_all_existing_views_without_primary_change() -> None:
    labels = _label_arrays()
    report, rows = audit_geometry_regression_target_views(
        label_arrays=labels,
        feature_arrays=_feature_arrays(labels["depth"]),
    )

    assert report.primary_target_view_changed is False
    assert report.no_final_labels is True
    assert len(rows) == len(TARGET_VIEWS) * 4
    assert {row["target_view"] for row in rows} == set(TARGET_VIEWS)
    assert all("permutation_margin" in row for row in rows)
    receiver_max_rows = [row for row in rows if row["target_view"] == "receiver_max"]
    assert any(row["single_receiver_extreme_sensitivity_warning"] for row in receiver_max_rows)
