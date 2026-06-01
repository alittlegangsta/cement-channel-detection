from __future__ import annotations

import numpy as np

from cement_channel.visualization.geometry_regression_review import (
    build_kernel_comparison_rows,
    build_selected_interval_rows,
)


def _labels() -> dict[str, np.ndarray]:
    depth = np.arange(10, dtype=np.float32)
    receiver_max = np.stack(
        [
            np.linspace(0.0, 0.4, depth.size),
            np.linspace(0.0, 0.5, depth.size),
            np.linspace(0.0, 0.3, depth.size),
            np.linspace(0.0, 0.8, depth.size),
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "depth": depth,
        "geometry_kernel": np.asarray(
            [
                "r7_reference_point",
                "midpoint_window",
                "uniform_source_receiver_interval",
                "triangular_midpoint_weighted",
            ]
        ),
        "receiver_max": receiver_max,
        "depth_label_confidence": np.ones((4, depth.size, 13), dtype=np.float32),
        "no_final_labels": np.asarray(True),
    }


def _features() -> dict[str, np.ndarray]:
    depth = np.arange(10, dtype=np.float32)
    return {
        "depth": depth,
        "depth_level_xsi_features": np.column_stack([depth, depth[::-1]]).astype(np.float32),
        "depth_level_xsi_feature_names": np.asarray(["energy", "reverse"]),
        "no_final_labels": np.asarray(True),
    }


def test_build_kernel_comparison_rows_extracts_audit_fields() -> None:
    audit = {
        "kernel_summaries": [
            {
                "geometry_kernel": "r7_reference_point",
                "sample_count": 10,
                "real_minus_permutation_margin": 0.1,
                "fold_stability": 1.0,
            }
        ]
    }

    rows = build_kernel_comparison_rows(audit)

    assert rows[0]["geometry_kernel"] == "r7_reference_point"
    assert rows[0]["real_minus_permutation_margin"] == 0.1


def test_build_selected_interval_rows_includes_expected_review_types() -> None:
    rows = build_selected_interval_rows(
        labels=_labels(),
        features=_features(),
        primary_kernel="triangular_midpoint_weighted",
    )

    review_types = {row["review_type"] for row in rows}
    assert "high_fraction_interval" in review_types
    assert "low_fraction_interval" in review_types
    assert "kernel_sensitive_interval" in review_types
    assert all(row["primary_kernel"] == "triangular_midpoint_weighted" for row in rows)
