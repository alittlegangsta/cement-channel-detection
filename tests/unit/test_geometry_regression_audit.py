from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_regression_audit import (
    GeometryRegressionAuditConfig,
    audit_geometry_regression,
)


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    depth = np.arange(40, dtype=np.float32)
    base = np.linspace(0.0, 1.0, depth.size, dtype=np.float32)
    targets = np.stack(
        [
            base * 0.4,
            base * 0.55,
            np.clip(base * 0.3 + 0.02, 0.0, 1.0),
            base * 0.75,
        ],
        axis=0,
    ).astype(np.float32)
    features = np.column_stack(
        [
            base,
            np.sin(base * np.pi).astype(np.float32),
        ]
    ).astype(np.float32)
    label_arrays = {
        "depth": depth,
        "geometry_kernel": np.asarray(
            [
                "r7_reference_point",
                "midpoint_window",
                "uniform_source_receiver_interval",
                "triangular_midpoint_weighted",
            ]
        ),
        "receiver_max": targets,
        "no_final_labels": np.asarray(True),
    }
    feature_arrays = {
        "depth": depth,
        "depth_level_xsi_features": features,
        "depth_level_xsi_feature_names": np.asarray(["energy", "sine_shape"]),
        "no_final_labels": np.asarray(True),
    }
    return label_arrays, feature_arrays


def test_audit_geometry_regression_reports_kernel_sanity_metrics() -> None:
    label_arrays, feature_arrays = _arrays()

    report, rows = audit_geometry_regression(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        config=GeometryRegressionAuditConfig(permutation_repeats=3),
    )

    assert report.errors == []
    assert report.best_kernel in {
        "r7_reference_point",
        "midpoint_window",
        "triangular_midpoint_weighted",
    }
    assert len(report.kernel_summaries) == 4
    assert len(rows) == 8
    assert report.kernel_summaries[0]["status"] == "runnable"
    assert report.kernel_summaries[0]["spearman_correlation"] is not None
    assert report.kernel_summaries[0]["cross_validated_mae"] is not None
    assert report.no_model_weights is True
    assert report.no_final_labels is True


def test_audit_geometry_regression_flags_leakage_feature_names() -> None:
    label_arrays, feature_arrays = _arrays()
    feature_arrays["depth_level_xsi_feature_names"] = np.asarray(["cast_zc_like", "energy"])

    report, _rows = audit_geometry_regression(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        config=GeometryRegressionAuditConfig(permutation_repeats=3),
    )

    assert any("Potential leakage" in warning for warning in report.warnings)
