from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_regression_audit import (
    GeometryRegressionAuditConfig,
    audit_geometry_regression,
)
from cement_channel.evaluation.geometry_regression_bounded_triage import (
    build_bounded_triage,
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
        "receiver_mean": np.mean(primary, axis=2),
        "receiver_max": np.max(primary, axis=2),
        "receiver_p90": np.percentile(primary, 90.0, axis=2).astype(np.float32),
        "full_360_fraction": np.max(primary, axis=2),
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


def test_bounded_triage_reports_mixed_unresolved_for_stop_audit() -> None:
    labels = _label_arrays()
    features = _feature_arrays(labels["depth"])
    audit, feature_rows = audit_geometry_regression(
        label_arrays=labels,
        feature_arrays=features,
        config=GeometryRegressionAuditConfig(permutation_repeats=2),
    )
    audit_dict = audit.to_dict()
    audit_dict["recommendation"] = "stop"

    report, rows = build_bounded_triage(
        label_arrays=labels,
        feature_arrays=features,
        inventory_report={
            "inventory_version": "v",
            "issue_classification": "report_ambiguity_only",
        },
        invariants_report={"passed": True},
        audit_report=audit_dict,
        feature_correlation_rows=[
            {
                "geometry_kernel": row["geometry_kernel"],
                "target_view": row["target_view"],
                "feature_name": row["feature_name"],
                "pearson": row["pearson_correlation"],
                "spearman": row["spearman_correlation"],
                "abs_pearson_rank": row["abs_pearson_rank"],
                "abs_spearman_rank": row["abs_spearman_rank"],
                "pearson_spearman_divergence_flag": str(
                    row["pearson_spearman_divergence_flag"]
                ),
            }
            for row in feature_rows
        ],
        inputs={},
    )

    assert report.errors == []
    assert report.eligibility["contract_invariants_passed"] is True
    assert report.no_model_training is True
    assert report.root_cause_classification in {
        "mixed_or_unresolved",
        "target_view_mismatch_resolved_but_cv_unstable",
    }
    assert rows
