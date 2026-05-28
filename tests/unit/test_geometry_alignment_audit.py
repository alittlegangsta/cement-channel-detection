from __future__ import annotations

import numpy as np

from cement_channel.evaluation.geometry_alignment_audit import (
    GeometryAlignmentAuditConfig,
    audit_geometry_alignment,
)


def _geometry_arrays() -> dict[str, np.ndarray]:
    n = 30
    mode = np.asarray(["r7_reference_depth", "r7_reference_depth"])
    sign = np.asarray([1, -1], dtype=np.int8)
    positive = np.zeros((2, n), dtype=bool)
    negative = np.zeros((2, n), dtype=bool)
    positive[:, :10] = True
    negative[:, 10:25] = True
    return {
        "depth": np.arange(n, dtype=np.float32),
        "mode": mode,
        "sign": sign,
        "high_confidence_positive_mask": positive,
        "clear_negative_mask": negative,
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }


def _features() -> dict[str, np.ndarray]:
    n = 30
    signal = np.zeros(n, dtype=np.float32)
    signal[:10] = 2.0
    signal[10:25] = -1.0
    return {
        "depth": np.arange(n, dtype=np.float32),
        "depth_level_xsi_features": np.column_stack(
            [signal, np.zeros(n, dtype=np.float32)]
        ).astype(np.float32),
        "depth_level_xsi_feature_names": np.asarray(["receiver_mean_peak_abs", "side_mean"]),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }


def test_geometry_alignment_audit_reports_best_feature_and_permutation_margin() -> None:
    report, rows = audit_geometry_alignment(
        geometry_arrays=_geometry_arrays(),
        feature_arrays=_features(),
        refinement_report={"no_final_labels": True},
        config=GeometryAlignmentAuditConfig(permutation_repeats=3),
    )

    assert report.report_version == "geometry_alignment_audit_v001"
    assert report.no_final_labels is True
    assert report.best_combo is not None
    assert report.best_combo["best_feature_name"] == "receiver_mean_peak_abs"
    assert report.best_combo["real_minus_permutation_margin"] is not None
    assert rows


def test_geometry_alignment_audit_skips_collapsed_sample_counts() -> None:
    arrays = _geometry_arrays()
    arrays["clear_negative_mask"][:] = False

    report, _rows = audit_geometry_alignment(
        geometry_arrays=arrays,
        feature_arrays=_features(),
        refinement_report={"no_final_labels": True},
        config=GeometryAlignmentAuditConfig(),
    )

    assert report.recommendation == "no_go"
    assert all(row["status"] == "skipped_sample_count" for row in report.combo_summaries)
