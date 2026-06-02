from __future__ import annotations

import numpy as np

from cement_channel.qc.cast_zc_physical_qc import audit_cast_zc_physical_qc


def _cast_arrays() -> dict[str, np.ndarray]:
    return {
        "cast_depth": np.asarray([0.0, 1.0], dtype=np.float32),
        "cast_azimuth_deg": np.asarray([0.0, 120.0, 240.0], dtype=np.float32),
        "cast_zc": np.asarray(
            [
                [-1.0, 1.0, 3.0],
                [1.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    }


def _label_arrays() -> dict[str, np.ndarray]:
    target = np.asarray([[[2.0 / 3.0], [1.0 / 3.0]]], dtype=np.float32)
    return {
        "depth": np.asarray([0.0, 1.0], dtype=np.float32),
        "geometry_kernel": np.asarray(["r7_reference_point"]),
        "receiver_index": np.asarray([1], dtype=np.int16),
        "weighted_channel_fraction_zc_lt_2p5": target,
        "interval_min_depth": np.asarray([[[0.0], [1.0]]], dtype=np.float32),
        "interval_max_depth": np.asarray([[[0.0], [1.0]]], dtype=np.float32),
        "midpoint_depth": np.asarray([[[0.0], [1.0]]], dtype=np.float32),
        "total_cell_count": np.asarray([[[3], [3]]], dtype=np.int32),
    }


def test_cast_zc_physical_qc_counts_negative_and_counterfactual_delta() -> None:
    report, rows = audit_cast_zc_physical_qc(
        cast_arrays=_cast_arrays(),
        label_arrays=_label_arrays(),
    )

    counts = report.global_cell_counts
    assert counts["zc_lt_0_count"] == 1
    assert counts["zc_eq_0_count"] == 0
    assert counts["zc_gt_0_lt_2p5_count"] == 2
    assert counts["zc_gte_2p5_count"] == 3
    assert report.counterfactual_summary["max_abs_delta"] > 0.16
    assert report.negative_zc_target_influence_significant is True
    assert report.no_formal_mask_applied is True
    assert any(row["section"] == "raw_depth" for row in rows)
    assert any(row["section"] == "kernel_summary" for row in rows)


def test_cast_zc_physical_qc_reports_boundary_no_overlap() -> None:
    labels = _label_arrays()
    labels["total_cell_count"] = np.asarray([[[0], [3]]], dtype=np.int32)

    report, _rows = audit_cast_zc_physical_qc(
        cast_arrays=_cast_arrays(),
        label_arrays=labels,
    )

    assert report.boundary_no_overlap_summary["no_overlap_region_count"] == 1
    assert report.boundary_no_overlap_summary["no_overlap_region_fraction"] == 0.5
