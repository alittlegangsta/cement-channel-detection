from __future__ import annotations

import numpy as np

from cement_channel.visualization.geometry_alignment_review import (
    build_geometry_mode_comparison_rows,
    build_interval_geometry_comparison,
)


def test_build_geometry_mode_comparison_rows_from_audit_report() -> None:
    rows = build_geometry_mode_comparison_rows(
        {
            "combo_summaries": [
                {
                    "mode": "r7_reference_depth",
                    "sign": 1,
                    "status": "runnable",
                    "positive_count": 10,
                    "negative_count": 5,
                }
            ]
        }
    )

    assert rows == [
        {
            "mode": "r7_reference_depth",
            "sign": 1,
            "status": "runnable",
            "positive_count": 10,
            "negative_count": 5,
            "positive_fraction": None,
            "best_feature_name": None,
            "best_abs_effect_size": None,
            "balanced_accuracy": None,
            "permutation_balanced_accuracy": None,
            "real_minus_permutation_margin": None,
            "predicted_positive_rate": None,
            "folds_above_permutation": None,
            "depends_on_5700_band": None,
            "sample_count_collapse": None,
        }
    ]


def test_interval_geometry_comparison_flags_clear_negative_change() -> None:
    depth = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
    arrays = {
        "depth": depth,
        "mode": np.asarray(["source_receiver_interval"]),
        "sign": np.asarray([1], dtype=np.int8),
        "candidate_fraction": np.asarray([[0.0, 0.4, 0.5]], dtype=np.float32),
        "has_channel_any": np.asarray([[False, True, True]]),
        "max_severity": np.asarray([[0, 2, 3]], dtype=np.int8),
        "max_confidence": np.asarray([[0.0, 0.8, 0.9]], dtype=np.float32),
        "max_relative_drop": np.asarray([[0.0, 0.5, 0.7]], dtype=np.float32),
        "depth_label_confidence": np.asarray([[1.0, 0.8, 0.9]], dtype=np.float32),
    }
    intervals = [
        {
            "review_id": "DLR-009",
            "interval_type": "clear_negative_like",
            "start_depth": 10.0,
            "end_depth": 12.0,
        }
    ]
    rows, interval_json = build_interval_geometry_comparison(
        intervals=intervals,
        geometry_arrays=arrays,
        original_cast_rows={
            "DLR-009": {"evidence_category": "clear_negative_evidence"}
        },
    )

    assert rows[0]["evidence_category_changed"] is True
    assert rows[0]["review_decision_should_be_revisited"] is True
    assert interval_json[0]["review_decision_should_be_revisited"] is True
