from __future__ import annotations

from pathlib import Path

import numpy as np

from cement_channel.labels.label_quality_schema import parse_label_quality_config
from cement_channel.labels.label_quality_subsets import (
    build_label_quality_subsets,
    connected_object_candidate_mask,
    review_interval_mask,
)
from tests.unit.test_label_quality_schema import _valid_raw_config


def _sample_arrays() -> dict[str, np.ndarray]:
    depth_values = np.arange(5680.0, 5692.0, 2.0, dtype=np.float32)
    depth = np.repeat(depth_values, 4)
    side = np.tile(np.arange(4, dtype=np.int16), depth_values.size)
    label = np.zeros(depth.size, dtype=np.int8)
    severity = np.zeros(depth.size, dtype=np.int8)
    confidence = np.full(depth.size, 0.05, dtype=np.float32)
    positive = (
        ((depth == 5680.0) & np.isin(side, [0, 1, 3]))
        | ((depth == 5682.0) & np.isin(side, [0, 1, 3]))
        | ((depth == 5684.0) & np.isin(side, [0, 1]))
    )
    label[positive] = 1
    severity[positive] = 2
    confidence[positive] = 0.8
    return {
        "sample_id": np.arange(depth.size, dtype=np.int64),
        "depth": depth,
        "side_index": side,
        "side_azimuth_deg": side.astype(np.float32) * 90.0,
        "label_presence_plus": label,
        "label_severity_plus": severity,
        "label_confidence_plus": confidence,
        "label_presence_minus_audit": label.copy(),
        "plus_minus_disagreement": np.zeros(depth.size, dtype=bool),
        "orientation_confidence": np.full(depth.size, 0.9, dtype=np.float32),
        "depth_match_error": np.zeros(depth.size, dtype=np.float32),
    }


def test_review_interval_mask_flags_configured_depth_band() -> None:
    config = parse_label_quality_config(_valid_raw_config())
    depth = np.asarray([5679.0, 5680.0, 5700.0, 5721.0], dtype=np.float32)

    mask = review_interval_mask(depth, config.exclude_review_intervals)

    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_connected_object_candidate_mask_keeps_circular_side_component() -> None:
    arrays = _sample_arrays()

    kept, object_id, summary = connected_object_candidate_mask(
        depth=arrays["depth"],
        side_index=arrays["side_index"],
        candidate_mask=arrays["label_presence_plus"] == 1,
        min_area_samples=6,
        min_depth_length_ft=2.0,
        circular_side_connectivity=True,
    )

    assert int(np.count_nonzero(kept)) == 8
    assert int(summary["kept_component_count"]) == 1
    assert np.max(object_id) == 1


def test_build_label_quality_subsets_uses_local_normal_clear_negative() -> None:
    raw = _valid_raw_config()
    raw["quality_policy"]["min_subset_samples_per_class"] = 2
    raw["subsets"]["exclude_review_intervals"][0]["depth_min_ft"] = 5690.0
    raw["subsets"]["exclude_review_intervals"][0]["depth_max_ft"] = 5700.0
    config = parse_label_quality_config(raw)

    output, report = build_label_quality_subsets(
        sample_arrays=_sample_arrays(),
        config=config,
        inputs={"sample_table_npz": "synthetic.npz"},
        output_npz=Path("synthetic_output.npz"),
    )

    assert report.errors == []
    assert report.subset_counts["quality_strong_positive"]["sample_count"] == 8
    assert report.subset_counts["quality_clear_negative"]["sample_count"] > 0
    assert output["quality_strong_positive_mask"].dtype == np.bool_
    assert output["no_final_labels"].item() is True


def test_build_label_quality_subsets_reports_single_class_error() -> None:
    raw = _valid_raw_config()
    raw["quality_policy"]["min_subset_samples_per_class"] = 2
    raw["subsets"]["exclude_review_intervals"][0]["depth_min_ft"] = 5690.0
    raw["subsets"]["exclude_review_intervals"][0]["depth_max_ft"] = 5700.0
    arrays = _sample_arrays()
    arrays["label_presence_plus"][:] = 1
    arrays["label_severity_plus"][:] = 2
    arrays["label_confidence_plus"][:] = 0.9

    _, report = build_label_quality_subsets(
        sample_arrays=arrays,
        config=parse_label_quality_config(raw),
    )

    assert any("single-class" in message for message in report.errors)
