from __future__ import annotations

import numpy as np

from cement_channel.labels.depth_level_labels import (
    build_depth_level_label_table,
    depth_review_interval_mask,
)
from cement_channel.labels.depth_level_schema import parse_depth_level_label_config
from tests.unit.test_depth_level_schema import _valid_raw_config


def _config():
    raw = _valid_raw_config()
    raw["quality_policy"]["strong_positive"]["min_candidate_fraction"] = 0.2
    raw["quality_policy"]["review_intervals"][0]["depth_min_ft"] = 5700.0
    raw["quality_policy"]["review_intervals"][0]["depth_max_ft"] = 5710.0
    return parse_depth_level_label_config(raw)


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    presence = np.zeros((3, 12), dtype=np.int8)
    severity = np.zeros_like(presence)
    confidence = np.full((3, 12), 0.2, dtype=np.float32)
    relative_drop = np.zeros((3, 12), dtype=np.float32)
    zc = np.full((3, 12), 6.0, dtype=np.float32)
    presence[0, [11, 0, 1]] = 1
    severity[0, [11, 0, 1]] = [2, 3, 2]
    confidence[0, [11, 0, 1]] = [0.7, 0.9, 0.8]
    relative_drop[0, [11, 0, 1]] = [0.4, 0.6, 0.5]
    zc[0, [11, 0, 1]] = [2.4, 2.0, 2.2]
    presence[2, 5] = 1
    severity[2, 5] = 1
    confidence[2, 5] = 0.4
    cast_arrays = {
        "cast_depth": np.asarray([100.0, 101.0, 102.0], dtype=np.float32),
        "cast_azimuth_aligned_deg": np.arange(12, dtype=np.float32) * 30.0,
        "presence_plus": presence,
        "severity_plus": severity,
        "label_confidence_plus": confidence,
        "presence_minus_ablation": np.zeros_like(presence),
        "relative_drop_plus": relative_drop,
        "cast_zc": zc,
        "no_final_labels": np.asarray(True),
    }
    xsi_arrays = {
        "xsi_depth": np.asarray([100.0, 101.0, 102.0], dtype=np.float32),
        "cast_depth_index": np.asarray([0, 1, 2], dtype=np.int32),
        "orientation_confidence_depth": np.asarray([1.0, 1.0, 0.4], dtype=np.float32),
        "orientation_confidence": np.ones((3, 8), dtype=np.float32),
        "no_final_labels": np.asarray(True),
    }
    return cast_arrays, xsi_arrays


def test_build_depth_level_label_table_aggregates_any_fraction_percentile_and_width() -> None:
    cast_arrays, xsi_arrays = _arrays()

    output, report = build_depth_level_label_table(
        cast_arrays=cast_arrays,
        xsi_arrays=xsi_arrays,
        config=_config(),
    )

    assert report.errors == []
    assert report.positive_count == 2
    assert report.strong_positive_count == 1
    assert report.clear_negative_count == 1
    np.testing.assert_array_equal(output["depth_has_channel_any"], [True, False, True])
    assert np.isclose(output["depth_candidate_fraction"][0], 3 / 12)
    assert output["depth_max_severity"][0] == 3
    assert np.isclose(output["depth_min_zc"][0], 2.0)
    assert output["depth_largest_azimuth_object_width"][0] == 90.0
    assert output["depth_clear_negative_mask"][1]
    assert bool(output["no_final_labels"]) is True


def test_depth_level_label_table_warns_when_raw_zc_is_missing() -> None:
    cast_arrays, xsi_arrays = _arrays()
    cast_arrays.pop("cast_zc")

    output, report = build_depth_level_label_table(
        cast_arrays=cast_arrays,
        xsi_arrays=xsi_arrays,
        config=_config(),
    )

    assert report.zc_source_field is None
    assert np.isnan(output["depth_min_zc"]).all()
    assert any("raw Zc" in message for message in report.warnings)


def test_depth_review_interval_mask_flags_configured_band() -> None:
    config = _config()
    depth = np.asarray([5699.0, 5700.0, 5705.0, 5711.0], dtype=np.float32)

    mask = depth_review_interval_mask(depth, config.quality_policy.review_intervals)

    np.testing.assert_array_equal(mask, [False, True, True, False])
