from __future__ import annotations

import numpy as np

from cement_channel.alignment.depth_audit import (
    audit_depth_axes,
    common_overlap_interval,
    summarize_depth_axis,
    summarize_receiver_depth_consistency,
)


def test_summarize_depth_axis_reports_basic_stats() -> None:
    result = summarize_depth_axis("depth", np.array([100.0, 100.5, 101.0]))

    assert result.length == 3
    assert result.min == 100.0
    assert result.max == 101.0
    assert result.monotonic is True
    assert result.direction == "increasing"
    assert result.median_step == 0.5
    assert result.nan_count == 0
    assert result.duplicate_count == 0


def test_summarize_depth_axis_blocks_non_monotonic() -> None:
    result = summarize_depth_axis("depth", np.array([100.0, 99.5, 100.5]))

    assert result.monotonic is False
    assert result.errors


def test_common_overlap_interval_uses_all_axes() -> None:
    overlap = common_overlap_interval(
        [
            np.array([0.0, 1.0, 2.0]),
            np.array([0.5, 1.5, 2.5]),
            np.array([1.0, 1.5, 3.0]),
        ]
    )

    assert overlap["min"] == 1.0
    assert overlap["max"] == 2.0
    assert overlap["length"] == 1.0


def test_receiver_consistency_accepts_matching_axes() -> None:
    axes = np.vstack([np.arange(5.0), np.arange(5.0) + 0.0001])

    result = summarize_receiver_depth_consistency(axes, expected_receiver_count=2)

    assert result.receiver_count == 2
    assert result.consistent is True
    assert not result.errors


def test_audit_depth_axes_returns_conditional_go_for_unknown_unit() -> None:
    depth = np.linspace(10.0, 20.0, 11)
    result = audit_depth_axes(
        cast_depth=depth,
        xsi_depth_by_receiver=np.vstack([depth, depth]),
        pose_depth=depth,
        expected_receiver_count=2,
        depth_unit="unknown_to_verify",
    )

    assert result.decision == "conditional_go"
    assert result.common_overlap_interval["length"] == 10.0
    assert not result.no_go_blockers
