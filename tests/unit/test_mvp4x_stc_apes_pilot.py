from __future__ import annotations

import numpy as np
import pytest

from cement_channel.features.mvp4x_stc_apes_pilot import (
    compute_apes_proxy_summary,
    compute_relative_stc_summary,
    select_representative_intervals,
)


def _snapshot(row_count: int = 220) -> dict[str, np.ndarray]:
    depth = np.linspace(5800.0, 2400.0, row_count, dtype=np.float32)
    x = np.linspace(0.0, 1.0, row_count, dtype=np.float32)
    morphology = np.column_stack([x, x[::-1], np.sin(x * np.pi) ** 2]).astype(np.float32)
    return {
        "depth": depth,
        "receiver_p90": x,
        "receiver_max": np.sqrt(x),
        "full_360_fraction": x[::-1],
        "orientation_confidence": np.clip(np.sin(x * np.pi), 0.0, 1.0),
        "no_overlap_flag": np.zeros(row_count, dtype=bool),
        "any_special_flag": x > 0.85,
        "broad_regime_id": np.where(x < 0.33, "A", np.where(x < 0.66, "B", "C")),
        "morphology_max_relative_drop": morphology,
        "morphology_largest_connected_component_fraction": morphology * 0.5,
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
    }


def test_select_representative_intervals_enforces_bounds_and_flags() -> None:
    intervals = select_representative_intervals(
        _snapshot(),
        interval_count=80,
        half_window_ft=2.5,
    )

    assert len(intervals) == 80
    assert intervals[0]["interval_id"] == "SA001"
    assert all(row["research_only"] is True for row in intervals)
    assert all(row["no_final_labels"] is True for row in intervals)
    assert len({round(float(row["depth_center"]), 1) for row in intervals}) == 80


def test_select_representative_intervals_rejects_unbounded_counts() -> None:
    with pytest.raises(ValueError, match="interval_count"):
        select_representative_intervals(_snapshot(), interval_count=200, half_window_ft=2.5)


def test_relative_stc_summary_returns_finite_peak_for_shifted_receivers() -> None:
    receiver_count = 5
    side_count = 2
    time_count = 128
    axis = np.arange(time_count, dtype=np.float32)
    base = np.sin(2.0 * np.pi * axis / 24.0)
    traces = np.zeros((receiver_count, side_count, time_count), dtype=np.float32)
    offsets = np.arange(receiver_count) - (receiver_count - 1) / 2
    for receiver, offset in enumerate(offsets):
        traces[receiver, 0, :] = np.roll(base, int(round(2.0 * offset)))
        traces[receiver, 1, :] = np.roll(base, int(round(-1.0 * offset)))

    summary = compute_relative_stc_summary(traces, side_labels=["A", "B"])

    assert np.isfinite(summary["peak_coherence"])
    assert summary["peak_coherence"] > 0.5
    assert summary["peak_side"] in {"A", "B"}


def test_apes_proxy_summary_peaks_near_sine_frequency() -> None:
    receiver_count = 4
    side_count = 2
    time_count = 160
    axis = np.arange(time_count, dtype=np.float32)
    traces = np.zeros((receiver_count, side_count, time_count), dtype=np.float32)
    sine = np.sin(2.0 * np.pi * 0.125 * axis)
    for receiver in range(receiver_count):
        traces[receiver, 0, :] = sine
        traces[receiver, 1, :] = np.sin(2.0 * np.pi * 0.25 * axis)

    summary = compute_apes_proxy_summary(
        traces,
        side_labels=["A", "B"],
        frequency_count=32,
        covariance_window=24,
    )

    assert np.isfinite(summary["peak_power"])
    assert 0.02 <= summary["peak_frequency"] <= 0.45
    assert summary["peak_side"] in {"A", "B"}
