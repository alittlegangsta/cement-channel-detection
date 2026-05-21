from __future__ import annotations

import numpy as np

from cement_channel.alignment.depth_reader import summarize_depth_only_array


def test_summarize_depth_only_array_reports_numeric_stats() -> None:
    result = summarize_depth_only_array("inc_deg", np.array([1.0, 2.0, 3.0]))

    assert result.shape == [3]
    assert result.finite_ratio == 1.0
    assert result.min == 1.0
    assert result.max == 3.0
    assert result.mean == 2.0
    assert not result.errors


def test_summarize_depth_only_array_warns_nonfinite() -> None:
    result = summarize_depth_only_array("relbearing_deg", np.array([1.0, np.nan]))

    assert result.finite_ratio == 0.5
    assert result.warnings
    assert not result.errors
