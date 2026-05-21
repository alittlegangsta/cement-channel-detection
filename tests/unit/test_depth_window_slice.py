from __future__ import annotations

import numpy as np

from cement_channel.data.small_slice_reader import (
    DepthWindow,
    select_depth_window_slices,
)


def test_depth_window_selection_uses_requested_window() -> None:
    depth = np.arange(100.0, 111.0)
    selection = select_depth_window_slices(
        {
            "cast_depth": depth,
            "pose_depth": depth,
            "xsi_depth_by_receiver": np.stack([depth, depth]),
        },
        DepthWindow(depth_start=103.0, depth_stop=107.0),
        max_depth_samples=3,
    )

    assert selection["cast"]["sample_count"] == 3
    assert selection["cast"]["matched_count"] == 5
    assert 3 <= selection["cast"]["source_start_index"] <= 5
    assert selection["pose"]["sample_count"] == 3
    assert selection["xsi_receivers"]["receiver_01"]["sample_count"] == 3


def test_depth_window_selection_warns_when_source_has_no_samples() -> None:
    depth = np.arange(100.0, 111.0)
    selection = select_depth_window_slices(
        {
            "cast_depth": depth,
            "pose_depth": depth,
            "xsi_depth_by_receiver": np.stack([depth]),
        },
        DepthWindow(depth_start=200.0, depth_stop=201.0),
        max_depth_samples=3,
    )

    assert selection["cast"]["sample_count"] == 0
    assert selection["cast"]["warnings"]
    assert selection["xsi_receivers"]["receiver_01"]["sample_count"] == 0
