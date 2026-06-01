from __future__ import annotations

import numpy as np

from cement_channel.alignment.xsi_geometry import ReceiverGeometry


def test_receiver_geometry_offsets_and_source_distance() -> None:
    geometry = ReceiverGeometry()

    assert geometry.receiver_offsets_ft[6] == 0.0
    assert geometry.receiver_offsets_ft[0] == -3.0
    assert geometry.receiver_offsets_ft[-1] == 3.0
    assert geometry.source_offset_ft == -4.0
    assert np.isclose(geometry.receiver_offsets_ft[0] - geometry.source_offset_ft, 1.0)
    assert geometry.depth_axis_sign == -1
    assert geometry.sign_convention_status == "human_confirmed"
    assert geometry.physical_receiver_offsets_ft[0] == 3.0
    assert geometry.physical_receiver_offsets_ft[-1] == -3.0
    assert geometry.physical_source_offset_ft == 4.0


def test_receiver_geometry_supports_depth_axis_signs() -> None:
    geometry = ReceiverGeometry()
    reference_depth = np.asarray([1000.0], dtype=np.float32)

    plus_receivers = geometry.receiver_depths(reference_depth, sign=1)
    minus_receivers = geometry.receiver_depths(reference_depth, sign=-1)
    plus_source = geometry.source_depths(reference_depth, sign=1)
    minus_source = geometry.source_depths(reference_depth, sign=-1)

    assert plus_receivers.shape == (1, 13)
    assert plus_receivers[0, 6] == 1000.0
    assert plus_receivers[0, 0] == 997.0
    assert plus_receivers[0, -1] == 1003.0
    assert plus_source[0] == 996.0
    assert minus_receivers[0, 0] == 1003.0
    assert minus_receivers[0, -1] == 997.0
    assert minus_source[0] == 1004.0


def test_receiver_geometry_midpoints_and_intervals() -> None:
    geometry = ReceiverGeometry()
    reference_depth = np.asarray([1000.0], dtype=np.float32)

    midpoints = geometry.source_receiver_midpoints(reference_depth, sign=1)
    lower, upper = geometry.source_receiver_interval_bounds(
        reference_depth,
        sign=1,
        padding_ft=0.25,
    )

    assert midpoints.shape == (1, 13)
    assert midpoints[0, 0] == 996.5
    assert midpoints[0, 6] == 998.0
    assert midpoints[0, -1] == 999.5
    assert lower[0, 0] == 995.75
    assert upper[0, 0] == 997.25
    assert lower[0, -1] == 995.75
    assert upper[0, -1] == 1003.25


def test_receiver_geometry_loads_nested_example_config() -> None:
    geometry = ReceiverGeometry.from_yaml("configs/xsi_geometry.example.yaml")

    assert geometry.reference_receiver_index == 7
    assert geometry.receiver_count == 13
    assert geometry.receiver_spacing_ft == 0.5
    assert geometry.audit_signs == (-1,)
    assert geometry.depth_axis_sign == -1
    assert geometry.sign_convention_status == "human_confirmed"
    assert geometry.depth_increases_toward == "deeper"
    assert geometry.sample_index_direction == "deep_to_shallow"
    assert geometry.receiver_index_direction == "R1_deep_to_R13_shallow"
    assert geometry.source_position == "deeper_than_R1"
    assert "source_receiver_interval" in geometry.alignment_modes
