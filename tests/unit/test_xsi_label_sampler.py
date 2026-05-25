from __future__ import annotations

import numpy as np

from cement_channel.evaluation.correlation_schema import parse_correlation_config
from cement_channel.evaluation.xsi_label_sampler import (
    build_xsi_label_samples_from_arrays,
    cast_azimuth_to_side_index,
    interpolate_depth_values,
    nearest_depth_indices,
)
from tests.unit.test_correlation_schema import _valid_raw_config


def _config():
    return parse_correlation_config(_valid_raw_config())


def _label_arrays() -> dict[str, np.ndarray]:
    presence_plus = np.zeros((2, 16), dtype=np.int8)
    severity_plus = np.zeros((2, 16), dtype=np.int8)
    confidence_plus = np.full((2, 16), 0.2, dtype=np.float32)
    presence_plus[0, 4] = 1
    severity_plus[0, 4] = 3
    confidence_plus[0, 4] = 0.8
    presence_minus = presence_plus.copy()
    severity_minus = severity_plus.copy()
    confidence_minus = confidence_plus.copy()
    presence_minus[0, 4] = 0
    severity_minus[0, 4] = 0
    confidence_minus[0, 4] = 0.2
    return {
        "cast_depth": np.array([100.0, 99.0], dtype=np.float32),
        "cast_azimuth_aligned_deg": np.arange(16, dtype=np.float32) * 22.5,
        "presence_plus": presence_plus,
        "severity_plus": severity_plus,
        "label_confidence_plus": confidence_plus,
        "presence_minus_ablation": presence_minus,
        "severity_minus_ablation": severity_minus,
        "label_confidence_minus_ablation": confidence_minus,
        "no_final_labels": np.asarray(True),
    }


def test_cast_azimuth_to_side_index_uses_nearest_side_center() -> None:
    side = np.arange(8, dtype=np.float32) * 45.0

    mapped = cast_azimuth_to_side_index(
        np.array([0.0, 22.0, 44.0, 90.0, 359.0], dtype=np.float32),
        side_azimuth_deg=side,
    )

    np.testing.assert_array_equal(mapped, np.array([0, 0, 1, 2, 0], dtype=np.int16))


def test_depth_helpers_handle_decreasing_depth_axes() -> None:
    source = np.array([100.0, 99.0, 98.0], dtype=np.float32)
    target = np.array([99.2, 98.1], dtype=np.float32)

    np.testing.assert_array_equal(nearest_depth_indices(source, target), np.array([1, 2]))
    interp = interpolate_depth_values(
        source_depth=source,
        source_values=np.array([1.0, 0.5, 0.0], dtype=np.float32),
        target_depth=target,
    )
    np.testing.assert_allclose(interp, np.array([0.6, 0.05], dtype=np.float32), atol=1e-5)


def test_build_xsi_label_samples_aggregates_cast_bins_to_xsi_sides() -> None:
    depth_arrays = {
        "xsi_depth_by_receiver": np.tile(
            np.array([100.0, 99.0], dtype=np.float32),
            (13, 1),
        )
    }
    orientation_arrays = {
        "pose_depth": np.array([100.0, 99.0], dtype=np.float32),
        "orientation_confidence": np.array([1.0, 0.2], dtype=np.float32),
    }

    arrays, stats = build_xsi_label_samples_from_arrays(
        label_arrays=_label_arrays(),
        depth_arrays=depth_arrays,
        orientation_arrays=orientation_arrays,
        correlation_config=_config(),
    )

    assert stats["errors"] == []
    assert arrays["label_presence_plus"].shape == (2, 8)
    assert arrays["label_presence_plus"][0, 2] == 1
    assert arrays["label_severity_plus"][0, 2] == 3
    assert arrays["label_confidence_plus"][0, 2] == np.float32(0.8)
    assert arrays["label_presence_minus_audit"][0, 2] == 0
    assert arrays["plus_minus_disagreement"][0, 2]
    assert arrays["valid_for_azimuthal_validation"][0, 2]
    assert not arrays["valid_for_azimuthal_validation"][1, 2]
    assert arrays["valid_for_non_azimuthal_summary"][1, 2]
    assert stats["coverage"]["high_confidence_candidate_count"] == 1


def test_build_xsi_label_samples_rejects_final_label_claim() -> None:
    labels = _label_arrays()
    labels["no_final_labels"] = np.asarray(False)
    depth_arrays = {"xsi_depth_by_receiver": np.tile(np.array([100.0, 99.0]), (13, 1))}
    orientation_arrays = {
        "pose_depth": np.array([100.0, 99.0], dtype=np.float32),
        "orientation_confidence": np.ones(2, dtype=np.float32),
    }

    _arrays, stats = build_xsi_label_samples_from_arrays(
        label_arrays=labels,
        depth_arrays=depth_arrays,
        orientation_arrays=orientation_arrays,
        correlation_config=_config(),
    )

    assert "no_final_labels=true" in stats["errors"][0]
