from __future__ import annotations

import numpy as np

from cement_channel.alignment.xsi_geometry import ReceiverGeometry
from cement_channel.labels.geometry_aware_cast_aggregation import (
    GeometryAwareAggregationConfig,
    build_geometry_aware_depth_labels,
)


def _cast_arrays() -> dict[str, np.ndarray]:
    presence = np.zeros((12, 4), dtype=np.int8)
    severity = np.zeros_like(presence)
    confidence = np.full((12, 4), 0.8, dtype=np.float32)
    presence[5, :] = 1
    severity[5, :] = 2
    presence[7, :2] = 1
    severity[7, :2] = 3
    minus = np.zeros_like(presence)
    return {
        "cast_depth": np.arange(12, dtype=np.float32),
        "presence_plus": presence,
        "severity_plus": severity,
        "label_confidence_plus": confidence,
        "presence_minus_ablation": minus,
        "relative_drop_plus": np.where(presence == 1, 0.6, 0.0).astype(np.float32),
        "orientation_confidence_on_cast_depth_plus": np.ones_like(confidence),
        "no_final_labels": np.asarray(True),
    }


def _depth_labels() -> dict[str, np.ndarray]:
    return {
        "depth": np.asarray([5.0, 8.0], dtype=np.float32),
        "depth_has_channel_any": np.asarray([True, False]),
        "depth_candidate_fraction": np.asarray([1.0, 0.0], dtype=np.float32),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }


def _features() -> dict[str, np.ndarray]:
    return {
        "depth": np.asarray([5.0, 8.0], dtype=np.float32),
        "depth_level_xsi_features": np.ones((2, 1), dtype=np.float32),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }


def test_geometry_aware_aggregation_outputs_all_modes_and_signs() -> None:
    arrays, report = build_geometry_aware_depth_labels(
        cast_arrays=_cast_arrays(),
        depth_label_arrays=_depth_labels(),
        feature_arrays=_features(),
        geometry=ReceiverGeometry(
            depth_axis_sign="audit_both",
            sign_convention_status="requires_audit",
        ),
        aggregation_config=GeometryAwareAggregationConfig(),
    )

    assert arrays["candidate_fraction"].shape == (8, 2)
    assert arrays["receiver_candidate_fraction"].shape == (8, 2, 13)
    assert set(arrays["mode"].astype(str)) == {
        "r7_reference_depth",
        "receiver_depth_shifted",
        "source_receiver_midpoint",
        "source_receiver_interval",
    }
    assert set(arrays["sign"].astype(int)) == {-1, 1}
    assert bool(arrays["no_final_labels"]) is True
    assert report.raw_zc_available is False
    assert any("raw Zc" in warning for warning in report.warnings)


def test_source_receiver_interval_can_change_local_reference_evidence() -> None:
    arrays, _report = build_geometry_aware_depth_labels(
        cast_arrays=_cast_arrays(),
        depth_label_arrays=_depth_labels(),
        feature_arrays=_features(),
        geometry=ReceiverGeometry(depth_axis_sign=1, sign_convention_status="requires_audit"),
        aggregation_config=GeometryAwareAggregationConfig(),
    )
    modes = arrays["mode"].astype(str)
    r7_index = int(np.flatnonzero(modes == "r7_reference_depth")[0])
    interval_index = int(np.flatnonzero(modes == "source_receiver_interval")[0])

    assert arrays["candidate_fraction"][r7_index, 1] == 0.0
    assert arrays["candidate_fraction"][interval_index, 1] > 0.0
    assert arrays["max_severity"][interval_index, 1] >= 2
