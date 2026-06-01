from __future__ import annotations

import numpy as np
import pytest

from cement_channel.alignment.xsi_geometry import ReceiverGeometry
from cement_channel.labels.cast_zc_source import build_cast_zc_source_from_arrays
from cement_channel.labels.geometry_aware_regression_labels import (
    GeometryAwareRegressionConfig,
    build_geometry_aware_regression_labels,
)


def _raw_source():
    depth = np.arange(96.0, 108.5, 0.5, dtype=np.float32)
    azimuth = np.arange(6, dtype=np.float32) * 60.0
    zc = np.full((depth.size, azimuth.size), 4.0, dtype=np.float32)
    zc[(depth >= 100.0) & (depth <= 101.0), :2] = 2.2
    zc[(depth >= 103.0) & (depth <= 105.0), 3:5] = 2.0
    return build_cast_zc_source_from_arrays(
        {"cast_depth": depth, "cast_azimuth_deg": azimuth, "cast_zc": zc},
        source_file="cast_label_input_v001.npz",
        min_finite_ratio=0.9,
    )


def _features() -> dict[str, np.ndarray]:
    depth = np.asarray([100.0, 103.0, 106.0], dtype=np.float32)
    return {
        "depth": depth,
        "depth_level_xsi_features": np.ones((depth.size, 2), dtype=np.float32),
        "depth_level_xsi_feature_names": np.asarray(["a", "b"]),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }


def _baseline(raw_zc_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    relative_drop = np.zeros(raw_zc_shape, dtype=np.float32)
    relative_drop[8:12, :2] = 0.5
    return {"relative_drop": relative_drop}


def test_geometry_aware_regression_labels_keep_continuous_receiver_targets() -> None:
    raw_source = _raw_source()
    arrays, report = build_geometry_aware_regression_labels(
        raw_source=raw_source,
        feature_arrays=_features(),
        geometry=ReceiverGeometry(),
        config=GeometryAwareRegressionConfig(midpoint_window_half_width_ft=0.5),
        baseline_arrays=_baseline(raw_source.cast_zc.shape),
    )

    assert report.errors == []
    assert arrays["weighted_channel_fraction_zc_lt_2p5"].shape == (4, 3, 13)
    assert arrays["receiver_mean"].shape == (4, 3)
    assert arrays["source_depth"].shape == (4, 3, 13)
    assert arrays["receiver_depth"].shape == (4, 3, 13)
    assert bool(arrays["no_final_labels"]) is True
    values = arrays["weighted_channel_fraction_zc_lt_2p5"]
    assert np.any((values > 0.0) & (values < 1.0))
    assert "derived_positive_at_fraction_0p05" in arrays
    assert report.kernel_summaries[0]["distribution"]["max"] is not None
    assert report.geometry_sign_status["depth_axis_sign"] == -1


def test_geometry_aware_regression_labels_reject_unconfirmed_geometry() -> None:
    raw_source = _raw_source()

    with pytest.raises(ValueError, match="human-confirmed sign=-1"):
        build_geometry_aware_regression_labels(
            raw_source=raw_source,
            feature_arrays=_features(),
            geometry=ReceiverGeometry(
                depth_axis_sign="audit_both",
                sign_convention_status="requires_audit",
            ),
            config=GeometryAwareRegressionConfig(),
        )
