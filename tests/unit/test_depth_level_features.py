from __future__ import annotations

import numpy as np

from cement_channel.features.depth_level_features import (
    build_depth_level_xsi_feature_table,
    compute_depth_level_xsi_features,
)


def _basic_arrays() -> dict[str, np.ndarray]:
    features = np.zeros((2, 4, 3, 2), dtype=np.float32)
    features[0, :, :, 0] = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0],
            [8.0, 9.0, 10.0],
        ],
        dtype=np.float32,
    )
    features[0, :, :, 1] = 2.0
    features[1, :, :, 0] = 1.0
    features[1, :, :, 1] = np.array(
        [
            [1.0, 2.0, 4.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 8.0],
            [2.0, 4.0, 8.0],
        ],
        dtype=np.float32,
    )
    return {
        "xsi_depth": np.asarray([100.0, 101.0], dtype=np.float32),
        "feature_names": np.asarray(["rms_energy", "late_over_early_ratio"]),
        "xsi_basic_features": features,
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }


def test_compute_depth_level_xsi_features_includes_required_groups() -> None:
    matrix, names, groups = compute_depth_level_xsi_features(
        _basic_arrays()["xsi_basic_features"],
        _basic_arrays()["feature_names"],
    )

    assert matrix.shape[0] == 2
    assert groups["side_mean"] == 2
    assert groups["receiver_mean"] == 2
    assert groups["near_far_receiver_ratio"] == 6
    assert groups["high_side_sector_audit"] == 4
    assert "late_over_early_side_max_late_over_early_ratio" in names
    assert "high_side_audit_rms_energy" in names


def test_build_depth_level_xsi_feature_table_preserves_guardrails_and_no_label_use() -> None:
    sample_arrays = {
        "depth": np.repeat(np.asarray([100.0, 101.0], dtype=np.float32), 3),
        "label_presence_plus": np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int8),
    }

    output, report = build_depth_level_xsi_feature_table(
        basic_arrays=_basic_arrays(),
        sample_arrays=sample_arrays,
    )

    assert report.errors == []
    assert report.used_label_information_for_feature_construction is False
    assert report.high_side_sector_summaries_audit_only is True
    assert output["depth_level_xsi_features"].shape[0] == 2
    assert bool(output["no_final_labels"]) is True
    assert bool(output["no_stc"]) is True
