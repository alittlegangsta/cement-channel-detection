from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_model_analysis import (
    build_feature_sets,
    summarize_feature_groups,
)


def test_build_feature_sets_creates_existing_waveform_and_combined() -> None:
    snapshot = {
        "xsi_features": np.ones((4, 2), dtype=np.float32),
        "xsi_feature_names": np.asarray(["a", "b"]),
        "xsi_feature_group": np.asarray(["g1", "g2"]),
    }
    waveform = {
        "waveform_depth_features": np.ones((4, 3), dtype=np.float32),
        "waveform_depth_feature_names": np.asarray(["c", "d", "e"]),
        "waveform_depth_feature_group": np.asarray(["wg1", "wg1", "wg2"]),
    }

    feature_sets = build_feature_sets(snapshot, waveform)

    assert set(feature_sets) == {
        "existing_features_only",
        "waveform_features_only",
        "combined_features",
    }
    assert feature_sets["combined_features"]["matrix"].shape == (4, 5)
    assert feature_sets["combined_features"]["names"].tolist() == ["a", "b", "c", "d", "e"]


def test_summarize_feature_groups_counts_prefixed_groups() -> None:
    snapshot = {
        "xsi_features": np.ones((4, 1), dtype=np.float32),
        "xsi_feature_names": np.asarray(["a"]),
        "xsi_feature_group": np.asarray(["g1"]),
    }
    waveform = {
        "waveform_depth_features": np.ones((4, 2), dtype=np.float32),
        "waveform_depth_feature_names": np.asarray(["c", "d"]),
        "waveform_depth_feature_group": np.asarray(["wg1", "wg1"]),
    }

    summary = summarize_feature_groups(build_feature_sets(snapshot, waveform))

    assert summary["existing_features_only"]["group_counts"] == {"existing:g1": 1}
    assert summary["waveform_features_only"]["group_counts"] == {"waveform:wg1": 2}
    assert summary["combined_features"]["feature_count"] == 3

