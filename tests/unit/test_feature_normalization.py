from __future__ import annotations

import numpy as np

from cement_channel.training.feature_normalization import (
    FeatureNormalizationConfig,
    derived_energy_features,
    enhance_sample_table_features,
    per_depth_side_rank,
    per_depth_side_zscore,
    per_side_depth_rolling_zscore,
)


def _toy_arrays() -> dict[str, np.ndarray]:
    depth = np.repeat(np.array([100.0, 101.0, 102.0], dtype=np.float32), 3)
    side = np.tile(np.array([0, 1, 2], dtype=np.int16), 3)
    raw = np.column_stack(
        [
            10.0 + side + depth * 0.0,
            20.0 + side,
            5.0 + side,
            100.0 - side,
            15.0 + side,
            np.array([0.10, 0.20, 0.30] * 3, dtype=np.float32),
        ]
    ).astype(np.float32)
    return {
        "sample_id": np.arange(depth.size, dtype=np.int64),
        "depth": depth,
        "side_index": side,
        "features": raw,
        "feature_names": np.array(
            [
                "rms_energy",
                "peak_abs",
                "mean_abs",
                "early_energy",
                "late_energy",
                "late_over_early_ratio",
            ]
        ),
        "transformed_features": np.log1p(raw).astype(np.float32),
        "transformed_feature_names": np.array(
            [
                "log1p_rms_energy",
                "log1p_peak_abs",
                "log1p_mean_abs",
                "log1p_early_energy",
                "log1p_late_energy",
                "log1p_late_over_early_ratio",
            ]
        ),
        "label_presence_plus": np.tile(np.array([0, 1, 0], dtype=np.int8), 3),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }


def test_per_depth_side_zscore_and_rank_are_depth_local() -> None:
    arrays = _toy_arrays()
    zscore = per_depth_side_zscore(
        arrays["depth"],
        arrays["features"][:, :1],
        epsilon=1.0e-6,
    )
    rank = per_depth_side_rank(arrays["depth"], arrays["features"][:, :1])

    np.testing.assert_allclose(zscore[:3, 0], [-1.2247448, 0.0, 1.2247448], atol=1e-5)
    np.testing.assert_allclose(rank[:3, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(rank[3:6, 0], [0.0, 0.5, 1.0])


def test_per_side_rolling_zscore_is_finite() -> None:
    arrays = _toy_arrays()

    rolling = per_side_depth_rolling_zscore(
        arrays["depth"],
        arrays["side_index"],
        arrays["features"][:, :2],
        window_samples=3,
        epsilon=1.0e-6,
    )

    assert rolling.shape == (arrays["depth"].size, 2)
    assert np.all(np.isfinite(rolling))


def test_derived_energy_features_do_not_use_labels() -> None:
    arrays = _toy_arrays()

    derived, names = derived_energy_features(
        arrays["features"],
        arrays["feature_names"],
        epsilon=1.0e-6,
    )

    assert names == ["log_late_over_early_ratio", "normalized_late_minus_early"]
    np.testing.assert_allclose(
        derived[:, 0],
        np.log1p(arrays["features"][:, -1]),
        atol=1.0e-6,
    )
    assert np.all(np.isfinite(derived))


def test_enhance_sample_table_features_appends_finite_features() -> None:
    arrays = _toy_arrays()

    updated, report = enhance_sample_table_features(
        arrays,
        config=FeatureNormalizationConfig(rolling_window_samples=3),
    )

    assert report.used_label_information_for_features is False
    assert report.errors == []
    assert report.added_feature_count == 26
    assert updated["transformed_features"].shape[1] == 32
    assert updated["base_transformed_feature_count"] == 6
    assert np.all(np.isfinite(updated["transformed_features"]))
    assert "per_depth_side_z_rms_energy" in updated["transformed_feature_names"].astype(str)
    assert "normalized_late_minus_early" in updated["transformed_feature_names"].astype(str)
