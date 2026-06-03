from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_baselines import (
    MODEL_NAMES,
    compute_regression_metrics,
    contiguous_depth_folds,
    run_mvp4x_baselines,
)


def _snapshot() -> dict[str, np.ndarray]:
    depth = np.arange(12, dtype=np.float32)
    features = np.column_stack([depth, np.sin(depth)]).astype(np.float32)
    target = depth / 10.0
    return {
        "depth": depth,
        "xsi_features": features,
        "xsi_feature_names": np.asarray(["side_mean_energy", "receiver_std_rms"]),
        "model_feature_mask": np.asarray([True, True]),
        "receiver_p90": target.astype(np.float32),
        "receiver_mean": target.astype(np.float32),
        "receiver_max": target.astype(np.float32),
        "broad_regime_id": np.asarray(["A"] * 4 + ["B"] * 4 + ["C"] * 4),
        "saturation_platform_flag": np.zeros(12, dtype=bool),
        "special_5680_flag": np.zeros(12, dtype=bool),
        "low_orientation_confidence_flag": np.zeros(12, dtype=bool),
        "any_special_flag": np.zeros(12, dtype=bool),
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }


def _config() -> dict:
    return {
        "baseline": {
            "target_views": ["receiver_p90"],
            "n_contiguous_folds": 3,
            "permutation_count": 2,
        }
    }


def test_compute_regression_metrics_reports_rank_signal() -> None:
    metrics = compute_regression_metrics(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.1, 1.1, 2.1]),
    )

    assert metrics["mae"] > 0.0
    assert metrics["spearman"] == 1.0


def test_contiguous_depth_folds_cover_selected_samples_once() -> None:
    folds = contiguous_depth_folds(
        np.arange(9, dtype=np.float32),
        np.ones(9, dtype=bool),
        n_folds=3,
    )

    assert len(folds) == 3
    assert np.sum(np.column_stack(folds), axis=1).tolist() == [1] * 9


def test_run_mvp4x_baselines_skips_when_sklearn_unavailable() -> None:
    report, rows = run_mvp4x_baselines(
        snapshot=_snapshot(),
        config=_config(),
        inputs={"snapshot_npz": "synthetic.npz"},
        feature_set_name="existing_features_only",
        feature_matrix_key="xsi_features",
        feature_name_key="xsi_feature_names",
    )

    if report.sklearn_available:
        assert report.model_backend == "scikit_learn"
    else:
        assert report.model_backend == "sklearn_unavailable_all_skipped"
        assert report.decision["stage3_recommended"] is True
        assert len(rows) == len(MODEL_NAMES)
    assert report.no_ground_truth_claim is True

