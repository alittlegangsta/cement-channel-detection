from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_regime_robustness import (
    bootstrap_metric_ci,
    select_robustness_candidates,
    summarize_values,
)


def test_select_robustness_candidates_maps_pooled_aliases() -> None:
    report = {
        "model_matrix": {
            "best_by_cohort": {
                "regime_bc_all": {
                    "feature_set": "waveform_features_only",
                    "target": "receiver_mean",
                    "model": "Ridge",
                    "spearman": 0.2,
                    "mae": 0.1,
                    "r2": -0.2,
                }
            }
        }
    }

    selected = select_robustness_candidates(
        stratified_baselines=report,
        requested_cohorts=["pooled_bc_all"],
    )

    assert selected["pooled_bc_all"]["source_cohort"] == "regime_bc_all"
    assert selected["pooled_bc_all"]["feature_set"] == "waveform_features_only"


def test_bootstrap_metric_ci_reports_positive_spearman_interval() -> None:
    y_true = np.arange(30, dtype=np.float32)
    y_pred = y_true + 0.1

    result = bootstrap_metric_ci(
        y_true=y_true,
        y_pred=y_pred,
        repeats=20,
        seed=7,
        ci_quantiles=[0.025, 0.975],
    )

    assert result["status"] == "completed"
    assert result["metrics"]["spearman"]["ci_lower"] > 0.9
    assert result["metrics"]["mae"]["mean"] > 0.0


def test_summarize_values_ignores_none() -> None:
    summary = summarize_values([None, 1.0, 2.0])

    assert summary["mean"] == 1.5
    assert summary["min"] == 1.0
