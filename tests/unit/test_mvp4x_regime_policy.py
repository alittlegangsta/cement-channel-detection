from __future__ import annotations

import numpy as np

from cement_channel.experiments.mvp4x_regime_policy import (
    build_policy_cohorts,
    build_regime_policy,
)


def _snapshot() -> dict[str, np.ndarray]:
    sample_count = 7108
    regimes = np.asarray(["A"] * 2370 + ["B"] * 2369 + ["C"] * 2369)
    low = np.zeros(sample_count, dtype=bool)
    low[:2370] = True
    low[2370 : 2370 + 1326] = True
    return {
        "depth": np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32),
        "xsi_features": np.ones((sample_count, 80), dtype=np.float32),
        "target_kernel": np.asarray("triangular_midpoint_weighted"),
        "broad_regime_id": regimes,
        "low_orientation_confidence_flag": low,
        "any_special_flag": np.zeros(sample_count, dtype=bool),
        "receiver_mean": np.linspace(0.0, 0.2, sample_count, dtype=np.float32),
        "receiver_p90": np.linspace(0.0, 0.2, sample_count, dtype=np.float32),
        "receiver_max": np.linspace(0.0, 0.2, sample_count, dtype=np.float32),
        "full_360_fraction": np.zeros(sample_count, dtype=np.float32),
        "receiver_std": np.ones(sample_count, dtype=np.float32) * 0.01,
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }


def test_policy_cohorts_capture_support_gaps() -> None:
    masks = build_policy_cohorts(_snapshot())

    assert masks["regime_a_high_orientation"].sum() == 0
    assert masks["regime_b_low_orientation"].sum() == 1326
    assert masks["regime_b_high_orientation"].sum() == 1043
    assert masks["regime_c_low_orientation"].sum() == 0
    assert masks["regime_c_high_orientation"].sum() == 2369


def test_regime_policy_marks_no_production_claims() -> None:
    report_flags = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }
    outputs = build_regime_policy(
        snapshot=_snapshot(),
        waveform={
            "waveform_depth_features": np.ones((7108, 342), dtype=np.float32),
            **{key: np.asarray(value) for key, value in report_flags.items()},
        },
        cf_decision={"decision": "x", **report_flags},
        cf_spatial={"status": "completed", **report_flags},
        stratified_decision={
            "decision": "exploratory_regime_specific_baseline_supported",
            "answers": {"8_most_stable_target_view": "receiver_mean"},
            **report_flags,
        },
        stratified_iteration_log="iteration",
        config={"regime_policy": {"version": "test"}},
        inputs={"snapshot_npz": "snapshot.npz"},
    )

    assert outputs.policy["research_only"]
    assert (
        outputs.policy["regime_policy"]["orientation"]["orientation_filter_formally_approved"]
        is False
    )
    assert (
        outputs.policy["regime_policy"]["model_use"]["absolute_channel_fraction_prediction"]
        is False
    )
    assert outputs.policy["feature_contract"]["combined_features"] == 422
