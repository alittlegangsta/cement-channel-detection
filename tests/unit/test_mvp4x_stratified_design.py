from __future__ import annotations

import numpy as np

from cement_channel.experiments.mvp4x_stratified_design import (
    build_stratified_cohorts,
    build_stratified_design,
)


def _snapshot() -> dict[str, np.ndarray]:
    depth = np.asarray([1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350], dtype=np.float32)
    low = np.asarray([True, True, False, False, True, False, False, False])
    regimes = np.asarray(["A", "A", "B", "B", "B", "C", "C", "C"])
    features = np.arange(depth.size * 4, dtype=np.float32).reshape(depth.size, 4)
    return {
        "depth": depth,
        "low_orientation_confidence_flag": low,
        "broad_regime_id": regimes,
        "saturation_platform_flag": np.zeros(depth.size, dtype=bool),
        "special_5680_flag": np.asarray([False, False, False, False, False, False, True, False]),
        "any_special_flag": np.asarray([False, False, False, False, False, False, True, False]),
        "xsi_features": features,
        "target_kernel": np.asarray("triangular_midpoint_weighted"),
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
        "receiver_mean": np.linspace(0.0, 0.2, depth.size, dtype=np.float32),
        "receiver_p90": np.linspace(0.0, 0.3, depth.size, dtype=np.float32),
        "receiver_max": np.linspace(0.0, 0.4, depth.size, dtype=np.float32),
        "full_360_fraction": np.linspace(0.0, 0.5, depth.size, dtype=np.float32),
        "receiver_std": np.linspace(0.0, 0.1, depth.size, dtype=np.float32),
    }


def _waveform() -> dict[str, np.ndarray]:
    return {
        "waveform_depth_features": np.ones((8, 2), dtype=np.float32),
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }


def _valid_snapshot() -> dict[str, np.ndarray]:
    sample_count = 7108
    depth = np.linspace(2350.0, 5760.0, sample_count, dtype=np.float32)
    regimes = np.asarray(["A"] * 2370 + ["B"] * 2369 + ["C"] * 2369)
    low = np.zeros(sample_count, dtype=bool)
    low[:2370] = True
    low[2370 : 2370 + 1326] = True
    features = np.ones((sample_count, 80), dtype=np.float32)
    targets = np.linspace(0.0, 0.2, sample_count, dtype=np.float32)
    return {
        "depth": depth,
        "low_orientation_confidence_flag": low,
        "broad_regime_id": regimes,
        "saturation_platform_flag": np.zeros(sample_count, dtype=bool),
        "special_5680_flag": np.zeros(sample_count, dtype=bool),
        "any_special_flag": np.zeros(sample_count, dtype=bool),
        "xsi_features": features,
        "target_kernel": np.asarray("triangular_midpoint_weighted"),
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
        "receiver_mean": targets,
        "receiver_p90": targets * 1.2,
        "receiver_max": targets * 1.5,
        "full_360_fraction": targets * 0.5,
        "receiver_std": targets * 0.1,
    }


def _valid_waveform() -> dict[str, np.ndarray]:
    sample_count = 7108
    return {
        "waveform_depth_features": np.ones((sample_count, 342), dtype=np.float32),
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }


def _cf_reports() -> dict[str, dict[str, object]]:
    base = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }
    return {
        "cf_decision_json": {
            **base,
            "decision": "exploratory_regime_specific_signal_only",
            "answers": {"1_orientation_effect_identifiable": False},
        },
        "cf_common_support_json": base,
        "cf_nuisance_json": base,
        "cf_spatial_json": base,
    }


def test_build_stratified_cohorts_defines_requested_support() -> None:
    masks = build_stratified_cohorts(_snapshot())

    assert masks["all_samples"].sum() == 8
    assert masks["high_orientation_cohort"].sum() == 5
    assert masks["low_orientation_cohort"].sum() == 3
    assert masks["regime_a_low_orientation"].sum() == 2
    assert masks["regime_b_low_orientation"].sum() == 1
    assert masks["regime_b_high_orientation"].sum() == 2
    assert masks["regime_c_high_orientation"].sum() == 3
    assert masks["regime_bc_high_orientation_exclude_5680"].sum() == 4


def test_build_stratified_design_records_research_policy() -> None:
    outputs = build_stratified_design(
        snapshot=_valid_snapshot(),
        waveform=_valid_waveform(),
        cf_reports=_cf_reports(),
        config={
            "stratified_design": {
                "cohorts": [
                    "all_samples",
                    "regime_a_low_orientation",
                    "regime_b_high_orientation",
                    "regime_c_high_orientation",
                ]
            }
        },
        inputs={"snapshot_npz": "snapshot.npz"},
    )

    assert outputs.cohort_masks.shape == (4, 7108)
    assert outputs.design["research_only"]
    assert outputs.design["policy"]["regime_boundary_research_evaluation_only"]
    assert outputs.design["policy"]["orientation_only_cohort_definition"]
    assert outputs.design["support"]["regime_a_high_orientation_support"] == 0
    assert outputs.design["support"]["regime_c_low_orientation_support"] == 0
