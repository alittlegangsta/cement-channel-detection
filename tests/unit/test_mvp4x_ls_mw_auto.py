from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_ls_mw_auto import (
    build_evaluation_set_policy,
    build_ls_mw_decision,
    build_parallel_label_candidates,
    diagnose_pilot_metric_collapse,
)


def _snapshot(sample_count: int = 36) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    regime = np.asarray(["A", "B", "C"] * (sample_count // 3), dtype="<U16")
    support = np.asarray(
        ["tier_4_unsupported_A", "tier_1_primary_supported", "tier_2_supported"]
        * (sample_count // 3),
        dtype="<U64",
    )
    morphology = np.tile(x.reshape(-1, 1), (1, 13)).astype(np.float32)
    return {
        "depth": np.linspace(1000.0, 1200.0, sample_count, dtype=np.float32),
        "xsi_features": np.column_stack([x, x**2, 1.0 - x]).astype(np.float32),
        "xsi_feature_names": np.asarray(["f0", "f1", "f2"]),
        "xsi_feature_group": np.asarray(["g", "g", "g"]),
        "model_feature_mask": np.asarray([True, True, True]),
        "target_kernel": np.asarray("triangular_midpoint_weighted"),
        "receiver_mean": x,
        "receiver_p90": np.clip(x + 0.1, 0.0, 1.0),
        "receiver_max": np.clip(x + 0.2, 0.0, 1.0),
        "full_360_fraction": x,
        "receiver_std": x * 0.1,
        "broad_regime_id": regime,
        "any_special_flag": x > 0.8,
        "low_orientation_confidence_flag": x < 0.2,
        "morphology_largest_connected_component_fraction": morphology,
        "morphology_max_azimuth_channel_fraction": morphology,
        "morphology_max_relative_drop": morphology,
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }, {"support_tier": support}


def test_parallel_label_candidates_are_audit_only_and_do_not_replace_v1() -> None:
    snapshot, final_scores = _snapshot()
    morphology_v3 = {
        "continuity_weighted_fraction": np.linspace(0.0, 1.0, 36, dtype=np.float32)
    }

    output = build_parallel_label_candidates(
        snapshot=snapshot,
        morphology_v3=morphology_v3,
        final_scores=final_scores,
    )

    assert output["report"]["v1_not_overwritten"] is True
    assert output["report"]["all_candidates_audit_only_except_v1_reference"] is True
    assert "receiver_mean_v1_reference" in output["arrays"]
    assert np.allclose(output["arrays"]["receiver_mean_v1_reference"], snapshot["receiver_mean"])
    assert output["report"]["targets"]["local_worst_sector_fraction"]["audit_only"] is True


def test_evaluation_set_policy_separates_population_supported_and_stress() -> None:
    snapshot, final_scores = _snapshot()

    policy = build_evaluation_set_policy(snapshot=snapshot, final_scores=final_scores)

    sets = policy["sets"]
    assert sets["population_like_screening_evaluation"]["sample_count"] == 36
    assert sets["supported_cohort_evaluation"]["sample_count"] == 24
    assert sets["stress_test_audit_set"]["sample_count"] > 0
    assert policy["rules"]["do_not_mix_stress_test_into_population_claims"] is True


def test_pilot_metric_collapse_marks_stress_set_not_bug() -> None:
    sample_count = 36
    y = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    matched = {
        "target__receiver_mean": y,
        "depth_center": np.linspace(1000.0, 1200.0, sample_count, dtype=np.float32),
        "features__existing_features_only": np.column_stack([1.0 - y, (1.0 - y) ** 2]),
        "selection_category": np.asarray(["stress"] * sample_count),
        "broad_regime_id": np.asarray(["A", "B", "C"] * 12),
        "support_tier": np.asarray(["tier_4", "tier_1", "tier_2"] * 12),
        "any_special_flag": np.asarray([False, True, False] * 12),
    }

    diagnostic = diagnose_pilot_metric_collapse(matched=matched)

    assert diagnostic["implementation_bug_detected"] is False
    assert diagnostic["pilot_selected_intervals_are_stress_test_not_population_evaluation"] is True
    assert diagnostic["top_k_denominator"]["target_top_count"] > 0


def test_decision_requests_human_review_before_multiwell_route() -> None:
    decision = build_ls_mw_decision(
        collapse={"implementation_bug_detected": False},
        policy={"sets": {}},
        candidates={"report": {"targets": {"a": {}, "b": {}}}},
        screening={"report": {"row_count": 10}},
        inventory={
            "well_count": 1,
            "single_well_only_multiwell_data_required_for_generalization": True,
        },
    )

    assert decision["decision"] == "request_human_review_parallel_label_candidates"
    assert decision["recommend_multiwell_onboarding"] is True
