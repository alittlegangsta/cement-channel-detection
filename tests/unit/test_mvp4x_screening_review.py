from __future__ import annotations

from cement_channel.visualization.mvp4x_screening_review import (
    build_decision,
    build_policy_comparison,
)


def test_policy_comparison_contains_requested_options() -> None:
    flags = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }
    robustness = {
        "candidate_rows": [
            {
                "cohort": "pooled_bc_high_orientation",
                "sample_count": 10,
                "repeated_cv": {"metrics": {"spearman": {"mean": 0.2}, "r2": {"mean": -0.1}}},
                "bootstrap_ci": {"metrics": {"spearman": {"ci_lower": 0.1}}},
                "blocked_gap": {},
                "permutation": {"margins": {"global": 0.2}},
                "special_band_sensitivity": {"dependency_flag": False},
            }
        ],
        **flags,
    }
    ranking = {
        "rows": [
            {
                "cohort": "pooled_bc_high_orientation",
                "ranking": {"top_k": {"top_10pct": {"lift": 2.0}}},
                "derived_ordinal_audit": {"macro_f1": 0.4},
            }
        ],
        **flags,
    }
    rows = build_policy_comparison(
        robustness,
        ranking,
        {"summary": {"domain_shift_warning": True}, **flags},
    )

    assert {row["policy_id"] for row in rows} == {"P0", "P1", "P2", "P3", "P4"}
    assert rows[1]["cohort"] == "pooled_bc_high_orientation"


def test_decision_requests_formal_policy_when_domain_shift_exists() -> None:
    decision = build_decision(
        scores={"sample_counts": {"supported_scored": 1}, "gap_stability": {"warning_count": 0}},
        policy={"supported_screening_cohorts": ["x"], "unsupported_or_audit_only_cohorts": []},
        comparison_rows=[
            {"top_10_lift": 1.0},
            {"top_10_lift": 2.0},
        ],
        error={"summary": {"domain_shift_warning": True}},
        model_manifest={"not_validated_for_deployment": True},
        regime_decision={"decision": "research_screening_baseline_domain_shift_limited"},
    )

    assert decision["decision"] == "request_formal_regime_policy_approval"
    assert decision["not_validated_for_deployment"]
