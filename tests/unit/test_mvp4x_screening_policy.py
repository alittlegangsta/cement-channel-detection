from __future__ import annotations

import numpy as np

from cement_channel.experiments.mvp4x_screening_policy import (
    build_screening_cohorts,
    build_screening_policy,
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


def _flags() -> dict[str, bool]:
    return {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
    }


def test_screening_cohorts_capture_supported_and_audit_sets() -> None:
    masks = build_screening_cohorts(_snapshot())

    assert masks["pooled_bc_all"].sum() == 4738
    assert masks["pooled_bc_high_orientation"].sum() == 3412
    assert masks["regime_b_high_orientation"].sum() == 1043
    assert masks["regime_c_all"].sum() == 2369
    assert masks["regime_a_all"].sum() == 2370


def test_screening_policy_keeps_deployment_forbidden() -> None:
    flags = _flags()
    outputs = build_screening_policy(
        snapshot=_snapshot(),
        waveform={
            "waveform_depth_features": np.ones((7108, 342), dtype=np.float32),
            **{key: np.asarray(value) for key, value in flags.items()},
        },
        regime_policy={**flags},
        robustness={"candidate_rows": [], **flags},
        ranking={"rows": [], **flags},
        error={
            "summary": {"max_abs_calibration_bias": 0.08},
            **flags,
        },
        screening={
            "supported_cohorts": ["pooled_bc_high_orientation"],
            "unsupported_cohorts": ["regime_b_all"],
            "domain_shift": True,
            **flags,
        },
        regime_decision={
            "decision": "research_screening_baseline_domain_shift_limited",
            "answers": {"18_stable_features": []},
            **flags,
        },
        config={"screening_baseline": {"random_seed": 1}},
        inputs={"snapshot_npz": "snapshot.npz"},
    )

    assert outputs.policy["research_only"]
    assert outputs.policy["not_validated_for_deployment"]
    assert outputs.policy["primary_research_score"]["target"] == "receiver_mean"
    assert outputs.policy["unsupported_uses"]["production_inference"]
    assert "production model" in outputs.policy["forbidden_claims"]
