from __future__ import annotations

from cement_channel.evaluation.geometry_regression_autonomous_review import (
    build_autonomous_decision,
)


def _reports() -> dict[str, dict]:
    return {
        "cast_qc": {
            "negative_zc_target_influence_significant": False,
            "global_cell_counts": {"zc_lt_0_count": 2, "nonfinite_count": 0},
        },
        "target_view": {
            "view_summaries": [
                {
                    "target_view": "receiver_max",
                    "single_receiver_extreme_sensitivity_warning": True,
                },
                {"target_view": "full_360_fraction", "saturation_warning": True},
            ]
        },
        "morphology": {
            "simple_low_zc_relationships": [
                {
                    "geometry_kernel": "triangular_midpoint_weighted",
                    "morphology_field": "largest_connected_component_fraction",
                    "pearson_vs_simple_low_zc_mean": 0.9,
                }
            ]
        },
        "depth_regime": {
            "regime_shift_flags": [
                "target_fold_regime_shift",
                "feature_fold_regime_shift",
                "morphology_fold_shift",
            ],
            "target_shift_summary": [
                {"geometry_kernel": "r7_reference_point", "shift_flag": True},
                {"geometry_kernel": "midpoint_window", "shift_flag": True},
                {"geometry_kernel": "uniform_source_receiver_interval", "shift_flag": True},
                {"geometry_kernel": "triangular_midpoint_weighted", "shift_flag": True},
            ],
        },
        "near_far": {
            "denominator_small_derivable": False,
            "feature_quantiles": [
                {
                    "feature_name": "near_far_ratio_mean_early_energy",
                    "extreme_value_count": 5,
                }
            ],
            "divergence_summary": [
                {
                    "global_divergence_flag": True,
                    "fold_divergence_flags": [False, False, False],
                }
            ],
        },
    }


def test_autonomous_decision_prioritizes_depth_regime_approval() -> None:
    decision = build_autonomous_decision(_reports())

    assert decision["decision"] == "stop_request_depth_regime_stratification_approval"
    answers = decision["answers"]
    assert answers["cast_zc_lt_0_significantly_affects_target"] is False
    assert answers["fold_0_1_2_significant_regime_shift"] is True
    assert answers["stage_10_stop_still_valid"] is True
    assert answers["stage_11_12_still_blocked"] is True
    assert (
        answers["mvp4c_stc_apes_deep_learning_final_labels_still_forbidden"] is True
    )
