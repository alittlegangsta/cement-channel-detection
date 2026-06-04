from __future__ import annotations

import json

import numpy as np

from cement_channel.modeling.mvp4x_stratified_baselines import (
    decide_time_frequency_v2,
    load_design,
)


def test_load_design_maps_names_to_masks() -> None:
    metadata = {
        "support": {"regime_a_high_orientation_support": 0},
        "policy": {"no_formal_cohort_production_filter": True},
    }
    design_npz = {
        "cohort_names": np.asarray(["all_samples", "regime_b_high_orientation"]),
        "cohort_masks": np.asarray([[True, True, False], [False, True, False]]),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }

    design = load_design(design_npz)

    assert design["cohort_names"] == ["all_samples", "regime_b_high_orientation"]
    assert design["masks"]["all_samples"].sum() == 2
    assert design["masks"]["regime_b_high_orientation"].sum() == 1
    assert design["metadata"]["policy"]["no_formal_cohort_production_filter"]


def test_time_frequency_v2_not_triggered_before_human_route_review() -> None:
    result = decide_time_frequency_v2(
        matrix={
            "best_by_cohort": {
                "regime_bc_high_orientation": {
                    "feature_set": "existing_features_only",
                    "spearman": 0.25,
                }
            }
        },
        strict={
            "by_cohort": {
                "regime_b_high_orientation": {
                    "best_candidate": {"spearman": 0.2},
                    "permutation_margins": {
                        "global": 0.2,
                        "within_depth_bin": 0.2,
                        "block": 0.2,
                    },
                }
            }
        },
        transfer={"domain_shift_warnings": ["B_to_C_weak_transfer_spearman"]},
        target_feature={"status": "completed"},
    )

    assert not result["time_frequency_v2_triggered"]
    assert result["no_raw_waveform_reread"]
    assert result["time_frequency_v2_improved"] == "not_triggered"
