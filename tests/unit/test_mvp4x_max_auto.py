from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_max_auto import (
    build_morphology_candidate,
    build_support_tier,
    rank_percentile_by_group,
)


def test_build_support_tier_keeps_unsupported_audit_only() -> None:
    selected = np.asarray(["P1", "P0", "", ""], dtype=str)
    regimes = np.asarray(["B", "C", "A", "B"], dtype=str)
    low = np.asarray([False, True, True, True])
    supported = np.asarray([True, True, False, False])

    tiers = build_support_tier(selected, regimes, low, supported)

    assert tiers[0] == "tier_1_primary_supported_pooled_bc_high_orientation"
    assert tiers[1] == "tier_2_supported_pooled_bc_all_fallback"
    assert tiers[2] == "tier_4_unsupported_regime_a_audit_only"
    assert tiers[3] == "tier_4_unsupported_low_orientation_audit_only"


def test_rank_percentile_by_group_does_not_rank_unsupported() -> None:
    score = np.asarray([0.3, 0.1, 0.2, np.nan], dtype=np.float32)
    group = np.asarray(["B", "B", "C", "C"], dtype=str)
    supported = np.asarray([True, True, True, False])

    ranked = rank_percentile_by_group(score, group, supported)

    assert ranked[0] == 1.0
    assert ranked[1] == 0.0
    assert ranked[2] == 1.0
    assert np.isnan(ranked[3])


def test_build_morphology_candidate_is_parallel_audit_only() -> None:
    snapshot = {
        "depth": np.asarray([1.0, 2.0], dtype=np.float32),
        "broad_regime_id": np.asarray(["B", "C"]),
        "receiver_mean": np.asarray([0.1, 0.2], dtype=np.float32),
        "target_kernel_index": np.asarray(0),
        "morphology_largest_connected_component_fraction": np.asarray(
            [[0.2], [0.8]], dtype=np.float32
        ),
        "morphology_max_azimuth_channel_fraction": np.asarray([[0.3], [0.9]], dtype=np.float32),
        "morphology_max_relative_drop": np.asarray([[0.4], [0.7]], dtype=np.float32),
        "morphology_candidate_cell_count": np.asarray([[2], [8]], dtype=np.float32),
        "morphology_total_cell_count": np.asarray([[10], [10]], dtype=np.float32),
        "label_confidence": np.asarray([1.0, 1.0], dtype=np.float32),
    }

    candidate = build_morphology_candidate(snapshot)

    assert candidate["report"]["audit_only"]
    assert candidate["report"]["parallel_comparison_only"]
    assert candidate["morphology_candidate_v2"].shape == (2,)
    assert candidate["morphology_candidate_v2"][1] > candidate["morphology_candidate_v2"][0]
