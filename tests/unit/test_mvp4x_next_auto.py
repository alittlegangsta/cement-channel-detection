from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_next_auto import (
    build_morphology_v003_variants,
    variant_gate_summary,
)


def test_build_morphology_v003_variants_uses_pre_registered_formulas() -> None:
    snapshot = {
        "depth": np.asarray([1.0, 2.0], dtype=np.float32),
        "target_kernel_index": np.asarray(0),
        "morphology_largest_connected_component_fraction": np.asarray(
            [[0.5], [0.0]], dtype=np.float32
        ),
        "morphology_max_azimuth_channel_fraction": np.asarray([[0.25], [1.0]], dtype=np.float32),
        "morphology_max_relative_drop": np.asarray([[0.1], [0.2]], dtype=np.float32),
        "morphology_candidate_cell_count": np.asarray([[2.0], [8.0]], dtype=np.float32),
        "morphology_total_cell_count": np.asarray([[10.0], [10.0]], dtype=np.float32),
        "label_confidence": np.asarray([1.0, 0.0], dtype=np.float32),
    }

    variants = build_morphology_v003_variants(snapshot)

    np.testing.assert_allclose(
        variants["continuity_weighted_fraction"],
        np.asarray([0.32, 0.48], dtype=np.float32),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        variants["broad_channel_weighted_fraction"],
        np.asarray([0.22, 0.88], dtype=np.float32),
        rtol=1.0e-6,
    )
    assert (
        variants["confidence_gated_combined_fraction"][0]
        > variants["confidence_gated_combined_fraction"][1]
    )


def test_variant_gate_summary_requires_all_improvement_checks() -> None:
    baseline = {
        "oof_metrics": {"spearman": 0.10},
        "ranking": {"top_k": {"top_10pct": {"lift": 1.0}}},
        "gap_stability": {"spearman_std": 0.01},
        "domain_shift_min_transfer_spearman": 0.05,
    }
    variant = {
        "oof_metrics": {"spearman": 0.14},
        "ranking": {"top_k": {"top_10pct": {"lift": 1.25}}},
        "gap_stability": {"spearman_std": 0.015},
        "permutation_margins": {
            "global": 0.01,
            "within_depth_bin": 0.02,
            "block": 0.03,
        },
        "domain_shift_min_transfer_spearman": 0.04,
    }

    summary = variant_gate_summary("candidate", variant, baseline)

    assert summary["passes_all_criteria"]
    assert summary["delta_spearman"] > 0.03
