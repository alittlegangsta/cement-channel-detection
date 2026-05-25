from __future__ import annotations

import numpy as np

from cement_channel.training.sample_weights import (
    SampleWeightPolicyConfig,
    build_azimuthal_valid_mask,
    build_policy_weights,
    build_reliability,
    rebuild_sample_weights,
    summarize_policy_weights,
)


def _toy_arrays() -> dict[str, np.ndarray]:
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    confidence = np.array([0.95, 0.90, 0.85, 0.80, 0.0, 0.55, 0.56, 0.57], dtype=np.float32)
    return {
        "depth": np.arange(labels.size, dtype=np.float32) * 10.0,
        "label_presence_plus": labels,
        "label_confidence_plus": confidence,
        "valid_for_azimuthal_validation": np.ones(labels.size, dtype=bool),
        "plus_minus_disagreement": np.array(
            [False, False, False, False, False, True, False, False]
        ),
        "depth_match_error": np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "exclude_large_depth_match_error": np.array(
            [False, False, True, False, False, False, False, False]
        ),
        "sample_weight": np.ones(labels.size, dtype=np.float32),
    }


def test_class_balanced_policy_caps_candidate_weight_fraction() -> None:
    arrays = _toy_arrays()
    config = SampleWeightPolicyConfig(
        min_label_confidence=0.5,
        max_depth_match_error_ft=0.5,
        target_candidate_weight_fraction=0.5,
        max_candidate_weight_fraction=0.6,
        n_splits=2,
        depth_block_size_ft=20.0,
        min_gap_ft=0.0,
        min_samples_per_class_per_fold=1,
    )

    updated, report = rebuild_sample_weights(arrays, config=config)

    confidence_fraction = report.policy_summary["confidence_only"][
        "candidate_effective_weight_fraction"
    ]
    capped_fraction = report.policy_summary["capped_class_balanced_confidence"][
        "candidate_effective_weight_fraction"
    ]
    assert isinstance(confidence_fraction, float)
    assert isinstance(capped_fraction, float)
    assert confidence_fraction > 0.6
    assert capped_fraction <= 0.6
    assert np.array_equal(
        updated["sample_weight"],
        updated["sample_weight_capped_class_balanced_confidence"],
    )
    assert updated["sample_weight"][2] == 0.0
    assert updated["sample_weight"][4] == 0.0
    assert report.errors == []


def test_valid_mask_zeroes_low_confidence_and_large_depth_error() -> None:
    arrays = _toy_arrays()
    config = SampleWeightPolicyConfig(min_label_confidence=0.5, max_depth_match_error_ft=0.5)

    mask = build_azimuthal_valid_mask(
        labels=arrays["label_presence_plus"],
        confidence=arrays["label_confidence_plus"],
        valid_for_azimuthal=arrays["valid_for_azimuthal_validation"],
        depth_match_error=arrays["depth_match_error"],
        large_depth_error=arrays["exclude_large_depth_match_error"],
        config=config,
    )

    assert mask.tolist() == [True, True, False, True, False, True, True, True]


def test_disagreement_can_be_excluded() -> None:
    arrays = _toy_arrays()
    config = SampleWeightPolicyConfig(disagreement_policy="exclude")
    mask = np.ones(arrays["label_presence_plus"].shape, dtype=bool)

    reliability = build_reliability(
        confidence=arrays["label_confidence_plus"],
        labels=arrays["label_presence_plus"],
        valid_mask=mask,
        disagreement=arrays["plus_minus_disagreement"],
        config=config,
    )

    assert reliability[5] == 0.0
    assert reliability[0] > 0.0


def test_unweighted_policy_removes_confidence_class_skew() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    reliability = np.array([1.0, 1.0, 0.1, 0.1], dtype=np.float32)
    config = SampleWeightPolicyConfig(n_splits=2)

    policies = build_policy_weights(
        labels=labels,
        reliability=reliability,
        disagreement=np.zeros(labels.shape, dtype=bool),
        config=config,
    )
    summary = summarize_policy_weights(
        labels,
        np.zeros(labels.shape, dtype=bool),
        np.ones(labels.shape, dtype=bool),
        policies["unweighted"],
    )

    assert summary["candidate_effective_weight_fraction"] == 0.5
