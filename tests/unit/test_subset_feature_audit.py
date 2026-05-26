from __future__ import annotations

import numpy as np

from cement_channel.evaluation.subset_feature_audit import (
    audit_subset_feature_separation,
    feature_group_indices,
    signal_enhancement_summary,
    subset_pair_masks,
)
from cement_channel.labels.label_quality_schema import parse_label_quality_config
from tests.unit.test_label_quality_schema import _valid_raw_config


def _sample_and_subsets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = 20
    label = np.zeros(n, dtype=np.int8)
    label[:10] = 1
    side_feature = np.zeros(n, dtype=np.float32)
    side_feature[:10] = 0.2
    receiver_feature = np.zeros(n, dtype=np.float32)
    receiver_feature[:4] = [3.0, 4.0, 5.0, 4.0]
    receiver_feature[4:10] = 0.2
    receiver_feature[10:16] = [0.0, 0.1, 0.0, 0.2, 0.1, 0.0]
    quality_pos = np.zeros(n, dtype=bool)
    quality_pos[:4] = True
    quality_neg = np.zeros(n, dtype=bool)
    quality_neg[10:16] = True
    strong = np.zeros(n, dtype=bool)
    strong[:8] = True
    clear = np.zeros(n, dtype=bool)
    clear[10:] = True
    sample = {
        "label_presence_plus": label,
        "transformed_features": np.column_stack([side_feature, receiver_feature]).astype(
            np.float32
        ),
        "transformed_feature_names": np.asarray(
            [
                "per_depth_side_z_late_over_early_ratio",
                "robust_scaled_receiver_far_minus_near_late_over_early_ratio",
            ]
        ),
        "receiver_transformed_feature_names_added": np.asarray(
            ["robust_scaled_receiver_far_minus_near_late_over_early_ratio"]
        ),
    }
    subsets = {
        "disagreement_free_mask": np.ones(n, dtype=bool),
        "high_confidence_orientation_mask": np.ones(n, dtype=bool),
        "connected_object_only_mask": strong,
        "review_exclusion_mask": np.zeros(n, dtype=bool),
        "strong_positive_mask": strong,
        "clear_negative_mask": clear,
        "quality_strong_positive_mask": quality_pos,
        "quality_clear_negative_mask": quality_neg,
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }
    return sample, subsets


def test_feature_group_indices_separate_side_and_receiver_features() -> None:
    sample, _subsets = _sample_and_subsets()

    groups = feature_group_indices(sample["transformed_feature_names"], sample)

    np.testing.assert_array_equal(groups["side_level_enhanced"], [0])
    np.testing.assert_array_equal(groups["receiver_derived"], [1])
    assert set(groups) >= {"late_over_early", "far_near_receiver"}


def test_subset_pair_masks_build_quality_strong_vs_clear_pair() -> None:
    sample, subsets = _sample_and_subsets()

    pairs = subset_pair_masks(sample["label_presence_plus"], subsets)

    assert int(np.count_nonzero(pairs["quality_strong_vs_clear"]["candidate"])) == 4
    assert int(np.count_nonzero(pairs["quality_strong_vs_clear"]["negative"])) == 6


def test_audit_subset_feature_separation_detects_quality_signal_enhancement() -> None:
    sample, subsets = _sample_and_subsets()
    raw = _valid_raw_config()
    raw["gate"]["signal_enhancement_effect_size_delta"] = 0.01
    raw["gate"]["strong_signal_effect_size_threshold"] = 0.1

    report, rows = audit_subset_feature_separation(
        sample_arrays=sample,
        subset_arrays=subsets,
        config=parse_label_quality_config(raw),
    )

    assert report.errors == []
    assert rows
    assert report.label_noise_likely is True
    assert report.no_model_training is True


def test_signal_enhancement_summary_can_remain_weak() -> None:
    raw = _valid_raw_config()
    summaries = [
        {
            "subset_name": "all_candidates_vs_non_candidates",
            "top_abs_standardized_difference": 0.20,
        },
        {
            "subset_name": "quality_strong_vs_clear",
            "top_abs_standardized_difference": 0.21,
        },
    ]

    result = signal_enhancement_summary(
        summaries,
        config=parse_label_quality_config(raw),
    )

    assert result["label_noise_likely"] is False
