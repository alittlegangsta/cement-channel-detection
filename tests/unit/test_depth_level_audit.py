from __future__ import annotations

import numpy as np

from cement_channel.evaluation.depth_level_audit import (
    audit_depth_level_separation,
    comparison_masks,
)
from cement_channel.labels.depth_level_schema import parse_depth_level_label_config
from tests.unit.test_depth_level_schema import _valid_raw_config


def _config():
    raw = _valid_raw_config()
    raw["gate"]["sanity_effect_size_threshold"] = 0.2
    raw["gate"]["depth_level_improvement_effect_size_delta"] = 0.05
    raw["quality_policy"]["strong_positive"]["min_candidate_fraction"] = 0.2
    raw["quality_policy"]["review_intervals"][0]["depth_min_ft"] = 5700.0
    raw["quality_policy"]["review_intervals"][0]["depth_max_ft"] = 5710.0
    return parse_depth_level_label_config(raw)


def _label_arrays() -> dict[str, np.ndarray]:
    depth = np.arange(20, dtype=np.float32)
    has_channel = np.zeros(20, dtype=bool)
    has_channel[:10] = True
    strong = np.zeros(20, dtype=bool)
    strong[:6] = True
    clear = np.zeros(20, dtype=bool)
    clear[10:18] = True
    return {
        "depth": depth,
        "depth_has_channel_any": has_channel,
        "depth_strong_positive_mask": strong,
        "depth_clear_negative_mask": clear,
        "depth_review_band_mask": np.zeros(20, dtype=bool),
        "depth_label_confidence": np.ones(20, dtype=np.float32),
        "depth_orientation_confidence": np.ones(20, dtype=np.float32),
        "depth_plus_minus_disagreement_fraction": np.zeros(20, dtype=np.float32),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }


def _feature_arrays() -> dict[str, np.ndarray]:
    signal = np.zeros(20, dtype=np.float32)
    signal[:10] = 3.0
    noise = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    return {
        "depth": np.arange(20, dtype=np.float32),
        "depth_level_xsi_features": np.column_stack([signal, noise]).astype(np.float32),
        "depth_level_xsi_feature_names": np.asarray(
            ["side_mean_late_over_early_ratio", "receiver_mean_rms_energy"]
        ),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }


def test_comparison_masks_include_required_depth_level_reviews() -> None:
    labels = {
        key: np.asarray(value).reshape(-1)
        for key, value in _label_arrays().items()
        if key.startswith("depth_")
    }
    comparisons = comparison_masks(labels, _config())

    assert "depth_has_channel_vs_no_channel" in comparisons
    assert "strong_positive_vs_clear_negative" in comparisons
    assert "exclude_5700_band" in comparisons
    assert int(np.count_nonzero(comparisons["strong_positive_vs_clear_negative"]["candidate"])) == 6


def test_audit_depth_level_separation_detects_improvement_over_side_level() -> None:
    report, rows = audit_depth_level_separation(
        label_arrays=_label_arrays(),
        feature_arrays=_feature_arrays(),
        config=_config(),
        side_level_audit_report={
            "signal_enhancement": {"all_candidate_best_abs_effect_size": 0.1}
        },
    )

    assert report.errors == []
    assert rows
    assert report.depth_level_separation_enhanced is True
    assert report.depth_level_baseline_sanity_candidate is True
    assert report.side_level_target_likely_too_fine is True
    assert report.no_model_training is True
