from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.labels.label_quality_schema import (
    MVP4B_LABEL_QUALITY_CONFIG_VERSION,
    active_review_intervals,
    load_label_quality_config,
    parse_label_quality_config,
    validate_label_quality_config,
)


def _valid_raw_config() -> dict:
    return {
        "config_version": MVP4B_LABEL_QUALITY_CONFIG_VERSION,
        "input_sample_table": "baseline_sample_table_receiver_enhanced_v001",
        "optional_cast_weak_labels": "cast_weak_label_candidates_v001",
        "primary_label": "plus",
        "audit_label": "minus_ablation",
        "label_status": "human_reviewed_candidate_v001",
        "subsets": {
            "strong_positive": {
                "label_presence_plus": 1,
                "min_severity": 2,
                "min_label_confidence": 0.5,
                "require_no_plus_minus_disagreement": True,
                "max_depth_match_error_ft": 0.5,
            },
            "clear_negative": {
                "label_presence_plus": 0,
                "min_label_confidence": 0.5,
                "require_no_plus_minus_disagreement": True,
                "max_depth_match_error_ft": 0.5,
            },
            "high_confidence_orientation": {
                "min_orientation_confidence": 0.7,
            },
            "connected_object_only": {
                "min_area_samples": 6,
                "min_depth_length_ft": 1.0,
                "circular_side_connectivity": True,
            },
            "exclude_review_intervals": [
                {
                    "name": "review_horizontal_severe_band_5700ft",
                    "depth_min_ft": 5680.0,
                    "depth_max_ft": 5720.0,
                    "reason": "review exclusion",
                    "apply_by_default": True,
                }
            ],
        },
        "quality_policy": {
            "min_subset_samples_per_class": 50,
            "high_confidence_orientation_thresholds": [0.5, 0.7],
            "disagreement_policy": "exclude_for_quality_subsets",
            "suspicious_band_policy": "exclude_and_report_sensitivity",
            "connected_object_policy": "candidate_only_filter",
        },
        "gate": {
            "signal_enhancement_effect_size_delta": 0.05,
            "strong_signal_effect_size_threshold": 0.30,
            "max_result_flip_fraction_from_review_exclusion": 0.50,
        },
        "allowed_scope": "label_quality_subset_diagnostics_only",
        "no_model_training": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }


def test_valid_label_quality_config_parses_guardrails_and_review_interval() -> None:
    config = parse_label_quality_config(_valid_raw_config())
    validation = validate_label_quality_config(config)

    assert validation.valid
    assert validation.errors == []
    assert config.strong_positive.min_severity == 2
    assert config.clear_negative.label_presence_plus == 0
    assert active_review_intervals(config)[0].name == "review_horizontal_severe_band_5700ft"


def test_label_quality_config_rejects_final_labels_and_forbidden_methods() -> None:
    raw = _valid_raw_config()
    raw["no_final_labels"] = False
    raw["no_stc"] = False
    raw["no_mvp4c"] = False

    validation = validate_label_quality_config(parse_label_quality_config(raw))

    assert not validation.valid
    assert "no_final_labels must be true." in validation.errors
    assert "no_stc must be true." in validation.errors
    assert "no_mvp4c must be true." in validation.errors


def test_label_quality_config_requires_disagreement_free_quality_subsets() -> None:
    raw = _valid_raw_config()
    raw["subsets"]["strong_positive"]["require_no_plus_minus_disagreement"] = False

    validation = validate_label_quality_config(parse_label_quality_config(raw))

    assert not validation.valid
    assert any("strong_positive" in message for message in validation.errors)


def test_label_quality_config_requires_5700_review_band() -> None:
    raw = _valid_raw_config()
    raw["subsets"]["exclude_review_intervals"][0]["depth_min_ft"] = 5800.0
    raw["subsets"]["exclude_review_intervals"][0]["depth_max_ft"] = 5810.0

    validation = validate_label_quality_config(parse_label_quality_config(raw))

    assert not validation.valid
    assert any("5700" in message for message in validation.errors)


def test_load_example_label_quality_config() -> None:
    config = load_label_quality_config("configs/mvp4b_label_quality_subsets.example.yaml")

    assert config.config_version == MVP4B_LABEL_QUALITY_CONFIG_VERSION
    assert config.no_final_labels is True
    assert config.no_deep_learning is True


def test_load_label_quality_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_label_quality_config(path)
