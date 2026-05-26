from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.labels.depth_level_schema import (
    DEPTH_LEVEL_LABEL_CONFIG_VERSION,
    DEPTH_LEVEL_REQUIRED_FIELDS,
    active_depth_review_intervals,
    load_depth_level_label_config,
    parse_depth_level_label_config,
    validate_depth_level_label_config,
)


def _valid_raw_config() -> dict:
    return {
        "schema_version": "schema_v001",
        "config_version": DEPTH_LEVEL_LABEL_CONFIG_VERSION,
        "stage": "MVP-4B-R4",
        "task": "depth_level_target_review",
        "input_cast_weak_labels": "cast_weak_label_candidates_v001",
        "input_xsi_label_samples": "xsi_label_samples_v001",
        "optional_input_sample_table": "baseline_sample_table_receiver_enhanced_v001",
        "output_depth_level_labels": "depth_level_labels_v001",
        "primary_label": "plus",
        "audit_label": "minus_ablation",
        "label_status": "human_reviewed_candidate_v001",
        "depth_label_fields": list(DEPTH_LEVEL_REQUIRED_FIELDS),
        "aggregation_policy": {
            "presence": {
                "has_channel_method": "any",
                "candidate_fraction_method": "fraction",
            },
            "severity": {"max_method": "max"},
            "confidence": {
                "max_method": "max",
                "depth_label_confidence_method": "confidence_weighted_fraction",
            },
            "zc": {"min_method": "min", "percentile_methods": ["p05", "p10"]},
            "relative_drop": {"max_method": "max"},
            "object_width": {"method": "largest_connected_azimuth_width"},
            "disagreement": {"plus_minus_method": "fraction"},
            "side_level_labels": {"usage": "audit_only", "train_target": False},
            "require_any_max_percentile_fraction": True,
            "forbid_mean_only": True,
        },
        "quality_policy": {
            "strong_positive": {
                "min_candidate_fraction": 0.25,
                "min_max_severity": 2,
                "min_label_confidence": 0.5,
                "max_plus_minus_disagreement_fraction": 0.25,
                "min_orientation_confidence": 0.5,
            },
            "clear_negative": {
                "max_candidate_fraction": 0.0,
                "min_label_confidence": 0.5,
                "max_plus_minus_disagreement_fraction": 0.25,
                "min_orientation_confidence": 0.5,
            },
            "review_intervals": [
                {
                    "name": "review_horizontal_severe_band_5700ft",
                    "depth_min_ft": 5680.0,
                    "depth_max_ft": 5720.0,
                    "reason": "review exclusion",
                    "apply_by_default": True,
                }
            ],
            "max_review_band_positive_fraction": 0.5,
        },
        "gate": {
            "min_depth_positive_count": 1,
            "min_depth_negative_count": 1,
            "max_5700_band_positive_fraction": 0.5,
            "depth_level_improvement_effect_size_delta": 0.05,
            "sanity_effect_size_threshold": 0.30,
        },
        "allowed_scope": "depth_level_target_review_only",
        "no_model_training": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }


def test_valid_depth_level_config_parses_required_fields_and_review_band() -> None:
    config = parse_depth_level_label_config(_valid_raw_config())
    validation = validate_depth_level_label_config(config)

    assert validation.valid
    assert validation.errors == []
    assert config.primary_label == "plus"
    assert active_depth_review_intervals(config)[0].name == "review_horizontal_severe_band_5700ft"


def test_depth_level_config_rejects_mean_only_policy_and_side_train_target() -> None:
    raw = _valid_raw_config()
    raw["aggregation_policy"]["forbid_mean_only"] = False
    raw["aggregation_policy"]["side_level_labels"]["train_target"] = True

    validation = validate_depth_level_label_config(parse_depth_level_label_config(raw))

    assert not validation.valid
    assert "aggregation_policy.forbid_mean_only must be true." in validation.errors
    assert "side_level_labels.train_target must be false." in validation.errors


def test_depth_level_config_rejects_final_labels_and_forbidden_scope() -> None:
    raw = _valid_raw_config()
    raw["no_final_labels"] = False
    raw["no_apes"] = False
    raw["no_mvp4c"] = False

    validation = validate_depth_level_label_config(parse_depth_level_label_config(raw))

    assert not validation.valid
    assert "no_final_labels must be true." in validation.errors
    assert "no_apes must be true." in validation.errors
    assert "no_mvp4c must be true." in validation.errors


def test_depth_level_config_requires_all_depth_label_fields() -> None:
    raw = _valid_raw_config()
    raw["depth_label_fields"] = ["depth_has_channel_any"]

    validation = validate_depth_level_label_config(parse_depth_level_label_config(raw))

    assert not validation.valid
    assert any(
        "depth_label_fields missing required field" in message
        for message in validation.errors
    )


def test_depth_level_config_requires_5700_review_band() -> None:
    raw = _valid_raw_config()
    raw["quality_policy"]["review_intervals"][0]["depth_min_ft"] = 5600.0
    raw["quality_policy"]["review_intervals"][0]["depth_max_ft"] = 5650.0

    validation = validate_depth_level_label_config(parse_depth_level_label_config(raw))

    assert not validation.valid
    assert any("5700" in message for message in validation.errors)


def test_load_example_depth_level_label_config() -> None:
    config = load_depth_level_label_config("configs/depth_level_label.example.yaml")

    assert config.config_version == DEPTH_LEVEL_LABEL_CONFIG_VERSION
    assert config.no_final_labels is True
    assert config.no_deep_learning is True
    assert "depth_candidate_fraction" in config.depth_label_fields


def test_load_depth_level_label_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_depth_level_label_config(path)
