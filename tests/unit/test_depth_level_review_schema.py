from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.evaluation.depth_level_review_schema import (
    DEPTH_LEVEL_MANUAL_REVIEW_CONFIG_VERSION,
    load_depth_level_manual_review_config,
    parse_depth_level_manual_review_config,
    validate_depth_level_manual_review_config,
)


def _valid_raw_config() -> dict:
    return {
        "schema_version": "schema_v001",
        "config_version": DEPTH_LEVEL_MANUAL_REVIEW_CONFIG_VERSION,
        "stage": "MVP-4B-R4c+",
        "task": "depth_level_manual_review_pack",
        "input_labels": "depth_level_labels_v001",
        "input_features": "depth_level_xsi_features_v001",
        "input_refinement_report": "depth_level_refinement_report_v001",
        "input_refinement_gate_report": "depth_level_refinement_gate_report",
        "target_variant": "high_confidence_positive_vs_clear_negative",
        "label_status": "weak_label_candidate",
        "selection_defaults": {
            "max_interval_gap_ft": 2.0,
            "min_interval_depth_span_ft": 0.0,
            "score_high_threshold": 0.7,
            "score_low_threshold": 0.3,
            "boundary_score_band": 0.1,
            "high_disagreement_threshold": 0.25,
            "low_confidence_threshold": 0.5,
        },
        "select_top_positive_intervals": {
            "enabled": True,
            "count": 8,
            "interval_type": "true_positive_like",
            "sort_by": "score_desc",
        },
        "select_clear_negative_intervals": {
            "enabled": True,
            "count": 8,
            "interval_type": "clear_negative_like",
            "sort_by": "score_asc",
        },
        "select_high_score_positive_intervals": {
            "enabled": True,
            "count": 8,
            "interval_type": "true_positive_like",
            "sort_by": "score_desc",
        },
        "select_high_score_negative_or_disagreement_intervals": {
            "enabled": True,
            "count": 8,
            "interval_type": "false_positive_like",
            "sort_by": "score_desc",
        },
        "select_low_score_positive_intervals": {
            "enabled": True,
            "count": 8,
            "interval_type": "false_negative_like",
            "sort_by": "score_asc",
        },
        "review_intervals": [
            {
                "name": "review_horizontal_severe_band_5700ft",
                "depth_min_ft": 5680.0,
                "depth_max_ft": 5720.0,
                "interval_type": "5700_band_review",
                "reason": "known review band",
            }
        ],
        "include_5700_band_sensitivity": True,
        "include_confidence_summary": True,
        "include_xsi_feature_summary": True,
        "include_cast_label_summary": True,
        "allowed_scope": "depth_level_manual_review_pack_only",
        "no_model_training_claim": True,
        "no_production_model": True,
        "no_final_labels": True,
        "no_mvp4c": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
    }


def test_valid_depth_level_manual_review_config_parses() -> None:
    config = parse_depth_level_manual_review_config(_valid_raw_config())
    validation = validate_depth_level_manual_review_config(config)

    assert validation.valid
    assert validation.errors == []
    assert config.no_final_labels is True
    assert config.selections["select_clear_negative_intervals"].interval_type == (
        "clear_negative_like"
    )


def test_depth_level_manual_review_config_rejects_forbidden_scope() -> None:
    raw = _valid_raw_config()
    raw["no_final_labels"] = False
    raw["no_mvp4c"] = False
    raw["no_stc"] = False

    validation = validate_depth_level_manual_review_config(
        parse_depth_level_manual_review_config(raw)
    )

    assert not validation.valid
    assert "no_final_labels must be true." in validation.errors
    assert "no_mvp4c must be true." in validation.errors
    assert "no_stc must be true." in validation.errors


def test_depth_level_manual_review_config_requires_core_selections() -> None:
    raw = _valid_raw_config()
    raw.pop("select_low_score_positive_intervals")

    validation = validate_depth_level_manual_review_config(
        parse_depth_level_manual_review_config(raw)
    )

    assert not validation.valid
    assert any("missing required review selection" in message for message in validation.errors)


def test_depth_level_manual_review_config_rejects_bad_selection_values() -> None:
    raw = _valid_raw_config()
    raw["select_clear_negative_intervals"]["count"] = 0
    raw["select_clear_negative_intervals"]["interval_type"] = "ground_truth"

    validation = validate_depth_level_manual_review_config(
        parse_depth_level_manual_review_config(raw)
    )

    assert not validation.valid
    assert "select_clear_negative_intervals.count must be positive when enabled." in (
        validation.errors
    )
    assert any("interval_type is unsupported" in message for message in validation.errors)


def test_load_example_depth_level_manual_review_config() -> None:
    config = load_depth_level_manual_review_config(
        "configs/depth_level_manual_review.example.yaml"
    )

    assert config.config_version == DEPTH_LEVEL_MANUAL_REVIEW_CONFIG_VERSION
    assert config.label_status == "weak_label_candidate"
    assert config.include_cast_label_summary is True
    assert config.no_apes is True


def test_load_depth_level_manual_review_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_depth_level_manual_review_config(path)
