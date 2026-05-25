from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.training.sample_schema import (
    MVP4B_FEATURE_NAMES,
    MVP4B_INPUT_FEATURES,
    MVP4B_INPUT_LABELS,
    MVP4B_SAMPLE_CONFIG_VERSION,
    load_sample_table_config,
    parse_sample_table_config,
    transformed_feature_names,
    validate_sample_table_config,
)


def _valid_raw_config() -> dict:
    return {
        "config_version": MVP4B_SAMPLE_CONFIG_VERSION,
        "input_features": MVP4B_INPUT_FEATURES,
        "input_labels": MVP4B_INPUT_LABELS,
        "primary_label": "plus",
        "audit_label": "minus_ablation",
        "feature_names": list(MVP4B_FEATURE_NAMES),
        "transforms": {
            "log1p": True,
            "robust_scaling": True,
            "clip_quantiles": [0.001, 0.999],
            "per_feature_scaling": True,
            "optional_per_depth_normalization": True,
            "optional_per_side_normalization": True,
        },
        "sample_policy": {
            "use_high_confidence_for_azimuthal": True,
            "min_label_confidence": 0.5,
            "low_confidence_usage": "non_azimuthal_or_excluded",
            "exclude_large_depth_match_error": True,
            "max_depth_match_error_ft": 0.5,
            "plus_minus_disagreement_policy": "audit_flag_or_downweight",
            "plus_minus_disagreement_weight_multiplier": 0.5,
            "depth_mismatch_weight_multiplier": 0.0,
        },
        "diagnostics": {
            "min_high_confidence_samples_per_class": 20,
            "max_nonfinite_transformed_fraction": 0.001,
            "min_positive_sample_weight_fraction": 0.05,
            "max_large_depth_match_error_fraction": 0.20,
        },
        "no_model_training": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
    }


def test_valid_sample_table_config_parses_and_validates() -> None:
    config = parse_sample_table_config(_valid_raw_config())
    validation = validate_sample_table_config(config)

    assert validation.valid
    assert validation.errors == []
    assert config.primary_label == "plus"
    assert config.audit_label == "minus_ablation"
    assert transformed_feature_names(config)[0] == "log1p_rms_energy"
    assert transformed_feature_names(config)[-1] == "robust_scaled_late_over_early_ratio"


def test_sample_table_config_rejects_training_and_final_labels() -> None:
    raw = _valid_raw_config()
    raw["no_model_training"] = False
    raw["no_final_labels"] = False

    validation = validate_sample_table_config(parse_sample_table_config(raw))

    assert not validation.valid
    assert "no_model_training must be true." in validation.errors
    assert "no_final_labels must be true." in validation.errors


def test_sample_table_config_rejects_unsupported_feature_and_policy() -> None:
    raw = _valid_raw_config()
    raw["feature_names"] = ["rms_energy", "late_energy", "stc_peak"]
    raw["sample_policy"]["low_confidence_usage"] = "strong_supervision"

    validation = validate_sample_table_config(parse_sample_table_config(raw))

    assert not validation.valid
    assert any("missing required feature" in message for message in validation.errors)
    assert any("unsupported feature" in message for message in validation.errors)
    assert any("low_confidence_usage" in message for message in validation.errors)


def test_sample_table_config_allows_todo_depth_match_warning() -> None:
    raw = _valid_raw_config()
    raw["sample_policy"]["max_depth_match_error_ft"] = "TODO_CONFIG"

    validation = validate_sample_table_config(parse_sample_table_config(raw))

    assert validation.valid
    assert any("TODO" in message for message in validation.warnings)


def test_load_example_config() -> None:
    config = load_sample_table_config("configs/mvp4b_sample_table.example.yaml")

    assert config.input_features == "xsi_basic_features_v001"
    assert config.input_labels == "xsi_label_samples_v001"
    assert config.no_model_training
    assert config.no_final_labels


def test_load_config_raises_on_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_sample_table_config(path)
