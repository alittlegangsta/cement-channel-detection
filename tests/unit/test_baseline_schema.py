from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.training.baseline_schema import (
    MVP4B_ALLOWED_SCOPE,
    MVP4B_INPUT_SAMPLE_TABLE,
    MVP4B_LABEL,
    MVP4B_LABEL_STATUS,
    MVP4B_SIMPLE_BASELINE_CONFIG_VERSION,
    load_baseline_config,
    parse_baseline_config,
    validate_baseline_config,
)


def _valid_raw_config() -> dict:
    return {
        "config_version": MVP4B_SIMPLE_BASELINE_CONFIG_VERSION,
        "input_sample_table": MVP4B_INPUT_SAMPLE_TABLE,
        "label": MVP4B_LABEL,
        "label_status": MVP4B_LABEL_STATUS,
        "primary_label": "plus",
        "audit_label": "minus_ablation",
        "no_final_labels": True,
        "model_type": ["logistic_regression", "linear_probe"],
        "feature_set": ["transformed_features"],
        "sample_filter": {
            "high_confidence_only": True,
            "valid_for_azimuthal_validation": True,
            "exclude_plus_minus_disagreement": False,
            "exclude_large_depth_match_error": True,
            "min_samples_per_class": 20,
        },
        "sample_weight": {"use_sample_weight": True, "source": "sample_weight"},
        "split": {
            "method": "depth_block_group_split",
            "n_splits": 3,
            "depth_block_size_ft": 250.0,
            "min_gap_ft": 5.0,
            "min_samples_per_class_per_fold": 10,
        },
        "evaluation": {
            "metrics": [
                "weighted_accuracy",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
                "calibration_summary",
            ],
            "permutation_check": True,
            "permutation_seed": 202405,
            "min_permutation_balanced_accuracy_margin": 0.02,
            "suspicious_metric_threshold": 0.98,
            "calibration_bins": 10,
        },
        "optimizer": {"max_iterations": 50, "learning_rate": 0.1, "l2_penalty": 0.0001},
        "allowed_scope": MVP4B_ALLOWED_SCOPE,
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "no_production_model": True,
    }


def test_valid_baseline_config_parses_scope_and_split() -> None:
    config = parse_baseline_config(_valid_raw_config())
    validation = validate_baseline_config(config)

    assert validation.valid
    assert validation.errors == []
    assert config.model_types == ("logistic_regression", "linear_probe")
    assert config.split_method == "depth_block_group_split"
    assert config.allowed_scope == "sanity_model_only"


def test_config_rejects_final_labels_and_production_scope() -> None:
    raw = _valid_raw_config()
    raw["no_final_labels"] = False
    raw["allowed_scope"] = "production_training"
    raw["no_production_model"] = False

    validation = validate_baseline_config(parse_baseline_config(raw))

    assert not validation.valid
    assert "no_final_labels must be true." in validation.errors
    assert "allowed_scope must be sanity_model_only." in validation.errors
    assert "no_production_model must be true." in validation.errors


def test_config_rejects_low_confidence_or_random_split() -> None:
    raw = _valid_raw_config()
    raw["sample_filter"]["high_confidence_only"] = False
    raw["split"]["method"] = "random_point_split"

    validation = validate_baseline_config(parse_baseline_config(raw))

    assert not validation.valid
    assert "sample_filter.high_confidence_only must be true." in validation.errors
    assert "split.method must be depth_block_group_split." in validation.errors


def test_config_rejects_deep_learning_stc_and_apes() -> None:
    raw = _valid_raw_config()
    raw["model_type"] = ["transformer"]
    raw["no_deep_learning"] = False
    raw["no_stc"] = False
    raw["no_apes"] = False

    validation = validate_baseline_config(parse_baseline_config(raw))

    assert not validation.valid
    assert any("Unsupported model_type" in message for message in validation.errors)
    assert "no_deep_learning must be true." in validation.errors
    assert "no_stc must be true." in validation.errors
    assert "no_apes must be true." in validation.errors


def test_load_example_config() -> None:
    config = load_baseline_config(Path("configs/mvp4b_simple_baseline.example.yaml"))

    assert config.input_sample_table == "baseline_sample_table_v001"
    assert config.label == "label_presence_plus"
    assert config.permutation_check is True


def test_load_config_raises_on_invalid_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_baseline_config(path)
