from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.training.depth_level_baseline_schema import (
    DEPTH_LEVEL_BASELINE_CONFIG_VERSION,
    load_depth_level_baseline_config,
    parse_depth_level_baseline_config,
    validate_depth_level_baseline_config,
)


def _valid_raw_config() -> dict:
    return {
        "schema_version": "schema_v001",
        "config_version": DEPTH_LEVEL_BASELINE_CONFIG_VERSION,
        "stage": "MVP-4B-R4b",
        "task": "depth_level_baseline_sanity_model",
        "input_labels": "depth_level_labels_v001",
        "input_features": "depth_level_xsi_features_v001",
        "primary_task": "depth_has_channel",
        "label_status": "weak_label_candidate",
        "model_type": ["logistic_regression", "linear_probe"],
        "feature_set": ["depth_level_xsi_features"],
        "target_variants": [
            "all_positive_vs_negative",
            "strong_positive_vs_clear_negative",
            "high_confidence_positive_vs_clear_negative",
        ],
        "target_filters": {
            "high_confidence_positive": {
                "min_label_confidence": 0.5,
                "min_orientation_confidence": 0.5,
                "max_plus_minus_disagreement_fraction": 0.25,
            },
            "clear_negative": {
                "min_label_confidence": 0.5,
                "min_orientation_confidence": 0.5,
                "max_plus_minus_disagreement_fraction": 0.25,
            },
            "exclude_review_band": True,
            "min_samples_per_class": 20,
            "min_samples_per_class_per_fold": 5,
            "warn_if_variant_too_small": True,
        },
        "split": {
            "method": "depth_block_split",
            "n_splits": 3,
            "depth_block_size_ft": 250.0,
            "min_gap_ft": 5.0,
        },
        "evaluation": {
            "metrics": [
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "permutation_margin",
            ],
            "permutation_check": True,
            "permutation_seed": 202405,
            "min_permutation_balanced_accuracy_margin": 0.03,
            "degenerate_prediction_min_positive_rate": 0.05,
            "degenerate_prediction_max_positive_rate": 0.95,
            "stable_fold_min_count": 2,
        },
        "optimizer": {
            "max_iterations": 350,
            "learning_rate": 0.1,
            "l2_penalty": 0.0001,
        },
        "allowed_scope": "depth_level_baseline_sanity_only",
        "no_model_training_claim": True,
        "no_production_model": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }


def test_valid_depth_level_baseline_config_parses_guardrails() -> None:
    config = parse_depth_level_baseline_config(_valid_raw_config())
    validation = validate_depth_level_baseline_config(config)

    assert validation.valid
    assert validation.errors == []
    assert config.input_labels == "depth_level_labels_v001"
    assert config.target_filters.warn_if_variant_too_small is True


def test_depth_level_baseline_config_rejects_forbidden_methods_and_claims() -> None:
    raw = _valid_raw_config()
    raw["no_final_labels"] = False
    raw["no_stc"] = False
    raw["no_mvp4c"] = False
    raw["no_model_training_claim"] = False

    validation = validate_depth_level_baseline_config(
        parse_depth_level_baseline_config(raw)
    )

    assert not validation.valid
    assert "no_final_labels must be true." in validation.errors
    assert "no_stc must be true." in validation.errors
    assert "no_mvp4c must be true." in validation.errors
    assert "no_model_training_claim must be true." in validation.errors


def test_depth_level_baseline_config_requires_all_target_variants() -> None:
    raw = _valid_raw_config()
    raw["target_variants"] = ["all_positive_vs_negative"]

    validation = validate_depth_level_baseline_config(
        parse_depth_level_baseline_config(raw)
    )

    assert not validation.valid
    assert any("target_variants missing" in message for message in validation.errors)


def test_depth_level_baseline_config_requires_permutation_and_depth_split() -> None:
    raw = _valid_raw_config()
    raw["evaluation"]["permutation_check"] = False
    raw["split"]["method"] = "random_split"

    validation = validate_depth_level_baseline_config(
        parse_depth_level_baseline_config(raw)
    )

    assert not validation.valid
    assert "evaluation.permutation_check must be true." in validation.errors
    assert "split.method must be depth_block_split." in validation.errors


def test_load_example_depth_level_baseline_config() -> None:
    config = load_depth_level_baseline_config("configs/depth_level_baseline.example.yaml")

    assert config.config_version == DEPTH_LEVEL_BASELINE_CONFIG_VERSION
    assert config.no_final_labels is True
    assert config.no_deep_learning is True
    assert "permutation_margin" in config.evaluation.metrics


def test_load_depth_level_baseline_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_depth_level_baseline_config(path)
