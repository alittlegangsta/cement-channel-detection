from __future__ import annotations

from pathlib import Path

import pytest

from cement_channel.training.depth_level_refinement_schema import (
    DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION,
    load_depth_level_refinement_config,
    parse_depth_level_refinement_config,
    validate_depth_level_refinement_config,
)


def _valid_raw_config() -> dict:
    return {
        "schema_version": "schema_v001",
        "config_version": DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION,
        "stage": "MVP-4B-R4c",
        "task": "controlled_depth_level_feature_refinement",
        "input_labels": "depth_level_labels_v001",
        "input_features": "depth_level_xsi_features_v001",
        "input_baseline_report": "depth_level_baseline_report_v001",
        "target_variant": "high_confidence_positive_vs_clear_negative",
        "label_status": "weak_label_candidate",
        "allowed_models": ["logistic_regression", "linear_probe"],
        "feature_groups": [
            "all_depth_features",
            "late_over_early_features",
            "energy_window_features",
            "side_contrast_features",
            "receiver_summary_features",
            "robust_top_features_from_baseline",
        ],
        "target_filters": {
            "clear_negative_min_label_confidence": 0.5,
            "clear_negative_min_orientation_confidence": 0.5,
            "max_plus_minus_disagreement_fraction": 0.25,
        },
        "review_intervals": [
            {
                "name": "review_horizontal_severe_band_5700ft",
                "depth_min_ft": 5680.0,
                "depth_max_ft": 5720.0,
                "reason": "known review band",
            }
        ],
        "robustness_checks": {
            "exclude_5700_band": [False, True],
            "confidence_thresholds": [0.4, 0.5, 0.6],
            "depth_block_splits": [3, 5],
            "permutation_repeats": 5,
            "feature_group_ablation": True,
            "fold_stability_required": True,
        },
        "split": {
            "method": "depth_block_split",
            "depth_block_size_ft": 250.0,
            "min_gap_ft": 5.0,
            "min_samples_per_class_per_fold": 5,
        },
        "optimizer": {
            "max_iterations": 350,
            "learning_rate": 0.1,
            "l2_penalty": 0.0001,
        },
        "gate_thresholds": {
            "min_margin_mean": 0.05,
            "min_margin_permutation": 0.03,
            "max_predicted_positive_rate": 0.85,
            "min_predicted_positive_rate": 0.15,
            "min_folds_above_permutation_fraction": 0.66,
            "suspicious_high_balanced_accuracy": 0.9,
        },
        "allowed_scope": "controlled_depth_level_refinement_only",
        "no_model_training_claim": True,
        "no_production_model": True,
        "no_final_labels": True,
        "no_mvp4c": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
    }


def test_valid_depth_level_refinement_config_parses() -> None:
    config = parse_depth_level_refinement_config(_valid_raw_config())
    validation = validate_depth_level_refinement_config(config)

    assert validation.valid
    assert validation.errors == []
    assert config.target_variant == "high_confidence_positive_vs_clear_negative"
    assert config.robustness_checks.permutation_repeats == 5
    assert config.no_final_labels is True


def test_depth_level_refinement_config_rejects_forbidden_scope() -> None:
    raw = _valid_raw_config()
    raw["no_mvp4c"] = False
    raw["no_stc"] = False
    raw["no_deep_learning"] = False

    validation = validate_depth_level_refinement_config(
        parse_depth_level_refinement_config(raw)
    )

    assert not validation.valid
    assert "no_mvp4c must be true." in validation.errors
    assert "no_stc must be true." in validation.errors
    assert "no_deep_learning must be true." in validation.errors


def test_depth_level_refinement_config_requires_fixed_target_variant() -> None:
    raw = _valid_raw_config()
    raw["target_variant"] = "all_positive_vs_negative"

    validation = validate_depth_level_refinement_config(
        parse_depth_level_refinement_config(raw)
    )

    assert not validation.valid
    assert any("target_variant must be" in message for message in validation.errors)


def test_depth_level_refinement_config_requires_robustness_matrix() -> None:
    raw = _valid_raw_config()
    raw["robustness_checks"]["exclude_5700_band"] = [False]
    raw["robustness_checks"]["depth_block_splits"] = [3]
    raw["robustness_checks"]["feature_group_ablation"] = False

    validation = validate_depth_level_refinement_config(
        parse_depth_level_refinement_config(raw)
    )

    assert not validation.valid
    assert any("exclude_5700_band" in message for message in validation.errors)
    assert any("depth_block_splits" in message for message in validation.errors)
    assert "robustness_checks.feature_group_ablation must be true." in validation.errors


def test_load_example_depth_level_refinement_config() -> None:
    config = load_depth_level_refinement_config(
        "configs/depth_level_refinement.example.yaml"
    )

    assert config.config_version == DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION
    assert config.no_final_labels is True
    assert config.no_apes is True
    assert "robust_top_features_from_baseline" in config.feature_groups


def test_load_depth_level_refinement_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_depth_level_refinement_config(path)
