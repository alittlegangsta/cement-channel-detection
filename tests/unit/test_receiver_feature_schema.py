from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cement_channel.features.receiver_feature_schema import (
    MVP4B_RECEIVER_FEATURE_CONFIG_VERSION,
    expected_receiver_feature_families,
    load_receiver_feature_config,
    parse_receiver_feature_config,
    receiver_group_zero_based,
    receiver_offsets_ft,
    receiver_source_distances_ft,
    validate_receiver_feature_config,
)


def _valid_raw_config() -> dict:
    return {
        "config_version": MVP4B_RECEIVER_FEATURE_CONFIG_VERSION,
        "input_basic_features": "xsi_basic_features_v001",
        "input_sample_table": "baseline_sample_table_enhanced_v001",
        "primary_label": "plus",
        "audit_label": "minus_ablation",
        "label_status": "human_reviewed_candidate_v001",
        "receiver_geometry": {
            "receiver_count": 13,
            "reference_receiver_index": 7,
            "receiver_spacing_ft": 0.5,
            "source_to_receiver1_ft": 1.0,
            "receiver_offsets_from_reference_ft": [
                -3.0,
                -2.5,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
            ],
            "near_receivers": [1, 2, 3, 4],
            "mid_receivers": [5, 6, 7, 8, 9],
            "far_receivers": [10, 11, 12, 13],
        },
        "source_feature_names": [
            "rms_energy",
            "peak_abs",
            "mean_abs",
            "early_energy",
            "late_energy",
            "late_over_early_ratio",
        ],
        "receiver_feature_set": [
            "receiver_mean_per_side_feature",
            "receiver_std_per_side_feature",
            "receiver_slope_per_side_feature",
            "near_receiver_mean",
            "mid_receiver_mean",
            "far_receiver_mean",
            "far_minus_near",
            "far_over_near",
            "receiver_peak_position",
            "receiver_energy_decay_slope",
            "receiver_consistency_cv",
            "per_side_receiver_normalized",
        ],
        "transforms": {
            "log1p_positive_features": True,
            "robust_scaling": True,
            "clip_quantiles": [0.001, 0.999],
            "epsilon": 1.0e-6,
        },
        "sample_policy": {
            "use_high_confidence_for_azimuthal": True,
            "min_label_confidence": 0.5,
            "low_confidence_usage": "non_azimuthal_or_excluded",
            "exclude_large_depth_match_error": True,
            "max_depth_match_error_ft": 0.5,
            "plus_minus_disagreement_policy": "audit_flag_or_downweight",
        },
        "ablation": {
            "required_margin_over_permutation": 0.03,
            "max_degenerate_positive_rate": 0.95,
            "min_degenerate_positive_rate": 0.05,
            "required_folds_above_permutation": 2,
        },
        "allowed_scope": "receiver_feature_remediation_only",
        "no_model_training": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }


def test_valid_receiver_feature_config_parses_geometry_and_guardrails() -> None:
    config = parse_receiver_feature_config(_valid_raw_config())
    validation = validate_receiver_feature_config(config)

    assert validation.valid
    assert validation.errors == []
    np.testing.assert_array_equal(receiver_group_zero_based(config, "near"), [0, 1, 2, 3])
    np.testing.assert_array_equal(receiver_group_zero_based(config, "mid"), [4, 5, 6, 7, 8])
    np.testing.assert_array_equal(receiver_group_zero_based(config, "far"), [9, 10, 11, 12])
    np.testing.assert_allclose(receiver_offsets_ft(config), np.linspace(-3.0, 3.0, 13))
    np.testing.assert_allclose(receiver_source_distances_ft(config), np.linspace(1.0, 7.0, 13))
    assert "receiver_energy_decay_slope" in expected_receiver_feature_families(config)


def test_receiver_feature_config_rejects_final_labels_and_forbidden_methods() -> None:
    raw = _valid_raw_config()
    raw["no_final_labels"] = False
    raw["no_stc"] = False
    raw["no_mvp4c"] = False

    validation = validate_receiver_feature_config(parse_receiver_feature_config(raw))

    assert not validation.valid
    assert "no_final_labels must be true." in validation.errors
    assert "no_stc must be true." in validation.errors
    assert "no_mvp4c must be true." in validation.errors


def test_receiver_feature_config_rejects_bad_receiver_groups() -> None:
    raw = _valid_raw_config()
    raw["receiver_geometry"]["near_receivers"] = [1, 2, 3]

    validation = validate_receiver_feature_config(parse_receiver_feature_config(raw))

    assert not validation.valid
    assert any("near_receivers" in message for message in validation.errors)
    assert any("cover receivers" in message for message in validation.errors)


def test_receiver_feature_config_rejects_unsupported_feature() -> None:
    raw = _valid_raw_config()
    raw["receiver_feature_set"].append("stc_peak")

    validation = validate_receiver_feature_config(parse_receiver_feature_config(raw))

    assert not validation.valid
    assert any("unsupported" in message for message in validation.errors)


def test_load_example_receiver_feature_config() -> None:
    config = load_receiver_feature_config("configs/mvp4b_receiver_features.example.yaml")

    assert config.config_version == MVP4B_RECEIVER_FEATURE_CONFIG_VERSION
    assert config.input_basic_features == "xsi_basic_features_v001"
    assert config.no_final_labels is True


def test_load_receiver_feature_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_receiver_feature_config(path)
