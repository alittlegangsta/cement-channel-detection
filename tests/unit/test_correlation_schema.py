from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cement_channel.evaluation.correlation_schema import (
    MVP4A_AUDIT_LABEL,
    MVP4A_CONFIG_VERSION,
    MVP4A_LABEL_SOURCE,
    MVP4A_PRIMARY_LABEL,
    expected_feature_names,
    load_correlation_config,
    parse_correlation_config,
    reference_receiver_zero_based,
    validate_correlation_config,
    xsi_side_azimuth_deg,
)


def _valid_raw_config() -> dict:
    return {
        "config_version": MVP4A_CONFIG_VERSION,
        "label_source": MVP4A_LABEL_SOURCE,
        "primary_label": MVP4A_PRIMARY_LABEL,
        "audit_label": MVP4A_AUDIT_LABEL,
        "use_label_confidence": True,
        "min_label_confidence_for_azimuthal_validation": 0.5,
        "allow_low_confidence_for_non_azimuthal_summary": True,
        "xsi_feature_set": [
            "rms_energy",
            "peak_abs",
            "mean_abs",
            "early_energy",
            "late_energy",
        ],
        "receiver_aggregation": {"method": "mean_or_median", "reference_receiver_index": 7},
        "side_mapping": {
            "side_a_offset_deg": 0.0,
            "side_order": "clockwise",
            "side_labels": list("ABCDEFGH"),
        },
        "sampler": {"noncandidate_azimuthal_validation_requires_label_confidence": False},
        "feature_extraction": {"chunk_depth_samples": 32, "max_time_samples": 1024},
        "correlation": {
            "high_confidence_min_samples_per_class": 20,
            "min_interpretable_abs_effect_size": 0.05,
            "min_interpretable_weighted_difference_fraction": 0.01,
        },
        "no_model_training": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
    }


def test_valid_correlation_config_parses_policy_and_geometry() -> None:
    config = parse_correlation_config(_valid_raw_config())
    validation = validate_correlation_config(config)

    assert validation.valid
    assert validation.errors == []
    assert reference_receiver_zero_based(config) == 6
    assert expected_feature_names(config)[-1] == "late_over_early_ratio"
    np.testing.assert_allclose(
        xsi_side_azimuth_deg(config),
        np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32),
    )


def test_config_rejects_training_and_final_label_modes() -> None:
    raw = _valid_raw_config()
    raw["no_model_training"] = False
    raw["no_final_labels"] = False

    validation = validate_correlation_config(parse_correlation_config(raw))

    assert not validation.valid
    assert "no_model_training must be true." in validation.errors
    assert "no_final_labels must be true." in validation.errors


def test_config_rejects_unsupported_feature() -> None:
    raw = _valid_raw_config()
    raw["xsi_feature_set"] = ["rms_energy", "peak_abs", "mean_abs", "early_energy", "stc_peak"]

    validation = validate_correlation_config(parse_correlation_config(raw))

    assert not validation.valid
    assert any("missing required feature" in message for message in validation.errors)
    assert any("unsupported feature" in message for message in validation.errors)


def test_load_example_config() -> None:
    path = Path("configs/mvp4a_xsi_cast_correlation.example.yaml")

    config = load_correlation_config(path)

    assert config.label_source == "cast_weak_label_candidates_v001"
    assert config.primary_label == "plus"
    assert config.audit_label == "minus_ablation"


def test_load_config_raises_on_invalid_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_correlation_config(path)
