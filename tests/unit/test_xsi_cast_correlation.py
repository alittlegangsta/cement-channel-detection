from __future__ import annotations

import numpy as np

from cement_channel.evaluation.correlation_schema import parse_correlation_config
from cement_channel.evaluation.xsi_cast_correlation import (
    evaluate_xsi_cast_correlation_from_arrays,
)
from tests.unit.test_correlation_schema import _valid_raw_config


def _config():
    return parse_correlation_config(_valid_raw_config())


def test_correlation_detects_feature_separation_without_training() -> None:
    presence = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int8)
    severity = np.where(presence == 1, 2, 0).astype(np.int8)
    values = np.zeros((4, 2, 2), dtype=np.float32)
    values[..., 0] = np.where(presence == 1, 10.0, 1.0)
    values[..., 1] = np.where(presence == 1, 2.0, 2.0)
    label_arrays = {
        "label_presence_plus": presence,
        "label_presence_minus_audit": np.zeros_like(presence),
        "label_severity_plus": severity,
        "label_confidence_plus": np.ones_like(values[..., 0]),
        "label_confidence_minus_audit": np.ones_like(values[..., 0]),
        "valid_for_azimuthal_validation": np.ones_like(presence, dtype=bool),
        "valid_for_non_azimuthal_summary": np.ones_like(presence, dtype=bool),
        "plus_minus_disagreement": presence == 1,
        "no_final_labels": np.asarray(True),
    }
    feature_arrays = {
        "xsi_basic_features_by_side": values,
        "feature_names": np.array(["rms_energy", "mean_abs"]),
        "no_model_training": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }

    rows, summary = evaluate_xsi_cast_correlation_from_arrays(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        correlation_config=_config(),
    )

    rms = next(
        row
        for row in rows
        if row["feature"] == "rms_energy" and row["subset"] == "high_confidence"
    )
    assert rms["candidate_mean"] == 10.0
    assert rms["non_candidate_mean"] == 1.0
    assert rms["mean_difference"] == 9.0
    assert rms["weighted_difference_fraction"] == 9.0
    assert summary["gate_observations"]["interpretable_signal_separation"] is True
    assert summary["gate_observations"]["no_model_training"] is True


def test_correlation_warns_when_high_confidence_subset_is_small() -> None:
    presence = np.array([[1, 0]], dtype=np.int8)
    feature_arrays = {
        "xsi_basic_features_by_side": np.ones((1, 2, 1), dtype=np.float32),
        "feature_names": np.array(["rms_energy"]),
        "no_model_training": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }
    label_arrays = {
        "label_presence_plus": presence,
        "label_presence_minus_audit": presence,
        "label_severity_plus": np.where(presence == 1, 1, 0).astype(np.int8),
        "label_confidence_plus": np.ones_like(presence, dtype=np.float32),
        "label_confidence_minus_audit": np.ones_like(presence, dtype=np.float32),
        "valid_for_azimuthal_validation": np.ones_like(presence, dtype=bool),
        "valid_for_non_azimuthal_summary": np.ones_like(presence, dtype=bool),
        "plus_minus_disagreement": np.zeros_like(presence, dtype=bool),
        "no_final_labels": np.asarray(True),
    }

    _rows, summary = evaluate_xsi_cast_correlation_from_arrays(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        correlation_config=_config(),
    )

    assert any("too few samples" in warning for warning in summary["warnings"])
