from __future__ import annotations

import numpy as np
import pytest

from cement_channel.labels.schema import (
    CAST_WEAK_LABEL_VERSION,
    CONVENTION_STATUS,
    LabelCandidateMetadata,
    LabelSource,
    PresenceLabel,
    SeverityLabel,
    validate_candidate_arrays,
    validate_candidate_metadata,
    validate_confidence_array,
    validate_convention_status,
    validate_label_source,
    validate_presence_array,
    validate_severity_array,
)


def test_label_schema_accepts_required_codes() -> None:
    presence = np.array(
        [
            PresenceLabel.UNKNOWN,
            PresenceLabel.NO_CHANNEL_CANDIDATE,
            PresenceLabel.CHANNEL_CANDIDATE,
        ],
        dtype=np.int8,
    )
    severity = np.array(
        [
            SeverityLabel.UNKNOWN,
            SeverityLabel.NONE,
            SeverityLabel.MILD,
            SeverityLabel.MODERATE,
            SeverityLabel.SEVERE,
        ],
        dtype=np.int8,
    )

    assert validate_presence_array(presence).valid
    assert validate_severity_array(severity).valid


def test_label_schema_rejects_invalid_codes() -> None:
    assert not validate_presence_array(np.array([0, 2], dtype=np.int8)).valid
    assert not validate_severity_array(np.array([0, 4], dtype=np.int8)).valid


def test_label_confidence_must_be_float_in_unit_interval() -> None:
    assert validate_confidence_array(np.array([0.0, 0.5, 1.0], dtype=np.float32)).valid
    invalid = validate_confidence_array(np.array([-0.1, 1.1], dtype=np.float32))
    assert not invalid.valid
    assert "within [0, 1]" in invalid.errors[0]


def test_candidate_array_shapes_must_match() -> None:
    result = validate_candidate_arrays(
        presence=np.zeros((2, 3), dtype=np.int8),
        severity=np.zeros((2, 3), dtype=np.int8),
        label_confidence=np.zeros((2, 2), dtype=np.float32),
    )

    assert not result.valid
    assert "shape mismatch" in result.errors[0]


def test_label_source_and_convention_status_are_strict() -> None:
    validate_label_source(LabelSource.CAST_WEAK_PLUS.value)
    validate_label_source(LabelSource.CAST_WEAK_MINUS_ABLATION.value)
    validate_convention_status(CONVENTION_STATUS)

    with pytest.raises(ValueError, match="Unsupported label_source"):
        validate_label_source("final_label")
    with pytest.raises(ValueError, match="Unsupported convention_status"):
        validate_convention_status("confirmed_plus")


def test_candidate_metadata_requires_no_final_labels() -> None:
    metadata = LabelCandidateMetadata(
        label_version=CAST_WEAK_LABEL_VERSION,
        label_source=LabelSource.CAST_WEAK_PLUS.value,
        convention="plus",
        convention_status=CONVENTION_STATUS,
        no_final_labels=True,
    )

    assert validate_candidate_metadata(metadata).valid

    invalid = LabelCandidateMetadata(
        label_version=CAST_WEAK_LABEL_VERSION,
        label_source=LabelSource.CAST_WEAK_PLUS.value,
        convention="plus",
        convention_status=CONVENTION_STATUS,
        no_final_labels=False,
    )
    assert not validate_candidate_metadata(invalid).valid
