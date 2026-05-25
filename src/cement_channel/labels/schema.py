from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum, IntFlag
from typing import Any

import numpy as np

LABEL_SCHEMA_VERSION = "label_schema_v001"
CAST_WEAK_LABEL_VERSION = "cast_weak_v001"
CONVENTION_STATUS = "specification_preferred_plus_data_unresolved"


class PresenceLabel(IntEnum):
    UNKNOWN = -1
    NO_CHANNEL_CANDIDATE = 0
    CHANNEL_CANDIDATE = 1


class SeverityLabel(IntEnum):
    UNKNOWN = -1
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class LabelSource(str, Enum):
    CAST_WEAK_PLUS = "cast_weak_plus"
    CAST_WEAK_MINUS_ABLATION = "cast_weak_minus_ablation"


class RelBearingConventionStatus(str, Enum):
    SPECIFICATION_PREFERRED_PLUS_DATA_UNRESOLVED = CONVENTION_STATUS


class EvidenceFlag(IntFlag):
    NONE = 0
    ABS_THRESHOLD = 1
    RELATIVE_DROP = 2
    AZIMUTH_GRADIENT = 4
    DEPTH_CONTINUITY = 8
    FUSED_RULE = 16
    UNCERTAIN = 32


@dataclass(frozen=True)
class LabelArrayValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelCandidateMetadata:
    label_version: str
    label_source: str
    convention: str
    convention_status: str
    no_final_labels: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VALID_PRESENCE_VALUES = frozenset(int(item.value) for item in PresenceLabel)
VALID_SEVERITY_VALUES = frozenset(int(item.value) for item in SeverityLabel)
VALID_LABEL_SOURCES = frozenset(item.value for item in LabelSource)
VALID_CONVENTION_STATUSES = frozenset(item.value for item in RelBearingConventionStatus)


def validate_presence_array(values: np.ndarray) -> LabelArrayValidation:
    array = np.asarray(values)
    errors: list[str] = []
    warnings: list[str] = []
    if not np.issubdtype(array.dtype, np.integer):
        errors.append(f"presence must be an integer array, observed {array.dtype}.")
    invalid = _invalid_values(array, VALID_PRESENCE_VALUES)
    if invalid:
        errors.append(f"presence contains invalid code(s): {invalid}.")
    if array.size == 0:
        errors.append("presence array is empty.")
    return LabelArrayValidation(valid=not errors, errors=errors, warnings=warnings)


def validate_severity_array(values: np.ndarray) -> LabelArrayValidation:
    array = np.asarray(values)
    errors: list[str] = []
    warnings: list[str] = []
    if not np.issubdtype(array.dtype, np.integer):
        errors.append(f"severity must be an integer array, observed {array.dtype}.")
    invalid = _invalid_values(array, VALID_SEVERITY_VALUES)
    if invalid:
        errors.append(f"severity contains invalid code(s): {invalid}.")
    if array.size == 0:
        errors.append("severity array is empty.")
    return LabelArrayValidation(valid=not errors, errors=errors, warnings=warnings)


def validate_confidence_array(values: np.ndarray) -> LabelArrayValidation:
    array = np.asarray(values)
    errors: list[str] = []
    warnings: list[str] = []
    if not np.issubdtype(array.dtype, np.floating):
        errors.append(f"label_confidence must be a floating array, observed {array.dtype}.")
    finite = np.isfinite(array)
    if array.size == 0:
        errors.append("label_confidence array is empty.")
    if np.any(finite & ((array < 0.0) | (array > 1.0))):
        errors.append("label_confidence must be within [0, 1].")
    if np.any(~finite):
        warnings.append("label_confidence contains non-finite values.")
    return LabelArrayValidation(valid=not errors, errors=errors, warnings=warnings)


def validate_label_source(value: str) -> None:
    if value not in VALID_LABEL_SOURCES:
        raise ValueError(f"Unsupported label_source: {value}")


def validate_convention_status(value: str) -> None:
    if value not in VALID_CONVENTION_STATUSES:
        raise ValueError(f"Unsupported convention_status: {value}")


def validate_candidate_metadata(metadata: LabelCandidateMetadata) -> LabelArrayValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if metadata.label_version != CAST_WEAK_LABEL_VERSION:
        errors.append(
            f"label_version must be {CAST_WEAK_LABEL_VERSION}, observed {metadata.label_version}."
        )
    if metadata.label_source not in VALID_LABEL_SOURCES:
        errors.append(f"Unsupported label_source: {metadata.label_source}.")
    if metadata.convention not in {"plus", "minus"}:
        errors.append(f"Unsupported convention: {metadata.convention}.")
    if metadata.convention_status != CONVENTION_STATUS:
        errors.append(
            "convention_status must remain "
            f"{CONVENTION_STATUS}, observed {metadata.convention_status}."
        )
    if not metadata.no_final_labels:
        errors.append("MVP-3 artifacts must set no_final_labels=true.")
    return LabelArrayValidation(valid=not errors, errors=errors, warnings=warnings)


def validate_candidate_arrays(
    *,
    presence: np.ndarray,
    severity: np.ndarray,
    label_confidence: np.ndarray,
) -> LabelArrayValidation:
    validations = [
        validate_presence_array(presence),
        validate_severity_array(severity),
        validate_confidence_array(label_confidence),
    ]
    errors = [message for result in validations for message in result.errors]
    warnings = [message for result in validations for message in result.warnings]
    if np.asarray(presence).shape != np.asarray(severity).shape:
        errors.append("presence and severity shape mismatch.")
    if np.asarray(presence).shape != np.asarray(label_confidence).shape:
        errors.append("presence and label_confidence shape mismatch.")
    return LabelArrayValidation(valid=not errors, errors=errors, warnings=warnings)


def _invalid_values(array: np.ndarray, valid_values: frozenset[int]) -> list[int]:
    if array.size == 0:
        return []
    unique = np.unique(array)
    return sorted(int(value) for value in unique if int(value) not in valid_values)
