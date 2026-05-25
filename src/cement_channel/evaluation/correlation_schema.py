from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MVP4A_CONFIG_VERSION = "mvp4a_xsi_cast_correlation_v001"
MVP4A_LABEL_SOURCE = "cast_weak_label_candidates_v001"
MVP4A_PRIMARY_LABEL = "plus"
MVP4A_AUDIT_LABEL = "minus_ablation"
MVP4A_SAMPLE_INDEX_VERSION = "xsi_label_samples_v001"
MVP4A_BASIC_FEATURE_VERSION = "xsi_basic_features_v001"
MVP4A_CORRELATION_VERSION = "xsi_cast_correlation_v001"
MVP4A_REVIEW_VERSION = "mvp4a_review_v001"
MVP4A_GATE_VERSION = "mvp4a_gate_v001"

SUPPORTED_XSI_BASIC_FEATURES = frozenset(
    {
        "rms_energy",
        "peak_abs",
        "mean_abs",
        "early_energy",
        "late_energy",
        "late_over_early_ratio",
    }
)
REQUIRED_XSI_BASIC_FEATURES = (
    "rms_energy",
    "peak_abs",
    "mean_abs",
    "early_energy",
    "late_energy",
)
SIDE_LABELS = tuple("ABCDEFGH")


@dataclass(frozen=True)
class CorrelationSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorrelationConfig:
    config_version: str
    label_source: str
    primary_label: str
    audit_label: str
    use_label_confidence: bool
    min_label_confidence_for_azimuthal_validation: float
    allow_low_confidence_for_non_azimuthal_summary: bool
    xsi_feature_set: tuple[str, ...]
    receiver_aggregation_method: str
    reference_receiver_index: int
    side_a_offset_deg: float
    side_order: str
    side_labels: tuple[str, ...]
    chunk_depth_samples: int
    max_time_samples: int
    high_confidence_min_samples_per_class: int
    min_interpretable_abs_effect_size: float
    min_interpretable_weighted_difference_fraction: float
    noncandidate_azimuthal_validation_requires_label_confidence: bool
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["xsi_feature_set"] = list(self.xsi_feature_set)
        data["side_labels"] = list(self.side_labels)
        return data


def load_correlation_config(path: Path | str) -> CorrelationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"MVP-4A config must contain a YAML mapping: {path}")
    config = parse_correlation_config(raw)
    validation = validate_correlation_config(config)
    if validation.errors:
        raise ValueError("Invalid MVP-4A config: " + "; ".join(validation.errors))
    return config


def parse_correlation_config(raw: dict[str, Any]) -> CorrelationConfig:
    receiver = _as_dict(raw.get("receiver_aggregation"))
    side_mapping = _as_dict(raw.get("side_mapping"))
    extraction = _as_dict(raw.get("feature_extraction"))
    correlation = _as_dict(raw.get("correlation"))
    sampler = _as_dict(raw.get("sampler"))
    features = tuple(str(value) for value in _as_list(raw.get("xsi_feature_set")))
    side_labels = tuple(str(value) for value in _as_list(side_mapping.get("side_labels")))
    return CorrelationConfig(
        config_version=str(raw.get("config_version", "")),
        label_source=str(raw.get("label_source", "")),
        primary_label=str(raw.get("primary_label", "")),
        audit_label=str(raw.get("audit_label", "")),
        use_label_confidence=bool(raw.get("use_label_confidence", False)),
        min_label_confidence_for_azimuthal_validation=float(
            raw.get("min_label_confidence_for_azimuthal_validation", 0.0)
        ),
        allow_low_confidence_for_non_azimuthal_summary=bool(
            raw.get("allow_low_confidence_for_non_azimuthal_summary", False)
        ),
        xsi_feature_set=features,
        receiver_aggregation_method=str(receiver.get("method", "")),
        reference_receiver_index=int(receiver.get("reference_receiver_index", 0)),
        side_a_offset_deg=float(side_mapping.get("side_a_offset_deg", 0.0)),
        side_order=str(side_mapping.get("side_order", "")),
        side_labels=side_labels,
        chunk_depth_samples=int(extraction.get("chunk_depth_samples", 0)),
        max_time_samples=int(extraction.get("max_time_samples", 0)),
        high_confidence_min_samples_per_class=int(
            correlation.get("high_confidence_min_samples_per_class", 0)
        ),
        min_interpretable_abs_effect_size=float(
            correlation.get("min_interpretable_abs_effect_size", 0.0)
        ),
        min_interpretable_weighted_difference_fraction=float(
            correlation.get("min_interpretable_weighted_difference_fraction", 0.0)
        ),
        noncandidate_azimuthal_validation_requires_label_confidence=bool(
            sampler.get("noncandidate_azimuthal_validation_requires_label_confidence", False)
        ),
        no_model_training=bool(raw.get("no_model_training", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
    )


def validate_correlation_config(config: CorrelationConfig) -> CorrelationSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.config_version != MVP4A_CONFIG_VERSION:
        errors.append(
            f"config_version must be {MVP4A_CONFIG_VERSION}, observed {config.config_version}."
        )
    if config.label_source != MVP4A_LABEL_SOURCE:
        errors.append(f"label_source must be {MVP4A_LABEL_SOURCE}.")
    if config.primary_label != MVP4A_PRIMARY_LABEL:
        errors.append("primary_label must remain plus for MVP-4A.")
    if config.audit_label != MVP4A_AUDIT_LABEL:
        errors.append("audit_label must remain minus_ablation for MVP-4A.")
    if not config.use_label_confidence:
        errors.append("use_label_confidence must be true.")
    if not 0.0 <= config.min_label_confidence_for_azimuthal_validation <= 1.0:
        errors.append("min_label_confidence_for_azimuthal_validation must be within [0, 1].")
    missing_features = [
        feature for feature in REQUIRED_XSI_BASIC_FEATURES if feature not in config.xsi_feature_set
    ]
    if missing_features:
        errors.append("xsi_feature_set missing required feature(s): " + ", ".join(missing_features))
    unsupported_features = [
        feature
        for feature in config.xsi_feature_set
        if feature not in SUPPORTED_XSI_BASIC_FEATURES
    ]
    if unsupported_features:
        errors.append(
            "xsi_feature_set contains unsupported feature(s): "
            + ", ".join(unsupported_features)
        )
    if config.receiver_aggregation_method not in {"mean", "median", "mean_or_median"}:
        errors.append("receiver_aggregation.method must be mean, median, or mean_or_median.")
    if not 1 <= config.reference_receiver_index <= 13:
        errors.append("receiver_aggregation.reference_receiver_index must be within [1, 13].")
    if config.side_labels != SIDE_LABELS:
        errors.append("side_mapping.side_labels must be [A, B, C, D, E, F, G, H].")
    if config.side_order != "clockwise":
        errors.append("side_mapping.side_order must remain clockwise unless separately audited.")
    if config.chunk_depth_samples <= 0:
        errors.append("feature_extraction.chunk_depth_samples must be positive.")
    if config.max_time_samples <= 0:
        errors.append("feature_extraction.max_time_samples must be positive.")
    if config.high_confidence_min_samples_per_class <= 0:
        errors.append("correlation.high_confidence_min_samples_per_class must be positive.")
    if config.min_interpretable_abs_effect_size < 0.0:
        errors.append("correlation.min_interpretable_abs_effect_size must be non-negative.")
    if config.min_interpretable_weighted_difference_fraction < 0.0:
        errors.append(
            "correlation.min_interpretable_weighted_difference_fraction must be non-negative."
        )
    if not config.allow_low_confidence_for_non_azimuthal_summary:
        warnings.append("Low-confidence intervals will be excluded from non-azimuthal summaries.")
    if not config.no_model_training:
        errors.append("no_model_training must be true.")
    if not config.no_final_labels:
        errors.append("no_final_labels must be true.")
    if not config.no_stc:
        errors.append("no_stc must be true.")
    if not config.no_apes:
        errors.append("no_apes must be true.")
    return CorrelationSchemaValidation(valid=not errors, errors=errors, warnings=warnings)


def expected_feature_names(config: CorrelationConfig) -> tuple[str, ...]:
    features = list(config.xsi_feature_set)
    if "late_over_early_ratio" not in features:
        features.append("late_over_early_ratio")
    return tuple(features)


def reference_receiver_zero_based(config: CorrelationConfig) -> int:
    return config.reference_receiver_index - 1


def xsi_side_azimuth_deg(config: CorrelationConfig) -> Any:
    import numpy as np

    count = len(config.side_labels)
    offsets = np.arange(count, dtype=np.float32) * (360.0 / count)
    if config.side_order == "clockwise":
        return (float(config.side_a_offset_deg) + offsets) % 360.0
    return (float(config.side_a_offset_deg) - offsets) % 360.0


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
