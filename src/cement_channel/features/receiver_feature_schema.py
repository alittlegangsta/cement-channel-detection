from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

MVP4B_RECEIVER_FEATURE_CONFIG_VERSION = "mvp4b_receiver_features_v001"
MVP4B_RECEIVER_FEATURE_VERSION = "receiver_derived_features_v001"
MVP4B_RECEIVER_FEATURE_REPORT_VERSION = "receiver_derived_feature_report_v001"
MVP4B_RECEIVER_ABLATION_VERSION = "receiver_feature_ablation_v001"
MVP4B_RECEIVER_GATE_VERSION = "receiver_feature_gate_v001"

MVP4B_BASIC_FEATURE_INPUT = "xsi_basic_features_v001"
MVP4B_SAMPLE_TABLE_INPUT = "baseline_sample_table_enhanced_v001"
MVP4B_LABEL_STATUS = "human_reviewed_candidate_v001"
MVP4B_PRIMARY_LABEL = "plus"
MVP4B_AUDIT_LABEL = "minus_ablation"
MVP4B_RECEIVER_ALLOWED_SCOPE = "receiver_feature_remediation_only"

SUPPORTED_SOURCE_FEATURES = frozenset(
    {
        "rms_energy",
        "peak_abs",
        "mean_abs",
        "early_energy",
        "late_energy",
        "late_over_early_ratio",
    }
)
REQUIRED_SOURCE_FEATURES = (
    "rms_energy",
    "peak_abs",
    "mean_abs",
    "early_energy",
    "late_energy",
    "late_over_early_ratio",
)
SUPPORTED_RECEIVER_FEATURES = frozenset(
    {
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
    }
)
REQUIRED_RECEIVER_FEATURES = tuple(sorted(SUPPORTED_RECEIVER_FEATURES))


@dataclass(frozen=True)
class ReceiverFeatureSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverGeometry:
    receiver_count: int
    reference_receiver_index: int
    receiver_spacing_ft: float
    source_to_receiver1_ft: float
    receiver_offsets_from_reference_ft: tuple[float, ...]
    near_receivers: tuple[int, ...]
    mid_receivers: tuple[int, ...]
    far_receivers: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverFeatureConfig:
    config_version: str
    input_basic_features: str
    input_sample_table: str
    primary_label: str
    audit_label: str
    label_status: str
    geometry: ReceiverGeometry
    source_feature_names: tuple[str, ...]
    receiver_feature_set: tuple[str, ...]
    log1p_positive_features: bool
    robust_scaling: bool
    clip_quantiles: tuple[float, float]
    epsilon: float
    use_high_confidence_for_azimuthal: bool
    min_label_confidence: float
    low_confidence_usage: str
    exclude_large_depth_match_error: bool
    max_depth_match_error_ft: float
    plus_minus_disagreement_policy: str
    required_margin_over_permutation: float
    max_degenerate_positive_rate: float
    min_degenerate_positive_rate: float
    required_folds_above_permutation: int
    allowed_scope: str
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_feature_names"] = list(self.source_feature_names)
        data["receiver_feature_set"] = list(self.receiver_feature_set)
        data["clip_quantiles"] = list(self.clip_quantiles)
        return data


def load_receiver_feature_config(path: Path | str) -> ReceiverFeatureConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"MVP-4B receiver feature config must be a YAML mapping: {path}")
    config = parse_receiver_feature_config(raw)
    validation = validate_receiver_feature_config(config)
    if validation.errors:
        raise ValueError(
            "Invalid MVP-4B receiver feature config: " + "; ".join(validation.errors)
        )
    return config


def parse_receiver_feature_config(raw: dict[str, Any]) -> ReceiverFeatureConfig:
    geometry_raw = _as_dict(raw.get("receiver_geometry"))
    transforms = _as_dict(raw.get("transforms"))
    sample_policy = _as_dict(raw.get("sample_policy"))
    ablation = _as_dict(raw.get("ablation"))
    clip_quantiles = tuple(float(value) for value in _as_list(transforms.get("clip_quantiles")))
    if len(clip_quantiles) != 2:
        clip_quantiles = (0.001, 0.999)
    geometry = ReceiverGeometry(
        receiver_count=int(geometry_raw.get("receiver_count", 0)),
        reference_receiver_index=int(geometry_raw.get("reference_receiver_index", 0)),
        receiver_spacing_ft=float(geometry_raw.get("receiver_spacing_ft", 0.0)),
        source_to_receiver1_ft=float(geometry_raw.get("source_to_receiver1_ft", 0.0)),
        receiver_offsets_from_reference_ft=tuple(
            float(value)
            for value in _as_list(geometry_raw.get("receiver_offsets_from_reference_ft"))
        ),
        near_receivers=tuple(int(value) for value in _as_list(geometry_raw.get("near_receivers"))),
        mid_receivers=tuple(int(value) for value in _as_list(geometry_raw.get("mid_receivers"))),
        far_receivers=tuple(int(value) for value in _as_list(geometry_raw.get("far_receivers"))),
    )
    return ReceiverFeatureConfig(
        config_version=str(raw.get("config_version", "")),
        input_basic_features=str(raw.get("input_basic_features", "")),
        input_sample_table=str(raw.get("input_sample_table", "")),
        primary_label=str(raw.get("primary_label", "")),
        audit_label=str(raw.get("audit_label", "")),
        label_status=str(raw.get("label_status", "")),
        geometry=geometry,
        source_feature_names=tuple(
            str(value) for value in _as_list(raw.get("source_feature_names"))
        ),
        receiver_feature_set=tuple(
            str(value) for value in _as_list(raw.get("receiver_feature_set"))
        ),
        log1p_positive_features=bool(transforms.get("log1p_positive_features", False)),
        robust_scaling=bool(transforms.get("robust_scaling", False)),
        clip_quantiles=(float(clip_quantiles[0]), float(clip_quantiles[1])),
        epsilon=float(transforms.get("epsilon", 1.0e-6)),
        use_high_confidence_for_azimuthal=bool(
            sample_policy.get("use_high_confidence_for_azimuthal", False)
        ),
        min_label_confidence=float(sample_policy.get("min_label_confidence", 0.0)),
        low_confidence_usage=str(sample_policy.get("low_confidence_usage", "")),
        exclude_large_depth_match_error=bool(
            sample_policy.get("exclude_large_depth_match_error", False)
        ),
        max_depth_match_error_ft=float(sample_policy.get("max_depth_match_error_ft", 0.0)),
        plus_minus_disagreement_policy=str(
            sample_policy.get("plus_minus_disagreement_policy", "")
        ),
        required_margin_over_permutation=float(
            ablation.get("required_margin_over_permutation", 0.0)
        ),
        max_degenerate_positive_rate=float(ablation.get("max_degenerate_positive_rate", 1.0)),
        min_degenerate_positive_rate=float(ablation.get("min_degenerate_positive_rate", 0.0)),
        required_folds_above_permutation=int(
            ablation.get("required_folds_above_permutation", 0)
        ),
        allowed_scope=str(raw.get("allowed_scope", "")),
        no_model_training=bool(raw.get("no_model_training", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
        no_deep_learning=bool(raw.get("no_deep_learning", False)),
        no_mvp4c=bool(raw.get("no_mvp4c", False)),
    )


def validate_receiver_feature_config(
    config: ReceiverFeatureConfig,
) -> ReceiverFeatureSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.config_version != MVP4B_RECEIVER_FEATURE_CONFIG_VERSION:
        errors.append(
            "config_version must be "
            f"{MVP4B_RECEIVER_FEATURE_CONFIG_VERSION}, observed {config.config_version}."
        )
    if config.input_basic_features != MVP4B_BASIC_FEATURE_INPUT:
        errors.append(f"input_basic_features must be {MVP4B_BASIC_FEATURE_INPUT}.")
    if config.input_sample_table != MVP4B_SAMPLE_TABLE_INPUT:
        errors.append(f"input_sample_table must be {MVP4B_SAMPLE_TABLE_INPUT}.")
    if config.primary_label != MVP4B_PRIMARY_LABEL:
        errors.append("primary_label must remain plus.")
    if config.audit_label != MVP4B_AUDIT_LABEL:
        errors.append("audit_label must remain minus_ablation.")
    if config.label_status != MVP4B_LABEL_STATUS:
        errors.append(f"label_status must be {MVP4B_LABEL_STATUS}.")
    _validate_geometry(config.geometry, errors)
    missing_source = [
        feature
        for feature in REQUIRED_SOURCE_FEATURES
        if feature not in config.source_feature_names
    ]
    if missing_source:
        errors.append(
            "source_feature_names missing required feature(s): " + ", ".join(missing_source)
        )
    unsupported_source = [
        feature
        for feature in config.source_feature_names
        if feature not in SUPPORTED_SOURCE_FEATURES
    ]
    if unsupported_source:
        errors.append(
            "source_feature_names contain unsupported feature(s): "
            + ", ".join(unsupported_source)
        )
    missing_receiver = [
        feature
        for feature in REQUIRED_RECEIVER_FEATURES
        if feature not in config.receiver_feature_set
    ]
    if missing_receiver:
        errors.append(
            "receiver_feature_set missing required feature(s): "
            + ", ".join(missing_receiver)
        )
    unsupported_receiver = [
        feature
        for feature in config.receiver_feature_set
        if feature not in SUPPORTED_RECEIVER_FEATURES
    ]
    if unsupported_receiver:
        errors.append(
            "receiver_feature_set contains unsupported feature(s): "
            + ", ".join(unsupported_receiver)
        )
    low, high = config.clip_quantiles
    if not 0.0 <= low < high <= 1.0:
        errors.append("transforms.clip_quantiles must satisfy 0 <= low < high <= 1.")
    if config.epsilon <= 0.0:
        errors.append("transforms.epsilon must be positive.")
    if not config.log1p_positive_features:
        warnings.append("log1p_positive_features is disabled.")
    if not config.robust_scaling:
        warnings.append("robust_scaling is disabled.")
    if not config.use_high_confidence_for_azimuthal:
        errors.append("sample_policy.use_high_confidence_for_azimuthal must be true.")
    if not 0.0 <= config.min_label_confidence <= 1.0:
        errors.append("sample_policy.min_label_confidence must be within [0, 1].")
    if config.low_confidence_usage != "non_azimuthal_or_excluded":
        errors.append("sample_policy.low_confidence_usage must be non_azimuthal_or_excluded.")
    if not config.exclude_large_depth_match_error:
        errors.append("sample_policy.exclude_large_depth_match_error must be true.")
    if config.max_depth_match_error_ft < 0.0:
        errors.append("sample_policy.max_depth_match_error_ft must be non-negative.")
    if config.plus_minus_disagreement_policy not in {"audit_flag_or_downweight", "exclude"}:
        errors.append(
            "sample_policy.plus_minus_disagreement_policy must be "
            "audit_flag_or_downweight or exclude."
        )
    if config.required_margin_over_permutation < 0.0:
        errors.append("ablation.required_margin_over_permutation must be non-negative.")
    if not 0.5 < config.max_degenerate_positive_rate <= 1.0:
        errors.append("ablation.max_degenerate_positive_rate must be in (0.5, 1].")
    if not 0.0 <= config.min_degenerate_positive_rate < 0.5:
        errors.append("ablation.min_degenerate_positive_rate must be in [0, 0.5).")
    if config.required_folds_above_permutation <= 0:
        errors.append("ablation.required_folds_above_permutation must be positive.")
    if config.allowed_scope != MVP4B_RECEIVER_ALLOWED_SCOPE:
        errors.append(f"allowed_scope must be {MVP4B_RECEIVER_ALLOWED_SCOPE}.")
    for field_name in (
        "no_model_training",
        "no_final_labels",
        "no_stc",
        "no_apes",
        "no_deep_learning",
        "no_mvp4c",
    ):
        if not bool(getattr(config, field_name)):
            errors.append(f"{field_name} must be true.")
    return ReceiverFeatureSchemaValidation(valid=not errors, errors=errors, warnings=warnings)


def receiver_group_zero_based(config: ReceiverFeatureConfig, group_name: str) -> np.ndarray:
    if group_name == "near":
        values = config.geometry.near_receivers
    elif group_name == "mid":
        values = config.geometry.mid_receivers
    elif group_name == "far":
        values = config.geometry.far_receivers
    else:
        raise ValueError(f"Unsupported receiver group: {group_name}")
    return np.asarray(values, dtype=np.int16) - 1


def receiver_offsets_ft(config: ReceiverFeatureConfig) -> np.ndarray:
    return np.asarray(config.geometry.receiver_offsets_from_reference_ft, dtype=np.float32)


def receiver_source_distances_ft(config: ReceiverFeatureConfig) -> np.ndarray:
    receiver_indices = np.arange(config.geometry.receiver_count, dtype=np.float32)
    return (
        float(config.geometry.source_to_receiver1_ft)
        + receiver_indices * float(config.geometry.receiver_spacing_ft)
    ).astype(np.float32)


def expected_receiver_feature_families(config: ReceiverFeatureConfig) -> tuple[str, ...]:
    return tuple(config.receiver_feature_set)


def _validate_geometry(geometry: ReceiverGeometry, errors: list[str]) -> None:
    if geometry.receiver_count != 13:
        errors.append("receiver_geometry.receiver_count must be 13.")
    if geometry.reference_receiver_index != 7:
        errors.append("receiver_geometry.reference_receiver_index must be 7.")
    if geometry.receiver_spacing_ft <= 0.0:
        errors.append("receiver_geometry.receiver_spacing_ft must be positive.")
    if geometry.source_to_receiver1_ft <= 0.0:
        errors.append("receiver_geometry.source_to_receiver1_ft must be positive.")
    if len(geometry.receiver_offsets_from_reference_ft) != geometry.receiver_count:
        errors.append(
            "receiver_geometry.receiver_offsets_from_reference_ft length must match "
            "receiver_count."
        )
    if geometry.near_receivers != (1, 2, 3, 4):
        errors.append("receiver_geometry.near_receivers must be [1, 2, 3, 4].")
    if geometry.mid_receivers != (5, 6, 7, 8, 9):
        errors.append("receiver_geometry.mid_receivers must be [5, 6, 7, 8, 9].")
    if geometry.far_receivers != (10, 11, 12, 13):
        errors.append("receiver_geometry.far_receivers must be [10, 11, 12, 13].")
    receiver_ids = set(geometry.near_receivers + geometry.mid_receivers + geometry.far_receivers)
    if receiver_ids != set(range(1, geometry.receiver_count + 1)):
        errors.append("receiver groups must cover receivers 1..13 exactly once.")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
