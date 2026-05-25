from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MVP4B_SAMPLE_CONFIG_VERSION = "mvp4b_sample_table_v001"
MVP4B_SAMPLE_TABLE_VERSION = "baseline_sample_table_v001"
MVP4B_PREPROCESSING_DIAGNOSTICS_VERSION = "feature_preprocessing_diagnostics_v001"
MVP4B_STAGE1_GATE_VERSION = "mvp4b_stage1_gate_v001"
MVP4B_INPUT_FEATURES = "xsi_basic_features_v001"
MVP4B_INPUT_LABELS = "xsi_label_samples_v001"
MVP4B_PRIMARY_LABEL = "plus"
MVP4B_AUDIT_LABEL = "minus_ablation"
MVP4B_FEATURE_NAMES = (
    "rms_energy",
    "peak_abs",
    "mean_abs",
    "early_energy",
    "late_energy",
    "late_over_early_ratio",
)


@dataclass(frozen=True)
class SampleSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SampleTableConfig:
    config_version: str
    input_features: str
    input_labels: str
    primary_label: str
    audit_label: str
    feature_names: tuple[str, ...]
    log1p: bool
    robust_scaling: bool
    clip_quantiles: tuple[float, float]
    per_feature_scaling: bool
    optional_per_depth_normalization: bool
    optional_per_side_normalization: bool
    use_high_confidence_for_azimuthal: bool
    min_label_confidence: float
    low_confidence_usage: str
    exclude_large_depth_match_error: bool
    max_depth_match_error_ft: float | None
    plus_minus_disagreement_policy: str
    plus_minus_disagreement_weight_multiplier: float
    depth_mismatch_weight_multiplier: float
    min_high_confidence_samples_per_class: int
    max_nonfinite_transformed_fraction: float
    min_positive_sample_weight_fraction: float
    max_large_depth_match_error_fraction: float
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_names"] = list(self.feature_names)
        data["clip_quantiles"] = list(self.clip_quantiles)
        return data


def load_sample_table_config(path: Path | str) -> SampleTableConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"MVP-4B sample table config must contain a YAML mapping: {path}")
    config = parse_sample_table_config(raw)
    validation = validate_sample_table_config(config)
    if validation.errors:
        raise ValueError("Invalid MVP-4B sample table config: " + "; ".join(validation.errors))
    return config


def parse_sample_table_config(raw: dict[str, Any]) -> SampleTableConfig:
    transforms = _as_dict(raw.get("transforms"))
    policy = _as_dict(raw.get("sample_policy"))
    diagnostics = _as_dict(raw.get("diagnostics"))
    clip_quantiles = _as_float_pair(transforms.get("clip_quantiles"), (0.001, 0.999))
    max_depth_error = policy.get("max_depth_match_error_ft")
    max_depth_match_error_ft = None if str(max_depth_error) == "TODO_CONFIG" else max_depth_error
    return SampleTableConfig(
        config_version=str(raw.get("config_version", "")),
        input_features=str(raw.get("input_features", "")),
        input_labels=str(raw.get("input_labels", "")),
        primary_label=str(raw.get("primary_label", "")),
        audit_label=str(raw.get("audit_label", "")),
        feature_names=tuple(str(value) for value in _as_list(raw.get("feature_names"))),
        log1p=bool(transforms.get("log1p", False)),
        robust_scaling=bool(transforms.get("robust_scaling", False)),
        clip_quantiles=clip_quantiles,
        per_feature_scaling=bool(transforms.get("per_feature_scaling", False)),
        optional_per_depth_normalization=bool(
            transforms.get("optional_per_depth_normalization", False)
        ),
        optional_per_side_normalization=bool(
            transforms.get("optional_per_side_normalization", False)
        ),
        use_high_confidence_for_azimuthal=bool(
            policy.get("use_high_confidence_for_azimuthal", False)
        ),
        min_label_confidence=float(policy.get("min_label_confidence", 0.0)),
        low_confidence_usage=str(policy.get("low_confidence_usage", "")),
        exclude_large_depth_match_error=bool(policy.get("exclude_large_depth_match_error", False)),
        max_depth_match_error_ft=(
            None if max_depth_match_error_ft is None else float(max_depth_match_error_ft)
        ),
        plus_minus_disagreement_policy=str(policy.get("plus_minus_disagreement_policy", "")),
        plus_minus_disagreement_weight_multiplier=float(
            policy.get("plus_minus_disagreement_weight_multiplier", 1.0)
        ),
        depth_mismatch_weight_multiplier=float(
            policy.get("depth_mismatch_weight_multiplier", 0.0)
        ),
        min_high_confidence_samples_per_class=int(
            diagnostics.get("min_high_confidence_samples_per_class", 20)
        ),
        max_nonfinite_transformed_fraction=float(
            diagnostics.get("max_nonfinite_transformed_fraction", 0.001)
        ),
        min_positive_sample_weight_fraction=float(
            diagnostics.get("min_positive_sample_weight_fraction", 0.05)
        ),
        max_large_depth_match_error_fraction=float(
            diagnostics.get("max_large_depth_match_error_fraction", 0.20)
        ),
        no_model_training=bool(raw.get("no_model_training", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
    )


def validate_sample_table_config(config: SampleTableConfig) -> SampleSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.config_version != MVP4B_SAMPLE_CONFIG_VERSION:
        errors.append(
            f"config_version must be {MVP4B_SAMPLE_CONFIG_VERSION}, "
            f"observed {config.config_version}."
        )
    if config.input_features != MVP4B_INPUT_FEATURES:
        errors.append(f"input_features must be {MVP4B_INPUT_FEATURES}.")
    if config.input_labels != MVP4B_INPUT_LABELS:
        errors.append(f"input_labels must be {MVP4B_INPUT_LABELS}.")
    if config.primary_label != MVP4B_PRIMARY_LABEL:
        errors.append("primary_label must remain plus.")
    if config.audit_label != MVP4B_AUDIT_LABEL:
        errors.append("audit_label must remain minus_ablation.")
    missing = [feature for feature in MVP4B_FEATURE_NAMES if feature not in config.feature_names]
    unsupported = [
        feature for feature in config.feature_names if feature not in MVP4B_FEATURE_NAMES
    ]
    if missing:
        errors.append("feature_names missing required feature(s): " + ", ".join(missing))
    if unsupported:
        errors.append("feature_names contains unsupported feature(s): " + ", ".join(unsupported))
    q_low, q_high = config.clip_quantiles
    if not 0.0 <= q_low < q_high <= 1.0:
        errors.append("transforms.clip_quantiles must satisfy 0 <= low < high <= 1.")
    if not config.log1p:
        warnings.append("log1p transform is disabled.")
    if not config.robust_scaling:
        warnings.append("robust scaling is disabled.")
    if not config.per_feature_scaling:
        warnings.append("per-feature scaling is disabled.")
    if not config.use_high_confidence_for_azimuthal:
        errors.append("sample_policy.use_high_confidence_for_azimuthal must be true.")
    if not 0.0 <= config.min_label_confidence <= 1.0:
        errors.append("sample_policy.min_label_confidence must be within [0, 1].")
    if config.low_confidence_usage != "non_azimuthal_or_excluded":
        errors.append("sample_policy.low_confidence_usage must be non_azimuthal_or_excluded.")
    if config.max_depth_match_error_ft is None:
        warnings.append("sample_policy.max_depth_match_error_ft is TODO/unconfigured.")
    elif config.max_depth_match_error_ft < 0.0:
        errors.append("sample_policy.max_depth_match_error_ft must be non-negative.")
    if config.plus_minus_disagreement_policy not in {"audit_flag_or_downweight", "audit_flag"}:
        errors.append(
            "sample_policy.plus_minus_disagreement_policy must be audit_flag_or_downweight "
            "or audit_flag."
        )
    if not 0.0 <= config.plus_minus_disagreement_weight_multiplier <= 1.0:
        errors.append("plus_minus_disagreement_weight_multiplier must be within [0, 1].")
    if not 0.0 <= config.depth_mismatch_weight_multiplier <= 1.0:
        errors.append("depth_mismatch_weight_multiplier must be within [0, 1].")
    if config.min_high_confidence_samples_per_class <= 0:
        errors.append("diagnostics.min_high_confidence_samples_per_class must be positive.")
    if not 0.0 <= config.max_nonfinite_transformed_fraction <= 1.0:
        errors.append("diagnostics.max_nonfinite_transformed_fraction must be within [0, 1].")
    if not 0.0 <= config.min_positive_sample_weight_fraction <= 1.0:
        errors.append("diagnostics.min_positive_sample_weight_fraction must be within [0, 1].")
    if not 0.0 <= config.max_large_depth_match_error_fraction <= 1.0:
        errors.append("diagnostics.max_large_depth_match_error_fraction must be within [0, 1].")
    if not config.no_model_training:
        errors.append("no_model_training must be true.")
    if not config.no_final_labels:
        errors.append("no_final_labels must be true.")
    if not config.no_stc:
        errors.append("no_stc must be true.")
    if not config.no_apes:
        errors.append("no_apes must be true.")
    return SampleSchemaValidation(valid=not errors, errors=errors, warnings=warnings)


def transformed_feature_names(config: SampleTableConfig) -> tuple[str, ...]:
    names: list[str] = []
    if config.log1p:
        names.extend(f"log1p_{feature}" for feature in config.feature_names)
    if config.robust_scaling:
        names.extend(f"robust_scaled_{feature}" for feature in config.feature_names)
    return tuple(names)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        return default
    return float(value[0]), float(value[1])
