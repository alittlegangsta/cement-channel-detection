from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEPTH_LEVEL_LABEL_CONFIG_VERSION = "depth_level_label_v001"
DEPTH_LEVEL_LABEL_VERSION = "depth_level_labels_v001"
DEPTH_LEVEL_LABEL_REPORT_VERSION = "depth_level_labels_report_v001"
DEPTH_LEVEL_XSI_FEATURE_VERSION = "depth_level_xsi_features_v001"
DEPTH_LEVEL_XSI_FEATURE_REPORT_VERSION = "depth_level_xsi_features_report_v001"
DEPTH_LEVEL_SEPARATION_AUDIT_VERSION = "depth_level_separation_audit_v001"
DEPTH_LEVEL_GATE_VERSION = "depth_level_target_gate_v001"

DEPTH_LEVEL_STAGE = "MVP-4B-R4"
DEPTH_LEVEL_TASK = "depth_level_target_review"
DEPTH_LEVEL_ALLOWED_SCOPE = "depth_level_target_review_only"
DEPTH_LEVEL_INPUT_CAST_LABELS = "cast_weak_label_candidates_v001"
DEPTH_LEVEL_INPUT_XSI_SAMPLES = "xsi_label_samples_v001"
DEPTH_LEVEL_OPTIONAL_SAMPLE_TABLE = "baseline_sample_table_receiver_enhanced_v001"
DEPTH_LEVEL_PRIMARY_LABEL = "plus"
DEPTH_LEVEL_AUDIT_LABEL = "minus_ablation"
DEPTH_LEVEL_LABEL_STATUS = "human_reviewed_candidate_v001"

DEPTH_LEVEL_REQUIRED_FIELDS = (
    "depth_has_channel_any",
    "depth_candidate_fraction",
    "depth_max_severity",
    "depth_max_confidence",
    "depth_min_zc",
    "depth_p05_zc",
    "depth_p10_zc",
    "depth_max_relative_drop",
    "depth_largest_azimuth_object_width",
    "depth_plus_minus_disagreement_fraction",
    "depth_orientation_confidence",
    "depth_label_confidence",
)


@dataclass(frozen=True)
class DepthLevelSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelStrongPositiveConfig:
    min_candidate_fraction: float
    min_max_severity: int
    min_label_confidence: float
    max_plus_minus_disagreement_fraction: float
    min_orientation_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelClearNegativeConfig:
    max_candidate_fraction: float
    min_label_confidence: float
    max_plus_minus_disagreement_fraction: float
    min_orientation_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelReviewIntervalConfig:
    name: str
    depth_min_ft: float
    depth_max_ft: float
    reason: str
    apply_by_default: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelQualityPolicy:
    strong_positive: DepthLevelStrongPositiveConfig
    clear_negative: DepthLevelClearNegativeConfig
    review_intervals: tuple[DepthLevelReviewIntervalConfig, ...]
    max_review_band_positive_fraction: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["review_intervals"] = [
            interval.to_dict() for interval in self.review_intervals
        ]
        return data


@dataclass(frozen=True)
class DepthLevelGateConfig:
    min_depth_positive_count: int
    min_depth_negative_count: int
    max_5700_band_positive_fraction: float
    depth_level_improvement_effect_size_delta: float
    sanity_effect_size_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelLabelConfig:
    schema_version: str
    config_version: str
    stage: str
    task: str
    input_cast_weak_labels: str
    input_xsi_label_samples: str
    optional_input_sample_table: str
    output_depth_level_labels: str
    primary_label: str
    audit_label: str
    label_status: str
    depth_label_fields: tuple[str, ...]
    aggregation_policy: dict[str, Any]
    quality_policy: DepthLevelQualityPolicy
    gate: DepthLevelGateConfig
    allowed_scope: str
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["depth_label_fields"] = list(self.depth_label_fields)
        data["quality_policy"] = self.quality_policy.to_dict()
        data["gate"] = self.gate.to_dict()
        return data


def load_depth_level_label_config(path: Path | str) -> DepthLevelLabelConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Depth-level label config must be a YAML mapping: {path}")
    config = parse_depth_level_label_config(raw)
    validation = validate_depth_level_label_config(config)
    if validation.errors:
        raise ValueError(
            "Invalid depth-level label config: " + "; ".join(validation.errors)
        )
    return config


def parse_depth_level_label_config(raw: dict[str, Any]) -> DepthLevelLabelConfig:
    quality = _as_dict(raw.get("quality_policy"))
    strong = _as_dict(quality.get("strong_positive"))
    clear = _as_dict(quality.get("clear_negative"))
    gate = _as_dict(raw.get("gate"))
    review_intervals = tuple(
        DepthLevelReviewIntervalConfig(
            name=str(item.get("name", "")),
            depth_min_ft=float(item.get("depth_min_ft", 0.0)),
            depth_max_ft=float(item.get("depth_max_ft", 0.0)),
            reason=str(item.get("reason", "")),
            apply_by_default=bool(item.get("apply_by_default", False)),
        )
        for item in _as_list(quality.get("review_intervals"))
        if isinstance(item, dict)
    )
    return DepthLevelLabelConfig(
        schema_version=str(raw.get("schema_version", "")),
        config_version=str(raw.get("config_version", "")),
        stage=str(raw.get("stage", "")),
        task=str(raw.get("task", "")),
        input_cast_weak_labels=str(raw.get("input_cast_weak_labels", "")),
        input_xsi_label_samples=str(raw.get("input_xsi_label_samples", "")),
        optional_input_sample_table=str(raw.get("optional_input_sample_table", "")),
        output_depth_level_labels=str(raw.get("output_depth_level_labels", "")),
        primary_label=str(raw.get("primary_label", "")),
        audit_label=str(raw.get("audit_label", "")),
        label_status=str(raw.get("label_status", "")),
        depth_label_fields=tuple(str(item) for item in _as_list(raw.get("depth_label_fields"))),
        aggregation_policy=_as_dict(raw.get("aggregation_policy")),
        quality_policy=DepthLevelQualityPolicy(
            strong_positive=DepthLevelStrongPositiveConfig(
                min_candidate_fraction=float(strong.get("min_candidate_fraction", -1.0)),
                min_max_severity=int(strong.get("min_max_severity", -99)),
                min_label_confidence=float(strong.get("min_label_confidence", -1.0)),
                max_plus_minus_disagreement_fraction=float(
                    strong.get("max_plus_minus_disagreement_fraction", -1.0)
                ),
                min_orientation_confidence=float(strong.get("min_orientation_confidence", -1.0)),
            ),
            clear_negative=DepthLevelClearNegativeConfig(
                max_candidate_fraction=float(clear.get("max_candidate_fraction", -1.0)),
                min_label_confidence=float(clear.get("min_label_confidence", -1.0)),
                max_plus_minus_disagreement_fraction=float(
                    clear.get("max_plus_minus_disagreement_fraction", -1.0)
                ),
                min_orientation_confidence=float(clear.get("min_orientation_confidence", -1.0)),
            ),
            review_intervals=review_intervals,
            max_review_band_positive_fraction=float(
                quality.get("max_review_band_positive_fraction", -1.0)
            ),
        ),
        gate=DepthLevelGateConfig(
            min_depth_positive_count=int(gate.get("min_depth_positive_count", 0)),
            min_depth_negative_count=int(gate.get("min_depth_negative_count", 0)),
            max_5700_band_positive_fraction=float(
                gate.get("max_5700_band_positive_fraction", -1.0)
            ),
            depth_level_improvement_effect_size_delta=float(
                gate.get("depth_level_improvement_effect_size_delta", -1.0)
            ),
            sanity_effect_size_threshold=float(gate.get("sanity_effect_size_threshold", -1.0)),
        ),
        allowed_scope=str(raw.get("allowed_scope", "")),
        no_model_training=bool(raw.get("no_model_training", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
        no_deep_learning=bool(raw.get("no_deep_learning", False)),
        no_mvp4c=bool(raw.get("no_mvp4c", False)),
    )


def validate_depth_level_label_config(
    config: DepthLevelLabelConfig,
) -> DepthLevelSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.schema_version != "schema_v001":
        errors.append("schema_version must be schema_v001.")
    if config.config_version != DEPTH_LEVEL_LABEL_CONFIG_VERSION:
        errors.append(
            "config_version must be "
            f"{DEPTH_LEVEL_LABEL_CONFIG_VERSION}, observed {config.config_version}."
        )
    if config.stage != DEPTH_LEVEL_STAGE:
        errors.append(f"stage must be {DEPTH_LEVEL_STAGE}.")
    if config.task != DEPTH_LEVEL_TASK:
        errors.append(f"task must be {DEPTH_LEVEL_TASK}.")
    if config.input_cast_weak_labels != DEPTH_LEVEL_INPUT_CAST_LABELS:
        errors.append(f"input_cast_weak_labels must be {DEPTH_LEVEL_INPUT_CAST_LABELS}.")
    if config.input_xsi_label_samples != DEPTH_LEVEL_INPUT_XSI_SAMPLES:
        errors.append(f"input_xsi_label_samples must be {DEPTH_LEVEL_INPUT_XSI_SAMPLES}.")
    if config.optional_input_sample_table != DEPTH_LEVEL_OPTIONAL_SAMPLE_TABLE:
        errors.append(
            f"optional_input_sample_table must be {DEPTH_LEVEL_OPTIONAL_SAMPLE_TABLE}."
        )
    if config.output_depth_level_labels != DEPTH_LEVEL_LABEL_VERSION:
        errors.append(f"output_depth_level_labels must be {DEPTH_LEVEL_LABEL_VERSION}.")
    if config.primary_label != DEPTH_LEVEL_PRIMARY_LABEL:
        errors.append("primary_label must remain plus.")
    if config.audit_label != DEPTH_LEVEL_AUDIT_LABEL:
        errors.append("audit_label must remain minus_ablation.")
    if config.label_status != DEPTH_LEVEL_LABEL_STATUS:
        errors.append(f"label_status must be {DEPTH_LEVEL_LABEL_STATUS}.")
    _validate_depth_fields(config.depth_label_fields, errors)
    _validate_aggregation_policy(config.aggregation_policy, errors)
    _validate_quality_policy(config.quality_policy, errors, warnings)
    _validate_gate(config.gate, errors)
    if config.allowed_scope != DEPTH_LEVEL_ALLOWED_SCOPE:
        errors.append(f"allowed_scope must be {DEPTH_LEVEL_ALLOWED_SCOPE}.")
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
    return DepthLevelSchemaValidation(valid=not errors, errors=errors, warnings=warnings)


def active_depth_review_intervals(
    config: DepthLevelLabelConfig,
) -> tuple[DepthLevelReviewIntervalConfig, ...]:
    return tuple(
        interval for interval in config.quality_policy.review_intervals if interval.apply_by_default
    )


def _validate_depth_fields(fields: tuple[str, ...], errors: list[str]) -> None:
    missing = [field for field in DEPTH_LEVEL_REQUIRED_FIELDS if field not in fields]
    extra = [field for field in fields if field not in DEPTH_LEVEL_REQUIRED_FIELDS]
    if missing:
        errors.append("depth_label_fields missing required field(s): " + ", ".join(missing))
    if extra:
        errors.append("depth_label_fields contains unsupported field(s): " + ", ".join(extra))


def _validate_aggregation_policy(policy: dict[str, Any], errors: list[str]) -> None:
    if not policy:
        errors.append("aggregation_policy must be configured.")
        return
    if not bool(policy.get("require_any_max_percentile_fraction", False)):
        errors.append("aggregation_policy.require_any_max_percentile_fraction must be true.")
    if not bool(policy.get("forbid_mean_only", False)):
        errors.append("aggregation_policy.forbid_mean_only must be true.")
    expected = {
        ("presence", "has_channel_method", "any"),
        ("presence", "candidate_fraction_method", "fraction"),
        ("severity", "max_method", "max"),
        ("confidence", "max_method", "max"),
        ("zc", "min_method", "min"),
        ("relative_drop", "max_method", "max"),
        ("object_width", "method", "largest_connected_azimuth_width"),
        ("disagreement", "plus_minus_method", "fraction"),
    }
    for section, key, expected_value in expected:
        section_data = _as_dict(policy.get(section))
        if section_data.get(key) != expected_value:
            errors.append(f"aggregation_policy.{section}.{key} must be {expected_value}.")
    percentiles = set(
        str(value) for value in _as_list(_as_dict(policy.get("zc")).get("percentile_methods"))
    )
    if not {"p05", "p10"}.issubset(percentiles):
        errors.append("aggregation_policy.zc.percentile_methods must include p05 and p10.")
    side_policy = _as_dict(policy.get("side_level_labels"))
    if side_policy.get("usage") != "audit_only":
        errors.append("side_level_labels.usage must be audit_only.")
    if bool(side_policy.get("train_target", True)):
        errors.append("side_level_labels.train_target must be false.")


def _validate_quality_policy(
    policy: DepthLevelQualityPolicy,
    errors: list[str],
    warnings: list[str],
) -> None:
    strong = policy.strong_positive
    clear = policy.clear_negative
    _validate_probability(
        strong.min_candidate_fraction,
        "quality_policy.strong_positive.min_candidate_fraction",
        errors,
    )
    if strong.min_candidate_fraction <= 0.0:
        errors.append("strong_positive.min_candidate_fraction must be positive.")
    if strong.min_max_severity not in {1, 2, 3}:
        errors.append("strong_positive.min_max_severity must be 1, 2, or 3.")
    _validate_probability(
        strong.min_label_confidence,
        "quality_policy.strong_positive.min_label_confidence",
        errors,
    )
    _validate_probability(
        strong.max_plus_minus_disagreement_fraction,
        "quality_policy.strong_positive.max_plus_minus_disagreement_fraction",
        errors,
    )
    _validate_probability(
        strong.min_orientation_confidence,
        "quality_policy.strong_positive.min_orientation_confidence",
        errors,
    )
    _validate_probability(
        clear.max_candidate_fraction,
        "quality_policy.clear_negative.max_candidate_fraction",
        errors,
    )
    _validate_probability(
        clear.min_label_confidence,
        "quality_policy.clear_negative.min_label_confidence",
        errors,
    )
    _validate_probability(
        clear.max_plus_minus_disagreement_fraction,
        "quality_policy.clear_negative.max_plus_minus_disagreement_fraction",
        errors,
    )
    _validate_probability(
        clear.min_orientation_confidence,
        "quality_policy.clear_negative.min_orientation_confidence",
        errors,
    )
    if clear.max_candidate_fraction >= strong.min_candidate_fraction:
        errors.append(
            "clear_negative.max_candidate_fraction must be lower than "
            "strong_positive.min_candidate_fraction."
        )
    _validate_probability(
        policy.max_review_band_positive_fraction,
        "quality_policy.max_review_band_positive_fraction",
        errors,
    )
    _validate_review_intervals(policy.review_intervals, errors, warnings)


def _validate_gate(config: DepthLevelGateConfig, errors: list[str]) -> None:
    if config.min_depth_positive_count <= 0:
        errors.append("gate.min_depth_positive_count must be positive.")
    if config.min_depth_negative_count <= 0:
        errors.append("gate.min_depth_negative_count must be positive.")
    _validate_probability(
        config.max_5700_band_positive_fraction,
        "gate.max_5700_band_positive_fraction",
        errors,
    )
    if config.depth_level_improvement_effect_size_delta < 0.0:
        errors.append("gate.depth_level_improvement_effect_size_delta must be non-negative.")
    if config.sanity_effect_size_threshold < 0.0:
        errors.append("gate.sanity_effect_size_threshold must be non-negative.")


def _validate_review_intervals(
    intervals: tuple[DepthLevelReviewIntervalConfig, ...],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not intervals:
        errors.append("quality_policy.review_intervals must include the ~5700 ft band.")
        return
    has_5700 = False
    for interval in intervals:
        if not interval.name:
            errors.append("review interval name must be non-empty.")
        if interval.depth_min_ft >= interval.depth_max_ft:
            errors.append(f"review interval {interval.name} has invalid depth bounds.")
        if interval.depth_min_ft <= 5700.0 <= interval.depth_max_ft:
            has_5700 = True
        if not interval.apply_by_default:
            warnings.append(f"review interval {interval.name} is not applied by default.")
    if not has_5700:
        errors.append("quality_policy.review_intervals must include a band covering ~5700 ft.")


def _validate_probability(value: float, name: str, errors: list[str]) -> None:
    if not 0.0 <= value <= 1.0:
        errors.append(f"{name} must be within [0, 1].")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
