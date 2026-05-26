from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MVP4B_LABEL_QUALITY_CONFIG_VERSION = "mvp4b_label_quality_subsets_v001"
MVP4B_LABEL_QUALITY_SUBSET_VERSION = "label_quality_subsets_v001"
MVP4B_LABEL_QUALITY_AUDIT_VERSION = "subset_feature_separation_audit_v001"
MVP4B_LABEL_QUALITY_GATE_VERSION = "label_quality_gate_v001"

MVP4B_LABEL_QUALITY_INPUT_SAMPLE_TABLE = "baseline_sample_table_receiver_enhanced_v001"
MVP4B_LABEL_QUALITY_OPTIONAL_CAST_LABELS = "cast_weak_label_candidates_v001"
MVP4B_LABEL_STATUS = "human_reviewed_candidate_v001"
MVP4B_PRIMARY_LABEL = "plus"
MVP4B_AUDIT_LABEL = "minus_ablation"
MVP4B_LABEL_QUALITY_ALLOWED_SCOPE = "label_quality_subset_diagnostics_only"


@dataclass(frozen=True)
class LabelQualitySchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrongPositiveSubsetConfig:
    label_presence_plus: int
    min_severity: int
    min_label_confidence: float
    require_no_plus_minus_disagreement: bool
    max_depth_match_error_ft: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClearNegativeSubsetConfig:
    label_presence_plus: int
    min_label_confidence: float
    allow_local_cast_normal: bool
    local_normal_requires_severity_none: bool
    min_local_normal_confidence: float
    require_no_plus_minus_disagreement: bool
    max_depth_match_error_ft: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrientationSubsetConfig:
    min_orientation_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConnectedObjectSubsetConfig:
    min_area_samples: int
    min_depth_length_ft: float
    circular_side_connectivity: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewIntervalConfig:
    name: str
    depth_min_ft: float
    depth_max_ft: float
    reason: str
    apply_by_default: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelQualityPolicy:
    min_subset_samples_per_class: int
    high_confidence_orientation_thresholds: tuple[float, ...]
    disagreement_policy: str
    suspicious_band_policy: str
    connected_object_policy: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["high_confidence_orientation_thresholds"] = list(
            self.high_confidence_orientation_thresholds
        )
        return data


@dataclass(frozen=True)
class LabelQualityGateConfig:
    signal_enhancement_effect_size_delta: float
    strong_signal_effect_size_threshold: float
    max_result_flip_fraction_from_review_exclusion: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelQualityConfig:
    config_version: str
    input_sample_table: str
    optional_cast_weak_labels: str
    primary_label: str
    audit_label: str
    label_status: str
    strong_positive: StrongPositiveSubsetConfig
    clear_negative: ClearNegativeSubsetConfig
    high_confidence_orientation: OrientationSubsetConfig
    connected_object_only: ConnectedObjectSubsetConfig
    exclude_review_intervals: tuple[ReviewIntervalConfig, ...]
    quality_policy: LabelQualityPolicy
    gate: LabelQualityGateConfig
    allowed_scope: str
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["exclude_review_intervals"] = [
            interval.to_dict() for interval in self.exclude_review_intervals
        ]
        data["quality_policy"] = self.quality_policy.to_dict()
        return data


def load_label_quality_config(path: Path | str) -> LabelQualityConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"MVP-4B label-quality config must be a YAML mapping: {path}")
    config = parse_label_quality_config(raw)
    validation = validate_label_quality_config(config)
    if validation.errors:
        raise ValueError(
            "Invalid MVP-4B label-quality subset config: " + "; ".join(validation.errors)
        )
    return config


def parse_label_quality_config(raw: dict[str, Any]) -> LabelQualityConfig:
    subsets = _as_dict(raw.get("subsets"))
    strong = _as_dict(subsets.get("strong_positive"))
    negative = _as_dict(subsets.get("clear_negative"))
    orientation = _as_dict(subsets.get("high_confidence_orientation"))
    connected = _as_dict(subsets.get("connected_object_only"))
    policy = _as_dict(raw.get("quality_policy"))
    gate = _as_dict(raw.get("gate"))
    intervals = tuple(
        ReviewIntervalConfig(
            name=str(item.get("name", "")),
            depth_min_ft=float(item.get("depth_min_ft", 0.0)),
            depth_max_ft=float(item.get("depth_max_ft", 0.0)),
            reason=str(item.get("reason", "")),
            apply_by_default=bool(item.get("apply_by_default", False)),
        )
        for item in _as_list(subsets.get("exclude_review_intervals"))
        if isinstance(item, dict)
    )
    return LabelQualityConfig(
        config_version=str(raw.get("config_version", "")),
        input_sample_table=str(raw.get("input_sample_table", "")),
        optional_cast_weak_labels=str(raw.get("optional_cast_weak_labels", "")),
        primary_label=str(raw.get("primary_label", "")),
        audit_label=str(raw.get("audit_label", "")),
        label_status=str(raw.get("label_status", "")),
        strong_positive=StrongPositiveSubsetConfig(
            label_presence_plus=int(strong.get("label_presence_plus", -99)),
            min_severity=int(strong.get("min_severity", -99)),
            min_label_confidence=float(strong.get("min_label_confidence", -1.0)),
            require_no_plus_minus_disagreement=bool(
                strong.get("require_no_plus_minus_disagreement", False)
            ),
            max_depth_match_error_ft=float(strong.get("max_depth_match_error_ft", -1.0)),
        ),
        clear_negative=ClearNegativeSubsetConfig(
            label_presence_plus=int(negative.get("label_presence_plus", -99)),
            min_label_confidence=float(negative.get("min_label_confidence", -1.0)),
            allow_local_cast_normal=bool(negative.get("allow_local_cast_normal", False)),
            local_normal_requires_severity_none=bool(
                negative.get("local_normal_requires_severity_none", False)
            ),
            min_local_normal_confidence=float(negative.get("min_local_normal_confidence", -1.0)),
            require_no_plus_minus_disagreement=bool(
                negative.get("require_no_plus_minus_disagreement", False)
            ),
            max_depth_match_error_ft=float(negative.get("max_depth_match_error_ft", -1.0)),
        ),
        high_confidence_orientation=OrientationSubsetConfig(
            min_orientation_confidence=float(orientation.get("min_orientation_confidence", -1.0))
        ),
        connected_object_only=ConnectedObjectSubsetConfig(
            min_area_samples=int(connected.get("min_area_samples", 0)),
            min_depth_length_ft=float(connected.get("min_depth_length_ft", -1.0)),
            circular_side_connectivity=bool(connected.get("circular_side_connectivity", False)),
        ),
        exclude_review_intervals=intervals,
        quality_policy=LabelQualityPolicy(
            min_subset_samples_per_class=int(policy.get("min_subset_samples_per_class", 0)),
            high_confidence_orientation_thresholds=tuple(
                float(value)
                for value in _as_list(policy.get("high_confidence_orientation_thresholds"))
            ),
            disagreement_policy=str(policy.get("disagreement_policy", "")),
            suspicious_band_policy=str(policy.get("suspicious_band_policy", "")),
            connected_object_policy=str(policy.get("connected_object_policy", "")),
        ),
        gate=LabelQualityGateConfig(
            signal_enhancement_effect_size_delta=float(
                gate.get("signal_enhancement_effect_size_delta", 0.0)
            ),
            strong_signal_effect_size_threshold=float(
                gate.get("strong_signal_effect_size_threshold", 0.0)
            ),
            max_result_flip_fraction_from_review_exclusion=float(
                gate.get("max_result_flip_fraction_from_review_exclusion", 0.0)
            ),
        ),
        allowed_scope=str(raw.get("allowed_scope", "")),
        no_model_training=bool(raw.get("no_model_training", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
        no_deep_learning=bool(raw.get("no_deep_learning", False)),
        no_mvp4c=bool(raw.get("no_mvp4c", False)),
    )


def validate_label_quality_config(config: LabelQualityConfig) -> LabelQualitySchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.config_version != MVP4B_LABEL_QUALITY_CONFIG_VERSION:
        errors.append(
            "config_version must be "
            f"{MVP4B_LABEL_QUALITY_CONFIG_VERSION}, observed {config.config_version}."
        )
    if config.input_sample_table != MVP4B_LABEL_QUALITY_INPUT_SAMPLE_TABLE:
        errors.append(f"input_sample_table must be {MVP4B_LABEL_QUALITY_INPUT_SAMPLE_TABLE}.")
    if config.optional_cast_weak_labels != MVP4B_LABEL_QUALITY_OPTIONAL_CAST_LABELS:
        errors.append(
            f"optional_cast_weak_labels must be {MVP4B_LABEL_QUALITY_OPTIONAL_CAST_LABELS}."
        )
    if config.primary_label != MVP4B_PRIMARY_LABEL:
        errors.append("primary_label must remain plus.")
    if config.audit_label != MVP4B_AUDIT_LABEL:
        errors.append("audit_label must remain minus_ablation.")
    if config.label_status != MVP4B_LABEL_STATUS:
        errors.append(f"label_status must be {MVP4B_LABEL_STATUS}.")
    _validate_strong_positive(config.strong_positive, errors)
    _validate_clear_negative(config.clear_negative, errors)
    _validate_orientation(config.high_confidence_orientation, errors)
    _validate_connected(config.connected_object_only, errors)
    _validate_intervals(config.exclude_review_intervals, errors, warnings)
    _validate_policy(config.quality_policy, errors)
    _validate_gate(config.gate, errors)
    if config.allowed_scope != MVP4B_LABEL_QUALITY_ALLOWED_SCOPE:
        errors.append(f"allowed_scope must be {MVP4B_LABEL_QUALITY_ALLOWED_SCOPE}.")
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
    return LabelQualitySchemaValidation(valid=not errors, errors=errors, warnings=warnings)


def active_review_intervals(config: LabelQualityConfig) -> tuple[ReviewIntervalConfig, ...]:
    return tuple(
        interval for interval in config.exclude_review_intervals if interval.apply_by_default
    )


def _validate_strong_positive(
    config: StrongPositiveSubsetConfig,
    errors: list[str],
) -> None:
    if config.label_presence_plus != 1:
        errors.append("strong_positive.label_presence_plus must be 1.")
    if config.min_severity not in {2, 3}:
        errors.append("strong_positive.min_severity must be 2 (moderate) or 3 (severe).")
    if not 0.0 <= config.min_label_confidence <= 1.0:
        errors.append("strong_positive.min_label_confidence must be within [0, 1].")
    if not config.require_no_plus_minus_disagreement:
        errors.append("strong_positive must require no plus/minus disagreement.")
    if config.max_depth_match_error_ft < 0.0:
        errors.append("strong_positive.max_depth_match_error_ft must be non-negative.")


def _validate_clear_negative(config: ClearNegativeSubsetConfig, errors: list[str]) -> None:
    if config.label_presence_plus != 0:
        errors.append("clear_negative.label_presence_plus must be 0.")
    if not 0.0 <= config.min_label_confidence <= 1.0:
        errors.append("clear_negative.min_label_confidence must be within [0, 1].")
    if not config.allow_local_cast_normal:
        errors.append("clear_negative.allow_local_cast_normal must be true.")
    if not config.local_normal_requires_severity_none:
        errors.append("clear_negative.local_normal_requires_severity_none must be true.")
    if not 0.0 <= config.min_local_normal_confidence <= 1.0:
        errors.append("clear_negative.min_local_normal_confidence must be within [0, 1].")
    if not config.require_no_plus_minus_disagreement:
        errors.append("clear_negative must require no plus/minus disagreement.")
    if config.max_depth_match_error_ft < 0.0:
        errors.append("clear_negative.max_depth_match_error_ft must be non-negative.")


def _validate_orientation(config: OrientationSubsetConfig, errors: list[str]) -> None:
    if not 0.0 <= config.min_orientation_confidence <= 1.0:
        errors.append(
            "high_confidence_orientation.min_orientation_confidence must be within [0, 1]."
        )


def _validate_connected(config: ConnectedObjectSubsetConfig, errors: list[str]) -> None:
    if config.min_area_samples <= 0:
        errors.append("connected_object_only.min_area_samples must be positive.")
    if config.min_depth_length_ft < 0.0:
        errors.append("connected_object_only.min_depth_length_ft must be non-negative.")
    if not config.circular_side_connectivity:
        errors.append("connected_object_only.circular_side_connectivity must be true.")


def _validate_intervals(
    intervals: tuple[ReviewIntervalConfig, ...],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not intervals:
        errors.append("exclude_review_intervals must include the ~5700 ft review band.")
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
        errors.append("exclude_review_intervals must include a band covering ~5700 ft.")


def _validate_policy(config: LabelQualityPolicy, errors: list[str]) -> None:
    if config.min_subset_samples_per_class <= 0:
        errors.append("quality_policy.min_subset_samples_per_class must be positive.")
    if not config.high_confidence_orientation_thresholds:
        errors.append("quality_policy.high_confidence_orientation_thresholds must be non-empty.")
    for threshold in config.high_confidence_orientation_thresholds:
        if not 0.0 <= threshold <= 1.0:
            errors.append("orientation thresholds must be within [0, 1].")
    if config.disagreement_policy != "exclude_for_quality_subsets":
        errors.append("quality_policy.disagreement_policy must be exclude_for_quality_subsets.")
    if config.suspicious_band_policy != "exclude_and_report_sensitivity":
        errors.append(
            "quality_policy.suspicious_band_policy must be exclude_and_report_sensitivity."
        )
    if config.connected_object_policy != "candidate_only_filter":
        errors.append("quality_policy.connected_object_policy must be candidate_only_filter.")


def _validate_gate(config: LabelQualityGateConfig, errors: list[str]) -> None:
    if config.signal_enhancement_effect_size_delta < 0.0:
        errors.append("gate.signal_enhancement_effect_size_delta must be non-negative.")
    if config.strong_signal_effect_size_threshold < 0.0:
        errors.append("gate.strong_signal_effect_size_threshold must be non-negative.")
    if not 0.0 <= config.max_result_flip_fraction_from_review_exclusion <= 1.0:
        errors.append(
            "gate.max_result_flip_fraction_from_review_exclusion must be within [0, 1]."
        )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
