from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEPTH_LEVEL_MANUAL_REVIEW_CONFIG_VERSION = "depth_level_manual_review_v001"
DEPTH_LEVEL_MANUAL_REVIEW_PACK_VERSION = "depth_level_manual_review_v001"
DEPTH_LEVEL_MANUAL_REVIEW_STAGE = "MVP-4B-R4c+"
DEPTH_LEVEL_MANUAL_REVIEW_TASK = "depth_level_manual_review_pack"
DEPTH_LEVEL_MANUAL_REVIEW_ALLOWED_SCOPE = "depth_level_manual_review_pack_only"
DEPTH_LEVEL_MANUAL_REVIEW_INPUT_LABELS = "depth_level_labels_v001"
DEPTH_LEVEL_MANUAL_REVIEW_INPUT_FEATURES = "depth_level_xsi_features_v001"
DEPTH_LEVEL_MANUAL_REVIEW_INPUT_REFINEMENT_REPORT = "depth_level_refinement_report_v001"
DEPTH_LEVEL_MANUAL_REVIEW_INPUT_GATE_REPORT = "depth_level_refinement_gate_report"
DEPTH_LEVEL_MANUAL_REVIEW_TARGET_VARIANT = (
    "high_confidence_positive_vs_clear_negative"
)
DEPTH_LEVEL_MANUAL_REVIEW_LABEL_STATUS = "weak_label_candidate"

SUPPORTED_REVIEW_INTERVAL_TYPES = frozenset(
    {
        "true_positive_like",
        "clear_negative_like",
        "false_positive_like",
        "false_negative_like",
        "high_uncertainty",
        "5700_band_review",
        "boundary_case",
    }
)
SUPPORTED_REVIEW_SORT_KEYS = frozenset(
    {
        "score_desc",
        "score_asc",
        "uncertainty_desc",
        "boundary_score_asc",
    }
)
REQUIRED_REVIEW_SELECTIONS = (
    "select_top_positive_intervals",
    "select_clear_negative_intervals",
    "select_high_score_positive_intervals",
    "select_high_score_negative_or_disagreement_intervals",
    "select_low_score_positive_intervals",
)


@dataclass(frozen=True)
class DepthLevelReviewSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelReviewSelectionDefaults:
    max_interval_gap_ft: float
    min_interval_depth_span_ft: float
    score_high_threshold: float
    score_low_threshold: float
    boundary_score_band: float
    high_disagreement_threshold: float
    low_confidence_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelReviewSelectionConfig:
    enabled: bool
    count: int
    interval_type: str
    sort_by: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelReviewIntervalConfig:
    name: str
    depth_min_ft: float
    depth_max_ft: float
    interval_type: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelManualReviewConfig:
    schema_version: str
    config_version: str
    stage: str
    task: str
    input_labels: str
    input_features: str
    input_refinement_report: str
    input_refinement_gate_report: str
    target_variant: str
    label_status: str
    selection_defaults: DepthLevelReviewSelectionDefaults
    selections: dict[str, DepthLevelReviewSelectionConfig]
    review_intervals: tuple[DepthLevelReviewIntervalConfig, ...]
    include_5700_band_sensitivity: bool
    include_confidence_summary: bool
    include_xsi_feature_summary: bool
    include_cast_label_summary: bool
    allowed_scope: str
    no_model_training_claim: bool
    no_production_model: bool
    no_final_labels: bool
    no_mvp4c: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["selection_defaults"] = self.selection_defaults.to_dict()
        data["selections"] = {
            key: value.to_dict() for key, value in self.selections.items()
        }
        data["review_intervals"] = [
            interval.to_dict() for interval in self.review_intervals
        ]
        return data


def load_depth_level_manual_review_config(
    path: Path | str,
) -> DepthLevelManualReviewConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Depth-level manual review config must be a YAML mapping: {path}"
        )
    config = parse_depth_level_manual_review_config(raw)
    validation = validate_depth_level_manual_review_config(config)
    if validation.errors:
        raise ValueError(
            "Invalid depth-level manual review config: "
            + "; ".join(validation.errors)
        )
    return config


def parse_depth_level_manual_review_config(
    raw: dict[str, Any],
) -> DepthLevelManualReviewConfig:
    defaults = _as_dict(raw.get("selection_defaults"))
    selections = {
        key: _parse_selection_config(raw.get(key))
        for key in _selection_keys(raw)
    }
    return DepthLevelManualReviewConfig(
        schema_version=str(raw.get("schema_version", "")),
        config_version=str(raw.get("config_version", "")),
        stage=str(raw.get("stage", "")),
        task=str(raw.get("task", "")),
        input_labels=str(raw.get("input_labels", "")),
        input_features=str(raw.get("input_features", "")),
        input_refinement_report=str(raw.get("input_refinement_report", "")),
        input_refinement_gate_report=str(raw.get("input_refinement_gate_report", "")),
        target_variant=str(raw.get("target_variant", "")),
        label_status=str(raw.get("label_status", "")),
        selection_defaults=DepthLevelReviewSelectionDefaults(
            max_interval_gap_ft=float(defaults.get("max_interval_gap_ft", -1.0)),
            min_interval_depth_span_ft=float(
                defaults.get("min_interval_depth_span_ft", -1.0)
            ),
            score_high_threshold=float(defaults.get("score_high_threshold", -1.0)),
            score_low_threshold=float(defaults.get("score_low_threshold", -1.0)),
            boundary_score_band=float(defaults.get("boundary_score_band", -1.0)),
            high_disagreement_threshold=float(
                defaults.get("high_disagreement_threshold", -1.0)
            ),
            low_confidence_threshold=float(
                defaults.get("low_confidence_threshold", -1.0)
            ),
        ),
        selections=selections,
        review_intervals=tuple(
            _parse_review_interval(value)
            for value in _as_list(raw.get("review_intervals"))
        ),
        include_5700_band_sensitivity=bool(
            raw.get("include_5700_band_sensitivity", False)
        ),
        include_confidence_summary=bool(raw.get("include_confidence_summary", False)),
        include_xsi_feature_summary=bool(raw.get("include_xsi_feature_summary", False)),
        include_cast_label_summary=bool(raw.get("include_cast_label_summary", False)),
        allowed_scope=str(raw.get("allowed_scope", "")),
        no_model_training_claim=bool(raw.get("no_model_training_claim", False)),
        no_production_model=bool(raw.get("no_production_model", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_mvp4c=bool(raw.get("no_mvp4c", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
        no_deep_learning=bool(raw.get("no_deep_learning", False)),
    )


def validate_depth_level_manual_review_config(
    config: DepthLevelManualReviewConfig,
) -> DepthLevelReviewSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.schema_version != "schema_v001":
        errors.append("schema_version must be schema_v001.")
    if config.config_version != DEPTH_LEVEL_MANUAL_REVIEW_CONFIG_VERSION:
        errors.append(
            "config_version must be "
            f"{DEPTH_LEVEL_MANUAL_REVIEW_CONFIG_VERSION}, observed {config.config_version}."
        )
    if config.stage != DEPTH_LEVEL_MANUAL_REVIEW_STAGE:
        errors.append(f"stage must be {DEPTH_LEVEL_MANUAL_REVIEW_STAGE}.")
    if config.task != DEPTH_LEVEL_MANUAL_REVIEW_TASK:
        errors.append(f"task must be {DEPTH_LEVEL_MANUAL_REVIEW_TASK}.")
    if config.input_labels != DEPTH_LEVEL_MANUAL_REVIEW_INPUT_LABELS:
        errors.append(f"input_labels must be {DEPTH_LEVEL_MANUAL_REVIEW_INPUT_LABELS}.")
    if config.input_features != DEPTH_LEVEL_MANUAL_REVIEW_INPUT_FEATURES:
        errors.append(
            f"input_features must be {DEPTH_LEVEL_MANUAL_REVIEW_INPUT_FEATURES}."
        )
    if config.input_refinement_report != DEPTH_LEVEL_MANUAL_REVIEW_INPUT_REFINEMENT_REPORT:
        errors.append(
            "input_refinement_report must be "
            f"{DEPTH_LEVEL_MANUAL_REVIEW_INPUT_REFINEMENT_REPORT}."
        )
    if (
        config.input_refinement_gate_report
        != DEPTH_LEVEL_MANUAL_REVIEW_INPUT_GATE_REPORT
    ):
        errors.append(
            "input_refinement_gate_report must be "
            f"{DEPTH_LEVEL_MANUAL_REVIEW_INPUT_GATE_REPORT}."
        )
    if config.target_variant != DEPTH_LEVEL_MANUAL_REVIEW_TARGET_VARIANT:
        errors.append(
            f"target_variant must be {DEPTH_LEVEL_MANUAL_REVIEW_TARGET_VARIANT}."
        )
    if config.label_status != DEPTH_LEVEL_MANUAL_REVIEW_LABEL_STATUS:
        errors.append(f"label_status must be {DEPTH_LEVEL_MANUAL_REVIEW_LABEL_STATUS}.")
    _validate_defaults(config.selection_defaults, errors)
    _validate_selections(config.selections, errors)
    _validate_review_intervals(config.review_intervals, errors, warnings)
    if not config.include_5700_band_sensitivity:
        errors.append("include_5700_band_sensitivity must be true.")
    if not config.include_confidence_summary:
        errors.append("include_confidence_summary must be true.")
    if not config.include_xsi_feature_summary:
        errors.append("include_xsi_feature_summary must be true.")
    if not config.include_cast_label_summary:
        errors.append("include_cast_label_summary must be true.")
    if config.allowed_scope != DEPTH_LEVEL_MANUAL_REVIEW_ALLOWED_SCOPE:
        errors.append(f"allowed_scope must be {DEPTH_LEVEL_MANUAL_REVIEW_ALLOWED_SCOPE}.")
    for field_name in (
        "no_model_training_claim",
        "no_production_model",
        "no_final_labels",
        "no_mvp4c",
        "no_stc",
        "no_apes",
        "no_deep_learning",
    ):
        if not bool(getattr(config, field_name)):
            errors.append(f"{field_name} must be true.")
    return DepthLevelReviewSchemaValidation(
        valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def _selection_keys(raw: dict[str, Any]) -> list[str]:
    return sorted(key for key in raw if key.startswith("select_"))


def _parse_selection_config(value: Any) -> DepthLevelReviewSelectionConfig:
    data = _as_dict(value)
    return DepthLevelReviewSelectionConfig(
        enabled=bool(data.get("enabled", False)),
        count=int(data.get("count", 0)),
        interval_type=str(data.get("interval_type", "")),
        sort_by=str(data.get("sort_by", "")),
    )


def _parse_review_interval(value: Any) -> DepthLevelReviewIntervalConfig:
    data = _as_dict(value)
    return DepthLevelReviewIntervalConfig(
        name=str(data.get("name", "")),
        depth_min_ft=float(data.get("depth_min_ft", 0.0)),
        depth_max_ft=float(data.get("depth_max_ft", 0.0)),
        interval_type=str(data.get("interval_type", "")),
        reason=str(data.get("reason", "")),
    )


def _validate_defaults(
    defaults: DepthLevelReviewSelectionDefaults,
    errors: list[str],
) -> None:
    if defaults.max_interval_gap_ft < 0.0:
        errors.append("selection_defaults.max_interval_gap_ft must be non-negative.")
    if defaults.min_interval_depth_span_ft < 0.0:
        errors.append("selection_defaults.min_interval_depth_span_ft must be non-negative.")
    if not 0.5 < defaults.score_high_threshold <= 1.0:
        errors.append("selection_defaults.score_high_threshold must be within (0.5, 1].")
    if not 0.0 <= defaults.score_low_threshold < 0.5:
        errors.append("selection_defaults.score_low_threshold must be within [0, 0.5).")
    if defaults.score_low_threshold >= defaults.score_high_threshold:
        errors.append("selection_defaults score_low_threshold must be below score_high_threshold.")
    if not 0.0 < defaults.boundary_score_band <= 0.5:
        errors.append("selection_defaults.boundary_score_band must be within (0, 0.5].")
    if not 0.0 <= defaults.high_disagreement_threshold <= 1.0:
        errors.append(
            "selection_defaults.high_disagreement_threshold must be within [0, 1]."
        )
    if not 0.0 <= defaults.low_confidence_threshold <= 1.0:
        errors.append("selection_defaults.low_confidence_threshold must be within [0, 1].")


def _validate_selections(
    selections: dict[str, DepthLevelReviewSelectionConfig],
    errors: list[str],
) -> None:
    missing = [key for key in REQUIRED_REVIEW_SELECTIONS if key not in selections]
    if missing:
        errors.append("missing required review selection(s): " + ", ".join(missing))
    if not selections:
        errors.append("at least one select_* interval rule must be configured.")
        return
    for key, selection in selections.items():
        if selection.enabled and selection.count <= 0:
            errors.append(f"{key}.count must be positive when enabled.")
        if selection.interval_type not in SUPPORTED_REVIEW_INTERVAL_TYPES:
            errors.append(f"{key}.interval_type is unsupported: {selection.interval_type}.")
        if selection.sort_by not in SUPPORTED_REVIEW_SORT_KEYS:
            errors.append(f"{key}.sort_by is unsupported: {selection.sort_by}.")


def _validate_review_intervals(
    intervals: tuple[DepthLevelReviewIntervalConfig, ...],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not intervals:
        errors.append("review_intervals must include the 5700 ft review interval.")
        return
    found_5700 = False
    for interval in intervals:
        if not interval.name:
            errors.append("review interval name must be non-empty.")
        if interval.depth_max_ft <= interval.depth_min_ft:
            errors.append(f"review interval {interval.name} must have max depth > min depth.")
        if interval.interval_type not in SUPPORTED_REVIEW_INTERVAL_TYPES:
            errors.append(
                f"review interval {interval.name} has unsupported interval_type "
                f"{interval.interval_type}."
            )
        if interval.depth_min_ft <= 5700.0 <= interval.depth_max_ft:
            found_5700 = True
    if not found_5700:
        warnings.append("review_intervals do not cover 5700 ft.")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
