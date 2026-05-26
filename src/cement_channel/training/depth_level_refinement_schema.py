from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION = "depth_level_refinement_v001"
DEPTH_LEVEL_REFINEMENT_REPORT_VERSION = "depth_level_refinement_v001"
DEPTH_LEVEL_REFINEMENT_CSV_VERSION = "depth_level_refinement_predictions_v001"
DEPTH_LEVEL_REFINEMENT_REVIEW_VERSION = "depth_level_refinement_review_v001"
DEPTH_LEVEL_REFINEMENT_GATE_VERSION = "depth_level_refinement_gate_v001"

DEPTH_LEVEL_REFINEMENT_STAGE = "MVP-4B-R4c"
DEPTH_LEVEL_REFINEMENT_TASK = "controlled_depth_level_feature_refinement"
DEPTH_LEVEL_REFINEMENT_ALLOWED_SCOPE = "controlled_depth_level_refinement_only"
DEPTH_LEVEL_REFINEMENT_INPUT_LABELS = "depth_level_labels_v001"
DEPTH_LEVEL_REFINEMENT_INPUT_FEATURES = "depth_level_xsi_features_v001"
DEPTH_LEVEL_REFINEMENT_INPUT_BASELINE_REPORT = "depth_level_baseline_report_v001"
DEPTH_LEVEL_REFINEMENT_TARGET_VARIANT = "high_confidence_positive_vs_clear_negative"
DEPTH_LEVEL_REFINEMENT_LABEL_STATUS = "weak_label_candidate"

SUPPORTED_REFINEMENT_MODELS = frozenset({"logistic_regression", "linear_probe"})
SUPPORTED_REFINEMENT_FEATURE_GROUPS = frozenset(
    {
        "all_depth_features",
        "late_over_early_features",
        "energy_window_features",
        "side_contrast_features",
        "receiver_summary_features",
        "robust_top_features_from_baseline",
    }
)
REQUIRED_REFINEMENT_FEATURE_GROUPS = tuple(sorted(SUPPORTED_REFINEMENT_FEATURE_GROUPS))


@dataclass(frozen=True)
class DepthLevelRefinementSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelRefinementTargetFilters:
    clear_negative_min_label_confidence: float
    clear_negative_min_orientation_confidence: float
    max_plus_minus_disagreement_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelRefinementReviewInterval:
    name: str
    depth_min_ft: float
    depth_max_ft: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelRefinementRobustnessChecks:
    exclude_5700_band: tuple[bool, ...]
    confidence_thresholds: tuple[float, ...]
    depth_block_splits: tuple[int, ...]
    permutation_repeats: int
    feature_group_ablation: bool
    fold_stability_required: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["exclude_5700_band"] = list(self.exclude_5700_band)
        data["confidence_thresholds"] = list(self.confidence_thresholds)
        data["depth_block_splits"] = list(self.depth_block_splits)
        return data


@dataclass(frozen=True)
class DepthLevelRefinementSplitConfig:
    method: str
    depth_block_size_ft: float | None
    min_gap_ft: float
    min_samples_per_class_per_fold: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelRefinementOptimizerConfig:
    max_iterations: int
    learning_rate: float
    l2_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelRefinementGateThresholds:
    min_margin_mean: float
    min_margin_permutation: float
    max_predicted_positive_rate: float
    min_predicted_positive_rate: float
    min_folds_above_permutation_fraction: float
    suspicious_high_balanced_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelRefinementConfig:
    schema_version: str
    config_version: str
    stage: str
    task: str
    input_labels: str
    input_features: str
    input_baseline_report: str
    target_variant: str
    label_status: str
    allowed_models: tuple[str, ...]
    feature_groups: tuple[str, ...]
    target_filters: DepthLevelRefinementTargetFilters
    review_intervals: tuple[DepthLevelRefinementReviewInterval, ...]
    robustness_checks: DepthLevelRefinementRobustnessChecks
    split: DepthLevelRefinementSplitConfig
    optimizer: DepthLevelRefinementOptimizerConfig
    gate_thresholds: DepthLevelRefinementGateThresholds
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
        data["allowed_models"] = list(self.allowed_models)
        data["feature_groups"] = list(self.feature_groups)
        data["target_filters"] = self.target_filters.to_dict()
        data["review_intervals"] = [
            interval.to_dict() for interval in self.review_intervals
        ]
        data["robustness_checks"] = self.robustness_checks.to_dict()
        data["split"] = self.split.to_dict()
        data["optimizer"] = self.optimizer.to_dict()
        data["gate_thresholds"] = self.gate_thresholds.to_dict()
        return data


def load_depth_level_refinement_config(path: Path | str) -> DepthLevelRefinementConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Depth-level refinement config must be a YAML mapping: {path}")
    config = parse_depth_level_refinement_config(raw)
    validation = validate_depth_level_refinement_config(config)
    if validation.errors:
        raise ValueError(
            "Invalid depth-level refinement config: " + "; ".join(validation.errors)
        )
    return config


def parse_depth_level_refinement_config(raw: dict[str, Any]) -> DepthLevelRefinementConfig:
    target_filters = _as_dict(raw.get("target_filters"))
    robustness = _as_dict(raw.get("robustness_checks"))
    split = _as_dict(raw.get("split"))
    optimizer = _as_dict(raw.get("optimizer"))
    gate = _as_dict(raw.get("gate_thresholds"))
    depth_block_size = split.get("depth_block_size_ft")
    return DepthLevelRefinementConfig(
        schema_version=str(raw.get("schema_version", "")),
        config_version=str(raw.get("config_version", "")),
        stage=str(raw.get("stage", "")),
        task=str(raw.get("task", "")),
        input_labels=str(raw.get("input_labels", "")),
        input_features=str(raw.get("input_features", "")),
        input_baseline_report=str(raw.get("input_baseline_report", "")),
        target_variant=str(raw.get("target_variant", "")),
        label_status=str(raw.get("label_status", "")),
        allowed_models=tuple(str(value) for value in _as_list(raw.get("allowed_models"))),
        feature_groups=tuple(str(value) for value in _as_list(raw.get("feature_groups"))),
        target_filters=DepthLevelRefinementTargetFilters(
            clear_negative_min_label_confidence=float(
                target_filters.get("clear_negative_min_label_confidence", -1.0)
            ),
            clear_negative_min_orientation_confidence=float(
                target_filters.get("clear_negative_min_orientation_confidence", -1.0)
            ),
            max_plus_minus_disagreement_fraction=float(
                target_filters.get("max_plus_minus_disagreement_fraction", -1.0)
            ),
        ),
        review_intervals=tuple(
            _parse_review_interval(value)
            for value in _as_list(raw.get("review_intervals"))
        ),
        robustness_checks=DepthLevelRefinementRobustnessChecks(
            exclude_5700_band=tuple(
                bool(value) for value in _as_list(robustness.get("exclude_5700_band"))
            ),
            confidence_thresholds=tuple(
                float(value)
                for value in _as_list(robustness.get("confidence_thresholds"))
            ),
            depth_block_splits=tuple(
                int(value) for value in _as_list(robustness.get("depth_block_splits"))
            ),
            permutation_repeats=int(robustness.get("permutation_repeats", 0)),
            feature_group_ablation=bool(robustness.get("feature_group_ablation", False)),
            fold_stability_required=bool(robustness.get("fold_stability_required", False)),
        ),
        split=DepthLevelRefinementSplitConfig(
            method=str(split.get("method", "")),
            depth_block_size_ft=(
                None if depth_block_size is None else float(depth_block_size)
            ),
            min_gap_ft=float(split.get("min_gap_ft", 0.0)),
            min_samples_per_class_per_fold=int(
                split.get("min_samples_per_class_per_fold", 0)
            ),
        ),
        optimizer=DepthLevelRefinementOptimizerConfig(
            max_iterations=int(optimizer.get("max_iterations", 0)),
            learning_rate=float(optimizer.get("learning_rate", 0.0)),
            l2_penalty=float(optimizer.get("l2_penalty", -1.0)),
        ),
        gate_thresholds=DepthLevelRefinementGateThresholds(
            min_margin_mean=float(gate.get("min_margin_mean", -1.0)),
            min_margin_permutation=float(gate.get("min_margin_permutation", -1.0)),
            max_predicted_positive_rate=float(gate.get("max_predicted_positive_rate", -1.0)),
            min_predicted_positive_rate=float(gate.get("min_predicted_positive_rate", -1.0)),
            min_folds_above_permutation_fraction=float(
                gate.get("min_folds_above_permutation_fraction", -1.0)
            ),
            suspicious_high_balanced_accuracy=float(
                gate.get("suspicious_high_balanced_accuracy", 1.1)
            ),
        ),
        allowed_scope=str(raw.get("allowed_scope", "")),
        no_model_training_claim=bool(raw.get("no_model_training_claim", False)),
        no_production_model=bool(raw.get("no_production_model", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_mvp4c=bool(raw.get("no_mvp4c", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
        no_deep_learning=bool(raw.get("no_deep_learning", False)),
    )


def validate_depth_level_refinement_config(
    config: DepthLevelRefinementConfig,
) -> DepthLevelRefinementSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.schema_version != "schema_v001":
        errors.append("schema_version must be schema_v001.")
    if config.config_version != DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION:
        errors.append(
            "config_version must be "
            f"{DEPTH_LEVEL_REFINEMENT_CONFIG_VERSION}, observed {config.config_version}."
        )
    if config.stage != DEPTH_LEVEL_REFINEMENT_STAGE:
        errors.append(f"stage must be {DEPTH_LEVEL_REFINEMENT_STAGE}.")
    if config.task != DEPTH_LEVEL_REFINEMENT_TASK:
        errors.append(f"task must be {DEPTH_LEVEL_REFINEMENT_TASK}.")
    if config.input_labels != DEPTH_LEVEL_REFINEMENT_INPUT_LABELS:
        errors.append(f"input_labels must be {DEPTH_LEVEL_REFINEMENT_INPUT_LABELS}.")
    if config.input_features != DEPTH_LEVEL_REFINEMENT_INPUT_FEATURES:
        errors.append(f"input_features must be {DEPTH_LEVEL_REFINEMENT_INPUT_FEATURES}.")
    if config.input_baseline_report != DEPTH_LEVEL_REFINEMENT_INPUT_BASELINE_REPORT:
        errors.append(
            f"input_baseline_report must be {DEPTH_LEVEL_REFINEMENT_INPUT_BASELINE_REPORT}."
        )
    if config.target_variant != DEPTH_LEVEL_REFINEMENT_TARGET_VARIANT:
        errors.append(
            f"target_variant must be {DEPTH_LEVEL_REFINEMENT_TARGET_VARIANT}."
        )
    if config.label_status != DEPTH_LEVEL_REFINEMENT_LABEL_STATUS:
        errors.append(f"label_status must be {DEPTH_LEVEL_REFINEMENT_LABEL_STATUS}.")
    _validate_models_and_groups(config, errors)
    _validate_target_filters(config.target_filters, errors)
    _validate_review_intervals(config.review_intervals, errors, warnings)
    _validate_robustness(config.robustness_checks, errors)
    _validate_split(config.split, errors)
    _validate_optimizer(config.optimizer, errors)
    _validate_gate_thresholds(config.gate_thresholds, errors)
    if config.allowed_scope != DEPTH_LEVEL_REFINEMENT_ALLOWED_SCOPE:
        errors.append(f"allowed_scope must be {DEPTH_LEVEL_REFINEMENT_ALLOWED_SCOPE}.")
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
    return DepthLevelRefinementSchemaValidation(
        valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def _parse_review_interval(value: Any) -> DepthLevelRefinementReviewInterval:
    data = _as_dict(value)
    return DepthLevelRefinementReviewInterval(
        name=str(data.get("name", "")),
        depth_min_ft=float(data.get("depth_min_ft", 0.0)),
        depth_max_ft=float(data.get("depth_max_ft", 0.0)),
        reason=str(data.get("reason", "")),
    )


def _validate_models_and_groups(
    config: DepthLevelRefinementConfig,
    errors: list[str],
) -> None:
    if not config.allowed_models:
        errors.append("allowed_models must include at least one simple model.")
    unsupported_models = [
        value for value in config.allowed_models if value not in SUPPORTED_REFINEMENT_MODELS
    ]
    if unsupported_models:
        errors.append("Unsupported allowed_models value(s): " + ", ".join(unsupported_models))
    missing_groups = [
        value for value in REQUIRED_REFINEMENT_FEATURE_GROUPS if value not in config.feature_groups
    ]
    if missing_groups:
        errors.append("feature_groups missing required value(s): " + ", ".join(missing_groups))
    unsupported_groups = [
        value
        for value in config.feature_groups
        if value not in SUPPORTED_REFINEMENT_FEATURE_GROUPS
    ]
    if unsupported_groups:
        errors.append("Unsupported feature_groups value(s): " + ", ".join(unsupported_groups))


def _validate_target_filters(
    config: DepthLevelRefinementTargetFilters,
    errors: list[str],
) -> None:
    for field_name, value in (
        ("clear_negative_min_label_confidence", config.clear_negative_min_label_confidence),
        (
            "clear_negative_min_orientation_confidence",
            config.clear_negative_min_orientation_confidence,
        ),
        (
            "max_plus_minus_disagreement_fraction",
            config.max_plus_minus_disagreement_fraction,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            errors.append(f"target_filters.{field_name} must be within [0, 1].")


def _validate_review_intervals(
    intervals: tuple[DepthLevelRefinementReviewInterval, ...],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not intervals:
        errors.append("review_intervals must include the 5700 ft review band.")
        return
    found_5700 = False
    for interval in intervals:
        if not interval.name:
            errors.append("review interval name must be non-empty.")
        if interval.depth_max_ft <= interval.depth_min_ft:
            errors.append(f"review interval {interval.name} must have max depth > min depth.")
        if interval.depth_min_ft <= 5700.0 <= interval.depth_max_ft:
            found_5700 = True
    if not found_5700:
        warnings.append("review_intervals do not cover 5700 ft.")


def _validate_robustness(
    config: DepthLevelRefinementRobustnessChecks,
    errors: list[str],
) -> None:
    if set(config.exclude_5700_band) != {False, True}:
        errors.append("robustness_checks.exclude_5700_band must include false and true.")
    if tuple(sorted(config.confidence_thresholds)) != config.confidence_thresholds:
        errors.append("robustness_checks.confidence_thresholds must be sorted.")
    if not all(0.0 < value < 1.0 for value in config.confidence_thresholds):
        errors.append("robustness_checks.confidence_thresholds must be within (0, 1).")
    if set(config.depth_block_splits) != {3, 5}:
        errors.append("robustness_checks.depth_block_splits must include 3 and 5.")
    if config.permutation_repeats < 1:
        errors.append("robustness_checks.permutation_repeats must be positive.")
    if not config.feature_group_ablation:
        errors.append("robustness_checks.feature_group_ablation must be true.")
    if not config.fold_stability_required:
        errors.append("robustness_checks.fold_stability_required must be true.")


def _validate_split(config: DepthLevelRefinementSplitConfig, errors: list[str]) -> None:
    if config.method != "depth_block_split":
        errors.append("split.method must be depth_block_split.")
    if config.depth_block_size_ft is not None and config.depth_block_size_ft <= 0.0:
        errors.append("split.depth_block_size_ft must be positive when configured.")
    if config.min_gap_ft < 0.0:
        errors.append("split.min_gap_ft must be non-negative.")
    if config.min_samples_per_class_per_fold <= 0:
        errors.append("split.min_samples_per_class_per_fold must be positive.")


def _validate_optimizer(
    config: DepthLevelRefinementOptimizerConfig,
    errors: list[str],
) -> None:
    if config.max_iterations <= 0:
        errors.append("optimizer.max_iterations must be positive.")
    if config.learning_rate <= 0.0:
        errors.append("optimizer.learning_rate must be positive.")
    if config.l2_penalty < 0.0:
        errors.append("optimizer.l2_penalty must be non-negative.")


def _validate_gate_thresholds(
    config: DepthLevelRefinementGateThresholds,
    errors: list[str],
) -> None:
    if config.min_margin_mean < 0.0:
        errors.append("gate_thresholds.min_margin_mean must be non-negative.")
    if config.min_margin_permutation < 0.0:
        errors.append("gate_thresholds.min_margin_permutation must be non-negative.")
    if not 0.0 <= config.min_predicted_positive_rate <= 0.5:
        errors.append("gate_thresholds.min_predicted_positive_rate must be within [0, 0.5].")
    if not 0.5 <= config.max_predicted_positive_rate <= 1.0:
        errors.append("gate_thresholds.max_predicted_positive_rate must be within [0.5, 1].")
    if config.min_predicted_positive_rate >= config.max_predicted_positive_rate:
        errors.append("gate predicted-positive min must be less than max.")
    if not 0.0 <= config.min_folds_above_permutation_fraction <= 1.0:
        errors.append(
            "gate_thresholds.min_folds_above_permutation_fraction must be within [0, 1]."
        )
    if not 0.5 <= config.suspicious_high_balanced_accuracy <= 1.0:
        errors.append(
            "gate_thresholds.suspicious_high_balanced_accuracy must be within [0.5, 1]."
        )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
