from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEPTH_LEVEL_BASELINE_CONFIG_VERSION = "depth_level_baseline_v001"
DEPTH_LEVEL_BASELINE_REPORT_VERSION = "depth_level_baseline_v001"
DEPTH_LEVEL_BASELINE_CSV_VERSION = "depth_level_baseline_predictions_v001"
DEPTH_LEVEL_BASELINE_REVIEW_VERSION = "depth_level_baseline_review_v001"
DEPTH_LEVEL_BASELINE_GATE_VERSION = "depth_level_baseline_gate_v001"

DEPTH_LEVEL_BASELINE_STAGE = "MVP-4B-R4b"
DEPTH_LEVEL_BASELINE_TASK = "depth_level_baseline_sanity_model"
DEPTH_LEVEL_BASELINE_ALLOWED_SCOPE = "depth_level_baseline_sanity_only"
DEPTH_LEVEL_BASELINE_INPUT_LABELS = "depth_level_labels_v001"
DEPTH_LEVEL_BASELINE_INPUT_FEATURES = "depth_level_xsi_features_v001"
DEPTH_LEVEL_BASELINE_PRIMARY_TASK = "depth_has_channel"
DEPTH_LEVEL_BASELINE_LABEL_STATUS = "weak_label_candidate"

SUPPORTED_DEPTH_LEVEL_MODEL_TYPES = frozenset({"logistic_regression", "linear_probe"})
SUPPORTED_DEPTH_LEVEL_FEATURE_SETS = frozenset({"depth_level_xsi_features"})
SUPPORTED_DEPTH_LEVEL_TARGET_VARIANTS = frozenset(
    {
        "all_positive_vs_negative",
        "strong_positive_vs_clear_negative",
        "high_confidence_positive_vs_clear_negative",
    }
)
REQUIRED_DEPTH_LEVEL_BASELINE_METRICS = (
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "permutation_margin",
)


@dataclass(frozen=True)
class DepthLevelBaselineSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelHighConfidenceFilter:
    min_label_confidence: float
    min_orientation_confidence: float
    max_plus_minus_disagreement_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelBaselineTargetFilters:
    high_confidence_positive: DepthLevelHighConfidenceFilter
    clear_negative: DepthLevelHighConfidenceFilter
    exclude_review_band: bool
    min_samples_per_class: int
    min_samples_per_class_per_fold: int
    warn_if_variant_too_small: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelBaselineSplitConfig:
    method: str
    n_splits: int
    depth_block_size_ft: float | None
    min_gap_ft: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelBaselineEvaluationConfig:
    metrics: tuple[str, ...]
    permutation_check: bool
    permutation_seed: int
    min_permutation_balanced_accuracy_margin: float
    degenerate_prediction_min_positive_rate: float
    degenerate_prediction_max_positive_rate: float
    stable_fold_min_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = list(self.metrics)
        return data


@dataclass(frozen=True)
class DepthLevelBaselineOptimizerConfig:
    max_iterations: int
    learning_rate: float
    l2_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthLevelBaselineConfig:
    schema_version: str
    config_version: str
    stage: str
    task: str
    input_labels: str
    input_features: str
    primary_task: str
    label_status: str
    model_types: tuple[str, ...]
    feature_set: tuple[str, ...]
    target_variants: tuple[str, ...]
    target_filters: DepthLevelBaselineTargetFilters
    split: DepthLevelBaselineSplitConfig
    evaluation: DepthLevelBaselineEvaluationConfig
    optimizer: DepthLevelBaselineOptimizerConfig
    allowed_scope: str
    no_model_training_claim: bool
    no_production_model: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_types"] = list(self.model_types)
        data["feature_set"] = list(self.feature_set)
        data["target_variants"] = list(self.target_variants)
        data["target_filters"] = self.target_filters.to_dict()
        data["split"] = self.split.to_dict()
        data["evaluation"] = self.evaluation.to_dict()
        data["optimizer"] = self.optimizer.to_dict()
        return data


def load_depth_level_baseline_config(path: Path | str) -> DepthLevelBaselineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Depth-level baseline config must be a YAML mapping: {path}")
    config = parse_depth_level_baseline_config(raw)
    validation = validate_depth_level_baseline_config(config)
    if validation.errors:
        raise ValueError(
            "Invalid depth-level baseline config: " + "; ".join(validation.errors)
        )
    return config


def parse_depth_level_baseline_config(raw: dict[str, Any]) -> DepthLevelBaselineConfig:
    filters = _as_dict(raw.get("target_filters"))
    high_positive = _as_dict(filters.get("high_confidence_positive"))
    clear_negative = _as_dict(filters.get("clear_negative"))
    split = _as_dict(raw.get("split"))
    evaluation = _as_dict(raw.get("evaluation"))
    optimizer = _as_dict(raw.get("optimizer"))
    depth_block_size = split.get("depth_block_size_ft")
    return DepthLevelBaselineConfig(
        schema_version=str(raw.get("schema_version", "")),
        config_version=str(raw.get("config_version", "")),
        stage=str(raw.get("stage", "")),
        task=str(raw.get("task", "")),
        input_labels=str(raw.get("input_labels", "")),
        input_features=str(raw.get("input_features", "")),
        primary_task=str(raw.get("primary_task", "")),
        label_status=str(raw.get("label_status", "")),
        model_types=tuple(str(value) for value in _as_list(raw.get("model_type"))),
        feature_set=tuple(str(value) for value in _as_list(raw.get("feature_set"))),
        target_variants=tuple(
            str(value) for value in _as_list(raw.get("target_variants"))
        ),
        target_filters=DepthLevelBaselineTargetFilters(
            high_confidence_positive=DepthLevelHighConfidenceFilter(
                min_label_confidence=float(high_positive.get("min_label_confidence", -1.0)),
                min_orientation_confidence=float(
                    high_positive.get("min_orientation_confidence", -1.0)
                ),
                max_plus_minus_disagreement_fraction=float(
                    high_positive.get("max_plus_minus_disagreement_fraction", -1.0)
                ),
            ),
            clear_negative=DepthLevelHighConfidenceFilter(
                min_label_confidence=float(clear_negative.get("min_label_confidence", -1.0)),
                min_orientation_confidence=float(
                    clear_negative.get("min_orientation_confidence", -1.0)
                ),
                max_plus_minus_disagreement_fraction=float(
                    clear_negative.get("max_plus_minus_disagreement_fraction", -1.0)
                ),
            ),
            exclude_review_band=bool(filters.get("exclude_review_band", False)),
            min_samples_per_class=int(filters.get("min_samples_per_class", 0)),
            min_samples_per_class_per_fold=int(
                filters.get("min_samples_per_class_per_fold", 0)
            ),
            warn_if_variant_too_small=bool(filters.get("warn_if_variant_too_small", False)),
        ),
        split=DepthLevelBaselineSplitConfig(
            method=str(split.get("method", "")),
            n_splits=int(split.get("n_splits", 0)),
            depth_block_size_ft=(
                None if depth_block_size is None else float(depth_block_size)
            ),
            min_gap_ft=float(split.get("min_gap_ft", 0.0)),
        ),
        evaluation=DepthLevelBaselineEvaluationConfig(
            metrics=tuple(str(value) for value in _as_list(evaluation.get("metrics"))),
            permutation_check=bool(evaluation.get("permutation_check", False)),
            permutation_seed=int(evaluation.get("permutation_seed", 0)),
            min_permutation_balanced_accuracy_margin=float(
                evaluation.get("min_permutation_balanced_accuracy_margin", -1.0)
            ),
            degenerate_prediction_min_positive_rate=float(
                evaluation.get("degenerate_prediction_min_positive_rate", -1.0)
            ),
            degenerate_prediction_max_positive_rate=float(
                evaluation.get("degenerate_prediction_max_positive_rate", -1.0)
            ),
            stable_fold_min_count=int(evaluation.get("stable_fold_min_count", 0)),
        ),
        optimizer=DepthLevelBaselineOptimizerConfig(
            max_iterations=int(optimizer.get("max_iterations", 0)),
            learning_rate=float(optimizer.get("learning_rate", 0.0)),
            l2_penalty=float(optimizer.get("l2_penalty", -1.0)),
        ),
        allowed_scope=str(raw.get("allowed_scope", "")),
        no_model_training_claim=bool(raw.get("no_model_training_claim", False)),
        no_production_model=bool(raw.get("no_production_model", False)),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
        no_deep_learning=bool(raw.get("no_deep_learning", False)),
        no_mvp4c=bool(raw.get("no_mvp4c", False)),
    )


def validate_depth_level_baseline_config(
    config: DepthLevelBaselineConfig,
) -> DepthLevelBaselineSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.schema_version != "schema_v001":
        errors.append("schema_version must be schema_v001.")
    if config.config_version != DEPTH_LEVEL_BASELINE_CONFIG_VERSION:
        errors.append(
            "config_version must be "
            f"{DEPTH_LEVEL_BASELINE_CONFIG_VERSION}, observed {config.config_version}."
        )
    if config.stage != DEPTH_LEVEL_BASELINE_STAGE:
        errors.append(f"stage must be {DEPTH_LEVEL_BASELINE_STAGE}.")
    if config.task != DEPTH_LEVEL_BASELINE_TASK:
        errors.append(f"task must be {DEPTH_LEVEL_BASELINE_TASK}.")
    if config.input_labels != DEPTH_LEVEL_BASELINE_INPUT_LABELS:
        errors.append(f"input_labels must be {DEPTH_LEVEL_BASELINE_INPUT_LABELS}.")
    if config.input_features != DEPTH_LEVEL_BASELINE_INPUT_FEATURES:
        errors.append(f"input_features must be {DEPTH_LEVEL_BASELINE_INPUT_FEATURES}.")
    if config.primary_task != DEPTH_LEVEL_BASELINE_PRIMARY_TASK:
        errors.append(f"primary_task must be {DEPTH_LEVEL_BASELINE_PRIMARY_TASK}.")
    if config.label_status != DEPTH_LEVEL_BASELINE_LABEL_STATUS:
        errors.append(f"label_status must be {DEPTH_LEVEL_BASELINE_LABEL_STATUS}.")
    _validate_models_and_targets(config, errors)
    _validate_filters(config.target_filters, errors)
    _validate_split(config.split, errors)
    _validate_evaluation(config.evaluation, errors)
    _validate_optimizer(config.optimizer, errors)
    if config.allowed_scope != DEPTH_LEVEL_BASELINE_ALLOWED_SCOPE:
        errors.append(f"allowed_scope must be {DEPTH_LEVEL_BASELINE_ALLOWED_SCOPE}.")
    for field_name in (
        "no_model_training_claim",
        "no_production_model",
        "no_final_labels",
        "no_stc",
        "no_apes",
        "no_deep_learning",
        "no_mvp4c",
    ):
        if not bool(getattr(config, field_name)):
            errors.append(f"{field_name} must be true.")
    if config.target_filters.warn_if_variant_too_small:
        warnings.append("target variants below minimum sample count will be skipped with warnings.")
    return DepthLevelBaselineSchemaValidation(
        valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def _validate_models_and_targets(
    config: DepthLevelBaselineConfig,
    errors: list[str],
) -> None:
    if not config.model_types:
        errors.append("model_type must include at least one simple model.")
    unsupported_models = [
        value for value in config.model_types if value not in SUPPORTED_DEPTH_LEVEL_MODEL_TYPES
    ]
    if unsupported_models:
        errors.append("Unsupported model_type value(s): " + ", ".join(unsupported_models))
    if not config.feature_set:
        errors.append("feature_set must include depth_level_xsi_features.")
    unsupported_features = [
        value for value in config.feature_set if value not in SUPPORTED_DEPTH_LEVEL_FEATURE_SETS
    ]
    if unsupported_features:
        errors.append("Unsupported feature_set value(s): " + ", ".join(unsupported_features))
    if "depth_level_xsi_features" not in config.feature_set:
        errors.append("feature_set must include depth_level_xsi_features.")
    missing_variants = [
        value
        for value in SUPPORTED_DEPTH_LEVEL_TARGET_VARIANTS
        if value not in config.target_variants
    ]
    if missing_variants:
        errors.append("target_variants missing required value(s): " + ", ".join(missing_variants))
    unsupported_variants = [
        value
        for value in config.target_variants
        if value not in SUPPORTED_DEPTH_LEVEL_TARGET_VARIANTS
    ]
    if unsupported_variants:
        errors.append("Unsupported target_variants value(s): " + ", ".join(unsupported_variants))


def _validate_filters(
    config: DepthLevelBaselineTargetFilters,
    errors: list[str],
) -> None:
    _validate_filter(config.high_confidence_positive, "high_confidence_positive", errors)
    _validate_filter(config.clear_negative, "clear_negative", errors)
    if not config.exclude_review_band:
        errors.append("target_filters.exclude_review_band must be true.")
    if config.min_samples_per_class <= 0:
        errors.append("target_filters.min_samples_per_class must be positive.")
    if config.min_samples_per_class_per_fold <= 0:
        errors.append("target_filters.min_samples_per_class_per_fold must be positive.")
    if not config.warn_if_variant_too_small:
        errors.append("target_filters.warn_if_variant_too_small must be true.")


def _validate_filter(
    config: DepthLevelHighConfidenceFilter,
    prefix: str,
    errors: list[str],
) -> None:
    for field_name, value in (
        ("min_label_confidence", config.min_label_confidence),
        ("min_orientation_confidence", config.min_orientation_confidence),
        (
            "max_plus_minus_disagreement_fraction",
            config.max_plus_minus_disagreement_fraction,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            errors.append(f"target_filters.{prefix}.{field_name} must be within [0, 1].")


def _validate_split(config: DepthLevelBaselineSplitConfig, errors: list[str]) -> None:
    if config.method != "depth_block_split":
        errors.append("split.method must be depth_block_split.")
    if config.n_splits not in {3, 5}:
        errors.append("split.n_splits must be 3 or 5.")
    if config.depth_block_size_ft is not None and config.depth_block_size_ft <= 0.0:
        errors.append("split.depth_block_size_ft must be positive when configured.")
    if config.min_gap_ft < 0.0:
        errors.append("split.min_gap_ft must be non-negative.")


def _validate_evaluation(
    config: DepthLevelBaselineEvaluationConfig,
    errors: list[str],
) -> None:
    missing = [
        metric for metric in REQUIRED_DEPTH_LEVEL_BASELINE_METRICS if metric not in config.metrics
    ]
    if missing:
        errors.append("evaluation.metrics missing required metric(s): " + ", ".join(missing))
    if not config.permutation_check:
        errors.append("evaluation.permutation_check must be true.")
    if config.min_permutation_balanced_accuracy_margin < 0.0:
        errors.append("evaluation.min_permutation_balanced_accuracy_margin must be non-negative.")
    if not 0.0 <= config.degenerate_prediction_min_positive_rate <= 0.5:
        errors.append("degenerate_prediction_min_positive_rate must be within [0, 0.5].")
    if not 0.5 <= config.degenerate_prediction_max_positive_rate <= 1.0:
        errors.append("degenerate_prediction_max_positive_rate must be within [0.5, 1].")
    if (
        config.degenerate_prediction_min_positive_rate
        >= config.degenerate_prediction_max_positive_rate
    ):
        errors.append("degenerate prediction min rate must be lower than max rate.")
    if config.stable_fold_min_count <= 0:
        errors.append("evaluation.stable_fold_min_count must be positive.")


def _validate_optimizer(
    config: DepthLevelBaselineOptimizerConfig,
    errors: list[str],
) -> None:
    if config.max_iterations <= 0:
        errors.append("optimizer.max_iterations must be positive.")
    if config.learning_rate <= 0.0:
        errors.append("optimizer.learning_rate must be positive.")
    if config.l2_penalty < 0.0:
        errors.append("optimizer.l2_penalty must be non-negative.")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
