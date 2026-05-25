from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MVP4B_SIMPLE_BASELINE_CONFIG_VERSION = "mvp4b_simple_baseline_v001"
MVP4B_SIMPLE_BASELINE_REPORT_VERSION = "simple_baseline_v001"
MVP4B_SIMPLE_BASELINE_CSV_VERSION = "simple_baseline_predictions_v001"
MVP4B_BASELINE_REVIEW_VERSION = "simple_baseline_review_v001"
MVP4B_STAGE2_GATE_VERSION = "mvp4b_stage2_gate_v001"

MVP4B_INPUT_SAMPLE_TABLE = "baseline_sample_table_v001"
MVP4B_LABEL = "label_presence_plus"
MVP4B_LABEL_STATUS = "human_reviewed_candidate_v001"
MVP4B_PRIMARY_LABEL = "plus"
MVP4B_AUDIT_LABEL = "minus_ablation"
MVP4B_ALLOWED_SCOPE = "sanity_model_only"

SUPPORTED_MODEL_TYPES = frozenset({"logistic_regression", "linear_probe"})
SUPPORTED_FEATURE_SETS = frozenset({"transformed_features"})
REQUIRED_METRICS = (
    "weighted_accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "calibration_summary",
)


@dataclass(frozen=True)
class BaselineSchemaValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineConfig:
    config_version: str
    input_sample_table: str
    label: str
    label_status: str
    primary_label: str
    audit_label: str
    no_final_labels: bool
    model_types: tuple[str, ...]
    feature_set: tuple[str, ...]
    high_confidence_only: bool
    valid_for_azimuthal_validation: bool
    exclude_plus_minus_disagreement: bool
    exclude_large_depth_match_error: bool
    min_samples_per_class: int
    use_sample_weight: bool
    sample_weight_source: str
    split_method: str
    n_splits: int
    depth_block_size_ft: float | None
    min_gap_ft: float
    min_samples_per_class_per_fold: int
    metrics: tuple[str, ...]
    permutation_check: bool
    permutation_seed: int
    min_permutation_balanced_accuracy_margin: float
    suspicious_metric_threshold: float
    calibration_bins: int
    max_iterations: int
    learning_rate: float
    l2_penalty: float
    allowed_scope: str
    no_deep_learning: bool
    no_stc: bool
    no_apes: bool
    no_production_model: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_types"] = list(self.model_types)
        data["feature_set"] = list(self.feature_set)
        data["metrics"] = list(self.metrics)
        return data


def load_baseline_config(path: Path | str) -> BaselineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"MVP-4B simple baseline config must contain a YAML mapping: {path}")
    config = parse_baseline_config(raw)
    validation = validate_baseline_config(config)
    if validation.errors:
        raise ValueError(
            "Invalid MVP-4B simple baseline config: " + "; ".join(validation.errors)
        )
    return config


def parse_baseline_config(raw: dict[str, Any]) -> BaselineConfig:
    sample_filter = _as_dict(raw.get("sample_filter"))
    sample_weight = _as_dict(raw.get("sample_weight"))
    split = _as_dict(raw.get("split"))
    evaluation = _as_dict(raw.get("evaluation"))
    optimizer = _as_dict(raw.get("optimizer"))
    depth_block_size = split.get("depth_block_size_ft")
    return BaselineConfig(
        config_version=str(raw.get("config_version", "")),
        input_sample_table=str(raw.get("input_sample_table", "")),
        label=str(raw.get("label", "")),
        label_status=str(raw.get("label_status", "")),
        primary_label=str(raw.get("primary_label", "")),
        audit_label=str(raw.get("audit_label", "")),
        no_final_labels=bool(raw.get("no_final_labels", False)),
        model_types=tuple(str(value) for value in _as_list(raw.get("model_type"))),
        feature_set=tuple(str(value) for value in _as_list(raw.get("feature_set"))),
        high_confidence_only=bool(sample_filter.get("high_confidence_only", False)),
        valid_for_azimuthal_validation=bool(
            sample_filter.get("valid_for_azimuthal_validation", False)
        ),
        exclude_plus_minus_disagreement=bool(
            sample_filter.get("exclude_plus_minus_disagreement", False)
        ),
        exclude_large_depth_match_error=bool(
            sample_filter.get("exclude_large_depth_match_error", False)
        ),
        min_samples_per_class=int(sample_filter.get("min_samples_per_class", 0)),
        use_sample_weight=bool(sample_weight.get("use_sample_weight", False)),
        sample_weight_source=str(sample_weight.get("source", "")),
        split_method=str(split.get("method", "")),
        n_splits=int(split.get("n_splits", 0)),
        depth_block_size_ft=(
            None if depth_block_size is None else float(depth_block_size)
        ),
        min_gap_ft=float(split.get("min_gap_ft", 0.0)),
        min_samples_per_class_per_fold=int(
            split.get("min_samples_per_class_per_fold", 0)
        ),
        metrics=tuple(str(value) for value in _as_list(evaluation.get("metrics"))),
        permutation_check=bool(evaluation.get("permutation_check", False)),
        permutation_seed=int(evaluation.get("permutation_seed", 0)),
        min_permutation_balanced_accuracy_margin=float(
            evaluation.get("min_permutation_balanced_accuracy_margin", 0.0)
        ),
        suspicious_metric_threshold=float(evaluation.get("suspicious_metric_threshold", 1.0)),
        calibration_bins=int(evaluation.get("calibration_bins", 10)),
        max_iterations=int(optimizer.get("max_iterations", 0)),
        learning_rate=float(optimizer.get("learning_rate", 0.0)),
        l2_penalty=float(optimizer.get("l2_penalty", 0.0)),
        allowed_scope=str(raw.get("allowed_scope", "")),
        no_deep_learning=bool(raw.get("no_deep_learning", False)),
        no_stc=bool(raw.get("no_stc", False)),
        no_apes=bool(raw.get("no_apes", False)),
        no_production_model=bool(raw.get("no_production_model", False)),
    )


def validate_baseline_config(config: BaselineConfig) -> BaselineSchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if config.config_version != MVP4B_SIMPLE_BASELINE_CONFIG_VERSION:
        errors.append(
            f"config_version must be {MVP4B_SIMPLE_BASELINE_CONFIG_VERSION}, "
            f"observed {config.config_version}."
        )
    if config.input_sample_table != MVP4B_INPUT_SAMPLE_TABLE:
        errors.append(f"input_sample_table must be {MVP4B_INPUT_SAMPLE_TABLE}.")
    if config.label != MVP4B_LABEL:
        errors.append("label must be label_presence_plus.")
    if config.label_status != MVP4B_LABEL_STATUS:
        errors.append(f"label_status must be {MVP4B_LABEL_STATUS}.")
    if config.primary_label != MVP4B_PRIMARY_LABEL:
        errors.append("primary_label must remain plus.")
    if config.audit_label != MVP4B_AUDIT_LABEL:
        errors.append("audit_label must remain minus_ablation.")
    if not config.no_final_labels:
        errors.append("no_final_labels must be true.")
    if not config.model_types:
        errors.append("model_type must include at least one simple model.")
    unsupported_models = [
        model_type for model_type in config.model_types if model_type not in SUPPORTED_MODEL_TYPES
    ]
    if unsupported_models:
        errors.append("Unsupported model_type value(s): " + ", ".join(unsupported_models))
    unsupported_feature_sets = [
        feature_set
        for feature_set in config.feature_set
        if feature_set not in SUPPORTED_FEATURE_SETS
    ]
    if unsupported_feature_sets:
        errors.append("Unsupported feature_set value(s): " + ", ".join(unsupported_feature_sets))
    if "transformed_features" not in config.feature_set:
        errors.append("feature_set must include transformed_features.")
    if not config.high_confidence_only:
        errors.append("sample_filter.high_confidence_only must be true.")
    if not config.valid_for_azimuthal_validation:
        errors.append("sample_filter.valid_for_azimuthal_validation must be true.")
    if not config.exclude_large_depth_match_error:
        errors.append("sample_filter.exclude_large_depth_match_error must be true.")
    if config.exclude_plus_minus_disagreement:
        warnings.append(
            "plus/minus disagreement samples will be excluded instead of preserved for audit."
        )
    if config.min_samples_per_class <= 0:
        errors.append("sample_filter.min_samples_per_class must be positive.")
    if not config.use_sample_weight:
        errors.append("sample_weight.use_sample_weight must be true.")
    if config.sample_weight_source != "sample_weight":
        errors.append("sample_weight.source must be sample_weight.")
    if config.split_method != "depth_block_group_split":
        errors.append("split.method must be depth_block_group_split.")
    if config.n_splits not in {3, 5}:
        errors.append("split.n_splits must be 3 or 5.")
    if config.depth_block_size_ft is not None and config.depth_block_size_ft <= 0.0:
        errors.append("split.depth_block_size_ft must be positive when configured.")
    if config.min_gap_ft < 0.0:
        errors.append("split.min_gap_ft must be non-negative.")
    if config.min_samples_per_class_per_fold <= 0:
        errors.append("split.min_samples_per_class_per_fold must be positive.")
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in config.metrics]
    if missing_metrics:
        errors.append(
            "evaluation.metrics missing required metric(s): " + ", ".join(missing_metrics)
        )
    if not config.permutation_check:
        errors.append("evaluation.permutation_check must be true.")
    if config.min_permutation_balanced_accuracy_margin < 0.0:
        errors.append("min_permutation_balanced_accuracy_margin must be non-negative.")
    if not 0.5 <= config.suspicious_metric_threshold <= 1.0:
        errors.append("suspicious_metric_threshold must be within [0.5, 1].")
    if config.calibration_bins <= 1:
        errors.append("evaluation.calibration_bins must be greater than 1.")
    if config.max_iterations <= 0:
        errors.append("optimizer.max_iterations must be positive.")
    if config.learning_rate <= 0.0:
        errors.append("optimizer.learning_rate must be positive.")
    if config.l2_penalty < 0.0:
        errors.append("optimizer.l2_penalty must be non-negative.")
    if config.allowed_scope != MVP4B_ALLOWED_SCOPE:
        errors.append("allowed_scope must be sanity_model_only.")
    if not config.no_deep_learning:
        errors.append("no_deep_learning must be true.")
    if not config.no_stc:
        errors.append("no_stc must be true.")
    if not config.no_apes:
        errors.append("no_apes must be true.")
    if not config.no_production_model:
        errors.append("no_production_model must be true.")
    return BaselineSchemaValidation(valid=not errors, errors=errors, warnings=warnings)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]
