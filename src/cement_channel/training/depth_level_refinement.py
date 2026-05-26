from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.depth_level_baseline import (
    binary_metrics_with_positive_rate,
)
from cement_channel.training.depth_level_baseline_schema import (
    DepthLevelBaselineConfig,
    DepthLevelBaselineEvaluationConfig,
    DepthLevelBaselineOptimizerConfig,
)
from cement_channel.training.depth_level_refinement_schema import (
    DEPTH_LEVEL_REFINEMENT_CSV_VERSION,
    DEPTH_LEVEL_REFINEMENT_REPORT_VERSION,
    DepthLevelRefinementConfig,
    load_depth_level_refinement_config,
)
from cement_channel.training.depth_splits import make_depth_block_splits
from cement_channel.training.simple_baseline import (
    fit_linear_probe,
    fit_logistic_regression,
    predict_scores,
)


@dataclass(frozen=True)
class DepthLevelRefinementReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    model_backend: str
    allowed_scope: str
    target_variant: str
    label_status: str
    scenario_count: int
    runnable_scenario_count: int
    passing_scenario_count: int
    best_result: dict[str, Any] | None
    best_feature_group: str | None
    best_configuration: dict[str, Any] | None
    scenario_summaries: list[dict[str, Any]]
    feature_group_summary: dict[str, dict[str, Any]]
    confidence_threshold_summary: dict[str, dict[str, Any]]
    split_summary: dict[str, dict[str, Any]]
    exclude_5700_summary: dict[str, dict[str, Any]]
    robustness_summary: dict[str, Any]
    top_features: dict[str, list[dict[str, float | str]]]
    output_csv_version: str
    recommendation: str
    manual_confirmation_required: bool
    manual_confirmation_items: list[str]
    production_training: bool
    no_model_training_claim: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_depth_level_refinement_from_config(
    *,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    baseline_report_json: Path | str,
    refinement_config_path: Path | str,
    output_report_md: Path | str | None = None,
    output_report_json: Path | str | None = None,
    output_csv: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[DepthLevelRefinementReport, list[dict[str, Any]]]:
    label_arrays = _load_npz(depth_level_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    baseline_report = _read_json(Path(baseline_report_json))
    report, rows = run_depth_level_refinement(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        baseline_report=baseline_report,
        config=load_depth_level_refinement_config(refinement_config_path),
        inputs={
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "baseline_report_json": str(baseline_report_json),
            "refinement_config_path": str(refinement_config_path),
        },
    )
    if output_report_md is not None and output_report_json is not None and output_csv is not None:
        write_depth_level_refinement_outputs(
            report,
            rows,
            output_report_md=Path(output_report_md),
            output_report_json=Path(output_report_json),
            output_csv=Path(output_csv),
            overwrite=overwrite,
        )
    return report, rows


def run_depth_level_refinement(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    baseline_report: dict[str, Any],
    config: DepthLevelRefinementConfig,
    inputs: dict[str, str] | None = None,
) -> tuple[DepthLevelRefinementReport, list[dict[str, Any]]]:
    warnings: list[str] = []
    errors: list[str] = []
    prepared = prepare_depth_level_refinement_inputs(label_arrays, feature_arrays)
    warnings.extend(prepared["warnings"])
    errors.extend(prepared["errors"])
    _validate_runtime_guardrails(label_arrays, feature_arrays, baseline_report, errors, warnings)
    feature_groups = build_feature_group_indices(prepared["feature_names"], baseline_report, config)
    scenario_summaries: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_summary_input: dict[str, list[np.ndarray]] = {}
    scenario_index = 0
    for feature_group in config.feature_groups:
        feature_indices = feature_groups.get(feature_group, np.asarray([], dtype=np.int32))
        for exclude_5700 in config.robustness_checks.exclude_5700_band:
            for confidence_threshold in config.robustness_checks.confidence_thresholds:
                target_data = build_refinement_target(
                    prepared,
                    config,
                    confidence_threshold=confidence_threshold,
                    exclude_5700_band=exclude_5700,
                )
                for n_splits in config.robustness_checks.depth_block_splits:
                    for model_type in config.allowed_models:
                        scenario_id = _scenario_id(
                            feature_group=feature_group,
                            exclude_5700_band=exclude_5700,
                            confidence_threshold=confidence_threshold,
                            n_splits=n_splits,
                            model_type=model_type,
                        )
                        result = run_refinement_scenario(
                            scenario_id=scenario_id,
                            scenario_index=scenario_index,
                            feature_group=feature_group,
                            feature_indices=feature_indices,
                            target_data=target_data,
                            n_splits=n_splits,
                            model_type=model_type,
                            config=config,
                        )
                        warnings.extend(result["warnings"])
                        scenario_summaries.append(result["summary"])
                        prediction_rows.extend(result["prediction_rows"])
                        if result["coefficients"]:
                            coefficient_summary_input[scenario_id] = result["coefficients"]
                        scenario_index += 1
    passing = [row for row in scenario_summaries if bool(row.get("passes_gate_thresholds"))]
    best = _best_scenario(passing or scenario_summaries)
    robustness = summarize_refinement_robustness(scenario_summaries, config)
    manual_items = manual_confirmation_items(robustness)
    if robustness["suspicious_leakage"]:
        errors.append(
            "Suspiciously high depth-level refinement performance; leakage review required."
        )
    if not passing:
        errors.append(
            "No refinement scenario beat permutation with required non-degenerate stability."
        )
    recommendation = _recommendation(errors, robustness, passing)
    top_features = summarize_refinement_coefficients(
        coefficient_summary_input,
        prepared["feature_names"],
        limit=20,
    )
    report = DepthLevelRefinementReport(
        report_version=DEPTH_LEVEL_REFINEMENT_REPORT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        model_backend="numpy_fallback",
        allowed_scope=config.allowed_scope,
        target_variant=config.target_variant,
        label_status=config.label_status,
        scenario_count=len(scenario_summaries),
        runnable_scenario_count=int(
            sum(str(row.get("status")) == "runnable" for row in scenario_summaries)
        ),
        passing_scenario_count=len(passing),
        best_result=best,
        best_feature_group=None if best is None else str(best.get("feature_group")),
        best_configuration=None if best is None else _best_configuration(best),
        scenario_summaries=scenario_summaries,
        feature_group_summary=_group_summary(scenario_summaries, "feature_group"),
        confidence_threshold_summary=_group_summary(scenario_summaries, "confidence_threshold"),
        split_summary=_group_summary(scenario_summaries, "n_splits"),
        exclude_5700_summary=_group_summary(scenario_summaries, "exclude_5700_band"),
        robustness_summary=robustness,
        top_features=top_features,
        output_csv_version=DEPTH_LEVEL_REFINEMENT_CSV_VERSION,
        recommendation=recommendation,
        manual_confirmation_required=bool(manual_items),
        manual_confirmation_items=manual_items,
        production_training=False,
        no_model_training_claim=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "formal model performance claim",
            "production model training",
            "model weight export",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
            "unconstrained hyperparameter search",
        ],
    )
    return report, prediction_rows


def prepare_depth_level_refinement_inputs(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    required_labels = (
        "depth",
        "depth_has_channel_any",
        "depth_clear_negative_mask",
        "depth_review_band_mask",
        "depth_label_confidence",
        "depth_orientation_confidence",
        "depth_plus_minus_disagreement_fraction",
    )
    required_features = ("depth", "depth_level_xsi_features", "depth_level_xsi_feature_names")
    missing_labels = [key for key in required_labels if key not in label_arrays]
    missing_features = [key for key in required_features if key not in feature_arrays]
    if missing_labels:
        raise KeyError("depth-level label NPZ missing field(s): " + ", ".join(missing_labels))
    if missing_features:
        raise KeyError(
            "depth-level feature NPZ missing field(s): " + ", ".join(missing_features)
        )
    warnings: list[str] = []
    errors: list[str] = []
    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    if features.ndim != 2:
        raise ValueError("depth_level_xsi_features must have shape [depth, feature].")
    if depth.size != features.shape[0]:
        raise ValueError("label depth count must match feature depth count.")
    if feature_depth.size == depth.size and not np.allclose(depth, feature_depth, atol=1e-3):
        warnings.append("label and feature depth arrays differ; using label depth order.")
    if feature_names.size != features.shape[1]:
        raise ValueError("depth_level_xsi_feature_names length must match feature count.")
    finite_rows = np.all(np.isfinite(features), axis=1)
    if not np.all(finite_rows):
        warnings.append("non-finite feature rows are excluded from refinement scenarios.")
    return {
        "depth": depth,
        "features": np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "feature_names": feature_names.astype(str).tolist(),
        "finite_rows": finite_rows,
        "has_channel": np.asarray(label_arrays["depth_has_channel_any"], dtype=bool).reshape(-1),
        "clear_negative": np.asarray(
            label_arrays["depth_clear_negative_mask"], dtype=bool
        ).reshape(-1),
        "review_band": np.asarray(label_arrays["depth_review_band_mask"], dtype=bool).reshape(-1),
        "label_confidence": np.asarray(
            label_arrays["depth_label_confidence"], dtype=np.float32
        ).reshape(-1),
        "orientation_confidence": np.asarray(
            label_arrays["depth_orientation_confidence"], dtype=np.float32
        ).reshape(-1),
        "disagreement_fraction": np.asarray(
            label_arrays["depth_plus_minus_disagreement_fraction"], dtype=np.float32
        ).reshape(-1),
        "warnings": warnings,
        "errors": errors,
    }


def build_feature_group_indices(
    feature_names: list[str],
    baseline_report: dict[str, Any],
    config: DepthLevelRefinementConfig,
) -> dict[str, np.ndarray]:
    names = np.asarray(feature_names).astype(str)
    groups: dict[str, np.ndarray] = {
        "all_depth_features": np.arange(names.size, dtype=np.int32),
        "late_over_early_features": _indices_matching(names, ("late_over_early",)),
        "energy_window_features": _indices_matching(names, ("energy",)),
        "side_contrast_features": _indices_matching(
            names, ("side_contrast", "max_side_anomaly")
        ),
        "receiver_summary_features": _indices_matching(names, ("receiver_", "near_far_ratio")),
        "robust_top_features_from_baseline": _robust_top_feature_indices(
            names, baseline_report
        ),
    }
    return {key: groups[key] for key in config.feature_groups}


def build_refinement_target(
    prepared: dict[str, Any],
    config: DepthLevelRefinementConfig,
    *,
    confidence_threshold: float,
    exclude_5700_band: bool,
) -> dict[str, Any]:
    valid = np.asarray(prepared["finite_rows"], dtype=bool).copy()
    if exclude_5700_band:
        valid &= ~np.asarray(prepared["review_band"], dtype=bool)
        valid &= ~_interval_mask(prepared["depth"], config)
    quality = (
        (prepared["label_confidence"] >= confidence_threshold)
        & (prepared["orientation_confidence"] >= confidence_threshold)
        & (
            prepared["disagreement_fraction"]
            <= config.target_filters.max_plus_minus_disagreement_fraction
        )
    )
    positive = prepared["has_channel"] & quality & valid
    negative = prepared["clear_negative"] & quality & valid
    selected = positive | negative
    label = np.where(positive[selected], 1, 0).astype(np.int8)
    sample_weight = _class_balanced_weights(label, prepared["label_confidence"][selected])
    return {
        "features": prepared["features"][selected],
        "depth": prepared["depth"][selected],
        "label": label,
        "sample_weight": sample_weight,
        "selected_global_index": np.flatnonzero(selected).astype(np.int32),
        "positive_count": int(np.count_nonzero(label == 1)),
        "negative_count": int(np.count_nonzero(label == 0)),
        "sample_count": int(label.size),
        "positive_fraction": None if label.size == 0 else float(np.mean(label == 1)),
        "effective_positive_weight_fraction": _effective_positive_weight_fraction(
            label, sample_weight
        ),
    }


def run_refinement_scenario(
    *,
    scenario_id: str,
    scenario_index: int,
    feature_group: str,
    feature_indices: np.ndarray,
    target_data: dict[str, Any],
    n_splits: int,
    model_type: str,
    config: DepthLevelRefinementConfig,
) -> dict[str, Any]:
    warnings: list[str] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficients: list[np.ndarray] = []
    label = np.asarray(target_data["label"], dtype=np.int8)
    summary_base = {
        "scenario_id": scenario_id,
        "scenario_index": scenario_index,
        "feature_group": feature_group,
        "feature_count": int(feature_indices.size),
        "exclude_5700_band": bool("exclude5700_true" in scenario_id),
        "confidence_threshold": _scenario_confidence_threshold(scenario_id),
        "n_splits": n_splits,
        "model_type": model_type,
        "sample_count": int(label.size),
        "positive_count": int(target_data["positive_count"]),
        "negative_count": int(target_data["negative_count"]),
        "positive_fraction": target_data["positive_fraction"],
        "effective_positive_weight_fraction": target_data["effective_positive_weight_fraction"],
    }
    if feature_indices.size == 0:
        return {
            "summary": {
                **summary_base,
                "status": "skipped_empty_feature_group",
                "passes_gate_thresholds": False,
                "fold_metrics": [],
            },
            "prediction_rows": [],
            "coefficients": [],
            "warnings": [f"{scenario_id}: feature group has no matching features."],
        }
    if label.size == 0 or np.unique(label).size < 2:
        return {
            "summary": {
                **summary_base,
                "status": "skipped_single_class_target",
                "passes_gate_thresholds": False,
                "fold_metrics": [],
            },
            "prediction_rows": [],
            "coefficients": [],
            "warnings": [f"{scenario_id}: target subset is empty or single-class."],
        }
    try:
        split_plan = make_depth_block_splits(
            depth=target_data["depth"],
            labels=label,
            n_splits=n_splits,
            min_gap_ft=config.split.min_gap_ft,
            block_size_ft=config.split.depth_block_size_ft,
            min_samples_per_class=config.split.min_samples_per_class_per_fold,
        )
    except ValueError as exc:
        return {
            "summary": {
                **summary_base,
                "status": "skipped_split_error",
                "split_error": str(exc),
                "passes_gate_thresholds": False,
                "fold_metrics": [],
            },
            "prediction_rows": [],
            "coefficients": [],
            "warnings": [f"{scenario_id}: {exc}"],
        }
    if split_plan.warnings:
        warnings.extend(f"{scenario_id}: {message}" for message in split_plan.warnings)
    fold_metrics: list[dict[str, Any]] = []
    metric_config = _metric_config(config)
    for fold in split_plan.folds:
        fold_result = _run_refinement_fold(
            scenario_id=scenario_id,
            feature_group=feature_group,
            model_type=model_type,
            target_data=target_data,
            feature_indices=feature_indices,
            fold_index=fold.fold_index,
            train_mask=fold.train_mask,
            validation_mask=fold.validation_mask,
            config=config,
            metric_config=metric_config,
        )
        warnings.extend(fold_result["warnings"])
        fold_metrics.extend(fold_result["fold_metrics"])
        prediction_rows.extend(fold_result["prediction_rows"])
        if fold_result["coefficient"] is not None:
            coefficients.append(fold_result["coefficient"])
    summary = summarize_scenario(
        summary_base,
        fold_metrics,
        config=config,
        split_summaries=split_plan.summaries(),
    )
    return {
        "summary": summary,
        "prediction_rows": prediction_rows,
        "coefficients": coefficients,
        "warnings": warnings,
    }


def summarize_scenario(
    summary_base: dict[str, Any],
    fold_metrics: list[dict[str, Any]],
    *,
    config: DepthLevelRefinementConfig,
    split_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    real_rows = [row for row in fold_metrics if not row["permutation"]]
    permutation_rows = [row for row in fold_metrics if row["permutation"]]
    real_ba = _metric_values(real_rows, "balanced_accuracy")
    perm_ba = _metric_values(permutation_rows, "balanced_accuracy")
    margins = _metric_values(real_rows, "margin")
    positive_rates = _metric_values(real_rows, "predicted_positive_rate")
    f1_values = _metric_values(real_rows, "f1")
    precision_values = _metric_values(real_rows, "precision")
    recall_values = _metric_values(real_rows, "recall")
    valid_folds = len(real_rows)
    folds_above = int(
        sum(
            bool(row.get("passes_permutation_margin"))
            and not bool(row.get("degenerate_prediction"))
            for row in real_rows
        )
    )
    fold_fraction = 0.0 if valid_folds == 0 else folds_above / valid_folds
    margin_mean = _nanmean(margins)
    predicted_positive_rate = _nanmean(positive_rates)
    degenerate = (
        predicted_positive_rate is None
        or predicted_positive_rate <= config.gate_thresholds.min_predicted_positive_rate
        or predicted_positive_rate >= config.gate_thresholds.max_predicted_positive_rate
    )
    passes = bool(
        valid_folds > 0
        and margin_mean is not None
        and margin_mean >= config.gate_thresholds.min_margin_mean
        and fold_fraction >= config.gate_thresholds.min_folds_above_permutation_fraction
        and not degenerate
    )
    status = "runnable" if valid_folds > 0 else "skipped_no_valid_folds"
    return {
        **summary_base,
        "status": status,
        "valid_fold_count": valid_folds,
        "fold_count_requested": int(summary_base["n_splits"]),
        "balanced_accuracy_mean": _nanmean(real_ba),
        "balanced_accuracy_std": _nanstd(real_ba),
        "permutation_balanced_accuracy_mean": _nanmean(perm_ba),
        "permutation_balanced_accuracy_std": _nanstd(perm_ba),
        "margin_mean": margin_mean,
        "margin_std": _nanstd(margins),
        "precision_mean": _nanmean(precision_values),
        "recall_mean": _nanmean(recall_values),
        "f1_mean": _nanmean(f1_values),
        "predicted_positive_rate": predicted_positive_rate,
        "degenerate_prediction": degenerate,
        "folds_above_permutation": folds_above,
        "folds_above_permutation_fraction": fold_fraction,
        "passes_gate_thresholds": passes,
        "split_summaries": split_summaries,
        "fold_metrics": real_rows,
    }


def summarize_refinement_robustness(
    summaries: list[dict[str, Any]],
    config: DepthLevelRefinementConfig,
) -> dict[str, Any]:
    passing = [row for row in summaries if bool(row.get("passes_gate_thresholds"))]
    passing_groups = sorted({str(row["feature_group"]) for row in passing})
    passing_thresholds = sorted({float(row["confidence_threshold"]) for row in passing})
    passing_splits = sorted({int(row["n_splits"]) for row in passing})
    passing_exclusions = sorted({bool(row["exclude_5700_band"]) for row in passing})
    include_pass = any(not bool(row["exclude_5700_band"]) for row in passing)
    exclude_pass = any(bool(row["exclude_5700_band"]) for row in passing)
    best = _best_scenario(summaries)
    best_ba = None if best is None else _as_float(best.get("balanced_accuracy_mean"))
    suspicious = (
        best_ba is not None
        and best_ba > config.gate_thresholds.suspicious_high_balanced_accuracy
    )
    return {
        "passing_scenario_count": len(passing),
        "passing_feature_groups": passing_groups,
        "passing_confidence_thresholds": passing_thresholds,
        "passing_depth_block_splits": passing_splits,
        "passing_exclude_5700_values": passing_exclusions,
        "depends_on_single_feature_group": len(passing_groups) <= 1,
        "depends_on_single_confidence_threshold": len(passing_thresholds) <= 1,
        "depends_on_single_split": len(passing_splits) <= 1,
        "depends_on_5700_band": include_pass and not exclude_pass,
        "exclude_5700_still_passes": exclude_pass,
        "stable_over_permutation": len(passing) > 0,
        "suspicious_leakage": suspicious,
        "suspicious_leakage_reason": (
            None
            if not suspicious
            else "best balanced_accuracy exceeds suspicious_high_balanced_accuracy"
        ),
    }


def manual_confirmation_items(robustness: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if robustness.get("depends_on_single_feature_group"):
        items.append("Confirm whether the passing feature group is physically acceptable.")
    if robustness.get("depends_on_single_confidence_threshold"):
        items.append("Confirm the confidence threshold before further refinement.")
    if robustness.get("depends_on_single_split"):
        items.append("Confirm depth-block split robustness or choose a split policy.")
    if robustness.get("depends_on_5700_band"):
        items.append("Confirm whether to keep or exclude the ~5700 ft review band.")
    if robustness.get("suspicious_leakage"):
        items.append("Review potential leakage before any further feature work.")
    return items


def write_depth_level_refinement_outputs(
    report: DepthLevelRefinementReport,
    prediction_rows: list[dict[str, Any]],
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    _ensure_can_write(output_csv, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_depth_level_refinement_markdown(report), encoding="utf-8")
    write_depth_level_refinement_csv(prediction_rows, output_csv)


def format_depth_level_refinement_markdown(report: DepthLevelRefinementReport) -> str:
    best = report.best_result or {}
    robustness = report.robustness_summary
    lines = [
        "# MVP-4B-R4c Depth-Level Refinement Report",
        "",
        "This is a controlled robustness sanity check against CAST weak-label "
        "candidates. It is not formal model performance, production training, "
        "or final-label generation.",
        "",
        f"- recommendation: `{report.recommendation}`",
        f"- best_feature_group: `{report.best_feature_group}`",
        f"- best_confidence_threshold: {best.get('confidence_threshold')}",
        f"- best_margin_mean: {best.get('margin_mean')}",
        "- best_permutation_balanced_accuracy_mean: "
        f"{best.get('permutation_balanced_accuracy_mean')}",
        f"- best_predicted_positive_rate: {best.get('predicted_positive_rate')}",
        f"- passing_scenario_count: {report.passing_scenario_count}",
        f"- depends_on_5700_band: `{robustness.get('depends_on_5700_band')}`",
        f"- stable_over_permutation: `{robustness.get('stable_over_permutation')}`",
        f"- manual_confirmation_required: `{report.manual_confirmation_required}`",
        "",
        "## Manual Confirmation Items",
        "",
    ]
    lines.extend(_message_lines(report.manual_confirmation_items))
    lines.extend(["", "## Robustness Summary", ""])
    for key, value in robustness.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Best Configuration", ""])
    lines.append(f"- {report.best_configuration if report.best_configuration else 'none'}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def write_depth_level_refinement_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "csv_version",
        "scenario_id",
        "feature_group",
        "exclude_5700_band",
        "confidence_threshold",
        "n_splits",
        "model_type",
        "fold_index",
        "depth",
        "label",
        "sample_weight",
        "score",
        "prediction",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _run_refinement_fold(
    *,
    scenario_id: str,
    feature_group: str,
    model_type: str,
    target_data: dict[str, Any],
    feature_indices: np.ndarray,
    fold_index: int,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: DepthLevelRefinementConfig,
    metric_config: DepthLevelBaselineConfig,
) -> dict[str, Any]:
    warnings: list[str] = []
    x_train_raw = target_data["features"][train_mask][:, feature_indices]
    x_val_raw = target_data["features"][validation_mask][:, feature_indices]
    y_train = target_data["label"][train_mask]
    y_val = target_data["label"][validation_mask]
    w_train = target_data["sample_weight"][train_mask]
    w_val = target_data["sample_weight"][validation_mask]
    if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
        warnings.append(f"{scenario_id}/fold_{fold_index}: split is single-class.")
        return {
            "warnings": warnings,
            "fold_metrics": [],
            "prediction_rows": [],
            "coefficient": None,
        }
    x_train, x_val = _standardize_by_train(x_train_raw, x_val_raw, w_train)
    coef = _fit_model(model_type, x_train, y_train, w_train, config)
    score = predict_scores(x_val, coef, model_type=model_type)
    real_metrics = binary_metrics_with_positive_rate(
        y_val,
        score,
        w_val,
        config=metric_config,
    )
    permutation_metrics = []
    rng = np.random.default_rng(_stable_seed(scenario_id, fold_index))
    for _ in range(config.robustness_checks.permutation_repeats):
        permuted = np.asarray(y_train, dtype=np.int8).copy()
        rng.shuffle(permuted)
        perm_coef = _fit_model(model_type, x_train, permuted, w_train, config)
        perm_score = predict_scores(x_val, perm_coef, model_type=model_type)
        permutation_metrics.append(
            binary_metrics_with_positive_rate(y_val, perm_score, w_val, config=metric_config)
        )
    perm_summary = _aggregate_metric_dicts(permutation_metrics)
    real_ba = _as_float(real_metrics.get("balanced_accuracy"))
    perm_ba = _as_float(perm_summary.get("balanced_accuracy"))
    margin = None if real_ba is None or perm_ba is None else real_ba - perm_ba
    degenerate = bool(real_metrics.get("degenerate_prediction"))
    passes = bool(
        margin is not None
        and margin >= config.gate_thresholds.min_margin_permutation
        and not degenerate
    )
    real_row = {
        "scenario_id": scenario_id,
        "feature_group": feature_group,
        "model_type": model_type,
        "fold_index": fold_index,
        "permutation": False,
        "metrics": real_metrics,
        "permutation_metrics": perm_summary,
        "margin": margin,
        "passes_permutation_margin": passes,
        "degenerate_prediction": degenerate,
    }
    permutation_row = {
        "scenario_id": scenario_id,
        "feature_group": feature_group,
        "model_type": model_type,
        "fold_index": fold_index,
        "permutation": True,
        "metrics": perm_summary,
    }
    return {
        "warnings": warnings,
        "fold_metrics": [real_row, permutation_row],
        "prediction_rows": _prediction_rows(
            scenario_id=scenario_id,
            feature_group=feature_group,
            model_type=model_type,
            fold_index=fold_index,
            depth=target_data["depth"][validation_mask],
            label=y_val,
            sample_weight=w_val,
            score=score,
        ),
        "coefficient": coef,
    }


def _metric_config(config: DepthLevelRefinementConfig) -> DepthLevelBaselineConfig:
    return DepthLevelBaselineConfig(
        schema_version="schema_v001",
        config_version="depth_level_baseline_v001",
        stage="MVP-4B-R4b",
        task="depth_level_baseline_sanity_model",
        input_labels="depth_level_labels_v001",
        input_features="depth_level_xsi_features_v001",
        primary_task="depth_has_channel",
        label_status="weak_label_candidate",
        model_types=config.allowed_models,
        feature_set=("depth_level_xsi_features",),
        target_variants=(config.target_variant,),
        target_filters=None,  # type: ignore[arg-type]
        split=None,  # type: ignore[arg-type]
        evaluation=DepthLevelBaselineEvaluationConfig(
            metrics=("balanced_accuracy", "precision", "recall", "f1", "permutation_margin"),
            permutation_check=True,
            permutation_seed=202405,
            min_permutation_balanced_accuracy_margin=config.gate_thresholds.min_margin_permutation,
            degenerate_prediction_min_positive_rate=(
                config.gate_thresholds.min_predicted_positive_rate
            ),
            degenerate_prediction_max_positive_rate=(
                config.gate_thresholds.max_predicted_positive_rate
            ),
            stable_fold_min_count=1,
        ),
        optimizer=DepthLevelBaselineOptimizerConfig(
            max_iterations=config.optimizer.max_iterations,
            learning_rate=config.optimizer.learning_rate,
            l2_penalty=config.optimizer.l2_penalty,
        ),
        allowed_scope="depth_level_baseline_sanity_only",
        no_model_training_claim=True,
        no_production_model=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
    )


def _fit_model(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    config: DepthLevelRefinementConfig,
) -> np.ndarray:
    if model_type == "logistic_regression":
        return fit_logistic_regression(
            x_train,
            y_train,
            sample_weight,
            max_iterations=config.optimizer.max_iterations,
            learning_rate=config.optimizer.learning_rate,
            l2_penalty=config.optimizer.l2_penalty,
        )
    if model_type == "linear_probe":
        return fit_linear_probe(
            x_train,
            y_train,
            sample_weight,
            l2_penalty=config.optimizer.l2_penalty,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _prediction_rows(
    *,
    scenario_id: str,
    feature_group: str,
    model_type: str,
    fold_index: int,
    depth: np.ndarray,
    label: np.ndarray,
    sample_weight: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, Any]]:
    parsed = _parse_scenario_id(scenario_id)
    return [
        {
            "csv_version": DEPTH_LEVEL_REFINEMENT_CSV_VERSION,
            "scenario_id": scenario_id,
            "feature_group": feature_group,
            "exclude_5700_band": parsed["exclude_5700_band"],
            "confidence_threshold": parsed["confidence_threshold"],
            "n_splits": parsed["n_splits"],
            "model_type": model_type,
            "fold_index": fold_index,
            "depth": float(depth[index]),
            "label": int(label[index]),
            "sample_weight": float(sample_weight[index]),
            "score": float(score[index]),
            "prediction": int(score[index] >= 0.5),
        }
        for index in range(label.size)
    ]


def summarize_refinement_coefficients(
    coefficients: dict[str, list[np.ndarray]],
    all_feature_names: list[str],
    *,
    limit: int,
) -> dict[str, list[dict[str, float | str]]]:
    result: dict[str, list[dict[str, float | str]]] = {}
    feature_names = np.asarray(all_feature_names).astype(str)
    for scenario_id, values in coefficients.items():
        if not values:
            continue
        feature_group = _parse_scenario_id(scenario_id)["feature_group"]
        indices = _feature_group_indices_for_summary(feature_names, feature_group)
        if len(indices) + 1 != values[0].shape[0]:
            names = [f"feature_{index}" for index in range(values[0].shape[0] - 1)]
        else:
            names = feature_names[indices].tolist()
        matrix = np.vstack(values)
        mean_coef = np.mean(matrix[:, 1:], axis=0)
        rows = [
            {
                "feature_name": name,
                "mean_coefficient": float(value),
                "mean_abs_coefficient": float(abs(value)),
            }
            for name, value in zip(names, mean_coef, strict=True)
        ]
        rows.sort(key=lambda row: float(row["mean_abs_coefficient"]), reverse=True)
        result[scenario_id] = rows[:limit]
    return result


def _feature_group_indices_for_summary(feature_names: np.ndarray, feature_group: str) -> np.ndarray:
    if feature_group == "all_depth_features":
        return np.arange(feature_names.size, dtype=np.int32)
    if feature_group == "late_over_early_features":
        return _indices_matching(feature_names, ("late_over_early",))
    if feature_group == "energy_window_features":
        return _indices_matching(feature_names, ("energy",))
    if feature_group == "side_contrast_features":
        return _indices_matching(feature_names, ("side_contrast", "max_side_anomaly"))
    if feature_group == "receiver_summary_features":
        return _indices_matching(feature_names, ("receiver_", "near_far_ratio"))
    return np.arange(min(20, feature_names.size), dtype=np.int32)


def _standardize_by_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = sample_weight.astype(np.float32)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        weights = np.ones(weights.shape, dtype=np.float32)
        weight_sum = float(weights.size)
    weights = weights / np.float32(weight_sum)
    mean = np.sum(x_train * weights[:, None], axis=0)
    centered = x_train - mean
    variance = np.sum((centered**2) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1.0e-6))
    return (
        ((x_train - mean) / scale).astype(np.float32),
        ((x_val - mean) / scale).astype(np.float32),
    )


def _class_balanced_weights(labels: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    base = np.asarray(confidence, dtype=np.float32)
    base = np.where(np.isfinite(base) & (base > 0.0), base, 1.0).astype(np.float32)
    weights = np.zeros(y.shape, dtype=np.float32)
    for label in (0, 1):
        mask = y == label
        if not np.any(mask):
            continue
        class_base = base[mask]
        class_sum = float(np.sum(class_base))
        if class_sum <= 0.0:
            class_base = np.ones(class_base.shape, dtype=np.float32)
            class_sum = float(class_base.size)
        weights[mask] = class_base * np.float32(0.5 / class_sum)
    max_weight = float(np.max(weights)) if weights.size else 0.0
    if max_weight > 0.0:
        weights = weights / np.float32(max_weight)
    return weights.astype(np.float32)


def _aggregate_metric_dicts(rows: list[dict[str, Any]]) -> dict[str, float | int | bool | None]:
    result: dict[str, float | int | bool | None] = {}
    for metric_name in (
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "predicted_positive_rate",
    ):
        result[metric_name] = _nanmean(
            np.asarray(
                [
                    np.nan
                    if row.get(metric_name) is None
                    else float(row[metric_name])
                    for row in rows
                ],
                dtype=np.float64,
            )
        )
    result["sample_count"] = int(sum(int(row.get("sample_count") or 0) for row in rows))
    result["weight_sum"] = float(sum(float(row.get("weight_sum") or 0.0) for row in rows))
    result["degenerate_prediction"] = bool(
        any(bool(row.get("degenerate_prediction")) for row in rows)
    )
    return result


def _metric_values(rows: list[dict[str, Any]], metric_name: str) -> np.ndarray:
    values = []
    for row in rows:
        if metric_name == "margin":
            value = row.get("margin")
        else:
            value = _as_dict(row.get("metrics")).get(metric_name)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def _group_summary(
    scenario_summaries: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    values = sorted({str(row.get(key)) for row in scenario_summaries})
    for value in values:
        rows = [row for row in scenario_summaries if str(row.get(key)) == value]
        margins = np.asarray(
            [
                np.nan if row.get("margin_mean") is None else float(row["margin_mean"])
                for row in rows
            ],
            dtype=np.float64,
        )
        result[value] = {
            "scenario_count": len(rows),
            "passing_scenario_count": int(
                sum(bool(row.get("passes_gate_thresholds")) for row in rows)
            ),
            "best_margin_mean": _nanmax(margins),
            "mean_margin_mean": _nanmean(margins),
        }
    return result


def _best_scenario(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("margin_mean") is not None]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            bool(row.get("passes_gate_thresholds")),
            float(row.get("margin_mean") or -999.0),
            float(row.get("balanced_accuracy_mean") or -999.0),
        ),
        reverse=True,
    )
    return candidates[0]


def _best_configuration(best: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": best.get("scenario_id"),
        "feature_group": best.get("feature_group"),
        "exclude_5700_band": best.get("exclude_5700_band"),
        "confidence_threshold": best.get("confidence_threshold"),
        "n_splits": best.get("n_splits"),
        "model_type": best.get("model_type"),
    }


def _recommendation(
    errors: list[str],
    robustness: dict[str, Any],
    passing: list[dict[str, Any]],
) -> str:
    if errors or not passing:
        return "no_go"
    if manual_confirmation_items(robustness):
        return "conditional_go"
    return "go"


def _validate_runtime_guardrails(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    baseline_report: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    for guardrail in ("no_final_labels", "no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        _check_npz_guardrail(label_arrays, guardrail, "label", errors, warnings)
        _check_npz_guardrail(feature_arrays, guardrail, "feature", errors, warnings)
        if baseline_report.get(guardrail) is not True:
            errors.append(f"baseline report does not set {guardrail}=true.")
    if baseline_report.get("production_training") is not False:
        errors.append("baseline report indicates production_training.")
    usable = baseline_report.get("usable_target_variants", [])
    if "high_confidence_positive_vs_clear_negative" not in usable:
        errors.append("baseline report does not allow high-confidence target refinement.")


def _check_npz_guardrail(
    arrays: dict[str, np.ndarray],
    guardrail: str,
    source: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if guardrail not in arrays:
        warnings.append(f"depth-level {source} NPZ has no {guardrail} field.")
        return
    if not bool(np.asarray(arrays[guardrail]).reshape(())):
        errors.append(f"depth-level {source} NPZ does not set {guardrail}=true.")


def _indices_matching(names: np.ndarray, needles: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(names.shape, dtype=bool)
    for needle in needles:
        mask |= np.char.find(names.astype(str), needle) >= 0
    return np.flatnonzero(mask).astype(np.int32)


def _robust_top_feature_indices(names: np.ndarray, baseline_report: dict[str, Any]) -> np.ndarray:
    top_features = _as_dict(baseline_report.get("top_features"))
    best = _as_dict(baseline_report.get("best_result"))
    key = f"{best.get('target_variant')}:{best.get('model_type')}"
    rows = top_features.get(key)
    if not isinstance(rows, list):
        rows = next((value for value in top_features.values() if isinstance(value, list)), [])
    selected: list[int] = []
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("feature_name", ""))
        matches = np.flatnonzero(names == name)
        if matches.size:
            selected.append(int(matches[0]))
    return np.asarray(selected, dtype=np.int32)


def _interval_mask(depth: np.ndarray, config: DepthLevelRefinementConfig) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    mask = np.zeros(values.shape, dtype=bool)
    for interval in config.review_intervals:
        mask |= (values >= interval.depth_min_ft) & (values <= interval.depth_max_ft)
    return mask


def _effective_positive_weight_fraction(labels: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    return float(np.sum(weights[labels == 1]) / total)


def _scenario_id(
    *,
    feature_group: str,
    exclude_5700_band: bool,
    confidence_threshold: float,
    n_splits: int,
    model_type: str,
) -> str:
    threshold = str(confidence_threshold).replace(".", "p")
    exclude = "true" if exclude_5700_band else "false"
    return (
        f"{feature_group}__exclude5700_{exclude}__conf_{threshold}"
        f"__split_{n_splits}__{model_type}"
    )


def _parse_scenario_id(scenario_id: str) -> dict[str, Any]:
    parts = scenario_id.split("__")
    feature_group = parts[0]
    exclude = parts[1].replace("exclude5700_", "") == "true"
    threshold = float(parts[2].replace("conf_", "").replace("p", "."))
    n_splits = int(parts[3].replace("split_", ""))
    model_type = parts[4]
    return {
        "feature_group": feature_group,
        "exclude_5700_band": exclude,
        "confidence_threshold": threshold,
        "n_splits": n_splits,
        "model_type": model_type,
    }


def _scenario_confidence_threshold(scenario_id: str) -> float:
    return float(_parse_scenario_id(scenario_id)["confidence_threshold"])


def _stable_seed(scenario_id: str, fold_index: int) -> int:
    value = 2166136261
    for char in f"{scenario_id}:{fold_index}":
        value ^= ord(char)
        value = (value * 16777619) % (2**32)
    return int(value)


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.mean(finite))


def _nanstd(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.std(finite))


def _nanmax(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.max(finite))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
