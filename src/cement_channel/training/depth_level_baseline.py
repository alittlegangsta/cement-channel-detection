from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.depth_level_baseline_schema import (
    DEPTH_LEVEL_BASELINE_CSV_VERSION,
    DEPTH_LEVEL_BASELINE_REPORT_VERSION,
    DepthLevelBaselineConfig,
    load_depth_level_baseline_config,
)
from cement_channel.training.depth_splits import make_depth_block_splits
from cement_channel.training.simple_baseline import (
    fit_linear_probe,
    fit_logistic_regression,
    predict_scores,
)


@dataclass(frozen=True)
class DepthLevelBaselineReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    model_backend: str
    allowed_scope: str
    target_variant_summaries: dict[str, dict[str, Any]]
    fold_metrics: list[dict[str, Any]]
    aggregate_metrics: dict[str, dict[str, dict[str, float | bool | None]]]
    permutation_metrics: dict[str, dict[str, dict[str, float | None]]]
    permutation_check: dict[str, dict[str, dict[str, float | bool | None]]]
    top_features: dict[str, list[dict[str, float | str]]]
    usable_target_variants: list[str]
    best_result: dict[str, Any] | None
    output_csv_version: str
    sanity_model_training_performed: bool
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


def run_depth_level_baseline_from_config(
    *,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    baseline_config_path: Path | str,
    output_report_md: Path | str | None = None,
    output_report_json: Path | str | None = None,
    output_csv: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[DepthLevelBaselineReport, list[dict[str, Any]]]:
    labels = _load_npz(depth_level_labels_npz)
    features = _load_npz(depth_level_features_npz)
    report, rows = run_depth_level_baseline(
        label_arrays=labels,
        feature_arrays=features,
        config=load_depth_level_baseline_config(baseline_config_path),
        inputs={
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "baseline_config_path": str(baseline_config_path),
        },
    )
    if output_report_md is not None and output_report_json is not None and output_csv is not None:
        write_depth_level_baseline_outputs(
            report,
            rows,
            output_report_md=Path(output_report_md),
            output_report_json=Path(output_report_json),
            output_csv=Path(output_csv),
            overwrite=overwrite,
        )
    return report, rows


def run_depth_level_baseline(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    config: DepthLevelBaselineConfig,
    inputs: dict[str, str] | None = None,
) -> tuple[DepthLevelBaselineReport, list[dict[str, Any]]]:
    warnings: list[str] = []
    errors: list[str] = []
    prepared = prepare_depth_level_baseline_inputs(label_arrays, feature_arrays)
    warnings.extend(prepared["warnings"])
    errors.extend(prepared["errors"])
    rng = np.random.default_rng(config.evaluation.permutation_seed)
    prediction_rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    coefficients: dict[str, list[np.ndarray]] = {}
    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant in config.target_variants:
        variant_data = build_target_variant(prepared, config, variant)
        variant_summaries[variant] = variant_data["summary"]
        warnings.extend(variant_data["warnings"])
        if not variant_data["runnable"]:
            continue
        split_plan = make_depth_block_splits(
            depth=variant_data["depth"],
            labels=variant_data["label"],
            n_splits=config.split.n_splits,
            min_gap_ft=config.split.min_gap_ft,
            block_size_ft=config.split.depth_block_size_ft,
            min_samples_per_class=config.target_filters.min_samples_per_class_per_fold,
        )
        variant_summaries[variant]["folds"] = split_plan.summaries()
        if split_plan.errors:
            warnings.extend(f"{variant}: {message}" for message in split_plan.errors)
            variant_summaries[variant]["status"] = "skipped_split_error"
            continue
        if split_plan.warnings:
            warnings.extend(f"{variant}: {message}" for message in split_plan.warnings)
        for model_type in config.model_types:
            for fold in split_plan.folds:
                result = _run_depth_fold(
                    variant=variant,
                    model_type=model_type,
                    prepared=variant_data,
                    fold_index=fold.fold_index,
                    train_mask=fold.train_mask,
                    validation_mask=fold.validation_mask,
                    config=config,
                    rng=rng,
                )
                warnings.extend(result["warnings"])
                fold_metrics.extend(result["fold_metrics"])
                prediction_rows.extend(result["prediction_rows"])
                if result["coefficient"] is not None:
                    coefficients.setdefault(f"{variant}:{model_type}", []).append(
                        result["coefficient"]
                    )
    aggregate = aggregate_depth_baseline_metrics(
        [row for row in fold_metrics if not row["permutation"]]
    )
    permutation = aggregate_depth_baseline_metrics(
        [row for row in fold_metrics if row["permutation"]]
    )
    permutation_check = depth_permutation_check(aggregate, permutation, config, fold_metrics)
    _annotate_variant_summaries(variant_summaries, aggregate, permutation_check, config)
    usable = usable_target_variants(variant_summaries, permutation_check)
    if not usable:
        errors.append("No target variant produced a non-degenerate permutation-safe baseline.")
    best = best_depth_baseline_result(variant_summaries, permutation_check)
    top_features = summarize_depth_coefficients(
        coefficients,
        prepared["feature_names"],
    )
    report = DepthLevelBaselineReport(
        report_version=DEPTH_LEVEL_BASELINE_REPORT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        model_backend="numpy_fallback",
        allowed_scope=config.allowed_scope,
        target_variant_summaries=variant_summaries,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        permutation_metrics=permutation,
        permutation_check=permutation_check,
        top_features=top_features,
        usable_target_variants=usable,
        best_result=best,
        output_csv_version=DEPTH_LEVEL_BASELINE_CSV_VERSION,
        sanity_model_training_performed=bool(fold_metrics),
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
        ],
    )
    return report, prediction_rows


def prepare_depth_level_baseline_inputs(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrail_arrays(label_arrays, feature_arrays, errors)
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
        warnings.append("non-finite feature rows are excluded from baseline variants.")
    return {
        "depth": depth,
        "features": np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "feature_names": feature_names.astype(str).tolist(),
        "finite_rows": finite_rows,
        "has_channel": np.asarray(label_arrays["depth_has_channel_any"], dtype=bool).reshape(-1),
        "strong_positive": np.asarray(
            label_arrays["depth_strong_positive_mask"],
            dtype=bool,
        ).reshape(-1),
        "clear_negative": np.asarray(
            label_arrays["depth_clear_negative_mask"],
            dtype=bool,
        ).reshape(-1),
        "review_band": np.asarray(label_arrays["depth_review_band_mask"], dtype=bool).reshape(-1),
        "label_confidence": np.asarray(
            label_arrays["depth_label_confidence"],
            dtype=np.float32,
        ).reshape(-1),
        "orientation_confidence": np.asarray(
            label_arrays["depth_orientation_confidence"],
            dtype=np.float32,
        ).reshape(-1),
        "disagreement_fraction": np.asarray(
            label_arrays["depth_plus_minus_disagreement_fraction"],
            dtype=np.float32,
        ).reshape(-1),
        "warnings": warnings,
        "errors": errors,
    }


def build_target_variant(
    prepared: dict[str, Any],
    config: DepthLevelBaselineConfig,
    variant: str,
) -> dict[str, Any]:
    base_valid = prepared["finite_rows"]
    if config.target_filters.exclude_review_band:
        base_valid = base_valid & ~prepared["review_band"]
    if variant == "all_positive_vs_negative":
        positive = prepared["has_channel"] & base_valid
        negative = (~prepared["has_channel"]) & base_valid
    elif variant == "strong_positive_vs_clear_negative":
        positive = prepared["strong_positive"] & base_valid
        negative = prepared["clear_negative"] & base_valid
    elif variant == "high_confidence_positive_vs_clear_negative":
        positive = prepared["has_channel"] & _high_confidence_mask(
            prepared,
            config.target_filters.high_confidence_positive,
        )
        negative = prepared["clear_negative"] & _high_confidence_mask(
            prepared,
            config.target_filters.clear_negative,
        )
        positive &= base_valid
        negative &= base_valid
    else:
        raise ValueError(f"Unsupported target variant: {variant}")
    selected = positive | negative
    label = np.where(positive[selected], 1, 0).astype(np.int8)
    sample_weight = _class_balanced_weights(
        label,
        prepared["label_confidence"][selected],
    )
    warnings: list[str] = []
    positive_count = int(np.count_nonzero(label == 1))
    negative_count = int(np.count_nonzero(label == 0))
    runnable = True
    status = "runnable"
    min_count = config.target_filters.min_samples_per_class
    if positive_count < min_count or negative_count < min_count:
        runnable = False
        status = "skipped_too_few_samples"
        warnings.append(
            f"{variant}: too few samples for baseline sanity "
            f"(positive={positive_count}, negative={negative_count}, min={min_count})."
        )
    summary = {
        "status": status,
        "sample_count": int(label.size),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction": None if label.size == 0 else positive_count / label.size,
        "effective_positive_weight_fraction": _effective_positive_weight_fraction(
            label,
            sample_weight,
        ),
    }
    return {
        "variant": variant,
        "features": prepared["features"][selected],
        "depth": prepared["depth"][selected],
        "label": label,
        "sample_weight": sample_weight,
        "selected_global_index": np.flatnonzero(selected).astype(np.int32),
        "summary": summary,
        "warnings": warnings,
        "runnable": runnable,
    }


def aggregate_depth_baseline_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | bool | None]]]:
    result: dict[str, dict[str, dict[str, float | bool | None]]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["target_variant"]), str(row["model_type"])), []).append(row)
    for (variant, model_type), group_rows in grouped.items():
        weights = np.asarray(
            [float(row["metrics"].get("weight_sum") or 0.0) for row in group_rows],
            dtype=np.float64,
        )
        metrics: dict[str, float | bool | None] = {}
        for metric_name in (
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "predicted_positive_rate",
        ):
            values = np.asarray(
                [
                    np.nan
                    if row["metrics"].get(metric_name) is None
                    else float(row["metrics"][metric_name])
                    for row in group_rows
                ],
                dtype=np.float64,
            )
            metrics[metric_name] = _weighted_nanmean(values, weights)
        metrics["sample_count"] = float(
            sum(int(row["metrics"].get("sample_count") or 0) for row in group_rows)
        )
        metrics["weight_sum"] = float(np.sum(weights))
        metrics["degenerate_prediction"] = bool(
            any(bool(row["metrics"].get("degenerate_prediction")) for row in group_rows)
        )
        result.setdefault(variant, {})[model_type] = metrics
    return result


def depth_permutation_check(
    aggregate: dict[str, dict[str, dict[str, float | bool | None]]],
    permutation: dict[str, dict[str, dict[str, float | bool | None]]],
    config: DepthLevelBaselineConfig,
    fold_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, dict[str, float | bool | None]]]:
    result: dict[str, dict[str, dict[str, float | bool | None]]] = {}
    for variant in config.target_variants:
        result[variant] = {}
        for model_type in config.model_types:
            aggregate_metrics = _as_dict(_as_dict(aggregate.get(variant)).get(model_type))
            permutation_metrics = _as_dict(_as_dict(permutation.get(variant)).get(model_type))
            real = _as_float(
                aggregate_metrics.get("balanced_accuracy")
            )
            permuted = _as_float(
                permutation_metrics.get("balanced_accuracy")
            )
            positive_rate = _as_float(
                aggregate_metrics.get("predicted_positive_rate")
            )
            margin = None if real is None or permuted is None else real - permuted
            fold_stability = _fold_stability_check(
                variant=variant,
                model_type=model_type,
                fold_metrics=fold_metrics or [],
                config=config,
            )
            degenerate = _degenerate_positive_rate(positive_rate, config) or bool(
                aggregate_metrics.get("degenerate_prediction")
            )
            stable_folds_pass = (
                fold_stability["stable_fold_count"]
                >= config.evaluation.stable_fold_min_count
            )
            result[variant][model_type] = {
                "real_balanced_accuracy": real,
                "permutation_balanced_accuracy": permuted,
                "balanced_accuracy_margin": margin,
                "required_margin": config.evaluation.min_permutation_balanced_accuracy_margin,
                "predicted_positive_rate": positive_rate,
                "degenerate_prediction": degenerate,
                "stable_fold_count": int(fold_stability["stable_fold_count"]),
                "stable_fold_min_count": config.evaluation.stable_fold_min_count,
                "stable_folds_pass": stable_folds_pass,
                "fold_checks": fold_stability["fold_checks"],
                "permutation_lower_than_real": (
                    None if real is None or permuted is None else permuted < real
                ),
                "passes_margin": (
                    bool(margin >= config.evaluation.min_permutation_balanced_accuracy_margin)
                    if margin is not None
                    else False
                ),
                "usable": (
                    bool(margin >= config.evaluation.min_permutation_balanced_accuracy_margin)
                    and not degenerate
                    and stable_folds_pass
                    if margin is not None
                    else False
                ),
            }
    return result


def _fold_stability_check(
    *,
    variant: str,
    model_type: str,
    fold_metrics: list[dict[str, Any]],
    config: DepthLevelBaselineConfig,
) -> dict[str, Any]:
    grouped: dict[int, dict[bool, dict[str, Any]]] = {}
    for row in fold_metrics:
        if row.get("target_variant") != variant or row.get("model_type") != model_type:
            continue
        grouped.setdefault(int(row["fold_index"]), {})[bool(row["permutation"])] = row

    fold_checks: list[dict[str, float | int | bool | None]] = []
    for fold_index in sorted(grouped):
        real_row = grouped[fold_index].get(False)
        permutation_row = grouped[fold_index].get(True)
        if real_row is None or permutation_row is None:
            continue
        real_metrics = _as_dict(real_row.get("metrics"))
        permutation_metrics = _as_dict(permutation_row.get("metrics"))
        real_balanced_accuracy = _as_float(real_metrics.get("balanced_accuracy"))
        permutation_balanced_accuracy = _as_float(
            permutation_metrics.get("balanced_accuracy")
        )
        margin = (
            None
            if real_balanced_accuracy is None or permutation_balanced_accuracy is None
            else real_balanced_accuracy - permutation_balanced_accuracy
        )
        predicted_positive_rate = _as_float(real_metrics.get("predicted_positive_rate"))
        degenerate = _degenerate_positive_rate(predicted_positive_rate, config) or bool(
            real_metrics.get("degenerate_prediction")
        )
        passes = (
            margin is not None
            and margin >= config.evaluation.min_permutation_balanced_accuracy_margin
            and not degenerate
        )
        fold_checks.append(
            {
                "fold_index": fold_index,
                "real_balanced_accuracy": real_balanced_accuracy,
                "permutation_balanced_accuracy": permutation_balanced_accuracy,
                "balanced_accuracy_margin": margin,
                "predicted_positive_rate": predicted_positive_rate,
                "degenerate_prediction": degenerate,
                "passes": bool(passes),
            }
        )
    return {
        "stable_fold_count": int(sum(bool(row["passes"]) for row in fold_checks)),
        "fold_checks": fold_checks,
    }


def summarize_depth_coefficients(
    coefficients: dict[str, list[np.ndarray]],
    feature_names: list[str],
    *,
    limit: int = 20,
) -> dict[str, list[dict[str, float | str]]]:
    result: dict[str, list[dict[str, float | str]]] = {}
    for key, values in coefficients.items():
        if not values:
            continue
        matrix = np.vstack(values)
        mean_coef = np.mean(matrix[:, 1:], axis=0)
        rows = [
            {
                "feature_name": name,
                "mean_coefficient": float(value),
                "mean_abs_coefficient": float(abs(value)),
            }
            for name, value in zip(feature_names, mean_coef, strict=True)
        ]
        rows.sort(key=lambda row: float(row["mean_abs_coefficient"]), reverse=True)
        result[key] = rows[:limit]
    return result


def usable_target_variants(
    summaries: dict[str, dict[str, Any]],
    checks: dict[str, dict[str, dict[str, float | bool | None]]],
) -> list[str]:
    usable: list[str] = []
    for variant, model_checks in checks.items():
        if summaries.get(variant, {}).get("status") != "runnable":
            continue
        if any(bool(check.get("usable")) for check in model_checks.values()):
            usable.append(variant)
    return usable


def best_depth_baseline_result(
    summaries: dict[str, dict[str, Any]],
    checks: dict[str, dict[str, dict[str, float | bool | None]]],
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for variant, model_checks in checks.items():
        if summaries.get(variant, {}).get("status") != "runnable":
            continue
        for model_type, check in model_checks.items():
            if not bool(check.get("usable")):
                continue
            rows.append(
                {
                    "target_variant": variant,
                    "model_type": model_type,
                    **check,
                }
            )
    if not rows:
        return None
    rows.sort(key=lambda row: float(row["balanced_accuracy_margin"] or 0.0), reverse=True)
    return rows[0]


def write_depth_level_baseline_outputs(
    report: DepthLevelBaselineReport,
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
    output_report_md.write_text(format_depth_level_baseline_markdown(report), encoding="utf-8")
    write_depth_level_prediction_csv(prediction_rows, output_csv)


def format_depth_level_baseline_markdown(report: DepthLevelBaselineReport) -> str:
    lines = [
        "# MVP-4B-R4b Depth-Level Baseline Sanity Report",
        "",
        "This is a simple baseline sanity check against CAST weak-label candidates. "
        "It is not formal model performance, not production training, and not final labels.",
        "",
        f"- report_version: `{report.report_version}`",
        f"- allowed_scope: `{report.allowed_scope}`",
        f"- usable_target_variants: {report.usable_target_variants}",
        f"- production_training: `{report.production_training}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        f"- no_stc: `{report.no_stc}`",
        f"- no_apes: `{report.no_apes}`",
        "",
        "## Target Variants",
        "",
    ]
    for variant, summary in report.target_variant_summaries.items():
        lines.append(
            "- "
            f"{variant}: status={summary.get('status')}, "
            f"sample_count={summary.get('sample_count')}, "
            f"positive={summary.get('positive_count')}, "
            f"negative={summary.get('negative_count')}, "
            f"positive_fraction={summary.get('positive_fraction')}"
        )
    lines.extend(["", "## Permutation Check", ""])
    for variant, model_checks in report.permutation_check.items():
        for model_type, check in model_checks.items():
            lines.append(
                "- "
                f"{variant} / {model_type}: "
                f"real_balanced_accuracy={check.get('real_balanced_accuracy')}, "
                f"permutation={check.get('permutation_balanced_accuracy')}, "
                f"margin={check.get('balanced_accuracy_margin')}, "
                f"predicted_positive_rate={check.get('predicted_positive_rate')}, "
                f"usable={check.get('usable')}"
            )
    lines.extend(["", "## Best Result", ""])
    lines.append(f"- {report.best_result if report.best_result else 'none'}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def write_depth_level_prediction_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "csv_version",
        "target_variant",
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


def _run_depth_fold(
    *,
    variant: str,
    model_type: str,
    prepared: dict[str, Any],
    fold_index: int,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: DepthLevelBaselineConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    warnings: list[str] = []
    x_train_raw = prepared["features"][train_mask]
    x_val_raw = prepared["features"][validation_mask]
    y_train = prepared["label"][train_mask]
    y_val = prepared["label"][validation_mask]
    w_train = prepared["sample_weight"][train_mask]
    w_val = prepared["sample_weight"][validation_mask]
    if not _has_two_classes(y_train) or not _has_two_classes(y_val):
        warnings.append(f"{variant}/{model_type}/fold_{fold_index}: split is single-class.")
        return {
            "warnings": warnings,
            "fold_metrics": [],
            "prediction_rows": [],
            "coefficient": None,
        }
    x_train, x_val = _standardize_by_train(x_train_raw, x_val_raw, w_train)
    coef = _fit_model(model_type, x_train, y_train, w_train, config)
    score = predict_scores(x_val, coef, model_type=model_type)
    metrics = binary_metrics_with_positive_rate(
        y_val,
        score,
        w_val,
        config=config,
    )
    fold_metrics = [
        {
            "target_variant": variant,
            "model_type": model_type,
            "fold_index": fold_index,
            "permutation": False,
            "metrics": metrics,
        }
    ]
    rows = _prediction_rows(
        variant=variant,
        model_type=model_type,
        fold_index=fold_index,
        depth=prepared["depth"][validation_mask],
        label=y_val,
        sample_weight=w_val,
        score=score,
    )
    permuted = np.asarray(y_train, dtype=np.int8).copy()
    rng.shuffle(permuted)
    perm_coef = _fit_model(model_type, x_train, permuted, w_train, config)
    perm_score = predict_scores(x_val, perm_coef, model_type=model_type)
    fold_metrics.append(
        {
            "target_variant": variant,
            "model_type": model_type,
            "fold_index": fold_index,
            "permutation": True,
            "metrics": binary_metrics_with_positive_rate(
                y_val,
                perm_score,
                w_val,
                config=config,
            ),
        }
    )
    return {
        "warnings": warnings,
        "fold_metrics": fold_metrics,
        "prediction_rows": rows,
        "coefficient": coef,
    }


def binary_metrics_with_positive_rate(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray,
    *,
    config: DepthLevelBaselineConfig,
) -> dict[str, float | int | bool | None]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    scores = np.asarray(score, dtype=np.float32).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    known = np.isin(y, [0, 1]) & np.isfinite(scores) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(known):
        return {
            "sample_count": 0,
            "weight_sum": 0.0,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "predicted_positive_rate": None,
            "degenerate_prediction": True,
        }
    y = y[known]
    scores = scores[known]
    weights = weights[known]
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(y.shape, dtype=np.float32)
    pred = (scores >= 0.5).astype(np.int8)
    tp = float(np.sum(weights[(pred == 1) & (y == 1)]))
    tn = float(np.sum(weights[(pred == 0) & (y == 0)]))
    fp = float(np.sum(weights[(pred == 1) & (y == 0)]))
    fn = float(np.sum(weights[(pred == 0) & (y == 1)]))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    positive_rate = float(np.mean(pred == 1))
    return {
        "sample_count": int(y.size),
        "weight_sum": float(np.sum(weights)),
        "balanced_accuracy": _safe_mean([recall, specificity]),
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "predicted_positive_rate": positive_rate,
        "degenerate_prediction": _degenerate_positive_rate(positive_rate, config),
    }


def _fit_model(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    config: DepthLevelBaselineConfig,
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


def _prediction_rows(
    *,
    variant: str,
    model_type: str,
    fold_index: int,
    depth: np.ndarray,
    label: np.ndarray,
    sample_weight: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "csv_version": DEPTH_LEVEL_BASELINE_CSV_VERSION,
            "target_variant": variant,
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


def _high_confidence_mask(
    prepared: dict[str, Any],
    filter_config: Any,
) -> np.ndarray:
    return (
        (prepared["label_confidence"] >= filter_config.min_label_confidence)
        & (prepared["orientation_confidence"] >= filter_config.min_orientation_confidence)
        & (
            prepared["disagreement_fraction"]
            <= filter_config.max_plus_minus_disagreement_fraction
        )
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


def _annotate_variant_summaries(
    summaries: dict[str, dict[str, Any]],
    aggregate: dict[str, dict[str, dict[str, float | bool | None]]],
    checks: dict[str, dict[str, dict[str, float | bool | None]]],
    config: DepthLevelBaselineConfig,
) -> None:
    for variant, summary in summaries.items():
        model_checks = checks.get(variant, {})
        summary["model_metrics"] = aggregate.get(variant, {})
        summary["permutation_check"] = model_checks
        summary["usable_model_count"] = int(
            sum(bool(check.get("usable")) for check in model_checks.values())
        )
        summary["stable_fold_count"] = max(
            (int(check.get("stable_fold_count") or 0) for check in model_checks.values()),
            default=0,
        )
        summary["stable_fold_min_count"] = config.evaluation.stable_fold_min_count


def _degenerate_positive_rate(
    positive_rate: float | None,
    config: DepthLevelBaselineConfig,
) -> bool:
    if positive_rate is None:
        return True
    return (
        positive_rate <= config.evaluation.degenerate_prediction_min_positive_rate
        or positive_rate >= config.evaluation.degenerate_prediction_max_positive_rate
    )


def _validate_guardrail_arrays(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    errors: list[str],
) -> None:
    required_labels = (
        "depth",
        "depth_has_channel_any",
        "depth_strong_positive_mask",
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
    for guardrail in ("no_final_labels", "no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        if guardrail in label_arrays and not bool(np.asarray(label_arrays[guardrail]).reshape(())):
            errors.append(f"depth-level label NPZ must keep {guardrail}=true.")
        if guardrail in feature_arrays and not bool(
            np.asarray(feature_arrays[guardrail]).reshape(())
        ):
            errors.append(f"depth-level feature NPZ must keep {guardrail}=true.")


def _has_two_classes(labels: np.ndarray) -> bool:
    return np.unique(labels).size == 2


def _effective_positive_weight_fraction(labels: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    return float(np.sum(weights[labels == 1]) / total)


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float | None:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return None
    return float(np.average(values[valid], weights=weights[valid]))


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _safe_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def _safe_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


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


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
