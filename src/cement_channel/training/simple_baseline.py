from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.baseline_schema import (
    MVP4B_SIMPLE_BASELINE_CSV_VERSION,
    MVP4B_SIMPLE_BASELINE_REPORT_VERSION,
    BaselineConfig,
    load_baseline_config,
)
from cement_channel.training.depth_splits import make_depth_block_splits


@dataclass(frozen=True)
class SimpleBaselineReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    model_backend: str
    allowed_scope: str
    sample_counts: dict[str, int | float | None]
    class_balance: dict[str, int | float | None]
    split: dict[str, Any]
    fold_metrics: list[dict[str, Any]]
    aggregate_metrics: dict[str, dict[str, float | None]]
    permutation_aggregate_metrics: dict[str, dict[str, float | None]]
    permutation_check: dict[str, dict[str, float | bool | None]]
    minus_audit_comparison: dict[str, dict[str, float | None]]
    disagreement_subset_analysis: dict[str, dict[str, float | int | None]]
    coefficient_summary: dict[str, dict[str, float | None]]
    output_csv_version: str
    sanity_model_training_performed: bool
    production_training: bool
    no_final_labels: bool
    no_deep_learning: bool
    no_stc: bool
    no_apes: bool
    no_production_model: bool
    leakage_suspected: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_simple_baseline_from_config(
    *,
    sample_table_npz: Path | str,
    baseline_config_path: Path | str,
) -> tuple[SimpleBaselineReport, list[dict[str, Any]]]:
    return run_simple_baseline(
        sample_table_npz=sample_table_npz,
        baseline_config=load_baseline_config(baseline_config_path),
        baseline_config_path=baseline_config_path,
    )


def run_simple_baseline(
    *,
    sample_table_npz: Path | str,
    baseline_config: BaselineConfig,
    baseline_config_path: Path | str | None = None,
) -> tuple[SimpleBaselineReport, list[dict[str, Any]]]:
    arrays = _load_npz(sample_table_npz)
    return run_simple_baseline_from_arrays(
        arrays=arrays,
        baseline_config=baseline_config,
        inputs={
            "sample_table_npz": str(sample_table_npz),
            "baseline_config_path": str(baseline_config_path) if baseline_config_path else "",
        },
    )


def run_simple_baseline_from_arrays(
    *,
    arrays: dict[str, np.ndarray],
    baseline_config: BaselineConfig,
    inputs: dict[str, str] | None = None,
) -> tuple[SimpleBaselineReport, list[dict[str, Any]]]:
    warnings: list[str] = []
    errors: list[str] = []
    prepared = prepare_baseline_samples(arrays, baseline_config)
    warnings.extend(prepared["warnings"])
    errors.extend(prepared["errors"])
    if prepared["selected_count"] == 0:
        errors.append("No samples remain after filtering.")
    if prepared["candidate_count"] < baseline_config.min_samples_per_class:
        errors.append(
            "High-confidence candidate sample count below "
            f"{baseline_config.min_samples_per_class}: {prepared['candidate_count']}."
        )
    if prepared["non_candidate_count"] < baseline_config.min_samples_per_class:
        errors.append(
            "High-confidence non-candidate sample count below "
            f"{baseline_config.min_samples_per_class}: {prepared['non_candidate_count']}."
        )
    if errors:
        report = _empty_error_report(
            baseline_config=baseline_config,
            inputs=inputs or {},
            prepared=prepared,
            warnings=warnings,
            errors=errors,
        )
        return report, []

    split_plan = make_depth_block_splits(
        depth=prepared["depth"],
        labels=prepared["label"],
        n_splits=baseline_config.n_splits,
        min_gap_ft=baseline_config.min_gap_ft,
        block_size_ft=baseline_config.depth_block_size_ft,
        min_samples_per_class=baseline_config.min_samples_per_class_per_fold,
    )
    warnings.extend(split_plan.warnings)
    errors.extend(split_plan.errors)
    if split_plan.warnings:
        errors.extend(
            f"Depth split class balance warning escalated for Stage 2: {message}"
            for message in split_plan.warnings
        )

    rng = np.random.default_rng(baseline_config.permutation_seed)
    fold_metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficients: dict[str, list[np.ndarray]] = {
        model_type: [] for model_type in baseline_config.model_types
    }
    feature_names = [str(value) for value in prepared["feature_names"]]
    for model_type in baseline_config.model_types:
        for fold in split_plan.folds:
            fold_result = _run_fold(
                prepared=prepared,
                fold_index=fold.fold_index,
                train_mask=fold.train_mask,
                validation_mask=fold.validation_mask,
                model_type=model_type,
                baseline_config=baseline_config,
                feature_names=feature_names,
                rng=rng,
            )
            warnings.extend(fold_result["warnings"])
            errors.extend(fold_result["errors"])
            fold_metrics.extend(fold_result["fold_metrics"])
            prediction_rows.extend(fold_result["prediction_rows"])
            if fold_result["coefficient"] is not None:
                coefficients[model_type].append(fold_result["coefficient"])

    aggregate = aggregate_fold_metrics(
        [metric for metric in fold_metrics if not metric["permutation"]]
    )
    permutation_aggregate = aggregate_fold_metrics(
        [metric for metric in fold_metrics if metric["permutation"]]
    )
    permutation_check = _permutation_check(aggregate, permutation_aggregate, baseline_config)
    for model_type, check in permutation_check.items():
        real_balanced = check["real_balanced_accuracy"]
        permutation_balanced = check["permutation_balanced_accuracy"]
        if (
            real_balanced is not None
            and permutation_balanced is not None
            and permutation_balanced >= real_balanced
        ):
            errors.append(
                f"{model_type}: permutation balanced_accuracy is not lower than real labels."
            )
        elif not check["passes_margin"]:
            warnings.append(
                f"{model_type}: balanced_accuracy margin over permutation is below configured "
                f"threshold: {check['balanced_accuracy_margin']}."
            )
    leakage_suspected = _leakage_suspected(aggregate, baseline_config)
    if leakage_suspected:
        warnings.append("Suspiciously high sanity metrics; leakage review is required.")
    minus_audit = minus_audit_comparison(prediction_rows)
    disagreement = disagreement_subset_analysis(prediction_rows)
    coefficient_summary = summarize_coefficients(coefficients, feature_names)
    report = SimpleBaselineReport(
        report_version=MVP4B_SIMPLE_BASELINE_REPORT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        model_backend="numpy_fallback",
        allowed_scope=baseline_config.allowed_scope,
        sample_counts={
            "total_samples": prepared["total_count"],
            "selected_samples": prepared["selected_count"],
            "excluded_samples": prepared["total_count"] - prepared["selected_count"],
            "excluded_large_depth_match_error": prepared["excluded_large_depth_match_error"],
            "excluded_plus_minus_disagreement": prepared["excluded_plus_minus_disagreement"],
            "excluded_zero_or_invalid_weight": prepared["excluded_zero_or_invalid_weight"],
        },
        class_balance={
            "candidate_count": prepared["candidate_count"],
            "non_candidate_count": prepared["non_candidate_count"],
            "candidate_fraction": _safe_div(
                prepared["candidate_count"],
                prepared["selected_count"],
            ),
        },
        split={
            "method": baseline_config.split_method,
            "n_splits": baseline_config.n_splits,
            "block_size_ft": split_plan.block_size_ft,
            "min_gap_ft": split_plan.min_gap_ft,
            "folds": split_plan.summaries(),
        },
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        permutation_aggregate_metrics=permutation_aggregate,
        permutation_check=permutation_check,
        minus_audit_comparison=minus_audit,
        disagreement_subset_analysis=disagreement,
        coefficient_summary=coefficient_summary,
        output_csv_version=MVP4B_SIMPLE_BASELINE_CSV_VERSION,
        sanity_model_training_performed=True,
        production_training=False,
        no_final_labels=True,
        no_deep_learning=True,
        no_stc=True,
        no_apes=True,
        no_production_model=True,
        leakage_suspected=leakage_suspected,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "production training",
            "production inference",
            "deep learning",
            "STC",
            "APES",
            "large-scale hyperparameter search",
            "final label generation",
            "model weight export",
            "MVP-4C",
            "MVP-5",
        ],
    )
    return report, prediction_rows


def prepare_baseline_samples(
    arrays: dict[str, np.ndarray],
    config: BaselineConfig,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    no_final_labels = bool(np.asarray(arrays.get("no_final_labels", False)).reshape(()))
    no_stc = bool(np.asarray(arrays.get("no_stc", False)).reshape(()))
    no_apes = bool(np.asarray(arrays.get("no_apes", False)).reshape(()))
    if not no_final_labels:
        errors.append("Sample table does not set no_final_labels=true.")
    if not no_stc:
        errors.append("Sample table does not set no_stc=true.")
    if not no_apes:
        errors.append("Sample table does not set no_apes=true.")
    features = np.asarray(arrays["transformed_features"], dtype=np.float32)
    labels = np.asarray(arrays[config.label], dtype=np.int8).reshape(-1)
    depth = np.asarray(arrays["depth"], dtype=np.float32).reshape(-1)
    if features.shape[0] != labels.size or depth.size != labels.size:
        errors.append("Feature, label, and depth arrays have incompatible sample counts.")
    weights = np.asarray(arrays[config.sample_weight_source], dtype=np.float32).reshape(-1)
    disagreement = np.asarray(arrays["plus_minus_disagreement"], dtype=bool).reshape(-1)
    large_depth_error = np.asarray(
        arrays.get("exclude_large_depth_match_error", np.zeros(labels.size, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    valid_azimuthal = np.asarray(
        arrays["valid_for_azimuthal_validation"],
        dtype=bool,
    ).reshape(-1)
    finite_features = np.all(np.isfinite(features), axis=1)
    finite_weights = np.isfinite(weights)
    known_label = np.isin(labels, [0, 1])
    mask = known_label & finite_features & finite_weights
    if config.high_confidence_only or config.valid_for_azimuthal_validation:
        mask &= valid_azimuthal
    if config.exclude_plus_minus_disagreement:
        mask &= ~disagreement
    if config.exclude_large_depth_match_error:
        mask &= ~large_depth_error
    if config.use_sample_weight:
        mask &= weights > 0.0
    else:
        weights = np.ones(labels.shape, dtype=np.float32)
        warnings.append("Sample weights disabled by config.")
    selected = np.flatnonzero(mask)
    selected_features = features[selected]
    feature_names = np.asarray(arrays["transformed_feature_names"]).astype(str)
    candidate_count = int(np.count_nonzero(labels[selected] == 1))
    non_candidate_count = int(np.count_nonzero(labels[selected] == 0))
    return {
        "features": selected_features,
        "feature_names": feature_names.tolist(),
        "label": labels[selected],
        "minus_label": np.asarray(arrays["label_presence_minus_audit"], dtype=np.int8).reshape(-1)[
            selected
        ],
        "depth": depth[selected],
        "side_index": np.asarray(arrays["side_index"], dtype=np.int16).reshape(-1)[selected],
        "sample_id": np.asarray(arrays["sample_id"], dtype=np.int64).reshape(-1)[selected],
        "sample_weight": np.clip(weights[selected], 0.0, 1.0).astype(np.float32),
        "plus_minus_disagreement": disagreement[selected],
        "candidate_count": candidate_count,
        "non_candidate_count": non_candidate_count,
        "selected_count": int(selected.size),
        "total_count": int(labels.size),
        "excluded_large_depth_match_error": int(np.count_nonzero(known_label & large_depth_error)),
        "excluded_plus_minus_disagreement": int(np.count_nonzero(known_label & disagreement)),
        "excluded_zero_or_invalid_weight": int(
            np.count_nonzero(known_label & (~finite_weights | (weights <= 0.0)))
        ),
        "warnings": warnings,
        "errors": errors,
    }


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    max_iterations: int,
    learning_rate: float,
    l2_penalty: float,
) -> np.ndarray:
    x_design = _add_intercept(x_train)
    y = y_train.astype(np.float32)
    weights = _normalized_weights(sample_weight)
    coef = np.zeros(x_design.shape[1], dtype=np.float32)
    for _ in range(max_iterations):
        score = _sigmoid(x_design @ coef)
        residual = (score - y) * weights
        gradient = x_design.T @ residual
        gradient[1:] += np.float32(l2_penalty) * coef[1:]
        coef -= np.float32(learning_rate) * gradient
    return coef.astype(np.float32)


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l2_penalty: float,
) -> np.ndarray:
    x_design = _add_intercept(x_train)
    sqrt_weight = np.sqrt(np.clip(sample_weight.astype(np.float32), 0.0, None))
    x_weighted = x_design * sqrt_weight[:, None]
    y_weighted = y_train.astype(np.float32) * sqrt_weight
    penalty = np.eye(x_design.shape[1], dtype=np.float32) * np.float32(l2_penalty)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x_weighted.T @ x_weighted + penalty) @ x_weighted.T @ y_weighted


def predict_scores(x_values: np.ndarray, coef: np.ndarray, *, model_type: str) -> np.ndarray:
    raw = _add_intercept(x_values) @ coef
    if model_type == "logistic_regression":
        return _sigmoid(raw).astype(np.float32)
    if model_type == "linear_probe":
        return np.clip(raw, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported model_type: {model_type}")


def binary_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray,
    *,
    calibration_bins: int,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    scores = np.asarray(score, dtype=np.float32).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    known = np.isin(y, [0, 1]) & np.isfinite(scores) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(known):
        return _empty_metrics()
    y = y[known]
    scores = scores[known]
    weights = weights[known]
    pred = (scores >= 0.5).astype(np.int8)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        weights = np.ones(y.shape, dtype=np.float32)
        weight_sum = float(y.size)
    tp = float(np.sum(weights[(pred == 1) & (y == 1)]))
    tn = float(np.sum(weights[(pred == 0) & (y == 0)]))
    fp = float(np.sum(weights[(pred == 1) & (y == 0)]))
    fn = float(np.sum(weights[(pred == 0) & (y == 1)]))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "sample_count": int(y.size),
        "weight_sum": weight_sum,
        "weighted_accuracy": _safe_div(tp + tn, weight_sum),
        "balanced_accuracy": _safe_mean([recall, specificity]),
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "candidate_fraction": float(np.mean(y == 1)),
        "mean_score": float(np.mean(scores)),
        "brier": _safe_div(float(np.sum(weights * (scores - y) ** 2)), weight_sum),
        "calibration_summary": calibration_summary(
            y,
            scores,
            weights,
            n_bins=calibration_bins,
        ),
    }


def calibration_summary(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray,
    *,
    n_bins: int,
) -> list[dict[str, float | int | None]]:
    bins = np.linspace(0.0, 1.0, num=n_bins + 1)
    rows: list[dict[str, float | int | None]] = []
    for index in range(n_bins):
        low = bins[index]
        high = bins[index + 1]
        if index == n_bins - 1:
            mask = (score >= low) & (score <= high)
        else:
            mask = (score >= low) & (score < high)
        weights = sample_weight[mask]
        weight_sum = float(np.sum(weights))
        rows.append(
            {
                "bin": index,
                "score_min": float(low),
                "score_max": float(high),
                "sample_count": int(np.count_nonzero(mask)),
                "weight_sum": weight_sum,
                "mean_score": (
                    float(np.average(score[mask], weights=weights))
                    if weight_sum > 0.0
                    else None
                ),
                "observed_candidate_rate": (
                    float(np.average(y_true[mask], weights=weights))
                    if weight_sum > 0.0
                    else None
                ),
            }
        )
    return rows


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in fold_metrics:
        by_model.setdefault(str(row["model_type"]), []).append(row)
    result: dict[str, dict[str, float | None]] = {}
    for model_type, rows in by_model.items():
        weights = np.array([float(row["metrics"].get("weight_sum") or 0.0) for row in rows])
        metrics: dict[str, float | None] = {}
        for metric_name in (
            "weighted_accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "brier",
        ):
            values = np.array(
                [
                    np.nan
                    if row["metrics"].get(metric_name) is None
                    else float(row["metrics"][metric_name])
                    for row in rows
                ],
                dtype=np.float32,
            )
            metrics[metric_name] = _weighted_nanmean(values, weights)
        metrics["sample_count"] = float(sum(int(row["metrics"]["sample_count"]) for row in rows))
        metrics["weight_sum"] = float(np.sum(weights))
        result[model_type] = metrics
    return result


def minus_audit_comparison(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for model_type in sorted({str(row["model_type"]) for row in prediction_rows}):
        rows = [
            row
            for row in prediction_rows
            if row["model_type"] == model_type and int(row["label_presence_minus_audit"]) in {0, 1}
        ]
        if not rows:
            result[model_type] = {}
            continue
        result[model_type] = binary_metrics(
            np.array([int(row["label_presence_minus_audit"]) for row in rows], dtype=np.int8),
            np.array([float(row["score"]) for row in rows], dtype=np.float32),
            np.array([float(row["sample_weight"]) for row in rows], dtype=np.float32),
            calibration_bins=10,
        )
    return result


def disagreement_subset_analysis(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for model_type in sorted({str(row["model_type"]) for row in prediction_rows}):
        model_rows = [row for row in prediction_rows if row["model_type"] == model_type]
        for subset_name, expected in (
            ("agreement", False),
            ("plus_minus_disagreement", True),
        ):
            rows = [
                row
                for row in model_rows
                if bool(row["plus_minus_disagreement"]) is expected
            ]
            key = f"{model_type}:{subset_name}"
            if not rows:
                result[key] = {"sample_count": 0, "mean_score": None, "candidate_fraction": None}
                continue
            result[key] = {
                "sample_count": len(rows),
                "mean_score": float(np.mean([float(row["score"]) for row in rows])),
                "candidate_fraction": float(
                    np.mean([int(row["label_presence_plus"]) == 1 for row in rows])
                ),
                "mean_sample_weight": float(
                    np.mean([float(row["sample_weight"]) for row in rows])
                ),
            }
    return result


def summarize_coefficients(
    coefficients: dict[str, list[np.ndarray]],
    feature_names: list[str],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for model_type, values in coefficients.items():
        if not values:
            continue
        matrix = np.vstack(values)
        mean_coef = np.mean(matrix[:, 1:], axis=0)
        for name, value in zip(feature_names, mean_coef, strict=True):
            result[f"{model_type}:{name}"] = {
                "mean_coefficient": float(value),
                "mean_abs_coefficient": float(abs(value)),
            }
    return result


def write_simple_baseline_outputs(
    report: SimpleBaselineReport,
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
    output_report_md.write_text(format_simple_baseline_markdown(report), encoding="utf-8")
    _write_prediction_csv(prediction_rows, output_csv)


def format_simple_baseline_markdown(report: SimpleBaselineReport) -> str:
    data = report.to_dict()
    lines = [
        "# MVP-4B Simple Baseline Sanity Report",
        "",
        f"- Version: {data['report_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Allowed scope: {data['allowed_scope']}",
        f"- Backend: {data['model_backend']}",
        f"- Selected samples: {data['sample_counts']['selected_samples']}",
        f"- Production training: {data['production_training']}",
        f"- No final labels: {data['no_final_labels']}",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for model_type, metrics in data["aggregate_metrics"].items():
        lines.append(
            f"- {model_type}: balanced_accuracy={metrics.get('balanced_accuracy')}, "
            f"f1={metrics.get('f1')}, weighted_accuracy={metrics.get('weighted_accuracy')}"
        )
    lines.extend(["", "## Permutation Check", ""])
    for model_type, check in data["permutation_check"].items():
        lines.append(
            f"- {model_type}: margin={check.get('balanced_accuracy_margin')}, "
            f"passes_margin={check.get('passes_margin')}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _run_fold(
    *,
    prepared: dict[str, Any],
    fold_index: int,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    model_type: str,
    baseline_config: BaselineConfig,
    feature_names: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    x_train_raw = prepared["features"][train_mask]
    x_val_raw = prepared["features"][validation_mask]
    y_train = prepared["label"][train_mask]
    y_val = prepared["label"][validation_mask]
    w_train = prepared["sample_weight"][train_mask]
    w_val = prepared["sample_weight"][validation_mask]
    if x_train_raw.shape[0] == 0 or x_val_raw.shape[0] == 0:
        return {
            "warnings": warnings,
            "errors": [f"{model_type} fold {fold_index}: empty train or validation split."],
            "fold_metrics": [],
            "prediction_rows": [],
            "coefficient": None,
        }
    x_train, x_val = _standardize_by_train(x_train_raw, x_val_raw, w_train)
    coef = _fit_model(
        model_type,
        x_train,
        y_train,
        w_train,
        baseline_config=baseline_config,
    )
    score = predict_scores(x_val, coef, model_type=model_type)
    metrics = binary_metrics(
        y_val,
        score,
        w_val,
        calibration_bins=baseline_config.calibration_bins,
    )
    fold_metrics = [
        {
            "model_type": model_type,
            "fold_index": fold_index,
            "permutation": False,
            "metrics": metrics,
        }
    ]
    prediction_rows = _prediction_rows(
        prepared=prepared,
        validation_mask=validation_mask,
        fold_index=fold_index,
        model_type=model_type,
        score=score,
    )

    if baseline_config.permutation_check:
        permuted = np.asarray(y_train, dtype=np.int8).copy()
        rng.shuffle(permuted)
        permutation_coef = _fit_model(
            model_type,
            x_train,
            permuted,
            w_train,
            baseline_config=baseline_config,
        )
        permutation_score = predict_scores(x_val, permutation_coef, model_type=model_type)
        permutation_metrics = binary_metrics(
            y_val,
            permutation_score,
            w_val,
            calibration_bins=baseline_config.calibration_bins,
        )
        fold_metrics.append(
            {
                "model_type": model_type,
                "fold_index": fold_index,
                "permutation": True,
                "metrics": permutation_metrics,
            }
        )

    return {
        "warnings": warnings,
        "errors": errors,
        "fold_metrics": fold_metrics,
        "prediction_rows": prediction_rows,
        "coefficient": coef,
    }


def _fit_model(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    baseline_config: BaselineConfig,
) -> np.ndarray:
    if model_type == "logistic_regression":
        return fit_logistic_regression(
            x_train,
            y_train,
            sample_weight,
            max_iterations=baseline_config.max_iterations,
            learning_rate=baseline_config.learning_rate,
            l2_penalty=baseline_config.l2_penalty,
        )
    if model_type == "linear_probe":
        return fit_linear_probe(
            x_train,
            y_train,
            sample_weight,
            l2_penalty=baseline_config.l2_penalty,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _prediction_rows(
    *,
    prepared: dict[str, Any],
    validation_mask: np.ndarray,
    fold_index: int,
    model_type: str,
    score: np.ndarray,
) -> list[dict[str, Any]]:
    selected_indices = np.flatnonzero(validation_mask)
    rows: list[dict[str, Any]] = []
    for local_index, sample_index in enumerate(selected_indices.tolist()):
        rows.append(
            {
                "csv_version": MVP4B_SIMPLE_BASELINE_CSV_VERSION,
                "model_type": model_type,
                "fold_index": fold_index,
                "sample_id": int(prepared["sample_id"][sample_index]),
                "depth": float(prepared["depth"][sample_index]),
                "side_index": int(prepared["side_index"][sample_index]),
                "label_presence_plus": int(prepared["label"][sample_index]),
                "label_presence_minus_audit": int(prepared["minus_label"][sample_index]),
                "plus_minus_disagreement": bool(
                    prepared["plus_minus_disagreement"][sample_index]
                ),
                "sample_weight": float(prepared["sample_weight"][sample_index]),
                "score": float(score[local_index]),
                "prediction": int(score[local_index] >= 0.5),
            }
        )
    return rows


def _standardize_by_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = _normalized_weights(sample_weight)
    mean = np.sum(x_train * weights[:, None], axis=0)
    centered = x_train - mean
    variance = np.sum((centered**2) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-6))
    return (
        ((x_train - mean) / scale).astype(np.float32),
        ((x_val - mean) / scale).astype(np.float32),
    )


def _permutation_check(
    aggregate: dict[str, dict[str, float | None]],
    permutation_aggregate: dict[str, dict[str, float | None]],
    config: BaselineConfig,
) -> dict[str, dict[str, float | bool | None]]:
    result: dict[str, dict[str, float | bool | None]] = {}
    for model_type in config.model_types:
        real = _as_float(_as_dict(aggregate.get(model_type)).get("balanced_accuracy"))
        permuted = _as_float(
            _as_dict(permutation_aggregate.get(model_type)).get("balanced_accuracy")
        )
        margin = None if real is None or permuted is None else real - permuted
        result[model_type] = {
            "real_balanced_accuracy": real,
            "permutation_balanced_accuracy": permuted,
            "balanced_accuracy_margin": margin,
            "required_margin": config.min_permutation_balanced_accuracy_margin,
            "passes_margin": (
                bool(margin >= config.min_permutation_balanced_accuracy_margin)
                if margin is not None
                else False
            ),
        }
    return result


def _leakage_suspected(
    aggregate: dict[str, dict[str, float | None]],
    config: BaselineConfig,
) -> bool:
    for metrics in aggregate.values():
        for key in ("weighted_accuracy", "balanced_accuracy", "f1"):
            value = metrics.get(key)
            if value is not None and value >= config.suspicious_metric_threshold:
                return True
    return False


def _empty_error_report(
    *,
    baseline_config: BaselineConfig,
    inputs: dict[str, str],
    prepared: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> SimpleBaselineReport:
    return SimpleBaselineReport(
        report_version=MVP4B_SIMPLE_BASELINE_REPORT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        model_backend="numpy_fallback",
        allowed_scope=baseline_config.allowed_scope,
        sample_counts={
            "total_samples": prepared.get("total_count"),
            "selected_samples": prepared.get("selected_count"),
            "excluded_samples": None,
            "excluded_large_depth_match_error": prepared.get("excluded_large_depth_match_error"),
            "excluded_plus_minus_disagreement": prepared.get("excluded_plus_minus_disagreement"),
            "excluded_zero_or_invalid_weight": prepared.get("excluded_zero_or_invalid_weight"),
        },
        class_balance={
            "candidate_count": prepared.get("candidate_count"),
            "non_candidate_count": prepared.get("non_candidate_count"),
            "candidate_fraction": None,
        },
        split={},
        fold_metrics=[],
        aggregate_metrics={},
        permutation_aggregate_metrics={},
        permutation_check={},
        minus_audit_comparison={},
        disagreement_subset_analysis={},
        coefficient_summary={},
        output_csv_version=MVP4B_SIMPLE_BASELINE_CSV_VERSION,
        sanity_model_training_performed=False,
        production_training=False,
        no_final_labels=True,
        no_deep_learning=True,
        no_stc=True,
        no_apes=True,
        no_production_model=True,
        leakage_suspected=False,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "production training",
            "production inference",
            "deep learning",
            "STC",
            "APES",
            "final label generation",
            "model weight export",
            "MVP-4C",
            "MVP-5",
        ],
    )


def _empty_metrics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "weight_sum": 0.0,
        "weighted_accuracy": None,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "candidate_fraction": None,
        "mean_score": None,
        "brier": None,
        "calibration_summary": [],
    }


def _write_prediction_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "csv_version",
        "model_type",
        "fold_index",
        "sample_id",
        "depth",
        "side_index",
        "label_presence_plus",
        "label_presence_minus_audit",
        "plus_minus_disagreement",
        "sample_weight",
        "score",
        "prediction",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _add_intercept(values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(values.shape[0], dtype=np.float32), values]).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float32)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / max(weights.size, 1), dtype=np.float32)
    return (weights / total).astype(np.float32)


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float | None:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return None
    return float(np.average(values[finite], weights=weights[finite]))


def _safe_div(numerator: float | int, denominator: float | int) -> float | None:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return None
    return float(numerator) / denominator_float


def _safe_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _safe_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return _safe_div(2.0 * precision * recall, precision + recall)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
