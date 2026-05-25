from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.baseline_schema import (
    BaselineConfig,
    load_baseline_config,
)
from cement_channel.training.depth_splits import make_depth_block_splits
from cement_channel.training.simple_baseline import (
    binary_metrics,
    prepare_baseline_samples,
)
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

BASELINE_FAILURE_DIAGNOSTICS_VERSION = "baseline_failure_diagnostics_v001"

DIAGNOSTIC_FIGURES = {
    "prediction_score_distribution": "01_prediction_score_distribution.png",
    "fold_weight_balance": "02_fold_weight_balance.png",
    "filter_strategy_effects": "03_filter_strategy_effects.png",
    "feature_effect_comparison": "04_feature_effect_comparison.png",
}


@dataclass(frozen=True)
class BaselineFailureDiagnosticsReport:
    diagnostics_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figures: dict[str, str]
    no_go_confirmed: bool
    no_go_reason_classes: list[str]
    answers: dict[str, Any]
    class_balance_by_fold: list[dict[str, Any]]
    sample_weight_by_fold: list[dict[str, Any]]
    prediction_distribution: dict[str, dict[str, Any]]
    real_vs_permutation_feature_distribution: dict[str, dict[str, float | int | None]]
    high_confidence_disagreement: dict[str, float | int | None]
    filter_strategy_summary: dict[str, dict[str, Any]]
    preprocessing_compression: dict[str, Any]
    late_over_early_threshold: dict[str, Any]
    recommendations: list[str]
    no_final_labels: bool
    no_deep_learning: bool
    no_stc: bool
    no_apes: bool
    no_mvp4c: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_baseline_failure_from_config(
    *,
    sample_table_npz: Path | str,
    simple_baseline_report_json: Path | str,
    baseline_config_path: Path | str,
    output_dir: Path | str,
    simple_baseline_csv: Path | str | None = None,
    overwrite: bool = False,
) -> BaselineFailureDiagnosticsReport:
    config = load_baseline_config(baseline_config_path)
    return diagnose_baseline_failure(
        sample_table_npz=sample_table_npz,
        simple_baseline_report_json=simple_baseline_report_json,
        baseline_config=config,
        output_dir=output_dir,
        baseline_config_path=baseline_config_path,
        simple_baseline_csv=simple_baseline_csv,
        overwrite=overwrite,
    )


def diagnose_baseline_failure(
    *,
    sample_table_npz: Path | str,
    simple_baseline_report_json: Path | str,
    baseline_config: BaselineConfig,
    output_dir: Path | str,
    baseline_config_path: Path | str | None = None,
    simple_baseline_csv: Path | str | None = None,
    overwrite: bool = False,
) -> BaselineFailureDiagnosticsReport:
    arrays = _load_npz(sample_table_npz)
    report = _read_json(Path(simple_baseline_report_json))
    csv_path = _default_prediction_csv(simple_baseline_report_json, simple_baseline_csv)
    prediction_rows = _read_prediction_rows(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_baseline_samples(arrays, baseline_config)
    warnings = list(prepared["warnings"])
    errors = list(prepared["errors"])
    split_plan = make_depth_block_splits(
        depth=prepared["depth"],
        labels=prepared["label"],
        n_splits=baseline_config.n_splits,
        min_gap_ft=baseline_config.min_gap_ft,
        block_size_ft=baseline_config.depth_block_size_ft,
        min_samples_per_class=baseline_config.min_samples_per_class_per_fold,
    )
    warnings.extend(split_plan.warnings)
    class_balance = [fold.summary.to_dict() for fold in split_plan.folds]
    weight_balance = fold_weight_balance(prepared, split_plan.folds)
    prediction_distribution = prediction_distribution_summary(prediction_rows)
    feature_distribution = real_vs_permutation_feature_distribution(
        prepared,
        seed=baseline_config.permutation_seed,
    )
    high_confidence_disagreement = high_confidence_disagreement_summary(prepared)
    filter_summary = filter_strategy_summary(arrays, baseline_config)
    preprocessing_compression = preprocessing_compression_summary(arrays, prepared)
    late_threshold = late_over_early_threshold_diagnostic(arrays, prepared)
    no_go_confirmed = _no_go_confirmed(report)
    reason_classes = classify_no_go_reasons(
        report=report,
        prepared=prepared,
        weight_balance=weight_balance,
        prediction_distribution=prediction_distribution,
        feature_distribution=feature_distribution,
        filter_summary=filter_summary,
        late_threshold=late_threshold,
    )
    answers = diagnostic_answers(
        prepared=prepared,
        weight_balance=weight_balance,
        prediction_distribution=prediction_distribution,
        filter_summary=filter_summary,
        reason_classes=reason_classes,
    )
    recommendations = recommendations_from_reasons(reason_classes, filter_summary)
    figures = {key: output / filename for key, filename in DIAGNOSTIC_FIGURES.items()}
    _save_prediction_score_distribution(
        prediction_rows,
        figures["prediction_score_distribution"],
        overwrite=overwrite,
    )
    _save_fold_weight_balance(
        weight_balance,
        figures["fold_weight_balance"],
        overwrite=overwrite,
    )
    _save_filter_strategy_effects(
        filter_summary,
        figures["filter_strategy_effects"],
        overwrite=overwrite,
    )
    _save_feature_effect_comparison(
        feature_distribution,
        figures["feature_effect_comparison"],
        overwrite=overwrite,
    )
    return BaselineFailureDiagnosticsReport(
        diagnostics_version=BASELINE_FAILURE_DIAGNOSTICS_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "sample_table_npz": str(sample_table_npz),
            "simple_baseline_report_json": str(simple_baseline_report_json),
            "simple_baseline_csv": str(csv_path),
            "baseline_config_path": str(baseline_config_path) if baseline_config_path else "",
        },
        output_dir=str(output),
        figures={key: str(path) for key, path in figures.items()},
        no_go_confirmed=no_go_confirmed,
        no_go_reason_classes=reason_classes,
        answers=answers,
        class_balance_by_fold=class_balance,
        sample_weight_by_fold=weight_balance,
        prediction_distribution=prediction_distribution,
        real_vs_permutation_feature_distribution=feature_distribution,
        high_confidence_disagreement=high_confidence_disagreement,
        filter_strategy_summary=filter_summary,
        preprocessing_compression=preprocessing_compression,
        late_over_early_threshold=late_threshold,
        recommendations=recommendations,
        no_final_labels=True,
        no_deep_learning=True,
        no_stc=True,
        no_apes=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "complex model training",
            "deep learning",
            "STC",
            "APES",
            "MVP-4C",
            "final label generation",
            "ground truth claim",
        ],
    )


def write_baseline_failure_diagnostics_outputs(
    report: BaselineFailureDiagnosticsReport,
    *,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_diagnostics_markdown(report), encoding="utf-8")


def fold_weight_balance(
    prepared: dict[str, Any],
    folds: list[Any],
) -> list[dict[str, Any]]:
    labels = prepared["label"]
    weights = prepared["sample_weight"]
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for split_name, mask in (
            ("train", fold.train_mask),
            ("validation", fold.validation_mask),
        ):
            rows.append(
                _weight_balance_row(
                    fold_index=fold.fold_index,
                    split_name=split_name,
                    labels=labels,
                    weights=weights,
                    mask=mask,
                )
            )
    return rows


def prediction_distribution_summary(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    model_types = sorted({str(row["model_type"]) for row in prediction_rows})
    for model_type in model_types:
        rows = [row for row in prediction_rows if row["model_type"] == model_type]
        scores = np.array([float(row["score"]) for row in rows], dtype=np.float32)
        predictions = np.array([int(row["prediction"]) for row in rows], dtype=np.int8)
        positive_rate = float(np.mean(predictions == 1)) if predictions.size else None
        result[model_type] = {
            "sample_count": len(rows),
            "predicted_positive_rate": positive_rate,
            "score_min": _finite_stat(scores, "min"),
            "score_median": _finite_stat(scores, "median"),
            "score_max": _finite_stat(scores, "max"),
            "degenerate_all_candidate": bool(
                positive_rate is not None and positive_rate >= 0.99
            ),
            "degenerate_all_non_candidate": bool(
                positive_rate is not None and positive_rate <= 0.01
            ),
        }
    return result


def real_vs_permutation_feature_distribution(
    prepared: dict[str, Any],
    *,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(prepared["label"], dtype=np.int8)
    permuted = labels.copy()
    rng.shuffle(permuted)
    features = np.asarray(prepared["features"], dtype=np.float32)
    feature_names = [str(value) for value in prepared["feature_names"]]
    result: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(feature_names):
        values = features[:, index]
        real_effect = standardized_difference(values, labels)
        permutation_effect = standardized_difference(values, permuted)
        result[name] = {
            "candidate_count": int(np.count_nonzero(labels == 1)),
            "non_candidate_count": int(np.count_nonzero(labels == 0)),
            "real_standardized_difference": real_effect,
            "permutation_standardized_difference": permutation_effect,
            "absolute_effect_margin": (
                None
                if real_effect is None or permutation_effect is None
                else abs(real_effect) - abs(permutation_effect)
            ),
        }
    return result


def high_confidence_disagreement_summary(prepared: dict[str, Any]) -> dict[str, float | int | None]:
    disagreement = np.asarray(prepared["plus_minus_disagreement"], dtype=bool)
    labels = np.asarray(prepared["label"], dtype=np.int8)
    weights = np.asarray(prepared["sample_weight"], dtype=np.float32)
    return {
        "selected_samples": int(labels.size),
        "disagreement_count": int(np.count_nonzero(disagreement)),
        "disagreement_fraction": float(np.mean(disagreement)) if labels.size else None,
        "disagreement_candidate_fraction": (
            float(np.mean(labels[disagreement] == 1)) if np.any(disagreement) else None
        ),
        "agreement_candidate_fraction": (
            float(np.mean(labels[~disagreement] == 1)) if np.any(~disagreement) else None
        ),
        "disagreement_weight_fraction": _safe_div(
            float(np.sum(weights[disagreement])),
            float(np.sum(weights)),
        ),
    }


def filter_strategy_summary(
    arrays: dict[str, np.ndarray],
    config: BaselineConfig,
) -> dict[str, dict[str, Any]]:
    base_mask = _base_mask(arrays)
    confidence = np.asarray(arrays["label_confidence_plus"], dtype=np.float32).reshape(-1)
    depth_error = np.abs(np.asarray(arrays["depth_match_error"], dtype=np.float32).reshape(-1))
    orientation = np.asarray(arrays["orientation_confidence"], dtype=np.float32).reshape(-1)
    disagreement = np.asarray(arrays["plus_minus_disagreement"], dtype=bool).reshape(-1)
    q75 = float(np.nanquantile(confidence[base_mask], 0.75)) if np.any(base_mask) else 1.0
    strategies = {
        "baseline_high_confidence": base_mask,
        "exclude_plus_minus_disagreement": base_mask & ~disagreement,
        "min_label_confidence_0p7": base_mask & (confidence >= 0.7),
        "max_depth_match_error_0p25": base_mask & (depth_error <= 0.25),
        "top_confidence_quartile": base_mask & (confidence >= q75),
        "stable_orientation_intervals": base_mask & (orientation >= 0.9),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, mask in strategies.items():
        result[name] = strategy_effect_summary(arrays, mask, config=config)
    baseline_effect = result["baseline_high_confidence"]["max_abs_transformed_effect_size"]
    for values in result.values():
        values["effect_size_delta_vs_baseline"] = _delta(
            values["max_abs_transformed_effect_size"],
            baseline_effect,
        )
    return result


def strategy_effect_summary(
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    config: BaselineConfig,
) -> dict[str, Any]:
    labels = np.asarray(arrays[config.label], dtype=np.int8).reshape(-1)
    weights = np.asarray(arrays[config.sample_weight_source], dtype=np.float32).reshape(-1)
    features = np.asarray(arrays["transformed_features"], dtype=np.float32)
    names = np.asarray(arrays["transformed_feature_names"]).astype(str).tolist()
    selected = mask & np.isin(labels, [0, 1]) & np.all(np.isfinite(features), axis=1)
    effects = {
        name: standardized_difference(features[selected, index], labels[selected])
        for index, name in enumerate(names)
    }
    max_abs = _max_abs_effect(effects)
    late_effect = _first_present(
        effects,
        ["robust_scaled_late_over_early_ratio", "log1p_late_over_early_ratio"],
    )
    selected_weights = weights[selected]
    selected_labels = labels[selected]
    candidate = selected_labels == 1
    non_candidate = selected_labels == 0
    return {
        "sample_count": int(np.count_nonzero(selected)),
        "candidate_count": int(np.count_nonzero(candidate)),
        "non_candidate_count": int(np.count_nonzero(non_candidate)),
        "candidate_fraction": (
            float(np.mean(candidate)) if selected_labels.size else None
        ),
        "weight_sum": float(np.sum(selected_weights)),
        "candidate_weight_fraction": _safe_div(
            float(np.sum(selected_weights[candidate])),
            float(np.sum(selected_weights)),
        ),
        "max_abs_transformed_effect_size": max_abs,
        "late_over_early_effect_size": late_effect,
        "top_effect_features": _top_effects(effects, limit=5),
    }


def preprocessing_compression_summary(
    arrays: dict[str, np.ndarray],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    raw_features = np.asarray(arrays["features"], dtype=np.float32)
    raw_names = np.asarray(arrays["feature_names"]).astype(str).tolist()
    transformed_features = np.asarray(arrays["transformed_features"], dtype=np.float32)
    transformed_names = np.asarray(arrays["transformed_feature_names"]).astype(str).tolist()
    selected_ids = np.asarray(prepared["sample_id"], dtype=np.int64)
    labels = np.asarray(prepared["label"], dtype=np.int8)
    raw_effects = {
        name: standardized_difference(raw_features[selected_ids, index], labels)
        for index, name in enumerate(raw_names)
    }
    transformed_effects = {
        name: standardized_difference(transformed_features[selected_ids, index], labels)
        for index, name in enumerate(transformed_names)
    }
    raw_max = _max_abs_effect(raw_effects)
    transformed_max = _max_abs_effect(transformed_effects)
    return {
        "raw_max_abs_effect_size": raw_max,
        "transformed_max_abs_effect_size": transformed_max,
        "transformed_to_raw_max_effect_ratio": (
            None if raw_max in {None, 0.0} or transformed_max is None else transformed_max / raw_max
        ),
        "possible_over_compression": bool(
            raw_max is not None
            and transformed_max is not None
            and raw_max >= 0.10
            and transformed_max < raw_max * 0.5
        ),
        "raw_top_effect_features": _top_effects(raw_effects, limit=6),
        "transformed_top_effect_features": _top_effects(transformed_effects, limit=6),
    }


def late_over_early_threshold_diagnostic(
    arrays: dict[str, np.ndarray],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    raw_names = np.asarray(arrays["feature_names"]).astype(str).tolist()
    if "late_over_early_ratio" not in raw_names:
        return {"available": False}
    index = raw_names.index("late_over_early_ratio")
    selected_ids = np.asarray(prepared["sample_id"], dtype=np.int64)
    values = np.asarray(arrays["features"], dtype=np.float32)[selected_ids, index]
    labels = np.asarray(prepared["label"], dtype=np.int8)
    weights = np.asarray(prepared["sample_weight"], dtype=np.float32)
    threshold = float(np.nanmedian(values))
    scores = (values >= threshold).astype(np.float32)
    metrics = binary_metrics(labels, scores, weights, calibration_bins=2)
    positive_rate = float(np.mean(scores >= 0.5)) if scores.size else None
    return {
        "available": True,
        "threshold": threshold,
        "predicted_positive_rate": positive_rate,
        "degenerate": bool(
            positive_rate is not None and (positive_rate <= 0.01 or positive_rate >= 0.99)
        ),
        "metrics": metrics,
        "candidate_median": _finite_stat(values[labels == 1], "median"),
        "non_candidate_median": _finite_stat(values[labels == 0], "median"),
        "standardized_difference": standardized_difference(values, labels),
    }


def classify_no_go_reasons(
    *,
    report: dict[str, Any],
    prepared: dict[str, Any],
    weight_balance: list[dict[str, Any]],
    prediction_distribution: dict[str, dict[str, Any]],
    feature_distribution: dict[str, dict[str, float | int | None]],
    filter_summary: dict[str, dict[str, Any]],
    late_threshold: dict[str, Any],
) -> list[str]:
    reasons: set[str] = set()
    if not _permutation_passes(report):
        reasons.add("insufficient_high_confidence_signal")
    if any(
        values["degenerate_all_candidate"] or values["degenerate_all_non_candidate"]
        for values in prediction_distribution.values()
    ):
        reasons.add("class_weight_failure")
    overall_weight_fraction = _candidate_weight_fraction(prepared)
    candidate_fraction = _safe_div(prepared["candidate_count"], prepared["selected_count"])
    if (
        overall_weight_fraction is not None
        and candidate_fraction is not None
        and abs(overall_weight_fraction - candidate_fraction) > 0.25
    ):
        reasons.add("sample_weight_failure")
    if (prepared["candidate_count"] <= 0) or (prepared["non_candidate_count"] <= 0):
        reasons.add("label_noise")
    disagreement = high_confidence_disagreement_summary(prepared)
    if (disagreement.get("disagreement_fraction") or 0.0) > 0.25:
        reasons.add("label_noise")
    if _max_abs_effect(
        {
            name: values["real_standardized_difference"]
            for name, values in feature_distribution.items()
        }
    ) is not None and (
        _max_abs_effect(
            {
                name: values["real_standardized_difference"]
                for name, values in feature_distribution.items()
            }
        )
        or 0.0
    ) < 0.20:
        reasons.add("feature_weakness")
    if _fold_weight_shift(weight_balance) > 0.30:
        reasons.add("split_distribution_shift")
    if _depth_split_invalid(report):
        reasons.add("depth_leakage_or_block_issue")
    late_metrics = _as_dict(late_threshold.get("metrics"))
    late_balanced = _as_float(late_metrics.get("balanced_accuracy"))
    if late_balanced is not None and late_balanced <= 0.55:
        reasons.add("feature_weakness")
    if not _filter_improves_signal(filter_summary):
        reasons.add("insufficient_high_confidence_signal")
    return sorted(reasons)


def diagnostic_answers(
    *,
    prepared: dict[str, Any],
    weight_balance: list[dict[str, Any]],
    prediction_distribution: dict[str, dict[str, Any]],
    filter_summary: dict[str, dict[str, Any]],
    reason_classes: list[str],
) -> dict[str, Any]:
    degenerate_models = [
        model
        for model, values in prediction_distribution.items()
        if values["degenerate_all_candidate"] or values["degenerate_all_non_candidate"]
    ]
    weight_fraction = _candidate_weight_fraction(prepared)
    count_fraction = _safe_div(prepared["candidate_count"], prepared["selected_count"])
    return {
        "model_degenerated_to_single_class": bool(degenerate_models),
        "degenerate_models": degenerate_models,
        "sample_weight_causes_effective_class_imbalance": bool(
            weight_fraction is not None
            and count_fraction is not None
            and abs(weight_fraction - count_fraction) > 0.25
        ),
        "candidate_count_fraction": count_fraction,
        "candidate_weight_fraction": weight_fraction,
        "depth_block_split_has_fold_imbalance": bool(_fold_weight_shift(weight_balance) > 0.30),
        "remove_disagreement_improves_signal": _strategy_improved(
            filter_summary,
            "exclude_plus_minus_disagreement",
        ),
        "higher_confidence_improves_signal": bool(
            _strategy_improved(filter_summary, "min_label_confidence_0p7")
            or _strategy_improved(filter_summary, "top_confidence_quartile")
        ),
        "should_return_to_label_or_feature_design": bool(
            "label_noise" in reason_classes
            or "feature_weakness" in reason_classes
            or "sample_weight_failure" in reason_classes
        ),
    }


def recommendations_from_reasons(
    reason_classes: list[str],
    filter_summary: dict[str, dict[str, Any]],
) -> list[str]:
    recommendations = ["no-go for modeling"]
    if "label_noise" in reason_classes or "sample_weight_failure" in reason_classes:
        recommendations.append("revise label sampling")
    if (
        "feature_weakness" in reason_classes
        or "insufficient_high_confidence_signal" in reason_classes
    ):
        recommendations.append("add receiver-level features")
        recommendations.append("add side-normalized features")
    if not _filter_improves_signal(filter_summary):
        recommendations.append("revise feature preprocessing")
    return sorted(set(recommendations))


def format_diagnostics_markdown(report: BaselineFailureDiagnosticsReport) -> str:
    data = report.to_dict()
    lines = [
        "# MVP-4B Baseline Failure Diagnostics",
        "",
        f"- Version: {data['diagnostics_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- No-go confirmed: {data['no_go_confirmed']}",
        f"- No-go reason classes: {', '.join(data['no_go_reason_classes'])}",
        f"- No MVP-4C: {data['no_mvp4c']}",
        "",
        "## Answers",
        "",
    ]
    for key, value in data["answers"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(_message_lines(data["recommendations"]))
    lines.extend(["", "## Prediction Degeneration", ""])
    for model, values in data["prediction_distribution"].items():
        lines.append(
            f"- {model}: positive_rate={values['predicted_positive_rate']}, "
            f"score_median={values['score_median']}, "
            f"all_candidate={values['degenerate_all_candidate']}, "
            f"all_non_candidate={values['degenerate_all_non_candidate']}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _weight_balance_row(
    *,
    fold_index: int,
    split_name: str,
    labels: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_labels = labels[mask]
    selected_weights = weights[mask]
    candidate = selected_labels == 1
    non_candidate = selected_labels == 0
    weight_sum = float(np.sum(selected_weights))
    return {
        "fold_index": int(fold_index),
        "split": split_name,
        "sample_count": int(np.count_nonzero(mask)),
        "candidate_count": int(np.count_nonzero(candidate)),
        "non_candidate_count": int(np.count_nonzero(non_candidate)),
        "weight_sum": weight_sum,
        "candidate_weight_sum": float(np.sum(selected_weights[candidate])),
        "non_candidate_weight_sum": float(np.sum(selected_weights[non_candidate])),
        "candidate_weight_fraction": _safe_div(
            float(np.sum(selected_weights[candidate])),
            weight_sum,
        ),
    }


def standardized_difference(values: np.ndarray, labels: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    label_values = np.asarray(labels, dtype=np.int8).reshape(-1)
    candidate = array[(label_values == 1) & np.isfinite(array)]
    non_candidate = array[(label_values == 0) & np.isfinite(array)]
    if candidate.size < 2 or non_candidate.size < 2:
        return None
    pooled = np.sqrt((np.var(candidate, ddof=1) + np.var(non_candidate, ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0.0:
        return None
    return float((np.mean(candidate) - np.mean(non_candidate)) / pooled)


def _base_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    labels = np.asarray(arrays["label_presence_plus"], dtype=np.int8).reshape(-1)
    valid = np.asarray(arrays["valid_for_azimuthal_validation"], dtype=bool).reshape(-1)
    large_depth = np.asarray(arrays["exclude_large_depth_match_error"], dtype=bool).reshape(-1)
    weights = np.asarray(arrays["sample_weight"], dtype=np.float32).reshape(-1)
    features = np.asarray(arrays["transformed_features"], dtype=np.float32)
    return (
        np.isin(labels, [0, 1])
        & valid
        & ~large_depth
        & np.isfinite(weights)
        & (weights > 0.0)
        & np.all(np.isfinite(features), axis=1)
    )


def _save_prediction_score_distribution(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for model_type in sorted({str(row["model_type"]) for row in rows}):
        scores = np.array(
            [float(row["score"]) for row in rows if row["model_type"] == model_type],
            dtype=np.float32,
        )
        ax.hist(scores[np.isfinite(scores)], bins=50, alpha=0.5, label=model_type)
    ax.set_xlabel("Predicted sanity score")
    ax.set_ylabel("Count")
    ax.set_title("Prediction score distribution")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_fold_weight_balance(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    labels = [f"f{row['fold_index']} {row['split']}" for row in rows]
    values = [row["candidate_weight_fraction"] or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.8), 5), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color="tab:blue")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(values)), labels=labels, rotation=35, ha="right")
    ax.set_ylabel("Candidate weight fraction")
    ax.set_title("Fold sample weight balance")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_filter_strategy_effects(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    names = list(summary)
    values = [summary[name]["max_abs_transformed_effect_size"] or 0.0 for name in names]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(np.arange(len(names)), values, color="tab:green")
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=35, ha="right")
    ax.set_ylabel("Max abs standardized difference")
    ax.set_title("Filter strategy signal diagnostics")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_feature_effect_comparison(
    summary: dict[str, dict[str, float | int | None]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    rows = sorted(
        [
            (
                name,
                values["real_standardized_difference"] or 0.0,
                values["permutation_standardized_difference"] or 0.0,
            )
            for name, values in summary.items()
        ],
        key=lambda row: abs(row[1]),
        reverse=True,
    )[:12]
    names = [row[0] for row in rows]
    real = [row[1] for row in rows]
    permuted = [row[2] for row in rows]
    x_values = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x_values - 0.18, real, width=0.36, label="real labels")
    ax.bar(x_values + 0.18, permuted, width=0.36, label="permuted labels")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x_values, labels=names, rotation=35, ha="right")
    ax.set_ylabel("Standardized difference")
    ax.set_title("Feature distribution: real vs permutation")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _default_prediction_csv(
    report_json: Path | str,
    csv_path: Path | str | None,
) -> Path:
    if csv_path is not None:
        return Path(csv_path)
    return Path(report_json).with_name("simple_baseline_v001.csv")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _no_go_confirmed(report: dict[str, Any]) -> bool:
    return not _permutation_passes(report)


def _permutation_passes(report: dict[str, Any]) -> bool:
    checks = _as_dict(report.get("permutation_check"))
    if not checks:
        return False
    return all(_as_dict(value).get("passes_margin") is True for value in checks.values())


def _depth_split_invalid(report: dict[str, Any]) -> bool:
    split = _as_dict(report.get("split"))
    return split.get("method") != "depth_block_group_split" or not _as_list(split.get("folds"))


def _fold_weight_shift(rows: list[dict[str, Any]]) -> float:
    values = [
        row["candidate_weight_fraction"]
        for row in rows
        if row["split"] == "validation" and row["candidate_weight_fraction"] is not None
    ]
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _candidate_weight_fraction(prepared: dict[str, Any]) -> float | None:
    labels = np.asarray(prepared["label"], dtype=np.int8)
    weights = np.asarray(prepared["sample_weight"], dtype=np.float32)
    return _safe_div(float(np.sum(weights[labels == 1])), float(np.sum(weights)))


def _filter_improves_signal(summary: dict[str, dict[str, Any]]) -> bool:
    baseline = summary.get("baseline_high_confidence", {}).get(
        "max_abs_transformed_effect_size"
    )
    if baseline is None:
        return False
    for name, values in summary.items():
        if name == "baseline_high_confidence":
            continue
        effect = values.get("max_abs_transformed_effect_size")
        if effect is not None and effect > baseline + 0.05:
            return True
    return False


def _strategy_improved(summary: dict[str, dict[str, Any]], name: str) -> bool:
    baseline = summary.get("baseline_high_confidence", {}).get(
        "max_abs_transformed_effect_size"
    )
    effect = summary.get(name, {}).get("max_abs_transformed_effect_size")
    return bool(baseline is not None and effect is not None and effect > baseline + 0.05)


def _finite_stat(values: np.ndarray, stat: str) -> float | None:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    if stat == "min":
        return float(np.min(finite))
    if stat == "max":
        return float(np.max(finite))
    if stat == "median":
        return float(np.median(finite))
    raise ValueError(f"Unsupported stat: {stat}")


def _top_effects(
    effects: dict[str, float | None],
    *,
    limit: int,
) -> list[dict[str, float | str | None]]:
    rows = sorted(
        effects.items(),
        key=lambda item: -1.0 if item[1] is None else abs(float(item[1])),
        reverse=True,
    )
    return [
        {"feature": name, "standardized_difference": value}
        for name, value in rows[:limit]
    ]


def _max_abs_effect(effects: dict[str, float | None]) -> float | None:
    values = [abs(float(value)) for value in effects.values() if value is not None]
    return max(values) if values else None


def _first_present(effects: dict[str, float | None], names: list[str]) -> float | None:
    for name in names:
        if name in effects:
            return effects[name]
    return None


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _safe_div(numerator: float | int, denominator: float | int) -> float | None:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return None
    return float(numerator) / denominator_float


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
