from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from cement_channel.evaluation.correlation_schema import (
    MVP4A_CORRELATION_VERSION,
    CorrelationConfig,
    load_correlation_config,
)


@dataclass(frozen=True)
class XsiCastCorrelationReport:
    correlation_version: str
    generated_at: str
    inputs: dict[str, str]
    label_source: str
    primary_label: str
    audit_label: str
    no_model_training: bool
    no_final_labels: bool
    subset_counts: dict[str, dict[str, int]]
    top_primary_effects: list[dict[str, Any]]
    gate_observations: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_xsi_cast_correlation_from_config(
    *,
    label_samples_npz: Path | str,
    basic_features_npz: Path | str,
    correlation_config_path: Path | str,
) -> tuple[XsiCastCorrelationReport, list[dict[str, Any]]]:
    config = load_correlation_config(correlation_config_path)
    return evaluate_xsi_cast_correlation(
        label_samples_npz=label_samples_npz,
        basic_features_npz=basic_features_npz,
        correlation_config=config,
        correlation_config_path=correlation_config_path,
    )


def evaluate_xsi_cast_correlation(
    *,
    label_samples_npz: Path | str,
    basic_features_npz: Path | str,
    correlation_config: CorrelationConfig,
    correlation_config_path: Path | str | None = None,
) -> tuple[XsiCastCorrelationReport, list[dict[str, Any]]]:
    labels = _load_npz(label_samples_npz)
    features = _load_npz(basic_features_npz)
    rows, summary = evaluate_xsi_cast_correlation_from_arrays(
        label_arrays=labels,
        feature_arrays=features,
        correlation_config=correlation_config,
    )
    report = XsiCastCorrelationReport(
        correlation_version=MVP4A_CORRELATION_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "label_samples_npz": str(label_samples_npz),
            "basic_features_npz": str(basic_features_npz),
            "correlation_config_path": (
                str(correlation_config_path) if correlation_config_path is not None else ""
            ),
        },
        label_source=correlation_config.label_source,
        primary_label=correlation_config.primary_label,
        audit_label=correlation_config.audit_label,
        no_model_training=True,
        no_final_labels=True,
        subset_counts=summary["subset_counts"],
        top_primary_effects=summary["top_primary_effects"],
        gate_observations=summary["gate_observations"],
        warnings=summary["warnings"],
        errors=summary["errors"],
        not_performed=[
            "model training",
            "train/test split",
            "AUC as model performance",
            "STC",
            "APES",
            "final label generation",
            "MVP-4B feature engineering",
        ],
    )
    return report, rows


def evaluate_xsi_cast_correlation_from_arrays(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    correlation_config: CorrelationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    warnings: list[str] = []
    errors: list[str] = []
    plus = np.asarray(label_arrays["label_presence_plus"], dtype=np.int8)
    minus = np.asarray(label_arrays["label_presence_minus_audit"], dtype=np.int8)
    severity = np.asarray(label_arrays["label_severity_plus"], dtype=np.int8)
    confidence_plus = np.asarray(label_arrays["label_confidence_plus"], dtype=np.float32)
    confidence_minus = np.asarray(label_arrays["label_confidence_minus_audit"], dtype=np.float32)
    valid_azimuthal = np.asarray(label_arrays["valid_for_azimuthal_validation"], dtype=bool)
    valid_non_azimuthal = np.asarray(label_arrays["valid_for_non_azimuthal_summary"], dtype=bool)
    disagreement = np.asarray(label_arrays["plus_minus_disagreement"], dtype=bool)
    feature_values = np.asarray(feature_arrays["xsi_basic_features_by_side"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["feature_names"]).astype(str).tolist()
    depth_count = min(plus.shape[0], feature_values.shape[0])
    if plus.shape[0] != feature_values.shape[0]:
        warnings.append(
            "Label sample depth count and feature depth count differ; using common prefix "
            f"{depth_count}."
        )
    plus = plus[:depth_count]
    minus = minus[:depth_count]
    severity = severity[:depth_count]
    confidence_plus = confidence_plus[:depth_count]
    confidence_minus = confidence_minus[:depth_count]
    valid_azimuthal = valid_azimuthal[:depth_count]
    valid_non_azimuthal = valid_non_azimuthal[:depth_count]
    disagreement = disagreement[:depth_count]
    feature_values = feature_values[:depth_count]
    no_final_labels = bool(np.asarray(label_arrays.get("no_final_labels", False)).reshape(()))
    if not no_final_labels:
        errors.append("label sample artifact does not set no_final_labels=true.")
    if not bool(np.asarray(feature_arrays.get("no_model_training", False)).reshape(())):
        errors.append("feature artifact does not set no_model_training=true.")
    if not bool(np.asarray(feature_arrays.get("no_stc", False)).reshape(())):
        errors.append("feature artifact does not set no_stc=true.")
    if not bool(np.asarray(feature_arrays.get("no_apes", False)).reshape(())):
        errors.append("feature artifact does not set no_apes=true.")

    subsets = {
        "all_known": plus >= 0,
        "high_confidence": valid_azimuthal & (plus >= 0),
        "low_confidence": valid_non_azimuthal & ~valid_azimuthal & (plus >= 0),
        "plus_minus_disagreement": disagreement & (plus >= 0),
    }
    minus_subsets = {
        "all_known_minus_audit": minus >= 0,
        "high_confidence_minus_audit": valid_azimuthal & (minus >= 0),
    }
    rows: list[dict[str, Any]] = []
    for subset_name, subset_mask in subsets.items():
        rows.extend(
            _rows_for_label_set(
                feature_values=feature_values,
                feature_names=feature_names,
                presence=plus,
                severity=severity,
                confidence=confidence_plus,
                subset_mask=subset_mask,
                label_convention="plus_primary",
                subset_name=subset_name,
            )
        )
    for subset_name, subset_mask in minus_subsets.items():
        rows.extend(
            _rows_for_label_set(
                feature_values=feature_values,
                feature_names=feature_names,
                presence=minus,
                severity=severity,
                confidence=confidence_minus,
                subset_mask=subset_mask,
                label_convention="minus_audit",
                subset_name=subset_name,
            )
        )
    subset_counts = {
        name: _subset_counts(plus if "minus" not in name else minus, mask)
        for name, mask in {**subsets, **minus_subsets}.items()
    }
    min_samples = correlation_config.high_confidence_min_samples_per_class
    high = subset_counts["high_confidence"]
    if high["candidate"] < min_samples or high["non_candidate"] < min_samples:
        warnings.append(
            "High-confidence subset has too few samples for one or both classes: "
            f"candidate={high['candidate']}, non_candidate={high['non_candidate']}, "
            f"minimum={min_samples}."
        )
    top_primary = _top_primary_effects(rows)
    has_interpretable_signal = any(
        abs(_as_float(row.get("point_biserial_effect_size")) or 0.0)
        >= correlation_config.min_interpretable_abs_effect_size
        or abs(_as_float(row.get("weighted_difference_fraction")) or 0.0)
        >= correlation_config.min_interpretable_weighted_difference_fraction
        for row in rows
        if row["label_convention"] == "plus_primary" and row["subset"] == "high_confidence"
    )
    if not has_interpretable_signal:
        warnings.append("High-confidence plus-primary subset shows no interpretable separation.")
    summary = {
        "subset_counts": subset_counts,
        "top_primary_effects": top_primary,
        "gate_observations": {
            "high_confidence_subset_exists": (
                high["candidate"] >= min_samples and high["non_candidate"] >= min_samples
            ),
            "interpretable_signal_separation": has_interpretable_signal,
            "low_confidence_policy_respected": True,
            "no_model_training": True,
            "no_final_labels": no_final_labels,
        },
        "warnings": warnings,
        "errors": errors,
    }
    return rows, summary


def write_xsi_cast_correlation_outputs(
    report: XsiCastCorrelationReport,
    rows: list[dict[str, Any]],
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
    output_report_md.write_text(format_xsi_cast_correlation_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_xsi_cast_correlation_markdown(report: XsiCastCorrelationReport) -> str:
    data = report.to_dict()
    lines = [
        "# XSI-CAST Weak-Label Correlation Report",
        "",
        f"- Version: {data['correlation_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Label source: {data['label_source']}",
        f"- Primary label: {data['primary_label']}",
        f"- Audit label: {data['audit_label']}",
        f"- No model training: {data['no_model_training']}",
        f"- No final labels: {data['no_final_labels']}",
        "",
        "## Gate Observations",
        "",
    ]
    for key, value in data["gate_observations"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Subset Counts", ""])
    for subset, counts in data["subset_counts"].items():
        lines.append(
            f"- {subset}: candidate={counts['candidate']}, "
            f"non_candidate={counts['non_candidate']}, known={counts['known']}"
        )
    lines.extend(["", "## Top Primary Effects", ""])
    if data["top_primary_effects"]:
        for row in data["top_primary_effects"]:
            lines.append(
                f"- {row['subset']} / {row['feature']}: "
                f"effect={row['point_biserial_effect_size']}, "
                f"weighted_diff_fraction={row['weighted_difference_fraction']}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _rows_for_label_set(
    *,
    feature_values: np.ndarray,
    feature_names: list[str],
    presence: np.ndarray,
    severity: np.ndarray,
    confidence: np.ndarray,
    subset_mask: np.ndarray,
    label_convention: str,
    subset_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        values = feature_values[..., feature_index]
        rows.append(
            _binary_stats_row(
                values=values,
                presence=presence,
                severity=severity,
                confidence=confidence,
                subset_mask=subset_mask,
                label_convention=label_convention,
                subset_name=subset_name,
                feature_name=feature_name,
            )
        )
    return rows


def _binary_stats_row(
    *,
    values: np.ndarray,
    presence: np.ndarray,
    severity: np.ndarray,
    confidence: np.ndarray,
    subset_mask: np.ndarray,
    label_convention: str,
    subset_name: str,
    feature_name: str,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    candidate = subset_mask & finite & (presence == 1)
    non_candidate = subset_mask & finite & (presence == 0)
    known = candidate | non_candidate
    candidate_values = values[candidate]
    non_candidate_values = values[non_candidate]
    candidate_weights = confidence[candidate]
    non_candidate_weights = confidence[non_candidate]
    candidate_mean = _mean(candidate_values)
    non_candidate_mean = _mean(non_candidate_values)
    weighted_candidate_mean = _weighted_mean(candidate_values, candidate_weights)
    weighted_non_candidate_mean = _weighted_mean(non_candidate_values, non_candidate_weights)
    difference = _difference(candidate_mean, non_candidate_mean)
    weighted_difference = _difference(weighted_candidate_mean, weighted_non_candidate_mean)
    pooled_std = _pooled_std(candidate_values, non_candidate_values)
    effect = None if difference is None or pooled_std is None else difference / pooled_std
    severity_means = {
        f"severity_mean_{level}": _mean(values[subset_mask & finite & (severity == level)])
        for level in range(4)
    }
    return {
        "label_convention": label_convention,
        "subset": subset_name,
        "feature": feature_name,
        "candidate_count": int(candidate_values.size),
        "non_candidate_count": int(non_candidate_values.size),
        "known_count": int(np.count_nonzero(known)),
        "candidate_mean": candidate_mean,
        "non_candidate_mean": non_candidate_mean,
        "candidate_median": _median(candidate_values),
        "non_candidate_median": _median(non_candidate_values),
        "mean_difference": difference,
        "mean_difference_fraction": _difference_fraction(difference, non_candidate_mean),
        "confidence_weighted_candidate_mean": weighted_candidate_mean,
        "confidence_weighted_non_candidate_mean": weighted_non_candidate_mean,
        "confidence_weighted_difference": weighted_difference,
        "weighted_difference_fraction": _difference_fraction(
            weighted_difference,
            weighted_non_candidate_mean,
        ),
        "point_biserial_effect_size": effect,
        "spearman_presence_r": _spearman(values[known], presence[known]),
        "spearman_severity_r": _spearman(
            values[subset_mask & finite & (severity >= 0)],
            severity[subset_mask & finite & (severity >= 0)],
        ),
        **severity_means,
        "severity_monotonic_non_decreasing": _monotonic(severity_means, direction="up"),
        "severity_monotonic_non_increasing": _monotonic(severity_means, direction="down"),
    }


def _subset_counts(presence: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        "known": int(np.count_nonzero(mask & (presence >= 0))),
        "candidate": int(np.count_nonzero(mask & (presence == 1))),
        "non_candidate": int(np.count_nonzero(mask & (presence == 0))),
        "unknown": int(np.count_nonzero(mask & (presence < 0))),
    }


def _top_primary_effects(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["label_convention"] == "plus_primary" and row["subset"] == "high_confidence"
    ]
    candidates.sort(
        key=lambda row: abs(_as_float(row.get("point_biserial_effect_size")) or 0.0),
        reverse=True,
    )
    keys = [
        "subset",
        "feature",
        "candidate_count",
        "non_candidate_count",
        "candidate_mean",
        "non_candidate_mean",
        "mean_difference",
        "weighted_difference_fraction",
        "point_biserial_effect_size",
        "spearman_presence_r",
        "spearman_severity_r",
    ]
    return [{key: row.get(key) for key in keys} for row in candidates[:limit]]


def _mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(finite):
        return None
    weight_sum = float(np.sum(weights[finite]))
    if weight_sum <= 0.0:
        return None
    return float(np.sum(values[finite] * weights[finite]) / weight_sum)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _difference_fraction(difference: float | None, baseline: float | None) -> float | None:
    if difference is None or baseline is None:
        return None
    denominator = max(abs(float(baseline)), 1.0e-12)
    return float(difference / denominator)


def _pooled_std(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left_var = float(np.var(left, ddof=1))
    right_var = float(np.var(right, ddof=1))
    denominator = left.size + right.size - 2
    if denominator <= 0:
        return None
    pooled = np.sqrt(((left.size - 1) * left_var + (right.size - 1) * right_var) / denominator)
    if not np.isfinite(pooled) or pooled <= 0.0:
        return None
    return float(pooled)


def _spearman(values: np.ndarray, labels: np.ndarray) -> float | None:
    if values.size < 3 or np.unique(labels).size < 2 or np.unique(values).size < 2:
        return None
    result = spearmanr(values, labels, nan_policy="omit")
    statistic = float(result.statistic)
    return statistic if np.isfinite(statistic) else None


def _monotonic(severity_means: dict[str, float | None], *, direction: str) -> bool | None:
    values = [severity_means[f"severity_mean_{level}"] for level in range(4)]
    finite_values = [value for value in values if value is not None and np.isfinite(value)]
    if len(finite_values) < 3:
        return None
    if direction == "up":
        return bool(
            all(left <= right for left, right in zip(finite_values, finite_values[1:], strict=True))
        )
    return bool(
        all(left >= right for left, right in zip(finite_values, finite_values[1:], strict=True))
    )


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
