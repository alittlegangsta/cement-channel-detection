from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TRIAGE_VERSION = "geometry_regression_bounded_triage_v001"
TRIAGE_TARGET_VIEWS = (
    "receiver_mean",
    "receiver_max",
    "receiver_p90",
    "full_360_fraction",
)


@dataclass(frozen=True)
class GeometryRegressionBoundedTriageReport:
    triage_version: str
    generated_at: str
    inputs: dict[str, str]
    eligibility: dict[str, Any]
    per_target_view_summary: list[dict[str, Any]]
    per_kernel_summary: list[dict[str, Any]]
    per_fold_summary: list[dict[str, Any]]
    pearson_spearman_divergence_summary: list[dict[str, Any]]
    root_cause_classification: str
    root_cause_flags: list[str]
    warnings: list[str]
    errors: list[str]
    no_model_training: bool
    no_model_weights: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_bounded_triage(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    inventory_report: dict[str, Any],
    invariants_report: dict[str, Any],
    audit_report: dict[str, Any],
    feature_correlation_rows: list[dict[str, Any]],
    inputs: dict[str, str],
) -> tuple[GeometryRegressionBoundedTriageReport, list[dict[str, Any]]]:
    warnings: list[str] = []
    errors: list[str] = []
    eligibility = _eligibility(inventory_report, invariants_report, audit_report)
    if not all(bool(value) for value in eligibility.values()):
        errors.append("Bounded triage eligibility conditions are not satisfied.")
    per_target = _per_target_view_summary(label_arrays)
    per_kernel = _per_kernel_summary(audit_report)
    per_fold = _per_fold_summary(audit_report)
    divergence = _divergence_summary(
        feature_correlation_rows,
        audit_report=audit_report,
    )
    flags = _root_cause_flags(audit_report, divergence)
    classification = "mixed_or_unresolved" if len(flags) > 1 else flags[0]
    report = GeometryRegressionBoundedTriageReport(
        triage_version=TRIAGE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        eligibility=eligibility,
        per_target_view_summary=per_target,
        per_kernel_summary=per_kernel,
        per_fold_summary=per_fold,
        pearson_spearman_divergence_summary=divergence,
        root_cause_classification=classification,
        root_cause_flags=flags,
        warnings=warnings,
        errors=errors,
        no_model_training=True,
        no_model_weights=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "new geometry kernel",
            "threshold scan",
            "new feature generation",
            "waveform reading",
            "formal model training",
            "hyperparameter tuning",
            "model weight export",
            "production metric claim",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, _triage_csv_rows(report)


def write_bounded_triage_outputs(
    report: GeometryRegressionBoundedTriageReport,
    rows: list[dict[str, Any]],
    *,
    output_md: Path,
    output_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_csv, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_bounded_triage_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_bounded_triage_markdown(report: GeometryRegressionBoundedTriageReport) -> str:
    lines = [
        "# Geometry Regression Bounded Root-Cause Triage",
        "",
        "This is a bounded MVP-4B-GR QA triage artifact. It uses existing arrays "
        "and existing XSI features only. It is not GR-D, not model training, and "
        "does not authorize MVP-4C/STC/APES/deep learning/final labels.",
        "",
        f"- triage_version: `{report.triage_version}`",
        f"- root_cause_classification: `{report.root_cause_classification}`",
        f"- root_cause_flags: `{report.root_cause_flags}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Eligibility",
        "",
    ]
    lines.extend(_dict_lines(report.eligibility))
    lines.extend(["", "## Per Target View Summary", ""])
    lines.extend(_dict_message_lines(report.per_target_view_summary))
    lines.extend(["", "## Per Kernel Summary", ""])
    lines.extend(_dict_message_lines(report.per_kernel_summary))
    lines.extend(["", "## Per Fold Summary", ""])
    lines.extend(_dict_message_lines(report.per_fold_summary))
    lines.extend(["", "## Pearson / Spearman Divergence", ""])
    lines.extend(_dict_message_lines(report.pearson_spearman_divergence_summary))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _eligibility(
    inventory_report: dict[str, Any],
    invariants_report: dict[str, Any],
    audit_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_inventory_generated": bool(inventory_report.get("inventory_version")),
        "contract_invariants_passed": bool(invariants_report.get("passed")),
        "audited_target_view_explicit": bool(audit_report.get("audited_target_view")),
        "no_implementation_mismatch": inventory_report.get("issue_classification")
        == "report_ambiguity_only",
        "stage10_recommendation_stop": audit_report.get("recommendation") == "stop",
    }


def _per_target_view_summary(label_arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    kernels = label_arrays["geometry_kernel"].astype(str)
    for view in TRIAGE_TARGET_VIEWS:
        if view not in label_arrays:
            continue
        values = np.asarray(label_arrays[view], dtype=np.float32)
        for index, kernel in enumerate(kernels):
            target = values[index]
            rows.append(
                {
                    "target_view": view,
                    "geometry_kernel": kernel,
                    "sample_count": int(target.size),
                    "zero_fraction": _fraction(target <= 0.0),
                    "nonzero_fraction": _fraction(target > 0.0),
                    "mean": _finite_mean(target),
                    "median": _finite_percentile(target, 50.0),
                    "p90": _finite_percentile(target, 90.0),
                    "p95": _finite_percentile(target, 95.0),
                    "max": _finite_max(target),
                }
            )
    return rows


def _per_kernel_summary(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in audit_report.get("kernel_summaries", []):
        rows.append(
            {
                "geometry_kernel": row.get("geometry_kernel"),
                "target_view": row.get("target_view"),
                "sample_count": row.get("sample_count"),
                "zero_fraction": row.get("zero_fraction"),
                "nonzero_fraction": row.get("nonzero_fraction"),
                "spearman": row.get("spearman_correlation"),
                "pearson": row.get("pearson_correlation"),
                "top_abs_pearson_feature": row.get("top_abs_pearson_feature"),
                "top_abs_pearson": row.get("top_abs_pearson"),
                "top_abs_spearman_feature": row.get("top_abs_spearman_feature"),
                "top_abs_spearman": row.get("top_abs_spearman"),
                "cv_mae": row.get("cross_validated_mae"),
                "cv_r2": row.get("cross_validated_r2_sanity"),
                "permutation_cv_r2": row.get("permutation_probe_r2"),
                "fold_stability": row.get("fold_stability"),
                "depends_on_5700_band": row.get("depends_on_5700_band"),
                "leakage_warnings": row.get("leakage_warnings"),
                "sample_support_collapse": row.get("sample_support_collapse"),
            }
        )
    return rows


def _per_fold_summary(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for summary in audit_report.get("kernel_summaries", []):
        probe = _as_dict(summary.get("simple_linear_sanity_probe"))
        permutation_r2 = probe.get("permutation_r2")
        for fold in probe.get("fold_metrics", []):
            rows.append(
                {
                    "geometry_kernel": summary.get("geometry_kernel"),
                    "target_view": summary.get("target_view"),
                    "fold": fold.get("fold"),
                    "fold_depth_min": fold.get("validation_depth_min"),
                    "fold_depth_max": fold.get("validation_depth_max"),
                    "train_count": fold.get("train_count"),
                    "validation_count": fold.get("validation_count"),
                    "target_mean": fold.get("target_mean"),
                    "target_zero_fraction": fold.get("target_zero_fraction"),
                    "pearson": summary.get("pearson_correlation"),
                    "spearman": summary.get("spearman_correlation"),
                    "cv_mae": fold.get("mae"),
                    "cv_r2": fold.get("r2"),
                    "permutation_cv_r2": permutation_r2,
                    "stability_flag": bool(
                        _as_float(summary.get("fold_stability")) is not None
                        and float(summary.get("fold_stability")) >= 0.5
                    ),
                }
            )
    return rows


def _divergence_summary(
    feature_rows: list[dict[str, Any]],
    *,
    audit_report: dict[str, Any],
) -> list[dict[str, Any]]:
    depth_blocks = audit_report.get("cv_block_boundaries", [])
    rows = []
    flagged = [
        row
        for row in feature_rows
        if str(row.get("pearson_spearman_divergence_flag", "")).lower() == "true"
    ]
    flagged = sorted(
        flagged,
        key=lambda row: (
            int(row.get("abs_pearson_rank") or 10**6),
            int(row.get("abs_spearman_rank") or 10**6),
        ),
    )
    for row in flagged[:20]:
        rows.append(
            {
                "geometry_kernel": row.get("geometry_kernel"),
                "target_view": row.get("target_view"),
                "feature_name": row.get("feature_name"),
                "pearson": _as_float(row.get("pearson")),
                "spearman": _as_float(row.get("spearman")),
                "abs_pearson_rank": _as_int(row.get("abs_pearson_rank")),
                "abs_spearman_rank": _as_int(row.get("abs_spearman_rank")),
                "affected_depth_blocks": depth_blocks,
            }
        )
    return rows


def _root_cause_flags(
    audit_report: dict[str, Any],
    divergence: list[dict[str, Any]],
) -> list[str]:
    flags = ["target_view_mismatch_resolved_but_cv_unstable"]
    fold_stability = [
        _as_float(row.get("fold_stability"))
        for row in audit_report.get("kernel_summaries", [])
    ]
    if any(value is not None and value < 0.5 for value in fold_stability):
        flags.append("fold_regime_shift_suspected")
    if divergence:
        flags.append("pearson_outlier_sensitivity_suspected")
    margins = [
        _as_float(row.get("real_minus_permutation_margin"))
        for row in audit_report.get("kernel_summaries", [])
    ]
    if margins and (
        max(value or -1.0 for value in margins) - min(value or 1.0 for value in margins)
    ) < 0.05:
        flags.append("kernel_choice_not_primary_issue")
    if audit_report.get("recommendation") == "stop":
        flags.append("existing_feature_set_insufficient_suspected")
    return flags


def _triage_csv_rows(report: GeometryRegressionBoundedTriageReport) -> list[dict[str, Any]]:
    rows = []
    for row in report.per_target_view_summary:
        rows.append({"section": "per_target_view", **row})
    for row in report.per_kernel_summary:
        rows.append({"section": "per_kernel", **row})
    for row in report.per_fold_summary:
        rows.append({"section": "per_fold", **row})
    for row in report.pearson_spearman_divergence_summary:
        rows.append({"section": "pearson_spearman_divergence", **row})
    return rows


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _finite_mean(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.mean(finite))


def _finite_max(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.max(finite))


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.percentile(finite, percentile))


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _dict_message_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row}" for row in rows]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
