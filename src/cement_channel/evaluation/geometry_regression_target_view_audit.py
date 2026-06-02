from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.evaluation.geometry_regression_audit import (
    DEFAULT_KERNELS,
    GeometryRegressionAuditConfig,
    audit_geometry_regression,
)

TARGET_VIEW_AUDIT_VERSION = "geometry_regression_target_view_audit_v001"
TARGET_VIEWS = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)

TARGET_VIEW_FORMULAS = {
    "receiver_mean": (
        "receiver_mean = mean(weighted_channel_fraction_zc_lt_2p5 over 13 receivers) "
        "for each geometry kernel and XSI reference depth"
    ),
    "receiver_p90": (
        "receiver_p90 = 90th percentile of weighted_channel_fraction_zc_lt_2p5 "
        "over 13 receivers for each geometry kernel and XSI reference depth"
    ),
    "receiver_max": (
        "receiver_max = max(weighted_channel_fraction_zc_lt_2p5 over 13 receivers) "
        "for each geometry kernel and XSI reference depth"
    ),
    "full_360_fraction": (
        "full_360_fraction = full source/receiver geometry depth-span fraction "
        "over all CAST azimuths for each geometry kernel and XSI reference depth"
    ),
    "receiver_std": (
        "receiver_std = standard deviation of weighted_channel_fraction_zc_lt_2p5 "
        "over 13 receivers for each geometry kernel and XSI reference depth"
    ),
}

TARGET_VIEW_REASONS = {
    "receiver_mean": "conservative reference view; not a primary-view change.",
    "receiver_p90": "robust anomaly candidate view; not a primary-view change.",
    "receiver_max": "sensitive audit view and current Stage 10 audited target.",
    "full_360_fraction": "auxiliary coverage view; not a primary target.",
    "receiver_std": "heterogeneity/disagreement view; not a primary target.",
}


@dataclass(frozen=True)
class GeometryRegressionTargetViewAuditReport:
    audit_version: str
    generated_at: str
    inputs: dict[str, str]
    target_views: list[str]
    kernels: list[str]
    audited_only: bool
    primary_target_view_changed: bool
    view_summaries: list[dict[str, Any]]
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


def audit_geometry_regression_target_views(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    inputs: dict[str, str] | None = None,
    target_views: tuple[str, ...] = TARGET_VIEWS,
) -> tuple[GeometryRegressionTargetViewAuditReport, list[dict[str, Any]]]:
    warnings: list[str] = []
    errors: list[str] = []
    _validate_inputs(label_arrays, target_views)

    rows: list[dict[str, Any]] = []
    kernels = [str(value) for value in np.asarray(label_arrays["geometry_kernel"])]
    for view in target_views:
        config = GeometryRegressionAuditConfig(
            target_field=view,
            target_formula=TARGET_VIEW_FORMULAS[view],
            target_selection_reason=TARGET_VIEW_REASONS[view],
        )
        audit_report, _feature_rows = audit_geometry_regression(
            label_arrays=label_arrays,
            feature_arrays=feature_arrays,
            config=config,
            inputs=inputs or {},
        )
        warnings.extend(audit_report.warnings)
        for summary in audit_report.kernel_summaries:
            rows.append(
                _view_kernel_row(
                    view=view,
                    summary=summary,
                    label_arrays=label_arrays,
                )
            )
    report = GeometryRegressionTargetViewAuditReport(
        audit_version=TARGET_VIEW_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        target_views=list(target_views),
        kernels=kernels,
        audited_only=True,
        primary_target_view_changed=False,
        view_summaries=rows,
        warnings=sorted(set(warnings)),
        errors=errors,
        no_model_training=True,
        no_model_weights=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "primary target-view change",
            "target semantics change",
            "new geometry kernel",
            "new XSI feature",
            "threshold search",
            "formal model training",
            "model weight export",
            "final label generation",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, rows


def audit_geometry_regression_target_views_from_paths(
    *,
    labels_npz: Path | str,
    features_npz: Path | str,
    output_md: Path | str,
    output_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> GeometryRegressionTargetViewAuditReport:
    label_arrays = _load_npz(Path(labels_npz))
    feature_arrays = _load_npz(Path(features_npz))
    report, rows = audit_geometry_regression_target_views(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        inputs={"labels_npz": str(labels_npz), "features_npz": str(features_npz)},
    )
    write_geometry_regression_target_view_outputs(
        report,
        rows,
        output_md=Path(output_md),
        output_json=Path(output_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def write_geometry_regression_target_view_outputs(
    report: GeometryRegressionTargetViewAuditReport,
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
    output_md.write_text(
        format_geometry_regression_target_view_markdown(report),
        encoding="utf-8",
    )
    _write_csv(rows, output_csv)


def format_geometry_regression_target_view_markdown(
    report: GeometryRegressionTargetViewAuditReport,
) -> str:
    lines = [
        "# Geometry Regression Target-View Audit",
        "",
        "This audit compares existing depth-level target views only. It does not "
        "change the primary Stage 10 audited view, add targets, train models, "
        "or authorize MVP-4C/STC/APES/deep learning/final labels.",
        "",
        f"- audit_version: `{report.audit_version}`",
        f"- target_views: `{report.target_views}`",
        f"- primary_target_view_changed: `{report.primary_target_view_changed}`",
        f"- audited_only: `{report.audited_only}`",
        "",
        "## View x Kernel Summary",
        "",
    ]
    lines.extend(_dict_message_lines(report.view_summaries))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _view_kernel_row(
    *,
    view: str,
    summary: dict[str, Any],
    label_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    distribution = _as_dict(summary.get("target_distribution"))
    fold_metrics = _fold_metrics(summary)
    kernel = str(summary.get("geometry_kernel"))
    row = {
        "target_view": view,
        "view_role": TARGET_VIEW_REASONS[view],
        "geometry_kernel": kernel,
        "sample_count": summary.get("sample_count"),
        "zero_fraction": summary.get("zero_fraction"),
        "nonzero_fraction": summary.get("nonzero_fraction"),
        "mean": distribution.get("mean"),
        "median": distribution.get("median"),
        "p90": distribution.get("p90"),
        "p95": distribution.get("p95"),
        "pearson": summary.get("pearson_correlation"),
        "spearman": summary.get("spearman_correlation"),
        "top_abs_pearson_feature": _as_optional_string(
            summary.get("top_abs_pearson_feature")
        ),
        "top_abs_pearson": summary.get("top_abs_pearson"),
        "top_abs_spearman_feature": _as_optional_string(
            summary.get("top_abs_spearman_feature")
        ),
        "top_abs_spearman": summary.get("top_abs_spearman"),
        "permutation_margin": summary.get("real_minus_permutation_margin"),
        "cv_mae": summary.get("cross_validated_mae"),
        "cv_r2": summary.get("cross_validated_r2_sanity"),
        "permutation_cv_r2": summary.get("permutation_probe_r2"),
        "fold_stability": summary.get("fold_stability"),
        "per_fold_target_mean": [
            fold.get("target_mean") for fold in fold_metrics
        ],
        "per_fold_zero_fraction": [
            fold.get("target_zero_fraction") for fold in fold_metrics
        ],
        "fold_metrics": fold_metrics,
        "depends_on_5700_band": summary.get("depends_on_5700_band"),
        "leakage_warnings": summary.get("leakage_warnings"),
        "sample_support_collapse": summary.get("sample_support_collapse"),
        "saturation_warning": _saturation_warning(view, summary),
        "single_receiver_extreme_sensitivity_warning": (
            _single_receiver_extreme_warning(view, kernel, label_arrays)
        ),
    }
    return row


def _fold_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    probe = _as_dict(summary.get("simple_linear_sanity_probe"))
    return [
        {
            "fold": fold.get("fold"),
            "validation_depth_min": fold.get("validation_depth_min"),
            "validation_depth_max": fold.get("validation_depth_max"),
            "target_mean": fold.get("target_mean"),
            "target_zero_fraction": fold.get("target_zero_fraction"),
            "mae": fold.get("mae"),
            "r2": fold.get("r2"),
        }
        for fold in probe.get("fold_metrics", [])
    ]


def _saturation_warning(view: str, summary: dict[str, Any]) -> bool:
    distribution = _as_dict(summary.get("target_distribution"))
    nonzero = _as_float(summary.get("nonzero_fraction"))
    p95 = _as_float(distribution.get("p95"))
    if view == "full_360_fraction" and nonzero is not None and nonzero >= 0.98:
        return True
    return bool(p95 is not None and p95 >= 0.95)


def _single_receiver_extreme_warning(
    view: str,
    kernel: str,
    label_arrays: dict[str, np.ndarray],
) -> bool:
    if view != "receiver_max":
        return False
    kernels = [str(value) for value in np.asarray(label_arrays["geometry_kernel"])]
    if kernel not in kernels:
        return False
    if "receiver_max" not in label_arrays or "receiver_p90" not in label_arrays:
        return False
    kernel_index = kernels.index(kernel)
    receiver_max = np.asarray(label_arrays["receiver_max"], dtype=np.float32)[kernel_index]
    receiver_p90 = np.asarray(label_arrays["receiver_p90"], dtype=np.float32)[kernel_index]
    gap = receiver_max - receiver_p90
    finite = gap[np.isfinite(gap)]
    if finite.size == 0:
        return False
    return bool(float(np.mean(finite)) >= 0.005 or float(np.percentile(finite, 95.0)) >= 0.02)


def _validate_inputs(
    label_arrays: dict[str, np.ndarray],
    target_views: tuple[str, ...],
) -> None:
    missing = sorted(set(target_views) - set(label_arrays))
    if missing:
        raise KeyError("Target-view audit missing label view(s): " + ", ".join(missing))
    kernels = tuple(str(value) for value in np.asarray(label_arrays["geometry_kernel"]))
    invalid = sorted(set(kernels) - set(DEFAULT_KERNELS))
    if invalid:
        raise ValueError("Target-view audit received unsupported kernel(s): " + ", ".join(invalid))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


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


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _dict_message_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row}" for row in rows]
