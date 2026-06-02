from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEPTH_REGIME_AUDIT_VERSION = "geometry_regression_depth_regime_audit_v001"
DEFAULT_TARGET_VIEW = "receiver_max"
DEFAULT_FOLD_COUNT = 3
DEFAULT_REVIEW_BIN_COUNT = 9
MORPHOLOGY_FIELDS = (
    "largest_connected_component_fraction",
    "max_azimuth_channel_fraction",
    "relative_anomaly_fraction",
    "combined_channel_fraction",
    "depth_label_confidence",
)


@dataclass(frozen=True)
class GeometryRegressionDepthRegimeConfig:
    target_view: str = DEFAULT_TARGET_VIEW
    fold_count: int = DEFAULT_FOLD_COUNT
    review_bin_count: int = DEFAULT_REVIEW_BIN_COUNT
    outlier_z_threshold: float = 5.0
    target_shift_delta_threshold: float = 0.10
    target_shift_ratio_threshold: float = 2.0
    feature_shift_z_threshold: float = 1.0
    invalid_zc_fold_fraction_threshold: float = 0.05


@dataclass(frozen=True)
class GeometryRegressionDepthRegimeAuditReport:
    audit_version: str
    generated_at: str
    inputs: dict[str, str]
    target_view: str
    cv_split_strategy_preserved: str
    formal_cv_protocol_changed: bool
    review_bin_count: int
    fold_target_summaries: list[dict[str, Any]]
    fold_feature_summaries: list[dict[str, Any]]
    fold_correlation_summaries: list[dict[str, Any]]
    fold_invalid_zc_summaries: list[dict[str, Any]]
    fold_morphology_summaries: list[dict[str, Any]]
    target_shift_summary: list[dict[str, Any]]
    feature_shift_summary: list[dict[str, Any]]
    review_bin_summaries: list[dict[str, Any]]
    selected_intervals: list[dict[str, Any]]
    regime_shift_flags: list[str]
    warnings: list[str]
    errors: list[str]
    no_cv_split_change: bool
    no_target_semantics_change: bool
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


def audit_geometry_regression_depth_regimes(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    cast_arrays: dict[str, np.ndarray] | None = None,
    config: GeometryRegressionDepthRegimeConfig | None = None,
    inputs: dict[str, str] | None = None,
) -> tuple[GeometryRegressionDepthRegimeAuditReport, list[dict[str, Any]]]:
    cfg = config or GeometryRegressionDepthRegimeConfig()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_inputs(label_arrays, feature_arrays, cfg)

    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    kernels = np.asarray(label_arrays["geometry_kernel"]).astype(str)
    target = np.asarray(label_arrays[cfg.target_view], dtype=np.float32)
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    finite_rows = np.all(np.isfinite(features), axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fold_ids = _fold_ids(depth, cfg.fold_count)
    review_bin_ids = _fold_ids(depth, cfg.review_bin_count)

    fold_target_rows = _fold_target_summaries(depth, kernels, target, fold_ids)
    fold_feature_rows = _fold_feature_summaries(
        depth,
        features,
        feature_names,
        fold_ids,
        outlier_z_threshold=cfg.outlier_z_threshold,
    )
    fold_correlation_rows = _fold_correlation_summaries(
        kernels,
        target,
        features,
        feature_names,
        fold_ids,
        finite_rows,
    )
    invalid_rows = _fold_invalid_zc_summaries(
        depth,
        fold_ids,
        cast_arrays=cast_arrays,
    )
    morphology_rows = _fold_morphology_summaries(
        label_arrays,
        depth=depth,
        kernels=kernels,
        fold_ids=fold_ids,
    )
    target_shift = _target_shift_summary(fold_target_rows, cfg)
    feature_shift = _feature_shift_summary(fold_feature_rows, cfg)
    review_bins = _review_bin_summaries(
        depth,
        kernels,
        target,
        review_bin_ids,
        bin_count=cfg.review_bin_count,
    )
    intervals = _selected_intervals(depth, kernels, target, fold_ids)
    flags = _regime_shift_flags(
        target_shift=target_shift,
        feature_shift=feature_shift,
        invalid_rows=invalid_rows,
        morphology_rows=morphology_rows,
        config=cfg,
    )
    rows: list[dict[str, Any]] = []
    for section, section_rows in (
        ("fold_target_summary", fold_target_rows),
        ("fold_feature_summary", fold_feature_rows),
        ("fold_correlation_summary", fold_correlation_rows),
        ("fold_invalid_zc_summary", invalid_rows),
        ("fold_morphology_summary", morphology_rows),
        ("target_shift_summary", target_shift),
        ("feature_shift_summary", feature_shift),
        ("review_bin_summary", review_bins),
        ("selected_interval", intervals),
    ):
        rows.extend({"section": section, **row} for row in section_rows)

    report = GeometryRegressionDepthRegimeAuditReport(
        audit_version=DEPTH_REGIME_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        target_view=cfg.target_view,
        cv_split_strategy_preserved="depth-sorted contiguous block split",
        formal_cv_protocol_changed=False,
        review_bin_count=cfg.review_bin_count,
        fold_target_summaries=fold_target_rows,
        fold_feature_summaries=fold_feature_rows,
        fold_correlation_summaries=fold_correlation_rows,
        fold_invalid_zc_summaries=invalid_rows,
        fold_morphology_summaries=morphology_rows,
        target_shift_summary=target_shift,
        feature_shift_summary=feature_shift,
        review_bin_summaries=review_bins,
        selected_intervals=intervals,
        regime_shift_flags=flags,
        warnings=warnings,
        errors=errors,
        no_cv_split_change=True,
        no_target_semantics_change=True,
        no_model_training=True,
        no_model_weights=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "formal CV split change",
            "depth-regime stratification",
            "target semantics change",
            "new geometry kernel",
            "new XSI feature",
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


def audit_geometry_regression_depth_regimes_from_paths(
    *,
    labels_npz: Path | str,
    features_npz: Path | str,
    cast_npz: Path | str | None,
    output_md: Path | str,
    output_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> GeometryRegressionDepthRegimeAuditReport:
    label_arrays = _load_npz(Path(labels_npz))
    feature_arrays = _load_npz(Path(features_npz))
    cast_arrays = None if cast_npz is None else _load_npz(Path(cast_npz))
    report, rows = audit_geometry_regression_depth_regimes(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        cast_arrays=cast_arrays,
        inputs={
            "labels_npz": str(labels_npz),
            "features_npz": str(features_npz),
            "cast_npz": str(cast_npz or ""),
        },
    )
    write_geometry_regression_depth_regime_outputs(
        report,
        rows,
        output_md=Path(output_md),
        output_json=Path(output_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def write_geometry_regression_depth_regime_outputs(
    report: GeometryRegressionDepthRegimeAuditReport,
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
        format_geometry_regression_depth_regime_markdown(report),
        encoding="utf-8",
    )
    _write_csv(rows, output_csv)


def format_geometry_regression_depth_regime_markdown(
    report: GeometryRegressionDepthRegimeAuditReport,
) -> str:
    lines = [
        "# Geometry Regression Depth-Regime Audit",
        "",
        "This audit uses existing contiguous depth folds and review-only depth "
        "bins. It does not change the formal CV protocol, target semantics, "
        "or authorize MVP-4C/STC/APES/deep learning/final labels.",
        "",
        f"- audit_version: `{report.audit_version}`",
        f"- target_view: `{report.target_view}`",
        f"- formal_cv_protocol_changed: `{report.formal_cv_protocol_changed}`",
        f"- regime_shift_flags: `{report.regime_shift_flags}`",
        "",
        "## Fold Target Summaries",
        "",
    ]
    lines.extend(_dict_message_lines(report.fold_target_summaries))
    lines.extend(["", "## Target Shift Summary", ""])
    lines.extend(_dict_message_lines(report.target_shift_summary))
    lines.extend(["", "## Top Feature Shift Summary", ""])
    lines.extend(_dict_message_lines(report.feature_shift_summary[:20]))
    lines.extend(["", "## Fold Invalid Zc Summary", ""])
    lines.extend(_dict_message_lines(report.fold_invalid_zc_summaries))
    lines.extend(["", "## Fold Morphology Summary", ""])
    lines.extend(_dict_message_lines(report.fold_morphology_summaries))
    lines.extend(["", "## Fold Correlation Summary", ""])
    lines.extend(_dict_message_lines(report.fold_correlation_summaries))
    lines.extend(["", "## Selected Representative Intervals", ""])
    lines.extend(_dict_message_lines(report.selected_intervals))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _fold_target_summaries(
    depth: np.ndarray,
    kernels: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for kernel_index, kernel in enumerate(kernels):
        values = target[kernel_index]
        for fold in sorted(set(int(value) for value in fold_ids)):
            mask = fold_ids == fold
            fold_values = values[mask]
            rows.append(
                {
                    "geometry_kernel": str(kernel),
                    "fold": fold,
                    "depth_min": _finite_min(depth[mask]),
                    "depth_max": _finite_max(depth[mask]),
                    "sample_count": int(np.count_nonzero(mask)),
                    "target_mean": _finite_mean(fold_values),
                    "target_median": _finite_percentile(fold_values, 50.0),
                    "target_p90": _finite_percentile(fold_values, 90.0),
                    "target_p95": _finite_percentile(fold_values, 95.0),
                    "target_zero_fraction": _fraction(fold_values <= 0.0),
                    "target_nonzero_fraction": _fraction(fold_values > 0.0),
                    "target_outlier_count": _robust_outlier_count(fold_values),
                }
            )
    return rows


def _fold_feature_summaries(
    depth: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    fold_ids: np.ndarray,
    *,
    outlier_z_threshold: float,
) -> list[dict[str, Any]]:
    global_median = np.nanmedian(features, axis=0)
    global_mad = np.nanmedian(np.abs(features - global_median), axis=0)
    robust_scale = np.where(global_mad > 0.0, 1.4826 * global_mad, np.nanstd(features, axis=0))
    robust_scale = np.where(robust_scale > 0.0, robust_scale, 1.0)
    rows = []
    for feature_index, name in enumerate(feature_names.astype(str)):
        values = features[:, feature_index]
        z = np.abs((values - global_median[feature_index]) / robust_scale[feature_index])
        for fold in sorted(set(int(value) for value in fold_ids)):
            mask = fold_ids == fold
            rows.append(
                {
                    "feature_index": int(feature_index),
                    "feature_name": str(name),
                    "fold": fold,
                    "depth_min": _finite_min(depth[mask]),
                    "depth_max": _finite_max(depth[mask]),
                    "sample_count": int(np.count_nonzero(mask)),
                    "feature_mean": _finite_mean(values[mask]),
                    "feature_std": _finite_std(values[mask]),
                    "feature_median": _finite_percentile(values[mask], 50.0),
                    "feature_p90": _finite_percentile(values[mask], 90.0),
                    "outlier_count": int(np.count_nonzero(z[mask] >= outlier_z_threshold)),
                    "outlier_fraction": _fraction(z[mask] >= outlier_z_threshold),
                }
            )
    return rows


def _fold_correlation_summaries(
    kernels: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    fold_ids: np.ndarray,
    finite_rows: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for kernel_index, kernel in enumerate(kernels):
        values = target[kernel_index]
        for fold in sorted(set(int(value) for value in fold_ids)):
            mask = (fold_ids == fold) & finite_rows & np.isfinite(values)
            feature_rows = []
            for feature_index, name in enumerate(feature_names.astype(str)):
                feature_values = features[:, feature_index]
                feature_rows.append(
                    {
                        "feature_index": int(feature_index),
                        "feature_name": str(name),
                        "pearson": _pearson(feature_values[mask], values[mask]),
                        "spearman": _spearman(feature_values[mask], values[mask]),
                    }
                )
            top_pearson = max(feature_rows, key=lambda row: _abs_rank(row["pearson"]))
            top_spearman = max(feature_rows, key=lambda row: _abs_rank(row["spearman"]))
            rows.append(
                {
                    "geometry_kernel": str(kernel),
                    "fold": fold,
                    "sample_count": int(np.count_nonzero(mask)),
                    "top_abs_pearson_feature": top_pearson["feature_name"],
                    "top_abs_pearson": _abs_or_none(top_pearson["pearson"]),
                    "top_abs_spearman_feature": top_spearman["feature_name"],
                    "top_abs_spearman": _abs_or_none(top_spearman["spearman"]),
                }
            )
    return rows


def _fold_invalid_zc_summaries(
    depth: np.ndarray,
    fold_ids: np.ndarray,
    *,
    cast_arrays: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    rows = []
    if cast_arrays is None:
        for fold in sorted(set(int(value) for value in fold_ids)):
            rows.append({"fold": fold, "cast_zc_available": False})
        return rows
    cast_depth = np.asarray(cast_arrays["cast_depth"], dtype=np.float32).reshape(-1)
    cast_zc = np.asarray(cast_arrays["cast_zc"], dtype=np.float32)
    finite = np.isfinite(cast_zc)
    negative = finite & (cast_zc < 0.0)
    upper = finite & (cast_zc > 12.0)
    for fold in sorted(set(int(value) for value in fold_ids)):
        mask = fold_ids == fold
        cast_mask = (cast_depth >= float(np.min(depth[mask]))) & (
            cast_depth <= float(np.max(depth[mask]))
        )
        zc_fold = cast_zc[cast_mask]
        rows.append(
            {
                "fold": fold,
                "cast_zc_available": True,
                "cast_depth_count": int(np.count_nonzero(cast_mask)),
                "negative_zc_count": int(np.count_nonzero(negative[cast_mask])),
                "negative_zc_fraction": _safe_fraction_count(
                    int(np.count_nonzero(negative[cast_mask])),
                    int(zc_fold.size),
                ),
                "nonfinite_zc_count": int(np.count_nonzero(~finite[cast_mask])),
                "nonfinite_zc_fraction": _safe_fraction_count(
                    int(np.count_nonzero(~finite[cast_mask])),
                    int(zc_fold.size),
                ),
                "upper_exceedance_count": int(np.count_nonzero(upper[cast_mask])),
                "upper_exceedance_fraction": _safe_fraction_count(
                    int(np.count_nonzero(upper[cast_mask])),
                    int(zc_fold.size),
                ),
            }
        )
    return rows


def _fold_morphology_summaries(
    label_arrays: dict[str, np.ndarray],
    *,
    depth: np.ndarray,
    kernels: np.ndarray,
    fold_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for kernel_index, kernel in enumerate(kernels):
        for fold in sorted(set(int(value) for value in fold_ids)):
            mask = fold_ids == fold
            row: dict[str, Any] = {
                "geometry_kernel": str(kernel),
                "fold": fold,
                "depth_min": _finite_min(depth[mask]),
                "depth_max": _finite_max(depth[mask]),
                "sample_count": int(np.count_nonzero(mask)),
            }
            for field in MORPHOLOGY_FIELDS:
                values = np.asarray(label_arrays[field], dtype=np.float32)[kernel_index]
                row[f"{field}_receiver_mean"] = _finite_mean(np.mean(values[mask], axis=1))
                row[f"{field}_receiver_max"] = _finite_mean(np.max(values[mask], axis=1))
            rows.append(row)
    return rows


def _target_shift_summary(
    fold_target_rows: list[dict[str, Any]],
    config: GeometryRegressionDepthRegimeConfig,
) -> list[dict[str, Any]]:
    by_kernel: dict[str, list[dict[str, Any]]] = {}
    for row in fold_target_rows:
        by_kernel.setdefault(str(row["geometry_kernel"]), []).append(row)
    output = []
    for kernel, rows in by_kernel.items():
        means = np.asarray([_as_float(row.get("target_mean")) or 0.0 for row in rows])
        zero = np.asarray([_as_float(row.get("target_zero_fraction")) or 0.0 for row in rows])
        min_mean = float(np.min(means))
        max_mean = float(np.max(means))
        ratio = None if min_mean <= 0.0 else float(max_mean / min_mean)
        output.append(
            {
                "geometry_kernel": kernel,
                "fold_target_means": [float(value) for value in means],
                "fold_zero_fractions": [float(value) for value in zero],
                "max_minus_min_target_mean": float(max_mean - min_mean),
                "max_over_min_target_mean": ratio,
                "shift_flag": bool(
                    (max_mean - min_mean) >= config.target_shift_delta_threshold
                    and (ratio is None or ratio >= config.target_shift_ratio_threshold)
                ),
            }
        )
    return output


def _feature_shift_summary(
    fold_feature_rows: list[dict[str, Any]],
    config: GeometryRegressionDepthRegimeConfig,
) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for row in fold_feature_rows:
        by_feature.setdefault(str(row["feature_name"]), []).append(row)
    output = []
    for feature, rows in by_feature.items():
        means = np.asarray([_as_float(row.get("feature_mean")) or 0.0 for row in rows])
        stds = np.asarray([_as_float(row.get("feature_std")) or 0.0 for row in rows])
        pooled = float(np.mean(stds[stds > 0.0])) if np.any(stds > 0.0) else 1.0
        shift = float(np.max(means) - np.min(means))
        output.append(
            {
                "feature_name": feature,
                "fold_feature_means": [float(value) for value in means],
                "max_minus_min_feature_mean": shift,
                "standardized_shift": shift / pooled if pooled > 0.0 else 0.0,
                "total_outlier_count": int(sum(int(row["outlier_count"]) for row in rows)),
                "shift_flag": bool(
                    pooled > 0.0 and shift / pooled >= config.feature_shift_z_threshold
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -float(row["standardized_shift"]),
            -int(row["total_outlier_count"]),
            row["feature_name"],
        ),
    )


def _review_bin_summaries(
    depth: np.ndarray,
    kernels: np.ndarray,
    target: np.ndarray,
    review_bin_ids: np.ndarray,
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for kernel_index, kernel in enumerate(kernels):
        values = target[kernel_index]
        for bin_index in range(bin_count):
            mask = review_bin_ids == bin_index
            rows.append(
                {
                    "geometry_kernel": str(kernel),
                    "review_bin": bin_index,
                    "depth_min": _finite_min(depth[mask]),
                    "depth_max": _finite_max(depth[mask]),
                    "sample_count": int(np.count_nonzero(mask)),
                    "target_mean": _finite_mean(values[mask]),
                    "target_p90": _finite_percentile(values[mask], 90.0),
                    "target_zero_fraction": _fraction(values[mask] <= 0.0),
                }
            )
    return rows


def _selected_intervals(
    depth: np.ndarray,
    kernels: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    fold_names = {
        0: "fold0_high_target",
        1: "fold1_low_target",
        2: "fold2_low_target",
    }
    for kernel_index, kernel in enumerate(kernels):
        values = target[kernel_index]
        for fold in sorted(set(int(value) for value in fold_ids)):
            mask = fold_ids == fold
            interval_type = fold_names.get(fold, f"fold{fold}_review")
            rows.append(
                {
                    "interval_type": interval_type,
                    "geometry_kernel": str(kernel),
                    "fold": fold,
                    "depth_min": _finite_min(depth[mask]),
                    "depth_max": _finite_max(depth[mask]),
                    "depth_count": int(np.count_nonzero(mask)),
                    "target_mean": _finite_mean(values[mask]),
                    "target_p90": _finite_percentile(values[mask], 90.0),
                    "target_zero_fraction": _fraction(values[mask] <= 0.0),
                }
            )
    return rows


def _regime_shift_flags(
    *,
    target_shift: list[dict[str, Any]],
    feature_shift: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    morphology_rows: list[dict[str, Any]],
    config: GeometryRegressionDepthRegimeConfig,
) -> list[str]:
    flags = []
    if any(bool(row.get("shift_flag")) for row in target_shift):
        flags.append("target_fold_regime_shift")
    if any(bool(row.get("shift_flag")) for row in feature_shift[:10]):
        flags.append("feature_fold_regime_shift")
    if any(
        (_as_float(row.get("negative_zc_fraction")) or 0.0)
        >= config.invalid_zc_fold_fraction_threshold
        for row in invalid_rows
    ):
        flags.append("invalid_zc_fold_concentration")
    combined_fold_means = [
        _as_float(row.get("combined_channel_fraction_receiver_mean"))
        for row in morphology_rows
    ]
    finite = [value for value in combined_fold_means if value is not None]
    if finite and max(finite) - min(finite) >= 0.10:
        flags.append("morphology_fold_shift")
    return flags or ["no_strong_regime_shift_flag"]


def _validate_inputs(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    config: GeometryRegressionDepthRegimeConfig,
) -> None:
    required_labels = {"depth", "geometry_kernel", config.target_view, *MORPHOLOGY_FIELDS}
    required_features = {
        "depth",
        "depth_level_xsi_features",
        "depth_level_xsi_feature_names",
    }
    missing_labels = sorted(required_labels - set(label_arrays))
    missing_features = sorted(required_features - set(feature_arrays))
    if missing_labels:
        raise KeyError("Depth-regime audit missing label field(s): " + ", ".join(missing_labels))
    if missing_features:
        raise KeyError(
            "Depth-regime audit missing feature field(s): " + ", ".join(missing_features)
        )
    label_depth = np.asarray(label_arrays["depth"]).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"]).reshape(-1)
    if label_depth.size != feature_depth.size:
        raise ValueError("Depth-regime audit label/feature depth counts differ.")


def _fold_ids(depth: np.ndarray, fold_count: int) -> np.ndarray:
    order = np.argsort(depth)
    folds = np.empty(depth.size, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(order, fold_count)):
        folds[indices] = fold
    return folds


def _robust_outlier_count(values: np.ndarray, threshold: float = 5.0) -> int:
    finite = _finite_values(values)
    if finite.size == 0:
        return 0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad if mad > 0.0 else float(np.std(finite))
    if scale <= 0.0:
        return 0
    return int(np.count_nonzero(np.abs((finite - median) / scale) >= threshold))


def _pearson(values: np.ndarray, target: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return None
    x = x[finite]
    y = y[finite]
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(values: np.ndarray, target: np.ndarray) -> float | None:
    return _pearson(_rank(values), _rank(target))


def _rank(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    return ranks


def _abs_rank(value: Any) -> float:
    number = _as_float(value)
    return -1.0 if number is None else abs(number)


def _abs_or_none(value: Any) -> float | None:
    number = _as_float(value)
    return None if number is None else abs(number)


def _safe_fraction_count(count: int, total: int) -> float | None:
    return None if total <= 0 else float(count / total)


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


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _finite_mean(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.mean(finite))


def _finite_std(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.std(finite))


def _finite_min(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.min(finite))


def _finite_max(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.max(finite))


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.percentile(finite, percentile))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _dict_message_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row}" for row in rows]
