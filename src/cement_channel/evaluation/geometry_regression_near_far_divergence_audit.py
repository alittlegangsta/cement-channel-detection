from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

NEAR_FAR_DIVERGENCE_AUDIT_VERSION = (
    "geometry_regression_near_far_divergence_audit_v001"
)
DEFAULT_TARGET_VIEW = "receiver_max"
DEFAULT_FOLD_COUNT = 3
DEFAULT_NEAR_FAR_FEATURES = (
    "near_far_ratio_mean_early_energy",
    "near_far_ratio_mean_rms_energy",
    "near_far_ratio_mean_peak_abs",
)
MORPHOLOGY_FIELDS = (
    "largest_connected_component_fraction",
    "combined_channel_fraction",
    "weighted_channel_fraction_zc_lt_2p5",
)


@dataclass(frozen=True)
class GeometryRegressionNearFarDivergenceConfig:
    target_view: str = DEFAULT_TARGET_VIEW
    feature_names: tuple[str, ...] = DEFAULT_NEAR_FAR_FEATURES
    fold_count: int = DEFAULT_FOLD_COUNT
    robust_z_threshold: float = 5.0
    max_outlier_depths_per_feature: int = 25
    divergence_abs_pearson_min: float = 0.30
    divergence_abs_spearman_max: float = 0.10
    divergence_abs_gap_min: float = 0.20
    review_band_min_ft: float = 5680.0
    review_band_max_ft: float = 5720.0
    fold_boundary_tolerance_ft: float = 10.0
    morphology_lcc_min: float = 0.15
    morphology_simple_gap_min: float = 0.10
    upper_zc_review_bound_mrayl: float = 12.0


@dataclass(frozen=True)
class GeometryRegressionNearFarDivergenceAuditReport:
    audit_version: str
    generated_at: str
    inputs: dict[str, str]
    target_view: str
    audited_feature_names: list[str]
    fold_count: int
    review_band: dict[str, float]
    denominator_small_derivable: bool
    denominator_small_derivation_reason: str
    global_correlations: list[dict[str, Any]]
    per_fold_correlations: list[dict[str, Any]]
    feature_quantiles: list[dict[str, Any]]
    outlier_depths: list[dict[str, Any]]
    outlier_overlap_summary: list[dict[str, Any]]
    divergence_summary: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    no_ratio_preprocessing_change: bool
    no_ratio_clip: bool
    no_outlier_removal: bool
    no_epsilon_change: bool
    no_new_xsi_feature: bool
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


def audit_geometry_regression_near_far_divergence(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    cast_arrays: dict[str, np.ndarray] | None = None,
    config: GeometryRegressionNearFarDivergenceConfig | None = None,
    inputs: dict[str, str] | None = None,
) -> tuple[GeometryRegressionNearFarDivergenceAuditReport, list[dict[str, Any]]]:
    cfg = config or GeometryRegressionNearFarDivergenceConfig()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_inputs(label_arrays, feature_arrays, cfg)

    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    kernels = np.asarray(label_arrays["geometry_kernel"]).astype(str)
    target = np.asarray(label_arrays[cfg.target_view], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    feature_matrix = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_indices = _feature_indices(feature_names, cfg.feature_names)
    selected_feature_names = [str(feature_names[index]) for index in feature_indices]
    selected_features = feature_matrix[:, feature_indices]
    fold_ids = _fold_ids(depth, cfg.fold_count)
    fold_boundaries = _fold_boundaries(depth, fold_ids)
    invalid_by_depth = _invalid_zc_by_depth(
        label_depth=depth,
        cast_arrays=cast_arrays,
        config=cfg,
    )
    morphology_masks = _morphology_sensitive_masks(label_arrays, cfg)

    global_rows = _global_correlations(
        kernels=kernels,
        target=target,
        features=selected_features,
        feature_names=selected_feature_names,
        config=cfg,
    )
    fold_rows = _per_fold_correlations(
        depth=depth,
        kernels=kernels,
        target=target,
        features=selected_features,
        feature_names=selected_feature_names,
        fold_ids=fold_ids,
        config=cfg,
    )
    quantile_rows = _feature_quantiles(
        features=selected_features,
        feature_names=selected_feature_names,
        config=cfg,
    )
    outlier_rows = _outlier_depths(
        depth=depth,
        kernels=kernels,
        target=target,
        features=selected_features,
        feature_names=selected_feature_names,
        fold_ids=fold_ids,
        fold_boundaries=fold_boundaries,
        invalid_by_depth=invalid_by_depth,
        morphology_masks=morphology_masks,
        config=cfg,
    )
    overlap_rows = _outlier_overlap_summary(
        feature_names=selected_feature_names,
        outlier_rows=outlier_rows,
    )
    divergence_rows = _divergence_summary(
        feature_names=selected_feature_names,
        kernels=kernels,
        global_rows=global_rows,
        fold_rows=fold_rows,
        overlap_rows=overlap_rows,
    )

    if cast_arrays is None:
        warnings.append("CAST NPZ unavailable; invalid-Zc overlap fields are false/zero.")

    rows: list[dict[str, Any]] = []
    for section, section_rows in (
        ("global_correlation", global_rows),
        ("per_fold_correlation", fold_rows),
        ("feature_quantile", quantile_rows),
        ("outlier_depth", outlier_rows),
        ("outlier_overlap_summary", overlap_rows),
        ("divergence_summary", divergence_rows),
    ):
        rows.extend({"section": section, **row} for row in section_rows)

    report = GeometryRegressionNearFarDivergenceAuditReport(
        audit_version=NEAR_FAR_DIVERGENCE_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        target_view=cfg.target_view,
        audited_feature_names=selected_feature_names,
        fold_count=cfg.fold_count,
        review_band={
            "min_ft": float(cfg.review_band_min_ft),
            "max_ft": float(cfg.review_band_max_ft),
        },
        denominator_small_derivable=False,
        denominator_small_derivation_reason=(
            "The current depth_level_xsi_features_v001 NPZ contains ratio features "
            "but not their near/far denominator components, so denominator-small "
            "counts are not safely derivable without changing feature extraction."
        ),
        global_correlations=global_rows,
        per_fold_correlations=fold_rows,
        feature_quantiles=quantile_rows,
        outlier_depths=outlier_rows,
        outlier_overlap_summary=overlap_rows,
        divergence_summary=divergence_rows,
        warnings=warnings,
        errors=errors,
        no_ratio_preprocessing_change=True,
        no_ratio_clip=True,
        no_outlier_removal=True,
        no_epsilon_change=True,
        no_new_xsi_feature=True,
        no_target_semantics_change=True,
        no_model_training=True,
        no_model_weights=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "ratio preprocessing change",
            "ratio clipping",
            "outlier removal",
            "epsilon change",
            "new XSI feature",
            "target semantics change",
            "formal model training",
            "model weight export",
            "final label generation",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    _ = feature_depth
    return report, rows


def audit_geometry_regression_near_far_divergence_from_paths(
    *,
    labels_npz: Path | str,
    features_npz: Path | str,
    cast_npz: Path | str | None,
    output_md: Path | str,
    output_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> GeometryRegressionNearFarDivergenceAuditReport:
    label_arrays = _load_npz(Path(labels_npz))
    feature_arrays = _load_npz(Path(features_npz))
    cast_arrays = None if cast_npz is None else _load_npz(Path(cast_npz))
    report, rows = audit_geometry_regression_near_far_divergence(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        cast_arrays=cast_arrays,
        inputs={
            "labels_npz": str(labels_npz),
            "features_npz": str(features_npz),
            "cast_npz": str(cast_npz or ""),
        },
    )
    write_geometry_regression_near_far_divergence_outputs(
        report,
        rows,
        output_md=Path(output_md),
        output_json=Path(output_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def write_geometry_regression_near_far_divergence_outputs(
    report: GeometryRegressionNearFarDivergenceAuditReport,
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
        format_geometry_regression_near_far_divergence_markdown(report),
        encoding="utf-8",
    )
    _write_csv(rows, output_csv)


def format_geometry_regression_near_far_divergence_markdown(
    report: GeometryRegressionNearFarDivergenceAuditReport,
) -> str:
    lines = [
        "# Geometry Regression Near/Far Ratio Divergence Audit",
        "",
        "This audit checks existing near/far ratio features only. It does not "
        "clip, remove outliers, alter epsilon handling, change preprocessing, "
        "add XSI features, train a model, or authorize MVP-4C/STC/APES/deep "
        "learning/final labels.",
        "",
        f"- audit_version: `{report.audit_version}`",
        f"- target_view: `{report.target_view}`",
        f"- audited_feature_names: `{report.audited_feature_names}`",
        f"- denominator_small_derivable: `{report.denominator_small_derivable}`",
        "",
        "## Global Pearson/Spearman",
        "",
    ]
    lines.extend(_dict_message_lines(report.global_correlations))
    lines.extend(["", "## Per-Fold Pearson/Spearman", ""])
    lines.extend(_dict_message_lines(report.per_fold_correlations))
    lines.extend(["", "## Robust Quantiles", ""])
    lines.extend(_dict_message_lines(report.feature_quantiles))
    lines.extend(["", "## Outlier Depths", ""])
    lines.extend(_dict_message_lines(report.outlier_depths))
    lines.extend(["", "## Outlier Overlap Summary", ""])
    lines.extend(_dict_message_lines(report.outlier_overlap_summary))
    lines.extend(["", "## Divergence Summary", ""])
    lines.extend(_dict_message_lines(report.divergence_summary))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _global_correlations(
    *,
    kernels: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    config: GeometryRegressionNearFarDivergenceConfig,
) -> list[dict[str, Any]]:
    rows = []
    for kernel_index, kernel in enumerate(kernels.astype(str)):
        target_values = target[kernel_index]
        for feature_index, feature_name in enumerate(feature_names):
            values = features[:, feature_index]
            pearson = _pearson(values, target_values)
            spearman = _spearman(values, target_values)
            rows.append(
                {
                    "geometry_kernel": str(kernel),
                    "feature_name": str(feature_name),
                    "sample_count": int(
                        np.count_nonzero(np.isfinite(values) & np.isfinite(target_values))
                    ),
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson_minus_abs_spearman": _abs_gap(pearson, spearman),
                    "pearson_spearman_divergence_flag": _divergence_flag(
                        pearson,
                        spearman,
                        config,
                    ),
                    "target_mean": _finite_mean(target_values),
                    "feature_median": _finite_percentile(values, 50.0),
                }
            )
    return rows


def _per_fold_correlations(
    *,
    depth: np.ndarray,
    kernels: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    fold_ids: np.ndarray,
    config: GeometryRegressionNearFarDivergenceConfig,
) -> list[dict[str, Any]]:
    rows = []
    for kernel_index, kernel in enumerate(kernels.astype(str)):
        target_values = target[kernel_index]
        for feature_index, feature_name in enumerate(feature_names):
            values = features[:, feature_index]
            for fold in sorted(set(int(value) for value in fold_ids)):
                mask = fold_ids == fold
                pearson = _pearson(values[mask], target_values[mask])
                spearman = _spearman(values[mask], target_values[mask])
                rows.append(
                    {
                        "geometry_kernel": str(kernel),
                        "feature_name": str(feature_name),
                        "fold": fold,
                        "depth_min": _finite_min(depth[mask]),
                        "depth_max": _finite_max(depth[mask]),
                        "sample_count": int(
                            np.count_nonzero(
                                np.isfinite(values[mask])
                                & np.isfinite(target_values[mask])
                            )
                        ),
                        "pearson": pearson,
                        "spearman": spearman,
                        "abs_pearson_minus_abs_spearman": _abs_gap(pearson, spearman),
                        "pearson_spearman_divergence_flag": _divergence_flag(
                            pearson,
                            spearman,
                            config,
                        ),
                    }
                )
    return rows


def _feature_quantiles(
    *,
    features: np.ndarray,
    feature_names: list[str],
    config: GeometryRegressionNearFarDivergenceConfig,
) -> list[dict[str, Any]]:
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        values = np.asarray(features[:, feature_index], dtype=np.float64)
        finite = values[np.isfinite(values)]
        z = _robust_z(values)
        extreme = np.isfinite(z) & (np.abs(z) >= config.robust_z_threshold)
        rows.append(
            {
                "feature_name": str(feature_name),
                "finite_count": int(finite.size),
                "nonfinite_count": int(values.size - finite.size),
                "min": _finite_min(values),
                "p01": _finite_percentile(values, 1.0),
                "p05": _finite_percentile(values, 5.0),
                "p10": _finite_percentile(values, 10.0),
                "median": _finite_percentile(values, 50.0),
                "p90": _finite_percentile(values, 90.0),
                "p95": _finite_percentile(values, 95.0),
                "p99": _finite_percentile(values, 99.0),
                "max": _finite_max(values),
                "robust_median": _finite_percentile(values, 50.0),
                "robust_scale": _robust_scale(values),
                "robust_z_threshold": float(config.robust_z_threshold),
                "extreme_value_count": int(np.count_nonzero(extreme)),
                "extreme_value_fraction": _fraction(extreme),
                "suspected_denominator_small_count": None,
                "denominator_small_derivable": False,
            }
        )
    return rows


def _outlier_depths(
    *,
    depth: np.ndarray,
    kernels: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    fold_ids: np.ndarray,
    fold_boundaries: list[dict[str, Any]],
    invalid_by_depth: dict[str, np.ndarray],
    morphology_masks: np.ndarray,
    config: GeometryRegressionNearFarDivergenceConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(feature_names):
        values = np.asarray(features[:, feature_index], dtype=np.float64)
        z = _robust_z(values)
        candidates = np.flatnonzero(np.isfinite(z) & (np.abs(z) >= config.robust_z_threshold))
        if candidates.size == 0:
            continue
        order = candidates[np.argsort(-np.abs(z[candidates]))]
        for index in order[: config.max_outlier_depths_per_feature]:
            sensitive_kernel_indices = [
                int(kernel_index)
                for kernel_index in range(len(kernels))
                if bool(morphology_masks[kernel_index, index])
            ]
            rows.append(
                {
                    "feature_name": str(feature_name),
                    "depth_index": int(index),
                    "depth": float(depth[index]),
                    "feature_value": float(values[index]),
                    "robust_z_score": float(z[index]),
                    "fold": int(fold_ids[index]),
                    "negative_zc_overlap": bool(invalid_by_depth["negative"][index]),
                    "nonfinite_zc_overlap": bool(invalid_by_depth["nonfinite"][index]),
                    "upper_zc_review_overlap": bool(invalid_by_depth["upper"][index]),
                    "invalid_zc_overlap": bool(
                        invalid_by_depth["negative"][index]
                        or invalid_by_depth["nonfinite"][index]
                    ),
                    "negative_zc_count": int(invalid_by_depth["negative_count"][index]),
                    "nonfinite_zc_count": int(invalid_by_depth["nonfinite_count"][index]),
                    "upper_exceedance_count": int(invalid_by_depth["upper_count"][index]),
                    "fold_boundary_overlap": _near_fold_boundary(
                        float(depth[index]),
                        fold_boundaries,
                        tolerance_ft=config.fold_boundary_tolerance_ft,
                    ),
                    "overlap_5700_band": bool(
                        config.review_band_min_ft
                        <= float(depth[index])
                        <= config.review_band_max_ft
                    ),
                    "morphology_sensitive_any_kernel": bool(sensitive_kernel_indices),
                    "morphology_sensitive_kernels": [
                        str(kernels[kernel_index])
                        for kernel_index in sensitive_kernel_indices
                    ],
                    "target_by_kernel": {
                        str(kernel): float(target[kernel_index, index])
                        for kernel_index, kernel in enumerate(kernels.astype(str))
                    },
                }
            )
    return rows


def _outlier_overlap_summary(
    *,
    feature_names: list[str],
    outlier_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for feature_name in feature_names:
        selected = [row for row in outlier_rows if row["feature_name"] == feature_name]
        count = len(selected)
        rows.append(
            {
                "feature_name": str(feature_name),
                "outlier_depth_count": count,
                "invalid_zc_overlap_count": sum(
                    1 for row in selected if bool(row["invalid_zc_overlap"])
                ),
                "invalid_zc_overlap_fraction": _safe_fraction_count(
                    sum(1 for row in selected if bool(row["invalid_zc_overlap"])),
                    count,
                ),
                "negative_zc_overlap_count": sum(
                    1 for row in selected if bool(row["negative_zc_overlap"])
                ),
                "nonfinite_zc_overlap_count": sum(
                    1 for row in selected if bool(row["nonfinite_zc_overlap"])
                ),
                "upper_zc_review_overlap_count": sum(
                    1 for row in selected if bool(row["upper_zc_review_overlap"])
                ),
                "fold_boundary_overlap_count": sum(
                    1 for row in selected if bool(row["fold_boundary_overlap"])
                ),
                "fold_boundary_overlap_fraction": _safe_fraction_count(
                    sum(1 for row in selected if bool(row["fold_boundary_overlap"])),
                    count,
                ),
                "review_5700_overlap_count": sum(
                    1 for row in selected if bool(row["overlap_5700_band"])
                ),
                "morphology_sensitive_overlap_count": sum(
                    1 for row in selected if bool(row["morphology_sensitive_any_kernel"])
                ),
                "morphology_sensitive_overlap_fraction": _safe_fraction_count(
                    sum(
                        1
                        for row in selected
                        if bool(row["morphology_sensitive_any_kernel"])
                    ),
                    count,
                ),
                "suspected_denominator_small_count": None,
                "denominator_small_derivable": False,
            }
        )
    return rows


def _divergence_summary(
    *,
    feature_names: list[str],
    kernels: np.ndarray,
    global_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    overlap_by_feature = {str(row["feature_name"]): row for row in overlap_rows}
    rows = []
    for feature_name in feature_names:
        for kernel in kernels.astype(str):
            global_match = next(
                row
                for row in global_rows
                if row["feature_name"] == feature_name and row["geometry_kernel"] == kernel
            )
            fold_matches = [
                row
                for row in fold_rows
                if row["feature_name"] == feature_name and row["geometry_kernel"] == kernel
            ]
            rows.append(
                {
                    "feature_name": str(feature_name),
                    "geometry_kernel": str(kernel),
                    "global_pearson": global_match["pearson"],
                    "global_spearman": global_match["spearman"],
                    "global_abs_gap": global_match["abs_pearson_minus_abs_spearman"],
                    "global_divergence_flag": bool(
                        global_match["pearson_spearman_divergence_flag"]
                    ),
                    "fold_pearson": [row["pearson"] for row in fold_matches],
                    "fold_spearman": [row["spearman"] for row in fold_matches],
                    "fold_divergence_flags": [
                        bool(row["pearson_spearman_divergence_flag"])
                        for row in fold_matches
                    ],
                    "any_divergence_flag": bool(
                        global_match["pearson_spearman_divergence_flag"]
                        or any(
                            bool(row["pearson_spearman_divergence_flag"])
                            for row in fold_matches
                        )
                    ),
                    "outlier_depth_count": int(
                        overlap_by_feature[feature_name]["outlier_depth_count"]
                    ),
                    "invalid_zc_overlap_count": int(
                        overlap_by_feature[feature_name]["invalid_zc_overlap_count"]
                    ),
                    "fold_boundary_overlap_count": int(
                        overlap_by_feature[feature_name]["fold_boundary_overlap_count"]
                    ),
                    "review_5700_overlap_count": int(
                        overlap_by_feature[feature_name]["review_5700_overlap_count"]
                    ),
                    "morphology_sensitive_overlap_count": int(
                        overlap_by_feature[feature_name][
                            "morphology_sensitive_overlap_count"
                        ]
                    ),
                }
            )
    return rows


def _validate_inputs(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    config: GeometryRegressionNearFarDivergenceConfig,
) -> None:
    required_labels = {
        "depth",
        "geometry_kernel",
        config.target_view,
        *MORPHOLOGY_FIELDS,
    }
    required_features = {
        "depth",
        "depth_level_xsi_features",
        "depth_level_xsi_feature_names",
    }
    missing_labels = sorted(required_labels - set(label_arrays))
    missing_features = sorted(required_features - set(feature_arrays))
    if missing_labels:
        raise KeyError(
            "Near/far divergence audit missing label field(s): "
            + ", ".join(missing_labels)
        )
    if missing_features:
        raise KeyError(
            "Near/far divergence audit missing feature field(s): "
            + ", ".join(missing_features)
        )
    label_depth = np.asarray(label_arrays["depth"]).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"]).reshape(-1)
    if label_depth.size != feature_depth.size:
        raise ValueError("Near/far divergence audit label/feature depth counts differ.")
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    missing_ratio = [name for name in config.feature_names if name not in set(feature_names)]
    if missing_ratio:
        raise KeyError(
            "Near/far divergence audit missing requested ratio feature(s): "
            + ", ".join(missing_ratio)
        )


def _feature_indices(feature_names: np.ndarray, requested: tuple[str, ...]) -> list[int]:
    lookup = {str(name): index for index, name in enumerate(feature_names.astype(str))}
    return [int(lookup[name]) for name in requested]


def _morphology_sensitive_masks(
    label_arrays: dict[str, np.ndarray],
    config: GeometryRegressionNearFarDivergenceConfig,
) -> np.ndarray:
    simple = np.asarray(label_arrays["weighted_channel_fraction_zc_lt_2p5"], dtype=np.float32)
    combined = np.asarray(label_arrays["combined_channel_fraction"], dtype=np.float32)
    lcc = np.asarray(label_arrays["largest_connected_component_fraction"], dtype=np.float32)
    simple_mean = np.mean(simple, axis=2)
    combined_mean = np.mean(combined, axis=2)
    lcc_mean = np.mean(lcc, axis=2)
    return ((combined_mean - simple_mean) >= config.morphology_simple_gap_min) | (
        lcc_mean >= config.morphology_lcc_min
    )


def _invalid_zc_by_depth(
    *,
    label_depth: np.ndarray,
    cast_arrays: dict[str, np.ndarray] | None,
    config: GeometryRegressionNearFarDivergenceConfig,
) -> dict[str, np.ndarray]:
    n = label_depth.size
    output = {
        "negative": np.zeros(n, dtype=bool),
        "nonfinite": np.zeros(n, dtype=bool),
        "upper": np.zeros(n, dtype=bool),
        "negative_count": np.zeros(n, dtype=np.int32),
        "nonfinite_count": np.zeros(n, dtype=np.int32),
        "upper_count": np.zeros(n, dtype=np.int32),
    }
    if cast_arrays is None:
        return output
    cast_depth = np.asarray(cast_arrays["cast_depth"], dtype=np.float64).reshape(-1)
    cast_zc = np.asarray(cast_arrays["cast_zc"], dtype=np.float32)
    if cast_depth.size != cast_zc.shape[0]:
        raise ValueError("CAST depth count does not match cast_zc depth dimension.")
    sort_order = np.argsort(label_depth)
    sorted_depth = np.asarray(label_depth, dtype=np.float64)[sort_order]
    edges = _depth_bin_edges(sorted_depth)
    assigned = np.searchsorted(edges, cast_depth, side="right") - 1
    valid = (assigned >= 0) & (assigned < n)
    finite = np.isfinite(cast_zc)
    negative = finite & (cast_zc < 0.0)
    upper = finite & (cast_zc > config.upper_zc_review_bound_mrayl)
    nonfinite = ~finite
    for sorted_index, depth_index in enumerate(sort_order):
        mask = valid & (assigned == sorted_index)
        if not np.any(mask):
            continue
        output["negative_count"][depth_index] = int(np.count_nonzero(negative[mask]))
        output["nonfinite_count"][depth_index] = int(np.count_nonzero(nonfinite[mask]))
        output["upper_count"][depth_index] = int(np.count_nonzero(upper[mask]))
        output["negative"][depth_index] = bool(output["negative_count"][depth_index] > 0)
        output["nonfinite"][depth_index] = bool(output["nonfinite_count"][depth_index] > 0)
        output["upper"][depth_index] = bool(output["upper_count"][depth_index] > 0)
    return output


def _depth_bin_edges(sorted_depth: np.ndarray) -> np.ndarray:
    if sorted_depth.size == 1:
        return np.asarray([sorted_depth[0] - 0.5, sorted_depth[0] + 0.5], dtype=np.float64)
    mids = (sorted_depth[:-1] + sorted_depth[1:]) / 2.0
    first = sorted_depth[0] - (mids[0] - sorted_depth[0])
    last = sorted_depth[-1] + (sorted_depth[-1] - mids[-1])
    return np.r_[first, mids, last]


def _fold_ids(depth: np.ndarray, fold_count: int) -> np.ndarray:
    order = np.argsort(depth)
    folds = np.empty(depth.size, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(order, fold_count)):
        folds[indices] = fold
    return folds


def _fold_boundaries(depth: np.ndarray, fold_ids: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for fold in sorted(set(int(value) for value in fold_ids)):
        mask = fold_ids == fold
        rows.append(
            {
                "fold": fold,
                "depth_min": _finite_min(depth[mask]),
                "depth_max": _finite_max(depth[mask]),
            }
        )
    return rows


def _near_fold_boundary(
    depth_value: float,
    boundaries: list[dict[str, Any]],
    *,
    tolerance_ft: float,
) -> bool:
    for row in boundaries:
        for key in ("depth_min", "depth_max"):
            boundary = _as_float(row.get(key))
            if boundary is not None and abs(depth_value - boundary) <= tolerance_ft:
                return True
    return False


def _robust_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    median = _finite_percentile(array, 50.0)
    scale = _robust_scale(array)
    z = np.full(array.shape, np.nan, dtype=np.float64)
    if median is None or scale <= 0.0:
        return z
    finite = np.isfinite(array)
    z[finite] = (array[finite] - median) / scale
    return z


def _robust_scale(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad if mad > 0.0 else float(np.std(finite))
    return float(scale) if scale > 0.0 else 0.0


def _divergence_flag(
    pearson: float | None,
    spearman: float | None,
    config: GeometryRegressionNearFarDivergenceConfig,
) -> bool:
    if pearson is None or spearman is None:
        return False
    return bool(
        abs(pearson) >= config.divergence_abs_pearson_min
        and abs(spearman) <= config.divergence_abs_spearman_max
        and abs(abs(pearson) - abs(spearman)) >= config.divergence_abs_gap_min
    )


def _abs_gap(pearson: float | None, spearman: float | None) -> float | None:
    if pearson is None or spearman is None:
        return None
    return float(abs(pearson) - abs(spearman))


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
