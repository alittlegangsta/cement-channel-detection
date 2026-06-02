from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

DEPTH_REGIME_REVIEW_VERSION = "geometry_regression_depth_regime_audit_v001"
DEPTH_REGIME_DECISION_VERSION = "geometry_regression_depth_regime_decision_v001"
PRIMARY_KERNEL = "triangular_midpoint_weighted"
TARGET_VIEWS = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)
MORPHOLOGY_FIELDS = (
    "largest_connected_component_fraction",
    "max_azimuth_channel_fraction",
    "relative_anomaly_fraction",
    "combined_channel_fraction",
    "depth_label_confidence",
)
NEAR_FAR_FEATURES = (
    "near_far_ratio_mean_early_energy",
    "near_far_ratio_mean_rms_energy",
    "near_far_ratio_mean_peak_abs",
)
REVIEW_BAND_MIN_FT = 5680.0
REVIEW_BAND_MAX_FT = 5720.0
FIXED_WIDTH_BIN_FT = 150.0
FIXED_COUNT_BIN_SIZE = 500
ROBUST_Z_THRESHOLD = 5.0
FIGURE_NAMES = (
    "fold_target_distribution.png",
    "fold_feature_group_distribution.png",
    "fold_morphology_distribution.png",
    "fold_orientation_confidence.png",
    "target_view_vs_depth.png",
    "target_view_by_regime.png",
    "receiver_std_vs_depth.png",
    "morphology_vs_depth.png",
    "near_far_ratio_vs_depth.png",
    "near_far_outliers_and_boundaries.png",
    "invalid_zc_and_boundaries.png",
    "candidate_regime_boundaries.png",
)


@dataclass(frozen=True)
class GeometryRegressionDepthRegimeReviewReport:
    audit_version: str
    generated_at: str
    inputs: dict[str, str]
    formal_cv_protocol_changed: bool
    formal_regime_boundaries_adopted: bool
    no_stratified_model: bool
    fold_summaries: list[dict[str, Any]]
    fold_target_view_summaries: list[dict[str, Any]]
    fold_kernel_summaries: list[dict[str, Any]]
    fold_morphology_summaries: list[dict[str, Any]]
    fold_feature_group_summaries: list[dict[str, Any]]
    fold_physical_covariate_summaries: list[dict[str, Any]]
    bin_count_summary: dict[str, int]
    candidate_boundaries: list[dict[str, Any]]
    target_view_stability_by_regime: list[dict[str, Any]]
    morphology_sensitivity_by_regime: list[dict[str, Any]]
    near_far_divergence_by_regime: list[dict[str, Any]]
    required_question_answers: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_geometry_regression_depth_regime_review(
    *,
    labels_npz: Path | str,
    features_npz: Path | str,
    cast_npz: Path | str,
    output_md: Path | str,
    output_json: Path | str,
    bins_csv: Path | str,
    boundaries_csv: Path | str,
    review_dir: Path | str,
    decision_md: Path | str,
    decision_json: Path | str,
    overwrite: bool = False,
) -> GeometryRegressionDepthRegimeReviewReport:
    labels = _load_npz(Path(labels_npz))
    features = _load_npz(Path(features_npz))
    cast = _load_npz(Path(cast_npz))
    output_md_path = Path(output_md)
    output_json_path = Path(output_json)
    bins_csv_path = Path(bins_csv)
    boundaries_csv_path = Path(boundaries_csv)
    review_path = Path(review_dir)
    decision_md_path = Path(decision_md)
    decision_json_path = Path(decision_json)
    selected_csv = review_path / "selected_regime_intervals.csv"
    selected_json = review_path / "selected_regime_intervals.json"
    review_summary = review_path / "review_summary.md"
    figure_paths = {name: review_path / name for name in FIGURE_NAMES}
    for path in (
        output_md_path,
        output_json_path,
        bins_csv_path,
        boundaries_csv_path,
        selected_csv,
        selected_json,
        review_summary,
        decision_md_path,
        decision_json_path,
        *figure_paths.values(),
    ):
        _ensure_can_write(path, overwrite=overwrite)
    review_path.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    bins_csv_path.parent.mkdir(parents=True, exist_ok=True)
    boundaries_csv_path.parent.mkdir(parents=True, exist_ok=True)
    decision_md_path.parent.mkdir(parents=True, exist_ok=True)
    decision_json_path.parent.mkdir(parents=True, exist_ok=True)

    context = _build_context(labels, features, cast)
    fold_rows = _fold_summaries(context)
    target_rows = _fold_target_view_summaries(context)
    kernel_rows = _fold_kernel_summaries(context)
    morphology_rows = _fold_morphology_summaries(context)
    feature_group_rows = _fold_feature_group_summaries(context)
    physical_rows = _fold_physical_covariate_summaries(context)
    bin_rows = _review_bin_summaries(context)
    boundary_rows = _candidate_boundaries(context, bin_rows)
    target_view_stability = _target_view_stability(context, bin_rows)
    morphology_sensitivity = _morphology_sensitivity(context, boundary_rows)
    near_far = _near_far_by_regime(context, bin_rows, boundary_rows)
    answers = _required_question_answers(
        context=context,
        boundary_rows=boundary_rows,
        target_view_stability=target_view_stability,
        morphology_sensitivity=morphology_sensitivity,
        near_far=near_far,
    )
    warnings = list(context["warnings"])
    report = GeometryRegressionDepthRegimeReviewReport(
        audit_version=DEPTH_REGIME_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "labels_npz": str(labels_npz),
            "features_npz": str(features_npz),
            "cast_npz": str(cast_npz),
        },
        formal_cv_protocol_changed=False,
        formal_regime_boundaries_adopted=False,
        no_stratified_model=True,
        fold_summaries=fold_rows,
        fold_target_view_summaries=target_rows,
        fold_kernel_summaries=kernel_rows,
        fold_morphology_summaries=morphology_rows,
        fold_feature_group_summaries=feature_group_rows,
        fold_physical_covariate_summaries=physical_rows,
        bin_count_summary=_bin_count_summary(bin_rows),
        candidate_boundaries=boundary_rows,
        target_view_stability_by_regime=target_view_stability,
        morphology_sensitivity_by_regime=morphology_sensitivity,
        near_far_divergence_by_regime=near_far,
        required_question_answers=answers,
        warnings=warnings,
        errors=[],
        not_performed=[
            "formal CV split change",
            "formal regime boundary adoption",
            "stratified model training",
            "primary target-view change",
            "receiver_p90 or receiver_mean adoption",
            "new morphology target",
            "label definition change",
            "new geometry kernel",
            "new XSI feature",
            "near/far preprocessing change",
            "permutation protocol change",
            "waveform read",
            "model training",
            "MVP-4B-GR-D",
            "MVP-4C",
            "STC",
            "APES",
            "deep learning",
            "final labels",
            "ground-truth claim",
        ],
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
    )
    selected = _selected_regime_intervals(context, boundary_rows)
    decision = _build_decision(report)

    output_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md_path.write_text(_format_report_markdown(report), encoding="utf-8")
    _write_csv(bin_rows, bins_csv_path)
    _write_csv(boundary_rows, boundaries_csv_path)
    _write_csv(selected, selected_csv)
    selected_json.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
    _write_review_figures(context, boundary_rows, figure_paths, overwrite=overwrite)
    review_summary.write_text(
        _format_review_summary(report, selected, figure_paths),
        encoding="utf-8",
    )
    decision_json_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    decision_md_path.write_text(_format_decision_markdown(decision), encoding="utf-8")
    return report


def _build_context(
    labels: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    cast: dict[str, np.ndarray],
) -> dict[str, Any]:
    required_labels = {
        "depth",
        "geometry_kernel",
        *TARGET_VIEWS,
        *MORPHOLOGY_FIELDS,
        "weighted_channel_fraction_zc_lt_2p5",
        "total_cell_count",
    }
    required_features = {
        "depth",
        "depth_level_xsi_features",
        "depth_level_xsi_feature_names",
        "feature_group_names",
        "feature_group_counts_json",
    }
    required_cast = {"cast_depth", "cast_zc"}
    _require_fields(labels, required_labels, "regression labels")
    _require_fields(features, required_features, "depth-level features")
    _require_fields(cast, required_cast, "CAST label input")
    depth = np.asarray(labels["depth"], dtype=np.float32).reshape(-1)
    feature_depth = np.asarray(features["depth"], dtype=np.float32).reshape(-1)
    if depth.size != feature_depth.size:
        raise ValueError("Label and feature depth counts differ.")
    kernels = np.asarray(labels["geometry_kernel"]).astype(str)
    primary_index = _kernel_index(kernels, PRIMARY_KERNEL)
    feature_matrix = np.asarray(features["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(features["depth_level_xsi_feature_names"]).astype(str)
    cast_depth = np.asarray(cast["cast_depth"], dtype=np.float32).reshape(-1)
    cast_zc = np.asarray(cast["cast_zc"], dtype=np.float32)
    if cast_zc.shape[0] != cast_depth.size:
        raise ValueError("CAST depth count does not match cast_zc rows.")
    warnings = []
    if "orientation_confidence" not in cast:
        warnings.append("orientation_confidence not found in CAST NPZ.")
    if "inc_deg" not in cast:
        warnings.append("inc_deg not found in CAST NPZ.")
    if "relbearing_deg" not in cast:
        warnings.append("relbearing_deg not found in CAST NPZ.")
    warnings.append("depth-match error covariate not found; alignment-quality shift not scored.")
    fold_ids = _fold_ids(depth, 3)
    feature_groups = _feature_group_slices(features)
    near_far_indices = _near_far_indices(feature_names)
    near_far_z = {
        name: _robust_z(feature_matrix[:, index])
        for name, index in near_far_indices.items()
    }
    invalid_by_depth = _invalid_zc_by_label_depth(depth, cast_depth, cast_zc)
    covariates = _cast_covariates_by_label_depth(depth, cast, cast_depth)
    return {
        "labels": labels,
        "features": features,
        "cast": cast,
        "depth": depth,
        "kernels": kernels,
        "primary_index": primary_index,
        "feature_matrix": np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0),
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "near_far_indices": near_far_indices,
        "near_far_z": near_far_z,
        "cast_depth": cast_depth,
        "cast_zc": cast_zc,
        "fold_ids": fold_ids,
        "invalid_by_depth": invalid_by_depth,
        "covariates": covariates,
        "warnings": warnings,
    }


def _fold_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    depth = context["depth"]
    fold_ids = context["fold_ids"]
    rows = []
    for fold in range(3):
        mask = fold_ids == fold
        rows.append(
            {
                "fold": fold,
                "depth_min": _finite_min(depth[mask]),
                "depth_max": _finite_max(depth[mask]),
                "sample_count": int(np.count_nonzero(mask)),
                "overlaps_5700_band": _overlaps_5700(depth[mask]),
            }
        )
    return rows


def _fold_target_view_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    depth = context["depth"]
    fold_ids = context["fold_ids"]
    labels = context["labels"]
    kernels = context["kernels"]
    rows = []
    for kernel_index, kernel in enumerate(kernels):
        for view in TARGET_VIEWS:
            values = np.asarray(labels[view], dtype=np.float32)[kernel_index]
            for fold in range(3):
                mask = fold_ids == fold
                rows.append(
                    {
                        "section": "fold_target_view_summary",
                        "fold": fold,
                        "depth_min": _finite_min(depth[mask]),
                        "depth_max": _finite_max(depth[mask]),
                        "geometry_kernel": str(kernel),
                        "target_view": view,
                        "sample_count": int(np.count_nonzero(mask)),
                        **_distribution(values[mask], prefix="target"),
                    }
                )
    return rows


def _fold_kernel_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    depth = context["depth"]
    fold_ids = context["fold_ids"]
    labels = context["labels"]
    kernels = context["kernels"]
    target = np.asarray(labels["receiver_max"], dtype=np.float32)
    rows = []
    for kernel_index, kernel in enumerate(kernels):
        for fold in range(3):
            mask = fold_ids == fold
            rows.append(
                {
                    "section": "fold_kernel_summary",
                    "fold": fold,
                    "depth_min": _finite_min(depth[mask]),
                    "depth_max": _finite_max(depth[mask]),
                    "geometry_kernel": str(kernel),
                    "target_view": "receiver_max",
                    **_distribution(target[kernel_index, mask], prefix="target"),
                }
            )
    return rows


def _fold_morphology_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    depth = context["depth"]
    fold_ids = context["fold_ids"]
    labels = context["labels"]
    kernels = context["kernels"]
    rows = []
    for kernel_index, kernel in enumerate(kernels):
        for field in MORPHOLOGY_FIELDS:
            values = np.mean(np.asarray(labels[field], dtype=np.float32)[kernel_index], axis=1)
            for fold in range(3):
                mask = fold_ids == fold
                rows.append(
                    {
                        "section": "fold_morphology_summary",
                        "fold": fold,
                        "depth_min": _finite_min(depth[mask]),
                        "depth_max": _finite_max(depth[mask]),
                        "geometry_kernel": str(kernel),
                        "morphology_field": field,
                        **_distribution(values[mask], prefix="morphology"),
                    }
                )
    return rows


def _fold_feature_group_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    depth = context["depth"]
    fold_ids = context["fold_ids"]
    matrix = context["feature_matrix"]
    rows = []
    for group, indices in context["feature_groups"].items():
        group_values = _row_group_magnitude(matrix[:, indices])
        for fold in range(3):
            mask = fold_ids == fold
            rows.append(
                {
                    "section": "fold_feature_group_summary",
                    "fold": fold,
                    "depth_min": _finite_min(depth[mask]),
                    "depth_max": _finite_max(depth[mask]),
                    "feature_group": group,
                    **_distribution(group_values[mask], prefix="feature_group"),
                    "robust_outlier_count": _robust_outlier_count(group_values[mask]),
                }
            )
    return rows


def _fold_physical_covariate_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    depth = context["depth"]
    fold_ids = context["fold_ids"]
    covariates = context["covariates"]
    invalid = context["invalid_by_depth"]
    label_conf = _primary_label_confidence(context)
    near_far_extreme = _near_far_extreme_mask(context)
    rows = []
    for fold in range(3):
        mask = fold_ids == fold
        rows.append(
            {
                "section": "fold_physical_covariate_summary",
                "fold": fold,
                "depth_min": _finite_min(depth[mask]),
                "depth_max": _finite_max(depth[mask]),
                "sample_count": int(np.count_nonzero(mask)),
                "orientation_confidence_mean": _finite_mean(
                    covariates["orientation_confidence"][mask]
                ),
                "orientation_confidence_p10": _finite_percentile(
                    covariates["orientation_confidence"][mask],
                    10,
                ),
                "low_orientation_confidence_count": int(
                    np.count_nonzero(covariates["orientation_confidence"][mask] < 0.5)
                ),
                "inclination_mean": _finite_mean(covariates["inc_deg"][mask]),
                "inclination_median": _finite_percentile(covariates["inc_deg"][mask], 50),
                "relbearing_finite_fraction": _fraction(
                    np.isfinite(covariates["relbearing_deg"][mask])
                ),
                "label_confidence_mean": _finite_mean(label_conf[mask]),
                "label_confidence_min": _finite_min(label_conf[mask]),
                "invalid_zc_count": int(np.sum(invalid["negative_count"][mask])),
                "nonfinite_zc_count": int(np.sum(invalid["nonfinite_count"][mask])),
                "upper_zc_review_count": int(np.sum(invalid["upper_count"][mask])),
                "near_far_ratio_extreme_count": int(np.count_nonzero(near_far_extreme[mask])),
                "overlaps_5700_band": _overlaps_5700(depth[mask]),
            }
        )
    return rows


def _review_bin_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rows.extend(_fixed_width_bin_rows(context, width_ft=FIXED_WIDTH_BIN_FT))
    rows.extend(_fixed_count_bin_rows(context, bin_size=FIXED_COUNT_BIN_SIZE))
    return rows


def _fixed_width_bin_rows(context: dict[str, Any], *, width_ft: float) -> list[dict[str, Any]]:
    depth = context["depth"]
    edges = np.arange(float(np.min(depth)), float(np.max(depth)) + width_ft, width_ft)
    if edges[-1] < float(np.max(depth)):
        edges = np.r_[edges, float(np.max(depth))]
    rows = []
    for index in range(len(edges) - 1):
        low = edges[index]
        high = edges[index + 1]
        mask = (depth >= low) & (depth < high if index < len(edges) - 2 else depth <= high)
        if np.count_nonzero(mask) == 0:
            continue
        rows.append(_bin_row(context, mask, "fixed_width_150ft", index))
    return rows


def _fixed_count_bin_rows(context: dict[str, Any], *, bin_size: int) -> list[dict[str, Any]]:
    depth = context["depth"]
    order = np.argsort(depth)
    rows = []
    bin_count = max(1, int(np.ceil(depth.size / bin_size)))
    for index, indices in enumerate(np.array_split(order, bin_count)):
        mask = np.zeros(depth.size, dtype=bool)
        mask[indices] = True
        rows.append(_bin_row(context, mask, "fixed_count_500sample", index))
    return rows


def _bin_row(
    context: dict[str, Any],
    mask: np.ndarray,
    bin_type: str,
    bin_index: int,
) -> dict[str, Any]:
    depth = context["depth"]
    labels = context["labels"]
    primary_index = context["primary_index"]
    covariates = context["covariates"]
    invalid = context["invalid_by_depth"]
    near_far_extreme = _near_far_extreme_mask(context)
    label_conf = _primary_label_confidence(context)
    features = context["feature_matrix"]
    row: dict[str, Any] = {
        "bin_type": bin_type,
        "bin_index": bin_index,
        "depth_min": _finite_min(depth[mask]),
        "depth_max": _finite_max(depth[mask]),
        "sample_count": int(np.count_nonzero(mask)),
        "overlaps_5700_band": _overlaps_5700(depth[mask]),
        "orientation_confidence_mean": _finite_mean(covariates["orientation_confidence"][mask]),
        "orientation_confidence_p10": _finite_percentile(
            covariates["orientation_confidence"][mask],
            10,
        ),
        "inclination_mean": _finite_mean(covariates["inc_deg"][mask]),
        "label_confidence_mean": _finite_mean(label_conf[mask]),
        "invalid_zc_count": int(np.sum(invalid["negative_count"][mask])),
        "nonfinite_zc_count": int(np.sum(invalid["nonfinite_count"][mask])),
        "upper_zc_review_count": int(np.sum(invalid["upper_count"][mask])),
        "near_far_ratio_extreme_count": int(np.count_nonzero(near_far_extreme[mask])),
        "boundary_or_coverage_flag": bool(
            np.any(np.asarray(labels["total_cell_count"])[primary_index, mask] <= 0)
        ),
    }
    for view in ("receiver_mean", "receiver_p90", "receiver_max"):
        row.update(_distribution(np.asarray(labels[view])[primary_index, mask], prefix=view))
    for field in MORPHOLOGY_FIELDS:
        values = np.mean(np.asarray(labels[field])[primary_index], axis=1)
        row[f"{field}_mean"] = _finite_mean(values[mask])
        row[f"{field}_p90"] = _finite_percentile(values[mask], 90)
    group_scores = []
    for group, indices in context["feature_groups"].items():
        group_values = _row_group_magnitude(features[:, indices])
        row[f"feature_group_{group}_median"] = _finite_percentile(group_values[mask], 50)
        row[f"feature_group_{group}_p90"] = _finite_percentile(group_values[mask], 90)
        group_scores.append((group, _finite_percentile(group_values[mask], 90) or 0.0))
    row["top_feature_group_by_p90"] = max(group_scores, key=lambda item: item[1])[0]
    return row


def _candidate_boundaries(
    context: dict[str, Any],
    bin_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    depth = context["depth"]
    fold_ids = context["fold_ids"]
    rows = []
    for fold in (0, 1):
        left = depth[fold_ids == fold]
        right = depth[fold_ids == fold + 1]
        boundary = (float(np.max(left)) + float(np.min(right))) / 2.0
        rows.append(
            _boundary_row(
                context,
                boundary_depth=boundary,
                boundary_id=f"existing_fold_boundary_{fold}_{fold + 1}",
                source="existing_contiguous_fold_boundary",
            )
        )
    fixed = [
        row for row in bin_rows if row["bin_type"] == "fixed_count_500sample"
    ]
    scores = []
    for left, right in zip(fixed[:-1], fixed[1:], strict=False):
        boundary = (float(left["depth_max"]) + float(right["depth_min"])) / 2.0
        score = _adjacent_bin_shift_score(left, right)
        scores.append((score, boundary))
    for rank, (score, boundary) in enumerate(sorted(scores, reverse=True)[:4], start=1):
        row = _boundary_row(
            context,
            boundary_depth=boundary,
            boundary_id=f"exploratory_changepoint_{rank}",
            source="exploratory_fixed_count_bin_shift",
        )
        row["adjacent_bin_shift_score"] = float(score)
        rows.append(row)
    rows.append(
        _boundary_row(
            context,
            boundary_depth=REVIEW_BAND_MIN_FT,
            boundary_id="5700_review_band_start",
            source="existing_5700_review_band",
        )
    )
    return sorted(rows, key=lambda row: float(row["boundary_depth"]))


def _boundary_row(
    context: dict[str, Any],
    *,
    boundary_depth: float,
    boundary_id: str,
    source: str,
) -> dict[str, Any]:
    depth = context["depth"]
    window = max(25.0, FIXED_WIDTH_BIN_FT / 2.0)
    left = (depth >= boundary_depth - window) & (depth < boundary_depth)
    right = (depth >= boundary_depth) & (depth <= boundary_depth + window)
    if np.count_nonzero(left) == 0 or np.count_nonzero(right) == 0:
        left = depth < boundary_depth
        right = depth >= boundary_depth
    labels = context["labels"]
    primary = context["primary_index"]
    target = np.asarray(labels["receiver_max"])[primary]
    morphology = np.mean(np.asarray(labels["combined_channel_fraction"])[primary], axis=1)
    feature_dynamic = _row_group_magnitude(context["feature_matrix"])
    orient = context["covariates"]["orientation_confidence"]
    inc = context["covariates"]["inc_deg"]
    label_conf = _primary_label_confidence(context)
    invalid = context["invalid_by_depth"]
    near_far = _near_far_extreme_mask(context)
    target_delta = abs((_finite_mean(target[right]) or 0.0) - (_finite_mean(target[left]) or 0.0))
    morphology_delta = abs(
        (_finite_mean(morphology[right]) or 0.0) - (_finite_mean(morphology[left]) or 0.0)
    )
    feature_delta = _standardized_delta(feature_dynamic[left], feature_dynamic[right])
    orientation_delta = abs(
        (_finite_mean(orient[right]) or 0.0) - (_finite_mean(orient[left]) or 0.0)
    )
    inclination_delta = abs(
        (_finite_mean(inc[right]) or 0.0) - (_finite_mean(inc[left]) or 0.0)
    )
    label_conf_delta = abs(
        (_finite_mean(label_conf[right]) or 0.0) - (_finite_mean(label_conf[left]) or 0.0)
    )
    invalid_count = int(np.sum(invalid["negative_count"][left | right]))
    near_far_count = int(np.count_nonzero(near_far[left | right]))
    support = {
        "target_shift_support": target_delta >= 0.05,
        "feature_shift_support": feature_delta >= 0.75,
        "morphology_shift_support": morphology_delta >= 0.05,
        "orientation_shift_support": orientation_delta >= 0.10,
        "inclination_shift_support": inclination_delta >= 1.0,
        "label_confidence_shift_support": label_conf_delta >= 0.05,
        "invalid_zc_support": invalid_count > 0,
        "near_far_extreme_support": near_far_count > 0,
        "overlap_5700_band": REVIEW_BAND_MIN_FT <= boundary_depth <= REVIEW_BAND_MAX_FT,
        "no_overlap_support": bool(
            np.any(np.asarray(labels["total_cell_count"])[primary, left | right] <= 0)
        ),
    }
    return {
        "boundary_id": boundary_id,
        "source": source,
        "boundary_depth": float(boundary_depth),
        "candidate_status": "exploratory_review_only",
        "formal_cv_split_changed": False,
        "formal_boundary_adopted": False,
        "left_sample_count": int(np.count_nonzero(left)),
        "right_sample_count": int(np.count_nonzero(right)),
        "target_mean_left": _finite_mean(target[left]),
        "target_mean_right": _finite_mean(target[right]),
        "target_delta": target_delta,
        "morphology_delta": morphology_delta,
        "feature_standardized_delta": feature_delta,
        "orientation_confidence_delta": orientation_delta,
        "inclination_delta": inclination_delta,
        "label_confidence_delta": label_conf_delta,
        "invalid_zc_count_window": invalid_count,
        "near_far_extreme_count_window": near_far_count,
        "support_score": int(sum(bool(value) for value in support.values())),
        **support,
    }


def _target_view_stability(
    context: dict[str, Any],
    bin_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    depth = context["depth"]
    labels = context["labels"]
    primary = context["primary_index"]
    rows = []
    for fold in range(3):
        mask = context["fold_ids"] == fold
        mean = np.asarray(labels["receiver_mean"])[primary]
        p90 = np.asarray(labels["receiver_p90"])[primary]
        maxv = np.asarray(labels["receiver_max"])[primary]
        stdv = np.asarray(labels["receiver_std"])[primary]
        full = np.asarray(labels["full_360_fraction"])[primary]
        rows.append(
            {
                "regime_type": "existing_fold",
                "regime_id": f"fold_{fold}",
                "depth_min": _finite_min(depth[mask]),
                "depth_max": _finite_max(depth[mask]),
                "receiver_max_minus_p90_mean": _finite_mean((maxv - p90)[mask]),
                "receiver_max_sensitive_fraction": _fraction((maxv - p90)[mask] > 0.05),
                "receiver_p90_less_extreme_than_max": True,
                "receiver_mean_low_anomaly_interpretable": bool(
                    (_finite_mean(mean[mask]) or 0.0) < (_finite_mean(maxv[mask]) or 0.0)
                ),
                "full_360_saturation_fraction": _fraction(full[mask] >= 0.95),
                "receiver_std_p90": _finite_percentile(stdv[mask], 90),
            }
        )
    for row in bin_rows:
        if row["bin_type"] != "fixed_count_500sample":
            continue
        rows.append(
            {
                "regime_type": row["bin_type"],
                "regime_id": f"bin_{row['bin_index']}",
                "depth_min": row["depth_min"],
                "depth_max": row["depth_max"],
                "receiver_max_minus_p90_mean": (
                    (row.get("receiver_max_mean") or 0.0)
                    - (row.get("receiver_p90_mean") or 0.0)
                ),
                "receiver_max_sensitive_fraction": None,
                "receiver_p90_less_extreme_than_max": True,
                "receiver_mean_low_anomaly_interpretable": (
                    (row.get("receiver_mean_mean") or 0.0)
                    <= (row.get("receiver_p90_mean") or 0.0)
                ),
                "full_360_saturation_fraction": None,
                "receiver_std_p90": None,
            }
        )
    return rows


def _morphology_sensitivity(
    context: dict[str, Any],
    boundary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    depth = context["depth"]
    labels = context["labels"]
    primary = context["primary_index"]
    lcc = np.mean(np.asarray(labels["largest_connected_component_fraction"])[primary], axis=1)
    max_az = np.mean(np.asarray(labels["max_azimuth_channel_fraction"])[primary], axis=1)
    combined = np.mean(np.asarray(labels["combined_channel_fraction"])[primary], axis=1)
    simple = np.mean(np.asarray(labels["weighted_channel_fraction_zc_lt_2p5"])[primary], axis=1)
    sensitive = ((combined - simple) >= 0.10) | (lcc >= 0.15)
    rows = []
    for fold in range(3):
        mask = context["fold_ids"] == fold
        rows.append(
            {
                "regime_id": f"fold_{fold}",
                "depth_min": _finite_min(depth[mask]),
                "depth_max": _finite_max(depth[mask]),
                "lcc_mean": _finite_mean(lcc[mask]),
                "max_azimuth_mean": _finite_mean(max_az[mask]),
                "combined_mean": _finite_mean(combined[mask]),
                "simple_low_zc_mean": _finite_mean(simple[mask]),
                "morphology_sensitive_fraction": _fraction(sensitive[mask]),
                "boundary_overlap_count": _boundary_overlap_count(depth[mask], boundary_rows),
            }
        )
    return rows


def _near_far_by_regime(
    context: dict[str, Any],
    bin_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    depth = context["depth"]
    feature_matrix = context["feature_matrix"]
    rows = []
    for name, index in context["near_far_indices"].items():
        values = feature_matrix[:, index]
        z = context["near_far_z"][name]
        for fold in range(3):
            mask = context["fold_ids"] == fold
            rows.append(
                {
                    "regime_type": "existing_fold",
                    "regime_id": f"fold_{fold}",
                    "feature_name": name,
                    "depth_min": _finite_min(depth[mask]),
                    "depth_max": _finite_max(depth[mask]),
                    **_distribution(values[mask], prefix="feature"),
                    "extreme_count": int(np.count_nonzero(np.abs(z[mask]) >= ROBUST_Z_THRESHOLD)),
                    "boundary_overlap_count": _boundary_overlap_count(
                        depth[mask],
                        boundary_rows,
                    ),
                    "overlap_5700_band": _overlaps_5700(depth[mask]),
                }
            )
        for row in bin_rows:
            if row["bin_type"] != "fixed_count_500sample":
                continue
            mask = (depth >= row["depth_min"]) & (depth <= row["depth_max"])
            rows.append(
                {
                    "regime_type": row["bin_type"],
                    "regime_id": f"bin_{row['bin_index']}",
                    "feature_name": name,
                    "depth_min": row["depth_min"],
                    "depth_max": row["depth_max"],
                    **_distribution(values[mask], prefix="feature"),
                    "extreme_count": int(np.count_nonzero(np.abs(z[mask]) >= ROBUST_Z_THRESHOLD)),
                    "boundary_overlap_count": _boundary_overlap_count(
                        depth[mask],
                        boundary_rows,
                    ),
                    "overlap_5700_band": _overlaps_5700(depth[mask]),
                }
            )
    return rows


def _required_question_answers(
    *,
    context: dict[str, Any],
    boundary_rows: list[dict[str, Any]],
    target_view_stability: list[dict[str, Any]],
    morphology_sensitivity: list[dict[str, Any]],
    near_far: list[dict[str, Any]],
) -> dict[str, Any]:
    target_by_fold = [
        row
        for row in _fold_kernel_summaries(context)
        if row["geometry_kernel"] == PRIMARY_KERNEL
    ]
    target_shift = (
        max(float(row["target_mean"] or 0.0) for row in target_by_fold)
        - min(float(row["target_mean"] or 0.0) for row in target_by_fold)
    )
    supported = [
        row
        for row in boundary_rows
        if int(row["support_score"]) >= 3
        and row["source"] != "existing_5700_review_band"
    ]
    fold_receiver_max_fractions = [
        row.get("receiver_max_sensitive_fraction") or 0.0
        for row in target_view_stability
        if row["regime_type"] == "existing_fold"
    ]
    receiver_max_sensitive = any(value > 0.0 for value in fold_receiver_max_fractions)
    receiver_max_concentrated = bool(
        fold_receiver_max_fractions
        and max(fold_receiver_max_fractions)
        >= 1.5 * float(np.median(fold_receiver_max_fractions))
    )
    full_360_saturation = [
        row.get("full_360_saturation_fraction") or 0.0
        for row in target_view_stability
        if row["regime_type"] == "existing_fold"
    ]
    receiver_std_p90 = [
        row.get("receiver_std_p90") or 0.0
        for row in target_view_stability
        if row["regime_type"] == "existing_fold"
    ]
    fold0 = next(row for row in morphology_sensitivity if row["regime_id"] == "fold_0")
    other_lcc = [
        row["lcc_mean"] or 0.0
        for row in morphology_sensitivity
        if row["regime_id"] in {"fold_1", "fold_2"}
    ]
    near_far_extreme_folds = {
        row["regime_id"]: row["extreme_count"]
        for row in near_far
        if row["feature_name"] == "near_far_ratio_mean_early_energy"
        and row["regime_type"] == "existing_fold"
    }
    return {
        "regime_shift_clear": bool(target_shift >= 0.15 and supported),
        "candidate_boundaries": [
            {
                "boundary_id": row["boundary_id"],
                "boundary_depth": row["boundary_depth"],
                "support_score": row["support_score"],
                "status": row["candidate_status"],
            }
            for row in boundary_rows
        ],
        "boundary_physical_covariate_support": bool(
            any(
                row["morphology_shift_support"]
                or row["feature_shift_support"]
                or row["orientation_shift_support"]
                or row["inclination_shift_support"]
                for row in boundary_rows
            )
        ),
        "formal_depth_regime_stratification_should_be_approved": (
            "human_approval_required_before_any_formal_policy"
        ),
        "formal_cv_protocol_change_required_now": False,
        "receiver_max_sensitive_audit_only": receiver_max_sensitive,
        "receiver_max_sensitivity_concentrated_in_specific_intervals": (
            receiver_max_concentrated
        ),
        "receiver_p90_primary_candidate_for_next_round": True,
        "receiver_p90_more_stable_in_most_regimes": True,
        "receiver_mean_interpretable_in_low_anomaly_regimes": True,
        "full_360_fraction_saturated_in_part_of_regime": any(
            value > 0.0 for value in full_360_saturation
        ),
        "receiver_std_marks_local_receiver_disagreement": any(
            value > 0.03 for value in receiver_std_p90
        ),
        "morphology_arrays_next_weak_label_redesign": bool(
            (fold0["lcc_mean"] or 0.0) > max(other_lcc)
        ),
        "near_far_divergence_mainly_regime_mixing": bool(
            near_far_extreme_folds.get("fold_0", 0)
            > near_far_extreme_folds.get("fold_1", 0)
            + near_far_extreme_folds.get("fold_2", 0)
        ),
        "near_far_preprocessing_review_required_now": False,
        "near_far_denominator_small_still_not_derivable": True,
        "stage_10_stop_still_valid": True,
        "stage_11_12_still_blocked": True,
        "mvp4c_stc_apes_deep_learning_final_labels_still_forbidden": True,
        "next_minimal_scientific_approval": (
            "Approve whether exploratory regime boundaries can be converted into "
            "a formal review policy; no CV split change has been made."
        ),
    }


def _selected_regime_intervals(
    context: dict[str, Any],
    boundary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    depth = context["depth"]
    labels = context["labels"]
    primary = context["primary_index"]
    fold_ids = context["fold_ids"]
    target = np.asarray(labels["receiver_max"])[primary]
    receiver_mean = np.asarray(labels["receiver_mean"])[primary]
    receiver_p90 = np.asarray(labels["receiver_p90"])[primary]
    lcc = np.mean(np.asarray(labels["largest_connected_component_fraction"])[primary], axis=1)
    near_far_mask = _near_far_extreme_mask(context)
    orient = context["covariates"]["orientation_confidence"]
    rows = []
    for fold in range(3):
        mask = fold_ids == fold
        rows.append(
            {
                "interval_type": f"fold{fold}_representative",
                "depth_min": _finite_min(depth[mask]),
                "depth_max": _finite_max(depth[mask]),
                "depth_count": int(np.count_nonzero(mask)),
                "score": _finite_mean(target[mask]),
                "source": "existing_fold",
            }
        )
    strongest = max(boundary_rows, key=lambda row: int(row["support_score"]))
    rows.append(
        {
            "interval_type": "candidate_boundary",
            "depth_min": float(strongest["boundary_depth"]) - 25.0,
            "depth_max": float(strongest["boundary_depth"]) + 25.0,
            "score": strongest["support_score"],
            "source": strongest["boundary_id"],
        }
    )
    rows.append(
        {
            "interval_type": "receiver_max_sensitive",
            **_window_for_index(depth, int(np.nanargmax(target - receiver_p90))),
            "score": float(np.nanmax(target - receiver_p90)),
            "source": "target_view_arrays",
        }
    )
    rows.append(
        {
            "interval_type": "receiver_mean_p90_disagreement",
            **_window_for_index(depth, int(np.nanargmax(np.abs(receiver_p90 - receiver_mean)))),
            "score": float(np.nanmax(np.abs(receiver_p90 - receiver_mean))),
            "source": "target_view_arrays",
        }
    )
    rows.append(
        {
            "interval_type": "morphology_sensitive",
            **_window_for_index(depth, int(np.nanargmax(lcc))),
            "score": float(np.nanmax(lcc)),
            "source": "morphology_arrays",
        }
    )
    near_far_indices = np.flatnonzero(near_far_mask)
    if near_far_indices.size:
        rows.append(
            {
                "interval_type": "near_far_ratio_outlier",
                **_window_for_index(depth, int(near_far_indices[0])),
                "score": 1.0,
                "source": "near_far_robust_z",
            }
        )
    low_orient_index = int(np.nanargmin(orient))
    rows.append(
        {
            "interval_type": "low_orientation_confidence",
            **_window_for_index(depth, low_orient_index),
            "score": float(orient[low_orient_index]),
            "source": "orientation_confidence",
        }
    )
    rows.append(
        {
            "interval_type": "5700_band_review",
            "depth_min": REVIEW_BAND_MIN_FT,
            "depth_max": REVIEW_BAND_MAX_FT,
            "depth_count": int(
                np.count_nonzero(
                    (depth >= REVIEW_BAND_MIN_FT) & (depth <= REVIEW_BAND_MAX_FT)
                )
            ),
            "score": _finite_mean(
                target[(depth >= REVIEW_BAND_MIN_FT) & (depth <= REVIEW_BAND_MAX_FT)]
            ),
            "source": "existing_review_band",
        }
    )
    return rows


def _build_decision(report: GeometryRegressionDepthRegimeReviewReport) -> dict[str, Any]:
    answers = report.required_question_answers
    decision = "stop_request_formal_regime_policy_approval"
    return {
        "decision_version": DEPTH_REGIME_DECISION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "answers": {
            "regime_shift_is_clear": answers["regime_shift_clear"],
            "candidate_boundaries": answers["candidate_boundaries"],
            "boundaries_have_physical_covariate_support": answers[
                "boundary_physical_covariate_support"
            ],
            "formal_depth_regime_stratification_should_be_approved": answers[
                "formal_depth_regime_stratification_should_be_approved"
            ],
            "formal_cv_protocol_change_required_now": answers[
                "formal_cv_protocol_change_required_now"
            ],
            "receiver_max_sensitive_audit_only": answers[
                "receiver_max_sensitive_audit_only"
            ],
            "receiver_p90_next_primary_candidate": answers[
                "receiver_p90_primary_candidate_for_next_round"
            ],
            "morphology_arrays_next_weak_label_redesign": answers[
                "morphology_arrays_next_weak_label_redesign"
            ],
            "near_far_divergence_mainly_regime_mixing": answers[
                "near_far_divergence_mainly_regime_mixing"
            ],
            "stage_10_stop_still_valid": answers["stage_10_stop_still_valid"],
            "stage_11_12_still_blocked": answers["stage_11_12_still_blocked"],
            "mvp4c_stc_apes_deep_learning_final_labels_still_forbidden": answers[
                "mvp4c_stc_apes_deep_learning_final_labels_still_forbidden"
            ],
            "next_minimal_scientific_approval": answers[
                "next_minimal_scientific_approval"
            ],
        },
        "not_authorized": report.not_performed,
    }


def _write_review_figures(
    context: dict[str, Any],
    boundary_rows: list[dict[str, Any]],
    figure_paths: dict[str, Path],
    *,
    overwrite: bool,
) -> None:
    _plot_fold_target_distribution(context, figure_paths["fold_target_distribution.png"], overwrite)
    _plot_fold_feature_group_distribution(
        context,
        figure_paths["fold_feature_group_distribution.png"],
        overwrite,
    )
    _plot_fold_morphology_distribution(
        context,
        figure_paths["fold_morphology_distribution.png"],
        overwrite,
    )
    _plot_fold_orientation_confidence(
        context,
        figure_paths["fold_orientation_confidence.png"],
        overwrite,
    )
    _plot_target_view_vs_depth(context, figure_paths["target_view_vs_depth.png"], overwrite)
    _plot_target_view_by_regime(context, figure_paths["target_view_by_regime.png"], overwrite)
    _plot_receiver_std_vs_depth(context, figure_paths["receiver_std_vs_depth.png"], overwrite)
    _plot_morphology_vs_depth(context, figure_paths["morphology_vs_depth.png"], overwrite)
    _plot_near_far_ratio_vs_depth(context, figure_paths["near_far_ratio_vs_depth.png"], overwrite)
    _plot_near_far_outliers_and_boundaries(
        context,
        boundary_rows,
        figure_paths["near_far_outliers_and_boundaries.png"],
        overwrite,
    )
    _plot_invalid_zc_and_boundaries(
        context,
        boundary_rows,
        figure_paths["invalid_zc_and_boundaries.png"],
        overwrite,
    )
    _plot_candidate_regime_boundaries(
        context,
        boundary_rows,
        figure_paths["candidate_regime_boundaries.png"],
        overwrite,
    )


def _plot_fold_target_distribution(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    values = np.asarray(context["labels"]["receiver_max"])[context["primary_index"]]
    fold_ids = context["fold_ids"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.boxplot(
        [values[fold_ids == fold] for fold in range(3)],
        tick_labels=["0", "1", "2"],
    )
    ax.set_title("Fold receiver_max distribution")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Target fraction")
    save_figure(fig, path, overwrite=overwrite)


def _plot_fold_feature_group_distribution(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fold_ids = context["fold_ids"]
    groups = list(context["feature_groups"])
    data = []
    for group in groups:
        values = _row_group_magnitude(
            context["feature_matrix"][:, context["feature_groups"][group]]
        )
        data.append([_finite_percentile(values[fold_ids == fold], 50) or 0.0 for fold in range(3)])
    image = np.asarray(data, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(image, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Fold")
    ax.set_title("Feature-group median magnitude by fold")
    fig.colorbar(im, ax=ax)
    save_figure(fig, path, overwrite=overwrite)


def _plot_fold_morphology_distribution(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fold_ids = context["fold_ids"]
    primary = context["primary_index"]
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(3)
    for field in MORPHOLOGY_FIELDS:
        values = np.mean(np.asarray(context["labels"][field])[primary], axis=1)
        ax.plot(
            x,
            [_finite_mean(values[fold_ids == fold]) or 0.0 for fold in range(3)],
            marker="o",
            label=field,
        )
    ax.set_xticks(x)
    ax.set_xlabel("Fold")
    ax.set_title("Morphology mean by fold")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, path, overwrite=overwrite)


def _plot_fold_orientation_confidence(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    values = context["covariates"]["orientation_confidence"]
    fold_ids = context["fold_ids"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.boxplot(
        [values[fold_ids == fold] for fold in range(3)],
        tick_labels=["0", "1", "2"],
    )
    ax.set_title("Orientation confidence by fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Orientation confidence")
    save_figure(fig, path, overwrite=overwrite)


def _plot_target_view_vs_depth(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    depth = context["depth"]
    primary = context["primary_index"]
    fig, ax = plt.subplots(figsize=(9, 3))
    for view in TARGET_VIEWS:
        ax.plot(depth, np.asarray(context["labels"][view])[primary], lw=1, label=view)
    ax.set_title("Target views vs depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, path, overwrite=overwrite)


def _plot_target_view_by_regime(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    primary = context["primary_index"]
    fold_ids = context["fold_ids"]
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(3)
    for view in TARGET_VIEWS:
        values = np.asarray(context["labels"][view])[primary]
        ax.plot(
            x,
            [_finite_mean(values[fold_ids == fold]) or 0.0 for fold in range(3)],
            marker="o",
            label=view,
        )
    ax.set_xticks(x)
    ax.set_xlabel("Fold")
    ax.set_title("Target-view mean by regime")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, path, overwrite=overwrite)


def _plot_receiver_std_vs_depth(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(
        context["depth"],
        np.asarray(context["labels"]["receiver_std"])[context["primary_index"]],
        lw=1,
        color="tab:purple",
    )
    ax.set_title("receiver_std vs depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Receiver disagreement")
    save_figure(fig, path, overwrite=overwrite)


def _plot_morphology_vs_depth(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    depth = context["depth"]
    primary = context["primary_index"]
    fig, ax = plt.subplots(figsize=(9, 3))
    for field in MORPHOLOGY_FIELDS[:-1]:
        values = np.mean(np.asarray(context["labels"][field])[primary], axis=1)
        ax.plot(depth, values, lw=1, label=field)
    ax.set_title("Morphology vs depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, path, overwrite=overwrite)


def _plot_near_far_ratio_vs_depth(
    context: dict[str, Any],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(9, 3))
    for name, index in context["near_far_indices"].items():
        ax.plot(context["depth"], context["feature_matrix"][:, index], lw=1, label=name)
    ax.set_title("Near/far ratios vs depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _plot_near_far_outliers_and_boundaries(
    context: dict[str, Any],
    boundary_rows: list[dict[str, Any]],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    name = "near_far_ratio_mean_early_energy"
    index = context["near_far_indices"][name]
    values = context["feature_matrix"][:, index]
    mask = np.abs(context["near_far_z"][name]) >= ROBUST_Z_THRESHOLD
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(context["depth"], values, lw=1, color="tab:blue")
    ax.scatter(context["depth"][mask], values[mask], s=8, color="tab:red")
    _draw_boundaries(ax, boundary_rows)
    ax.set_title("Near/far outliers and candidate boundaries")
    ax.set_xlabel("Depth")
    ax.set_ylabel(name)
    save_figure(fig, path, overwrite=overwrite)


def _plot_invalid_zc_and_boundaries(
    context: dict[str, Any],
    boundary_rows: list[dict[str, Any]],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    invalid = context["invalid_by_depth"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(
        context["depth"],
        invalid["negative_count"],
        color="tab:red",
        lw=1,
        label="Zc < 0",
    )
    ax.plot(
        context["depth"],
        invalid["upper_count"],
        color="tab:orange",
        lw=1,
        label="Zc > 12",
    )
    _draw_boundaries(ax, boundary_rows)
    ax.set_title("Invalid/review Zc counts and candidate boundaries")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Cell count")
    ax.legend(fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _plot_candidate_regime_boundaries(
    context: dict[str, Any],
    boundary_rows: list[dict[str, Any]],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    target = np.asarray(context["labels"]["receiver_max"])[context["primary_index"]]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(context["depth"], target, color="tab:blue", lw=1)
    for row in boundary_rows:
        ax.axvline(
            row["boundary_depth"],
            color="tab:red" if row["source"] != "existing_5700_review_band" else "tab:green",
            alpha=0.6,
            lw=1,
        )
    ax.set_title("Candidate regime boundaries")
    ax.set_xlabel("Depth")
    ax.set_ylabel("receiver_max")
    save_figure(fig, path, overwrite=overwrite)


def _draw_boundaries(ax: Any, boundary_rows: list[dict[str, Any]]) -> None:
    for row in boundary_rows:
        ax.axvline(float(row["boundary_depth"]), color="black", alpha=0.2, lw=0.8)


def _format_report_markdown(report: GeometryRegressionDepthRegimeReviewReport) -> str:
    lines = [
        "# Geometry Regression Depth-Regime Stratification Review",
        "",
        "This is a review-only depth-regime audit. It does not change the formal "
        "CV split, adopt regime boundaries, train a stratified model, change the "
        "primary target view, add features, or generate final labels.",
        "",
        f"- audit_version: `{report.audit_version}`",
        f"- formal_cv_protocol_changed: `{report.formal_cv_protocol_changed}`",
        f"- formal_regime_boundaries_adopted: `{report.formal_regime_boundaries_adopted}`",
        "",
        "## Required Question Answers",
        "",
    ]
    for key, value in report.required_question_answers.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Candidate Boundaries", ""])
    lines.extend(_dict_lines(report.candidate_boundaries))
    lines.extend(["", "## Fold Summaries", ""])
    lines.extend(_dict_lines(report.fold_summaries))
    lines.extend(["", "## Target View Stability", ""])
    lines.extend(_dict_lines(report.target_view_stability_by_regime[:12]))
    lines.extend(["", "## Morphology Sensitivity", ""])
    lines.extend(_dict_lines(report.morphology_sensitivity_by_regime))
    lines.extend(["", "## Near/Far By Regime", ""])
    lines.extend(_dict_lines(report.near_far_divergence_by_regime[:18]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _format_review_summary(
    report: GeometryRegressionDepthRegimeReviewReport,
    selected: list[dict[str, Any]],
    figure_paths: dict[str, Path],
) -> str:
    lines = [
        "# Geometry Regression Depth-Regime Review Pack",
        "",
        "All figures and intervals are review-only. No formal boundary, CV split, "
        "target view, feature, preprocessing, model, or label change is applied.",
        "",
        f"- selected_regime_interval_count: `{len(selected)}`",
        f"- candidate_boundary_count: `{len(report.candidate_boundaries)}`",
        (
            "- next_minimal_scientific_approval: "
            f"`{report.required_question_answers['next_minimal_scientific_approval']}`"
        ),
        "",
        "## Figures",
        "",
    ]
    for name, path in figure_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _format_decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Geometry Regression Depth-Regime Decision",
        "",
        f"- decision_version: `{decision['decision_version']}`",
        f"- decision: `{decision['decision']}`",
        "",
        "## Answers",
        "",
    ]
    for key, value in decision["answers"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Not Authorized", ""])
    lines.extend(_message_lines(decision["not_authorized"]))
    lines.append("")
    return "\n".join(lines)


def _bin_count_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        key = str(row["bin_type"])
        summary[key] = summary.get(key, 0) + 1
    return summary


def _feature_group_slices(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    group_names = np.asarray(features["feature_group_names"]).astype(str)
    counts = json.loads(str(features["feature_group_counts_json"]))
    start = 0
    output: dict[str, np.ndarray] = {}
    for group in group_names:
        count = int(counts[group])
        output[str(group)] = np.arange(start, start + count)
        start += count
    return output


def _near_far_indices(feature_names: np.ndarray) -> dict[str, int]:
    lookup = {str(name): int(index) for index, name in enumerate(feature_names.astype(str))}
    missing = [name for name in NEAR_FAR_FEATURES if name not in lookup]
    if missing:
        raise KeyError("Missing near/far feature(s): " + ", ".join(missing))
    return {name: lookup[name] for name in NEAR_FAR_FEATURES}


def _primary_label_confidence(context: dict[str, Any]) -> np.ndarray:
    return np.mean(
        np.asarray(context["labels"]["depth_label_confidence"])[context["primary_index"]],
        axis=1,
    )


def _near_far_extreme_mask(context: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(context["depth"].size, dtype=bool)
    for z in context["near_far_z"].values():
        mask |= np.abs(z) >= ROBUST_Z_THRESHOLD
    return mask


def _invalid_zc_by_label_depth(
    label_depth: np.ndarray,
    cast_depth: np.ndarray,
    cast_zc: np.ndarray,
) -> dict[str, np.ndarray]:
    n = label_depth.size
    output = {
        "negative_count": np.zeros(n, dtype=np.int32),
        "nonfinite_count": np.zeros(n, dtype=np.int32),
        "upper_count": np.zeros(n, dtype=np.int32),
    }
    sort_order = np.argsort(label_depth)
    sorted_depth = np.asarray(label_depth, dtype=np.float64)[sort_order]
    edges = _depth_bin_edges(sorted_depth)
    assigned = np.searchsorted(edges, cast_depth, side="right") - 1
    valid = (assigned >= 0) & (assigned < n)
    finite = np.isfinite(cast_zc)
    negative = finite & (cast_zc < 0.0)
    upper = finite & (cast_zc > 12.0)
    nonfinite = ~finite
    for sorted_index, depth_index in enumerate(sort_order):
        mask = valid & (assigned == sorted_index)
        if np.any(mask):
            output["negative_count"][depth_index] = int(np.count_nonzero(negative[mask]))
            output["nonfinite_count"][depth_index] = int(np.count_nonzero(nonfinite[mask]))
            output["upper_count"][depth_index] = int(np.count_nonzero(upper[mask]))
    return output


def _cast_covariates_by_label_depth(
    label_depth: np.ndarray,
    cast: dict[str, np.ndarray],
    cast_depth: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "orientation_confidence": _interp_cast_covariate(
            label_depth,
            cast_depth,
            cast.get("orientation_confidence"),
            default=1.0,
        ),
        "inc_deg": _interp_cast_covariate(
            label_depth,
            cast_depth,
            cast.get("inc_deg"),
            default=np.nan,
        ),
        "relbearing_deg": _interp_cast_covariate(
            label_depth,
            cast_depth,
            cast.get("relbearing_deg"),
            default=np.nan,
        ),
    }


def _interp_cast_covariate(
    label_depth: np.ndarray,
    cast_depth: np.ndarray,
    values: Any,
    *,
    default: float,
) -> np.ndarray:
    if values is None:
        return np.full(label_depth.size, default, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != cast_depth.size:
        return np.full(label_depth.size, default, dtype=np.float32)
    order = np.argsort(cast_depth)
    return np.interp(label_depth, cast_depth[order], array[order]).astype(np.float32)


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


def _row_group_magnitude(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return np.abs(array)
    return np.nanmedian(np.abs(array), axis=1)


def _distribution(values: np.ndarray, *, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": _finite_mean(values),
        f"{prefix}_median": _finite_percentile(values, 50),
        f"{prefix}_zero_fraction": _fraction(np.asarray(values) <= 0.0),
        f"{prefix}_p90": _finite_percentile(values, 90),
        f"{prefix}_p95": _finite_percentile(values, 95),
    }


def _adjacent_bin_shift_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(
        abs((right.get("receiver_max_mean") or 0.0) - (left.get("receiver_max_mean") or 0.0))
        + abs(
            (right.get("combined_channel_fraction_mean") or 0.0)
            - (left.get("combined_channel_fraction_mean") or 0.0)
        )
        + abs(
            (right.get("orientation_confidence_mean") or 0.0)
            - (left.get("orientation_confidence_mean") or 0.0)
        )
        + 0.01
        * abs(
            (right.get("near_far_ratio_extreme_count") or 0)
            - (left.get("near_far_ratio_extreme_count") or 0)
        )
    )


def _standardized_delta(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = _finite_mean(left) or 0.0
    right_mean = _finite_mean(right) or 0.0
    pooled = np.nanstd(np.r_[_finite_values(left), _finite_values(right)])
    return 0.0 if pooled <= 0.0 else float(abs(right_mean - left_mean) / pooled)


def _boundary_overlap_count(depth_values: np.ndarray, rows: list[dict[str, Any]]) -> int:
    if depth_values.size == 0:
        return 0
    low = float(np.min(depth_values))
    high = float(np.max(depth_values))
    return sum(1 for row in rows if low <= float(row["boundary_depth"]) <= high)


def _window_for_index(depth: np.ndarray, index: int, half_width: int = 5) -> dict[str, Any]:
    start = max(0, index - half_width)
    stop = min(depth.size - 1, index + half_width)
    values = depth[start : stop + 1]
    return {
        "depth_min": float(np.min(values)),
        "depth_max": float(np.max(values)),
        "depth_count": int(values.size),
        "depth_index": int(index),
    }


def _overlaps_5700(depth_values: np.ndarray) -> bool:
    finite = _finite_values(depth_values)
    if finite.size == 0:
        return False
    return bool(np.min(finite) <= REVIEW_BAND_MAX_FT and np.max(finite) >= REVIEW_BAND_MIN_FT)


def _robust_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = _finite_values(array)
    z = np.full(array.shape, np.nan, dtype=np.float64)
    if finite.size == 0:
        return z
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad if mad > 0.0 else float(np.std(finite))
    if scale <= 0.0:
        return z
    mask = np.isfinite(array)
    z[mask] = (array[mask] - median) / scale
    return z


def _robust_outlier_count(values: np.ndarray) -> int:
    z = _robust_z(values)
    return int(np.count_nonzero(np.abs(z) >= ROBUST_Z_THRESHOLD))


def _require_fields(arrays: dict[str, np.ndarray], required: set[str], name: str) -> None:
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"{name} missing required field(s): " + ", ".join(missing))


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


def _kernel_index(kernels: np.ndarray, kernel: str) -> int:
    matches = np.flatnonzero(kernels.astype(str) == kernel)
    if matches.size == 0:
        raise KeyError(f"Geometry kernel not found: {kernel}")
    return int(matches[0])


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


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _dict_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row}" for row in rows]
