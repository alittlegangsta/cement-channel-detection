from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CAST_ZC_PHYSICAL_QC_VERSION = "cast_zc_physical_qc_v001"
DEFAULT_PLAUSIBLE_UPPER_BOUND_MRAYL = 12.0
DEFAULT_ZC_THRESHOLD_MRAYL = 2.5
DEFAULT_FOLD_COUNT = 3


@dataclass(frozen=True)
class CastZcPhysicalQcConfig:
    zc_threshold_mrayl: float = DEFAULT_ZC_THRESHOLD_MRAYL
    plausible_upper_bound_mrayl: float = DEFAULT_PLAUSIBLE_UPPER_BOUND_MRAYL
    fold_count: int = DEFAULT_FOLD_COUNT
    practical_delta_threshold: float = 0.001
    significant_practical_affected_depth_fraction: float = 0.01
    significant_mean_abs_delta: float = 1e-4
    significant_max_abs_delta: float = 0.05


@dataclass(frozen=True)
class CastZcPhysicalQcReport:
    qc_version: str
    generated_at: str
    inputs: dict[str, str]
    raw_zc_shape: list[int]
    label_target_shape: list[int]
    zc_threshold_mrayl: float
    plausible_upper_bound_mrayl: float
    global_cell_counts: dict[str, Any]
    depth_summary: dict[str, Any]
    azimuth_summary: dict[str, Any]
    fold_summaries: list[dict[str, Any]]
    kernel_summaries: list[dict[str, Any]]
    receiver_summaries: list[dict[str, Any]]
    boundary_no_overlap_summary: dict[str, Any]
    counterfactual_summary: dict[str, Any]
    negative_zc_target_influence_significant: bool
    warnings: list[str]
    errors: list[str]
    no_formal_mask_applied: bool
    no_threshold_change: bool
    no_final_labels: bool
    no_model_training: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_cast_zc_physical_qc(
    *,
    cast_arrays: dict[str, np.ndarray],
    label_arrays: dict[str, np.ndarray],
    config: CastZcPhysicalQcConfig | None = None,
    inputs: dict[str, str] | None = None,
) -> tuple[CastZcPhysicalQcReport, list[dict[str, Any]]]:
    cfg = config or CastZcPhysicalQcConfig()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_inputs(cast_arrays, label_arrays)

    cast_depth = np.asarray(cast_arrays["cast_depth"], dtype=np.float32).reshape(-1)
    cast_zc = np.asarray(cast_arrays["cast_zc"], dtype=np.float32)
    cast_azimuth = np.asarray(
        cast_arrays.get("cast_azimuth_deg", np.arange(cast_zc.shape[1])),
        dtype=np.float32,
    ).reshape(-1)
    label_depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    kernels = np.asarray(label_arrays["geometry_kernel"]).astype(str)
    receiver_index = np.asarray(label_arrays["receiver_index"]).astype(int)
    original_target = np.asarray(
        label_arrays["weighted_channel_fraction_zc_lt_2p5"],
        dtype=np.float32,
    )
    interval_min = np.asarray(label_arrays["interval_min_depth"], dtype=np.float32)
    interval_max = np.asarray(label_arrays["interval_max_depth"], dtype=np.float32)
    midpoint = np.asarray(label_arrays["midpoint_depth"], dtype=np.float32)
    total_cell_count = np.asarray(label_arrays["total_cell_count"], dtype=np.int64)

    sorted_order = np.argsort(cast_depth)
    cast_depth = cast_depth[sorted_order]
    cast_zc = cast_zc[sorted_order]
    finite = np.isfinite(cast_zc)
    negative = finite & (cast_zc < 0.0)
    zero = finite & (cast_zc == 0.0)
    low_positive = finite & (cast_zc > 0.0) & (cast_zc < cfg.zc_threshold_mrayl)
    high_or_equal = finite & (cast_zc >= cfg.zc_threshold_mrayl)
    upper_exceed = finite & (cast_zc > cfg.plausible_upper_bound_mrayl)
    nonfinite = ~finite
    original_candidate = finite & (cast_zc < cfg.zc_threshold_mrayl)

    context = _CounterfactualContext.from_masks(
        cast_depth=cast_depth,
        finite=finite,
        negative=negative,
        original_candidate=original_candidate,
    )
    counterfactual = _counterfactual_targets(
        original_target=original_target,
        interval_min=interval_min,
        interval_max=interval_max,
        midpoint=midpoint,
        kernels=kernels,
        context=context,
    )
    absolute_delta = np.abs(counterfactual - original_target)
    relative_delta = np.divide(
        absolute_delta,
        np.abs(original_target),
        out=np.zeros_like(absolute_delta, dtype=np.float32),
        where=np.abs(original_target) > 0.0,
    )
    numerical_delta_nonzero = absolute_delta > 1e-8
    affected = absolute_delta >= cfg.practical_delta_threshold
    no_overlap = total_cell_count <= 0
    possible_cells = _possible_cell_count(
        interval_min,
        interval_max,
        cast_depth,
        cast_zc.shape[1],
    )
    finite_coverage = np.divide(
        total_cell_count,
        np.maximum(possible_cells, 1),
        out=np.zeros_like(total_cell_count, dtype=np.float64),
        where=True,
    )

    global_counts = _global_counts(
        negative=negative,
        zero=zero,
        low_positive=low_positive,
        high_or_equal=high_or_equal,
        nonfinite=nonfinite,
        upper_exceed=upper_exceed,
        total_cells=cast_zc.size,
    )
    fold_ids = _fold_ids(label_depth, cfg.fold_count)
    fold_summaries = _fold_summaries(
        label_depth=label_depth,
        cast_depth=cast_depth,
        negative_by_cast_depth=np.count_nonzero(negative, axis=1),
        nonfinite_by_cast_depth=np.count_nonzero(nonfinite, axis=1),
        upper_exceed_by_cast_depth=np.count_nonzero(upper_exceed, axis=1),
        target=original_target,
        counterfactual=counterfactual,
        affected=affected,
        no_overlap=no_overlap,
        fold_ids=fold_ids,
    )
    kernel_summaries = _kernel_summaries(
        kernels=kernels,
        target=original_target,
        counterfactual=counterfactual,
        affected=affected,
        no_overlap=no_overlap,
        finite_coverage=finite_coverage,
    )
    receiver_summaries = _receiver_summaries(
        receiver_index=receiver_index,
        target=original_target,
        counterfactual=counterfactual,
        affected=affected,
        no_overlap=no_overlap,
        finite_coverage=finite_coverage,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(
        _depth_rows(
            cast_depth,
            cast_zc,
            finite,
            negative,
            zero,
            low_positive,
            high_or_equal,
            nonfinite,
            upper_exceed,
        )
    )
    rows.extend(
        _azimuth_rows(
            cast_azimuth,
            cast_zc,
            finite,
            negative,
            zero,
            low_positive,
            high_or_equal,
            nonfinite,
            upper_exceed,
        )
    )
    rows.extend({"section": "fold_summary", **row} for row in fold_summaries)
    rows.extend({"section": "kernel_summary", **row} for row in kernel_summaries)
    rows.extend({"section": "receiver_summary", **row} for row in receiver_summaries)

    counterfactual_summary = {
        "original_target_mean": _finite_mean(original_target),
        "counterfactual_target_mean_excluding_zc_lt_0": _finite_mean(counterfactual),
        "mean_abs_delta": _finite_mean(absolute_delta),
        "max_abs_delta": _finite_max(absolute_delta),
        "mean_relative_delta_where_original_positive": _finite_mean(
            relative_delta[original_target > 0.0]
        ),
        "max_relative_delta_where_original_positive": _finite_max(
            relative_delta[original_target > 0.0]
        ),
        "practical_delta_threshold": cfg.practical_delta_threshold,
        "numerical_delta_nonzero_target_cell_fraction": _fraction(
            numerical_delta_nonzero
        ),
        "affected_target_cell_fraction": _fraction(affected),
        "affected_depth_fraction": _affected_depth_fraction(affected),
        "affected_receiver_fraction": _affected_receiver_fraction(affected),
        "affected_fold_fraction": _affected_fold_fraction(affected, fold_ids),
        "counterfactual_formula": (
            "review-only recomputation of weighted_channel_fraction_zc_lt_2p5 "
            "after removing Zc < 0 cells from candidate numerator and finite denominator"
        ),
    }
    boundary_summary = {
        "region_count": int(no_overlap.size),
        "no_overlap_region_count": int(np.count_nonzero(no_overlap)),
        "no_overlap_region_fraction": _fraction(no_overlap),
        "finite_coverage_mean": _finite_mean(finite_coverage),
        "finite_coverage_min": _finite_min(finite_coverage),
        "finite_coverage_p05": _finite_percentile(finite_coverage, 5.0),
        "finite_coverage_below_0p5_fraction": _fraction(finite_coverage < 0.5),
    }
    significant = bool(
        (counterfactual_summary["mean_abs_delta"] or 0.0) >= cfg.significant_mean_abs_delta
        or (counterfactual_summary["affected_depth_fraction"] or 0.0)
        >= cfg.significant_practical_affected_depth_fraction
        or (counterfactual_summary["max_abs_delta"] or 0.0)
        >= cfg.significant_max_abs_delta
    )
    if significant:
        warnings.append(
            "Zc < 0 has review-significant target influence; formal invalid-Zc "
            "mask requires human approval."
        )

    report = CastZcPhysicalQcReport(
        qc_version=CAST_ZC_PHYSICAL_QC_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        raw_zc_shape=list(cast_zc.shape),
        label_target_shape=list(original_target.shape),
        zc_threshold_mrayl=cfg.zc_threshold_mrayl,
        plausible_upper_bound_mrayl=cfg.plausible_upper_bound_mrayl,
        global_cell_counts=global_counts,
        depth_summary=_depth_summary(cast_depth, negative, nonfinite, upper_exceed),
        azimuth_summary=_azimuth_summary(cast_azimuth, negative, nonfinite, upper_exceed),
        fold_summaries=fold_summaries,
        kernel_summaries=kernel_summaries,
        receiver_summaries=receiver_summaries,
        boundary_no_overlap_summary=boundary_summary,
        counterfactual_summary=counterfactual_summary,
        negative_zc_target_influence_significant=significant,
        warnings=warnings,
        errors=errors,
        no_formal_mask_applied=True,
        no_threshold_change=True,
        no_final_labels=True,
        no_model_training=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "formal invalid-Zc masking",
            "Zc threshold change",
            "target semantics change",
            "final label generation",
            "model training",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, rows


def audit_cast_zc_physical_qc_from_paths(
    *,
    cast_npz: Path | str,
    labels_npz: Path | str,
    output_md: Path | str,
    output_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
    config: CastZcPhysicalQcConfig | None = None,
) -> CastZcPhysicalQcReport:
    cast_arrays = _load_npz(Path(cast_npz))
    label_arrays = _load_npz(Path(labels_npz))
    report, rows = audit_cast_zc_physical_qc(
        cast_arrays=cast_arrays,
        label_arrays=label_arrays,
        config=config,
        inputs={"cast_npz": str(cast_npz), "labels_npz": str(labels_npz)},
    )
    write_cast_zc_physical_qc_outputs(
        report,
        rows,
        output_md=Path(output_md),
        output_json=Path(output_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def write_cast_zc_physical_qc_outputs(
    report: CastZcPhysicalQcReport,
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
    output_md.write_text(format_cast_zc_physical_qc_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_cast_zc_physical_qc_markdown(report: CastZcPhysicalQcReport) -> str:
    lines = [
        "# CAST Zc Physical QC Audit",
        "",
        "This is an audit-only physical QC report for controlled raw CAST Zc and "
        "continuous geometry regression weak-label targets. It does not apply an "
        "invalid-Zc mask, change the 2.5 MRayl threshold, generate final labels, "
        "or authorize MVP-4C/STC/APES/deep learning.",
        "",
        f"- qc_version: `{report.qc_version}`",
        f"- zc_threshold_mrayl: `{report.zc_threshold_mrayl}`",
        f"- plausible_upper_bound_mrayl: `{report.plausible_upper_bound_mrayl}`",
        f"- raw_zc_shape: `{report.raw_zc_shape}`",
        f"- label_target_shape: `{report.label_target_shape}`",
        "- negative_zc_target_influence_significant: "
        f"`{report.negative_zc_target_influence_significant}`",
        f"- no_formal_mask_applied: `{report.no_formal_mask_applied}`",
        "",
        "## Global Cell Counts",
        "",
    ]
    lines.extend(_dict_lines(report.global_cell_counts))
    lines.extend(["", "## Boundary / Coverage", ""])
    lines.extend(_dict_lines(report.boundary_no_overlap_summary))
    lines.extend(["", "## Counterfactual Target Summary", ""])
    lines.extend(_dict_lines(report.counterfactual_summary))
    lines.extend(["", "## Fold Summaries", ""])
    lines.extend(_dict_message_lines(report.fold_summaries))
    lines.extend(["", "## Kernel Summaries", ""])
    lines.extend(_dict_message_lines(report.kernel_summaries))
    lines.extend(["", "## Receiver Summaries", ""])
    lines.extend(_dict_message_lines(report.receiver_summaries))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class _CounterfactualContext:
    depth: np.ndarray
    finite_count_by_depth: np.ndarray
    negative_count_by_depth: np.ndarray
    original_candidate_count_by_depth: np.ndarray
    finite_cumsum: np.ndarray
    negative_cumsum: np.ndarray
    original_candidate_cumsum: np.ndarray

    @classmethod
    def from_masks(
        cls,
        *,
        cast_depth: np.ndarray,
        finite: np.ndarray,
        negative: np.ndarray,
        original_candidate: np.ndarray,
    ) -> _CounterfactualContext:
        finite_count = np.count_nonzero(finite, axis=1).astype(np.int64)
        negative_count = np.count_nonzero(negative, axis=1).astype(np.int64)
        candidate_count = np.count_nonzero(original_candidate, axis=1).astype(np.int64)
        return cls(
            depth=cast_depth,
            finite_count_by_depth=finite_count,
            negative_count_by_depth=negative_count,
            original_candidate_count_by_depth=candidate_count,
            finite_cumsum=_count_cumsum(finite_count),
            negative_cumsum=_count_cumsum(negative_count),
            original_candidate_cumsum=_count_cumsum(candidate_count),
        )


def _counterfactual_targets(
    *,
    original_target: np.ndarray,
    interval_min: np.ndarray,
    interval_max: np.ndarray,
    midpoint: np.ndarray,
    kernels: np.ndarray,
    context: _CounterfactualContext,
) -> np.ndarray:
    output = np.zeros_like(original_target, dtype=np.float32)
    for index in np.ndindex(original_target.shape):
        kernel = str(kernels[index[0]])
        lower = float(interval_min[index])
        upper = float(interval_max[index])
        middle = float(midpoint[index])
        start, stop = _depth_region_bounds(context.depth, lower, upper)
        if stop <= start:
            output[index] = 0.0
            continue
        if kernel == "triangular_midpoint_weighted":
            output[index] = _weighted_counterfactual_fraction(
                start,
                stop,
                lower,
                upper,
                midpoint=middle,
                context=context,
            )
        else:
            output[index] = _unweighted_counterfactual_fraction(start, stop, context)
    return output


def _unweighted_counterfactual_fraction(
    start: int,
    stop: int,
    context: _CounterfactualContext,
) -> float:
    finite = _range_sum(context.finite_cumsum, start, stop)
    candidate = _range_sum(context.original_candidate_cumsum, start, stop)
    negative = _range_sum(context.negative_cumsum, start, stop)
    denominator = finite - negative
    if denominator <= 0:
        return 0.0
    return float(max(candidate - negative, 0) / denominator)


def _weighted_counterfactual_fraction(
    start: int,
    stop: int,
    lower: float,
    upper: float,
    *,
    midpoint: float,
    context: _CounterfactualContext,
) -> float:
    weights = _depth_weights(
        context.depth[start:stop],
        lower=lower,
        upper=upper,
        midpoint=midpoint,
    )
    denominator = float(
        np.sum(
            weights
            * (
                context.finite_count_by_depth[start:stop]
                - context.negative_count_by_depth[start:stop]
            )
        )
    )
    if denominator <= 0.0:
        return 0.0
    numerator = float(
        np.sum(
            weights
            * (
                context.original_candidate_count_by_depth[start:stop]
                - context.negative_count_by_depth[start:stop]
            )
        )
    )
    return max(numerator, 0.0) / denominator


def _validate_inputs(
    cast_arrays: dict[str, np.ndarray],
    label_arrays: dict[str, np.ndarray],
) -> None:
    required_cast = {"cast_depth", "cast_zc"}
    required_labels = {
        "depth",
        "geometry_kernel",
        "receiver_index",
        "weighted_channel_fraction_zc_lt_2p5",
        "interval_min_depth",
        "interval_max_depth",
        "midpoint_depth",
        "total_cell_count",
    }
    missing_cast = sorted(required_cast - set(cast_arrays))
    missing_labels = sorted(required_labels - set(label_arrays))
    if missing_cast:
        raise KeyError("CAST Zc QC missing cast array(s): " + ", ".join(missing_cast))
    if missing_labels:
        raise KeyError("CAST Zc QC missing label array(s): " + ", ".join(missing_labels))
    zc = np.asarray(cast_arrays["cast_zc"])
    if zc.ndim != 2:
        raise ValueError("cast_zc must be a 2D [depth, azimuth] array.")
    target = np.asarray(label_arrays["weighted_channel_fraction_zc_lt_2p5"])
    if target.ndim != 3:
        raise ValueError("weighted_channel_fraction_zc_lt_2p5 must be [kernel, depth, receiver].")


def _global_counts(
    *,
    negative: np.ndarray,
    zero: np.ndarray,
    low_positive: np.ndarray,
    high_or_equal: np.ndarray,
    nonfinite: np.ndarray,
    upper_exceed: np.ndarray,
    total_cells: int,
) -> dict[str, Any]:
    return {
        "total_cells": int(total_cells),
        "zc_lt_0_count": int(np.count_nonzero(negative)),
        "zc_lt_0_fraction": _fraction(negative),
        "zc_eq_0_count": int(np.count_nonzero(zero)),
        "zc_eq_0_fraction": _fraction(zero),
        "zc_gt_0_lt_2p5_count": int(np.count_nonzero(low_positive)),
        "zc_gt_0_lt_2p5_fraction": _fraction(low_positive),
        "zc_gte_2p5_count": int(np.count_nonzero(high_or_equal)),
        "zc_gte_2p5_fraction": _fraction(high_or_equal),
        "nonfinite_count": int(np.count_nonzero(nonfinite)),
        "nonfinite_fraction": _fraction(nonfinite),
        "plausible_upper_bound_exceedance_count": int(np.count_nonzero(upper_exceed)),
        "plausible_upper_bound_exceedance_fraction": _fraction(upper_exceed),
    }


def _depth_summary(
    depth: np.ndarray,
    negative: np.ndarray,
    nonfinite: np.ndarray,
    upper_exceed: np.ndarray,
) -> dict[str, Any]:
    neg_depth = np.any(negative, axis=1)
    nonfinite_depth = np.any(nonfinite, axis=1)
    upper_depth = np.any(upper_exceed, axis=1)
    return {
        "cast_depth_count": int(depth.size),
        "depths_with_zc_lt_0_count": int(np.count_nonzero(neg_depth)),
        "depths_with_zc_lt_0_fraction": _fraction(neg_depth),
        "depths_with_nonfinite_count": int(np.count_nonzero(nonfinite_depth)),
        "depths_with_nonfinite_fraction": _fraction(nonfinite_depth),
        "depths_with_upper_exceedance_count": int(np.count_nonzero(upper_depth)),
        "depths_with_upper_exceedance_fraction": _fraction(upper_depth),
        "first_negative_depth": _first_value(depth[neg_depth]),
        "last_negative_depth": _last_value(depth[neg_depth]),
    }


def _azimuth_summary(
    azimuth: np.ndarray,
    negative: np.ndarray,
    nonfinite: np.ndarray,
    upper_exceed: np.ndarray,
) -> dict[str, Any]:
    neg_az = np.any(negative, axis=0)
    nonfinite_az = np.any(nonfinite, axis=0)
    upper_az = np.any(upper_exceed, axis=0)
    return {
        "azimuth_count": int(azimuth.size),
        "azimuths_with_zc_lt_0_count": int(np.count_nonzero(neg_az)),
        "azimuths_with_zc_lt_0_fraction": _fraction(neg_az),
        "azimuths_with_nonfinite_count": int(np.count_nonzero(nonfinite_az)),
        "azimuths_with_nonfinite_fraction": _fraction(nonfinite_az),
        "azimuths_with_upper_exceedance_count": int(np.count_nonzero(upper_az)),
        "azimuths_with_upper_exceedance_fraction": _fraction(upper_az),
        "negative_azimuth_degrees": [float(value) for value in azimuth[neg_az][:20]],
    }


def _fold_summaries(
    *,
    label_depth: np.ndarray,
    cast_depth: np.ndarray,
    negative_by_cast_depth: np.ndarray,
    nonfinite_by_cast_depth: np.ndarray,
    upper_exceed_by_cast_depth: np.ndarray,
    target: np.ndarray,
    counterfactual: np.ndarray,
    affected: np.ndarray,
    no_overlap: np.ndarray,
    fold_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for fold in sorted(set(int(value) for value in fold_ids)):
        mask = fold_ids == fold
        cast_mask = (cast_depth >= float(np.min(label_depth[mask]))) & (
            cast_depth <= float(np.max(label_depth[mask]))
        )
        fold_delta = np.abs(counterfactual[:, mask, :] - target[:, mask, :])
        rows.append(
            {
                "fold": fold,
                "depth_min": _finite_min(label_depth[mask]),
                "depth_max": _finite_max(label_depth[mask]),
                "label_depth_count": int(np.count_nonzero(mask)),
                "raw_zc_lt_0_cell_count": int(np.sum(negative_by_cast_depth[cast_mask])),
                "raw_nonfinite_cell_count": int(np.sum(nonfinite_by_cast_depth[cast_mask])),
                "raw_upper_exceedance_cell_count": int(
                    np.sum(upper_exceed_by_cast_depth[cast_mask])
                ),
                "original_target_mean": _finite_mean(target[:, mask, :]),
                "counterfactual_target_mean_excluding_zc_lt_0": _finite_mean(
                    counterfactual[:, mask, :]
                ),
                "mean_abs_delta": _finite_mean(fold_delta),
                "max_abs_delta": _finite_max(fold_delta),
                "affected_depth_fraction": _affected_depth_fraction(affected[:, mask, :]),
                "affected_receiver_fraction": _affected_receiver_fraction(affected[:, mask, :]),
                "no_overlap_region_fraction": _fraction(no_overlap[:, mask, :]),
            }
        )
    return rows


def _kernel_summaries(
    *,
    kernels: np.ndarray,
    target: np.ndarray,
    counterfactual: np.ndarray,
    affected: np.ndarray,
    no_overlap: np.ndarray,
    finite_coverage: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, kernel in enumerate(kernels):
        delta = np.abs(counterfactual[index] - target[index])
        rows.append(
            {
                "geometry_kernel": str(kernel),
                "original_target_mean": _finite_mean(target[index]),
                "counterfactual_target_mean_excluding_zc_lt_0": _finite_mean(
                    counterfactual[index]
                ),
                "mean_abs_delta": _finite_mean(delta),
                "max_abs_delta": _finite_max(delta),
                "affected_depth_fraction": _affected_depth_fraction(affected[index]),
                "affected_receiver_fraction": _affected_receiver_fraction(affected[index]),
                "no_overlap_region_fraction": _fraction(no_overlap[index]),
                "finite_coverage_mean": _finite_mean(finite_coverage[index]),
            }
        )
    return rows


def _receiver_summaries(
    *,
    receiver_index: np.ndarray,
    target: np.ndarray,
    counterfactual: np.ndarray,
    affected: np.ndarray,
    no_overlap: np.ndarray,
    finite_coverage: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for axis_index, receiver in enumerate(receiver_index):
        delta = np.abs(counterfactual[:, :, axis_index] - target[:, :, axis_index])
        receiver_affected = affected[:, :, axis_index]
        rows.append(
            {
                "receiver": int(receiver),
                "original_target_mean": _finite_mean(target[:, :, axis_index]),
                "counterfactual_target_mean_excluding_zc_lt_0": _finite_mean(
                    counterfactual[:, :, axis_index]
                ),
                "mean_abs_delta": _finite_mean(delta),
                "max_abs_delta": _finite_max(delta),
                "affected_depth_fraction": float(
                    np.mean(np.any(receiver_affected, axis=0))
                ),
                "affected_kernel_fraction": float(
                    np.mean(np.any(receiver_affected, axis=1))
                ),
                "no_overlap_region_fraction": _fraction(no_overlap[:, :, axis_index]),
                "finite_coverage_mean": _finite_mean(finite_coverage[:, :, axis_index]),
            }
        )
    return rows


def _depth_rows(
    depth: np.ndarray,
    zc: np.ndarray,
    finite: np.ndarray,
    negative: np.ndarray,
    zero: np.ndarray,
    low_positive: np.ndarray,
    high_or_equal: np.ndarray,
    nonfinite: np.ndarray,
    upper_exceed: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    azimuth_count = zc.shape[1]
    for index, value in enumerate(depth):
        rows.append(
            {
                "section": "raw_depth",
                "cast_depth": float(value),
                "finite_coverage": float(np.count_nonzero(finite[index]) / azimuth_count),
                "zc_lt_0_count": int(np.count_nonzero(negative[index])),
                "zc_eq_0_count": int(np.count_nonzero(zero[index])),
                "zc_gt_0_lt_2p5_count": int(np.count_nonzero(low_positive[index])),
                "zc_gte_2p5_count": int(np.count_nonzero(high_or_equal[index])),
                "nonfinite_count": int(np.count_nonzero(nonfinite[index])),
                "upper_exceedance_count": int(np.count_nonzero(upper_exceed[index])),
            }
        )
    return rows


def _azimuth_rows(
    azimuth: np.ndarray,
    zc: np.ndarray,
    finite: np.ndarray,
    negative: np.ndarray,
    zero: np.ndarray,
    low_positive: np.ndarray,
    high_or_equal: np.ndarray,
    nonfinite: np.ndarray,
    upper_exceed: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    depth_count = zc.shape[0]
    for index, value in enumerate(azimuth):
        rows.append(
            {
                "section": "raw_azimuth",
                "cast_azimuth_deg": float(value),
                "finite_coverage": float(np.count_nonzero(finite[:, index]) / depth_count),
                "zc_lt_0_count": int(np.count_nonzero(negative[:, index])),
                "zc_eq_0_count": int(np.count_nonzero(zero[:, index])),
                "zc_gt_0_lt_2p5_count": int(np.count_nonzero(low_positive[:, index])),
                "zc_gte_2p5_count": int(np.count_nonzero(high_or_equal[:, index])),
                "nonfinite_count": int(np.count_nonzero(nonfinite[:, index])),
                "upper_exceedance_count": int(np.count_nonzero(upper_exceed[:, index])),
            }
        )
    return rows


def _possible_cell_count(
    interval_min: np.ndarray,
    interval_max: np.ndarray,
    cast_depth: np.ndarray,
    azimuth_count: int,
) -> np.ndarray:
    output = np.zeros(interval_min.shape, dtype=np.int64)
    for index in np.ndindex(interval_min.shape):
        start, stop = _depth_region_bounds(
            cast_depth,
            float(interval_min[index]),
            float(interval_max[index]),
        )
        output[index] = max(stop - start, 0) * azimuth_count
    return output


def _fold_ids(depth: np.ndarray, fold_count: int) -> np.ndarray:
    order = np.argsort(depth)
    folds = np.empty(depth.size, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(order, fold_count)):
        folds[indices] = fold
    return folds


def _depth_region_bounds(depth: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    if depth.size == 0:
        return 0, 0
    if np.isclose(lower, upper):
        insert = int(np.searchsorted(depth, lower, side="left"))
        if insert <= 0:
            nearest = 0
        elif insert >= depth.size:
            nearest = depth.size - 1
        else:
            before = insert - 1
            after = insert
            nearest = (
                before
                if abs(float(depth[before]) - lower)
                <= abs(float(depth[after]) - lower)
                else after
            )
        return nearest, nearest + 1
    return (
        int(np.searchsorted(depth, min(lower, upper), side="left")),
        int(np.searchsorted(depth, max(lower, upper), side="right")),
    )


def _depth_weights(
    depth: np.ndarray,
    *,
    lower: float,
    upper: float,
    midpoint: float,
) -> np.ndarray:
    half_width = max(abs(upper - midpoint), abs(midpoint - lower), 1e-6)
    weights = 1.0 - np.abs(depth.astype(np.float32) - midpoint) / half_width
    return np.clip(weights, 0.0, 1.0).astype(np.float32)


def _count_cumsum(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)])


def _range_sum(cumsum: np.ndarray, start: int, stop: int) -> int:
    return int(cumsum[stop] - cumsum[start])


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


def _first_value(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(values[0])


def _last_value(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(values[-1])


def _affected_depth_fraction(affected: np.ndarray) -> float | None:
    array = np.asarray(affected, dtype=bool)
    if array.size == 0:
        return None
    if array.ndim == 3:
        return float(np.mean(np.any(array, axis=(0, 2))))
    if array.ndim == 2:
        return float(np.mean(np.any(array, axis=-1)))
    return float(np.mean(array))


def _affected_receiver_fraction(affected: np.ndarray) -> float | None:
    array = np.asarray(affected, dtype=bool)
    if array.ndim == 3:
        return float(np.mean(np.any(array, axis=(0, 1))))
    if array.ndim == 2:
        return float(np.mean(np.any(array, axis=0)))
    return _fraction(array)


def _affected_kernel_fraction(affected: np.ndarray) -> float | None:
    array = np.asarray(affected, dtype=bool)
    if array.ndim == 2:
        return float(np.mean(np.any(array, axis=1)))
    return _fraction(array)


def _affected_fold_fraction(affected: np.ndarray, fold_ids: np.ndarray) -> float | None:
    flags = []
    for fold in sorted(set(int(value) for value in fold_ids)):
        flags.append(bool(np.any(affected[:, fold_ids == fold, :])))
    return None if not flags else float(np.mean(flags))


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
