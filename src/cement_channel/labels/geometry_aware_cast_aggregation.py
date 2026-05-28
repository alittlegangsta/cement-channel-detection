from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.alignment.xsi_geometry import AlignmentMode, DepthAxisSign, ReceiverGeometry

GEOMETRY_AWARE_DEPTH_LABEL_VERSION = "geometry_aware_depth_labels_v001"
GEOMETRY_AWARE_DEPTH_LABEL_REPORT_VERSION = "geometry_aware_depth_labels_report_v001"


@dataclass(frozen=True)
class GeometryAwareAggregationConfig:
    interval_padding_ft: float = 0.0
    nearest_tolerance_ft: float | None = None
    strong_candidate_fraction: float = 0.25
    strong_min_severity: int = 2
    min_label_confidence: float = 0.5
    max_disagreement_fraction: float = 0.25
    min_orientation_confidence: float = 0.5


@dataclass(frozen=True)
class GeometryAwareDepthLabelReport:
    report_version: str
    label_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    geometry_version: str
    depth_count: int
    receiver_count: int
    mode_sign_summaries: list[dict[str, Any]]
    baseline_difference_summary: dict[str, Any]
    raw_zc_available: bool
    zc_source_field: str | None
    sign_convention_warning: str
    no_model_training: bool
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


def build_geometry_aware_depth_labels_from_paths(
    *,
    cast_weak_label_npz: Path | str,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    geometry_config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
    aggregation_config: GeometryAwareAggregationConfig | None = None,
) -> GeometryAwareDepthLabelReport:
    cast_arrays = _load_npz(cast_weak_label_npz)
    label_arrays = _load_npz(depth_level_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    geometry = ReceiverGeometry.from_yaml(geometry_config_path)
    arrays, report = build_geometry_aware_depth_labels(
        cast_arrays=cast_arrays,
        depth_label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        geometry=geometry,
        aggregation_config=aggregation_config or GeometryAwareAggregationConfig(),
        inputs={
            "cast_weak_label_npz": str(cast_weak_label_npz),
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "geometry_config_path": str(geometry_config_path),
        },
        output_npz=Path(output_npz),
    )
    write_geometry_aware_depth_label_outputs(
        arrays,
        report,
        output_npz=Path(output_npz),
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return report


def build_geometry_aware_depth_labels(
    *,
    cast_arrays: dict[str, np.ndarray],
    depth_label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    geometry: ReceiverGeometry,
    aggregation_config: GeometryAwareAggregationConfig | None = None,
    inputs: dict[str, str] | None = None,
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], GeometryAwareDepthLabelReport]:
    config = aggregation_config or GeometryAwareAggregationConfig()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(cast_arrays, depth_label_arrays, feature_arrays, errors)

    reference_depth = _reference_depth(depth_label_arrays, feature_arrays, warnings)
    cast_depth = np.asarray(cast_arrays["cast_depth"], dtype=np.float32).reshape(-1)
    sorted_context = _build_sorted_context(cast_arrays, cast_depth, warnings)
    if sorted_context.zc_source_field is None:
        warnings.append(
            "CAST weak-label candidate NPZ does not contain raw Zc; zc_min, zc_p05, "
            "and zc_p10 are NaN review fields. zc_ratio is not used as raw Zc."
        )
    if geometry.depth_axis_sign == "audit_both":
        warnings.append(
            "Depth-axis sign is unconfirmed; both sign=+1 and sign=-1 are audit outputs."
        )

    combo_arrays: list[dict[str, np.ndarray]] = []
    mode_names: list[str] = []
    signs: list[int] = []
    for mode in geometry.alignment_modes:
        for sign in geometry.audit_signs:
            receiver_metrics = aggregate_receiver_cast_evidence(
                reference_depth=reference_depth,
                mode=mode,
                sign=sign,
                geometry=geometry,
                context=sorted_context,
                config=config,
            )
            summary = summarize_receiver_metrics(receiver_metrics, config=config)
            table = geometry.geometry_table(
                reference_depth,
                sign=sign,
                interval_padding_ft=config.interval_padding_ft,
            )
            combo_arrays.append({**table, **receiver_metrics, **summary})
            mode_names.append(mode)
            signs.append(sign)

    arrays = _stack_combo_arrays(combo_arrays, mode_names, signs, geometry)
    arrays.update(
        {
            "depth": reference_depth.astype(np.float32),
            "geometry_aware_depth_label_version": np.asarray(
                GEOMETRY_AWARE_DEPTH_LABEL_VERSION
            ),
            "geometry_aware_depth_label_metadata_json": np.asarray(
                json.dumps(
                    {
                        "label_status": "weak_label_candidate",
                        "source": "CAST weak-label candidate aggregation",
                        "geometry_interpretation": "source_to_receiver_interval_audit",
                        "no_final_labels": True,
                        "raw_zc_available": sorted_context.zc_source_field is not None,
                        "zc_source_field": sorted_context.zc_source_field,
                    },
                    sort_keys=True,
                )
            ),
            "no_model_training": np.asarray(True),
            "no_final_labels": np.asarray(True),
            "no_stc": np.asarray(True),
            "no_apes": np.asarray(True),
            "no_deep_learning": np.asarray(True),
            "no_mvp4c": np.asarray(True),
        }
    )
    errors.extend(_sample_count_errors(arrays))
    report = _build_report(
        arrays,
        depth_label_arrays=depth_label_arrays,
        geometry=geometry,
        inputs=inputs or {},
        output_npz=output_npz,
        zc_source_field=sorted_context.zc_source_field,
        warnings=warnings,
        errors=errors,
    )
    return arrays, report


def aggregate_receiver_cast_evidence(
    *,
    reference_depth: np.ndarray,
    mode: AlignmentMode,
    sign: DepthAxisSign,
    geometry: ReceiverGeometry,
    context: SortedCastContext,
    config: GeometryAwareAggregationConfig,
) -> dict[str, np.ndarray]:
    if mode == "source_receiver_interval":
        lower, upper = geometry.source_receiver_interval_bounds(
            reference_depth,
            sign=sign,
            padding_ft=config.interval_padding_ft,
        )
        return _interval_receiver_metrics(context, lower, upper)

    target_depths = geometry.target_depths_for_mode(reference_depth, mode=mode, sign=sign)
    positions, distances = _nearest_positions(context.sorted_depth, target_depths)
    if config.nearest_tolerance_ft is not None:
        outside = distances > config.nearest_tolerance_ft
        positions = np.where(outside, -1, positions)
    return _point_receiver_metrics(context, positions)


def summarize_receiver_metrics(
    receiver_metrics: dict[str, np.ndarray],
    *,
    config: GeometryAwareAggregationConfig,
) -> dict[str, np.ndarray]:
    candidate_count = receiver_metrics["receiver_candidate_count"].sum(axis=1)
    valid_count = receiver_metrics["receiver_valid_count"].sum(axis=1)
    possible_count = receiver_metrics["receiver_possible_count"].sum(axis=1)
    disagreement_count = receiver_metrics["receiver_disagreement_count"].sum(axis=1)
    disagreement_denominator = receiver_metrics["receiver_disagreement_denominator"].sum(axis=1)
    orientation_sum = receiver_metrics["receiver_orientation_sum"].sum(axis=1)
    orientation_count = receiver_metrics["receiver_orientation_count"].sum(axis=1)

    candidate_fraction = _safe_divide(candidate_count, valid_count)
    valid_fraction = _safe_divide(valid_count, possible_count)
    disagreement_fraction = _safe_divide(disagreement_count, disagreement_denominator)
    orientation = _safe_divide(orientation_sum, orientation_count, default=0.0)
    receiver_disagreement = _safe_divide(
        receiver_metrics["receiver_disagreement_count"],
        receiver_metrics["receiver_disagreement_denominator"],
    )
    receiver_orientation = _safe_divide(
        receiver_metrics["receiver_orientation_sum"],
        receiver_metrics["receiver_orientation_count"],
    )
    receiver_valid_fraction = _safe_divide(
        receiver_metrics["receiver_valid_count"],
        receiver_metrics["receiver_possible_count"],
    )
    receiver_label_confidence = _depth_label_confidence(
        has_channel=receiver_metrics["receiver_candidate_count"] > 0,
        candidate_max_confidence=receiver_metrics["receiver_candidate_max_confidence"],
        valid_fraction=receiver_valid_fraction,
        orientation_confidence=receiver_orientation,
        disagreement_fraction=receiver_disagreement,
    )
    max_confidence = _nanmax_axis(receiver_metrics["receiver_max_confidence"], axis=1, default=0.0)
    candidate_max_confidence = _nanmax_axis(
        receiver_metrics["receiver_candidate_max_confidence"], axis=1, default=0.0
    )
    has_channel = candidate_count > 0
    label_confidence = _depth_label_confidence(
        has_channel=has_channel,
        candidate_max_confidence=candidate_max_confidence,
        valid_fraction=valid_fraction,
        orientation_confidence=orientation,
        disagreement_fraction=disagreement_fraction,
    )
    max_severity = np.max(receiver_metrics["receiver_max_severity"], axis=1)
    max_severity = np.where(has_channel, max_severity, 0).astype(np.int8)
    high_confidence = (
        has_channel
        & (label_confidence >= config.min_label_confidence)
        & (orientation >= config.min_orientation_confidence)
        & (disagreement_fraction <= config.max_disagreement_fraction)
    )
    strong_positive = (
        has_channel
        & (candidate_fraction >= config.strong_candidate_fraction)
        & (max_severity >= config.strong_min_severity)
        & (label_confidence >= config.min_label_confidence)
        & (orientation >= config.min_orientation_confidence)
        & (disagreement_fraction <= config.max_disagreement_fraction)
    )
    clear_negative = (
        (candidate_fraction <= 0.0)
        & (label_confidence >= config.min_label_confidence)
        & (orientation >= config.min_orientation_confidence)
        & (disagreement_fraction <= config.max_disagreement_fraction)
    )
    return {
        "candidate_fraction": candidate_fraction.astype(np.float32),
        "has_channel_any": has_channel.astype(bool),
        "max_severity": max_severity.astype(np.int8),
        "max_confidence": max_confidence.astype(np.float32),
        "max_relative_drop": _nanmax_axis(
            receiver_metrics["receiver_max_relative_drop"], axis=1, default=np.nan
        ).astype(np.float32),
        "plus_minus_disagreement_fraction": disagreement_fraction.astype(np.float32),
        "depth_label_confidence": label_confidence.astype(np.float32),
        "orientation_confidence": orientation.astype(np.float32),
        "zc_min": _nanmin_axis(receiver_metrics["receiver_zc_min"], axis=1).astype(np.float32),
        "zc_p05": _nanmin_axis(receiver_metrics["receiver_zc_p05"], axis=1).astype(np.float32),
        "zc_p10": _nanmin_axis(receiver_metrics["receiver_zc_p10"], axis=1).astype(np.float32),
        "cast_sample_count": receiver_metrics["receiver_cast_sample_count"].sum(axis=1).astype(
            np.int32
        ),
        "valid_fraction": valid_fraction.astype(np.float32),
        "high_confidence_positive_mask": high_confidence.astype(bool),
        "strong_positive_mask": strong_positive.astype(bool),
        "clear_negative_mask": clear_negative.astype(bool),
        "receiver_positive_fraction": np.mean(
            receiver_metrics["receiver_candidate_count"] > 0,
            axis=1,
        ).astype(np.float32),
        "receiver_candidate_fraction_max": np.max(
            receiver_metrics["receiver_candidate_fraction"],
            axis=1,
        ).astype(np.float32),
        "receiver_candidate_fraction_p90": np.percentile(
            receiver_metrics["receiver_candidate_fraction"],
            90.0,
            axis=1,
        ).astype(np.float32),
        "receiver_label_confidence": receiver_label_confidence.astype(np.float32),
    }


@dataclass(frozen=True)
class SortedCastContext:
    sorted_depth: np.ndarray
    azimuth_count: int
    row_metrics: dict[str, np.ndarray]
    prefixes: dict[str, np.ndarray]
    zc_source_field: str | None


def write_geometry_aware_depth_label_outputs(
    arrays: dict[str, np.ndarray],
    report: GeometryAwareDepthLabelReport,
    *,
    output_npz: Path,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_geometry_aware_depth_label_markdown(report), encoding="utf-8")


def format_geometry_aware_depth_label_markdown(report: GeometryAwareDepthLabelReport) -> str:
    lines = [
        "# Geometry-Aware CAST Depth Weak-Label Candidate Report",
        "",
        "This is an MVP-4B-G review artifact. It aggregates CAST weak-label "
        "candidates under explicit XSI source-to-receiver geometry. It is not "
        "ground truth and does not create final labels.",
        "",
        f"- label_version: `{report.label_version}`",
        f"- depth_count: {report.depth_count}",
        f"- receiver_count: {report.receiver_count}",
        f"- raw_zc_available: `{report.raw_zc_available}`",
        f"- zc_source_field: `{report.zc_source_field}`",
        f"- sign_convention_warning: {report.sign_convention_warning}",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Mode And Sign Summaries",
        "",
    ]
    for summary in report.mode_sign_summaries:
        lines.append(
            "- "
            f"{summary['mode']} sign={summary['sign']}: "
            f"sample_count={summary['sample_count']}, "
            f"positive_fraction={summary['positive_fraction']}, "
            f"high_confidence_positive_count={summary['high_confidence_positive_count']}, "
            f"candidate_fraction_median={summary['candidate_fraction_distribution']['median']}, "
            f"max_severity={summary['max_severity_distribution']}"
        )
    lines.extend(["", "## Difference From Existing R7 Baseline", ""])
    lines.extend(_dict_lines(report.baseline_difference_summary))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _build_sorted_context(
    cast_arrays: dict[str, np.ndarray],
    cast_depth: np.ndarray,
    warnings: list[str],
) -> SortedCastContext:
    order = np.argsort(cast_depth)
    presence = np.asarray(cast_arrays["presence_plus"], dtype=np.int8)[order]
    severity = np.asarray(cast_arrays["severity_plus"], dtype=np.int8)[order]
    confidence = np.asarray(cast_arrays["label_confidence_plus"], dtype=np.float32)[order]
    minus = np.asarray(cast_arrays["presence_minus_ablation"], dtype=np.int8)[order]
    relative_drop = np.asarray(
        cast_arrays.get("relative_drop_plus", np.full(confidence.shape, np.nan)),
        dtype=np.float32,
    )[order]
    orientation_key = "orientation_confidence_on_cast_depth_plus"
    if orientation_key in cast_arrays:
        orientation = np.asarray(cast_arrays[orientation_key], dtype=np.float32)[order]
    else:
        warnings.append(
            "orientation_confidence_on_cast_depth_plus is unavailable; using 1.0 fallback."
        )
        orientation = np.ones(confidence.shape, dtype=np.float32)
    zc_values, zc_source = _optional_zc_values(cast_arrays, order)
    row_metrics = _row_metrics(
        presence=presence,
        severity=severity,
        confidence=confidence,
        minus=minus,
        relative_drop=relative_drop,
        orientation=orientation,
        zc_values=zc_values,
    )
    prefixes = {
        key: _prefix_sum(row_metrics[key])
        for key in (
            "valid_count",
            "candidate_count",
            "possible_count",
            "disagreement_count",
            "disagreement_denominator",
            "orientation_sum",
            "orientation_count",
        )
    }
    return SortedCastContext(
        sorted_depth=cast_depth[order].astype(np.float32),
        azimuth_count=int(presence.shape[1]),
        row_metrics=row_metrics,
        prefixes=prefixes,
        zc_source_field=zc_source,
    )


def _row_metrics(
    *,
    presence: np.ndarray,
    severity: np.ndarray,
    confidence: np.ndarray,
    minus: np.ndarray,
    relative_drop: np.ndarray,
    orientation: np.ndarray,
    zc_values: np.ndarray | None,
) -> dict[str, np.ndarray]:
    valid = presence >= 0
    candidate = presence == 1
    disagreement_known = valid & (minus >= 0)
    disagreement = disagreement_known & (presence != minus)
    valid_count = np.count_nonzero(valid, axis=1).astype(np.float32)
    candidate_count = np.count_nonzero(candidate, axis=1).astype(np.float32)
    candidate_max_confidence = _row_nanmax(
        np.where(candidate, confidence, np.nan),
        default=0.0,
    )
    row_orientation = np.where(valid & np.isfinite(orientation), orientation, np.nan)
    row_metrics = {
        "valid_count": valid_count,
        "candidate_count": candidate_count,
        "possible_count": np.full(presence.shape[0], presence.shape[1], dtype=np.float32),
        "disagreement_count": np.count_nonzero(disagreement, axis=1).astype(np.float32),
        "disagreement_denominator": np.count_nonzero(disagreement_known, axis=1).astype(
            np.float32
        ),
        "orientation_sum": np.nansum(row_orientation, axis=1).astype(np.float32),
        "orientation_count": np.count_nonzero(np.isfinite(row_orientation), axis=1).astype(
            np.float32
        ),
        "candidate_fraction": _safe_divide(candidate_count, valid_count),
        "max_severity": np.where(
            candidate_count > 0,
            np.max(np.where(candidate, severity, -1), axis=1),
            0,
        ).astype(np.int8),
        "max_confidence": _row_nanmax(np.where(valid, confidence, np.nan), default=0.0),
        "candidate_max_confidence": candidate_max_confidence,
        "max_relative_drop": _row_nanmax(
            np.where(valid, relative_drop, np.nan),
            default=np.nan,
        ),
    }
    if zc_values is None:
        nan_row = np.full(presence.shape[0], np.nan, dtype=np.float32)
        row_metrics.update({"zc_min": nan_row, "zc_p05": nan_row, "zc_p10": nan_row})
    else:
        zc = np.where(valid & np.isfinite(zc_values), zc_values, np.nan)
        row_metrics.update(
            {
                "zc_min": _row_nanpercentile(zc, 0.0),
                "zc_p05": _row_nanpercentile(zc, 5.0),
                "zc_p10": _row_nanpercentile(zc, 10.0),
            }
        )
    return row_metrics


def _point_receiver_metrics(
    context: SortedCastContext,
    positions: np.ndarray,
) -> dict[str, np.ndarray]:
    invalid = positions < 0
    safe_positions = np.where(invalid, 0, positions)
    metrics = {
        "receiver_valid_count": context.row_metrics["valid_count"][safe_positions],
        "receiver_candidate_count": context.row_metrics["candidate_count"][safe_positions],
        "receiver_possible_count": context.row_metrics["possible_count"][safe_positions],
        "receiver_disagreement_count": context.row_metrics["disagreement_count"][safe_positions],
        "receiver_disagreement_denominator": context.row_metrics[
            "disagreement_denominator"
        ][safe_positions],
        "receiver_orientation_sum": context.row_metrics["orientation_sum"][safe_positions],
        "receiver_orientation_count": context.row_metrics["orientation_count"][safe_positions],
        "receiver_candidate_fraction": context.row_metrics["candidate_fraction"][safe_positions],
        "receiver_max_severity": context.row_metrics["max_severity"][safe_positions],
        "receiver_max_confidence": context.row_metrics["max_confidence"][safe_positions],
        "receiver_candidate_max_confidence": context.row_metrics[
            "candidate_max_confidence"
        ][safe_positions],
        "receiver_max_relative_drop": context.row_metrics["max_relative_drop"][safe_positions],
        "receiver_zc_min": context.row_metrics["zc_min"][safe_positions],
        "receiver_zc_p05": context.row_metrics["zc_p05"][safe_positions],
        "receiver_zc_p10": context.row_metrics["zc_p10"][safe_positions],
        "receiver_cast_sample_count": np.where(invalid, 0, 1).astype(np.int32),
    }
    for key, value in metrics.items():
        if key in {"receiver_max_severity", "receiver_cast_sample_count"}:
            metrics[key] = np.where(invalid, 0, value)
        else:
            metrics[key] = np.where(invalid, np.nan, value)
    return metrics


def _interval_receiver_metrics(
    context: SortedCastContext,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, np.ndarray]:
    left = np.searchsorted(context.sorted_depth, lower, side="left")
    right = np.searchsorted(context.sorted_depth, upper, side="right")
    row_count = np.maximum(right - left, 0).astype(np.int32)
    metrics = {
        "receiver_valid_count": _range_sum(context.prefixes["valid_count"], left, right),
        "receiver_candidate_count": _range_sum(
            context.prefixes["candidate_count"], left, right
        ),
        "receiver_possible_count": _range_sum(context.prefixes["possible_count"], left, right),
        "receiver_disagreement_count": _range_sum(
            context.prefixes["disagreement_count"], left, right
        ),
        "receiver_disagreement_denominator": _range_sum(
            context.prefixes["disagreement_denominator"], left, right
        ),
        "receiver_orientation_sum": _range_sum(context.prefixes["orientation_sum"], left, right),
        "receiver_orientation_count": _range_sum(
            context.prefixes["orientation_count"], left, right
        ),
        "receiver_max_severity": _range_max(
            context.row_metrics["max_severity"], left, right, default=0
        ).astype(np.int8),
        "receiver_max_confidence": _range_nanmax(
            context.row_metrics["max_confidence"], left, right, default=0.0
        ),
        "receiver_candidate_max_confidence": _range_nanmax(
            context.row_metrics["candidate_max_confidence"], left, right, default=0.0
        ),
        "receiver_max_relative_drop": _range_nanmax(
            context.row_metrics["max_relative_drop"], left, right, default=np.nan
        ),
        "receiver_zc_min": _range_nanmin(
            context.row_metrics["zc_min"], left, right, default=np.nan
        ),
        "receiver_zc_p05": _range_nanmin(
            context.row_metrics["zc_p05"], left, right, default=np.nan
        ),
        "receiver_zc_p10": _range_nanmin(
            context.row_metrics["zc_p10"], left, right, default=np.nan
        ),
        "receiver_cast_sample_count": row_count,
    }
    metrics["receiver_candidate_fraction"] = _safe_divide(
        metrics["receiver_candidate_count"],
        metrics["receiver_valid_count"],
    )
    return metrics


def _stack_combo_arrays(
    combo_arrays: list[dict[str, np.ndarray]],
    mode_names: list[str],
    signs: list[int],
    geometry: ReceiverGeometry,
) -> dict[str, np.ndarray]:
    stack_keys = tuple(combo_arrays[0].keys())
    output = {key: np.stack([arrays[key] for arrays in combo_arrays], axis=0) for key in stack_keys}
    output["mode"] = np.asarray(mode_names)
    output["sign"] = np.asarray(signs, dtype=np.int8)
    output["receiver_label"] = np.asarray(geometry.receiver_labels)
    output["receiver_offset_relative_to_r7_ft"] = geometry.receiver_offsets_ft.astype(np.float32)
    output["source_offset_relative_to_r7_ft"] = np.asarray(
        geometry.source_offset_ft,
        dtype=np.float32,
    )
    output["interval_min_depth"] = np.min(output["interval_min_depths"], axis=2).astype(
        np.float32
    )
    output["interval_max_depth"] = np.max(output["interval_max_depths"], axis=2).astype(
        np.float32
    )
    return output


def _build_report(
    arrays: dict[str, np.ndarray],
    *,
    depth_label_arrays: dict[str, np.ndarray],
    geometry: ReceiverGeometry,
    inputs: dict[str, str],
    output_npz: Path | None,
    zc_source_field: str | None,
    warnings: list[str],
    errors: list[str],
) -> GeometryAwareDepthLabelReport:
    summaries = []
    for index, mode in enumerate(arrays["mode"].astype(str)):
        candidate_fraction = arrays["candidate_fraction"][index]
        has_channel = arrays["has_channel_any"][index]
        summaries.append(
            {
                "mode": mode,
                "sign": int(arrays["sign"][index]),
                "sample_count": int(candidate_fraction.size),
                "valid_sample_count": int(np.count_nonzero(arrays["valid_fraction"][index] > 0.0)),
                "positive_count": int(np.count_nonzero(has_channel)),
                "positive_fraction": _fraction(has_channel),
                "high_confidence_positive_count": int(
                    np.count_nonzero(arrays["high_confidence_positive_mask"][index])
                ),
                "strong_positive_count": int(
                    np.count_nonzero(arrays["strong_positive_mask"][index])
                ),
                "clear_negative_count": int(np.count_nonzero(arrays["clear_negative_mask"][index])),
                "candidate_fraction_distribution": _numeric_distribution(candidate_fraction),
                "max_severity_distribution": _severity_distribution(arrays["max_severity"][index]),
                "disagreement_fraction_distribution": _numeric_distribution(
                    arrays["plus_minus_disagreement_fraction"][index]
                ),
                "depth_label_confidence_distribution": _numeric_distribution(
                    arrays["depth_label_confidence"][index]
                ),
            }
        )
    return GeometryAwareDepthLabelReport(
        report_version=GEOMETRY_AWARE_DEPTH_LABEL_REPORT_VERSION,
        label_version=GEOMETRY_AWARE_DEPTH_LABEL_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        output_npz=str(output_npz) if output_npz else "",
        geometry_version="xsi_geometry_v001",
        depth_count=int(arrays["depth"].size),
        receiver_count=geometry.receiver_count,
        mode_sign_summaries=summaries,
        baseline_difference_summary=_baseline_difference_summary(arrays, depth_label_arrays),
        raw_zc_available=zc_source_field is not None,
        zc_source_field=zc_source_field,
        sign_convention_warning=(
            "depth_axis_sign is unconfirmed; compare sign=+1 and sign=-1 before use."
        ),
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "final label generation",
            "ground truth claim",
            "raw waveform reading",
            "model training",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )


def _baseline_difference_summary(
    arrays: dict[str, np.ndarray],
    depth_label_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    if "depth_candidate_fraction" not in depth_label_arrays:
        return {"available": False, "reason": "depth_candidate_fraction unavailable"}
    baseline = np.asarray(depth_label_arrays["depth_candidate_fraction"], dtype=np.float32)
    rows: dict[str, Any] = {"available": True}
    for index, mode in enumerate(arrays["mode"].astype(str)):
        if mode != "r7_reference_depth":
            continue
        key = f"{mode}_sign_{int(arrays['sign'][index]):+d}"
        current = np.asarray(arrays["candidate_fraction"][index], dtype=np.float32)
        delta = current - baseline
        rows[key] = {
            "mean_abs_candidate_fraction_delta": _nanmean(np.abs(delta)),
            "max_abs_candidate_fraction_delta": _nanmax(delta=np.abs(delta)),
            "positive_fraction_delta": _fraction(arrays["has_channel_any"][index])
            - _fraction(np.asarray(depth_label_arrays["depth_has_channel_any"], dtype=bool)),
        }
    return rows


def _reference_depth(
    depth_label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    warnings: list[str],
) -> np.ndarray:
    label_depth = np.asarray(depth_label_arrays["depth"], dtype=np.float32).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    if feature_depth.size != label_depth.size:
        raise ValueError("Depth-level label and feature depth counts differ.")
    if not np.allclose(label_depth, feature_depth, atol=1e-3):
        warnings.append("Depth-level label and feature depth arrays differ; using label depth.")
    return label_depth


def _validate_guardrails(
    cast_arrays: dict[str, np.ndarray],
    depth_label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    errors: list[str],
) -> None:
    required_cast = (
        "cast_depth",
        "presence_plus",
        "severity_plus",
        "label_confidence_plus",
        "presence_minus_ablation",
    )
    required_labels = ("depth", "depth_has_channel_any", "no_final_labels")
    required_features = ("depth", "depth_level_xsi_features", "no_final_labels")
    missing_cast = [key for key in required_cast if key not in cast_arrays]
    missing_labels = [key for key in required_labels if key not in depth_label_arrays]
    missing_features = [key for key in required_features if key not in feature_arrays]
    if missing_cast:
        raise KeyError("CAST weak-label NPZ missing field(s): " + ", ".join(missing_cast))
    if missing_labels:
        raise KeyError("Depth-level label NPZ missing field(s): " + ", ".join(missing_labels))
    if missing_features:
        raise KeyError("Depth-level feature NPZ missing field(s): " + ", ".join(missing_features))
    for name, arrays in (
        ("CAST weak-label input", cast_arrays),
        ("depth-level label input", depth_label_arrays),
        ("depth-level feature input", feature_arrays),
    ):
        if "no_final_labels" in arrays and not bool(np.asarray(arrays["no_final_labels"])):
            errors.append(f"{name} must preserve no_final_labels=true.")
    for forbidden in ("no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        for name, arrays in (
            ("depth-level label input", depth_label_arrays),
            ("depth-level feature input", feature_arrays),
        ):
            if forbidden in arrays and not bool(np.asarray(arrays[forbidden])):
                errors.append(f"{name} must preserve {forbidden}=true.")


def _sample_count_errors(arrays: dict[str, np.ndarray]) -> list[str]:
    errors: list[str] = []
    valid = np.asarray(arrays["valid_fraction"], dtype=np.float32)
    for index, mode in enumerate(arrays["mode"].astype(str)):
        sign = int(arrays["sign"][index])
        if np.count_nonzero(valid[index] > 0.0) == 0:
            errors.append(f"Sample counts collapsed for {mode} sign={sign}.")
    return errors


def _nearest_positions(
    sorted_depth: np.ndarray,
    target_depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(target_depths, dtype=np.float32).reshape(-1)
    insert = np.searchsorted(sorted_depth, flat)
    insert = np.clip(insert, 1, sorted_depth.size - 1)
    left = insert - 1
    right = insert
    choose_right = np.abs(sorted_depth[right] - flat) < np.abs(sorted_depth[left] - flat)
    positions = np.where(choose_right, right, left)
    distances = np.abs(sorted_depth[positions] - flat)
    return positions.reshape(target_depths.shape), distances.reshape(target_depths.shape)


def _optional_zc_values(
    cast_arrays: dict[str, np.ndarray],
    order: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    for key in ("zc_plus", "cast_zc_plus", "cast_zc", "zc", "Zc"):
        if key in cast_arrays:
            return np.asarray(cast_arrays[key], dtype=np.float32)[order], key
    return None, None


def _row_nanmax(values: np.ndarray, *, default: float) -> np.ndarray:
    finite = np.isfinite(values)
    replaced = np.where(finite, values, -np.inf)
    result = np.max(replaced, axis=1)
    return np.where(np.isfinite(result), result, default).astype(np.float32)


def _row_nanpercentile(values: np.ndarray, percentile: float) -> np.ndarray:
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    for index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            output[index] = np.percentile(finite, percentile)
    return output


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.asarray([0.0], dtype=np.float64), np.cumsum(values, dtype=np.float64)]
    )


def _range_sum(prefix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (prefix[right] - prefix[left]).astype(np.float32)


def _range_max(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    default: int,
) -> np.ndarray:
    output = np.full(left.shape, default, dtype=np.int16)
    for index in np.ndindex(left.shape):
        if right[index] > left[index]:
            output[index] = int(np.max(values[left[index] : right[index]]))
    return output


def _range_nanmax(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    default: float,
) -> np.ndarray:
    output = np.full(left.shape, default, dtype=np.float32)
    for index in np.ndindex(left.shape):
        if right[index] <= left[index]:
            continue
        finite = values[left[index] : right[index]]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            output[index] = float(np.max(finite))
    return output


def _range_nanmin(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    default: float,
) -> np.ndarray:
    output = np.full(left.shape, default, dtype=np.float32)
    for index in np.ndindex(left.shape):
        if right[index] <= left[index]:
            continue
        finite = values[left[index] : right[index]]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            output[index] = float(np.min(finite))
    return output


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    default: float = 0.0,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.shape(numerator), default, dtype=np.float32),
        where=np.asarray(denominator) > 0,
    ).astype(np.float32)


def _depth_label_confidence(
    *,
    has_channel: np.ndarray,
    candidate_max_confidence: np.ndarray,
    valid_fraction: np.ndarray,
    orientation_confidence: np.ndarray,
    disagreement_fraction: np.ndarray,
) -> np.ndarray:
    evidence_confidence = np.where(has_channel, candidate_max_confidence, valid_fraction)
    confidence = evidence_confidence * orientation_confidence * np.maximum(
        0.0,
        1.0 - disagreement_fraction,
    )
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _nanmax_axis(values: np.ndarray, *, axis: int, default: float) -> np.ndarray:
    finite = np.isfinite(values)
    replaced = np.where(finite, values, -np.inf)
    result = np.max(replaced, axis=axis)
    return np.where(np.isfinite(result), result, default)


def _nanmin_axis(values: np.ndarray, *, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    replaced = np.where(finite, values, np.inf)
    result = np.min(replaced, axis=axis)
    return np.where(np.isfinite(result), result, np.nan)


def _numeric_distribution(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "min": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    percentiles = np.percentile(finite, [0, 5, 10, 25, 50, 75, 90, 95, 100])
    return {
        "min": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "median": float(percentiles[4]),
        "p75": float(percentiles[5]),
        "p90": float(percentiles[6]),
        "p95": float(percentiles[7]),
        "max": float(percentiles[8]),
        "mean": float(np.mean(finite)),
    }


def _severity_distribution(values: np.ndarray) -> dict[str, int]:
    array = np.asarray(values, dtype=np.int16).reshape(-1)
    return {str(level): int(np.count_nonzero(array == level)) for level in (-1, 0, 1, 2, 3)}


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _nanmean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.mean(finite))


def _nanmax(*, delta: np.ndarray) -> float | None:
    finite = np.asarray(delta, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.max(finite))


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
