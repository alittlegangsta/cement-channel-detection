from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.labels.depth_level_schema import (
    DEPTH_LEVEL_LABEL_REPORT_VERSION,
    DEPTH_LEVEL_LABEL_VERSION,
    DepthLevelLabelConfig,
    DepthLevelReviewIntervalConfig,
    active_depth_review_intervals,
    load_depth_level_label_config,
)


@dataclass(frozen=True)
class DepthLevelLabelReport:
    report_version: str
    label_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    depth_count: int
    positive_count: int
    negative_count: int
    positive_fraction: float | None
    strong_positive_count: int
    clear_negative_count: int
    candidate_fraction_distribution: dict[str, float | None]
    max_severity_distribution: dict[str, int]
    plus_minus_disagreement_distribution: dict[str, float | None]
    confidence_distribution: dict[str, float | None]
    review_band_impact: dict[str, int | float | None]
    example_strong_positive_intervals: list[dict[str, float | int | None]]
    example_clear_negative_intervals: list[dict[str, float | int | None]]
    zc_source_field: str | None
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


def build_depth_level_labels_from_config(
    *,
    cast_weak_label_npz: Path | str,
    xsi_label_samples_npz: Path | str,
    depth_level_config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    sample_table_npz: Path | str | None = None,
    overwrite: bool = False,
) -> DepthLevelLabelReport:
    config = load_depth_level_label_config(depth_level_config_path)
    cast_arrays = _load_npz(cast_weak_label_npz)
    xsi_arrays = _load_npz(xsi_label_samples_npz)
    _sample_arrays = _load_npz(sample_table_npz) if sample_table_npz else None
    output, report = build_depth_level_label_table(
        cast_arrays=cast_arrays,
        xsi_arrays=xsi_arrays,
        config=config,
        inputs={
            "cast_weak_label_npz": str(cast_weak_label_npz),
            "xsi_label_samples_npz": str(xsi_label_samples_npz),
            "depth_level_config_path": str(depth_level_config_path),
            "sample_table_npz": str(sample_table_npz or ""),
        },
        output_npz=Path(output_npz),
    )
    write_depth_level_label_outputs(
        output,
        report,
        output_npz=Path(output_npz),
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return report


def build_depth_level_label_table(
    *,
    cast_arrays: dict[str, np.ndarray],
    xsi_arrays: dict[str, np.ndarray],
    config: DepthLevelLabelConfig,
    inputs: dict[str, str] | None = None,
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], DepthLevelLabelReport]:
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(cast_arrays, xsi_arrays, errors)
    depth = np.asarray(xsi_arrays["xsi_depth"], dtype=np.float32).reshape(-1)
    cast_depth = np.asarray(cast_arrays["cast_depth"], dtype=np.float32).reshape(-1)
    cast_depth_index = _cast_depth_index(xsi_arrays, cast_depth, depth)
    cast_depth_matched = cast_depth[cast_depth_index]
    presence = np.asarray(cast_arrays["presence_plus"], dtype=np.int8)[cast_depth_index]
    severity = np.asarray(cast_arrays["severity_plus"], dtype=np.int8)[cast_depth_index]
    confidence = np.asarray(cast_arrays["label_confidence_plus"], dtype=np.float32)[
        cast_depth_index
    ]
    presence_minus = np.asarray(cast_arrays["presence_minus_ablation"], dtype=np.int8)[
        cast_depth_index
    ]
    zc_values, zc_source = _optional_zc_values(cast_arrays, cast_depth_index)
    if zc_values is None:
        warnings.append(
            "CAST weak-label NPZ does not contain raw Zc; depth_min_zc, depth_p05_zc, "
            "and depth_p10_zc are NaN review fields."
        )
        zc_values = np.full(presence.shape, np.nan, dtype=np.float32)
    relative_drop = np.asarray(
        cast_arrays.get("relative_drop_plus", np.full_like(confidence, np.nan)),
        dtype=np.float32,
    )[cast_depth_index]
    orientation = _orientation_confidence_depth(xsi_arrays, depth.size)
    cast_azimuth = _cast_azimuth_deg(cast_arrays, presence.shape[1])

    valid = presence >= 0
    candidate = presence == 1
    valid_count = np.count_nonzero(valid, axis=1).astype(np.int32)
    candidate_count = np.count_nonzero(candidate, axis=1).astype(np.int32)
    valid_denominator = np.maximum(valid_count, 1)
    candidate_fraction = (candidate_count / valid_denominator).astype(np.float32)
    has_channel = candidate_count > 0
    max_severity = _row_max_int(np.where(candidate, severity, -1), default=-1)
    max_severity = np.where(has_channel, max_severity, 0).astype(np.int8)
    max_confidence = _row_nanmax(np.where(valid, confidence, np.nan), default=0.0)
    candidate_max_confidence = _row_nanmax(np.where(candidate, confidence, np.nan), default=0.0)
    valid_fraction = (valid_count / max(presence.shape[1], 1)).astype(np.float32)
    disagreement_known = valid & (presence_minus >= 0)
    disagreement = disagreement_known & (presence != presence_minus)
    disagreement_denominator = np.maximum(np.count_nonzero(disagreement_known, axis=1), 1)
    disagreement_fraction = (
        np.count_nonzero(disagreement, axis=1) / disagreement_denominator
    ).astype(np.float32)
    zc_min = _row_nanpercentile(zc_values, 0.0)
    zc_p05 = _row_nanpercentile(zc_values, 5.0)
    zc_p10 = _row_nanpercentile(zc_values, 10.0)
    max_relative_drop = _row_nanmax(np.where(valid, relative_drop, np.nan), default=np.nan)
    largest_width = np.asarray(
        [
            _largest_circular_run_width(row, cast_azimuth)
            for row in candidate.astype(bool)
        ],
        dtype=np.float32,
    )
    depth_label_confidence = _depth_label_confidence(
        has_channel=has_channel,
        candidate_max_confidence=candidate_max_confidence,
        valid_fraction=valid_fraction,
        orientation_confidence=orientation,
        disagreement_fraction=disagreement_fraction,
    )

    review_mask = depth_review_interval_mask(depth, active_depth_review_intervals(config))
    strong_positive = (
        has_channel
        & (candidate_fraction >= config.quality_policy.strong_positive.min_candidate_fraction)
        & (max_severity >= config.quality_policy.strong_positive.min_max_severity)
        & (depth_label_confidence >= config.quality_policy.strong_positive.min_label_confidence)
        & (
            disagreement_fraction
            <= config.quality_policy.strong_positive.max_plus_minus_disagreement_fraction
        )
        & (orientation >= config.quality_policy.strong_positive.min_orientation_confidence)
        & ~review_mask
    )
    clear_negative = (
        (candidate_fraction <= config.quality_policy.clear_negative.max_candidate_fraction)
        & (depth_label_confidence >= config.quality_policy.clear_negative.min_label_confidence)
        & (
            disagreement_fraction
            <= config.quality_policy.clear_negative.max_plus_minus_disagreement_fraction
        )
        & (orientation >= config.quality_policy.clear_negative.min_orientation_confidence)
        & ~review_mask
    )
    output = {
        "depth": depth.astype(np.float32),
        "cast_depth_index": cast_depth_index.astype(np.int32),
        "cast_depth_matched": cast_depth_matched.astype(np.float32),
        "depth_has_channel_any": has_channel.astype(bool),
        "depth_candidate_fraction": candidate_fraction.astype(np.float32),
        "depth_max_severity": max_severity.astype(np.int8),
        "depth_max_confidence": max_confidence.astype(np.float32),
        "depth_min_zc": zc_min.astype(np.float32),
        "depth_p05_zc": zc_p05.astype(np.float32),
        "depth_p10_zc": zc_p10.astype(np.float32),
        "depth_max_relative_drop": max_relative_drop.astype(np.float32),
        "depth_largest_azimuth_object_width": largest_width.astype(np.float32),
        "depth_plus_minus_disagreement_fraction": disagreement_fraction.astype(np.float32),
        "depth_orientation_confidence": orientation.astype(np.float32),
        "depth_label_confidence": depth_label_confidence.astype(np.float32),
        "depth_valid_fraction": valid_fraction.astype(np.float32),
        "depth_candidate_count": candidate_count.astype(np.int32),
        "depth_valid_azimuth_count": valid_count.astype(np.int32),
        "depth_strong_positive_mask": strong_positive.astype(bool),
        "depth_clear_negative_mask": clear_negative.astype(bool),
        "depth_review_band_mask": review_mask.astype(bool),
        "depth_label_version": np.asarray(DEPTH_LEVEL_LABEL_VERSION),
        "depth_label_metadata_json": np.asarray(
            json.dumps(
                {
                    "config_version": config.config_version,
                    "primary_label": config.primary_label,
                    "audit_label": config.audit_label,
                    "side_level_labels": "audit_only",
                    "no_final_labels": True,
                    "zc_source_field": zc_source,
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
    errors.extend(_stop_condition_errors(output, config))
    report = _build_report(
        output,
        config=config,
        inputs=inputs or {},
        output_npz=output_npz,
        zc_source=zc_source,
        warnings=warnings,
        errors=errors,
    )
    return output, report


def depth_review_interval_mask(
    depth: np.ndarray,
    intervals: tuple[DepthLevelReviewIntervalConfig, ...],
) -> np.ndarray:
    mask = np.zeros(depth.size, dtype=bool)
    for interval in intervals:
        mask |= (depth >= interval.depth_min_ft) & (depth <= interval.depth_max_ft)
    return mask


def write_depth_level_label_outputs(
    arrays: dict[str, np.ndarray],
    report: DepthLevelLabelReport,
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
    output_md.write_text(format_depth_level_label_markdown(report), encoding="utf-8")


def format_depth_level_label_markdown(report: DepthLevelLabelReport) -> str:
    lines = [
        "# MVP-4B-R4 Depth-Level CAST Weak-Label Candidate Report",
        "",
        "This table is a depth-level review target built from CAST weak-label candidates. "
        "It is not final labels and does not claim ground truth.",
        "",
        f"- label_version: `{report.label_version}`",
        f"- depth_count: {report.depth_count}",
        f"- positive_fraction: {report.positive_fraction}",
        f"- strong_positive_count: {report.strong_positive_count}",
        f"- clear_negative_count: {report.clear_negative_count}",
        f"- no_final_labels: `{report.no_final_labels}`",
        f"- zc_source_field: `{report.zc_source_field}`",
        "",
        "## Candidate Fraction Distribution",
        "",
    ]
    lines.extend(_dict_lines(report.candidate_fraction_distribution))
    lines.extend(["", "## Max Severity Distribution", ""])
    lines.extend(_dict_lines(report.max_severity_distribution))
    lines.extend(["", "## Plus/Minus Disagreement By Depth", ""])
    lines.extend(_dict_lines(report.plus_minus_disagreement_distribution))
    lines.extend(["", "## Confidence Distribution", ""])
    lines.extend(_dict_lines(report.confidence_distribution))
    lines.extend(["", "## 5700 ft Review Band Impact", ""])
    lines.extend(_dict_lines(report.review_band_impact))
    lines.extend(["", "## Example Strong Positive Intervals", ""])
    lines.extend(_interval_lines(report.example_strong_positive_intervals))
    lines.extend(["", "## Example Clear Negative Intervals", ""])
    lines.extend(_interval_lines(report.example_clear_negative_intervals))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _build_report(
    output: dict[str, np.ndarray],
    *,
    config: DepthLevelLabelConfig,
    inputs: dict[str, str],
    output_npz: Path | None,
    zc_source: str | None,
    warnings: list[str],
    errors: list[str],
) -> DepthLevelLabelReport:
    has_channel = np.asarray(output["depth_has_channel_any"], dtype=bool)
    negative = ~has_channel
    positive_count = int(np.count_nonzero(has_channel))
    negative_count = int(np.count_nonzero(negative))
    strong = np.asarray(output["depth_strong_positive_mask"], dtype=bool)
    clear = np.asarray(output["depth_clear_negative_mask"], dtype=bool)
    return DepthLevelLabelReport(
        report_version=DEPTH_LEVEL_LABEL_REPORT_VERSION,
        label_version=DEPTH_LEVEL_LABEL_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        output_npz=str(output_npz) if output_npz else "",
        depth_count=int(has_channel.size),
        positive_count=positive_count,
        negative_count=negative_count,
        positive_fraction=None if has_channel.size == 0 else positive_count / has_channel.size,
        strong_positive_count=int(np.count_nonzero(strong)),
        clear_negative_count=int(np.count_nonzero(clear)),
        candidate_fraction_distribution=_numeric_distribution(
            output["depth_candidate_fraction"]
        ),
        max_severity_distribution=_severity_distribution(output["depth_max_severity"]),
        plus_minus_disagreement_distribution=_numeric_distribution(
            output["depth_plus_minus_disagreement_fraction"]
        ),
        confidence_distribution=_numeric_distribution(output["depth_label_confidence"]),
        review_band_impact=_review_band_impact(output, config),
        example_strong_positive_intervals=_example_intervals(
            output["depth"],
            strong,
            score=output["depth_candidate_fraction"],
        ),
        example_clear_negative_intervals=_example_intervals(
            output["depth"],
            clear,
            score=output["depth_label_confidence"],
        ),
        zc_source_field=zc_source,
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
            "side-level training target creation",
            "model training",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )


def _stop_condition_errors(
    output: dict[str, np.ndarray],
    config: DepthLevelLabelConfig,
) -> list[str]:
    errors: list[str] = []
    positive_count = int(np.count_nonzero(output["depth_strong_positive_mask"]))
    negative_count = int(np.count_nonzero(output["depth_clear_negative_mask"]))
    if positive_count < config.gate.min_depth_positive_count:
        errors.append("depth-level strong-positive subset is empty or below minimum.")
    if negative_count < config.gate.min_depth_negative_count:
        errors.append("depth-level clear-negative subset is empty or below minimum.")
    review = np.asarray(output["depth_review_band_mask"], dtype=bool)
    has_channel = np.asarray(output["depth_has_channel_any"], dtype=bool)
    positive_any = int(np.count_nonzero(has_channel))
    review_positive = int(np.count_nonzero(has_channel & review))
    review_fraction = 0.0 if positive_any == 0 else review_positive / positive_any
    if review_fraction > config.gate.max_5700_band_positive_fraction:
        errors.append("depth-level positive subset is dominated by the ~5700 ft review band.")
    return errors


def _review_band_impact(
    output: dict[str, np.ndarray],
    config: DepthLevelLabelConfig,
) -> dict[str, int | float | None]:
    review = np.asarray(output["depth_review_band_mask"], dtype=bool)
    has_channel = np.asarray(output["depth_has_channel_any"], dtype=bool)
    strong = np.asarray(output["depth_strong_positive_mask"], dtype=bool)
    clear = np.asarray(output["depth_clear_negative_mask"], dtype=bool)
    positive_total = int(np.count_nonzero(has_channel))
    review_positive = int(np.count_nonzero(review & has_channel))
    return {
        "review_depth_count": int(np.count_nonzero(review)),
        "review_positive_count": review_positive,
        "review_strong_positive_count": int(np.count_nonzero(review & strong)),
        "review_clear_negative_count": int(np.count_nonzero(review & clear)),
        "positive_fraction_in_review_band": (
            None if positive_total == 0 else review_positive / positive_total
        ),
        "max_allowed_positive_fraction_in_review_band": (
            config.gate.max_5700_band_positive_fraction
        ),
    }


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


def _largest_circular_run_width(candidate_row: np.ndarray, azimuth_deg: np.ndarray) -> float:
    row = np.asarray(candidate_row, dtype=bool)
    if row.size == 0 or not np.any(row):
        return 0.0
    if np.all(row):
        return 360.0
    step = _azimuth_step_deg(azimuth_deg, row.size)
    doubled = np.concatenate([row, row])
    best = 0
    current = 0
    for value in doubled:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
        if best >= row.size:
            break
    return float(min(best, row.size) * step)


def _azimuth_step_deg(azimuth_deg: np.ndarray, count: int) -> float:
    values = np.asarray(azimuth_deg, dtype=np.float64).reshape(-1)
    if values.size > 1:
        diffs = np.diff(np.sort(values))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            return float(np.median(diffs))
    return 360.0 / max(count, 1)


def _cast_depth_index(
    xsi_arrays: dict[str, np.ndarray],
    cast_depth: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    if "cast_depth_index" in xsi_arrays:
        return np.asarray(xsi_arrays["cast_depth_index"], dtype=np.int32).reshape(-1)
    order = np.argsort(cast_depth)
    sorted_depth = cast_depth[order]
    insert = np.searchsorted(sorted_depth, target_depth)
    insert = np.clip(insert, 1, sorted_depth.size - 1)
    left = insert - 1
    right = insert
    choose_right = np.abs(sorted_depth[right] - target_depth) < np.abs(
        sorted_depth[left] - target_depth
    )
    return order[np.where(choose_right, right, left)].astype(np.int32)


def _optional_zc_values(
    cast_arrays: dict[str, np.ndarray],
    cast_depth_index: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    for key in ("zc_plus", "cast_zc_plus", "cast_zc", "zc", "Zc"):
        if key in cast_arrays:
            return np.asarray(cast_arrays[key], dtype=np.float32)[cast_depth_index], key
    return None, None


def _orientation_confidence_depth(
    xsi_arrays: dict[str, np.ndarray],
    depth_count: int,
) -> np.ndarray:
    if "orientation_confidence_depth" in xsi_arrays:
        values = np.asarray(
            xsi_arrays["orientation_confidence_depth"],
            dtype=np.float32,
        ).reshape(-1)
    else:
        orientation = np.asarray(xsi_arrays["orientation_confidence"], dtype=np.float32)
        values = (
            np.nanmean(orientation, axis=1).astype(np.float32)
            if orientation.ndim == 2
            else orientation.reshape(-1)
        )
    if values.size != depth_count:
        raise ValueError(
            "orientation confidence depth length does not match XSI depth length: "
            f"{values.size} != {depth_count}."
        )
    return np.where(np.isfinite(values), values, 0.0).astype(np.float32)


def _cast_azimuth_deg(cast_arrays: dict[str, np.ndarray], azimuth_count: int) -> np.ndarray:
    values = np.asarray(
        cast_arrays.get(
            "cast_azimuth_aligned_deg",
            np.linspace(0.0, 360.0, num=azimuth_count, endpoint=False),
        ),
        dtype=np.float32,
    ).reshape(-1)
    if values.size != azimuth_count:
        return np.linspace(0.0, 360.0, num=azimuth_count, endpoint=False, dtype=np.float32)
    return values


def _validate_guardrails(
    cast_arrays: dict[str, np.ndarray],
    xsi_arrays: dict[str, np.ndarray],
    errors: list[str],
) -> None:
    required_cast = (
        "cast_depth",
        "presence_plus",
        "severity_plus",
        "label_confidence_plus",
        "presence_minus_ablation",
    )
    required_xsi = ("xsi_depth", "orientation_confidence")
    missing_cast = [key for key in required_cast if key not in cast_arrays]
    missing_xsi = [key for key in required_xsi if key not in xsi_arrays]
    if missing_cast:
        raise KeyError("CAST weak-label NPZ missing required field(s): " + ", ".join(missing_cast))
    if missing_xsi:
        raise KeyError("XSI label sample NPZ missing required field(s): " + ", ".join(missing_xsi))
    if not bool(np.asarray(cast_arrays.get("no_final_labels", False)).reshape(())):
        errors.append("CAST weak-label candidate input must set no_final_labels=true.")
    if not bool(np.asarray(xsi_arrays.get("no_final_labels", False)).reshape(())):
        errors.append("XSI label sample input must set no_final_labels=true.")
    for forbidden in ("no_stc", "no_apes"):
        if forbidden in cast_arrays and not bool(np.asarray(cast_arrays[forbidden]).reshape(())):
            errors.append(f"CAST weak-label input must preserve {forbidden}=true.")


def _row_max_int(values: np.ndarray, *, default: int) -> np.ndarray:
    if values.shape[1] == 0:
        return np.full(values.shape[0], default, dtype=np.int16)
    return np.max(values, axis=1).astype(np.int16)


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


def _example_intervals(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    score: np.ndarray,
    limit: int = 5,
) -> list[dict[str, float | int | None]]:
    intervals: list[dict[str, float | int | None]] = []
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return intervals
    groups = np.split(selected, np.where(np.diff(selected) > 1)[0] + 1)
    for group in groups:
        group_score = np.asarray(score)[group]
        intervals.append(
            {
                "depth_min_ft": float(np.min(depth[group])),
                "depth_max_ft": float(np.max(depth[group])),
                "sample_count": int(group.size),
                "max_score": (
                    None
                    if not np.any(np.isfinite(group_score))
                    else float(np.nanmax(group_score))
                ),
            }
        )
    intervals.sort(
        key=lambda item: (float(item["max_score"] or 0.0), item["sample_count"]),
        reverse=True,
    )
    return intervals[:limit]


def _load_npz(path: Path | str | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _interval_lines(intervals: list[dict[str, float | int | None]]) -> list[str]:
    if not intervals:
        return ["- none"]
    return [
        "- "
        f"{item['depth_min_ft']} to {item['depth_max_ft']} ft: "
        f"samples={item['sample_count']}, max_score={item['max_score']}"
        for item in intervals
    ]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
