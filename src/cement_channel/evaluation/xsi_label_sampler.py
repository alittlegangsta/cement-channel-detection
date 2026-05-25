from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.evaluation.correlation_schema import (
    MVP4A_SAMPLE_INDEX_VERSION,
    CorrelationConfig,
    load_correlation_config,
    reference_receiver_zero_based,
    xsi_side_azimuth_deg,
)


@dataclass(frozen=True)
class XsiLabelSampleReport:
    sample_index_version: str
    generated_at: str
    inputs: dict[str, str]
    shape: dict[str, int]
    label_source: str
    primary_label: str
    audit_label: str
    no_final_labels: bool
    coverage: dict[str, float | int | None]
    confidence: dict[str, float | None]
    orientation_confidence: dict[str, float | None]
    depth_match: dict[str, float | None]
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_xsi_label_samples_from_config(
    *,
    label_candidate_npz: Path | str,
    depth_only_npz: Path | str,
    orientation_confidence_npz: Path | str,
    correlation_config_path: Path | str,
) -> tuple[XsiLabelSampleReport, dict[str, np.ndarray]]:
    return build_xsi_label_samples(
        label_candidate_npz=label_candidate_npz,
        depth_only_npz=depth_only_npz,
        orientation_confidence_npz=orientation_confidence_npz,
        correlation_config=load_correlation_config(correlation_config_path),
        correlation_config_path=correlation_config_path,
    )


def build_xsi_label_samples(
    *,
    label_candidate_npz: Path | str,
    depth_only_npz: Path | str,
    orientation_confidence_npz: Path | str,
    correlation_config: CorrelationConfig,
    correlation_config_path: Path | str | None = None,
) -> tuple[XsiLabelSampleReport, dict[str, np.ndarray]]:
    label_arrays = _load_npz(label_candidate_npz)
    depth_arrays = _load_npz(depth_only_npz)
    orientation_arrays = _load_npz(orientation_confidence_npz)
    arrays, stats = build_xsi_label_samples_from_arrays(
        label_arrays=label_arrays,
        depth_arrays=depth_arrays,
        orientation_arrays=orientation_arrays,
        correlation_config=correlation_config,
    )
    report = XsiLabelSampleReport(
        sample_index_version=MVP4A_SAMPLE_INDEX_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "label_candidate_npz": str(label_candidate_npz),
            "depth_only_npz": str(depth_only_npz),
            "orientation_confidence_npz": str(orientation_confidence_npz),
            "correlation_config_path": (
                str(correlation_config_path) if correlation_config_path is not None else ""
            ),
        },
        shape={
            "depth": int(arrays["xsi_depth"].shape[0]),
            "side": int(arrays["xsi_side_azimuth_deg"].shape[0]),
        },
        label_source=correlation_config.label_source,
        primary_label=correlation_config.primary_label,
        audit_label=correlation_config.audit_label,
        no_final_labels=bool(np.asarray(arrays["no_final_labels"]).reshape(())),
        coverage=stats["coverage"],
        confidence=stats["confidence"],
        orientation_confidence=stats["orientation_confidence"],
        depth_match=stats["depth_match"],
        warnings=stats["warnings"],
        errors=stats["errors"],
        not_performed=[
            "final label generation",
            "XSI waveform reading",
            "XSI feature extraction",
            "STC",
            "APES",
            "model training",
        ],
    )
    return report, arrays


def build_xsi_label_samples_from_arrays(
    *,
    label_arrays: dict[str, np.ndarray],
    depth_arrays: dict[str, np.ndarray],
    orientation_arrays: dict[str, np.ndarray],
    correlation_config: CorrelationConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    warnings: list[str] = []
    errors: list[str] = []
    no_final_labels = bool(np.asarray(label_arrays.get("no_final_labels", False)).reshape(()))
    if not no_final_labels:
        errors.append("label candidate artifact does not set no_final_labels=true.")

    cast_depth = np.asarray(label_arrays["cast_depth"], dtype=np.float32).reshape(-1)
    cast_azimuth = np.asarray(
        label_arrays.get(
            "cast_azimuth_aligned_deg",
            np.linspace(
                0.0,
                360.0,
                num=np.asarray(label_arrays["presence_plus"]).shape[1],
                endpoint=False,
            ),
        ),
        dtype=np.float32,
    ).reshape(-1)
    xsi_depth_by_receiver = np.asarray(depth_arrays["xsi_depth_by_receiver"], dtype=np.float32)
    reference_index = reference_receiver_zero_based(correlation_config)
    if xsi_depth_by_receiver.ndim != 2:
        raise ValueError("depth_only_npz.xsi_depth_by_receiver must be rank 2.")
    if reference_index >= xsi_depth_by_receiver.shape[0]:
        raise ValueError("reference receiver index is outside xsi_depth_by_receiver.")
    xsi_depth = np.asarray(xsi_depth_by_receiver[reference_index], dtype=np.float32).reshape(-1)
    xsi_depth_index = np.arange(xsi_depth.size, dtype=np.int32)
    cast_depth_index = nearest_depth_indices(cast_depth, xsi_depth).astype(np.int32)
    cast_depth_matched = cast_depth[cast_depth_index]
    depth_mismatch = np.abs(cast_depth_matched - xsi_depth).astype(np.float32)
    if np.any(~np.isfinite(depth_mismatch)):
        warnings.append("Non-finite depth mismatch observed during CAST-to-XSI matching.")

    side_azimuth = xsi_side_azimuth_deg(correlation_config).astype(np.float32)
    side_bin = cast_azimuth_to_side_index(
        cast_azimuth,
        side_azimuth_deg=side_azimuth,
    )
    plus = _aggregate_label_set(
        presence=np.asarray(label_arrays["presence_plus"], dtype=np.int8)[cast_depth_index],
        severity=np.asarray(label_arrays["severity_plus"], dtype=np.int8)[cast_depth_index],
        confidence=np.asarray(label_arrays["label_confidence_plus"], dtype=np.float32)[
            cast_depth_index
        ],
        side_bin=side_bin,
        side_count=side_azimuth.size,
    )
    minus = _aggregate_label_set(
        presence=np.asarray(label_arrays["presence_minus_ablation"], dtype=np.int8)[
            cast_depth_index
        ],
        severity=np.asarray(label_arrays["severity_minus_ablation"], dtype=np.int8)[
            cast_depth_index
        ],
        confidence=np.asarray(label_arrays["label_confidence_minus_ablation"], dtype=np.float32)[
            cast_depth_index
        ],
        side_bin=side_bin,
        side_count=side_azimuth.size,
    )

    orientation_depth = np.asarray(orientation_arrays["pose_depth"], dtype=np.float32)
    orientation_values = np.asarray(
        orientation_arrays["orientation_confidence"],
        dtype=np.float32,
    )
    orientation_confidence_depth = interpolate_depth_values(
        source_depth=orientation_depth,
        source_values=orientation_values,
        target_depth=xsi_depth,
    )
    orientation_confidence_depth = np.where(
        np.isfinite(orientation_confidence_depth),
        orientation_confidence_depth,
        0.0,
    ).astype(np.float32)
    orientation_confidence = np.broadcast_to(
        orientation_confidence_depth.reshape(-1, 1),
        plus["presence"].shape,
    ).astype(np.float32)
    disagreement = plus["presence"] != minus["presence"]
    valid_known = plus["presence"] >= 0
    min_conf = correlation_config.min_label_confidence_for_azimuthal_validation
    if correlation_config.noncandidate_azimuthal_validation_requires_label_confidence:
        label_confidence_ok = plus["confidence"] >= min_conf
    else:
        label_confidence_ok = (plus["presence"] == 0) | (plus["confidence"] >= min_conf)
    valid_for_azimuthal = (
        valid_known & (orientation_confidence >= min_conf) & label_confidence_ok
    )
    if correlation_config.allow_low_confidence_for_non_azimuthal_summary:
        valid_for_non_azimuthal = valid_known
    else:
        valid_for_non_azimuthal = valid_for_azimuthal

    output = {
        "xsi_depth": xsi_depth.astype(np.float32),
        "xsi_depth_index": xsi_depth_index,
        "xsi_reference_receiver_index": np.asarray(
            correlation_config.reference_receiver_index,
            dtype=np.int16,
        ),
        "cast_depth_index": cast_depth_index,
        "cast_depth_matched": cast_depth_matched.astype(np.float32),
        "cast_depth_mismatch": depth_mismatch,
        "xsi_side_azimuth_deg": side_azimuth.astype(np.float32),
        "side_labels": np.asarray(correlation_config.side_labels),
        "label_presence_plus": plus["presence"].astype(np.int8),
        "label_severity_plus": plus["severity"].astype(np.int8),
        "label_confidence_plus": plus["confidence"].astype(np.float32),
        "label_presence_minus_audit": minus["presence"].astype(np.int8),
        "label_severity_minus_audit": minus["severity"].astype(np.int8),
        "label_confidence_minus_audit": minus["confidence"].astype(np.float32),
        "plus_minus_disagreement": disagreement.astype(bool),
        "orientation_confidence": orientation_confidence,
        "orientation_confidence_depth": orientation_confidence_depth,
        "valid_for_azimuthal_validation": valid_for_azimuthal.astype(bool),
        "valid_for_non_azimuthal_summary": valid_for_non_azimuthal.astype(bool),
        "no_final_labels": np.asarray(True),
    }
    metadata = {
        "sample_index_version": MVP4A_SAMPLE_INDEX_VERSION,
        "label_source": correlation_config.label_source,
        "primary_label": correlation_config.primary_label,
        "audit_label": correlation_config.audit_label,
        "reference_receiver_index": correlation_config.reference_receiver_index,
        "side_a_offset_deg": correlation_config.side_a_offset_deg,
        "side_order": correlation_config.side_order,
        "no_model_training": correlation_config.no_model_training,
        "no_final_labels": True,
    }
    output["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    stats = {
        "coverage": _coverage(output),
        "confidence": _numeric_summary(output["label_confidence_plus"]),
        "orientation_confidence": _numeric_summary(orientation_confidence_depth),
        "depth_match": _numeric_summary(depth_mismatch),
        "warnings": warnings,
        "errors": errors,
    }
    return output, stats


def write_xsi_label_sample_outputs(
    report: XsiLabelSampleReport,
    arrays: dict[str, np.ndarray],
    *,
    output_npz: Path,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_xsi_label_sample_markdown(report), encoding="utf-8")


def format_xsi_label_sample_markdown(report: XsiLabelSampleReport) -> str:
    data = report.to_dict()
    lines = [
        "# XSI Label Sample Index Report",
        "",
        f"- Version: {data['sample_index_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Label source: {data['label_source']}",
        f"- Primary label: {data['primary_label']}",
        f"- Audit label: {data['audit_label']}",
        f"- No final labels: {data['no_final_labels']}",
        f"- Shape: depth={data['shape']['depth']}, side={data['shape']['side']}",
        "",
        "## Coverage",
        "",
    ]
    for key, value in data["coverage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Confidence", ""])
    for key, value in data["confidence"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Orientation Confidence", ""])
    for key, value in data["orientation_confidence"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Depth Match", ""])
    for key, value in data["depth_match"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def nearest_depth_indices(source_depth: np.ndarray, target_depth: np.ndarray) -> np.ndarray:
    source = np.asarray(source_depth, dtype=np.float32).reshape(-1)
    target = np.asarray(target_depth, dtype=np.float32).reshape(-1)
    if source.size == 0:
        raise ValueError("source_depth is empty.")
    order = np.argsort(source)
    sorted_depth = source[order]
    positions = np.searchsorted(sorted_depth, target, side="left")
    positions = np.clip(positions, 0, sorted_depth.size - 1)
    prev_positions = np.clip(positions - 1, 0, sorted_depth.size - 1)
    next_diff = np.abs(sorted_depth[positions] - target)
    prev_diff = np.abs(sorted_depth[prev_positions] - target)
    nearest_sorted = np.where(prev_diff <= next_diff, prev_positions, positions)
    return order[nearest_sorted]


def interpolate_depth_values(
    *,
    source_depth: np.ndarray,
    source_values: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(source_depth, dtype=np.float32).reshape(-1)
    values = np.asarray(source_values, dtype=np.float32).reshape(-1)
    target = np.asarray(target_depth, dtype=np.float32).reshape(-1)
    valid = np.isfinite(depth) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.full(target.shape, np.nan, dtype=np.float32)
    depth = depth[valid]
    values = values[valid]
    order = np.argsort(depth)
    depth = depth[order]
    values = values[order]
    unique_depth, unique_indices = np.unique(depth, return_index=True)
    unique_values = values[unique_indices]
    return np.interp(
        target,
        unique_depth,
        unique_values,
        left=np.nan,
        right=np.nan,
    ).astype(np.float32)


def cast_azimuth_to_side_index(
    cast_azimuth_deg: np.ndarray,
    *,
    side_azimuth_deg: np.ndarray,
) -> np.ndarray:
    azimuth = np.asarray(cast_azimuth_deg, dtype=np.float32).reshape(-1)
    side_azimuth = np.asarray(side_azimuth_deg, dtype=np.float32).reshape(-1)
    if side_azimuth.size == 0:
        raise ValueError("side_azimuth_deg is empty.")
    distance = np.abs(((azimuth[:, None] - side_azimuth[None, :] + 180.0) % 360.0) - 180.0)
    return np.argmin(distance, axis=1).astype(np.int16)


def _aggregate_label_set(
    *,
    presence: np.ndarray,
    severity: np.ndarray,
    confidence: np.ndarray,
    side_bin: np.ndarray,
    side_count: int,
) -> dict[str, np.ndarray]:
    depth_count = presence.shape[0]
    output_presence = np.full((depth_count, side_count), -1, dtype=np.int8)
    output_severity = np.full((depth_count, side_count), -1, dtype=np.int8)
    output_confidence = np.zeros((depth_count, side_count), dtype=np.float32)
    for side_index in range(side_count):
        mask = side_bin == side_index
        if not np.any(mask):
            continue
        side_presence = presence[:, mask]
        side_severity = severity[:, mask]
        side_confidence = confidence[:, mask]
        known = side_presence >= 0
        candidate = side_presence == 1
        has_known = np.any(known, axis=1)
        has_candidate = np.any(candidate, axis=1)
        output_presence[has_known, side_index] = 0
        output_presence[has_candidate, side_index] = 1
        candidate_severity = np.max(np.where(candidate, side_severity, -1), axis=1)
        output_severity[has_known, side_index] = 0
        output_severity[has_candidate, side_index] = np.maximum(
            candidate_severity[has_candidate],
            1,
        ).astype(np.int8)
        output_confidence[:, side_index] = _aggregate_confidence(
            side_confidence,
            known=known,
            candidate=candidate,
            has_candidate=has_candidate,
        )
    return {
        "presence": output_presence,
        "severity": output_severity,
        "confidence": output_confidence,
    }


def _aggregate_confidence(
    confidence: np.ndarray,
    *,
    known: np.ndarray,
    candidate: np.ndarray,
    has_candidate: np.ndarray,
) -> np.ndarray:
    valid_confidence = np.where(known & np.isfinite(confidence), confidence, np.nan)
    candidate_confidence = np.where(candidate & np.isfinite(confidence), confidence, np.nan)
    max_valid = _safe_nanmax(valid_confidence, axis=1)
    max_candidate = _safe_nanmax(candidate_confidence, axis=1)
    result = np.where(has_candidate, max_candidate, max_valid)
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


def _safe_nanmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    filled = np.where(finite, array, -np.inf)
    maximum = np.max(filled, axis=axis)
    has_finite = np.any(finite, axis=axis)
    return np.where(has_finite, maximum, np.nan).astype(np.float32)


def _coverage(arrays: dict[str, np.ndarray]) -> dict[str, float | int | None]:
    plus = np.asarray(arrays["label_presence_plus"])
    valid_az = np.asarray(arrays["valid_for_azimuthal_validation"], dtype=bool)
    valid_non_az = np.asarray(arrays["valid_for_non_azimuthal_summary"], dtype=bool)
    disagreement = np.asarray(arrays["plus_minus_disagreement"], dtype=bool)
    candidate = plus == 1
    known = plus >= 0
    return {
        "depth_count": int(plus.shape[0]),
        "side_count": int(plus.shape[1]),
        "candidate_fraction_plus": _fraction(candidate, known),
        "known_fraction_plus": float(np.mean(known)) if known.size else None,
        "plus_minus_disagreement_fraction": _fraction(disagreement, known),
        "valid_for_azimuthal_validation_count": int(np.count_nonzero(valid_az)),
        "valid_for_azimuthal_validation_fraction": (
            float(np.mean(valid_az)) if valid_az.size else None
        ),
        "valid_for_non_azimuthal_summary_count": int(np.count_nonzero(valid_non_az)),
        "valid_for_non_azimuthal_summary_fraction": (
            float(np.mean(valid_non_az)) if valid_non_az.size else None
        ),
        "high_confidence_candidate_count": int(np.count_nonzero(valid_az & candidate)),
        "high_confidence_non_candidate_count": int(np.count_nonzero(valid_az & (plus == 0))),
    }


def _numeric_summary(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {"finite_ratio": None, "min": None, "max": None, "mean": None, "median": None}
    finite = np.isfinite(array)
    finite_ratio = float(np.mean(finite))
    if not np.any(finite):
        return {
            "finite_ratio": finite_ratio,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    finite_values = array[finite]
    return {
        "finite_ratio": finite_ratio,
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def _fraction(mask: np.ndarray, denominator_mask: np.ndarray) -> float | None:
    denominator = int(np.count_nonzero(denominator_mask))
    if denominator == 0:
        return None
    return float(np.count_nonzero(mask & denominator_mask) / denominator)


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
