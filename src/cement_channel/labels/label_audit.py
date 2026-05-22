from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as connected_label

from cement_channel.labels.cast_label_input import load_label_config
from cement_channel.labels.schema import PresenceLabel

LABEL_AUDIT_VERSION = "cast_weak_label_audit_v001"


@dataclass(frozen=True)
class ComponentSummary:
    component_count: int
    object_depth_length_m: dict[str, float | None]
    object_azimuth_width_deg: dict[str, float | None]
    isolated_speckle_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelAuditReport:
    label_audit_version: str
    generated_at: str
    inputs: dict[str, str]
    coverage: dict[str, Any]
    plus_minus_disagreement_rate: float | None
    confidence_distribution: dict[str, int]
    severity_distribution: dict[str, dict[str, int]]
    components: dict[str, dict[str, Any]]
    low_confidence_fraction: dict[str, float | None]
    no_final_labels: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_cast_weak_labels_from_config(
    *,
    weak_label_npz: Path | str,
    label_config_path: Path | str,
) -> LabelAuditReport:
    return audit_cast_weak_labels(
        weak_label_npz=weak_label_npz,
        label_config=load_label_config(label_config_path),
        label_config_path=label_config_path,
    )


def audit_cast_weak_labels(
    *,
    weak_label_npz: Path | str,
    label_config: dict[str, Any],
    label_config_path: Path | str | None = None,
) -> LabelAuditReport:
    arrays = _load_npz(weak_label_npz)
    depth = np.asarray(arrays["cast_depth"], dtype=np.float32)
    azimuth = np.asarray(arrays["cast_azimuth_aligned_deg"], dtype=np.float32)
    presence_plus = np.asarray(arrays["presence_plus"], dtype=np.int8)
    presence_minus = np.asarray(arrays["presence_minus_ablation"], dtype=np.int8)
    confidence_plus = np.asarray(arrays["label_confidence_plus"], dtype=np.float32)
    confidence_minus = np.asarray(arrays["label_confidence_minus_ablation"], dtype=np.float32)
    severity_plus = np.asarray(arrays["severity_plus"], dtype=np.int8)
    severity_minus = np.asarray(arrays["severity_minus_ablation"], dtype=np.int8)
    no_final_labels = bool(np.asarray(arrays.get("no_final_labels", False)).reshape(()))

    warnings: list[str] = []
    errors: list[str] = []
    audit_config = _as_dict(label_config.get("audit"))
    low_conf_threshold = float(audit_config.get("low_confidence_threshold", 0.25))
    isolated_max_pixels = int(audit_config.get("isolated_object_max_pixels", 3))
    disagreement_warning = float(audit_config.get("max_plus_minus_disagreement_warning", 0.25))
    disagreement_blocking = float(audit_config.get("max_plus_minus_disagreement_blocking", 0.50))
    threshold_config = _as_dict(label_config.get("threshold"))

    plus_coverage = _candidate_coverage(presence_plus)
    minus_coverage = _candidate_coverage(presence_minus)
    disagreement = _disagreement_rate(presence_plus, presence_minus)
    _coverage_warning(plus_coverage, "plus", threshold_config, warnings, errors)
    _coverage_warning(minus_coverage, "minus_ablation", threshold_config, warnings, errors)
    if disagreement is not None:
        if disagreement > disagreement_blocking:
            errors.append(f"plus/minus disagreement is blocking high: {disagreement}.")
        elif disagreement > disagreement_warning:
            warnings.append(f"plus/minus disagreement is high: {disagreement}.")
    if not no_final_labels:
        errors.append("Candidate NPZ does not set no_final_labels=true.")

    components_plus = _component_summary(
        presence_plus == PresenceLabel.CHANNEL_CANDIDATE,
        depth,
        azimuth,
        isolated_max_pixels=isolated_max_pixels,
    )
    components_minus = _component_summary(
        presence_minus == PresenceLabel.CHANNEL_CANDIDATE,
        depth,
        azimuth,
        isolated_max_pixels=isolated_max_pixels,
    )
    report = LabelAuditReport(
        label_audit_version=LABEL_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "weak_label_npz": str(weak_label_npz),
            "label_config_path": str(label_config_path) if label_config_path is not None else "",
        },
        coverage={
            "plus": plus_coverage,
            "minus_ablation": minus_coverage,
            "plus_by_depth": _axis_ratio_summary(
                presence_plus == PresenceLabel.CHANNEL_CANDIDATE,
                axis=1,
            ),
            "plus_by_azimuth": _axis_ratio_summary(
                presence_plus == PresenceLabel.CHANNEL_CANDIDATE,
                axis=0,
            ),
            "minus_by_depth": _axis_ratio_summary(
                presence_minus == PresenceLabel.CHANNEL_CANDIDATE,
                axis=1,
            ),
            "minus_by_azimuth": _axis_ratio_summary(
                presence_minus == PresenceLabel.CHANNEL_CANDIDATE,
                axis=0,
            ),
        },
        plus_minus_disagreement_rate=disagreement,
        confidence_distribution=_confidence_distribution(confidence_plus, confidence_minus),
        severity_distribution={
            "plus": _severity_distribution(severity_plus),
            "minus_ablation": _severity_distribution(severity_minus),
        },
        components={
            "plus": components_plus.to_dict(),
            "minus_ablation": components_minus.to_dict(),
        },
        low_confidence_fraction={
            "plus": _low_confidence_fraction(presence_plus, confidence_plus, low_conf_threshold),
            "minus_ablation": _low_confidence_fraction(
                presence_minus,
                confidence_minus,
                low_conf_threshold,
            ),
        },
        no_final_labels=no_final_labels,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "final label approval",
            "manual review approval",
            "feature extraction",
            "model training",
            "MVP-4 correlation validation",
        ],
    )
    return report


def write_label_audit_outputs(
    report: LabelAuditReport,
    *,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_label_audit_markdown(report), encoding="utf-8")


def format_label_audit_markdown(report: LabelAuditReport) -> str:
    data = report.to_dict()
    lines = [
        "# CAST Weak-Label Audit",
        "",
        f"- Version: {data['label_audit_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Plus coverage: {data['coverage']['plus']}",
        f"- Minus coverage: {data['coverage']['minus_ablation']}",
        f"- Plus/minus disagreement: {data['plus_minus_disagreement_rate']}",
        f"- No final labels: {data['no_final_labels']}",
        "",
        "## Components",
        "",
    ]
    for key, value in data["components"].items():
        lines.append(
            f"- {key}: count={value['component_count']}, "
            f"isolated_speckle_ratio={value['isolated_speckle_ratio']}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _component_summary(
    mask: np.ndarray,
    depth: np.ndarray,
    azimuth: np.ndarray,
    *,
    isolated_max_pixels: int,
) -> ComponentSummary:
    candidate = np.asarray(mask, dtype=bool)
    if candidate.size == 0 or not np.any(candidate):
        return ComponentSummary(
            component_count=0,
            object_depth_length_m=_stats([]),
            object_azimuth_width_deg=_stats([]),
            isolated_speckle_ratio=None,
        )
    tiled = np.concatenate([candidate, candidate, candidate], axis=1)
    labels, _count = connected_label(tiled, structure=np.ones((3, 3), dtype=bool))
    azimuth_count = candidate.shape[1]
    center = labels[:, azimuth_count : 2 * azimuth_count]
    component_ids = np.unique(center[center > 0])
    parent = {int(component_id): int(component_id) for component_id in component_ids}
    for row in range(center.shape[0]):
        for delta in (-1, 0, 1):
            other_row = row + delta
            if other_row < 0 or other_row >= center.shape[0]:
                continue
            left_id = int(center[row, 0])
            right_id = int(center[other_row, -1])
            if left_id > 0 and right_id > 0:
                _union(parent, left_id, right_id)
    rows_all, cols_all = np.where(center > 0)
    label_ids = center[rows_all, cols_all]
    roots = np.asarray([_find(parent, int(label_id)) for label_id in label_ids], dtype=np.int32)
    if roots.size == 0:
        return ComponentSummary(
            component_count=0,
            object_depth_length_m=_stats([]),
            object_azimuth_width_deg=_stats([]),
            isolated_speckle_ratio=None,
        )
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_rows = rows_all[order]
    sorted_cols = cols_all[order]
    bounds = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.diff(sorted_inverse)) + 1,
            np.array([sorted_inverse.size], dtype=np.int64),
        ]
    )
    depth_lengths: list[float] = []
    azimuth_widths: list[float] = []
    isolated_cells = 0
    total_cells = int(np.count_nonzero(candidate))
    depth_step = _median_step(depth)
    azimuth_step = _median_step(azimuth) or (360.0 / max(azimuth_count, 1))
    for start, stop in zip(bounds[:-1], bounds[1:], strict=False):
        rows = sorted_rows[start:stop]
        cols = sorted_cols[start:stop]
        if rows.size == 0:
            continue
        area = int(rows.size)
        if area <= isolated_max_pixels:
            isolated_cells += area
        depth_lengths.append((float(rows.max() - rows.min()) + 1.0) * depth_step)
        azimuth_widths.append(float(np.unique(cols).size) * azimuth_step)
    return ComponentSummary(
        component_count=int(unique_roots.size),
        object_depth_length_m=_stats(depth_lengths),
        object_azimuth_width_deg=_stats(azimuth_widths),
        isolated_speckle_ratio=None if total_cells == 0 else isolated_cells / total_cells,
    )


def _find(parent: dict[int, int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: dict[int, int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _candidate_coverage(presence: np.ndarray) -> float | None:
    valid = presence != PresenceLabel.UNKNOWN
    if not np.any(valid):
        return None
    return float(np.mean(presence[valid] == PresenceLabel.CHANNEL_CANDIDATE))


def _disagreement_rate(presence_a: np.ndarray, presence_b: np.ndarray) -> float | None:
    valid = (presence_a != PresenceLabel.UNKNOWN) & (presence_b != PresenceLabel.UNKNOWN)
    if not np.any(valid):
        return None
    return float(
        np.mean(
            (presence_a[valid] == PresenceLabel.CHANNEL_CANDIDATE)
            != (presence_b[valid] == PresenceLabel.CHANNEL_CANDIDATE)
        )
    )


def _axis_ratio_summary(mask: np.ndarray, *, axis: int) -> dict[str, float | None]:
    if mask.size == 0:
        return _stats([])
    ratios = np.mean(mask, axis=axis)
    return _stats([float(item) for item in ratios])


def _confidence_distribution(
    confidence_plus: np.ndarray,
    confidence_minus: np.ndarray,
) -> dict[str, int]:
    values = np.concatenate([confidence_plus.reshape(-1), confidence_minus.reshape(-1)])
    finite = values[np.isfinite(values)]
    bins = {
        "0.00-0.25": int(np.count_nonzero((finite >= 0.0) & (finite < 0.25))),
        "0.25-0.50": int(np.count_nonzero((finite >= 0.25) & (finite < 0.50))),
        "0.50-0.75": int(np.count_nonzero((finite >= 0.50) & (finite < 0.75))),
        "0.75-1.00": int(np.count_nonzero((finite >= 0.75) & (finite <= 1.0))),
    }
    return bins


def _severity_distribution(severity: np.ndarray) -> dict[str, int]:
    return {
        "unknown": int(np.count_nonzero(severity == -1)),
        "none": int(np.count_nonzero(severity == 0)),
        "mild": int(np.count_nonzero(severity == 1)),
        "moderate": int(np.count_nonzero(severity == 2)),
        "severe": int(np.count_nonzero(severity == 3)),
    }


def _low_confidence_fraction(
    presence: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> float | None:
    candidate = presence == PresenceLabel.CHANNEL_CANDIDATE
    if not np.any(candidate):
        return None
    return float(np.mean(confidence[candidate] < threshold))


def _coverage_warning(
    coverage: float | None,
    prefix: str,
    config: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> None:
    if coverage is None:
        errors.append(f"{prefix} coverage is undefined.")
        return
    warning_min = float(config.get("candidate_coverage_warning_min", 0.001))
    warning_max = float(config.get("candidate_coverage_warning_max", 0.40))
    blocking_min = float(config.get("candidate_coverage_blocking_min", 0.000001))
    blocking_max = float(config.get("candidate_coverage_blocking_max", 0.80))
    if coverage < blocking_min or coverage > blocking_max:
        errors.append(f"{prefix} coverage is extreme: {coverage}.")
    elif coverage < warning_min or coverage > warning_max:
        warnings.append(f"{prefix} coverage is outside warning range: {coverage}.")


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _median_step(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 1.0
    diffs = np.abs(np.diff(array))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
