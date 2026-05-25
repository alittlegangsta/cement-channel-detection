from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.alignment.azimuth_normalization import (
    orientation_confidence_from_inclination,
    orientation_uncertain_mask,
)

ORIENTATION_CONFIDENCE_VERSION = "orientation_confidence_v001"
DEFAULT_I_MIN_DEG = 1.0
DEFAULT_I_STABLE_DEG = 5.0


@dataclass(frozen=True)
class OrientationConfidenceStats:
    count: int
    finite_count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrientationConfidenceReport:
    orientation_confidence_version: str
    generated_at: str
    inputs: dict[str, str]
    thresholds: dict[str, float | str]
    inc_stats: OrientationConfidenceStats
    confidence_stats: OrientationConfidenceStats
    low_inclination_ratio: float | None
    stable_inclination_ratio: float | None
    confidence_distribution: dict[str, int]
    arrays: dict[str, dict[str, Any]]
    relbearing_sign_dependency: str
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "orientation_confidence_version": self.orientation_confidence_version,
            "generated_at": self.generated_at,
            "inputs": self.inputs,
            "thresholds": self.thresholds,
            "inc_stats": self.inc_stats.to_dict(),
            "confidence_stats": self.confidence_stats.to_dict(),
            "low_inclination_ratio": self.low_inclination_ratio,
            "stable_inclination_ratio": self.stable_inclination_ratio,
            "confidence_distribution": self.confidence_distribution,
            "arrays": self.arrays,
            "relbearing_sign_dependency": self.relbearing_sign_dependency,
            "warnings": self.warnings,
            "errors": self.errors,
            "not_performed": self.not_performed,
        }


def build_orientation_confidence(
    *,
    depth_only_npz: Path | str,
    i_min_deg: float = DEFAULT_I_MIN_DEG,
    i_stable_deg: float = DEFAULT_I_STABLE_DEG,
) -> tuple[OrientationConfidenceReport, dict[str, np.ndarray]]:
    if i_stable_deg <= i_min_deg:
        raise ValueError("i_stable_deg must be greater than i_min_deg.")

    with np.load(depth_only_npz) as data:
        arrays = {key: data[key] for key in data.files}

    pose_depth = _require_array(arrays, "pose_depth").reshape(-1).astype(np.float32)
    inc_deg = _require_array(arrays, "inc_deg").reshape(-1).astype(np.float32)
    confidence = np.asarray(
        orientation_confidence_from_inclination(
            inc_deg,
            i_min_deg=i_min_deg,
            i_stable_deg=i_stable_deg,
        ),
        dtype=np.float32,
    )
    low_inc_mask = np.asarray(
        orientation_uncertain_mask(inc_deg, i_min_deg=i_min_deg),
        dtype=bool,
    )
    stable_inc_mask = (np.isfinite(inc_deg)) & (inc_deg >= float(i_stable_deg))
    orientation_uncertain = low_inc_mask | (~np.isfinite(inc_deg))

    output_arrays = {
        "pose_depth": pose_depth,
        "inc_deg": inc_deg,
        "orientation_confidence": confidence,
        "low_inc_mask": low_inc_mask,
        "stable_inc_mask": stable_inc_mask,
        "orientation_uncertain": orientation_uncertain,
    }

    warnings: list[str] = []
    errors: list[str] = []
    if pose_depth.size != inc_deg.size:
        errors.append(
            f"pose_depth and inc_deg length mismatch: pose_depth={pose_depth.size}, "
            f"inc_deg={inc_deg.size}"
        )
    if not np.any(np.isfinite(inc_deg)):
        errors.append("inc_deg has no finite samples.")
    if np.any(~np.isfinite(inc_deg)):
        warnings.append("inc_deg contains non-finite samples; confidence set to 0 there.")

    low_ratio = _ratio(low_inc_mask, inc_deg.size)
    stable_ratio = _ratio(stable_inc_mask, inc_deg.size)
    if low_ratio is not None and low_ratio > 0.25:
        warnings.append(
            "Low-inclination sample ratio exceeds 25%; high-side orientation is uncertain "
            "over a substantial part of the preview."
        )

    report = OrientationConfidenceReport(
        orientation_confidence_version=ORIENTATION_CONFIDENCE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={"depth_only_npz": str(depth_only_npz)},
        thresholds={
            "i_min_deg": float(i_min_deg),
            "i_stable_deg": float(i_stable_deg),
            "transition": "linear",
        },
        inc_stats=_numeric_stats(inc_deg),
        confidence_stats=_numeric_stats(confidence),
        low_inclination_ratio=low_ratio,
        stable_inclination_ratio=stable_ratio,
        confidence_distribution=_confidence_distribution(confidence),
        arrays={key: _array_summary(value) for key, value in output_arrays.items()},
        relbearing_sign_dependency="independent_of_plus_minus_convention",
        warnings=warnings,
        errors=errors,
        not_performed=[
            "RelBearing sign selection",
            "RelBearing plus/minus rotation",
            "depth alignment",
            "weak label generation",
            "feature extraction",
            "model training",
        ],
    )
    return report, output_arrays


def write_orientation_confidence_outputs(
    report: OrientationConfidenceReport,
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
    output_report_md.write_text(format_orientation_confidence_markdown(report), encoding="utf-8")


def format_orientation_confidence_markdown(report: OrientationConfidenceReport) -> str:
    data = report.to_dict()
    lines = [
        "# Orientation Confidence Report",
        "",
        f"- Version: {data['orientation_confidence_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- RelBearing sign dependency: {data['relbearing_sign_dependency']}",
        f"- I_min_deg: {data['thresholds']['i_min_deg']}",
        f"- I_stable_deg: {data['thresholds']['i_stable_deg']}",
        "",
        "## Inc Statistics",
        "",
    ]
    for key, value in data["inc_stats"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Confidence Statistics", ""])
    for key, value in data["confidence_stats"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Inclination Ratios",
            "",
            f"- low_inclination_ratio: {data['low_inclination_ratio']}",
            f"- stable_inclination_ratio: {data['stable_inclination_ratio']}",
            "",
            "## Confidence Distribution",
            "",
        ]
    )
    for key, value in data["confidence_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _require_array(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"Depth-only NPZ is missing array: {key}")
    return np.asarray(arrays[key])


def _numeric_stats(values: np.ndarray) -> OrientationConfidenceStats:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return OrientationConfidenceStats(
            count=int(array.size),
            finite_count=0,
            min=None,
            max=None,
            mean=None,
            median=None,
        )
    return OrientationConfidenceStats(
        count=int(array.size),
        finite_count=int(finite.size),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
    )


def _confidence_distribution(confidence: np.ndarray) -> dict[str, int]:
    values = np.asarray(confidence, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    return {
        "eq_0": int(np.sum(finite == 0.0)),
        "gt_0_lt_0_25": int(np.sum((finite > 0.0) & (finite < 0.25))),
        "ge_0_25_lt_0_5": int(np.sum((finite >= 0.25) & (finite < 0.5))),
        "ge_0_5_lt_0_75": int(np.sum((finite >= 0.5) & (finite < 0.75))),
        "ge_0_75_lt_1": int(np.sum((finite >= 0.75) & (finite < 1.0))),
        "eq_1": int(np.sum(finite == 1.0)),
    }


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else np.ones_like(array)
    finite_ratio = float(np.mean(finite)) if array.size else None
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "finite_ratio": finite_ratio,
    }


def _ratio(mask: np.ndarray, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(np.sum(mask) / denominator)


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")
