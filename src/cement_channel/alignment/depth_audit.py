from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.data.manifest import load_paths_config
from cement_channel.data.small_slice_reader import (
    MatReadRequest,
    load_mapping_config,
    read_mat_file_slices,
)

DEPTH_AXIS_AUDIT_VERSION = "depth_axis_audit_v001"
DEFAULT_MAX_DEPTH_SAMPLES = 5_000_000


@dataclass(frozen=True)
class DepthAxisStats:
    name: str
    length: int
    finite_count: int
    nan_count: int
    inf_count: int
    duplicate_count: int
    min: float | None
    max: float | None
    direction: str
    monotonic: bool
    median_step: float | None
    min_step: float | None
    max_step: float | None
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverDepthConsistency:
    receiver_count: int
    expected_receiver_count: int | None
    common_length: int
    max_start_delta: float | None
    max_stop_delta: float | None
    max_median_step_delta: float | None
    max_pairwise_delta_on_common_prefix: float | None
    median_pairwise_delta_on_common_prefix: float | None
    consistent: bool
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthAxisAuditResult:
    audit_version: str
    generated_at: str
    depth_unit: str
    cast_depth: DepthAxisStats
    xsi_depth_by_receiver: dict[str, DepthAxisStats]
    pose_depth: DepthAxisStats
    receiver_consistency: ReceiverDepthConsistency
    common_overlap_interval: dict[str, float | None]
    candidate_canonical_depth_grid: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    no_go_blockers: list[str]
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": self.audit_version,
            "generated_at": self.generated_at,
            "depth_unit": self.depth_unit,
            "cast_depth": self.cast_depth.to_dict(),
            "xsi_depth_by_receiver": {
                key: value.to_dict() for key, value in self.xsi_depth_by_receiver.items()
            },
            "pose_depth": self.pose_depth.to_dict(),
            "receiver_consistency": self.receiver_consistency.to_dict(),
            "common_overlap_interval": self.common_overlap_interval,
            "candidate_canonical_depth_grid": self.candidate_canonical_depth_grid,
            "warnings": self.warnings,
            "errors": self.errors,
            "no_go_blockers": self.no_go_blockers,
            "decision": self.decision,
        }


def read_depth_axes_from_configs(
    paths_config: Path | str,
    mapping_path: Path | str,
    *,
    max_depth_samples: int = DEFAULT_MAX_DEPTH_SAMPLES,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    config = load_paths_config(paths_config)
    mapping = load_mapping_config(mapping_path)
    arrays = read_depth_axes(config, mapping, max_depth_samples=max_depth_samples)
    return arrays, config, mapping


def read_depth_axes(
    paths_config: dict[str, Any],
    mapping: dict[str, Any],
    *,
    max_depth_samples: int = DEFAULT_MAX_DEPTH_SAMPLES,
) -> dict[str, np.ndarray]:
    data_config = _as_dict(paths_config.get("data"))
    raw_dir = Path(str(data_config.get("raw", "")))
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    cast = _as_dict(mapping.get("cast"))
    pose = _as_dict(mapping.get("pose"))
    xsi = _as_dict(mapping.get("xsi"))
    arrays: dict[str, np.ndarray] = {}

    cast_path = raw_dir / str(cast.get("file", "CAST.mat"))
    cast_request = MatReadRequest(
        variable_path=str(cast.get("depth_variable", "")),
        role="depth",
        source_orientation=_as_str_list(cast.get("depth_source_shape_order"), ["depth"]),
        canonical_orientation=["depth"],
        max_depth_samples=max_depth_samples,
        max_time_samples=1,
        max_cast_azimuth=1,
    )
    arrays["cast_depth"] = _read_single_depth(cast_path, cast_request)

    pose_path = raw_dir / str(pose.get("file", "D2_XSI_RelBearing_Inclination.mat"))
    pose_request = MatReadRequest(
        variable_path=str(pose.get("depth_variable", "")),
        role="depth",
        source_orientation=_as_str_list(pose.get("source_shape_order"), ["depth"]),
        canonical_orientation=["depth"],
        max_depth_samples=max_depth_samples,
        max_time_samples=1,
        max_cast_azimuth=1,
    )
    arrays["pose_depth"] = _read_single_depth(pose_path, pose_request)

    receiver_dir = raw_dir / str(xsi.get("receiver_dir", "XSILMR"))
    expected_receiver_files = int(xsi.get("expected_receiver_files", 13))
    receiver_depths: list[np.ndarray] = []
    for receiver_index in range(1, expected_receiver_files + 1):
        receiver_file = receiver_dir / f"XSILMR{receiver_index:02d}.mat"
        request = MatReadRequest(
            variable_path=_format_pattern(
                str(xsi.get("depth_variable_pattern", "")),
                receiver_index,
            ),
            role="depth",
            source_orientation=_as_str_list(xsi.get("depth_source_shape_order"), ["depth"]),
            canonical_orientation=["depth"],
            max_depth_samples=max_depth_samples,
            max_time_samples=1,
            max_cast_azimuth=1,
        )
        receiver_depths.append(_read_single_depth(receiver_file, request))
    arrays["xsi_depth_by_receiver"] = np.stack(receiver_depths, axis=0).astype(np.float32)
    return arrays


def audit_depth_axes(
    *,
    cast_depth: np.ndarray,
    xsi_depth_by_receiver: np.ndarray,
    pose_depth: np.ndarray,
    expected_receiver_count: int | None = None,
    depth_unit: str = "unknown_to_verify",
) -> DepthAxisAuditResult:
    cast_stats = summarize_depth_axis("cast_depth", cast_depth)
    pose_stats = summarize_depth_axis("pose_depth", pose_depth)
    xsi_values = np.asarray(xsi_depth_by_receiver)
    if xsi_values.ndim == 1:
        xsi_values = xsi_values.reshape(1, -1)
    xsi_stats = {
        f"receiver_{index + 1:02d}": summarize_depth_axis(
            f"xsi_depth_receiver_{index + 1:02d}",
            xsi_values[index],
        )
        for index in range(xsi_values.shape[0])
    }
    receiver_consistency = summarize_receiver_depth_consistency(
        xsi_values,
        expected_receiver_count=expected_receiver_count,
    )
    overlap = common_overlap_interval([cast_depth, pose_depth, *list(xsi_values)])
    candidate_grid = propose_candidate_canonical_grid(
        [cast_stats, pose_stats, *list(xsi_stats.values())],
        overlap,
    )
    warnings, errors, blockers = _collect_audit_messages(
        cast_stats=cast_stats,
        pose_stats=pose_stats,
        xsi_stats=xsi_stats,
        receiver_consistency=receiver_consistency,
        overlap=overlap,
        depth_unit=depth_unit,
    )
    decision = _decision(blockers, warnings)
    return DepthAxisAuditResult(
        audit_version=DEPTH_AXIS_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        depth_unit=depth_unit,
        cast_depth=cast_stats,
        xsi_depth_by_receiver=xsi_stats,
        pose_depth=pose_stats,
        receiver_consistency=receiver_consistency,
        common_overlap_interval=overlap,
        candidate_canonical_depth_grid=candidate_grid,
        warnings=warnings,
        errors=errors,
        no_go_blockers=blockers,
        decision=decision,
    )


def summarize_depth_axis(name: str, depth: np.ndarray) -> DepthAxisStats:
    values = np.asarray(depth, dtype=np.float64).reshape(-1)
    warnings: list[str] = []
    errors: list[str] = []
    length = int(values.size)
    if length == 0:
        errors.append(f"{name} is empty.")
        return DepthAxisStats(
            name=name,
            length=0,
            finite_count=0,
            nan_count=0,
            inf_count=0,
            duplicate_count=0,
            min=None,
            max=None,
            direction="empty",
            monotonic=False,
            median_step=None,
            min_step=None,
            max_step=None,
            warnings=warnings,
            errors=errors,
        )

    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    finite_count = int(finite_values.size)
    if finite_count == 0:
        errors.append(f"{name} has no finite depth values.")
        return DepthAxisStats(
            name=name,
            length=length,
            finite_count=0,
            nan_count=nan_count,
            inf_count=inf_count,
            duplicate_count=0,
            min=None,
            max=None,
            direction="non_finite",
            monotonic=False,
            median_step=None,
            min_step=None,
            max_step=None,
            warnings=warnings,
            errors=errors,
        )

    if finite_count != length:
        warnings.append(f"{name} contains {length - finite_count} non-finite values.")
    diffs = np.diff(finite_values)
    duplicate_count = int(np.sum(diffs == 0.0))
    positive_count = int(np.sum(diffs > 0.0))
    negative_count = int(np.sum(diffs < 0.0))
    zero_count = int(np.sum(diffs == 0.0))
    positive = bool(diffs.size == 0 or positive_count == diffs.size)
    nondecreasing = bool(diffs.size == 0 or negative_count == 0)
    negative = bool(diffs.size > 0 and negative_count == diffs.size)
    nonincreasing = bool(diffs.size > 0 and positive_count == 0)
    mostly_positive = _is_mostly_one_direction(positive_count, negative_count + zero_count)
    mostly_negative = _is_mostly_one_direction(negative_count, positive_count + zero_count)
    if positive:
        direction = "increasing"
        monotonic = True
    elif nondecreasing:
        direction = "nondecreasing"
        monotonic = True
        warnings.append(f"{name} contains duplicate depth samples.")
    elif negative:
        direction = "decreasing"
        monotonic = True
        warnings.append(f"{name} is decreasing in raw order; reverse before interpolation.")
    elif nonincreasing:
        direction = "nonincreasing"
        monotonic = True
        warnings.append(
            f"{name} is nonincreasing in raw order with duplicate samples; "
            "reverse and de-duplicate before interpolation."
        )
    elif mostly_positive:
        direction = "mostly_increasing"
        monotonic = False
        warnings.append(
            f"{name} is mostly increasing with {negative_count} reverse and "
            f"{zero_count} duplicate steps."
        )
    elif mostly_negative:
        direction = "mostly_decreasing"
        monotonic = False
        warnings.append(
            f"{name} is mostly decreasing with {positive_count} forward and "
            f"{zero_count} duplicate steps; reverse and de-duplicate before interpolation."
        )
    else:
        direction = "non_monotonic"
        monotonic = False
        errors.append(f"{name} is not monotonic.")

    nonzero_abs_steps = np.abs(diffs[diffs != 0.0])
    median_step = float(np.median(nonzero_abs_steps)) if nonzero_abs_steps.size else None
    min_step = float(np.min(nonzero_abs_steps)) if nonzero_abs_steps.size else None
    max_step = float(np.max(nonzero_abs_steps)) if nonzero_abs_steps.size else None
    if median_step is None:
        warnings.append(f"{name} has no positive depth step.")

    return DepthAxisStats(
        name=name,
        length=length,
        finite_count=finite_count,
        nan_count=nan_count,
        inf_count=inf_count,
        duplicate_count=duplicate_count,
        min=float(np.min(finite_values)),
        max=float(np.max(finite_values)),
        direction=direction,
        monotonic=monotonic,
        median_step=median_step,
        min_step=min_step,
        max_step=max_step,
        warnings=warnings,
        errors=errors,
    )


def summarize_receiver_depth_consistency(
    xsi_depth_by_receiver: np.ndarray,
    *,
    expected_receiver_count: int | None,
) -> ReceiverDepthConsistency:
    values = np.asarray(xsi_depth_by_receiver, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    warnings: list[str] = []
    errors: list[str] = []
    receiver_count = int(values.shape[0])
    if expected_receiver_count is not None and receiver_count != expected_receiver_count:
        errors.append(
            f"Observed {receiver_count} XSI receiver depth axes; "
            f"expected {expected_receiver_count}."
        )
    if receiver_count == 0:
        errors.append("No XSI receiver depth axes were provided.")
        return ReceiverDepthConsistency(
            0, expected_receiver_count, 0, None, None, None, None, None, False, warnings, errors
        )

    common_length = int(min(values.shape[1], *(len(row) for row in values)))
    starts = np.asarray([_finite_min(row) for row in values], dtype=np.float64)
    stops = np.asarray([_finite_max(row) for row in values], dtype=np.float64)
    steps = np.asarray(
        [summarize_depth_axis("receiver", row).median_step for row in values],
        dtype=np.float64,
    )
    max_start_delta = _finite_range(starts)
    max_stop_delta = _finite_range(stops)
    max_median_step_delta = _finite_range(steps)
    max_pairwise_delta: float | None = None
    median_pairwise_delta: float | None = None
    if common_length > 0 and receiver_count > 1:
        common = values[:, :common_length]
        reference = common[0]
        deltas = np.abs(common - reference)
        finite_deltas = deltas[np.isfinite(deltas)]
        if finite_deltas.size:
            max_pairwise_delta = float(np.max(finite_deltas))
            median_pairwise_delta = float(np.median(finite_deltas))

    median_step = float(np.nanmedian(steps)) if np.any(np.isfinite(steps)) else None
    tolerance = _receiver_tolerance(median_step)
    consistent = not errors
    if max_median_step_delta is not None and max_median_step_delta > tolerance:
        warnings.append(
            "XSI receiver median depth steps differ by "
            f"{max_median_step_delta:.6g}, tolerance {tolerance:.6g}."
        )
        consistent = False
    range_tolerance = max(tolerance * 10.0, tolerance)
    if max_start_delta is not None and max_start_delta > range_tolerance:
        warnings.append(
            "XSI receiver start depths differ by "
            f"{max_start_delta:.6g}, tolerance {range_tolerance:.6g}."
        )
    if max_stop_delta is not None and max_stop_delta > range_tolerance:
        warnings.append(
            "XSI receiver stop depths differ by "
            f"{max_stop_delta:.6g}, tolerance {range_tolerance:.6g}."
        )
    if max_pairwise_delta is not None and max_pairwise_delta > range_tolerance:
        warnings.append(
            "XSI receiver common-prefix depth values differ by "
            f"{max_pairwise_delta:.6g}, tolerance {range_tolerance:.6g}."
        )
    return ReceiverDepthConsistency(
        receiver_count=receiver_count,
        expected_receiver_count=expected_receiver_count,
        common_length=common_length,
        max_start_delta=max_start_delta,
        max_stop_delta=max_stop_delta,
        max_median_step_delta=max_median_step_delta,
        max_pairwise_delta_on_common_prefix=max_pairwise_delta,
        median_pairwise_delta_on_common_prefix=median_pairwise_delta,
        consistent=consistent,
        warnings=warnings,
        errors=errors,
    )


def common_overlap_interval(arrays: list[np.ndarray]) -> dict[str, float | None]:
    mins = [_finite_min(array) for array in arrays]
    maxes = [_finite_max(array) for array in arrays]
    finite_mins = [value for value in mins if value is not None]
    finite_maxes = [value for value in maxes if value is not None]
    if len(finite_mins) != len(arrays) or len(finite_maxes) != len(arrays):
        return {"min": None, "max": None, "length": None}
    start = max(finite_mins)
    stop = min(finite_maxes)
    return {
        "min": float(start),
        "max": float(stop),
        "length": float(stop - start) if stop >= start else float(stop - start),
    }


def propose_candidate_canonical_grid(
    stats: list[DepthAxisStats],
    overlap: dict[str, float | None],
) -> dict[str, Any]:
    start = overlap.get("min")
    stop = overlap.get("max")
    steps = [
        stat.median_step
        for stat in stats
        if stat.median_step is not None and np.isfinite(stat.median_step) and stat.median_step > 0
    ]
    if start is None or stop is None or stop <= start or not steps:
        return {
            "depth_start": None,
            "depth_stop": None,
            "depth_step": None,
            "sample_count": 0,
            "basis": "No valid common overlap or positive median step.",
        }
    step = float(max(steps))
    sample_count = int(np.floor((float(stop) - float(start)) / step)) + 1
    return {
        "depth_start": float(start),
        "depth_stop": float(start + step * (sample_count - 1)),
        "depth_step": step,
        "sample_count": sample_count,
        "basis": "Conservative Stage-1 candidate uses common overlap and coarsest median step.",
        "source_median_steps": [float(value) for value in steps],
    }


def format_depth_axis_audit_markdown(result: DepthAxisAuditResult) -> str:
    data = result.to_dict()
    lines = [
        "# Depth Axis Audit Report",
        "",
        f"- Audit version: {data['audit_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Decision: {data['decision']}",
        f"- Depth unit: {data['depth_unit']}",
        "",
        "## CAST Depth",
        "",
    ]
    lines.extend(_stats_lines(data["cast_depth"]))
    lines.extend(["", "## XSI Depth By Receiver", ""])
    for receiver, stats in data["xsi_depth_by_receiver"].items():
        lines.append(f"### {receiver}")
        lines.extend(_stats_lines(stats))
        lines.append("")
    lines.extend(["## Pose Depth", ""])
    lines.extend(_stats_lines(data["pose_depth"]))
    lines.extend(["", "## Receiver Consistency", ""])
    for key, value in data["receiver_consistency"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Common Overlap Interval", ""])
    for key, value in data["common_overlap_interval"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate Canonical Depth Grid", ""])
    for key, value in data["candidate_canonical_depth_grid"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## No-Go Blockers", ""])
    lines.extend(_message_lines(data["no_go_blockers"]))
    lines.extend(
        [
            "",
            "## Not Performed",
            "",
            "- waveform reading",
            "- full CAST Zc reading",
            "- interpolation",
            "- label generation",
            "- feature extraction",
            "- model training",
            "",
        ]
    )
    return "\n".join(lines)


def write_depth_axis_audit_outputs(
    result: DepthAxisAuditResult,
    *,
    output_json: Path,
    output_md: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_md, overwrite=overwrite)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_depth_axis_audit_markdown(result), encoding="utf-8")


def _read_single_depth(path: Path, request: MatReadRequest) -> np.ndarray:
    data = read_mat_file_slices(path, [request])
    return np.asarray(data[request.variable_path], dtype=np.float32).reshape(-1)


def _collect_audit_messages(
    *,
    cast_stats: DepthAxisStats,
    pose_stats: DepthAxisStats,
    xsi_stats: dict[str, DepthAxisStats],
    receiver_consistency: ReceiverDepthConsistency,
    overlap: dict[str, float | None],
    depth_unit: str,
) -> tuple[list[str], list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    blockers: list[str] = []
    all_stats = [cast_stats, pose_stats, *list(xsi_stats.values())]
    for stats in all_stats:
        warnings.extend(f"{stats.name}: {message}" for message in stats.warnings)
        errors.extend(f"{stats.name}: {message}" for message in stats.errors)
        if stats.errors:
            blockers.extend(f"{stats.name}: {message}" for message in stats.errors)
        if stats.nan_count + stats.inf_count > max(1, int(stats.length * 0.01)):
            blockers.append(f"{stats.name}: severe non-finite depth values.")
    warnings.extend(f"receiver_consistency: {message}" for message in receiver_consistency.warnings)
    errors.extend(f"receiver_consistency: {message}" for message in receiver_consistency.errors)
    blockers.extend(f"receiver_consistency: {message}" for message in receiver_consistency.errors)
    if not receiver_consistency.consistent and receiver_consistency.errors:
        blockers.append("XSI receiver depth axes are not usable as a receiver set.")
    if overlap.get("min") is None or overlap.get("max") is None or overlap.get("length") is None:
        blockers.append("No finite common overlap interval could be computed.")
    elif float(overlap["length"]) <= 0.0:
        blockers.append("CAST, XSI, and pose depth axes have no positive common overlap.")
    if str(depth_unit).lower().startswith("unknown"):
        warnings.append("depth unit is unknown_to_verify; conditional_go requires human review.")
    return warnings, errors, blockers


def _decision(blockers: list[str], warnings: list[str]) -> str:
    if blockers:
        return "no_go"
    if warnings:
        return "conditional_go"
    return "go"


def _stats_lines(stats: dict[str, Any]) -> list[str]:
    keys = [
        "length",
        "min",
        "max",
        "direction",
        "monotonic",
        "median_step",
        "nan_count",
        "inf_count",
        "duplicate_count",
    ]
    return [f"- {key}: {stats[key]}" for key in keys]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _receiver_tolerance(median_step: float | None) -> float:
    if median_step is None or not np.isfinite(median_step) or median_step <= 0.0:
        return 1.0e-3
    return max(float(median_step) * 0.5, 1.0e-3)


def _is_mostly_one_direction(main_count: int, violation_count: int) -> bool:
    total = main_count + violation_count
    if total == 0:
        return False
    allowed = 0 if total < 100 else max(25, int(total * 0.005))
    return main_count > 0 and violation_count <= allowed


def _finite_min(array: np.ndarray) -> float | None:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else None


def _finite_max(array: np.ndarray) -> float | None:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def _finite_range(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(finite) - np.min(finite))


def _format_pattern(pattern: str, receiver: int) -> str:
    return pattern.format(receiver=receiver)


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(default or [])
