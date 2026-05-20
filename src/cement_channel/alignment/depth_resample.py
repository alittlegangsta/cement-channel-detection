from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEPTH_RESAMPLE_PREVIEW_VERSION = "depth_resample_preview_v001"


@dataclass(frozen=True)
class ResampledArraySummary:
    name: str
    shape: list[int]
    finite_ratio: float | None
    nan_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthResamplePreviewReport:
    preview_version: str
    generated_at: str
    inputs: dict[str, str | None]
    canonical_grid: dict[str, Any]
    arrays: dict[str, ResampledArraySummary]
    small_slice: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview_version": self.preview_version,
            "generated_at": self.generated_at,
            "inputs": self.inputs,
            "canonical_grid": self.canonical_grid,
            "arrays": {key: value.to_dict() for key, value in self.arrays.items()},
            "small_slice": self.small_slice,
            "warnings": self.warnings,
            "errors": self.errors,
            "not_performed": self.not_performed,
        }


def load_depth_grid_proposal(path: Path | str) -> dict[str, Any]:
    proposal_path = Path(path)
    data = json.loads(proposal_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Depth grid proposal must contain an object: {proposal_path}")
    return data


def build_depth_resample_preview(
    *,
    depth_only_npz: Path | str,
    depth_grid_proposal_json: Path | str,
    small_slice_npz: Path | str | None = None,
    max_preview_depth_samples: int = 16,
) -> tuple[DepthResamplePreviewReport, dict[str, np.ndarray]]:
    proposal = load_depth_grid_proposal(depth_grid_proposal_json)
    canonical_depth = canonical_depth_from_proposal(proposal)
    arrays: dict[str, np.ndarray] = {"canonical_depth": canonical_depth.astype(np.float32)}
    warnings = [str(item) for item in proposal.get("warnings", []) if item]
    errors = [str(item) for item in proposal.get("errors", []) if item]
    blockers = [str(item) for item in proposal.get("no_go_blockers", []) if item]
    if blockers or proposal.get("decision") == "no_go":
        errors.extend(["Depth grid proposal is no_go."] + blockers)

    with np.load(depth_only_npz) as data:
        depth_only = {key: data[key] for key in data.files}
    _require_depth_only_arrays(depth_only)

    arrays.update(_resample_depth_only_arrays(depth_only, canonical_depth))
    small_slice_path = Path(small_slice_npz) if small_slice_npz is not None else None
    small_slice_status = _resample_small_slice(
        small_slice_path,
        canonical_depth,
        arrays,
        warnings,
        max_preview_depth_samples=max_preview_depth_samples,
    )
    summaries = {key: summarize_resampled_array(key, value) for key, value in arrays.items()}
    return (
        DepthResamplePreviewReport(
            preview_version=DEPTH_RESAMPLE_PREVIEW_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            inputs={
                "depth_only_npz": str(depth_only_npz),
                "depth_grid_proposal_json": str(depth_grid_proposal_json),
                "small_slice_npz": str(small_slice_path) if small_slice_path else None,
            },
            canonical_grid={
                "depth_start": _as_float_or_none(proposal.get("depth_start")),
                "depth_stop": _as_float_or_none(proposal.get("depth_stop")),
                "depth_step": _as_float_or_none(proposal.get("depth_step")),
                "sample_count": int(canonical_depth.size),
                "allow_extrapolation": False,
            },
            arrays=summaries,
            small_slice=small_slice_status,
            warnings=warnings,
            errors=errors,
            not_performed=[
                "full waveform reading",
                "full CAST Zc reading",
                "formal alignment HDF5 writing",
                "label generation",
                "feature extraction",
                "STC/APES",
                "model training",
            ],
        ),
        arrays,
    )


def canonical_depth_from_proposal(proposal: dict[str, Any]) -> np.ndarray:
    start = _as_float_or_none(proposal.get("depth_start"))
    step = _as_float_or_none(proposal.get("depth_step"))
    sample_count = int(proposal.get("sample_count", 0) or 0)
    if start is None or step is None or step <= 0.0 or sample_count < 1:
        raise ValueError("Depth grid proposal does not define a valid canonical grid.")
    return start + step * np.arange(sample_count, dtype=np.float64)


def interpolate_1d(
    source_depth: np.ndarray,
    values: np.ndarray,
    target_depth: np.ndarray,
    *,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    source, source_values = prepare_depth_series(source_depth, values)
    target = np.asarray(target_depth, dtype=np.float64).reshape(-1)
    output_shape = (target.size, *source_values.shape[1:])
    if source.size < 2:
        return np.full(output_shape, np.nan, dtype=np.float32)
    flat_values = source_values.reshape(source_values.shape[0], -1)
    flat_output = np.empty((target.size, flat_values.shape[1]), dtype=np.float32)
    for column in range(flat_values.shape[1]):
        flat_output[:, column] = np.interp(target, source, flat_values[:, column]).astype(
            np.float32
        )
    if not allow_extrapolation:
        outside = (target < source[0]) | (target > source[-1])
        flat_output[outside, :] = np.nan
    return flat_output.reshape(output_shape)


def interpolate_angle_deg(
    source_depth: np.ndarray,
    angle_deg: np.ndarray,
    target_depth: np.ndarray,
    *,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    source, source_values = prepare_depth_series(source_depth, angle_deg)
    if source.size < 2:
        return np.full(np.asarray(target_depth).reshape(-1).shape, np.nan, dtype=np.float32)
    radians = np.unwrap(np.deg2rad(source_values.reshape(-1).astype(np.float64)))
    interpolated = interpolate_1d(
        source,
        radians,
        target_depth,
        allow_extrapolation=allow_extrapolation,
    ).reshape(-1)
    return np.mod(np.rad2deg(interpolated), 360.0).astype(np.float32)


def prepare_depth_series(
    source_depth: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(source_depth, dtype=np.float64).reshape(-1)
    value_array = np.asarray(values)
    if value_array.shape[0] != depth.size:
        raise ValueError(
            "Values must have depth as the first dimension: "
            f"depth={depth.size}, values={value_array.shape}"
        )
    finite_depth = np.isfinite(depth)
    depth = depth[finite_depth]
    value_array = value_array[finite_depth]
    order = np.argsort(depth, kind="mergesort")
    sorted_depth = depth[order]
    sorted_values = value_array[order]
    unique_depth, unique_indices = np.unique(sorted_depth, return_index=True)
    return unique_depth.astype(np.float64), sorted_values[unique_indices].astype(np.float32)


def summarize_resampled_array(name: str, array: np.ndarray) -> ResampledArraySummary:
    values = np.asarray(array)
    if values.size == 0:
        return ResampledArraySummary(
            name=name, shape=[int(item) for item in values.shape], finite_ratio=None, nan_count=0
        )
    finite = np.isfinite(values)
    return ResampledArraySummary(
        name=name,
        shape=[int(item) for item in values.shape],
        finite_ratio=float(np.mean(finite)),
        nan_count=int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
    )


def write_depth_resample_preview_outputs(
    report: DepthResamplePreviewReport,
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
    np.savez_compressed(output_npz, **arrays)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_depth_resample_preview_markdown(report), encoding="utf-8")


def format_depth_resample_preview_markdown(report: DepthResamplePreviewReport) -> str:
    data = report.to_dict()
    lines = [
        "# Depth Resample Preview Report",
        "",
        f"- Preview version: {data['preview_version']}",
        f"- Generated at: {data['generated_at']}",
        "",
        "## Canonical Grid",
        "",
    ]
    for key, value in data["canonical_grid"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Arrays", ""])
    for key, summary in data["arrays"].items():
        lines.append(
            f"- {key}: shape={summary['shape']}, "
            f"finite_ratio={summary['finite_ratio']}, nan_count={summary['nan_count']}"
        )
    lines.extend(["", "## Small Slice", ""])
    for key, value in data["small_slice"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _resample_depth_only_arrays(
    depth_only: dict[str, np.ndarray],
    canonical_depth: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    arrays["cast_source_index_on_grid"] = interpolate_1d(
        depth_only["cast_depth"],
        np.arange(depth_only["cast_depth"].size, dtype=np.float32),
        canonical_depth,
    )
    arrays["pose_source_index_on_grid"] = interpolate_1d(
        depth_only["pose_depth"],
        np.arange(depth_only["pose_depth"].size, dtype=np.float32),
        canonical_depth,
    )
    arrays["inc_deg_on_grid"] = interpolate_1d(
        depth_only["pose_depth"],
        depth_only["inc_deg"],
        canonical_depth,
    )
    arrays["relbearing_deg_on_grid"] = interpolate_angle_deg(
        depth_only["pose_depth"],
        depth_only["relbearing_deg"],
        canonical_depth,
    )
    xsi_depth = np.asarray(depth_only["xsi_depth_by_receiver"])
    source_indices = []
    valid_masks = []
    for receiver_index in range(xsi_depth.shape[0]):
        source_indices.append(
            interpolate_1d(
                xsi_depth[receiver_index],
                np.arange(xsi_depth.shape[1], dtype=np.float32),
                canonical_depth,
            )
        )
        valid_masks.append(_coverage_mask(xsi_depth[receiver_index], canonical_depth))
    arrays["xsi_source_index_on_grid"] = np.stack(source_indices, axis=0).astype(np.float32)
    arrays["cast_valid_mask"] = _coverage_mask(depth_only["cast_depth"], canonical_depth)
    arrays["pose_valid_mask"] = _coverage_mask(depth_only["pose_depth"], canonical_depth)
    arrays["xsi_valid_mask_by_receiver"] = np.stack(valid_masks, axis=0)
    return arrays


def _resample_small_slice(
    small_slice_path: Path | None,
    canonical_depth: np.ndarray,
    arrays: dict[str, np.ndarray],
    warnings: list[str],
    *,
    max_preview_depth_samples: int,
) -> dict[str, Any]:
    arrays["small_slice_preview_depth"] = np.empty((0,), dtype=np.float32)
    arrays["small_slice_cast_zc_on_preview"] = np.empty((0, 0), dtype=np.float32)
    arrays["small_slice_xsi_waveform_on_preview"] = np.empty((0, 0, 0, 0), dtype=np.float32)
    if small_slice_path is None or not small_slice_path.exists():
        warnings.append("small_slice_v001.npz is missing; skipped small-slice array preview.")
        return {"status": "skipped_missing", "warnings": ["small slice NPZ missing"]}

    with np.load(small_slice_path) as data:
        small = {key: data[key] for key in data.files}
    status_warnings: list[str] = []
    preview_depth = _preview_depth_for_small_slice(
        small, canonical_depth, max_preview_depth_samples
    )
    arrays["small_slice_preview_depth"] = preview_depth.astype(np.float32)
    if preview_depth.size == 0:
        message = "small-slice depth ranges do not overlap the proposed canonical grid."
        warnings.append(message)
        status_warnings.append(message)
        return {"status": "skipped_no_common_overlap", "warnings": status_warnings}

    if "cast_zc" in small and "cast_depth" in small:
        arrays["small_slice_cast_zc_on_preview"] = interpolate_1d(
            small["cast_depth"],
            small["cast_zc"],
            preview_depth,
        )
    if "xsi_waveform" in small and "xsi_depth" in small:
        arrays["small_slice_xsi_waveform_on_preview"] = _interpolate_small_xsi_waveform(
            small["xsi_depth"],
            small["xsi_waveform"],
            preview_depth,
        )
    return {
        "status": "completed",
        "preview_depth_count": int(preview_depth.size),
        "warnings": status_warnings,
    }


def _preview_depth_for_small_slice(
    small: dict[str, np.ndarray],
    canonical_depth: np.ndarray,
    max_preview_depth_samples: int,
) -> np.ndarray:
    ranges: list[tuple[float, float]] = []
    for key in ["cast_depth", "pose_depth"]:
        if key in small:
            ranges.append(_finite_range(small[key]))
    if "xsi_depth" in small:
        xsi = np.asarray(small["xsi_depth"])
        for receiver_index in range(xsi.shape[0]):
            ranges.append(_finite_range(xsi[receiver_index]))
    finite_ranges = [item for item in ranges if np.isfinite(item[0]) and np.isfinite(item[1])]
    if not finite_ranges:
        return np.empty((0,), dtype=np.float32)
    start = max(item[0] for item in finite_ranges)
    stop = min(item[1] for item in finite_ranges)
    mask = (canonical_depth >= start) & (canonical_depth <= stop)
    candidates = canonical_depth[mask]
    return candidates[:max_preview_depth_samples]


def _interpolate_small_xsi_waveform(
    xsi_depth_by_receiver: np.ndarray,
    waveform: np.ndarray,
    preview_depth: np.ndarray,
) -> np.ndarray:
    xsi_depth = np.asarray(xsi_depth_by_receiver)
    values = np.asarray(waveform)
    receiver_arrays = []
    for receiver_index in range(values.shape[1]):
        receiver_values = values[:, receiver_index, :, :]
        receiver_arrays.append(
            interpolate_1d(xsi_depth[receiver_index], receiver_values, preview_depth)
        )
    return np.stack(receiver_arrays, axis=1).astype(np.float32)


def _coverage_mask(source_depth: np.ndarray, target_depth: np.ndarray) -> np.ndarray:
    source, _values = prepare_depth_series(
        source_depth,
        np.arange(np.asarray(source_depth).size, dtype=np.float32),
    )
    if source.size < 2:
        return np.zeros(np.asarray(target_depth).shape, dtype=bool)
    target = np.asarray(target_depth, dtype=np.float64)
    return ((target >= source[0]) & (target <= source[-1])).astype(bool)


def _finite_range(array: np.ndarray) -> tuple[float, float]:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return float(np.min(finite)), float(np.max(finite))


def _require_depth_only_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = [
        "cast_depth",
        "xsi_depth_by_receiver",
        "pose_depth",
        "inc_deg",
        "relbearing_deg",
    ]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError("Depth-only NPZ is missing arrays: " + ", ".join(missing))


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")
