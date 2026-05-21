from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.alignment.depth_audit import (
    DEFAULT_MAX_DEPTH_SAMPLES,
    read_depth_axes,
    summarize_depth_axis,
)
from cement_channel.data.manifest import load_paths_config
from cement_channel.data.small_slice_reader import (
    MatReadRequest,
    load_mapping_config,
    read_mat_file_slices,
)

DEPTH_ONLY_VERSION = "depth_only_v001"
SCHEMA_VERSION = "schema_v001"
DATA_VERSION = "data_v001"


@dataclass(frozen=True)
class DepthOnlyArraySummary:
    name: str
    shape: list[int]
    dtype: str
    finite_ratio: float | None
    min: float | None
    max: float | None
    mean: float | None
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthOnlyReadResult:
    depth_only_version: str
    schema_version: str
    data_version: str
    mapping_path: str
    created_at: str
    source_files: dict[str, Any]
    arrays: dict[str, DepthOnlyArraySummary]
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_only_version": self.depth_only_version,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "mapping_path": self.mapping_path,
            "created_at": self.created_at,
            "source_files": self.source_files,
            "arrays": {key: value.to_dict() for key, value in self.arrays.items()},
            "warnings": self.warnings,
            "errors": self.errors,
            "not_performed": self.not_performed,
        }


def read_depth_only_from_configs(
    paths_config: Path | str,
    mapping_path: Path | str,
    *,
    max_depth_samples: int = DEFAULT_MAX_DEPTH_SAMPLES,
) -> tuple[DepthOnlyReadResult, dict[str, np.ndarray]]:
    config = load_paths_config(paths_config)
    mapping = load_mapping_config(mapping_path)
    return read_depth_only(
        config,
        mapping,
        mapping_path=mapping_path,
        max_depth_samples=max_depth_samples,
    )


def read_depth_only(
    paths_config: dict[str, Any],
    mapping: dict[str, Any],
    *,
    mapping_path: Path | str,
    max_depth_samples: int = DEFAULT_MAX_DEPTH_SAMPLES,
) -> tuple[DepthOnlyReadResult, dict[str, np.ndarray]]:
    arrays = read_depth_axes(paths_config, mapping, max_depth_samples=max_depth_samples)
    pose_arrays = _read_pose_arrays(paths_config, mapping, max_depth_samples=max_depth_samples)
    arrays["pose_depth"] = pose_arrays["pose_depth"]
    arrays["inc_deg"] = pose_arrays["inc_deg"]
    arrays["relbearing_deg"] = pose_arrays["relbearing_deg"]

    warnings: list[str] = []
    errors: list[str] = []
    summaries = {key: summarize_depth_only_array(key, value) for key, value in arrays.items()}
    for key, summary in summaries.items():
        warnings.extend(f"{key}: {message}" for message in summary.warnings)
        errors.extend(f"{key}: {message}" for message in summary.errors)
    warnings.extend(_depth_axis_warnings(arrays))
    warnings.extend(_pose_range_warnings(arrays["inc_deg"], arrays["relbearing_deg"]))
    errors.extend(_pose_length_errors(arrays))

    result = DepthOnlyReadResult(
        depth_only_version=DEPTH_ONLY_VERSION,
        schema_version=SCHEMA_VERSION,
        data_version=DATA_VERSION,
        mapping_path=str(mapping_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        source_files=_source_files(paths_config, mapping),
        arrays=summaries,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "XSI waveform reading",
            "CAST Zc reading",
            "interpolation",
            "alignment",
            "label generation",
            "feature extraction",
            "model training",
        ],
    )
    return result, arrays


def summarize_depth_only_array(name: str, array: np.ndarray) -> DepthOnlyArraySummary:
    values = np.asarray(array)
    warnings: list[str] = []
    errors: list[str] = []
    if values.size == 0:
        errors.append(f"{name} is empty.")
        return DepthOnlyArraySummary(
            name=name,
            shape=[int(item) for item in values.shape],
            dtype=str(values.dtype),
            finite_ratio=None,
            min=None,
            max=None,
            mean=None,
            warnings=warnings,
            errors=errors,
        )
    finite = np.isfinite(values)
    finite_ratio = float(np.mean(finite))
    if finite_ratio < 1.0:
        warnings.append(f"{name} contains non-finite values.")
    finite_values = values[finite]
    if finite_values.size == 0:
        errors.append(f"{name} has no finite values.")
        return DepthOnlyArraySummary(
            name=name,
            shape=[int(item) for item in values.shape],
            dtype=str(values.dtype),
            finite_ratio=finite_ratio,
            min=None,
            max=None,
            mean=None,
            warnings=warnings,
            errors=errors,
        )
    return DepthOnlyArraySummary(
        name=name,
        shape=[int(item) for item in values.shape],
        dtype=str(values.dtype),
        finite_ratio=finite_ratio,
        min=float(np.min(finite_values)),
        max=float(np.max(finite_values)),
        mean=float(np.mean(finite_values)),
        warnings=warnings,
        errors=errors,
    )


def write_depth_only_outputs(
    result: DepthOnlyReadResult,
    arrays: dict[str, np.ndarray],
    *,
    output_npz: Path,
    output_summary_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    _ensure_can_write(output_summary_json, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_summary_json.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_pose_arrays(
    paths_config: dict[str, Any],
    mapping: dict[str, Any],
    *,
    max_depth_samples: int,
) -> dict[str, np.ndarray]:
    data_config = _as_dict(paths_config.get("data"))
    raw_dir = Path(str(data_config.get("raw", "")))
    pose = _as_dict(mapping.get("pose"))
    pose_path = raw_dir / str(pose.get("file", "D2_XSI_RelBearing_Inclination.mat"))
    orientation = _as_str_list(pose.get("source_shape_order"), ["depth"])
    requests = [
        MatReadRequest(
            variable_path=str(pose.get("depth_variable", "")),
            role="depth",
            source_orientation=orientation,
            canonical_orientation=["depth"],
            max_depth_samples=max_depth_samples,
            max_time_samples=1,
            max_cast_azimuth=1,
        ),
        MatReadRequest(
            variable_path=str(pose.get("inclination_variable", "")),
            role="depth",
            source_orientation=orientation,
            canonical_orientation=["depth"],
            max_depth_samples=max_depth_samples,
            max_time_samples=1,
            max_cast_azimuth=1,
        ),
        MatReadRequest(
            variable_path=str(pose.get("relbearing_variable", "")),
            role="depth",
            source_orientation=orientation,
            canonical_orientation=["depth"],
            max_depth_samples=max_depth_samples,
            max_time_samples=1,
            max_cast_azimuth=1,
        ),
    ]
    data = read_mat_file_slices(pose_path, requests)
    return {
        "pose_depth": np.asarray(data[requests[0].variable_path], dtype=np.float32).reshape(-1),
        "inc_deg": np.asarray(data[requests[1].variable_path], dtype=np.float32).reshape(-1),
        "relbearing_deg": np.asarray(data[requests[2].variable_path], dtype=np.float32).reshape(-1),
    }


def _depth_axis_warnings(arrays: dict[str, np.ndarray]) -> list[str]:
    warnings: list[str] = []
    depth_items = {
        "cast_depth": arrays["cast_depth"],
        "pose_depth": arrays["pose_depth"],
    }
    xsi = np.asarray(arrays["xsi_depth_by_receiver"])
    for index in range(xsi.shape[0]):
        depth_items[f"xsi_depth_receiver_{index + 1:02d}"] = xsi[index]
    for name, values in depth_items.items():
        stats = summarize_depth_axis(name, values)
        warnings.extend(stats.warnings)
        warnings.extend(stats.errors)
    return warnings


def _pose_range_warnings(inc_deg: np.ndarray, relbearing_deg: np.ndarray) -> list[str]:
    warnings: list[str] = []
    finite_inc = np.asarray(inc_deg)[np.isfinite(inc_deg)]
    finite_rel = np.asarray(relbearing_deg)[np.isfinite(relbearing_deg)]
    if finite_inc.size and (np.min(finite_inc) < 0.0 or np.max(finite_inc) > 180.0):
        warnings.append("Inc values fall outside the broad [0, 180] degree sanity range.")
    if finite_rel.size and (np.min(finite_rel) < 0.0 or np.max(finite_rel) >= 360.0):
        warnings.append(
            "RelBearing values fall outside [0, 360); wrap convention must be verified."
        )
    return warnings


def _pose_length_errors(arrays: dict[str, np.ndarray]) -> list[str]:
    lengths = {
        "pose_depth": int(np.asarray(arrays["pose_depth"]).size),
        "inc_deg": int(np.asarray(arrays["inc_deg"]).size),
        "relbearing_deg": int(np.asarray(arrays["relbearing_deg"]).size),
    }
    if len(set(lengths.values())) == 1:
        return []
    return [f"Pose arrays have mismatched lengths: {lengths}"]


def _source_files(paths_config: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    data_config = _as_dict(paths_config.get("data"))
    raw_dir = Path(str(data_config.get("raw", "")))
    cast = _as_dict(mapping.get("cast"))
    pose = _as_dict(mapping.get("pose"))
    xsi = _as_dict(mapping.get("xsi"))
    receiver_dir = raw_dir / str(xsi.get("receiver_dir", "XSILMR"))
    receiver_count = int(xsi.get("expected_receiver_files", 13))
    return {
        "cast": str(raw_dir / str(cast.get("file", "CAST.mat"))),
        "pose": str(raw_dir / str(pose.get("file", "D2_XSI_RelBearing_Inclination.mat"))),
        "xsi_receiver_files": [
            str(receiver_dir / f"XSILMR{receiver_index:02d}.mat")
            for receiver_index in range(1, receiver_count + 1)
        ],
    }


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(default or [])
