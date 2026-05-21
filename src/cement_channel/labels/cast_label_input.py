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
from cement_channel.utils.angles import wrap_deg

CAST_LABEL_INPUT_VERSION = "cast_label_input_v001"
SCHEMA_VERSION = "schema_v001"
DATA_VERSION = "data_v001"
MAX_DEPTH_SAMPLES = 10_000_000
DEFAULT_CHUNK_DEPTH_SAMPLES = 2048


@dataclass(frozen=True)
class ArraySummary:
    name: str
    shape: list[int]
    dtype: str
    finite_ratio: float | None
    nan_count: int
    min: float | None
    max: float | None
    mean: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CastLabelInputReport:
    cast_label_input_version: str
    schema_version: str
    data_version: str
    generated_at: str
    inputs: dict[str, str]
    source_files: dict[str, str]
    chunking: dict[str, int | float]
    arrays: dict[str, ArraySummary]
    relbearing_interpolation: str
    cast_azimuth: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cast_label_input_version": self.cast_label_input_version,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "generated_at": self.generated_at,
            "inputs": self.inputs,
            "source_files": self.source_files,
            "chunking": self.chunking,
            "arrays": {key: value.to_dict() for key, value in self.arrays.items()},
            "relbearing_interpolation": self.relbearing_interpolation,
            "cast_azimuth": self.cast_azimuth,
            "warnings": self.warnings,
            "errors": self.errors,
            "not_performed": self.not_performed,
        }


def prepare_cast_label_input_from_configs(
    *,
    paths_config: Path | str,
    mapping_path: Path | str,
    label_config_path: Path | str,
    orientation_confidence_npz: Path | str | None = None,
    chunk_depth_samples: int = DEFAULT_CHUNK_DEPTH_SAMPLES,
) -> tuple[CastLabelInputReport, dict[str, np.ndarray]]:
    paths = load_paths_config(paths_config)
    mapping = load_mapping_config(mapping_path)
    label_config = load_label_config(label_config_path)
    resolved_orientation = (
        Path(orientation_confidence_npz)
        if orientation_confidence_npz is not None
        else _default_interim_path(paths, "orientation_confidence_v001.npz")
    )
    return prepare_cast_label_input(
        paths,
        mapping,
        label_config,
        paths_config_path=Path(paths_config),
        mapping_path=Path(mapping_path),
        label_config_path=Path(label_config_path),
        orientation_confidence_npz=resolved_orientation,
        chunk_depth_samples=chunk_depth_samples,
    )


def load_label_config(path: Path | str) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Label config must contain a YAML mapping: {config_path}")
    return data


def prepare_cast_label_input(
    paths: dict[str, Any],
    mapping: dict[str, Any],
    label_config: dict[str, Any],
    *,
    paths_config_path: Path | str,
    mapping_path: Path | str,
    label_config_path: Path | str,
    orientation_confidence_npz: Path | str,
    chunk_depth_samples: int = DEFAULT_CHUNK_DEPTH_SAMPLES,
) -> tuple[CastLabelInputReport, dict[str, np.ndarray]]:
    if chunk_depth_samples <= 0:
        raise ValueError("chunk_depth_samples must be positive.")

    data_config = _as_dict(paths.get("data"))
    raw_dir = Path(str(data_config.get("raw", "")))
    cast = _as_dict(mapping.get("cast"))
    pose = _as_dict(mapping.get("pose"))
    cast_path = raw_dir / str(cast.get("file", "CAST.mat"))
    pose_path = raw_dir / str(pose.get("file", "D2_XSI_RelBearing_Inclination.mat"))
    warnings: list[str] = []
    errors: list[str] = []

    cast_depth = _read_depth_array(cast_path, str(cast.get("depth_variable", "")))
    cast_azimuth = _cast_azimuth_axis(mapping, label_config)
    zc = _read_cast_zc_chunks(
        cast_path,
        variable_path=str(cast.get("zc_variable", "")),
        depth_count=cast_depth.size,
        azimuth_count=cast_azimuth.size,
        chunk_depth_samples=chunk_depth_samples,
    )
    pose_depth = _read_depth_array(pose_path, str(pose.get("depth_variable", "")))
    pose_inc = _read_depth_array(pose_path, str(pose.get("inclination_variable", "")))
    pose_relbearing = _read_depth_array(pose_path, str(pose.get("relbearing_variable", "")))

    inc_interp = _interp_linear(pose_depth, pose_inc, cast_depth).astype(np.float32)
    relbearing_interp = _interp_angle_deg(pose_depth, pose_relbearing, cast_depth).astype(
        np.float32
    )
    orientation_arrays = _load_orientation_confidence(orientation_confidence_npz)
    orientation_depth = _require_orientation_array(orientation_arrays, "pose_depth")
    orientation_confidence = _interp_linear(
        orientation_depth,
        _require_orientation_array(orientation_arrays, "orientation_confidence"),
        cast_depth,
        fill_value=0.0,
    ).astype(np.float32)
    low_inc_mask = _interp_mask(
        orientation_depth,
        _require_orientation_array(orientation_arrays, "low_inc_mask"),
        cast_depth,
    )
    orientation_uncertain = _interp_mask(
        orientation_depth,
        _require_orientation_array(orientation_arrays, "orientation_uncertain"),
        cast_depth,
    )

    arrays = {
        "cast_depth": cast_depth.astype(np.float32),
        "cast_zc": zc.astype(np.float32),
        "cast_azimuth_deg": cast_azimuth.astype(np.float32),
        "pose_depth": pose_depth.astype(np.float32),
        "pose_inc_deg": pose_inc.astype(np.float32),
        "pose_relbearing_deg": wrap_deg(pose_relbearing).astype(np.float32),
        "inc_deg": inc_interp,
        "relbearing_deg": relbearing_interp,
        "orientation_confidence": np.clip(orientation_confidence, 0.0, 1.0).astype(np.float32),
        "low_inc_mask": low_inc_mask.astype(bool),
        "orientation_uncertain": orientation_uncertain.astype(bool),
    }

    if zc.shape != (cast_depth.size, cast_azimuth.size):
        errors.append(
            "CAST.Zc canonical shape mismatch: "
            f"expected {(cast_depth.size, cast_azimuth.size)}, observed {zc.shape}."
        )
    if not np.allclose(cast_azimuth, np.arange(180, dtype=np.float32) * 2.0):
        warnings.append("CAST azimuth axis is not exactly [0, 2, ..., 358].")
    if np.any(~np.isfinite(arrays["relbearing_deg"])):
        warnings.append("Interpolated RelBearing contains non-finite values.")
    if np.any(~np.isfinite(arrays["orientation_confidence"])):
        warnings.append("Interpolated orientation_confidence contains non-finite values.")

    summaries = {key: summarize_array(key, value) for key, value in arrays.items()}
    report = CastLabelInputReport(
        cast_label_input_version=CAST_LABEL_INPUT_VERSION,
        schema_version=SCHEMA_VERSION,
        data_version=DATA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "paths_config": str(paths_config_path),
            "mapping_path": str(mapping_path),
            "label_config_path": str(label_config_path),
            "orientation_confidence_npz": str(orientation_confidence_npz),
        },
        source_files={"cast": str(cast_path), "pose": str(pose_path)},
        chunking={
            "chunk_depth_samples": int(chunk_depth_samples),
            "cast_depth_samples": int(cast_depth.size),
            "cast_azimuth_count": int(cast_azimuth.size),
            "largest_chunk_bytes": int(
                min(chunk_depth_samples, max(cast_depth.size, 1))
                * max(cast_azimuth.size, 1)
                * np.dtype(np.float32).itemsize
            ),
        },
        arrays=summaries,
        relbearing_interpolation="circular_sin_cos_to_cast_depth",
        cast_azimuth={
            "canonical_shape": "[depth, cast_azimuth]",
            "values": "0,2,4,...,358",
            "direction": _as_dict(label_config.get("azimuth")).get(
                "cast_azimuth_direction", "normal"
            ),
        },
        warnings=warnings,
        errors=errors,
        not_performed=[
            "XSI waveform reading",
            "XSI feature extraction",
            "STFT",
            "STC",
            "APES",
            "weak label generation",
            "final label generation",
            "model training",
        ],
    )
    return report, arrays


def summarize_array(name: str, values: np.ndarray) -> ArraySummary:
    array = np.asarray(values)
    if array.dtype == np.bool_:
        numeric = array.astype(np.float32)
    else:
        numeric = array.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    finite_values = numeric[finite]
    if numeric.size == 0 or finite_values.size == 0:
        return ArraySummary(
            name=name,
            shape=[int(item) for item in array.shape],
            dtype=str(array.dtype),
            finite_ratio=None if numeric.size == 0 else float(np.mean(finite)),
            nan_count=int(np.count_nonzero(~finite)),
            min=None,
            max=None,
            mean=None,
        )
    return ArraySummary(
        name=name,
        shape=[int(item) for item in array.shape],
        dtype=str(array.dtype),
        finite_ratio=float(np.mean(finite)),
        nan_count=int(np.count_nonzero(~finite)),
        min=float(np.min(finite_values)),
        max=float(np.max(finite_values)),
        mean=float(np.mean(finite_values)),
    )


def write_cast_label_input_outputs(
    report: CastLabelInputReport,
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
    output_report_md.write_text(format_cast_label_input_markdown(report), encoding="utf-8")


def format_cast_label_input_markdown(report: CastLabelInputReport) -> str:
    data = report.to_dict()
    lines = [
        "# CAST Label Input Summary",
        "",
        f"- Version: {data['cast_label_input_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- CAST source: {data['source_files']['cast']}",
        f"- Pose source: {data['source_files']['pose']}",
        f"- Chunk depth samples: {data['chunking']['chunk_depth_samples']}",
        f"- Largest chunk bytes: {data['chunking']['largest_chunk_bytes']}",
        f"- RelBearing interpolation: {data['relbearing_interpolation']}",
        "",
        "## Arrays",
        "",
    ]
    for name, summary in data["arrays"].items():
        lines.append(
            f"- {name}: shape={summary['shape']}, finite_ratio={summary['finite_ratio']}, "
            f"range=[{summary['min']}, {summary['max']}]"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _read_depth_array(path: Path, variable_path: str) -> np.ndarray:
    request = MatReadRequest(
        variable_path=variable_path,
        role="depth",
        source_orientation=["depth"],
        canonical_orientation=["depth"],
        max_depth_samples=MAX_DEPTH_SAMPLES,
        max_time_samples=1,
        max_cast_azimuth=1,
    )
    return np.asarray(read_mat_file_slices(path, [request])[variable_path], dtype=np.float32)


def _read_cast_zc_chunks(
    path: Path,
    *,
    variable_path: str,
    depth_count: int,
    azimuth_count: int,
    chunk_depth_samples: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, depth_count, chunk_depth_samples):
        count = min(chunk_depth_samples, depth_count - start)
        request = MatReadRequest(
            variable_path=variable_path,
            role="cast_zc",
            source_orientation=["cast_azimuth", "depth"],
            canonical_orientation=["depth", "cast_azimuth"],
            max_depth_samples=count,
            max_time_samples=1,
            max_cast_azimuth=azimuth_count,
            source_start_index=start,
        )
        chunk = read_mat_file_slices(path, [request])[variable_path]
        chunks.append(np.asarray(chunk, dtype=np.float32))
    if not chunks:
        return np.empty((0, azimuth_count), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _cast_azimuth_axis(mapping: dict[str, Any], label_config: dict[str, Any]) -> np.ndarray:
    cast = _as_dict(mapping.get("cast"))
    azimuth = _as_dict(label_config.get("azimuth"))
    count = int(azimuth.get("cast_azimuth_count", cast.get("azimuth_count", 180)))
    step = float(azimuth.get("cast_azimuth_step_deg", cast.get("azimuth_step_deg", 2.0)))
    start = float(cast.get("azimuth_start_deg", 0.0))
    return wrap_deg(start + np.arange(count, dtype=np.float32) * step).astype(np.float32)


def _load_orientation_confidence(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _require_orientation_array(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"orientation confidence NPZ missing required array: {key}")
    return np.asarray(arrays[key])


def _interp_linear(
    source_depth: np.ndarray,
    source_values: np.ndarray,
    target_depth: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    depth, values = _finite_sorted(source_depth, source_values)
    if depth.size == 0:
        return np.full(target_depth.shape, fill_value, dtype=np.float32)
    return np.interp(target_depth, depth, values, left=fill_value, right=fill_value).astype(
        np.float32
    )


def _interp_angle_deg(
    source_depth: np.ndarray,
    source_angle_deg: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    depth, angle = _finite_sorted(source_depth, source_angle_deg)
    if depth.size == 0:
        return np.full(target_depth.shape, np.nan, dtype=np.float32)
    radians = np.deg2rad(angle)
    sin_values = np.interp(target_depth, depth, np.sin(radians), left=np.nan, right=np.nan)
    cos_values = np.interp(target_depth, depth, np.cos(radians), left=np.nan, right=np.nan)
    return wrap_deg(np.rad2deg(np.arctan2(sin_values, cos_values))).astype(np.float32)


def _interp_mask(
    source_depth: np.ndarray,
    source_mask: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    interpolated = _interp_linear(
        source_depth,
        np.asarray(source_mask, dtype=np.float32),
        target_depth,
        fill_value=1.0,
    )
    return interpolated >= 0.5


def _finite_sorted(
    source_depth: np.ndarray,
    source_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(source_depth, dtype=np.float64).reshape(-1)
    values = np.asarray(source_values, dtype=np.float64).reshape(-1)
    count = min(depth.size, values.size)
    depth = depth[:count]
    values = values[:count]
    finite = np.isfinite(depth) & np.isfinite(values)
    depth = depth[finite]
    values = values[finite]
    if depth.size == 0:
        return depth, values
    order = np.argsort(depth)
    return depth[order], values[order]


def _default_interim_path(config: dict[str, Any], filename: str) -> Path:
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if not interim:
        raise ValueError("data.interim is not configured.")
    return Path(str(interim)) / filename


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
