from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

TINY_HDF5_VERSION = "tiny_aligned_prototype_v001"
SCHEMA_VERSION = "schema_v001"
DATA_VERSION = "data_v001"

REQUIRED_DATASETS = [
    "/aligned/xsi_waveform",
    "/aligned/cast_zc",
    "/axis/depth",
    "/axis/xsi_side_azimuth_deg",
    "/axis/cast_azimuth_deg",
    "/pose/inc_deg",
    "/pose/rel_bearing_deg",
    "/metadata/schema_version",
    "/metadata/data_version",
    "/metadata/mapping_version",
    "/metadata/source_files",
    "/metadata/created_at",
]


@dataclass(frozen=True)
class TinyHDF5BuildResult:
    output_hdf5: str
    created_at: str
    datasets: dict[str, list[int]]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HDF5SchemaValidationResult:
    path: str
    is_valid: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_tiny_hdf5_prototype(
    *,
    small_slice_npz: Path | str,
    small_slice_summary: Path | str,
    output_hdf5: Path | str,
    overwrite: bool = False,
) -> TinyHDF5BuildResult:
    npz_path = Path(small_slice_npz)
    summary_path = Path(small_slice_summary)
    output_path = Path(output_hdf5)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite.")
    summary = _read_summary(summary_path)
    warnings = list(summary.get("warnings", []))
    errors = list(summary.get("errors", []))
    if errors:
        return TinyHDF5BuildResult(
            output_hdf5=str(output_path),
            created_at=datetime.now(timezone.utc).isoformat(),
            datasets={},
            warnings=warnings,
            errors=["small-slice summary contains errors"] + errors,
        )

    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    _require_arrays(
        arrays,
        [
            "xsi_waveform",
            "cast_zc",
            "cast_depth",
            "pose_inc_deg",
            "pose_rel_bearing_deg",
            "cast_azimuth_deg",
            "xsi_side_azimuth_deg",
        ],
    )
    warnings.extend(_depth_warnings(arrays))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        _write_dataset(h5, "/aligned/xsi_waveform", arrays["xsi_waveform"], dtype="float32")
        _write_dataset(h5, "/aligned/cast_zc", arrays["cast_zc"], dtype="float32")
        _write_dataset(h5, "/axis/depth", arrays["cast_depth"], dtype="float32")
        if _time_unit_known(summary):
            _write_dataset(
                h5,
                "/axis/time_ms",
                _time_axis_from_tad(arrays),
                dtype="float32",
            )
        else:
            time_count = int(arrays["xsi_waveform"].shape[-1])
            _write_dataset(
                h5,
                "/axis/time_sample_index",
                np.arange(time_count, dtype=np.int32),
                dtype="int32",
            )
            warnings.append(
                "time unit unknown; wrote /axis/time_sample_index instead of /axis/time_ms."
            )
        _write_dataset(
            h5,
            "/axis/xsi_side_azimuth_deg",
            arrays["xsi_side_azimuth_deg"],
            dtype="float32",
        )
        _write_dataset(
            h5,
            "/axis/cast_azimuth_deg",
            arrays["cast_azimuth_deg"],
            dtype="float32",
        )
        _write_dataset(
            h5,
            "/axis/receiver_index",
            arrays.get("receiver_index", np.arange(1, arrays["xsi_waveform"].shape[1] + 1)),
            dtype="int16",
        )
        _write_dataset(
            h5,
            "/axis/side_index",
            arrays.get("side_index", np.arange(1, arrays["xsi_waveform"].shape[2] + 1)),
            dtype="int16",
        )
        _write_dataset(h5, "/pose/inc_deg", arrays["pose_inc_deg"], dtype="float32")
        _write_dataset(
            h5,
            "/pose/rel_bearing_deg",
            arrays["pose_rel_bearing_deg"],
            dtype="float32",
        )
        _write_metadata(h5, summary, warnings)

    validation = validate_tiny_hdf5_schema(output_path)
    return TinyHDF5BuildResult(
        output_hdf5=str(output_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        datasets=_dataset_shapes(output_path),
        warnings=warnings + validation.warnings,
        errors=validation.errors,
    )


def validate_tiny_hdf5_schema(path: Path | str) -> HDF5SchemaValidationResult:
    hdf5_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    if not hdf5_path.exists():
        return HDF5SchemaValidationResult(
            path=str(hdf5_path),
            is_valid=False,
            errors=[f"HDF5 file does not exist: {hdf5_path}"],
            warnings=[],
        )

    with h5py.File(hdf5_path, "r") as h5:
        for dataset in REQUIRED_DATASETS:
            if dataset not in h5:
                errors.append(f"Missing required dataset: {dataset}")
        if "/axis/time_ms" not in h5 and "/axis/time_sample_index" not in h5:
            errors.append("Missing required time axis: /axis/time_ms or /axis/time_sample_index")
        if errors:
            return HDF5SchemaValidationResult(str(hdf5_path), False, errors, warnings)

        waveform_shape = h5["/aligned/xsi_waveform"].shape
        cast_shape = h5["/aligned/cast_zc"].shape
        if len(waveform_shape) != 4:
            errors.append(f"/aligned/xsi_waveform must be rank 4, observed {waveform_shape}")
        if len(cast_shape) != 2:
            errors.append(f"/aligned/cast_zc must be rank 2, observed {cast_shape}")
        if len(waveform_shape) == 4:
            depth, _receiver, side, time = waveform_shape
            if side != h5["/axis/xsi_side_azimuth_deg"].shape[0]:
                errors.append("XSI side dimension does not match /axis/xsi_side_azimuth_deg.")
            time_axis = "/axis/time_ms" if "/axis/time_ms" in h5 else "/axis/time_sample_index"
            if time != h5[time_axis].shape[0]:
                errors.append("XSI time dimension does not match the time axis.")
            if depth != h5["/axis/depth"].shape[0]:
                warnings.append(
                    "Prototype depth axis is not a formal alignment result; "
                    "depth dimensions are only checked for tiny shape consistency."
                )
        if len(cast_shape) == 2:
            if cast_shape[1] != h5["/axis/cast_azimuth_deg"].shape[0]:
                errors.append("CAST azimuth dimension does not match /axis/cast_azimuth_deg.")
            if cast_shape[0] != h5["/axis/depth"].shape[0]:
                warnings.append("CAST depth dimension differs from /axis/depth.")

    return HDF5SchemaValidationResult(
        path=str(hdf5_path),
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def _write_dataset(h5: h5py.File, path: str, array: np.ndarray, *, dtype: str) -> None:
    h5.create_dataset(path, data=np.asarray(array, dtype=dtype), compression="gzip")


def _write_metadata(h5: h5py.File, summary: dict[str, Any], warnings: list[str]) -> None:
    metadata = h5.require_group("metadata")
    _write_string(metadata, "schema_version", str(summary.get("schema_version", SCHEMA_VERSION)))
    _write_string(metadata, "data_version", str(summary.get("data_version", DATA_VERSION)))
    mapping_version = str(summary.get("mapping_version", "raw_variable_mapping_v001"))
    source_files = json.dumps(summary.get("source_files", {}), ensure_ascii=False)
    _write_string(metadata, "mapping_version", mapping_version)
    _write_string(metadata, "source_files", source_files)
    _write_string(metadata, "created_at", datetime.now(timezone.utc).isoformat())
    _write_string(metadata, "tiny_hdf5_version", TINY_HDF5_VERSION)
    _write_string(metadata, "depth_unit", "unknown_to_verify")
    _write_string(metadata, "time_unit", "unknown_to_verify")
    _write_string(metadata, "azimuth_unit", "degree")
    _write_string(metadata, "zc_unit", "MRayl")
    _write_string(metadata, "warnings", json.dumps(warnings, ensure_ascii=False))
    _write_string(metadata, "not_performed", json.dumps(_not_performed(), ensure_ascii=False))


def _write_string(group: h5py.Group, name: str, value: str) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=value, dtype=dtype)


def _read_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Small-slice summary must contain an object: {path}")
    return data


def _require_arrays(arrays: dict[str, np.ndarray], names: list[str]) -> None:
    missing = [name for name in names if name not in arrays]
    if missing:
        raise ValueError("Small-slice NPZ is missing arrays: " + ", ".join(missing))


def _depth_warnings(arrays: dict[str, np.ndarray]) -> list[str]:
    warnings: list[str] = []
    cast_depth = arrays.get("cast_depth")
    pose_depth = arrays.get("pose_depth")
    xsi_depth = arrays.get("xsi_depth")
    if cast_depth is not None and pose_depth is not None:
        if cast_depth.shape != pose_depth.shape or not np.allclose(cast_depth, pose_depth):
            warnings.append("CAST and pose depth slices differ; no depth alignment was performed.")
    if cast_depth is not None and xsi_depth is not None:
        first_xsi_depth = np.asarray(xsi_depth)[0]
        if cast_depth.shape != first_xsi_depth.shape or not np.allclose(
            cast_depth,
            first_xsi_depth,
        ):
            warnings.append("CAST and XSI depth slices differ; no depth alignment was performed.")
    return warnings


def _time_unit_known(summary: dict[str, Any]) -> bool:
    warnings = [str(warning).lower() for warning in summary.get("warnings", [])]
    return not any("time unit is unknown" in warning for warning in warnings)


def _time_axis_from_tad(arrays: dict[str, np.ndarray]) -> np.ndarray:
    time_count = int(arrays["xsi_waveform"].shape[-1])
    tad = np.asarray(arrays.get("xsi_tad", [1.0]), dtype=np.float32).reshape(-1)
    step = float(tad[0]) if tad.size else 1.0
    return np.arange(time_count, dtype=np.float32) * step


def _dataset_shapes(path: Path) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}

    def collect_shape(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset):
            shapes["/" + name] = list(item.shape)

    with h5py.File(path, "r") as h5:
        h5.visititems(collect_shape)
    return shapes


def _not_performed() -> list[str]:
    return [
        "depth alignment",
        "RelBearing rotation",
        "label generation",
        "feature extraction",
        "model training",
    ]
