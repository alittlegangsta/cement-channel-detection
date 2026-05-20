from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from cement_channel.data.manifest import load_paths_config

SMALL_SLICE_VERSION = "small_slice_v001"
SCHEMA_VERSION = "schema_v001"
DATA_VERSION = "data_v001"

MAX_DEPTH_LIMIT = 64
MAX_TIME_LIMIT = 64
MAX_RECEIVER_LIMIT = 13
MAX_SIDE_LIMIT = 8
MAX_CAST_AZIMUTH_LIMIT = 180

MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15

MX_STRUCT = 2

MI_NUMPY_DTYPES = {
    MI_INT8: "i1",
    MI_UINT8: "u1",
    MI_INT16: "i2",
    MI_UINT16: "u2",
    MI_INT32: "i4",
    MI_UINT32: "u4",
    MI_SINGLE: "f4",
    MI_DOUBLE: "f8",
    MI_INT64: "i8",
    MI_UINT64: "u8",
}


@dataclass(frozen=True)
class SmallSliceLimits:
    max_depth_samples: int = MAX_DEPTH_LIMIT
    max_time_samples: int = MAX_TIME_LIMIT
    max_receivers: int = MAX_RECEIVER_LIMIT
    max_sides: int = MAX_SIDE_LIMIT
    max_cast_azimuth: int = MAX_CAST_AZIMUTH_LIMIT


@dataclass(frozen=True)
class DepthWindow:
    depth_start: float
    depth_stop: float


@dataclass(frozen=True)
class DepthWindowSelection:
    source_start_index: int
    sample_count: int
    matched_count: int
    observed_min: float | None
    observed_max: float | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VariableSliceSummary:
    name: str
    path: str
    shape: list[int]
    dtype: str
    finite_ratio: float | None
    min: float | None
    max: float | None
    mean: float | None
    source_orientation: list[str]
    canonical_orientation_suggestion: list[str]
    warnings: list[str]
    errors: list[str]


@dataclass(frozen=True)
class SmallSliceResult:
    slice_version: str
    schema_version: str
    data_version: str
    mapping_path: str
    created_at: str
    limits: dict[str, int]
    depth_window: dict[str, Any] | None
    source_files: dict[str, Any]
    variables: dict[str, VariableSliceSummary]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatReadRequest:
    variable_path: str
    role: str
    source_orientation: list[str]
    canonical_orientation: list[str]
    max_depth_samples: int
    max_time_samples: int
    max_cast_azimuth: int
    source_start_index: int = 0


@dataclass
class MatDataTag:
    data_type: int
    nbytes: int
    inline_data: bytes | None = None


@dataclass
class MatrixHeader:
    matlab_class: int
    dims: list[int]
    name: str


class MatSliceReadError(RuntimeError):
    """Raised for recoverable MAT slice parsing errors."""


class _FileReader:
    def __init__(self, file_obj: BinaryIO) -> None:
        self.file_obj = file_obj

    def read(self, nbytes: int) -> bytes:
        data = self.file_obj.read(nbytes)
        if len(data) != nbytes:
            raise MatSliceReadError("Unexpected end of MAT file.")
        return data

    def skip(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        self.file_obj.seek(nbytes, 1)


class _LimitedReader:
    def __init__(self, parent: Any, limit: int) -> None:
        self.parent = parent
        self.limit = limit
        self.consumed = 0

    def read(self, nbytes: int) -> bytes:
        if self.consumed + nbytes > self.limit:
            raise MatSliceReadError("MAT element parser attempted to read past element limit.")
        data = self.parent.read(nbytes)
        self.consumed += nbytes
        return data

    def skip(self, nbytes: int) -> None:
        if self.consumed + nbytes > self.limit:
            raise MatSliceReadError("MAT element parser attempted to skip past element limit.")
        self.parent.skip(nbytes)
        self.consumed += nbytes

    def skip_remaining(self) -> None:
        self.skip(self.limit - self.consumed)


class _CompressedReader:
    def __init__(self, file_obj: BinaryIO, compressed_nbytes: int) -> None:
        self.file_obj = file_obj
        self.remaining_compressed = compressed_nbytes
        self.decompressor = zlib.decompressobj()
        self.buffer = b""

    def read(self, nbytes: int) -> bytes:
        while len(self.buffer) < nbytes:
            if self.remaining_compressed <= 0:
                self.buffer += self.decompressor.flush()
                break
            chunk = self.file_obj.read(min(65536, self.remaining_compressed))
            if not chunk:
                break
            self.remaining_compressed -= len(chunk)
            self.buffer += self.decompressor.decompress(chunk)
        if len(self.buffer) < nbytes:
            raise MatSliceReadError("Unexpected end of compressed MAT payload.")
        data = self.buffer[:nbytes]
        self.buffer = self.buffer[nbytes:]
        return data

    def skip(self, nbytes: int) -> None:
        remaining = nbytes
        while remaining > 0:
            chunk = self.read(min(65536, remaining))
            remaining -= len(chunk)


def load_mapping_config(mapping_path: Path | str) -> dict[str, Any]:
    import yaml

    path = Path(mapping_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Mapping config must contain a YAML mapping: {path}")
    return data


def read_small_slice_from_configs(
    paths_config: Path | str,
    mapping_path: Path | str,
    *,
    limits: SmallSliceLimits | None = None,
    depth_window: DepthWindow | None = None,
    depth_reference_npz: Path | str | None = None,
) -> tuple[SmallSliceResult, dict[str, np.ndarray]]:
    config = load_paths_config(paths_config)
    mapping = load_mapping_config(mapping_path)
    depth_reference_arrays = (
        load_depth_reference_arrays(depth_reference_npz)
        if depth_reference_npz is not None
        else None
    )
    return read_small_slice(
        config,
        mapping,
        mapping_path=Path(mapping_path),
        limits=limits,
        depth_window=depth_window,
        depth_reference_arrays=depth_reference_arrays,
    )


def read_small_slice(
    paths_config: dict[str, Any],
    mapping: dict[str, Any],
    *,
    mapping_path: Path | str,
    limits: SmallSliceLimits | None = None,
    depth_window: DepthWindow | None = None,
    depth_reference_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[SmallSliceResult, dict[str, np.ndarray]]:
    active_limits = _validate_limits(limits or SmallSliceLimits())
    data_config = _as_dict(paths_config.get("data"))
    raw_dir = Path(str(data_config.get("raw", "")))
    warnings: list[str] = []
    errors: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    summaries: dict[str, VariableSliceSummary] = {}
    source_files: dict[str, Any] = {}
    window_selection: dict[str, Any] | None = None

    if not raw_dir.exists():
        errors.append(f"Raw directory does not exist: {raw_dir}")
        result = _result(
            mapping_path,
            active_limits,
            window_selection,
            source_files,
            summaries,
            warnings,
            errors,
        )
        return result, arrays

    if depth_window is not None:
        if depth_reference_arrays is None:
            errors.append("depth_reference_arrays are required when depth_window is provided.")
            result = _result(
                mapping_path,
                active_limits,
                window_selection,
                source_files,
                summaries,
                warnings,
                errors,
            )
            return result, arrays
        window_selection = select_depth_window_slices(
            depth_reference_arrays,
            depth_window,
            max_depth_samples=active_limits.max_depth_samples,
        )
        warnings.extend(_window_selection_warnings(window_selection))

    _read_cast(
        raw_dir,
        mapping,
        active_limits,
        window_selection,
        arrays,
        summaries,
        source_files,
        errors,
    )
    _read_pose(
        raw_dir,
        mapping,
        active_limits,
        window_selection,
        arrays,
        summaries,
        source_files,
        errors,
    )
    _read_xsi(
        raw_dir,
        mapping,
        active_limits,
        window_selection,
        arrays,
        summaries,
        source_files,
        errors,
    )

    if _unit_unknown(mapping, ("cast", "depth_unit")):
        warnings.append("CAST depth unit is unknown_to_verify.")
    if _unit_unknown(mapping, ("xsi", "time_unit")):
        warnings.append("XSI time unit is unknown_to_verify.")
    if _unit_unknown(mapping, ("xsi", "depth_unit")):
        warnings.append("XSI depth unit is unknown_to_verify.")

    return (
        _result(
            mapping_path,
            active_limits,
            window_selection,
            source_files,
            summaries,
            warnings,
            errors,
        ),
        arrays,
    )


def write_small_slice_outputs(
    result: SmallSliceResult,
    arrays: dict[str, np.ndarray],
    *,
    output_json: Path,
    output_npz: Path | None,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_json, overwrite=overwrite)
    if output_npz is not None:
        _ensure_can_write(output_npz, overwrite=overwrite)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, **arrays)


def load_depth_reference_arrays(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Depth reference NPZ does not exist: {npz_path}")
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def depth_window_from_center(
    *,
    depth_center: float,
    depth_window_size: float,
) -> DepthWindow:
    if depth_window_size <= 0.0:
        raise ValueError("depth_window_size must be positive.")
    half = depth_window_size / 2.0
    return DepthWindow(
        depth_start=float(depth_center - half), depth_stop=float(depth_center + half)
    )


def depth_window_from_grid_proposal(
    proposal_json: Path | str,
    *,
    depth_window_size: float = 2.0,
) -> DepthWindow:
    proposal = json.loads(Path(proposal_json).read_text(encoding="utf-8"))
    if not isinstance(proposal, dict):
        raise ValueError(f"Depth grid proposal must contain an object: {proposal_json}")
    start = _first_float(
        proposal.get("common_overlap_min"),
        proposal.get("depth_start"),
    )
    stop = _first_float(
        proposal.get("common_overlap_max"),
        proposal.get("depth_stop"),
    )
    if start is None or stop is None or stop <= start:
        raise ValueError("Depth grid proposal does not define a positive common overlap.")
    return depth_window_from_center(
        depth_center=(start + stop) / 2.0,
        depth_window_size=min(float(depth_window_size), 2.0),
    )


def select_depth_window_slices(
    depth_reference_arrays: dict[str, np.ndarray],
    depth_window: DepthWindow,
    *,
    max_depth_samples: int,
) -> dict[str, Any]:
    if depth_window.depth_stop <= depth_window.depth_start:
        raise ValueError("depth_stop must be greater than depth_start.")
    selection: dict[str, Any] = {
        "requested": asdict(depth_window),
        "max_depth_samples": int(max_depth_samples),
    }
    selection["cast"] = _select_single_depth_window(
        np.asarray(depth_reference_arrays.get("cast_depth", [])),
        depth_window,
        max_depth_samples=max_depth_samples,
        name="cast_depth",
    ).to_dict()
    selection["pose"] = _select_single_depth_window(
        np.asarray(depth_reference_arrays.get("pose_depth", [])),
        depth_window,
        max_depth_samples=max_depth_samples,
        name="pose_depth",
    ).to_dict()
    xsi = np.asarray(depth_reference_arrays.get("xsi_depth_by_receiver", []))
    receiver_selections: dict[str, dict[str, Any]] = {}
    if xsi.ndim == 1:
        xsi = xsi.reshape(1, -1)
    if xsi.ndim != 2 or xsi.size == 0:
        receiver_selections["receiver_01"] = DepthWindowSelection(
            source_start_index=0,
            sample_count=0,
            matched_count=0,
            observed_min=None,
            observed_max=None,
            warnings=["xsi_depth_by_receiver is missing or not rank 2."],
        ).to_dict()
    else:
        for index in range(xsi.shape[0]):
            receiver_selections[f"receiver_{index + 1:02d}"] = _select_single_depth_window(
                xsi[index],
                depth_window,
                max_depth_samples=max_depth_samples,
                name=f"xsi_depth_receiver_{index + 1:02d}",
            ).to_dict()
    selection["xsi_receivers"] = receiver_selections
    return selection


def read_mat_file_slices(
    path: Path | str,
    requests: list[MatReadRequest],
) -> dict[str, np.ndarray]:
    request_by_top: dict[str, list[MatReadRequest]] = {}
    for request in requests:
        top = request.variable_path.split(".")[0]
        request_by_top.setdefault(top, []).append(request)

    found: dict[str, np.ndarray] = {}
    with Path(path).open("rb") as file_obj:
        header = file_obj.read(128)
        if len(header) != 128:
            raise MatSliceReadError(f"Invalid MAT header: {path}")
        endian = _endian_from_header(header)
        reader = _FileReader(file_obj)
        file_size = Path(path).stat().st_size
        while file_obj.tell() < file_size and len(found) < len(requests):
            tag = _read_tag_or_none(reader, endian)
            if tag is None:
                break
            data_start = file_obj.tell()
            if tag.data_type == MI_COMPRESSED:
                compressed_reader = _CompressedReader(file_obj, tag.nbytes)
                _read_slices_from_stream(compressed_reader, endian, request_by_top, found)
                file_obj.seek(data_start + tag.nbytes)
            elif tag.data_type == MI_MATRIX:
                _read_matrix_for_requests(reader, tag, endian, request_by_top, found)
            else:
                _skip_tag_payload(reader, tag)

    missing = sorted(
        request.variable_path for request in requests if request.variable_path not in found
    )
    if missing:
        raise MatSliceReadError("Missing MAT variable path(s): " + ", ".join(missing))
    return found


def _read_cast(
    raw_dir: Path,
    mapping: dict[str, Any],
    limits: SmallSliceLimits,
    window_selection: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
    summaries: dict[str, VariableSliceSummary],
    source_files: dict[str, Any],
    errors: list[str],
) -> None:
    cast = _as_dict(mapping.get("cast"))
    path = raw_dir / str(cast.get("file", "CAST.mat"))
    source_files["cast"] = str(path)
    selection = _selection_for_source(window_selection, "cast")
    if selection is not None and int(selection.get("sample_count", 0)) <= 0:
        errors.append("cast_depth has no samples in the requested depth window.")
        return
    depth_count = _selected_depth_count(selection, limits.max_depth_samples)
    source_start_index = _selected_start_index(selection)
    requests = [
        MatReadRequest(
            variable_path=str(cast.get("zc_variable", "")),
            role="cast_zc",
            source_orientation=_as_str_list(cast.get("zc_source_shape_order")),
            canonical_orientation=_as_str_list(cast.get("zc_canonical_shape_order")),
            max_depth_samples=depth_count,
            max_time_samples=limits.max_time_samples,
            max_cast_azimuth=limits.max_cast_azimuth,
            source_start_index=source_start_index,
        ),
        MatReadRequest(
            variable_path=str(cast.get("depth_variable", "")),
            role="depth",
            source_orientation=_as_str_list(cast.get("depth_source_shape_order"), ["depth"]),
            canonical_orientation=["depth"],
            max_depth_samples=depth_count,
            max_time_samples=limits.max_time_samples,
            max_cast_azimuth=limits.max_cast_azimuth,
            source_start_index=source_start_index,
        ),
    ]
    _read_request_group(
        path,
        requests,
        arrays,
        summaries,
        errors,
        key_by_role={"cast_zc": "cast_zc", "depth": "cast_depth"},
    )
    if "cast_zc" in arrays:
        arrays["cast_azimuth_deg"] = _cast_azimuth_axis(cast, arrays["cast_zc"].shape[1])


def _read_pose(
    raw_dir: Path,
    mapping: dict[str, Any],
    limits: SmallSliceLimits,
    window_selection: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
    summaries: dict[str, VariableSliceSummary],
    source_files: dict[str, Any],
    errors: list[str],
) -> None:
    pose = _as_dict(mapping.get("pose"))
    path = raw_dir / str(pose.get("file", "D2_XSI_RelBearing_Inclination.mat"))
    source_files["pose"] = str(path)
    orientation = _as_str_list(pose.get("source_shape_order"), ["depth"])
    selection = _selection_for_source(window_selection, "pose")
    if selection is not None and int(selection.get("sample_count", 0)) <= 0:
        errors.append("pose_depth has no samples in the requested depth window.")
        return
    depth_count = _selected_depth_count(selection, limits.max_depth_samples)
    source_start_index = _selected_start_index(selection)
    requests = [
        MatReadRequest(
            variable_path=str(pose.get("depth_variable", "")),
            role="depth",
            source_orientation=orientation,
            canonical_orientation=["depth"],
            max_depth_samples=depth_count,
            max_time_samples=limits.max_time_samples,
            max_cast_azimuth=limits.max_cast_azimuth,
            source_start_index=source_start_index,
        ),
        MatReadRequest(
            variable_path=str(pose.get("inclination_variable", "")),
            role="pose_inc",
            source_orientation=orientation,
            canonical_orientation=["depth"],
            max_depth_samples=depth_count,
            max_time_samples=limits.max_time_samples,
            max_cast_azimuth=limits.max_cast_azimuth,
            source_start_index=source_start_index,
        ),
        MatReadRequest(
            variable_path=str(pose.get("relbearing_variable", "")),
            role="pose_relbearing",
            source_orientation=orientation,
            canonical_orientation=["depth"],
            max_depth_samples=depth_count,
            max_time_samples=limits.max_time_samples,
            max_cast_azimuth=limits.max_cast_azimuth,
            source_start_index=source_start_index,
        ),
    ]
    key_by_role = {
        "depth": "pose_depth",
        "pose_inc": "pose_inc_deg",
        "pose_relbearing": "pose_rel_bearing_deg",
    }
    _read_request_group(path, requests, arrays, summaries, errors, key_by_role=key_by_role)


def _read_xsi(
    raw_dir: Path,
    mapping: dict[str, Any],
    limits: SmallSliceLimits,
    window_selection: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
    summaries: dict[str, VariableSliceSummary],
    source_files: dict[str, Any],
    errors: list[str],
) -> None:
    xsi = _as_dict(mapping.get("xsi"))
    receiver_dir = raw_dir / str(xsi.get("receiver_dir", "XSILMR"))
    receiver_count = min(int(xsi.get("expected_receiver_files", 13)), limits.max_receivers)
    side_labels = _as_str_list(xsi.get("side_labels"), list("ABCDEFGH"))[: limits.max_sides]
    source_files["xsi_receiver_files"] = []
    waveform_slices: list[np.ndarray] = []
    depth_slices: list[np.ndarray] = []
    tad_values: list[float] = []
    waveform_summaries: list[VariableSliceSummary] = []
    depth_summaries: list[VariableSliceSummary] = []
    tad_summaries: list[VariableSliceSummary] = []

    for receiver_index in range(1, receiver_count + 1):
        receiver_file = receiver_dir / f"XSILMR{receiver_index:02d}.mat"
        source_files["xsi_receiver_files"].append(str(receiver_file))
        receiver_selection = _selection_for_xsi_receiver(window_selection, receiver_index)
        if receiver_selection is not None and int(receiver_selection.get("sample_count", 0)) <= 0:
            errors.append(f"xsi receiver {receiver_index}: no depth samples in requested window.")
            continue
        depth_count = _selected_depth_count(receiver_selection, limits.max_depth_samples)
        source_start_index = _selected_start_index(receiver_selection)
        requests = [
            MatReadRequest(
                variable_path=_format_pattern(
                    str(xsi.get("depth_variable_pattern", "")),
                    receiver_index,
                ),
                role="depth",
                source_orientation=_as_str_list(xsi.get("depth_source_shape_order"), ["depth"]),
                canonical_orientation=["depth"],
                max_depth_samples=depth_count,
                max_time_samples=limits.max_time_samples,
                max_cast_azimuth=limits.max_cast_azimuth,
                source_start_index=source_start_index,
            ),
            MatReadRequest(
                variable_path=_format_pattern(
                    str(xsi.get("time_variable_pattern", "")),
                    receiver_index,
                ),
                role="xsi_time",
                source_orientation=["scalar"],
                canonical_orientation=["receiver"],
                max_depth_samples=depth_count,
                max_time_samples=limits.max_time_samples,
                max_cast_azimuth=limits.max_cast_azimuth,
            ),
        ]
        requests.extend(
            MatReadRequest(
                variable_path=_format_pattern(
                    str(xsi.get("waveform_variable_pattern", "")),
                    receiver_index,
                    side=side,
                ),
                role="xsi_waveform",
                source_orientation=_as_str_list(xsi.get("waveform_source_shape_order")),
                canonical_orientation=_as_str_list(xsi.get("waveform_canonical_shape_order")),
                max_depth_samples=depth_count,
                max_time_samples=limits.max_time_samples,
                max_cast_azimuth=limits.max_cast_azimuth,
                source_start_index=source_start_index,
            )
            for side in side_labels
        )
        try:
            data = read_mat_file_slices(receiver_file, requests)
        except Exception as exc:
            errors.append(f"xsi receiver {receiver_index}: {type(exc).__name__}: {exc}")
            continue
        depth_key = requests[0].variable_path
        time_key = requests[1].variable_path
        depth_slices.append(np.asarray(data[depth_key], dtype=np.float32))
        tad_values.append(float(np.asarray(data[time_key]).reshape(-1)[0]))
        side_arrays = [
            np.asarray(data[request.variable_path], dtype=np.float32) for request in requests[2:]
        ]
        waveform_slices.append(np.stack(side_arrays, axis=1))
        depth_summaries.append(_summary(depth_key, depth_key, data[depth_key], requests[0]))
        tad_summaries.append(_summary(time_key, time_key, data[time_key], requests[1]))
        waveform_summaries.extend(
            _summary(
                request.variable_path,
                request.variable_path,
                data[request.variable_path],
                request,
            )
            for request in requests[2:]
        )

    if waveform_slices:
        arrays["xsi_waveform"] = np.stack(waveform_slices, axis=1).astype(np.float32)
        summaries["xsi_waveform"] = _aggregate_summary(
            "xsi_waveform",
            arrays["xsi_waveform"],
            waveform_summaries,
            ["depth", "receiver", "side", "time"],
        )
    if depth_slices:
        arrays["xsi_depth"] = np.stack(depth_slices, axis=0).astype(np.float32)
        summaries["xsi_depth"] = _aggregate_summary(
            "xsi_depth",
            arrays["xsi_depth"],
            depth_summaries,
            ["receiver", "depth"],
        )
    if tad_values:
        arrays["xsi_tad"] = np.asarray(tad_values, dtype=np.float32)
        summaries["xsi_tad"] = _aggregate_summary(
            "xsi_tad",
            arrays["xsi_tad"],
            tad_summaries,
            ["receiver"],
        )
    arrays["receiver_index"] = np.arange(1, receiver_count + 1, dtype=np.int16)
    arrays["side_index"] = np.arange(1, len(side_labels) + 1, dtype=np.int16)
    arrays["xsi_side_azimuth_deg"] = np.linspace(
        0.0,
        360.0,
        num=len(side_labels),
        endpoint=False,
        dtype=np.float32,
    )


def _read_request_group(
    path: Path,
    requests: list[MatReadRequest],
    arrays: dict[str, np.ndarray],
    summaries: dict[str, VariableSliceSummary],
    errors: list[str],
    *,
    key_prefix: str | None = None,
    key_by_role: dict[str, str] | None = None,
) -> None:
    try:
        data = read_mat_file_slices(path, requests)
    except Exception as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return
    for request in requests:
        key = key_by_role.get(request.role) if key_by_role else None
        if key is None:
            key = f"{key_prefix}_{request.role}" if key_prefix else request.role
        arrays[key] = np.asarray(data[request.variable_path])
        summaries[key] = _summary(key, request.variable_path, arrays[key], request)


def _read_slices_from_stream(
    reader: Any,
    endian: str,
    requests_by_top: dict[str, list[MatReadRequest]],
    found: dict[str, np.ndarray],
) -> None:
    while True:
        tag = _read_tag_or_none(reader, endian)
        if tag is None:
            return
        if tag.data_type == MI_MATRIX:
            _read_matrix_for_requests(reader, tag, endian, requests_by_top, found)
            return
        _skip_tag_payload(reader, tag)


def _read_matrix_for_requests(
    reader: Any,
    tag: MatDataTag,
    endian: str,
    requests_by_top: dict[str, list[MatReadRequest]],
    found: dict[str, np.ndarray],
) -> None:
    payload = _LimitedReader(reader, tag.nbytes)
    header = _read_matrix_header(payload, endian)
    top_requests = requests_by_top.get(header.name, [])
    if not top_requests:
        payload.skip_remaining()
        _skip_padding(reader, tag.nbytes)
        return

    if header.matlab_class == MX_STRUCT:
        _read_struct_fields(payload, endian, header.name, top_requests, found)
    else:
        for request in top_requests:
            if request.variable_path == header.name:
                found[request.variable_path] = _read_numeric_from_matrix_payload(
                    payload,
                    endian,
                    header,
                    request,
                )
                break
    payload.skip_remaining()
    _skip_padding(reader, tag.nbytes)


def _read_struct_fields(
    payload: _LimitedReader,
    endian: str,
    top_name: str,
    requests: list[MatReadRequest],
    found: dict[str, np.ndarray],
) -> None:
    field_name_length_data = _read_element_data(payload, _read_tag(payload, endian))
    field_name_length = int(np.frombuffer(field_name_length_data[:4], dtype=endian + "i4")[0])
    field_names_data = _read_element_data(payload, _read_tag(payload, endian))
    field_names = _decode_field_names(field_names_data, field_name_length)
    request_by_field = {
        request.variable_path.split(".", 1)[1]: request
        for request in requests
        if "." in request.variable_path
    }
    for field_name in field_names:
        field_tag = _read_tag(payload, endian)
        request = request_by_field.get(field_name)
        if request is None:
            _skip_tag_payload(payload, field_tag)
            continue
        field_payload = _LimitedReader(payload, field_tag.nbytes)
        header = _read_matrix_header(field_payload, endian)
        found[f"{top_name}.{field_name}"] = _read_numeric_from_matrix_payload(
            field_payload,
            endian,
            header,
            request,
        )
        field_payload.skip_remaining()
        _skip_padding(payload, field_tag.nbytes)


def _read_numeric_from_matrix_payload(
    payload: _LimitedReader,
    endian: str,
    header: MatrixHeader,
    request: MatReadRequest,
) -> np.ndarray:
    data_tag = _read_tag(payload, endian)
    if data_tag.data_type not in MI_NUMPY_DTYPES:
        raise MatSliceReadError(
            f"{request.variable_path} has unsupported numeric data type {data_tag.data_type}"
        )
    dtype = np.dtype(endian + MI_NUMPY_DTYPES[data_tag.data_type])
    skip_count = _source_skip_element_count(header.dims, request)
    read_count = _required_source_element_count(header.dims, request)
    total_count = _element_count(header.dims)
    if skip_count + read_count > total_count:
        raise MatSliceReadError(
            f"{request.variable_path} requested start={skip_count}, count={read_count} "
            f"from shape {header.dims}"
        )
    values = _read_numeric_values(payload, data_tag, dtype, read_count, skip_count=skip_count)
    if data_tag.inline_data is None:
        remaining_bytes = data_tag.nbytes - ((skip_count + read_count) * dtype.itemsize)
        if remaining_bytes > 0:
            payload.skip(remaining_bytes)
        _skip_padding(payload, data_tag.nbytes)
    return _canonicalize_values(values, header.dims, request)


def _read_matrix_header(payload: _LimitedReader, endian: str) -> MatrixHeader:
    flags = _read_element_data(payload, _read_tag(payload, endian))
    if len(flags) < 4:
        raise MatSliceReadError("Invalid MAT matrix flags.")
    matlab_class = int(np.frombuffer(flags[:4], dtype=endian + "u4")[0] & 0xFF)
    dims_data = _read_element_data(payload, _read_tag(payload, endian))
    dims = [int(value) for value in np.frombuffer(dims_data, dtype=endian + "i4")]
    name_data = _read_element_data(payload, _read_tag(payload, endian))
    name = name_data.decode("utf-8", errors="ignore")
    return MatrixHeader(matlab_class=matlab_class, dims=dims, name=name)


def _read_numeric_values(
    payload: _LimitedReader,
    data_tag: MatDataTag,
    dtype: np.dtype,
    count: int,
    *,
    skip_count: int,
) -> np.ndarray:
    nbytes = count * dtype.itemsize
    if data_tag.inline_data is not None:
        start = skip_count * dtype.itemsize
        data = data_tag.inline_data[start : start + nbytes]
    else:
        if skip_count:
            payload.skip(skip_count * dtype.itemsize)
        data = payload.read(nbytes)
    if len(data) != nbytes:
        raise MatSliceReadError("Numeric MAT payload ended before requested slice.")
    return np.frombuffer(data, dtype=dtype, count=count).copy()


def _canonicalize_values(
    values: np.ndarray,
    dims: list[int],
    request: MatReadRequest,
) -> np.ndarray:
    role = request.role
    if role == "cast_zc":
        rows, cols = _matrix_rows_cols(dims)
        depth_count = _requested_depth_count(cols, request)
        azimuth_count = min(request.max_cast_azimuth, rows)
        source = values.reshape((rows, depth_count), order="F")
        return source[:azimuth_count, :].T.astype(np.float32)
    if role == "xsi_waveform":
        rows, cols = _matrix_rows_cols(dims)
        depth_count = _requested_depth_count(cols, request)
        time_count = min(request.max_time_samples, rows)
        source = values.reshape((rows, depth_count), order="F")
        return source[:time_count, :].T.astype(np.float32)
    if role == "xsi_time":
        return values[:1].astype(np.float32)
    return values[: request.max_depth_samples].reshape(-1).astype(np.float32)


def _required_source_element_count(dims: list[int], request: MatReadRequest) -> int:
    if request.role == "cast_zc":
        rows, cols = _matrix_rows_cols(dims)
        return rows * _requested_depth_count(cols, request)
    if request.role == "xsi_waveform":
        rows, cols = _matrix_rows_cols(dims)
        return rows * _requested_depth_count(cols, request)
    if request.role == "xsi_time":
        return 1
    total_count = _element_count(dims)
    return max(min(request.max_depth_samples, total_count - request.source_start_index), 0)


def _source_skip_element_count(dims: list[int], request: MatReadRequest) -> int:
    start_index = max(int(request.source_start_index), 0)
    if request.role in {"cast_zc", "xsi_waveform"}:
        rows, _cols = _matrix_rows_cols(dims)
        return rows * start_index
    if request.role == "xsi_time":
        return 0
    return start_index


def _requested_depth_count(source_depth_count: int, request: MatReadRequest) -> int:
    remaining = max(source_depth_count - max(int(request.source_start_index), 0), 0)
    return min(request.max_depth_samples, remaining)


def _read_tag_or_none(reader: Any, endian: str) -> MatDataTag | None:
    try:
        return _read_tag(reader, endian)
    except MatSliceReadError:
        return None


def _read_tag(reader: Any, endian: str) -> MatDataTag:
    raw = reader.read(4)
    first = struct.unpack(endian + "I", raw)[0]
    small_nbytes = first >> 16
    small_type = first & 0xFFFF
    if small_nbytes and small_nbytes <= 4 and small_type in _known_data_types():
        inline = reader.read(4)[:small_nbytes]
        return MatDataTag(data_type=small_type, nbytes=small_nbytes, inline_data=inline)
    nbytes = struct.unpack(endian + "I", reader.read(4))[0]
    return MatDataTag(data_type=first, nbytes=nbytes)


def _read_element_data(reader: Any, tag: MatDataTag) -> bytes:
    if tag.inline_data is not None:
        return tag.inline_data
    data = reader.read(tag.nbytes)
    _skip_padding(reader, tag.nbytes)
    return data


def _skip_tag_payload(reader: Any, tag: MatDataTag) -> None:
    if tag.inline_data is not None:
        return
    reader.skip(_padded_size(tag.nbytes))


def _skip_padding(reader: Any, nbytes: int) -> None:
    padding = _padded_size(nbytes) - nbytes
    if padding:
        reader.skip(padding)


def _known_data_types() -> set[int]:
    return {
        MI_INT8,
        MI_UINT8,
        MI_INT16,
        MI_UINT16,
        MI_INT32,
        MI_UINT32,
        MI_SINGLE,
        MI_DOUBLE,
        MI_INT64,
        MI_UINT64,
        MI_MATRIX,
        MI_COMPRESSED,
    }


def _decode_field_names(field_names_data: bytes, field_name_length: int) -> list[str]:
    if field_name_length <= 0:
        return []
    field_count = len(field_names_data) // field_name_length
    names = []
    for index in range(field_count):
        raw = field_names_data[index * field_name_length : (index + 1) * field_name_length]
        names.append(raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore"))
    return names


def _summary(
    name: str,
    path: str,
    array: np.ndarray,
    request: MatReadRequest,
) -> VariableSliceSummary:
    stats = _numeric_stats(array)
    return VariableSliceSummary(
        name=name,
        path=path,
        shape=[int(value) for value in array.shape],
        dtype=str(array.dtype),
        finite_ratio=stats["finite_ratio"],
        min=stats["min"],
        max=stats["max"],
        mean=stats["mean"],
        source_orientation=request.source_orientation,
        canonical_orientation_suggestion=request.canonical_orientation,
        warnings=[],
        errors=[],
    )


def _aggregate_summary(
    name: str,
    array: np.ndarray,
    component_summaries: list[VariableSliceSummary],
    orientation: list[str],
) -> VariableSliceSummary:
    stats = _numeric_stats(array)
    warnings = [warning for summary in component_summaries for warning in summary.warnings]
    errors = [error for summary in component_summaries for error in summary.errors]
    return VariableSliceSummary(
        name=name,
        path="; ".join(summary.path for summary in component_summaries[:8]),
        shape=[int(value) for value in array.shape],
        dtype=str(array.dtype),
        finite_ratio=stats["finite_ratio"],
        min=stats["min"],
        max=stats["max"],
        mean=stats["mean"],
        source_orientation=[],
        canonical_orientation_suggestion=orientation,
        warnings=warnings,
        errors=errors,
    )


def _numeric_stats(array: np.ndarray) -> dict[str, float | None]:
    if array.size == 0:
        return {"finite_ratio": None, "min": None, "max": None, "mean": None}
    finite = np.isfinite(array)
    finite_ratio = float(np.mean(finite))
    if not np.any(finite):
        return {"finite_ratio": finite_ratio, "min": None, "max": None, "mean": None}
    finite_values = array[finite]
    return {
        "finite_ratio": finite_ratio,
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
    }


def _result(
    mapping_path: Path | str,
    limits: SmallSliceLimits,
    depth_window: dict[str, Any] | None,
    source_files: dict[str, Any],
    summaries: dict[str, VariableSliceSummary],
    warnings: list[str],
    errors: list[str],
) -> SmallSliceResult:
    return SmallSliceResult(
        slice_version=SMALL_SLICE_VERSION,
        schema_version=SCHEMA_VERSION,
        data_version=DATA_VERSION,
        mapping_path=str(mapping_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        limits=asdict(limits),
        depth_window=depth_window,
        source_files=source_files,
        variables=summaries,
        warnings=warnings,
        errors=errors,
    )


def _validate_limits(limits: SmallSliceLimits) -> SmallSliceLimits:
    checks = {
        "max_depth_samples": (limits.max_depth_samples, MAX_DEPTH_LIMIT),
        "max_time_samples": (limits.max_time_samples, MAX_TIME_LIMIT),
        "max_receivers": (limits.max_receivers, MAX_RECEIVER_LIMIT),
        "max_sides": (limits.max_sides, MAX_SIDE_LIMIT),
        "max_cast_azimuth": (limits.max_cast_azimuth, MAX_CAST_AZIMUTH_LIMIT),
    }
    for name, (value, maximum) in checks.items():
        if value < 1 or value > maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}; observed {value}")
    return limits


def _select_single_depth_window(
    depth: np.ndarray,
    depth_window: DepthWindow,
    *,
    max_depth_samples: int,
    name: str,
) -> DepthWindowSelection:
    values = np.asarray(depth, dtype=np.float64).reshape(-1)
    warnings: list[str] = []
    finite = np.isfinite(values)
    if values.size == 0 or not np.any(finite):
        return DepthWindowSelection(
            source_start_index=0,
            sample_count=0,
            matched_count=0,
            observed_min=None,
            observed_max=None,
            warnings=[f"{name} has no finite depth samples for window selection."],
        )
    low = min(depth_window.depth_start, depth_window.depth_stop)
    high = max(depth_window.depth_start, depth_window.depth_stop)
    matched = np.flatnonzero(finite & (values >= low) & (values <= high))
    if matched.size == 0:
        finite_values = values[finite]
        center = (low + high) / 2.0
        nearest = int(np.flatnonzero(finite)[np.argmin(np.abs(finite_values - center))])
        return DepthWindowSelection(
            source_start_index=nearest,
            sample_count=0,
            matched_count=0,
            observed_min=None,
            observed_max=None,
            warnings=[f"{name} has no samples inside depth window [{low}, {high}]."],
        )
    center = (low + high) / 2.0
    center_index = int(matched[np.argmin(np.abs(values[matched] - center))])
    matched_start = int(matched[0])
    matched_stop_exclusive = int(matched[-1]) + 1
    desired_count = min(int(max_depth_samples), int(matched.size))
    half = desired_count // 2
    start = max(matched_start, center_index - half)
    stop = start + desired_count
    if stop > matched_stop_exclusive:
        stop = matched_stop_exclusive
        start = max(matched_start, stop - desired_count)
    observed = values[start:stop]
    finite_observed = observed[np.isfinite(observed)]
    return DepthWindowSelection(
        source_start_index=int(start),
        sample_count=int(stop - start),
        matched_count=int(matched.size),
        observed_min=float(np.min(finite_observed)) if finite_observed.size else None,
        observed_max=float(np.max(finite_observed)) if finite_observed.size else None,
        warnings=warnings,
    )


def _window_selection_warnings(window_selection: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for source in ["cast", "pose"]:
        selection = _as_dict(window_selection.get(source))
        warnings.extend(str(item) for item in selection.get("warnings", []) if item)
    for receiver, selection_value in _as_dict(window_selection.get("xsi_receivers")).items():
        selection = _as_dict(selection_value)
        warnings.extend(f"{receiver}: {item}" for item in selection.get("warnings", []) if item)
    return warnings


def _selection_for_source(
    window_selection: dict[str, Any] | None,
    source: str,
) -> dict[str, Any] | None:
    if window_selection is None:
        return None
    return _as_dict(window_selection.get(source))


def _selection_for_xsi_receiver(
    window_selection: dict[str, Any] | None,
    receiver_index: int,
) -> dict[str, Any] | None:
    if window_selection is None:
        return None
    return _as_dict(
        _as_dict(window_selection.get("xsi_receivers")).get(f"receiver_{receiver_index:02d}")
    )


def _selected_start_index(selection: dict[str, Any] | None) -> int:
    if selection is None:
        return 0
    return max(int(selection.get("source_start_index", 0)), 0)


def _selected_depth_count(selection: dict[str, Any] | None, fallback: int) -> int:
    if selection is None:
        return fallback
    return max(int(selection.get("sample_count", 0)), 0)


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _format_pattern(pattern: str, receiver: int, side: str | None = None) -> str:
    return pattern.format(receiver=receiver, side=side or "")


def _cast_azimuth_axis(cast_mapping: dict[str, Any], count: int) -> np.ndarray:
    start = float(cast_mapping.get("azimuth_start_deg", 0.0))
    step = float(cast_mapping.get("azimuth_step_deg", 360.0 / max(count, 1)))
    return (start + step * np.arange(count, dtype=np.float32)).astype(np.float32)


def _unit_unknown(mapping: dict[str, Any], path: tuple[str, str]) -> bool:
    section = _as_dict(mapping.get(path[0]))
    return str(section.get(path[1], "")).lower().startswith("unknown")


def _matrix_rows_cols(dims: list[int]) -> tuple[int, int]:
    if not dims:
        return 1, 1
    if len(dims) == 1:
        return int(dims[0]), 1
    return int(dims[0]), int(dims[1])


def _element_count(dims: list[int]) -> int:
    if not dims:
        return 1
    return int(math.prod(dims))


def _padded_size(nbytes: int) -> int:
    return ((nbytes + 7) // 8) * 8


def _endian_from_header(header: bytes) -> str:
    marker = header[126:128]
    if marker == b"IM":
        return "<"
    if marker == b"MI":
        return ">"
    raise MatSliceReadError("Unsupported MAT endian marker.")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(default or [])
