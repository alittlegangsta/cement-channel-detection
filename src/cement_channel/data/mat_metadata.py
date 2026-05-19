from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAT_METADATA_VERSION = "mat_metadata_v001"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
HDF5_SIGNATURE_OFFSETS = (0, 512, 1024, 2048)

NUMERIC_MATLAB_CLASSES = {
    "double",
    "single",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "logical",
}

ROLE_HINT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class MatVariableMetadata:
    name: str
    shape: list[int]
    dtype_or_class: str
    is_numeric: bool
    element_count: int | None
    role_hint: str = ROLE_HINT_UNKNOWN


@dataclass(frozen=True)
class MatFileMetadata:
    path: str
    filename: str
    file_role: str | None
    receiver_index: int | None
    can_open: bool
    mat_format: str
    variables: list[MatVariableMetadata]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_mat_file(
    path: Path | str,
    *,
    file_role: str | None = None,
    receiver_index: int | None = None,
) -> MatFileMetadata:
    mat_path = Path(path)
    base = _empty_file_metadata(
        mat_path,
        file_role=file_role,
        receiver_index=receiver_index,
        mat_format="missing",
    )
    if not mat_path.exists():
        return _replace_file_metadata(
            base,
            errors=[f"file does not exist: {mat_path}"],
        )
    if not mat_path.is_file():
        return _replace_file_metadata(
            base,
            mat_format="not_a_file",
            errors=[f"path is not a file: {mat_path}"],
        )

    mat_format = detect_mat_format(mat_path)
    if mat_format == "matlab_v7.3_hdf5":
        return _inspect_hdf5_metadata(
            mat_path,
            file_role=file_role,
            receiver_index=receiver_index,
            mat_format=mat_format,
        )

    return _inspect_scipy_whosmat(
        mat_path,
        file_role=file_role,
        receiver_index=receiver_index,
        mat_format=mat_format,
    )


def inspect_manifest_mat_metadata(
    manifest: dict[str, Any],
    *,
    max_files: int | None = None,
) -> dict[str, Any]:
    file_records = _manifest_file_records(manifest)
    warnings: list[str] = []
    if max_files is not None and len(file_records) > max_files:
        warnings.append(f"max_files applied: inspecting {max_files} of {len(file_records)} files")
        file_records = file_records[: max(max_files, 0)]

    files = [
        inspect_mat_file(
            file_record["path"],
            file_role=file_record.get("file_role"),
            receiver_index=_optional_int(file_record.get("receiver_index")),
        )
        for file_record in file_records
        if file_record.get("path")
    ]

    return {
        "metadata_version": MAT_METADATA_VERSION,
        "schema_version": manifest.get("schema_version"),
        "data_version": manifest.get("data_version"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_version": manifest.get("manifest_version"),
        "summary": _build_metadata_summary(files),
        "files": [file_metadata.to_dict() for file_metadata in files],
        "warnings": warnings,
    }


def detect_mat_format(path: Path | str) -> str:
    mat_path = Path(path)
    try:
        with mat_path.open("rb") as mat_file:
            header = mat_file.read(4096)
    except OSError:
        return "unknown"

    if _contains_hdf5_signature(header) or b"MATLAB 7.3 MAT-file" in header:
        return "matlab_v7.3_hdf5"
    if b"MATLAB" in header[:128]:
        return "matlab_v5_or_v7"
    return "unknown"


def infer_variable_role_hint(name: str, shape: list[int] | tuple[int, ...]) -> str:
    normalized_name = _normalize_variable_name(name)
    shape_values = [int(value) for value in shape]

    if _is_relbearing_name(normalized_name):
        return "relbearing_candidate"
    if _is_inclination_name(normalized_name):
        return "inclination_candidate"
    if _is_depth_name(normalized_name):
        return "depth_candidate"
    if _is_cast_zc_name(normalized_name, shape_values):
        return "cast_zc_candidate"
    if _is_xsi_waveform_name(normalized_name, shape_values):
        return "xsi_waveform_candidate"
    return ROLE_HINT_UNKNOWN


def _inspect_scipy_whosmat(
    mat_path: Path,
    *,
    file_role: str | None,
    receiver_index: int | None,
    mat_format: str,
) -> MatFileMetadata:
    try:
        from scipy.io import whosmat
    except ModuleNotFoundError:
        return _empty_file_metadata(
            mat_path,
            file_role=file_role,
            receiver_index=receiver_index,
            mat_format=mat_format,
            warnings=["scipy is not available; cannot inspect MATLAB v5/v7 metadata"],
            errors=["unable to read MAT metadata without scipy.io.whosmat"],
        )

    try:
        entries = whosmat(str(mat_path))
    except NotImplementedError as exc:
        return _empty_file_metadata(
            mat_path,
            file_role=file_role,
            receiver_index=receiver_index,
            mat_format="invalid_or_unsupported_mat",
            warnings=["scipy.io.whosmat cannot inspect this MAT file format"],
            errors=[f"failed to read MAT metadata: {type(exc).__name__}: {exc}"],
        )
    except Exception as exc:
        return _empty_file_metadata(
            mat_path,
            file_role=file_role,
            receiver_index=receiver_index,
            mat_format="invalid_or_unsupported_mat",
            errors=[f"failed to read MAT metadata: {type(exc).__name__}: {exc}"],
        )

    variables = [
        _variable_from_whosmat_entry(name, shape, matlab_class)
        for name, shape, matlab_class in entries
    ]
    return MatFileMetadata(
        path=str(mat_path.resolve()),
        filename=mat_path.name,
        file_role=file_role,
        receiver_index=receiver_index,
        can_open=True,
        mat_format="matlab_v5_or_v7" if mat_format == "unknown" else mat_format,
        variables=variables,
        warnings=[],
        errors=[],
    )


def _inspect_hdf5_metadata(
    mat_path: Path,
    *,
    file_role: str | None,
    receiver_index: int | None,
    mat_format: str,
) -> MatFileMetadata:
    try:
        import h5py
    except ModuleNotFoundError:
        return _empty_file_metadata(
            mat_path,
            file_role=file_role,
            receiver_index=receiver_index,
            mat_format=mat_format,
            warnings=["h5py is not available; cannot inspect MATLAB v7.3/HDF5 metadata"],
        )

    variables: list[MatVariableMetadata] = []
    try:
        with h5py.File(mat_path, "r") as h5_file:
            for name, item in h5_file.items():
                variables.append(_variable_from_hdf5_item(name, item))
    except OSError as exc:
        return _empty_file_metadata(
            mat_path,
            file_role=file_role,
            receiver_index=receiver_index,
            mat_format=mat_format,
            errors=[f"failed to read HDF5 MAT metadata: {exc}"],
        )

    return MatFileMetadata(
        path=str(mat_path.resolve()),
        filename=mat_path.name,
        file_role=file_role,
        receiver_index=receiver_index,
        can_open=True,
        mat_format=mat_format,
        variables=variables,
        warnings=[],
        errors=[],
    )


def _variable_from_whosmat_entry(
    name: str,
    shape: tuple[int, ...],
    matlab_class: str,
) -> MatVariableMetadata:
    shape_values = [int(value) for value in shape]
    dtype_or_class = str(matlab_class)
    return MatVariableMetadata(
        name=name,
        shape=shape_values,
        dtype_or_class=dtype_or_class,
        is_numeric=dtype_or_class.lower() in NUMERIC_MATLAB_CLASSES,
        element_count=_element_count(shape_values),
        role_hint=infer_variable_role_hint(name, shape_values),
    )


def _variable_from_hdf5_item(name: str, item: Any) -> MatVariableMetadata:
    shape = [int(value) for value in getattr(item, "shape", [])]
    dtype = getattr(item, "dtype", None)
    dtype_or_class = str(dtype) if dtype is not None else "group"
    dtype_kind = getattr(dtype, "kind", "")
    is_numeric = dtype_kind in {"b", "i", "u", "f", "c"}
    return MatVariableMetadata(
        name=name,
        shape=shape,
        dtype_or_class=dtype_or_class,
        is_numeric=is_numeric,
        element_count=_element_count(shape) if dtype is not None else None,
        role_hint=infer_variable_role_hint(name, shape),
    )


def _manifest_file_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if isinstance(files, list):
        return [file_record for file_record in files if isinstance(file_record, dict)]

    records: list[dict[str, Any]] = []
    wells = manifest.get("wells")
    if isinstance(wells, list):
        for well in wells:
            if not isinstance(well, dict):
                continue
            well_files = well.get("files")
            if isinstance(well_files, list):
                records.extend(
                    file_record for file_record in well_files if isinstance(file_record, dict)
                )
    return records


def _build_metadata_summary(files: list[MatFileMetadata]) -> dict[str, int]:
    return {
        "file_count": len(files),
        "can_open_count": sum(1 for file_metadata in files if file_metadata.can_open),
        "error_file_count": sum(1 for file_metadata in files if file_metadata.errors),
        "warning_file_count": sum(1 for file_metadata in files if file_metadata.warnings),
        "variable_count": sum(len(file_metadata.variables) for file_metadata in files),
    }


def _contains_hdf5_signature(header: bytes) -> bool:
    return any(
        header[offset : offset + len(HDF5_SIGNATURE)] == HDF5_SIGNATURE
        for offset in HDF5_SIGNATURE_OFFSETS
    )


def _normalize_variable_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_relbearing_name(normalized_name: str) -> bool:
    return "relbearing" in normalized_name or "relativebearing" in normalized_name


def _is_inclination_name(normalized_name: str) -> bool:
    return normalized_name in {"inc", "incl", "inclination"} or "inclination" in normalized_name


def _is_depth_name(normalized_name: str) -> bool:
    return normalized_name in {"depth", "md", "tvd"} or normalized_name.endswith("depth")


def _is_cast_zc_name(normalized_name: str, shape: list[int]) -> bool:
    if normalized_name in {"zc", "castzc"} or normalized_name.endswith("zc"):
        return True
    if "impedance" in normalized_name:
        return True
    return "cast" in normalized_name and 180 in shape


def _is_xsi_waveform_name(normalized_name: str, shape: list[int]) -> bool:
    if any(token in normalized_name for token in ("waveform", "wave", "xsilmr")):
        return True
    if "xsi" in normalized_name and "relbearing" not in normalized_name:
        return True
    return len(shape) >= 2 and (1024 in shape or 8 in shape and 13 in shape)


def _element_count(shape: list[int]) -> int:
    count = 1
    for value in shape:
        count *= int(value)
    return count


def _empty_file_metadata(
    path: Path,
    *,
    file_role: str | None,
    receiver_index: int | None,
    mat_format: str,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> MatFileMetadata:
    return MatFileMetadata(
        path=str(path),
        filename=path.name,
        file_role=file_role,
        receiver_index=receiver_index,
        can_open=False,
        mat_format=mat_format,
        variables=[],
        warnings=warnings or [],
        errors=errors or [],
    )


def _replace_file_metadata(
    metadata: MatFileMetadata,
    *,
    mat_format: str | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> MatFileMetadata:
    return MatFileMetadata(
        path=metadata.path,
        filename=metadata.filename,
        file_role=metadata.file_role,
        receiver_index=metadata.receiver_index,
        can_open=False,
        mat_format=mat_format or metadata.mat_format,
        variables=[],
        warnings=warnings or metadata.warnings,
        errors=errors or metadata.errors,
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
