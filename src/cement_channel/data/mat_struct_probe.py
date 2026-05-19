from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.data.mat_metadata import infer_variable_role_hint

MAT_STRUCT_PROBE_VERSION = "mat_struct_probe_v001"
STRUCT_CLASS_NAMES = {"struct", "mat_struct"}


@dataclass(frozen=True)
class StructFieldProbe:
    top_variable: str
    field_path: str
    shape: list[int]
    dtype_or_class: str
    role_hint: str
    element_count: int | None
    preview_stats: dict[str, Any]


@dataclass(frozen=True)
class FileStructProbe:
    path: str
    filename: str
    file_role: str | None
    receiver_index: int | None
    can_probe: bool
    mat_format: str
    probed_variables: list[str]
    fields: list[StructFieldProbe]
    warnings: list[str]
    errors: list[str]


@dataclass(frozen=True)
class StructProbeResult:
    probe_version: str
    metadata_json_path: str
    manifest_path: str | None
    created_at: str
    summary: dict[str, int]
    files: list[FileStructProbe]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json_object(path: Path | str) -> dict[str, Any]:
    json_path = Path(path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    return data


def probe_structs_from_metadata(
    metadata: dict[str, Any],
    *,
    metadata_json_path: Path | str,
    manifest_path: Path | str | None = None,
    max_files: int = 3,
    max_variables_per_file: int = 3,
    max_field_depth: int = 2,
    max_array_elements_preview: int = 20,
) -> StructProbeResult:
    files = _metadata_files(metadata)
    warnings: list[str] = []
    if len(files) > max_files:
        warnings.append(f"max_files applied: probing {max_files} of {len(files)} files")
    selected_files = files[: max(max_files, 0)]
    file_results = [
        probe_file_structs(
            file_metadata,
            max_variables_per_file=max_variables_per_file,
            max_field_depth=max_field_depth,
            max_array_elements_preview=max_array_elements_preview,
        )
        for file_metadata in selected_files
    ]
    return StructProbeResult(
        probe_version=MAT_STRUCT_PROBE_VERSION,
        metadata_json_path=str(metadata_json_path),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        created_at=datetime.now(timezone.utc).isoformat(),
        summary=_build_summary(file_results),
        files=file_results,
        warnings=warnings,
    )


def probe_file_structs(
    file_metadata: dict[str, Any],
    *,
    max_variables_per_file: int = 3,
    max_field_depth: int = 2,
    max_array_elements_preview: int = 20,
) -> FileStructProbe:
    path = str(file_metadata.get("path", ""))
    variables = _metadata_variables(file_metadata)
    selected_variables = _select_top_level_variables(variables, max_variables_per_file)
    base = {
        "path": path,
        "filename": str(file_metadata.get("filename", Path(path).name)),
        "file_role": _optional_str(file_metadata.get("file_role")),
        "receiver_index": _optional_int(file_metadata.get("receiver_index")),
        "mat_format": str(file_metadata.get("mat_format", "unknown")),
        "probed_variables": [str(variable.get("name", "")) for variable in selected_variables],
    }

    if not file_metadata.get("can_open"):
        return FileStructProbe(
            **base,
            can_probe=False,
            fields=[],
            warnings=[],
            errors=[f"metadata marked file as not openable: {path}"],
        )
    if not selected_variables:
        return FileStructProbe(
            **base,
            can_probe=False,
            fields=[],
            warnings=["no top-level variables selected for struct probe"],
            errors=[],
        )
    if base["mat_format"] == "matlab_v7.3_hdf5":
        return FileStructProbe(
            **base,
            can_probe=False,
            fields=[],
            warnings=["MAT v7.3/HDF5 struct probe is not implemented in scipy loadmat path"],
            errors=[],
        )

    fields: list[StructFieldProbe] = []
    warnings: list[str] = []
    errors: list[str] = []
    for variable in selected_variables:
        variable_name = str(variable.get("name", ""))
        try:
            loaded = _load_single_variable(Path(path), variable_name)
        except Exception as exc:
            errors.append(
                f"{variable_name}: failed controlled loadmat: {type(exc).__name__}: {exc}"
            )
            continue
        if variable_name not in loaded:
            warnings.append(f"{variable_name}: variable not returned by loadmat")
            continue
        fields.extend(
            probe_loaded_variable(
                variable_name,
                loaded[variable_name],
                max_field_depth=max_field_depth,
                max_array_elements_preview=max_array_elements_preview,
            )
        )

    return FileStructProbe(
        **base,
        can_probe=not errors,
        fields=fields,
        warnings=warnings,
        errors=errors,
    )


def probe_loaded_variable(
    top_variable: str,
    value: Any,
    *,
    max_field_depth: int,
    max_array_elements_preview: int,
) -> list[StructFieldProbe]:
    probes: list[StructFieldProbe] = []
    _probe_value(
        top_variable=top_variable,
        field_path=top_variable,
        value=value,
        depth=0,
        max_field_depth=max_field_depth,
        max_array_elements_preview=max_array_elements_preview,
        probes=probes,
    )
    return probes


def role_hint_for_field_path(field_path: str, shape: list[int]) -> str:
    leaf_name = field_path.split(".")[-1]
    leaf_lower = leaf_name.lower()
    if "azimuth" in leaf_lower or leaf_lower in {"azi", "azim"}:
        return "cast_azimuth_candidate"
    if leaf_lower in {"time", "t", "timems", "time_ms", "tad"}:
        return "xsi_time_candidate"
    if "depth" in leaf_lower or leaf_lower in {"md", "tvd"}:
        return "depth_candidate"
    if leaf_lower.startswith("xsilmr") and shape == [1, 1]:
        return "unknown"
    inherited_hint = infer_variable_role_hint(leaf_name, shape)
    if inherited_hint != "unknown":
        return inherited_hint
    if "zc" in leaf_lower or "impedance" in leaf_lower:
        return "cast_zc_candidate"
    if "wave" in leaf_lower or leaf_lower == "data":
        return "xsi_waveform_candidate"
    return "unknown"


def format_struct_probe_report(result: StructProbeResult) -> str:
    data = result.to_dict()
    lines = [
        "# MAT Struct Probe Report",
        "",
        f"- Metadata JSON: {data['metadata_json_path']}",
        f"- Manifest: {data['manifest_path']}",
        f"- Generated at: {data['created_at']}",
        "",
        "## Summary",
        "",
    ]
    for key, value in data["summary"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    if data["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in data["warnings"]])
    lines.extend(["", "## Files", ""])
    for file_probe in data["files"]:
        lines.extend(
            [
                f"### {file_probe['filename']}",
                "",
                f"- Role: {file_probe['file_role']}",
                f"- Receiver index: {file_probe['receiver_index']}",
                f"- Can probe: {file_probe['can_probe']}",
                f"- Probed variables: {file_probe['probed_variables']}",
                f"- Field count: {len(file_probe['fields'])}",
            ]
        )
        if file_probe["errors"]:
            lines.append("- Errors: " + "; ".join(file_probe["errors"]))
        if file_probe["warnings"]:
            lines.append("- Warnings: " + "; ".join(file_probe["warnings"]))
        for field in file_probe["fields"][:20]:
            lines.append(
                "- "
                f"{field['field_path']} "
                f"shape={field['shape']} "
                f"dtype={field['dtype_or_class']} "
                f"role_hint={field['role_hint']}"
            )
        lines.append("")
    return "\n".join(lines)


def _load_single_variable(path: Path, variable_name: str) -> dict[str, Any]:
    from scipy.io import loadmat

    return loadmat(
        str(path),
        variable_names=[variable_name],
        struct_as_record=False,
        squeeze_me=False,
    )


def _probe_value(
    *,
    top_variable: str,
    field_path: str,
    value: Any,
    depth: int,
    max_field_depth: int,
    max_array_elements_preview: int,
    probes: list[StructFieldProbe],
) -> None:
    probes.append(_field_probe(top_variable, field_path, value, max_array_elements_preview))
    if depth >= max_field_depth:
        return

    for child_name, child_value in _iter_struct_children(value):
        _probe_value(
            top_variable=top_variable,
            field_path=f"{field_path}.{child_name}",
            value=child_value,
            depth=depth + 1,
            max_field_depth=max_field_depth,
            max_array_elements_preview=max_array_elements_preview,
            probes=probes,
        )


def _field_probe(
    top_variable: str,
    field_path: str,
    value: Any,
    max_array_elements_preview: int,
) -> StructFieldProbe:
    shape = _shape(value)
    dtype_or_class = _dtype_or_class(value)
    return StructFieldProbe(
        top_variable=top_variable,
        field_path=field_path,
        shape=shape,
        dtype_or_class=dtype_or_class,
        role_hint=role_hint_for_field_path(field_path, shape),
        element_count=_element_count(shape),
        preview_stats=_preview_stats(value, max_array_elements_preview),
    )


def _iter_struct_children(value: Any) -> list[tuple[str, Any]]:
    if _is_mat_struct(value):
        return [(name, getattr(value, name)) for name in getattr(value, "_fieldnames", [])]

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            first = _first_array_item(value)
            if first is None:
                return []
            return [(name, first[name]) for name in value.dtype.names]
        first = _first_array_item(value)
        if _is_mat_struct(first):
            return [(name, getattr(first, name)) for name in getattr(first, "_fieldnames", [])]
    return []


def _is_mat_struct(value: Any) -> bool:
    return hasattr(value, "_fieldnames")


def _first_array_item(array: np.ndarray) -> Any:
    if array.size == 0:
        return None
    return array.flat[0]


def _shape(value: Any) -> list[int]:
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.shape]
    return []


def _dtype_or_class(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.dtype == object and value.size and _is_mat_struct(value.flat[0]):
            return "struct"
        if value.dtype.names:
            return "struct"
        return str(value.dtype)
    if _is_mat_struct(value):
        return "struct"
    return type(value).__name__


def _element_count(shape: list[int]) -> int | None:
    if not shape:
        return None
    count = 1
    for item in shape:
        count *= item
    return count


def _preview_stats(value: Any, max_array_elements_preview: int) -> dict[str, Any]:
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        return {}
    preview = value.ravel()[: max(max_array_elements_preview, 0)]
    finite_preview = preview[np.isfinite(preview)]
    if finite_preview.size == 0:
        return {"preview_count": int(preview.size), "finite_count": 0}
    return {
        "preview_count": int(preview.size),
        "finite_count": int(finite_preview.size),
        "min": float(np.min(finite_preview)),
        "max": float(np.max(finite_preview)),
        "mean": float(np.mean(finite_preview)),
    }


def _metadata_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    files = metadata.get("files")
    if not isinstance(files, list):
        return []
    return [item for item in files if isinstance(item, dict)]


def _metadata_variables(file_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    variables = file_metadata.get("variables")
    if not isinstance(variables, list):
        return []
    return [item for item in variables if isinstance(item, dict)]


def _select_top_level_variables(
    variables: list[dict[str, Any]],
    max_variables_per_file: int,
) -> list[dict[str, Any]]:
    def priority(variable: dict[str, Any]) -> tuple[int, str]:
        dtype_or_class = str(variable.get("dtype_or_class", "")).lower()
        shape = variable.get("shape")
        is_struct_like = dtype_or_class in STRUCT_CLASS_NAMES or shape == [1, 1]
        return (0 if is_struct_like else 1, str(variable.get("name", "")))

    return sorted(variables, key=priority)[: max(max_variables_per_file, 0)]


def _build_summary(files: list[FileStructProbe]) -> dict[str, int]:
    return {
        "file_count": len(files),
        "can_probe_count": sum(1 for file_probe in files if file_probe.can_probe),
        "file_error_count": sum(1 for file_probe in files if file_probe.errors),
        "field_count": sum(len(file_probe.fields) for file_probe in files),
    }


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
