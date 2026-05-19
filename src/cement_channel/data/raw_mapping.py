from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_METADATA_AUDIT_VERSION = "raw_metadata_audit_v001"
RAW_VARIABLE_MAPPING_VERSION = "raw_variable_mapping_v001"
EXPECTED_XSI_RECEIVER_COUNT = 13

CANDIDATE_KEYS = [
    "cast_zc_candidates",
    "cast_depth_candidates",
    "pose_depth_candidates",
    "inclination_candidates",
    "relbearing_candidates",
    "xsi_waveform_candidates",
    "xsi_depth_candidates",
    "xsi_time_candidates",
]


@dataclass(frozen=True)
class VariableCandidate:
    file_path: str
    file_role: str | None
    receiver_index: int | None
    variable_name: str
    shape: list[int]
    dtype_or_class: str
    role_hint: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class FileAuditSummary:
    path: str
    filename: str
    file_role: str | None
    receiver_index: int | None
    can_open: bool
    mat_format: str
    variables_count: int
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class RawMetadataAuditResult:
    audit_version: str
    metadata_json_path: str
    generated_at: str
    status: str
    statistics: dict[str, int]
    file_summaries: list[FileAuditSummary]
    candidates: dict[str, list[VariableCandidate]]
    cast_warnings: list[str]
    pose_warnings: list[str]
    xsi_warnings: list[str]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_mat_metadata_json(path: Path | str) -> dict[str, Any]:
    metadata_path = Path(path)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MAT metadata JSON must contain an object: {metadata_path}")
    return data


def audit_raw_metadata(
    metadata: dict[str, Any],
    *,
    metadata_json_path: Path | str,
) -> RawMetadataAuditResult:
    files = _metadata_files(metadata)
    file_summaries = [_file_summary(file_metadata) for file_metadata in files]
    candidates = _collect_candidates(files)
    cast_warnings = _audit_cast(candidates, file_summaries)
    pose_warnings = _audit_pose(candidates, file_summaries)
    xsi_warnings = _audit_xsi_receivers(files, candidates)
    warnings = _collect_global_warnings(metadata, file_summaries)
    errors = [] if files else ["mat_metadata JSON contains no file entries"]
    statistics = _build_statistics(file_summaries)
    all_warnings = warnings + cast_warnings + pose_warnings + xsi_warnings
    status = _overall_status(errors, all_warnings)

    return RawMetadataAuditResult(
        audit_version=RAW_METADATA_AUDIT_VERSION,
        metadata_json_path=str(metadata_json_path),
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        statistics=statistics,
        file_summaries=file_summaries,
        candidates=candidates,
        cast_warnings=cast_warnings,
        pose_warnings=pose_warnings,
        xsi_warnings=xsi_warnings,
        warnings=warnings,
        errors=errors,
    )


def format_raw_metadata_report(result: RawMetadataAuditResult) -> str:
    data = result.to_dict()
    lines = [
        "# Raw MAT Metadata Audit Report",
        "",
        f"- Input metadata-json: {data['metadata_json_path']}",
        f"- Generated at: {data['generated_at']}",
        f"- Overall status: {data['status']}",
        "",
        "## File Statistics",
        "",
    ]
    for key, value in data["statistics"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")

    candidate_groups = data["candidates"]
    lines.extend(["", "## CAST Audit", ""])
    lines.extend(
        _candidate_lines("Candidate Zc variables", candidate_groups["cast_zc_candidates"])
    )
    lines.extend(
        _candidate_lines(
            "Candidate depth variables",
            candidate_groups["cast_depth_candidates"],
        )
    )
    lines.extend(_cast_shape_lines(candidate_groups["cast_zc_candidates"]))
    lines.extend(_warning_lines(data["cast_warnings"]))

    lines.extend(["", "## Pose Audit", ""])
    lines.extend(
        _candidate_lines(
            "Candidate Inc / Inclination variables",
            candidate_groups["inclination_candidates"],
        )
    )
    lines.extend(
        _candidate_lines(
            "Candidate RelBearing variables",
            candidate_groups["relbearing_candidates"],
        )
    )
    lines.extend(
        _candidate_lines(
            "Candidate depth variables",
            candidate_groups["pose_depth_candidates"],
        )
    )
    lines.extend(_warning_lines(data["pose_warnings"]))

    lines.extend(["", "## XSI Audit", ""])
    lines.append(f"- Receiver files: {data['statistics']['xsi_receiver_files']}")
    lines.extend(
        _candidate_lines("Waveform candidates", candidate_groups["xsi_waveform_candidates"])
    )
    lines.extend(_candidate_lines("Depth candidates", candidate_groups["xsi_depth_candidates"]))
    lines.extend(_candidate_lines("Time candidates", candidate_groups["xsi_time_candidates"]))
    lines.extend(_warning_lines(data["xsi_warnings"]))

    lines.extend(["", "## Recommended Human Review", ""])
    lines.extend(
        [
            "- Confirm variable names before controlled small-slice MAT reading.",
            "- Confirm whether CAST Zc shape is depth x azimuth or azimuth x depth.",
            "- Confirm whether XSI receiver waveform shape is depth x time or time x depth.",
            "- Confirm depth units and time units.",
        ]
    )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(
        [
            "- Manually confirm configs/raw_variable_mapping.yaml.",
            "- Then proceed to MVP-1 Step 5: controlled small-slice MAT reader.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def format_mapping_template(result: RawMetadataAuditResult, *, well_id: str) -> str:
    data = result.to_dict()
    cast_file = _first_file_name(data["file_summaries"], "cast", "CAST.mat")
    pose_file = _first_file_name(
        data["file_summaries"],
        "pose",
        "D2_XSI_RelBearing_Inclination.mat",
    )
    lines = [
        "schema_version: schema_v001",
        f"mapping_version: {RAW_VARIABLE_MAPPING_VERSION}",
        f"well_id: {_yaml_scalar(well_id)}",
        "status: draft_requires_human_review",
        "",
        "cast:",
        f"  file: {_yaml_scalar(cast_file)}",
        "  zc_variable: TODO_CONFIRM",
        "  depth_variable: TODO_CONFIRM",
        "  azimuth_variable: TODO_CONFIRM",
        "  candidates:",
        "    zc:",
    ]
    lines.extend(_yaml_candidate_items(data["candidates"]["cast_zc_candidates"], indent=6))
    lines.append("    depth:")
    lines.extend(_yaml_candidate_items(data["candidates"]["cast_depth_candidates"], indent=6))
    lines.append("    azimuth: []")
    lines.extend(
        [
            "",
            "pose:",
            f"  file: {_yaml_scalar(pose_file)}",
            "  depth_variable: TODO_CONFIRM",
            "  inclination_variable: TODO_CONFIRM",
            "  relbearing_variable: TODO_CONFIRM",
            "  candidates:",
            "    depth:",
        ]
    )
    lines.extend(_yaml_candidate_items(data["candidates"]["pose_depth_candidates"], indent=6))
    lines.append("    inclination:")
    lines.extend(_yaml_candidate_items(data["candidates"]["inclination_candidates"], indent=6))
    lines.append("    relbearing:")
    lines.extend(_yaml_candidate_items(data["candidates"]["relbearing_candidates"], indent=6))
    lines.extend(
        [
            "",
            "xsi:",
            "  receiver_dir: XSILMR",
            "  expected_receiver_files: 13",
            "  receiver_file_pattern: XSILMR*.mat",
            "  waveform_variable: TODO_CONFIRM",
            "  depth_variable: TODO_CONFIRM",
            "  time_variable: TODO_CONFIRM",
            "  receiver_index_source: filename",
            "  candidates:",
            "    waveform:",
        ]
    )
    lines.extend(_yaml_candidate_items(data["candidates"]["xsi_waveform_candidates"], indent=6))
    lines.append("    depth:")
    lines.extend(_yaml_candidate_items(data["candidates"]["xsi_depth_candidates"], indent=6))
    lines.append("    time:")
    lines.extend(_yaml_candidate_items(data["candidates"]["xsi_time_candidates"], indent=6))
    lines.extend(
        [
            "",
            "human_review:",
            "  required: true",
            "  notes:",
            "    - Confirm variable names before controlled small-slice MAT reading.",
            "    - Confirm whether CAST Zc shape is depth x azimuth or azimuth x depth.",
            "    - Confirm whether XSI receiver waveform shape is depth x time or time x depth.",
            "    - Confirm depth units and time units.",
            "",
        ]
    )
    return "\n".join(lines)


def _metadata_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    files = metadata.get("files")
    if not isinstance(files, list):
        return []
    return [file_metadata for file_metadata in files if isinstance(file_metadata, dict)]


def _file_summary(file_metadata: dict[str, Any]) -> FileAuditSummary:
    variables = file_metadata.get("variables")
    warnings = file_metadata.get("warnings")
    errors = file_metadata.get("errors")
    return FileAuditSummary(
        path=str(file_metadata.get("path", "")),
        filename=str(file_metadata.get("filename", "")),
        file_role=file_metadata.get("file_role"),
        receiver_index=_optional_int(file_metadata.get("receiver_index")),
        can_open=bool(file_metadata.get("can_open")),
        mat_format=str(file_metadata.get("mat_format", "unknown")),
        variables_count=len(variables) if isinstance(variables, list) else 0,
        errors=[str(error) for error in errors] if isinstance(errors, list) else [],
        warnings=[str(warning) for warning in warnings] if isinstance(warnings, list) else [],
    )


def _collect_candidates(files: list[dict[str, Any]]) -> dict[str, list[VariableCandidate]]:
    candidates: dict[str, list[VariableCandidate]] = {key: [] for key in CANDIDATE_KEYS}
    for file_metadata in files:
        variables = file_metadata.get("variables")
        if not file_metadata.get("can_open") or not isinstance(variables, list):
            continue
        for variable in variables:
            if not isinstance(variable, dict):
                continue
            for key, candidate in _candidates_for_variable(file_metadata, variable):
                candidates[key].append(candidate)
    return candidates


def _candidates_for_variable(
    file_metadata: dict[str, Any],
    variable: dict[str, Any],
) -> list[tuple[str, VariableCandidate]]:
    file_role = _optional_str(file_metadata.get("file_role"))
    name = str(variable.get("name", ""))
    shape = _shape_list(variable.get("shape"))
    role_hint = str(variable.get("role_hint", "unknown"))
    normalized = _normalize_name(name)
    items: list[tuple[str, VariableCandidate]] = []

    if file_role == "cast":
        if _is_cast_zc_candidate(normalized, shape, role_hint):
            items.append(
                (
                    "cast_zc_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.9,
                        _reason(role_hint, "CAST Zc heuristic"),
                    ),
                )
            )
        if _is_depth_candidate(normalized, role_hint):
            items.append(
                (
                    "cast_depth_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.9,
                        _reason(role_hint, "CAST depth heuristic"),
                    ),
                )
            )
    elif file_role == "pose":
        if _is_depth_candidate(normalized, role_hint):
            items.append(
                (
                    "pose_depth_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.9,
                        _reason(role_hint, "pose depth heuristic"),
                    ),
                )
            )
        if _is_inclination_candidate(normalized, role_hint):
            items.append(
                (
                    "inclination_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.95,
                        _reason(role_hint, "inclination heuristic"),
                    ),
                )
            )
        if _is_relbearing_candidate(normalized, role_hint):
            items.append(
                (
                    "relbearing_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.95,
                        _reason(role_hint, "RelBearing heuristic"),
                    ),
                )
            )
    elif file_role == "xsi_receiver":
        if _is_xsi_waveform_candidate(normalized, shape, role_hint):
            items.append(
                (
                    "xsi_waveform_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.85,
                        _reason(role_hint, "XSI waveform heuristic"),
                    ),
                )
            )
        if _is_depth_candidate(normalized, role_hint):
            items.append(
                (
                    "xsi_depth_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.85,
                        _reason(role_hint, "XSI depth heuristic"),
                    ),
                )
            )
        if _is_time_candidate(normalized, shape):
            items.append(
                (
                    "xsi_time_candidates",
                    _candidate(
                        file_metadata,
                        variable,
                        0.75,
                        "variable name/shape suggests time axis",
                    ),
                )
            )
    return items


def _candidate(
    file_metadata: dict[str, Any],
    variable: dict[str, Any],
    confidence: float,
    reason: str,
) -> VariableCandidate:
    return VariableCandidate(
        file_path=str(file_metadata.get("path", "")),
        file_role=_optional_str(file_metadata.get("file_role")),
        receiver_index=_optional_int(file_metadata.get("receiver_index")),
        variable_name=str(variable.get("name", "")),
        shape=_shape_list(variable.get("shape")),
        dtype_or_class=str(variable.get("dtype_or_class", "")),
        role_hint=str(variable.get("role_hint", "unknown")),
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
    )


def _audit_cast(
    candidates: dict[str, list[VariableCandidate]],
    file_summaries: list[FileAuditSummary],
) -> list[str]:
    warnings: list[str] = []
    if not any(file_summary.file_role == "cast" for file_summary in file_summaries):
        warnings.append("No CAST file metadata found.")
        return warnings
    if not candidates["cast_zc_candidates"]:
        warnings.append("No CAST Zc candidate variable found.")
    if not candidates["cast_depth_candidates"]:
        warnings.append("No CAST depth candidate variable found.")
    for candidate in candidates["cast_zc_candidates"]:
        if _cast_shape_orientation(candidate.shape) == "unknown":
            warnings.append(
                "Unable to infer CAST Zc orientation for "
                f"{candidate.variable_name} shape {candidate.shape}."
            )
    return warnings


def _audit_pose(
    candidates: dict[str, list[VariableCandidate]],
    file_summaries: list[FileAuditSummary],
) -> list[str]:
    warnings: list[str] = []
    if not any(file_summary.file_role == "pose" for file_summary in file_summaries):
        warnings.append("No pose file metadata found.")
        return warnings
    if not candidates["inclination_candidates"]:
        warnings.append("No Inc / Inclination candidate variable found.")
    if not candidates["relbearing_candidates"]:
        warnings.append("No RelBearing candidate variable found.")
    if not candidates["pose_depth_candidates"]:
        warnings.append("No pose depth candidate variable found.")
    return warnings


def _audit_xsi_receivers(
    files: list[dict[str, Any]],
    candidates: dict[str, list[VariableCandidate]],
) -> list[str]:
    warnings: list[str] = []
    receiver_files = [
        file_metadata
        for file_metadata in files
        if file_metadata.get("file_role") == "xsi_receiver"
    ]
    receiver_indexes = sorted(
        index
        for index in (
            _optional_int(file_metadata.get("receiver_index"))
            for file_metadata in receiver_files
        )
        if index is not None
    )
    expected_indexes = list(range(1, EXPECTED_XSI_RECEIVER_COUNT + 1))
    if receiver_indexes != expected_indexes:
        warnings.append(
            "XSI receiver indexes are incomplete or out of order: "
            f"expected {expected_indexes}, observed {receiver_indexes}."
        )
    if not candidates["xsi_waveform_candidates"]:
        warnings.append("No XSI waveform candidate variable found.")

    open_receiver_files = [
        file_metadata
        for file_metadata in receiver_files
        if file_metadata.get("can_open")
    ]
    variable_name_signatures = {
        tuple(sorted(_variable_names(file_metadata))) for file_metadata in open_receiver_files
    }
    if len(variable_name_signatures) > 1:
        warnings.append("XSI receiver files do not share a consistent top-level variable name set.")

    shape_signatures: dict[str, set[tuple[int, ...]]] = {}
    for file_metadata in open_receiver_files:
        for variable in _variables(file_metadata):
            name = str(variable.get("name", ""))
            shape_signatures.setdefault(name, set()).add(
                tuple(_shape_list(variable.get("shape")))
            )
    inconsistent_shapes = sorted(
        name for name, shapes in shape_signatures.items() if len(shapes) > 1
    )
    if inconsistent_shapes:
        warnings.append(
            "XSI receiver files have inconsistent shapes for variables: "
            + ", ".join(inconsistent_shapes)
            + "."
        )
    return warnings


def _collect_global_warnings(
    metadata: dict[str, Any],
    file_summaries: list[FileAuditSummary],
) -> list[str]:
    warnings = [str(warning) for warning in metadata.get("warnings", []) if warning]
    files_with_errors = [
        file_summary.filename for file_summary in file_summaries if file_summary.errors
    ]
    if files_with_errors:
        warnings.append(
            "Some MAT files reported metadata errors: "
            + ", ".join(files_with_errors)
            + "."
        )
    return warnings


def _build_statistics(file_summaries: list[FileAuditSummary]) -> dict[str, int]:
    return {
        "total_files": len(file_summaries),
        "cast_files": sum(
            1 for file_summary in file_summaries if file_summary.file_role == "cast"
        ),
        "pose_files": sum(
            1 for file_summary in file_summaries if file_summary.file_role == "pose"
        ),
        "xsi_receiver_files": sum(
            1 for file_summary in file_summaries if file_summary.file_role == "xsi_receiver"
        ),
        "files_with_errors": sum(1 for file_summary in file_summaries if file_summary.errors),
        "files_with_warnings": sum(
            1 for file_summary in file_summaries if file_summary.warnings
        ),
    }


def _overall_status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "fail"
    if warnings:
        return "warning"
    return "pass"


def _is_cast_zc_candidate(normalized: str, shape: list[int], role_hint: str) -> bool:
    return (
        role_hint == "cast_zc_candidate"
        or normalized in {"zc", "castzc"}
        or normalized.endswith("zc")
        or "impedance" in normalized
        or ("cast" in normalized and 180 in shape)
    )


def _is_depth_candidate(normalized: str, role_hint: str) -> bool:
    return (
        role_hint == "depth_candidate"
        or normalized in {"depth", "md", "tvd"}
        or normalized.endswith("depth")
    )


def _is_inclination_candidate(normalized: str, role_hint: str) -> bool:
    return (
        role_hint == "inclination_candidate"
        or normalized in {"inc", "incl", "inclination"}
        or "inclination" in normalized
    )


def _is_relbearing_candidate(normalized: str, role_hint: str) -> bool:
    return (
        role_hint == "relbearing_candidate"
        or "relbearing" in normalized
        or "relativebearing" in normalized
    )


def _is_xsi_waveform_candidate(normalized: str, shape: list[int], role_hint: str) -> bool:
    return (
        role_hint == "xsi_waveform_candidate"
        or normalized in {"data", "waveform", "wave"}
        or "waveform" in normalized
        or "xsi" in normalized
        or "xsilmr" in normalized
        or len(shape) >= 2
        and (1024 in shape or 8 in shape)
    )


def _is_time_candidate(normalized: str, shape: list[int]) -> bool:
    return (
        normalized in {"time", "t", "timems", "timeaxis"}
        or normalized.endswith("time")
        or shape == [1024]
        or shape == [1, 1024]
    )


def _cast_shape_orientation(shape: list[int]) -> str:
    if len(shape) < 2 or 180 not in shape:
        return "unknown"
    if shape[-1] == 180:
        return "possibly_depth_x_azimuth"
    if shape[0] == 180:
        return "possibly_azimuth_x_depth"
    return "contains_azimuth_dimension"


def _cast_shape_lines(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return ["- Shape orientation: no Zc candidate available"]
    lines = ["- Shape orientation notes:"]
    for candidate in candidates:
        orientation = _cast_shape_orientation(candidate["shape"])
        lines.append(
            f"  - {candidate['variable_name']}: "
            f"shape {candidate['shape']} -> {orientation}"
        )
    return lines


def _candidate_lines(title: str, candidates: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", ""]
    if not candidates:
        lines.append("- none")
        return lines
    for candidate in candidates:
        lines.append(
            "- "
            f"{candidate['variable_name']} "
            f"file_role={candidate['file_role']} "
            f"receiver_index={candidate['receiver_index']} "
            f"shape={candidate['shape']} "
            f"dtype={candidate['dtype_or_class']} "
            f"confidence={candidate['confidence']}"
        )
    return lines


def _warning_lines(warnings: list[str]) -> list[str]:
    lines = ["### Warnings", ""]
    if not warnings:
        return lines + ["- none"]
    return lines + [f"- {warning}" for warning in warnings]


def _yaml_candidate_items(candidates: list[dict[str, Any]], *, indent: int) -> list[str]:
    prefix = " " * indent
    if not candidates:
        return [f"{prefix}[]"]
    lines: list[str] = []
    for candidate in candidates[:10]:
        lines.extend(
            [
                f"{prefix}- variable: {_yaml_scalar(candidate['variable_name'])}",
                f"{prefix}  file_role: {_yaml_scalar(candidate['file_role'])}",
                f"{prefix}  receiver_index: {_yaml_scalar(candidate['receiver_index'])}",
                f"{prefix}  shape: {candidate['shape']}",
                f"{prefix}  dtype_or_class: {_yaml_scalar(candidate['dtype_or_class'])}",
                f"{prefix}  role_hint: {_yaml_scalar(candidate['role_hint'])}",
                f"{prefix}  confidence: {candidate['confidence']}",
                f"{prefix}  reason: {_yaml_scalar(candidate['reason'])}",
            ]
        )
    return lines


def _first_file_name(file_summaries: list[dict[str, Any]], file_role: str, default: str) -> str:
    for file_summary in file_summaries:
        if file_summary.get("file_role") == file_role and file_summary.get("filename"):
            return str(file_summary["filename"])
    return default


def _reason(role_hint: str, fallback: str) -> str:
    if role_hint != "unknown":
        return f"role_hint={role_hint}"
    return fallback


def _variables(file_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    variables = file_metadata.get("variables")
    if not isinstance(variables, list):
        return []
    return [variable for variable in variables if isinstance(variable, dict)]


def _variable_names(file_metadata: dict[str, Any]) -> list[str]:
    return [str(variable.get("name", "")) for variable in _variables(file_metadata)]


def _shape_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#[]{}*,&!|>'\"%@`"):
        return json.dumps(text)
    return text
