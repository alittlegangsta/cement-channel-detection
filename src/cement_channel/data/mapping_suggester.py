from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_MAPPING_SUGGESTION_VERSION = "raw_variable_mapping_suggestions_v001"
RAW_MAPPING_DRAFT_VERSION = "raw_variable_mapping_draft_v001"
TODO_CONFIRM = "TODO_CONFIRM"

RECOMMENDATION_TARGETS = [
    "cast.zc_variable",
    "cast.depth_variable",
    "pose.depth_variable",
    "pose.inclination_variable",
    "pose.relbearing_variable",
    "xsi.waveform_variable",
    "xsi.depth_variable",
    "xsi.time_variable",
]


@dataclass(frozen=True)
class MappingCandidate:
    file_path: str
    filename: str
    file_role: str | None
    receiver_index: int | None
    field_path: str
    shape: list[int]
    dtype_or_class: str
    role_hint: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class MappingRecommendation:
    target: str
    variable_path: str
    confidence: float
    reason: str
    candidates: list[MappingCandidate]
    requires_human_confirmation: bool = True


@dataclass(frozen=True)
class RawMappingSuggestionResult:
    suggestion_version: str
    struct_probe_json_path: str
    generated_at: str
    status: str
    well_id: str
    recommendations: dict[str, MappingRecommendation]
    human_review_required: bool
    human_review_fields: list[str]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_struct_probe_json(path: Path | str) -> dict[str, Any]:
    probe_path = Path(path)
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Struct probe JSON must contain an object: {probe_path}")
    return data


def suggest_raw_variable_mapping(
    probe: dict[str, Any],
    *,
    struct_probe_json_path: Path | str,
    well_id: str = "TODO_CONFIRM",
) -> RawMappingSuggestionResult:
    files = _probe_files(probe)
    errors = [] if files else ["struct probe JSON contains no file entries"]
    source_warnings = [str(warning) for warning in probe.get("warnings", []) if warning]
    file_warnings = _source_file_warnings(files)
    recommendations = _build_recommendations(files)
    todo_targets = [
        target
        for target in RECOMMENDATION_TARGETS
        if recommendations[target].variable_path == TODO_CONFIRM
    ]
    warnings = source_warnings + file_warnings
    warnings.extend([f"No reliable recommendation for {target}." for target in todo_targets])
    status = _status(errors, warnings)

    return RawMappingSuggestionResult(
        suggestion_version=RAW_MAPPING_SUGGESTION_VERSION,
        struct_probe_json_path=str(struct_probe_json_path),
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        well_id=well_id,
        recommendations=recommendations,
        human_review_required=True,
        human_review_fields=RECOMMENDATION_TARGETS.copy(),
        warnings=warnings,
        errors=errors,
    )


def format_mapping_suggestions_report(result: RawMappingSuggestionResult) -> str:
    data = result.to_dict()
    lines = [
        "# Raw Variable Mapping Suggestions",
        "",
        f"- Input struct-probe-json: {data['struct_probe_json_path']}",
        f"- Generated at: {data['generated_at']}",
        f"- Overall status: {data['status']}",
        f"- Well ID: {data['well_id']}",
        f"- Human review required: {data['human_review_required']}",
        "",
        "## Recommendations",
        "",
    ]
    for target in RECOMMENDATION_TARGETS:
        recommendation = data["recommendations"][target]
        lines.extend(
            [
                f"### {target}",
                "",
                f"- Suggested variable: {recommendation['variable_path']}",
                f"- Confidence: {recommendation['confidence']:.2f}",
                f"- Reason: {recommendation['reason']}",
                f"- Candidate count: {len(recommendation['candidates'])}",
                "",
            ]
        )
        for candidate in recommendation["candidates"][:8]:
            lines.append(
                "- "
                f"{candidate['field_path']} "
                f"file={candidate['filename']} "
                f"shape={candidate['shape']} "
                f"dtype={candidate['dtype_or_class']} "
                f"confidence={candidate['confidence']:.2f}"
            )
        if recommendation["candidates"]:
            lines.append("")

    lines.extend(["## Human Review Required", ""])
    lines.extend([f"- {target}" for target in data["human_review_fields"]])
    if data["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in data["warnings"]])
    if data["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend([f"- {error}" for error in data["errors"]])
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- Manually inspect the generated draft mapping.",
            "- Copy confirmed values into configs/raw_variable_mapping.yaml only after review.",
            "- Proceed to the controlled small-slice MAT reader only after confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def format_mapping_draft_yaml(result: RawMappingSuggestionResult) -> str:
    data = result.to_dict()
    recommendations = data["recommendations"]
    cast_zc = _yaml_scalar(_recommended_value(recommendations, "cast.zc_variable"))
    cast_depth = _yaml_scalar(_recommended_value(recommendations, "cast.depth_variable"))
    pose_depth = _yaml_scalar(_recommended_value(recommendations, "pose.depth_variable"))
    pose_inclination = _yaml_scalar(
        _recommended_value(recommendations, "pose.inclination_variable")
    )
    pose_relbearing = _yaml_scalar(
        _recommended_value(recommendations, "pose.relbearing_variable")
    )
    xsi_waveform = _yaml_scalar(_recommended_value(recommendations, "xsi.waveform_variable"))
    xsi_depth = _yaml_scalar(_recommended_value(recommendations, "xsi.depth_variable"))
    xsi_time = _yaml_scalar(_recommended_value(recommendations, "xsi.time_variable"))
    lines = [
        "schema_version: schema_v001",
        f"mapping_version: {RAW_MAPPING_DRAFT_VERSION}",
        f"well_id: {_yaml_scalar(data['well_id'])}",
        "status: draft_requires_human_review",
        f"source_struct_probe_json: {_yaml_scalar(data['struct_probe_json_path'])}",
        "",
        "cast:",
        f"  zc_variable: {cast_zc}",
        f"  depth_variable: {cast_depth}",
        "  azimuth_variable: TODO_CONFIRM",
        "  recommendations:",
        "    zc:",
    ]
    lines.extend(_yaml_candidate_items(recommendations["cast.zc_variable"]["candidates"], 6))
    lines.append("    depth:")
    lines.extend(_yaml_candidate_items(recommendations["cast.depth_variable"]["candidates"], 6))
    lines.append("    azimuth: []")
    lines.extend(
        [
            "",
            "pose:",
            f"  depth_variable: {pose_depth}",
            f"  inclination_variable: {pose_inclination}",
            f"  relbearing_variable: {pose_relbearing}",
            "  recommendations:",
            "    depth:",
        ]
    )
    lines.extend(_yaml_candidate_items(recommendations["pose.depth_variable"]["candidates"], 6))
    lines.append("    inclination:")
    lines.extend(
        _yaml_candidate_items(recommendations["pose.inclination_variable"]["candidates"], 6)
    )
    lines.append("    relbearing:")
    lines.extend(
        _yaml_candidate_items(recommendations["pose.relbearing_variable"]["candidates"], 6)
    )
    lines.extend(
        [
            "",
            "xsi:",
            "  receiver_dir: XSILMR",
            "  expected_receiver_files: 13",
            "  receiver_file_pattern: XSILMR*.mat",
            f"  waveform_variable: {xsi_waveform}",
            f"  depth_variable: {xsi_depth}",
            f"  time_variable: {xsi_time}",
            "  receiver_index_source: filename",
            "  side_field_pattern: TODO_CONFIRM",
            "  recommendations:",
            "    waveform:",
        ]
    )
    lines.extend(
        _yaml_candidate_items(recommendations["xsi.waveform_variable"]["candidates"], 6)
    )
    lines.append("    depth:")
    lines.extend(_yaml_candidate_items(recommendations["xsi.depth_variable"]["candidates"], 6))
    lines.append("    time:")
    lines.extend(_yaml_candidate_items(recommendations["xsi.time_variable"]["candidates"], 6))
    lines.extend(
        [
            "",
            "human_review:",
            "  required: true",
            "  fields:",
        ]
    )
    lines.extend([f"    - {_yaml_scalar(target)}" for target in data["human_review_fields"]])
    lines.extend(
        [
            "  notes:",
            "    - Confirm all variable paths before reading slices.",
            "    - Confirm whether CAST Zc shape is depth x azimuth or azimuth x depth.",
            "    - Confirm whether XSI waveform fields encode Side A-H separately.",
            "    - Confirm depth units and time units.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_recommendations(
    files: list[dict[str, Any]],
) -> dict[str, MappingRecommendation]:
    candidate_map = _collect_candidate_map(files)
    return {
        "cast.zc_variable": _recommend_best(
            "cast.zc_variable",
            candidate_map["cast.zc_variable"],
        ),
        "cast.depth_variable": _recommend_best(
            "cast.depth_variable",
            candidate_map["cast.depth_variable"],
        ),
        "pose.depth_variable": _recommend_best(
            "pose.depth_variable",
            candidate_map["pose.depth_variable"],
        ),
        "pose.inclination_variable": _recommend_best(
            "pose.inclination_variable",
            candidate_map["pose.inclination_variable"],
        ),
        "pose.relbearing_variable": _recommend_best(
            "pose.relbearing_variable",
            candidate_map["pose.relbearing_variable"],
        ),
        "xsi.waveform_variable": _recommend_xsi_waveform(
            candidate_map["xsi.waveform_variable"],
        ),
        "xsi.depth_variable": _recommend_best(
            "xsi.depth_variable",
            candidate_map["xsi.depth_variable"],
        ),
        "xsi.time_variable": _recommend_best(
            "xsi.time_variable",
            candidate_map["xsi.time_variable"],
        ),
    }


def _collect_candidate_map(
    files: list[dict[str, Any]],
) -> dict[str, list[MappingCandidate]]:
    candidates: dict[str, list[MappingCandidate]] = {
        target: [] for target in RECOMMENDATION_TARGETS
    }
    for file_probe in files:
        for field in _probe_fields(file_probe):
            for target in _targets_for_field(file_probe, field):
                candidates[target].append(_candidate_for_target(file_probe, field, target))
    for target, items in candidates.items():
        candidates[target] = sorted(
            items,
            key=lambda item: (-item.confidence, item.field_path),
        )
    return candidates


def _targets_for_field(file_probe: dict[str, Any], field: dict[str, Any]) -> list[str]:
    file_role = str(file_probe.get("file_role") or "")
    role_hint = str(field.get("role_hint") or "unknown")
    leaf = _normalize_leaf(str(field.get("field_path", "")))
    shape = _shape_list(field.get("shape"))
    targets: list[str] = []

    if file_role == "cast":
        if _is_cast_zc(leaf, shape, role_hint):
            targets.append("cast.zc_variable")
        if _is_depth(leaf, role_hint):
            targets.append("cast.depth_variable")
    elif file_role == "pose":
        if _is_depth(leaf, role_hint):
            targets.append("pose.depth_variable")
        if _is_inclination(leaf, role_hint):
            targets.append("pose.inclination_variable")
        if _is_relbearing(leaf, role_hint):
            targets.append("pose.relbearing_variable")
    elif file_role == "xsi_receiver":
        if _is_xsi_waveform(leaf, shape, role_hint):
            targets.append("xsi.waveform_variable")
        if _is_depth(leaf, role_hint):
            targets.append("xsi.depth_variable")
        if _is_xsi_time(leaf, shape, role_hint):
            targets.append("xsi.time_variable")
    return targets


def _candidate_for_target(
    file_probe: dict[str, Any],
    field: dict[str, Any],
    target: str,
) -> MappingCandidate:
    confidence, reason = _score_field_for_target(field, target)
    return MappingCandidate(
        file_path=str(file_probe.get("path", "")),
        filename=str(file_probe.get("filename", "")),
        file_role=_optional_str(file_probe.get("file_role")),
        receiver_index=_optional_int(file_probe.get("receiver_index")),
        field_path=str(field.get("field_path", "")),
        shape=_shape_list(field.get("shape")),
        dtype_or_class=str(field.get("dtype_or_class", "")),
        role_hint=str(field.get("role_hint", "unknown")),
        confidence=confidence,
        reason=reason,
    )


def _score_field_for_target(field: dict[str, Any], target: str) -> tuple[float, str]:
    field_path = str(field.get("field_path", ""))
    leaf = _normalize_leaf(field_path)
    role_hint = str(field.get("role_hint") or "unknown")
    shape = _shape_list(field.get("shape"))
    score = 0.45
    reasons: list[str] = []

    expected_hint = {
        "cast.zc_variable": "cast_zc_candidate",
        "cast.depth_variable": "depth_candidate",
        "pose.depth_variable": "depth_candidate",
        "pose.inclination_variable": "inclination_candidate",
        "pose.relbearing_variable": "relbearing_candidate",
        "xsi.waveform_variable": "xsi_waveform_candidate",
        "xsi.depth_variable": "depth_candidate",
        "xsi.time_variable": "xsi_time_candidate",
    }[target]
    if role_hint == expected_hint:
        score += 0.32
        reasons.append(f"role_hint={role_hint}")

    if target == "cast.zc_variable" and 180 in shape:
        score += 0.12
        reasons.append("shape includes 180 CAST azimuth sectors")
    if target.endswith("depth_variable") and leaf in {"depth", "md", "depthinc"}:
        score += 0.12
        reasons.append("depth-like field name")
    if target == "pose.inclination_variable" and leaf in {"inc", "incl", "inclination"}:
        score += 0.15
        reasons.append("inclination-like field name")
    if target == "pose.relbearing_variable" and "relbearing" in leaf:
        score += 0.15
        reasons.append("RelBearing-like field name")
    if target == "xsi.waveform_variable" and _side_suffix(field_path):
        score += 0.10
        reasons.append("Side A-H waveform field")
    if target == "xsi.time_variable" and leaf in {"time", "t", "tad", "timems"}:
        score += 0.08
        reasons.append("time-like field name")

    if _is_numeric_dtype(str(field.get("dtype_or_class", ""))):
        score += 0.04
        reasons.append("numeric dtype")
    return min(score, 0.99), "; ".join(reasons) or "name/shape heuristic"


def _recommend_best(
    target: str,
    candidates: list[MappingCandidate],
    *,
    minimum_confidence: float = 0.70,
) -> MappingRecommendation:
    if not candidates or candidates[0].confidence < minimum_confidence:
        return MappingRecommendation(
            target=target,
            variable_path=TODO_CONFIRM,
            confidence=0.0,
            reason="no candidate exceeded confidence threshold",
            candidates=candidates,
        )
    best = candidates[0]
    return MappingRecommendation(
        target=target,
        variable_path=best.field_path,
        confidence=best.confidence,
        reason=best.reason,
        candidates=candidates,
    )


def _recommend_xsi_waveform(candidates: list[MappingCandidate]) -> MappingRecommendation:
    if not candidates:
        return MappingRecommendation(
            target="xsi.waveform_variable",
            variable_path=TODO_CONFIRM,
            confidence=0.0,
            reason="no waveform candidate found",
            candidates=[],
        )
    side_group = _best_side_group(candidates)
    if side_group is not None:
        prefix, group_candidates = side_group
        confidence = min(0.99, max(candidate.confidence for candidate in group_candidates) + 0.04)
        return MappingRecommendation(
            target="xsi.waveform_variable",
            variable_path=f"{prefix}{{A-H}}",
            confidence=confidence,
            reason="detected Side A-H waveform field group with matching shape",
            candidates=group_candidates + [
                candidate for candidate in candidates if candidate not in group_candidates
            ],
        )
    return _recommend_best("xsi.waveform_variable", candidates, minimum_confidence=0.72)


def _best_side_group(
    candidates: list[MappingCandidate],
) -> tuple[str, list[MappingCandidate]] | None:
    grouped: dict[tuple[str, tuple[int, ...]], list[MappingCandidate]] = {}
    for candidate in candidates:
        prefix = _side_prefix(candidate.field_path)
        if prefix is None:
            continue
        grouped.setdefault((prefix, tuple(candidate.shape)), []).append(candidate)
    complete_groups: list[tuple[str, list[MappingCandidate]]] = []
    for (prefix, _shape), group_candidates in grouped.items():
        sides = {_side_suffix(candidate.field_path) for candidate in group_candidates}
        if sides == set("ABCDEFGH"):
            sorted_group = sorted(group_candidates, key=lambda item: item.field_path)
            complete_groups.append((prefix, sorted_group))
    if not complete_groups:
        return None
    return sorted(
        complete_groups,
        key=lambda item: (-sum(candidate.confidence for candidate in item[1]), item[0]),
    )[0]


def _probe_files(probe: dict[str, Any]) -> list[dict[str, Any]]:
    files = probe.get("files")
    if not isinstance(files, list):
        return []
    return [item for item in files if isinstance(item, dict)]


def _probe_fields(file_probe: dict[str, Any]) -> list[dict[str, Any]]:
    fields = file_probe.get("fields")
    if not isinstance(fields, list):
        return []
    return [item for item in fields if isinstance(item, dict)]


def _source_file_warnings(files: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for file_probe in files:
        filename = str(file_probe.get("filename", ""))
        file_errors = [str(error) for error in file_probe.get("errors", []) if error]
        if file_errors:
            warnings.append(f"{filename} has struct probe errors: {'; '.join(file_errors)}")
        if not file_probe.get("can_probe"):
            warnings.append(f"{filename} was not probed successfully.")
    return warnings


def _status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "fail"
    if warnings:
        return "warning"
    return "warning"


def _is_cast_zc(leaf: str, shape: list[int], role_hint: str) -> bool:
    return (
        role_hint == "cast_zc_candidate"
        or leaf in {"zc", "castzc"}
        or leaf.endswith("zc")
        or "impedance" in leaf
        or ("zc" in leaf and len(shape) >= 2)
    )


def _is_depth(leaf: str, role_hint: str) -> bool:
    return (
        role_hint == "depth_candidate"
        or leaf in {"depth", "md", "tvd", "depthinc"}
        or leaf.endswith("depth")
    )


def _is_inclination(leaf: str, role_hint: str) -> bool:
    return (
        role_hint == "inclination_candidate"
        or leaf in {"inc", "incl", "inclination"}
        or "inclination" in leaf
    )


def _is_relbearing(leaf: str, role_hint: str) -> bool:
    return (
        role_hint == "relbearing_candidate"
        or "relbearing" in leaf
        or "relativebearing" in leaf
    )


def _is_xsi_waveform(leaf: str, shape: list[int], role_hint: str) -> bool:
    return (
        role_hint == "xsi_waveform_candidate"
        or leaf in {"waveform", "wave", "data"}
        or "waveform" in leaf
        or ("wave" in leaf and len(shape) >= 2)
    )


def _is_xsi_time(leaf: str, shape: list[int], role_hint: str) -> bool:
    return (
        role_hint == "xsi_time_candidate"
        or leaf in {"time", "t", "tad", "timeaxis", "timems"}
        or ("time" in leaf and len(shape) <= 2)
    )


def _side_prefix(field_path: str) -> str | None:
    match = re.match(r"^(?P<prefix>.+Side)(?P<side>[A-H])$", field_path)
    if match is None:
        return None
    return match.group("prefix")


def _side_suffix(field_path: str) -> str | None:
    match = re.search(r"Side([A-H])$", field_path)
    if match is None:
        return None
    return match.group(1)


def _normalize_leaf(field_path: str) -> str:
    leaf = field_path.split(".")[-1]
    return re.sub(r"[^a-z0-9]+", "", leaf.lower())


def _shape_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _is_numeric_dtype(dtype_or_class: str) -> bool:
    dtype = dtype_or_class.lower()
    return any(
        token in dtype
        for token in ("int", "float", "double", "single", "uint", "logical")
    )


def _recommended_value(recommendations: dict[str, Any], target: str) -> str:
    value = recommendations[target]["variable_path"]
    return str(value or TODO_CONFIRM)


def _yaml_candidate_items(candidates: list[dict[str, Any]], indent: int) -> list[str]:
    if not candidates:
        return [" " * indent + "[]"]
    pad = " " * indent
    lines: list[str] = []
    for candidate in candidates[:12]:
        lines.extend(
            [
                f"{pad}- variable_path: {_yaml_scalar(candidate['field_path'])}",
                f"{pad}  file: {_yaml_scalar(candidate['filename'])}",
                f"{pad}  receiver_index: {_yaml_scalar(candidate['receiver_index'])}",
                f"{pad}  shape: {candidate['shape']}",
                f"{pad}  dtype_or_class: {_yaml_scalar(candidate['dtype_or_class'])}",
                f"{pad}  role_hint: {_yaml_scalar(candidate['role_hint'])}",
                f"{pad}  confidence: {candidate['confidence']:.2f}",
                f"{pad}  reason: {_yaml_scalar(candidate['reason'])}",
            ]
        )
    return lines


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if re.match(r"^[A-Za-z0-9_./{}:*+-]+$", text):
        return text
    return json.dumps(text, ensure_ascii=False)
