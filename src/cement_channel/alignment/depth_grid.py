from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEPTH_GRID_PROPOSAL_VERSION = "depth_grid_proposal_v001"


@dataclass(frozen=True)
class DepthGridProposal:
    proposal_version: str
    generated_at: str
    source_audit_report: str
    decision: str
    common_overlap_min: float | None
    common_overlap_max: float | None
    depth_start: float | None
    depth_stop: float | None
    depth_step: float | None
    sample_count: int
    grid_order: str
    allow_extrapolation: bool
    step_strategy: str
    source_median_steps: dict[str, float]
    rationale: list[str]
    warnings: list[str]
    errors: list[str]
    no_go_blockers: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_depth_axis_audit(path: Path | str) -> dict[str, Any]:
    audit_path = Path(path)
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Depth axis audit report must contain an object: {audit_path}")
    return data


def propose_depth_grid(
    audit_report: dict[str, Any],
    *,
    source_audit_report: Path | str,
    step_strategy: str = "coarsest_median_step",
    step_ratio_warning_threshold: float = 2.0,
) -> DepthGridProposal:
    warnings = [str(item) for item in audit_report.get("warnings", []) if item]
    errors: list[str] = []
    blockers = [str(item) for item in audit_report.get("no_go_blockers", []) if item]
    rationale = [
        "Use only Stage-1 depth-axis audit JSON.",
        "Use increasing canonical grid order.",
        "Do not interpolate or align in this proposal stage.",
        "Disable extrapolation by default.",
    ]

    overlap = _as_dict(audit_report.get("common_overlap_interval"))
    overlap_min = _as_float_or_none(overlap.get("min"))
    overlap_max = _as_float_or_none(overlap.get("max"))
    source_steps = _source_median_steps(audit_report)
    if audit_report.get("decision") == "no_go":
        blockers.append("Stage-1 depth axis audit decision is no_go.")
    if overlap_min is None or overlap_max is None or overlap_max <= overlap_min:
        blockers.append("No positive common overlap interval is available.")
    if not source_steps:
        blockers.append("No positive source median depth steps are available.")

    depth_start: float | None = None
    depth_stop: float | None = None
    depth_step: float | None = None
    sample_count = 0
    if not blockers:
        depth_step = _select_step(source_steps, step_strategy)
        if depth_step is None or depth_step <= 0.0:
            blockers.append("Selected depth step is not positive.")
        else:
            assert overlap_min is not None
            assert overlap_max is not None
            depth_start = float(overlap_min)
            sample_count = int((float(overlap_max) - depth_start) // depth_step) + 1
            depth_stop = float(depth_start + depth_step * (sample_count - 1))
            if sample_count < 2:
                blockers.append("Candidate canonical depth grid has fewer than two samples.")
            rationale.append(
                "Step uses the coarsest source median step to avoid inventing higher "
                "depth resolution during validation."
            )

    if source_steps:
        min_step = min(source_steps.values())
        max_step = max(source_steps.values())
        if min_step > 0.0 and max_step / min_step > step_ratio_warning_threshold:
            warnings.append(
                "Source depth median steps differ by "
                f"{max_step / min_step:.3f}x; proposal keeps the coarsest step."
            )
    if _depth_unit_unknown(audit_report):
        warnings.append("Depth unit is unknown_to_verify; human review remains required.")

    decision = _decision(blockers, warnings)
    return DepthGridProposal(
        proposal_version=DEPTH_GRID_PROPOSAL_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_audit_report=str(source_audit_report),
        decision=decision,
        common_overlap_min=overlap_min,
        common_overlap_max=overlap_max,
        depth_start=depth_start,
        depth_stop=depth_stop,
        depth_step=depth_step,
        sample_count=sample_count,
        grid_order="increasing",
        allow_extrapolation=False,
        step_strategy=step_strategy,
        source_median_steps=source_steps,
        rationale=rationale,
        warnings=warnings,
        errors=errors,
        no_go_blockers=blockers,
        not_performed=[
            "MAT reading",
            "waveform reading",
            "CAST Zc reading",
            "interpolation",
            "alignment",
            "label generation",
            "feature extraction",
            "model training",
        ],
    )


def format_depth_grid_proposal_markdown(proposal: DepthGridProposal) -> str:
    data = proposal.to_dict()
    lines = [
        "# Depth Grid Proposal",
        "",
        f"- Proposal version: {data['proposal_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Decision: {data['decision']}",
        f"- Source audit report: {data['source_audit_report']}",
        "",
        "## Common Overlap",
        "",
        f"- common_overlap_min: {data['common_overlap_min']}",
        f"- common_overlap_max: {data['common_overlap_max']}",
        "",
        "## Proposed Canonical Grid",
        "",
        f"- depth_start: {data['depth_start']}",
        f"- depth_stop: {data['depth_stop']}",
        f"- depth_step: {data['depth_step']}",
        f"- sample_count: {data['sample_count']}",
        f"- grid_order: {data['grid_order']}",
        f"- allow_extrapolation: {data['allow_extrapolation']}",
        f"- step_strategy: {data['step_strategy']}",
        "",
        "## Source Median Steps",
        "",
    ]
    for key, value in data["source_median_steps"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Rationale", ""])
    lines.extend(_message_lines(data["rationale"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## No-Go Blockers", ""])
    lines.extend(_message_lines(data["no_go_blockers"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def depth_grid_config_dict(proposal: DepthGridProposal) -> dict[str, Any]:
    return {
        "schema_version": "schema_v001",
        "alignment_config_version": "alignment_depth_grid_v001",
        "status": "example_requires_human_review"
        if proposal.decision != "go"
        else "example_generated",
        "source_report": "reports/depth_axis_audit_report.json",
        "canonical_depth_grid": {
            "depth_start": proposal.depth_start,
            "depth_stop": proposal.depth_stop,
            "depth_step": proposal.depth_step,
            "sample_count": proposal.sample_count,
            "grid_order": proposal.grid_order,
            "depth_unit": "unknown_to_verify",
            "allow_extrapolation": proposal.allow_extrapolation,
            "step_strategy": proposal.step_strategy,
        },
        "quality_gate": {
            "proposal_decision": proposal.decision,
            "requires_human_review": bool(proposal.warnings or proposal.no_go_blockers),
            "warnings": proposal.warnings,
            "no_go_blockers": proposal.no_go_blockers,
        },
        "not_performed": proposal.not_performed,
    }


def write_depth_grid_outputs(
    proposal: DepthGridProposal,
    *,
    output_json: Path,
    output_md: Path,
    output_config: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_config, overwrite=overwrite)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(proposal.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_depth_grid_proposal_markdown(proposal), encoding="utf-8")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(depth_grid_config_dict(proposal), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _source_median_steps(audit_report: dict[str, Any]) -> dict[str, float]:
    steps: dict[str, float] = {}
    for key in ["cast_depth", "pose_depth"]:
        stats = _as_dict(audit_report.get(key))
        step = _as_float_or_none(stats.get("median_step"))
        if step is not None and step > 0.0:
            steps[key] = step
    for receiver, stats_value in _as_dict(audit_report.get("xsi_depth_by_receiver")).items():
        stats = _as_dict(stats_value)
        step = _as_float_or_none(stats.get("median_step"))
        if step is not None and step > 0.0:
            steps[f"xsi_{receiver}"] = step
    return steps


def _select_step(source_steps: dict[str, float], step_strategy: str) -> float | None:
    if not source_steps:
        return None
    if step_strategy != "coarsest_median_step":
        raise ValueError(f"Unsupported depth grid step strategy: {step_strategy}")
    return float(max(source_steps.values()))


def _decision(blockers: list[str], warnings: list[str]) -> str:
    if blockers:
        return "no_go"
    if warnings:
        return "conditional_go"
    return "go"


def _depth_unit_unknown(audit_report: dict[str, Any]) -> bool:
    return str(audit_report.get("depth_unit", "")).lower().startswith("unknown")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
