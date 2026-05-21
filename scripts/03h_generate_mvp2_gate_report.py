from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class MVP2GateReportError(RuntimeError):
    """Raised when the MVP-2 gate report cannot be generated safely."""


LEGACY_DOCUMENTATION_STATUS = "documentation_preferred_plus_data_unresolved"
SPECIFICATION_PREFERRED_STATUS = "specification_preferred_plus_data_unresolved"
SPECIFICATION_PREFERRED_CONCLUSION: dict[str, Any] = {
    "relbearing_sign_status": SPECIFICATION_PREFERRED_STATUS,
    "documentation_preferred_sign": "plus",
    "primary_convention": "plus",
    "ablation_convention": "minus",
    "documentation_formula": "theta_aligned = (theta_raw + RelBearing) mod 360",
    "data_driven_validation": "insufficient_evidence",
    "single_sign_alignment_approved": False,
    "single_sign_alignment_approval_basis": "specification_only_not_data_confirmed",
    "approved_downstream_mode": "plus_primary_minus_ablation",
    "documentation_basis": (
        "Halliburton Relative Bearing documentation suggests a plus convention when raw side "
        "azimuth is clockwise from tool key and measured looking downhole."
    ),
    "manual_confirmations": {
        "side_a_aligned_with_cast_0deg": True,
        "side_a_offset_deg": 0.0,
        "side_a_offset_status": "manually_confirmed",
        "xsi_side_order": "clockwise",
        "xsi_side_order_status": "manually_confirmed",
        "cast_azimuth_direction": "normal",
        "cast_azimuth_values": "0,2,4,...,358",
        "cast_azimuth_direction_status": "manually_confirmed",
    },
    "xsi_geometry": {
        "receiver_count": 13,
        "reference_receiver_index": 7,
        "receiver_spacing_ft": 0.5,
        "source_to_receiver1_ft": 1.0,
        "source_to_reference_receiver_ft": 4.0,
        "receiver_offsets_from_R7_ft": {
            "R1": -3.0,
            "R2": -2.5,
            "R3": -2.0,
            "R4": -1.5,
            "R5": -1.0,
            "R6": -0.5,
            "R7": 0.0,
            "R8": 0.5,
            "R9": 1.0,
            "R10": 1.5,
            "R11": 2.0,
            "R12": 2.5,
            "R13": 3.0,
        },
    },
    "unresolved_assumptions": [
        "Data-driven plus/minus/no/random metrics are not sign-discriminative.",
        "Plus is specification-preferred, not data-confirmed.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MVP-2 gate report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--depth-axis-audit", default=None)
    parser.add_argument("--depth-grid-proposal", default=None)
    parser.add_argument("--depth-only-summary", default=None)
    parser.add_argument("--depth-resample-preview", default=None)
    parser.add_argument("--depth-resample-overlap-preview", default=None)
    parser.add_argument("--relbearing-validation", default=None)
    parser.add_argument("--relbearing-validation-overlap", default=None)
    parser.add_argument("--orientation-confidence", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        paths = _resolve_paths(config, args)
        report = _build_gate_report(paths)
        markdown = _format_markdown(report)
        _ensure_report_output(config, paths["output_report_md"])
        _ensure_report_output(config, paths["output_report_json"])
        if not args.dry_run:
            _write_outputs(
                report,
                markdown,
                output_md=paths["output_report_md"],
                output_json=paths["output_report_json"],
                overwrite=args.overwrite,
            )
    except (ManifestBuildError, MVP2GateReportError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-2 gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"approved_downstream_mode={report['approved_downstream_mode']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    data = _as_dict(config.get("data"))
    reports = Path(str(data.get("reports", "")))
    interim = Path(str(data.get("interim", "")))
    return {
        "depth_axis_audit": Path(
            args.depth_axis_audit or reports / "depth_axis_audit_report.json"
        ),
        "depth_grid_proposal": Path(
            args.depth_grid_proposal or reports / "depth_grid_proposal.json"
        ),
        "depth_only_summary": Path(
            args.depth_only_summary or interim / "depth_only_summary_v001.json"
        ),
        "depth_resample_preview": Path(
            args.depth_resample_preview or reports / "depth_resample_preview_report.json"
        ),
        "depth_resample_overlap_preview": Path(
            args.depth_resample_overlap_preview
            or reports / "depth_resample_overlap_preview_report.json"
        ),
        "relbearing_validation": Path(
            args.relbearing_validation or reports / "relbearing_sign_validation_report.json"
        ),
        "relbearing_validation_overlap": Path(
            args.relbearing_validation_overlap
            or reports / "relbearing_sign_validation_overlap_report.json"
        ),
        "orientation_confidence": Path(
            args.orientation_confidence or reports / "orientation_confidence_report.json"
        ),
        "output_report_md": Path(args.output_report_md or reports / "mvp2_gate_report.md"),
        "output_report_json": Path(args.output_report_json or reports / "mvp2_gate_report.json"),
    }


def _build_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    statuses: dict[str, dict[str, Any]] = {
        "depth_axis_audit": _depth_axis_audit_status(paths["depth_axis_audit"]),
        "depth_grid_proposal": _depth_grid_status(paths["depth_grid_proposal"]),
        "depth_only_reader": _depth_only_status(paths["depth_only_summary"]),
        "initial_resampling": _initial_resample_status(paths["depth_resample_preview"]),
        "overlap_targeted_resampling": _overlap_resample_status(
            paths["depth_resample_overlap_preview"]
        ),
        "relbearing_validation": _relbearing_status(
            paths["relbearing_validation"],
            source="initial",
        ),
        "relbearing_validation_overlap": _relbearing_status(
            paths["relbearing_validation_overlap"],
            source="overlap",
        ),
        "orientation_confidence": _orientation_confidence_status(
            paths["orientation_confidence"]
        ),
    }

    blocking: list[str] = []
    warnings: list[str] = []
    for name, status in statuses.items():
        blocking.extend(f"{name}: {message}" for message in status["errors"])
        warnings.extend(f"{name}: {message}" for message in status["warnings"])

    relbearing_conclusion = _relbearing_conclusion(statuses["relbearing_validation_overlap"])
    blocking.extend(_relbearing_blockers(relbearing_conclusion))

    required_ready = all(
        statuses[name]["ready"]
        for name in (
            "depth_axis_audit",
            "depth_grid_proposal",
            "depth_only_reader",
            "overlap_targeted_resampling",
            "orientation_confidence",
        )
    )
    if not required_ready:
        blocking.append("One or more required MVP-2 alignment artifacts are not ready.")

    decision = _decision(blocking, relbearing_conclusion)
    unresolved = [
        "Data-driven RelBearing sign validation remains insufficient_evidence.",
        "Plus is specification-preferred but not data-confirmed.",
        "Depth unit remains unknown_to_verify in current audit/proposal warnings.",
        "Low-inclination intervals require orientation_uncertain handling.",
    ]
    return {
        "stage": "MVP-2",
        "task": "mvp2_gate_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "statuses": statuses,
        "relbearing_convention": relbearing_conclusion,
        "documentation_preferred_convention": relbearing_conclusion[
            "documentation_preferred_sign"
        ],
        "primary_convention": relbearing_conclusion["primary_convention"],
        "ablation_convention": relbearing_conclusion["ablation_convention"],
        "documentation_formula": relbearing_conclusion["documentation_formula"],
        "data_driven_validation": relbearing_conclusion["data_driven_validation"],
        "single_sign_alignment_approved": relbearing_conclusion[
            "single_sign_alignment_approved"
        ],
        "single_sign_alignment_approval_basis": relbearing_conclusion[
            "single_sign_alignment_approval_basis"
        ],
        "approved_downstream_mode": relbearing_conclusion["approved_downstream_mode"],
        "manual_confirmations": relbearing_conclusion["manual_confirmations"],
        "xsi_geometry": relbearing_conclusion["xsi_geometry"],
        "blocking_issues": blocking,
        "warnings": warnings,
        "unresolved_issues": unresolved,
        "decision": decision,
        "next_recommended_stage": (
            "MVP-3_plus_primary_minus_ablation_only"
            if decision == "conditional_go"
            else "fix_mvp2_blockers"
        ),
        "not_allowed": [
            "single-sign production alignment",
            "direct final weak label generation",
            "feature extraction",
            "STC/APES",
            "model training",
        ],
        "not_performed": [
            "weak label generation",
            "feature extraction",
            "STC/APES",
            "model training",
            "MVP-3 execution",
        ],
    }


def _depth_axis_audit_status(path: Path) -> dict[str, Any]:
    data, read_errors = _read_json_status(path)
    errors = list(read_errors)
    warnings = _warnings(data)
    details = {"path": str(path)}
    if data:
        errors.extend(_errors(data))
        errors.extend(str(item) for item in data.get("no_go_blockers", []) if item)
        if data.get("decision") == "no_go":
            errors.append("depth axis audit decision is no_go")
        overlap = _as_dict(data.get("common_overlap_interval"))
        if _as_float(overlap.get("length")) is None or float(overlap.get("length", 0.0)) <= 0.0:
            errors.append("depth axis audit has no positive common overlap interval")
        details.update(
            {
                "decision": data.get("decision"),
                "common_overlap_interval": overlap,
                "depth_unit": data.get("depth_unit"),
            }
        )
    return _status("pass", errors, warnings, details)


def _depth_grid_status(path: Path) -> dict[str, Any]:
    data, read_errors = _read_json_status(path)
    errors = list(read_errors)
    warnings = _warnings(data)
    details = {"path": str(path)}
    if data:
        errors.extend(_errors(data))
        errors.extend(str(item) for item in data.get("no_go_blockers", []) if item)
        if data.get("decision") == "no_go":
            errors.append("depth grid proposal decision is no_go")
        if _as_float(data.get("depth_step")) is None:
            errors.append("depth grid proposal has no depth_step")
        if int(data.get("sample_count", 0) or 0) <= 0:
            errors.append("depth grid proposal sample_count is not positive")
        details.update(
            {
                "decision": data.get("decision"),
                "depth_start": data.get("depth_start"),
                "depth_stop": data.get("depth_stop"),
                "depth_step": data.get("depth_step"),
                "sample_count": data.get("sample_count"),
            }
        )
    return _status("pass", errors, warnings, details)


def _depth_only_status(path: Path) -> dict[str, Any]:
    data, read_errors = _read_json_status(path)
    errors = list(read_errors)
    warnings = _warnings(data)
    details = {"path": str(path)}
    required = {"cast_depth", "xsi_depth_by_receiver", "pose_depth", "inc_deg", "relbearing_deg"}
    if data:
        errors.extend(_errors(data))
        arrays = _as_dict(data.get("arrays"))
        missing = sorted(required.difference(arrays))
        errors.extend(f"missing depth-only array summary: {name}" for name in missing)
        details.update({"array_names": sorted(arrays)})
    return _status("pass", errors, warnings, details)


def _initial_resample_status(path: Path) -> dict[str, Any]:
    data, read_errors = _read_json_status(path)
    errors = list(read_errors)
    warnings = _warnings(data)
    details = {"path": str(path)}
    if data:
        errors.extend(_errors(data))
        small_slice = _as_dict(data.get("small_slice"))
        status = str(small_slice.get("status"))
        if status != "completed":
            warnings.append(f"initial small-slice preview status is {status}")
        details.update({"small_slice_status": status, "canonical_grid": data.get("canonical_grid")})
    return _status("pass", errors, warnings, details)


def _overlap_resample_status(path: Path) -> dict[str, Any]:
    data, read_errors = _read_json_status(path)
    errors = list(read_errors)
    warnings = _warnings(data)
    details = {"path": str(path)}
    if data:
        errors.extend(_errors(data))
        small_slice = _as_dict(data.get("small_slice"))
        status = str(small_slice.get("status"))
        if status != "completed":
            errors.append(f"overlap-targeted small-slice preview status is {status}")
        arrays = _as_dict(data.get("arrays"))
        for name in ("small_slice_cast_zc_on_preview", "small_slice_xsi_waveform_on_preview"):
            if name not in arrays:
                errors.append(f"overlap resample report is missing {name}")
        details.update({"small_slice_status": status, "canonical_grid": data.get("canonical_grid")})
    return _status("pass", errors, warnings, details)


def _relbearing_status(path: Path, *, source: str) -> dict[str, Any]:
    data, read_errors = _read_json_status(path)
    errors = list(read_errors)
    warnings = _warnings(data)
    details = {"path": str(path), "source": source}
    if data:
        errors.extend(_errors(data))
        decision = data.get("decision")
        if decision != "insufficient_evidence":
            warnings.append(f"RelBearing validation decision is {decision}; expected unresolved.")
        if data.get("selected_convention") is not None:
            errors.append("RelBearing validation selected a convention before approval")
        conclusion = _as_dict(data.get("convention_conclusion"))
        if not conclusion:
            warnings.append(
                "RelBearing validation report lacks convention_conclusion; using gate policy."
            )
        details.update(
            {
                "decision": decision,
                "selected_convention": data.get("selected_convention"),
                "convention_conclusion": conclusion,
            }
        )
    return _status("pass", errors, warnings, details)


def _orientation_confidence_status(path: Path) -> dict[str, Any]:
    data, read_errors = _read_json_status(path)
    errors = list(read_errors)
    warnings = _warnings(data)
    details = {"path": str(path)}
    if data:
        errors.extend(_errors(data))
        arrays = _as_dict(data.get("arrays"))
        for name in ("orientation_confidence", "orientation_uncertain", "low_inc_mask"):
            if name not in arrays:
                errors.append(f"orientation confidence report is missing {name}")
        if data.get("relbearing_sign_dependency") != "independent_of_plus_minus_convention":
            errors.append(
                "orientation confidence report does not state RelBearing sign independence"
            )
        details.update(
            {
                "low_inclination_ratio": data.get("low_inclination_ratio"),
                "stable_inclination_ratio": data.get("stable_inclination_ratio"),
            }
        )
    return _status("pass", errors, warnings, details)


def _relbearing_conclusion(overlap_status: dict[str, Any]) -> dict[str, Any]:
    details = _as_dict(overlap_status.get("details"))
    report_conclusion = _as_dict(details.get("convention_conclusion"))
    conclusion = dict(SPECIFICATION_PREFERRED_CONCLUSION)
    if report_conclusion:
        conclusion.update(report_conclusion)
    if conclusion.get("relbearing_sign_status") == LEGACY_DOCUMENTATION_STATUS:
        conclusion["relbearing_sign_status"] = SPECIFICATION_PREFERRED_STATUS
    conclusion["primary_convention"] = "plus"
    conclusion["ablation_convention"] = "minus"
    conclusion["documentation_preferred_sign"] = "plus"
    conclusion["documentation_formula"] = SPECIFICATION_PREFERRED_CONCLUSION[
        "documentation_formula"
    ]
    conclusion["single_sign_alignment_approval_basis"] = (
        "specification_only_not_data_confirmed"
    )
    conclusion["approved_downstream_mode"] = "plus_primary_minus_ablation"
    conclusion["manual_confirmations"] = SPECIFICATION_PREFERRED_CONCLUSION[
        "manual_confirmations"
    ]
    conclusion["xsi_geometry"] = SPECIFICATION_PREFERRED_CONCLUSION["xsi_geometry"]
    conclusion["unresolved_assumptions"] = SPECIFICATION_PREFERRED_CONCLUSION[
        "unresolved_assumptions"
    ]
    conclusion.pop("unconfirmed_assumptions", None)
    return conclusion


def _relbearing_blockers(conclusion: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if conclusion.get("relbearing_sign_status") != SPECIFICATION_PREFERRED_STATUS:
        blockers.append(
            f"RelBearing sign status is not {SPECIFICATION_PREFERRED_STATUS}."
        )
    if conclusion.get("primary_convention") != "plus":
        blockers.append("Primary RelBearing convention is not plus.")
    if conclusion.get("ablation_convention") != "minus":
        blockers.append("Ablation RelBearing convention is not minus.")
    if conclusion.get("data_driven_validation") != "insufficient_evidence":
        blockers.append("Data-driven RelBearing validation is not insufficient_evidence.")
    if conclusion.get("single_sign_alignment_approved") is not False:
        blockers.append("single_sign_alignment_approved must be false.")
    if conclusion.get("approved_downstream_mode") != "plus_primary_minus_ablation":
        blockers.append("approved_downstream_mode must be plus_primary_minus_ablation.")
    return blockers


def _decision(blocking: list[str], conclusion: dict[str, Any]) -> str:
    if blocking:
        return "no_go"
    if conclusion.get("relbearing_sign_status") == SPECIFICATION_PREFERRED_STATUS:
        return "conditional_go"
    return "no_go"


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-2 Gate Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Decision: {report['decision']}",
        f"- Next recommended stage: {report['next_recommended_stage']}",
        f"- Documentation-preferred convention: {report['documentation_preferred_convention']}",
        f"- Primary convention: {report['primary_convention']}",
        f"- Ablation convention: {report['ablation_convention']}",
        f"- Data-driven validation: {report['data_driven_validation']}",
        f"- Approved downstream mode: {report['approved_downstream_mode']}",
        f"- Single-sign alignment approved: {report['single_sign_alignment_approved']}",
        (
            "- Single-sign alignment approval basis: "
            f"{report['single_sign_alignment_approval_basis']}"
        ),
        "",
        "## Status Summary",
        "",
    ]
    for name, status in report["statuses"].items():
        lines.append(f"- {name}: {status['status']}")
    lines.extend(["", "## RelBearing Convention", ""])
    for key, value in report["relbearing_convention"].items():
        if key in {"unconfirmed_assumptions", "unresolved_assumptions"}:
            lines.append(f"- {key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(_items(report["blocking_issues"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_items(report["warnings"]))
    lines.extend(["", "## Unresolved Issues", ""])
    lines.extend(_items(report["unresolved_issues"]))
    lines.extend(["", "## Not Allowed", ""])
    lines.extend(_items(report["not_allowed"]))
    lines.extend(["", "## Go / No-Go", ""])
    if report["decision"] == "conditional_go":
        lines.append(
            "- Conditional go only for MVP-3 plus-primary / minus-ablation workflow. "
            "Single-sign production alignment is not approved."
        )
    else:
        lines.append("- No-go for MVP-3 until MVP-2 blocking issues are fixed.")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(
    report: dict[str, Any],
    markdown: str,
    *,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json_status(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"required input does not exist: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"invalid JSON in {path}: {exc}"]
    if not isinstance(data, dict):
        return {}, [f"JSON input must contain an object: {path}"]
    return data, []


def _status(
    base_status: str,
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    status = "fail" if errors else "warning" if warnings else base_status
    return {
        "status": status,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def _warnings(data: dict[str, Any]) -> list[str]:
    return [str(item) for item in data.get("warnings", []) if item]


def _errors(data: dict[str, Any]) -> list[str]:
    return [str(item) for item in data.get("errors", []) if item]


def _items(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_report_output(config: dict[str, Any], path: Path) -> None:
    data = _as_dict(config.get("data"))
    reports = Path(str(data.get("reports", ""))).resolve()
    if not str(reports):
        raise MVP2GateReportError("data.reports is not configured.")
    try:
        path.resolve().relative_to(reports)
    except ValueError as exc:
        raise MVP2GateReportError(
            f"Refusing to write MVP-2 gate report outside data.reports: {path}"
        ) from exc


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MVP2GateReportError(f"Output already exists: {path}. Pass --overwrite.")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
