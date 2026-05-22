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


class MVP3GateReportError(RuntimeError):
    """Raised when the MVP-3 gate report cannot be generated safely."""


PROVISIONAL_RECOMMENDED_PARAMETER_SET = {
    "alpha": 0.35,
    "zc_min_limit": 2.5,
    "severity_thresholds": [0.30, 0.45, 0.60],
    "status": "provisional_after_sensitivity",
    "requires_human_review": True,
    "no_final_labels": True,
    "preserve_plus_minus_ablation": True,
    "mvp4_allowed": False,
    "reason": "thresholds are provisional and plus/minus disagreement remains non-negligible",
}
NON_NEGLIGIBLE_PLUS_MINUS_DISAGREEMENT = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MVP-3 gate report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--cast-label-input-summary", default=None)
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--weak-label-report", default=None)
    parser.add_argument("--label-audit-report", default=None)
    parser.add_argument("--label-review-summary", default=None)
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
    except (ManifestBuildError, MVP3GateReportError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-3 gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"mvp4_allowed={report['mvp4_allowed']}."
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
    return {
        "cast_label_input_summary": Path(
            args.cast_label_input_summary or reports / "cast_label_input_summary_v001.json"
        ),
        "baseline_report": Path(
            args.baseline_report or reports / "cast_zc_baseline_report_v001.json"
        ),
        "weak_label_report": Path(
            args.weak_label_report or reports / "cast_weak_label_candidates_report_v001.json"
        ),
        "label_audit_report": Path(
            args.label_audit_report or reports / "cast_weak_label_audit_v001.json"
        ),
        "label_review_summary": Path(
            args.label_review_summary
            or reports / "label_review_v001" / "label_review_summary_v001.json"
        ),
        "output_report_md": Path(args.output_report_md or reports / "mvp3_gate_report.md"),
        "output_report_json": Path(args.output_report_json or reports / "mvp3_gate_report.json"),
    }


def _build_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    statuses = {
        "cast_label_input": _status_from_report(
            paths["cast_label_input_summary"],
            required_keys=["cast_label_input_version", "arrays"],
        ),
        "baseline": _baseline_status(paths["baseline_report"]),
        "weak_label_candidates": _weak_label_status(paths["weak_label_report"]),
        "label_audit": _audit_status(paths["label_audit_report"]),
        "label_review": _review_status(paths["label_review_summary"]),
    }
    blocking: list[str] = []
    warnings: list[str] = []
    for name, status in statuses.items():
        blocking.extend(f"{name}: {message}" for message in status["errors"])
        warnings.extend(f"{name}: {message}" for message in status["warnings"])

    weak_report = statuses["weak_label_candidates"]["data"]
    audit_report = statuses["label_audit"]["data"]
    review_report = statuses["label_review"]["data"]
    if not _data_bool(weak_report, "no_final_labels"):
        blocking.append("weak_label_candidates: no_final_labels is not true.")
    if not _data_bool(audit_report, "no_final_labels"):
        blocking.append("label_audit: no_final_labels is not true.")
    if not _data_bool(review_report, "no_final_labels"):
        blocking.append("label_review: no_final_labels is not true.")

    threshold_status = (
        _as_dict(weak_report.get("threshold")).get("zc_min_limit_status")
        if isinstance(weak_report, dict)
        else None
    )
    if threshold_status == "requires_human_threshold_confirmation":
        warnings.append("weak_label_candidates: zc_min_limit requires human confirmation.")

    coverage = _as_dict(weak_report.get("coverage")) if isinstance(weak_report, dict) else {}
    plus_minus_disagreement = _as_float(coverage.get("plus_minus_disagreement"))
    if (
        plus_minus_disagreement is not None
        and plus_minus_disagreement >= NON_NEGLIGIBLE_PLUS_MINUS_DISAGREEMENT
    ):
        warnings.append(
            "weak_label_candidates: plus/minus disagreement remains non-negligible "
            f"({plus_minus_disagreement:.6g})."
        )

    recommended_parameter_set = dict(PROVISIONAL_RECOMMENDED_PARAMETER_SET)
    warnings.append(
        "weak_label_candidates: recommended parameter set is provisional_after_sensitivity "
        "and requires human review."
    )

    no_final_labels = (
        _data_bool(weak_report, "no_final_labels")
        and _data_bool(audit_report, "no_final_labels")
        and _data_bool(review_report, "no_final_labels")
    )
    plus_minus_preserved = (
        coverage.get("plus") is not None
        and coverage.get("minus_ablation") is not None
        and "minus_ablation" in _as_dict(weak_report.get("confidence"))
    )
    decision = _decision(blocking, warnings)
    mvp4_allowed = (
        decision == "go"
        and not recommended_parameter_set["requires_human_review"]
        and plus_minus_disagreement is not None
        and plus_minus_disagreement < NON_NEGLIGIBLE_PLUS_MINUS_DISAGREEMENT
    )
    return {
        "stage": "MVP-3",
        "task": "cast_weak_label_candidate_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "statuses": {key: _public_status(value) for key, value in statuses.items()},
        "recommended_parameter_set": recommended_parameter_set,
        "no_final_labels": no_final_labels,
        "plus_primary_minus_ablation_preserved": plus_minus_preserved,
        "plus_minus_disagreement": plus_minus_disagreement,
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "mvp4_allowed": mvp4_allowed,
        "mvp4_allowed_reason": (
            "all MVP-3 gate conditions resolved"
            if mvp4_allowed
            else recommended_parameter_set["reason"]
        ),
        "conditional_requirements": _conditional_requirements(warnings),
        "not_allowed": [
            "final label approval",
            "XSI feature extraction",
            "STFT",
            "STC",
            "APES",
            "model training",
            "MVP-4 correlation validation before gate conditions are resolved",
        ],
    }


def _status_from_report(path: Path, *, required_keys: list[str]) -> dict[str, Any]:
    data = _read_json(path)
    errors = list(_as_list(data.get("errors")))
    warnings = list(_as_list(data.get("warnings")))
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    return {"ready": not errors, "errors": errors, "warnings": warnings, "data": data}


def _baseline_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(path, required_keys=["cast_baseline_version", "arrays"])
    data = status["data"]
    if _as_float(data.get("baseline_valid_ratio")) is not None:
        if float(data["baseline_valid_ratio"]) < 0.70:
            status["errors"].append("baseline_valid_ratio is below 0.70.")
    else:
        status["errors"].append("baseline_valid_ratio is missing.")
    return status


def _weak_label_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=[
            "cast_weak_label_candidate_version",
            "label_version",
            "convention_status",
            "coverage",
            "confidence",
        ],
    )
    data = status["data"]
    if data.get("convention_status") != "specification_preferred_plus_data_unresolved":
        status["errors"].append("convention_status is not specification-preferred unresolved.")
    coverage = _as_dict(data.get("coverage"))
    if coverage.get("plus") is None or coverage.get("minus_ablation") is None:
        status["errors"].append("plus/minus coverage is missing.")
    return status


def _audit_status(path: Path) -> dict[str, Any]:
    return _status_from_report(
        path,
        required_keys=["label_audit_version", "coverage", "components"],
    )


def _review_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=["label_review_version", "figures", "review_summary_template"],
    )
    figures = _as_dict(status["data"].get("figures"))
    if len(figures) < 9:
        status["errors"].append("label review does not list all 9 required figures.")
    return status


def _decision(blocking: list[str], warnings: list[str]) -> str:
    if blocking:
        return "no_go"
    if warnings:
        return "conditional_go"
    return "go"


def _conditional_requirements(warnings: list[str]) -> list[str]:
    requirements: list[str] = []
    if any("zc_min_limit" in warning for warning in warnings):
        requirements.append("Human confirmation required for zc_min_limit.")
    if any("provisional_after_sensitivity" in warning for warning in warnings):
        requirements.append("Human review required for provisional weak-label parameters.")
    if any("plus/minus disagreement" in warning for warning in warnings):
        requirements.append("Review plus/minus disagreement before any MVP-4 transition.")
    if warnings:
        requirements.append("Human review required for warning items before MVP-4.")
    return requirements


def _public_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "ready": status["ready"],
        "errors": status["errors"],
        "warnings": status["warnings"],
    }


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-3 Gate Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Decision: {report['decision']}",
        f"- MVP-4 allowed: {report['mvp4_allowed']}",
        f"- MVP-4 allowed reason: {report['mvp4_allowed_reason']}",
        f"- No final labels: {report['no_final_labels']}",
        f"- Plus primary / minus ablation preserved: "
        f"{report['plus_primary_minus_ablation_preserved']}",
        "",
        "## Recommended Parameter Set",
        "",
        f"- alpha: {report['recommended_parameter_set']['alpha']}",
        f"- zc_min_limit: {report['recommended_parameter_set']['zc_min_limit']}",
        "- severity_thresholds: "
        f"{report['recommended_parameter_set']['severity_thresholds']}",
        f"- status: {report['recommended_parameter_set']['status']}",
        "- requires_human_review: "
        f"{report['recommended_parameter_set']['requires_human_review']}",
        f"- reason: {report['recommended_parameter_set']['reason']}",
        "",
        "## Statuses",
        "",
    ]
    for name, status in report["statuses"].items():
        lines.append(
            f"- {name}: ready={status['ready']}, "
            f"errors={len(status['errors'])}, warnings={len(status['warnings'])}"
        )
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(_message_lines(report["blocking_issues"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report["warnings"]))
    lines.extend(["", "## Conditional Requirements", ""])
    lines.extend(_message_lines(report["conditional_requirements"]))
    lines.extend(["", "## Not Allowed", ""])
    lines.extend(_message_lines(report["not_allowed"]))
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


def _ensure_report_output(config: dict[str, Any], path: Path) -> None:
    data = _as_dict(config.get("data"))
    reports = Path(str(data.get("reports", ""))).resolve()
    if not str(reports):
        raise MVP3GateReportError("data.reports is not configured.")
    try:
        path.resolve().relative_to(reports)
    except ValueError as exc:
        raise MVP3GateReportError(f"Refusing to write report outside data.reports: {path}") from exc


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MVP3GateReportError(
            f"Refusing to overwrite existing file without --overwrite: {path}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MVP3GateReportError(f"Missing required MVP-3 input report: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MVP3GateReportError(f"Input report must be a JSON object: {path}")
    return data


def _data_bool(data: Any, key: str) -> bool:
    return isinstance(data, dict) and bool(data.get(key, False))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
