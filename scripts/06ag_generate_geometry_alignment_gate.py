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

GEOMETRY_ALIGNMENT_GATE_VERSION = "geometry_alignment_gate_v001"


class GeometryAlignmentGateCliError(RuntimeError):
    """Raised when geometry alignment gate cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometry-aware alignment gate report."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--geometry-alignment-audit-json", default=None)
    parser.add_argument("--geometry-aware-label-report-json", default=None)
    parser.add_argument("--geometry-review-summary-md", default=None)
    parser.add_argument("--depth-level-refinement-gate-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        audit_json = _resolve_report_path(
            paths,
            args.geometry_alignment_audit_json,
            "geometry_alignment_audit_v001.json",
        )
        label_report_json = _resolve_report_path(
            paths,
            args.geometry_aware_label_report_json,
            "geometry_aware_depth_labels_report_v001.json",
        )
        review_summary_md = _resolve_report_path(
            paths,
            args.geometry_review_summary_md,
            "geometry_aware_manual_review_v001/review_summary.md",
        )
        prior_gate_json = _resolve_report_path(
            paths,
            args.depth_level_refinement_gate_json,
            "depth_level_refinement_gate_report.json",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "geometry_alignment_gate_report.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "geometry_alignment_gate_report.json",
        )
        for path in (audit_json, label_report_json, review_summary_md, prior_gate_json):
            _ensure_path_within(paths, path, key="reports", action="read")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        report = build_geometry_alignment_gate_report(
            geometry_alignment_audit=_read_json(audit_json),
            geometry_label_report=_read_json(label_report_json),
            review_summary_md=review_summary_md.read_text(encoding="utf-8")
            if review_summary_md.exists()
            else "",
            review_summary_json=_read_optional_json(
                review_summary_md.parent / "geometry_alignment_review_summary_v001.json"
            ),
            prior_refinement_gate=_read_json(prior_gate_json),
            inputs={
                "geometry_alignment_audit_json": str(audit_json),
                "geometry_aware_label_report_json": str(label_report_json),
                "geometry_review_summary_md": str(review_summary_md),
                "depth_level_refinement_gate_json": str(prior_gate_json),
            },
        )
        if not args.dry_run:
            write_geometry_alignment_gate_report(
                report,
                output_md=output_md,
                output_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        GeometryAlignmentGateCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry alignment gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"best_mode={report['evidence'].get('best_mode')}; "
        f"best_sign={report['evidence'].get('best_sign')}; "
        f"no_final_labels={not report['final_labels_allowed']}."
    )
    if args.dry_run:
        print("Dry run: no gate report outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 0


def build_geometry_alignment_gate_report(
    *,
    geometry_alignment_audit: dict[str, Any],
    geometry_label_report: dict[str, Any],
    review_summary_md: str,
    review_summary_json: dict[str, Any] | None,
    prior_refinement_gate: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    warnings: list[str] = []
    blocking_issues: list[str] = []
    manual_items: list[str] = []
    combo_summaries = [
        row
        for row in _as_list(geometry_alignment_audit.get("combo_summaries"))
        if isinstance(row, dict)
    ]
    best = _as_dict(geometry_alignment_audit.get("best_combo"))
    r7 = _as_dict(geometry_alignment_audit.get("r7_reference_summary"))
    interval = _as_dict(geometry_alignment_audit.get("source_receiver_interval_summary"))
    audit_errors = _as_list(geometry_alignment_audit.get("errors"))
    if audit_errors:
        blocking_issues.extend(f"audit_error: {message}" for message in audit_errors)
    if prior_refinement_gate.get("decision") != "go":
        warnings.append(
            "Prior depth-level refinement gate is not go; geometry review remains limited."
        )
    if not best:
        blocking_issues.append("No best geometry-aware mode is available.")
    best_margin = _as_float(best.get("real_minus_permutation_margin"))
    best_predicted_positive_rate = _as_float(best.get("predicted_positive_rate"))
    if best_margin is None or best_margin < 0.03:
        blocking_issues.append(
            "Best geometry-aware sanity result does not exceed permutation margin."
        )
    if best_predicted_positive_rate is None or not (0.15 <= best_predicted_positive_rate <= 0.85):
        blocking_issues.append("Best geometry-aware predicted positive rate is degenerate.")
    if bool(best.get("depends_on_5700_band")):
        blocking_issues.append("Best geometry-aware result depends on the 5700 ft review band.")
    if bool(best.get("suspicious_leakage")):
        blocking_issues.append("Best geometry-aware result has suspicious leakage warning.")

    collapsed = [row for row in combo_summaries if bool(row.get("sample_count_collapse"))]
    if collapsed:
        warnings.append(
            "Some geometry mode/sign targets have collapsed positive or negative support."
        )
        manual_items.append("Review collapsed geometry targets before selecting any target.")
    if collapsed and len(collapsed) == len(combo_summaries):
        blocking_issues.append("All geometry mode/sign targets collapsed.")
    if interval and bool(interval.get("sample_count_collapse")):
        warnings.append("source_receiver_interval target is not adoptable in this audit.")

    sign_sensitive = [
        row
        for row in _as_list(geometry_alignment_audit.get("sign_sensitivity"))
        if row.get("sign_sensitive")
    ]
    if sign_sensitive:
        warnings.append("Depth-axis sign changes geometry-aware audit results.")
        manual_items.append("Confirm depth-axis sign before choosing a non-R7 geometry mode.")

    review_counts = _review_counts(review_summary_json, review_summary_md)
    if review_counts["revisit_interval_count"] > 0:
        warnings.append("Manual review supplement changes evidence category for DLR intervals.")
        manual_items.append("Re-review DLR intervals flagged by the geometry-aware supplement.")

    r7_margin = _as_float(r7.get("real_minus_permutation_margin"))
    interval_margin = _as_float(interval.get("real_minus_permutation_margin"))
    best_matches_r7 = (
        best_margin is not None
        and r7_margin is not None
        and best_margin >= r7_margin - 0.01
    )
    interval_matches_r7 = (
        interval_margin is not None
        and r7_margin is not None
        and interval_margin >= r7_margin - 0.01
    )
    source_receiver_interval_adoption_allowed = bool(
        interval
        and not bool(interval.get("sample_count_collapse"))
        and interval_margin is not None
        and r7_margin is not None
        and interval_margin >= r7_margin + 0.01
    )

    if blocking_issues:
        decision = "no_go"
    elif (
        warnings
        or manual_items
        or geometry_alignment_audit.get("recommendation") == "conditional_go"
    ):
        decision = "conditional_go"
    else:
        decision = "go"

    evidence = {
        "best_mode": best.get("mode"),
        "best_sign": best.get("sign"),
        "best_margin": best_margin,
        "best_predicted_positive_rate": best_predicted_positive_rate,
        "r7_margin": r7_margin,
        "source_receiver_interval_margin": interval_margin,
        "best_matches_or_exceeds_r7": best_matches_r7,
        "source_receiver_interval_matches_r7": interval_matches_r7,
        "source_receiver_interval_adoption_allowed": source_receiver_interval_adoption_allowed,
        "raw_zc_available": geometry_label_report.get("raw_zc_available"),
        "geometry_audit_recommendation": geometry_alignment_audit.get("recommendation"),
        **review_counts,
    }
    return {
        "gate_version": GEOMETRY_ALIGNMENT_GATE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "decision": decision,
        "evidence": evidence,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "manual_confirmation_required": bool(manual_items) or decision == "conditional_go",
        "manual_confirmation_items": _dedupe(manual_items),
        "next_branch_requires_human_approval": True,
        "geometry_alignment_review_supported": decision in {"go", "conditional_go"},
        "depth_axis_sign_confirmation_required": bool(sign_sensitive),
        "manual_review_revisit_required": review_counts["revisit_interval_count"] > 0,
        "mvp4c_allowed": False,
        "stc_allowed": False,
        "apes_allowed": False,
        "deep_learning_allowed": False,
        "final_labels_allowed": False,
        "production_model_allowed": False,
        "recommendation": _recommendation_text(decision, evidence, blocking_issues, warnings),
        "not_performed": [
            "MVP-4C",
            "STC",
            "APES",
            "deep learning",
            "new model training",
            "production inference",
            "final label generation",
            "ground truth claim",
        ],
    }


def write_geometry_alignment_gate_report(
    report: dict[str, Any],
    *,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_geometry_alignment_gate_markdown(report), encoding="utf-8")


def format_geometry_alignment_gate_markdown(report: dict[str, Any]) -> str:
    evidence = _as_dict(report.get("evidence"))
    lines = [
        "# Geometry-Aware Alignment Gate",
        "",
        "This gate is a review-only MVP-4B-G decision. It does not authorize "
        "MVP-4C, STC/APES, deep learning, production modeling, final labels, or "
        "ground-truth claims for CAST weak-label candidates.",
        "",
        f"- decision: `{report['decision']}`",
        f"- best_mode: `{evidence.get('best_mode')}`",
        f"- best_sign: `{evidence.get('best_sign')}`",
        f"- best_margin: {evidence.get('best_margin')}",
        f"- r7_margin: {evidence.get('r7_margin')}",
        "- source_receiver_interval_adoption_allowed: "
        f"`{evidence.get('source_receiver_interval_adoption_allowed')}`",
        "- depth_axis_sign_confirmation_required: "
        f"`{report['depth_axis_sign_confirmation_required']}`",
        f"- manual_review_revisit_required: `{report['manual_review_revisit_required']}`",
        f"- final_labels_allowed: `{report['final_labels_allowed']}`",
        "",
        "## Blocking Issues",
        "",
    ]
    lines.extend(_message_lines(_as_list(report.get("blocking_issues"))))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(_as_list(report.get("warnings"))))
    lines.extend(["", "## Manual Confirmation Items", ""])
    lines.extend(_message_lines(_as_list(report.get("manual_confirmation_items"))))
    lines.extend(["", "## Recommendation", "", str(report.get("recommendation", "")), ""])
    return "\n".join(lines)


def _review_counts(
    review_summary_json: dict[str, Any] | None,
    review_summary_md: str,
) -> dict[str, int]:
    if review_summary_json:
        return {
            "changed_interval_count": int(
                review_summary_json.get("evidence_category_changed_interval_count", 0)
            ),
            "revisit_interval_count": int(review_summary_json.get("revisit_interval_count", 0)),
        }
    changed = _parse_md_count(review_summary_md, "evidence_category_changed_interval_count")
    revisit = _parse_md_count(review_summary_md, "revisit_interval_count")
    return {"changed_interval_count": changed, "revisit_interval_count": revisit}


def _parse_md_count(text: str, key: str) -> int:
    for line in text.splitlines():
        if key in line:
            try:
                return int(line.rsplit(":", maxsplit=1)[1].strip())
            except (IndexError, ValueError):
                return 0
    return 0


def _recommendation_text(
    decision: str,
    evidence: dict[str, Any],
    blocking_issues: list[str],
    warnings: list[str],
) -> str:
    if decision == "no_go":
        return "Geometry-aware alignment is not usable until blockers are resolved: " + "; ".join(
            blocking_issues
        )
    if decision == "conditional_go":
        return (
            "Geometry-aware review may continue only with human confirmation. "
            f"Best mode/sign is {evidence.get('best_mode')} sign={evidence.get('best_sign')}; "
            "source-receiver interval is not automatically adopted; MVP-4C/STC/APES/deep "
            "learning/final labels remain blocked. Warnings: "
            + "; ".join(warnings)
        )
    return (
        "Geometry-aware alignment review may proceed as a review target only; "
        "human approval is still required before any new branch."
    )


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryAlignmentGateCliError("data.reports is not configured.")


def _ensure_path_within(
    config: dict[str, Any],
    path: Path,
    *,
    key: str,
    action: str,
) -> None:
    data = _as_dict(config.get("data"))
    root = Path(str(data.get(key, ""))).resolve()
    if not str(root):
        raise GeometryAlignmentGateCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryAlignmentGateCliError(
            f"Refusing to {action} geometry gate path outside data.{key}: {path}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _dedupe(values: list[str]) -> list[str]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _message_lines(messages: list[Any]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
