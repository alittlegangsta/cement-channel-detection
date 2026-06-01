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

GEOMETRY_REGRESSION_GATE_VERSION = "geometry_regression_gate_v001"
FORBIDDEN_RAW_ZC_SOURCE_FIELDS = {"zc_ratio", "relative_drop", "relative_drop_plus"}


class GeometryRegressionGateCliError(RuntimeError):
    """Raised when geometry regression gate cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometry-aware regression alignment gate report."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--regression-label-report-json", default=None)
    parser.add_argument("--regression-audit-json", default=None)
    parser.add_argument("--regression-review-summary-json", default=None)
    parser.add_argument("--selected-interval-review-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        label_report_json = _resolve_report_path(
            paths,
            args.regression_label_report_json,
            "geometry_aware_regression_labels_report_v001.json",
        )
        audit_json = _resolve_report_path(
            paths,
            args.regression_audit_json,
            "geometry_regression_audit_v001.json",
        )
        review_summary_json = _resolve_report_path(
            paths,
            args.regression_review_summary_json,
            "geometry_regression_manual_review_v001/geometry_regression_review_summary_v001.json",
        )
        selected_json = _resolve_report_path(
            paths,
            args.selected_interval_review_json,
            "geometry_regression_manual_review_v001/selected_interval_review_list.json",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "geometry_regression_gate_report.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "geometry_regression_gate_report.json",
        )
        for path in (label_report_json, audit_json, review_summary_json, selected_json):
            _ensure_path_within(paths, path, key="reports", action="read")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        report = build_geometry_regression_gate_report(
            regression_label_report=_read_json(label_report_json),
            regression_audit=_read_json(audit_json),
            review_summary=_read_json(review_summary_json),
            selected_intervals=_read_json_list(selected_json),
            inputs={
                "regression_label_report_json": str(label_report_json),
                "regression_audit_json": str(audit_json),
                "regression_review_summary_json": str(review_summary_json),
                "selected_interval_review_json": str(selected_json),
            },
        )
        if not args.dry_run:
            write_geometry_regression_gate_outputs(
                report,
                output_md=output_md,
                output_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        GeometryRegressionGateCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"most_reasonable_kernel={report['answers']['most_reasonable_kernel']}; "
        f"manual_review_required={report['answers']['manual_review_required']}; "
        f"no_final_labels={not report['final_labels_allowed']}."
    )
    if args.dry_run:
        print("Dry run: no gate report outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 0


def build_geometry_regression_gate_report(
    *,
    regression_label_report: dict[str, Any],
    regression_audit: dict[str, Any],
    review_summary: dict[str, Any],
    selected_intervals: list[dict[str, Any]],
    inputs: dict[str, str],
) -> dict[str, Any]:
    warnings: list[str] = []
    blocking_issues: list[str] = []
    manual_items: list[str] = []
    kernel_summaries = [
        row
        for row in _as_list(regression_audit.get("kernel_summaries"))
        if isinstance(row, dict)
    ]
    label_kernel_summaries = [
        row
        for row in _as_list(regression_label_report.get("kernel_summaries"))
        if isinstance(row, dict)
    ]
    geometry_sign_confirmed = _geometry_sign_confirmed(regression_label_report)
    if not geometry_sign_confirmed:
        blocking_issues.append("Human-confirmed geometry sign is not fixed as depth_axis_sign=-1.")

    raw_zc = _raw_zc_status(regression_label_report)
    if not raw_zc["raw_zc_controlled"]:
        blocking_issues.append("Raw CAST Zc is not available from a controlled source.")

    audit_errors = _as_list(regression_audit.get("errors"))
    if audit_errors:
        blocking_issues.extend(f"audit_error: {message}" for message in audit_errors)

    leakage_warnings = _leakage_warnings(kernel_summaries, regression_audit)
    if leakage_warnings:
        blocking_issues.extend(leakage_warnings)

    best_kernel = _best_kernel(regression_audit, kernel_summaries)
    if best_kernel is None:
        blocking_issues.append("No most-reasonable regression kernel is available.")

    audit_only_kernels = _audit_only_kernels(kernel_summaries, label_kernel_summaries)
    collapse = _sample_support_collapse(kernel_summaries, label_kernel_summaries)
    if collapse["all_kernels_collapsed"]:
        blocking_issues.append("All regression kernels still show sample-support collapse.")
    elif collapse["any_kernel_collapsed"]:
        warnings.append("Some regression kernels still show sample-support collapse.")

    if _all_kernels_below_permutation(kernel_summaries):
        blocking_issues.append("All regression kernels are unstable or below permutation.")

    if _all_depend_on_5700(kernel_summaries):
        blocking_issues.append("All runnable regression kernels depend on the 5700 ft band.")
    elif _any_depends_on_5700(kernel_summaries):
        warnings.append("At least one regression kernel depends on the 5700 ft band.")

    continuous_health = _continuous_target_health(label_kernel_summaries)
    if not continuous_health["continuous_target_healthier_than_old_binary"]:
        warnings.append("Continuous target health is not clearly better than old binary support.")

    selected = _selected_interval_ids(selected_intervals)
    if selected:
        manual_items.append("Review selected regression intervals before any next experiment.")
    manual_items.append("Wait for human review before any MVP-4C/STC/APES/deep-learning step.")

    if blocking_issues:
        decision = "no_go"
    elif warnings or manual_items:
        decision = "conditional_go"
    else:
        decision = "go"

    answers = {
        "human_confirmed_geometry_sign_fixed_minus_one": geometry_sign_confirmed,
        "most_reasonable_kernel": best_kernel,
        "audit_only_kernels": audit_only_kernels,
        "continuous_target_healthier_than_old_binary": continuous_health[
            "continuous_target_healthier_than_old_binary"
        ],
        "manual_review_required": True,
        "intervals_requiring_manual_review": selected,
        "raw_zc_correctly_controlled": raw_zc["raw_zc_controlled"],
        "raw_zc_source": raw_zc["raw_zc_source"],
        "sample_support_collapse": collapse,
        "still_forbidden": {
            "mvp4c": True,
            "stc": True,
            "apes": True,
            "deep_learning": True,
            "final_labels": True,
            "production_claims": True,
        },
    }
    evidence = {
        "best_kernel": best_kernel,
        "audit_recommendation": regression_audit.get("recommendation"),
        "review_primary_kernel": review_summary.get("primary_kernel"),
        "review_interval_count": review_summary.get("interval_count"),
        "figure_count": review_summary.get("figure_count"),
        "target_health": continuous_health,
        "raw_zc_status": raw_zc,
    }
    return {
        "gate_version": GEOMETRY_REGRESSION_GATE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "decision": decision,
        "answers": answers,
        "evidence": evidence,
        "blocking_issues": _dedupe(blocking_issues),
        "warnings": _dedupe(warnings),
        "manual_confirmation_required": True,
        "manual_confirmation_items": _dedupe(manual_items),
        "next_step_must_wait_human_review": True,
        "mvp4c_allowed": False,
        "stc_allowed": False,
        "apes_allowed": False,
        "deep_learning_allowed": False,
        "final_labels_allowed": False,
        "production_claims_allowed": False,
        "recommendation": _recommendation_text(decision, answers, blocking_issues, warnings),
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


def write_geometry_regression_gate_outputs(
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
    output_md.write_text(format_geometry_regression_gate_markdown(report), encoding="utf-8")


def format_geometry_regression_gate_markdown(report: dict[str, Any]) -> str:
    answers = _as_dict(report.get("answers"))
    lines = [
        "# Geometry-Aware Regression Gate",
        "",
        "This gate summarizes MVP-4B-GR regression weak-label candidate readiness. "
        "It does not authorize MVP-4C, STC/APES, deep learning, final labels, "
        "production claims, or ground-truth claims. The next step must wait for "
        "human review.",
        "",
        f"- decision: `{report['decision']}`",
        "- human_confirmed_geometry_sign_fixed_minus_one: "
        f"`{answers.get('human_confirmed_geometry_sign_fixed_minus_one')}`",
        f"- most_reasonable_kernel: `{answers.get('most_reasonable_kernel')}`",
        f"- audit_only_kernels: `{answers.get('audit_only_kernels')}`",
        "- continuous_target_healthier_than_old_binary: "
        f"`{answers.get('continuous_target_healthier_than_old_binary')}`",
        f"- manual_review_required: `{answers.get('manual_review_required')}`",
        f"- raw_zc_correctly_controlled: `{answers.get('raw_zc_correctly_controlled')}`",
        f"- sample_support_collapse: `{answers.get('sample_support_collapse')}`",
        f"- final_labels_allowed: `{report['final_labels_allowed']}`",
        "",
        "## Intervals Requiring Manual Review",
        "",
    ]
    lines.extend(_message_lines(_as_list(answers.get("intervals_requiring_manual_review"))))
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(_message_lines(_as_list(report.get("blocking_issues"))))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(_as_list(report.get("warnings"))))
    lines.extend(["", "## Manual Confirmation Items", ""])
    lines.extend(_message_lines(_as_list(report.get("manual_confirmation_items"))))
    lines.extend(["", "## Recommendation", "", str(report.get("recommendation", "")), ""])
    lines.extend(["## Not Performed", ""])
    lines.extend(_message_lines(_as_list(report.get("not_performed"))))
    lines.append("")
    return "\n".join(lines)


def _geometry_sign_confirmed(report: dict[str, Any]) -> bool:
    status = _as_dict(report.get("geometry_sign_status"))
    return (
        int(status.get("depth_axis_sign", 0)) == -1
        and status.get("sign_convention_status") == "human_confirmed"
        and status.get("relbearing_plus_minus_independent") is True
    )


def _raw_zc_status(report: dict[str, Any]) -> dict[str, Any]:
    source = _as_dict(report.get("raw_zc_source"))
    field = str(source.get("source_field", ""))
    controlled = (
        report.get("raw_zc_available") is True
        and bool(field)
        and field not in FORBIDDEN_RAW_ZC_SOURCE_FIELDS
    )
    return {
        "raw_zc_controlled": controlled,
        "raw_zc_source": {
            "source_file": source.get("source_file"),
            "source_field": field,
            "finite_ratio": source.get("finite_ratio"),
            "shape": source.get("shape"),
        },
    }


def _best_kernel(
    audit: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> str | None:
    best = audit.get("best_kernel")
    kernels = {str(row.get("geometry_kernel")) for row in summaries}
    if isinstance(best, str) and best in kernels:
        return best
    runnable = [
        row
        for row in summaries
        if row.get("status") == "runnable"
        and _as_float(row.get("real_minus_permutation_margin")) is not None
    ]
    if not runnable:
        return None
    return str(
        max(runnable, key=lambda row: float(row["real_minus_permutation_margin"]))[
            "geometry_kernel"
        ]
    )


def _audit_only_kernels(
    audit_summaries: list[dict[str, Any]],
    label_summaries: list[dict[str, Any]],
) -> list[str]:
    audit_only: set[str] = set()
    for row in audit_summaries:
        kernel = str(row.get("geometry_kernel"))
        distribution = _as_dict(row.get("target_distribution"))
        median = _as_float(distribution.get("median"))
        p90 = _as_float(distribution.get("p90"))
        if bool(row.get("sample_support_collapse")):
            audit_only.add(kernel)
        if "uniform_source_receiver_interval" == kernel and median is not None and p90 is not None:
            if abs(p90 - median) < 1e-6:
                audit_only.add(kernel)
    for row in label_summaries:
        kernel = str(row.get("geometry_kernel"))
        zero = _as_float(row.get("zero_fraction"))
        nonzero = _as_float(row.get("nonzero_fraction"))
        if zero == 1.0 or nonzero == 1.0:
            audit_only.add(kernel)
    return sorted(audit_only)


def _sample_support_collapse(
    audit_summaries: list[dict[str, Any]],
    label_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    collapsed: set[str] = set()
    kernels: set[str] = set()
    for row in audit_summaries:
        kernel = str(row.get("geometry_kernel"))
        kernels.add(kernel)
        if bool(row.get("sample_support_collapse")):
            collapsed.add(kernel)
    for row in label_summaries:
        kernel = str(row.get("geometry_kernel"))
        kernels.add(kernel)
        zero = _as_float(row.get("zero_fraction"))
        nonzero = _as_float(row.get("nonzero_fraction"))
        if zero == 1.0 or nonzero == 1.0:
            collapsed.add(kernel)
    return {
        "any_kernel_collapsed": bool(collapsed),
        "all_kernels_collapsed": bool(kernels) and collapsed == kernels,
        "collapsed_kernels": sorted(collapsed),
    }


def _continuous_target_health(label_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    healthy = []
    for row in label_summaries:
        zero = _as_float(row.get("zero_fraction"))
        nonzero = _as_float(row.get("nonzero_fraction"))
        distribution = _as_dict(row.get("distribution"))
        median = _as_float(distribution.get("median"))
        p95 = _as_float(distribution.get("p95"))
        if (
            zero is not None
            and nonzero is not None
            and 0.0 < zero < 1.0
            and 0.0 < nonzero < 1.0
            and p95 is not None
            and median is not None
            and p95 > median
        ):
            healthy.append(str(row.get("geometry_kernel")))
    return {
        "continuous_target_healthier_than_old_binary": bool(healthy),
        "healthy_kernels": healthy,
    }


def _all_kernels_below_permutation(summaries: list[dict[str, Any]]) -> bool:
    runnable = [row for row in summaries if row.get("status") == "runnable"]
    if not runnable:
        return True
    return all(
        (_as_float(row.get("real_minus_permutation_margin")) or -1.0) <= 0.0
        for row in runnable
    )


def _any_depends_on_5700(summaries: list[dict[str, Any]]) -> bool:
    return any(bool(row.get("depends_on_5700_band")) for row in summaries)


def _all_depend_on_5700(summaries: list[dict[str, Any]]) -> bool:
    runnable = [row for row in summaries if row.get("status") == "runnable"]
    return bool(runnable) and all(bool(row.get("depends_on_5700_band")) for row in runnable)


def _leakage_warnings(summaries: list[dict[str, Any]], audit: dict[str, Any]) -> list[str]:
    warnings = [
        str(message)
        for message in _as_list(audit.get("warnings"))
        if "leakage" in str(message).lower()
    ]
    for row in summaries:
        for message in _as_list(row.get("leakage_warnings")):
            if "leakage" in str(message).lower():
                warnings.append(str(message))
    return _dedupe(warnings)


def _selected_interval_ids(selected_intervals: list[dict[str, Any]]) -> list[str]:
    rows = sorted(
        selected_intervals,
        key=lambda row: float(row.get("target_fraction") or 0.0),
        reverse=True,
    )
    return [str(row.get("review_id")) for row in rows[:20] if row.get("review_id")]


def _recommendation_text(
    decision: str,
    answers: dict[str, Any],
    blocking_issues: list[str],
    warnings: list[str],
) -> str:
    if decision == "no_go":
        return (
            "Do not proceed. Resolve blockers and complete human review first: "
            + "; ".join(blocking_issues)
        )
    if decision == "conditional_go":
        return (
            "Regression review artifacts are usable only for human review. "
            f"Most reasonable kernel candidate: {answers.get('most_reasonable_kernel')}. "
            "MVP-4C/STC/APES/deep learning/final labels remain forbidden. Warnings: "
            + "; ".join(warnings)
        )
    return (
        "Regression gate is go for manual review only. Human approval is still required "
        "before any next-stage experiment."
    )


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionGateCliError("data.reports is not configured.")


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
        raise GeometryRegressionGateCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionGateCliError(
            f"Refusing to {action} geometry regression gate path outside data.{key}: {path}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GeometryRegressionGateCliError(f"JSON must contain an object: {path}")
    return data


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise GeometryRegressionGateCliError(f"JSON must contain a list: {path}")
    return [row for row in data if isinstance(row, dict)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in {float("inf"), float("-inf")} else None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
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
