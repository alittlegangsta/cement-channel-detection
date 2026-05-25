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
from cement_channel.training.baseline_schema import MVP4B_STAGE2_GATE_VERSION  # noqa: E402


class MVP4BStage2GateReportError(RuntimeError):
    """Raised when the MVP-4B Stage 2 gate report cannot be generated safely."""


MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MVP-4B Stage 2 gate report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--simple-baseline-report", default=None)
    parser.add_argument("--baseline-review-summary", default=None)
    parser.add_argument("--mvp4b-stage1-gate-report", default=None)
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
        for key, path in paths.items():
            if key.startswith("output_"):
                _ensure_path_within(config, path, key="reports", action="write")
            else:
                _ensure_path_within(config, path, key="reports", action="read")
        report = _build_gate_report(paths)
        markdown = _format_markdown(report)
        if not args.dry_run:
            _write_outputs(
                report,
                markdown,
                output_md=paths["output_report_md"],
                output_json=paths["output_report_json"],
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        MVP4BStage2GateReportError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4B Stage 2 gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"mvp4c_allowed={report['mvp4c_allowed']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    reports = Path(str(_as_dict(config.get("data")).get("reports", "")))
    if not str(reports):
        raise MVP4BStage2GateReportError("data.reports is not configured.")
    return {
        "simple_baseline_report": Path(
            args.simple_baseline_report or reports / "simple_baseline_report_v001.json"
        ),
        "baseline_review_summary": Path(
            args.baseline_review_summary
            or reports
            / "simple_baseline_review_v001"
            / "simple_baseline_review_summary_v001.json"
        ),
        "mvp4b_stage1_gate_report": Path(
            args.mvp4b_stage1_gate_report or reports / "mvp4b_stage1_gate_report.json"
        ),
        "output_report_md": Path(
            args.output_report_md or reports / "mvp4b_stage2_gate_report.md"
        ),
        "output_report_json": Path(
            args.output_report_json or reports / "mvp4b_stage2_gate_report.json"
        ),
    }


def _build_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    statuses = {
        "mvp4b_stage1_gate": _stage1_status(paths["mvp4b_stage1_gate_report"]),
        "simple_baseline": _simple_baseline_status(paths["simple_baseline_report"]),
        "baseline_review": _review_status(paths["baseline_review_summary"]),
    }
    blocking: list[str] = []
    warnings: list[str] = []
    for name, status in statuses.items():
        blocking.extend(f"{name}: {message}" for message in status["errors"])
        warnings.extend(f"{name}: {message}" for message in status["warnings"])

    baseline = statuses["simple_baseline"]["data"]
    review = statuses["baseline_review"]["data"]
    class_balance = _as_dict(baseline.get("class_balance"))
    candidate_count = _as_int(class_balance.get("candidate_count")) or 0
    non_candidate_count = _as_int(class_balance.get("non_candidate_count")) or 0
    depth_split_valid = _depth_split_valid(baseline)
    high_confidence_sufficient = (
        candidate_count >= MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS
        and non_candidate_count >= MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS
    )
    permutation_passes = _permutation_passes(baseline)
    leakage_absent = not bool(baseline.get("leakage_suspected"))
    interpretable = bool(_as_dict(baseline.get("coefficient_summary")))
    plus_minus_documented = bool(_as_dict(baseline.get("minus_audit_comparison")))
    no_final_labels = all(
        [
            _data_bool(baseline, "no_final_labels"),
            _data_bool(review, "no_final_labels"),
        ]
    )
    no_production_model = all(
        [
            baseline.get("production_training") is False,
            _data_bool(baseline, "no_production_model"),
            _data_bool(review, "no_production_model"),
        ]
    )
    no_forbidden_methods = all(
        [
            _data_bool(baseline, "no_deep_learning"),
            _data_bool(baseline, "no_stc"),
            _data_bool(baseline, "no_apes"),
            _data_bool(review, "no_deep_learning"),
            _data_bool(review, "no_stc"),
            _data_bool(review, "no_apes"),
        ]
    )
    stage1_allowed = _stage1_allowed(statuses["mvp4b_stage1_gate"]["data"])

    if not stage1_allowed:
        blocking.append("mvp4b_stage1_gate: Stage 1 gate does not allow Stage 2.")
    if not depth_split_valid:
        blocking.append("simple_baseline: depth-block split is invalid.")
    if not high_confidence_sufficient:
        blocking.append(
            "simple_baseline: high-confidence subset is insufficient: "
            f"candidate={candidate_count}, non_candidate={non_candidate_count}."
        )
    if not permutation_passes:
        blocking.append(
            "simple_baseline: metrics are not above the permutation sanity baseline."
        )
    if not leakage_absent:
        blocking.append("simple_baseline: leakage_suspected is true.")
    if not interpretable:
        blocking.append("simple_baseline: coefficient summary is missing.")
    if not plus_minus_documented:
        blocking.append("simple_baseline: plus primary vs minus audit comparison is missing.")
    if not no_final_labels:
        blocking.append("mvp4b_stage2: reports indicate final labels or missing guard flag.")
    if not no_production_model:
        blocking.append("mvp4b_stage2: reports indicate production model/training.")
    if not no_forbidden_methods:
        blocking.append("mvp4b_stage2: report permits deep learning, STC, or APES.")

    decision = _decision(blocking, warnings)
    mvp4c_allowed = decision in {"go", "conditional_go"} and not blocking
    return {
        "gate_version": MVP4B_STAGE2_GATE_VERSION,
        "stage": "MVP-4B Stage 2",
        "task": "simple_baseline_sanity_model_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "statuses": {key: _public_status(value) for key, value in statuses.items()},
        "gate_conditions": {
            "stage1_allowed": stage1_allowed,
            "depth_block_split_valid": depth_split_valid,
            "candidate_count": candidate_count,
            "non_candidate_count": non_candidate_count,
            "high_confidence_subset_sufficient": high_confidence_sufficient,
            "metrics_above_permutation": permutation_passes,
            "no_evidence_of_leakage": leakage_absent,
            "interpretable_coefficients_available": interpretable,
            "plus_minus_audit_documented": plus_minus_documented,
            "no_final_labels": no_final_labels,
            "no_production_model": no_production_model,
            "no_deep_learning_stc_apes": no_forbidden_methods,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "mvp4c_allowed": mvp4c_allowed,
        "next_stage_allowed": (
            "MVP-4C advanced feature engineering only" if mvp4c_allowed else None
        ),
        "not_performed": [
            "production training",
            "production inference",
            "deep learning",
            "STC",
            "APES",
            "final label generation",
            "model weight export",
            "MVP-5",
        ],
    }


def _stage1_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(path, required_keys=["decision"])
    data = status["data"]
    if data.get("decision") not in {"go", "conditional_go"}:
        status["errors"].append("Stage 1 decision must be go or conditional_go.")
    if not _stage1_allowed(data):
        status["errors"].append("Stage 1 report does not allow Stage 2.")
    return status


def _simple_baseline_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=[
            "report_version",
            "class_balance",
            "split",
            "aggregate_metrics",
            "permutation_check",
            "minus_audit_comparison",
            "coefficient_summary",
            "production_training",
            "no_final_labels",
            "no_deep_learning",
            "no_stc",
            "no_apes",
            "no_production_model",
        ],
    )
    data = status["data"]
    if data.get("report_version") != "simple_baseline_v001":
        status["errors"].append("report_version must be simple_baseline_v001.")
    if data.get("production_training") is not False:
        status["errors"].append("production_training must be false.")
    return status


def _review_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=[
            "review_version",
            "figures",
            "review_summary_template",
            "no_final_labels",
            "no_deep_learning",
            "no_stc",
            "no_apes",
            "no_production_model",
        ],
    )
    data = status["data"]
    if data.get("review_version") != "simple_baseline_review_v001":
        status["errors"].append("review_version must be simple_baseline_review_v001.")
    if len(_as_dict(data.get("figures"))) < 7:
        status["warnings"].append("expected 7 simple baseline review figures.")
    return status


def _status_from_report(path: Path, *, required_keys: list[str]) -> dict[str, Any]:
    data = _read_json(path)
    errors = list(_as_list(data.get("errors")))
    warnings = list(_as_list(data.get("warnings")))
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    return {"ready": not errors, "errors": errors, "warnings": warnings, "data": data}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MVP4BStage2GateReportError(f"Required report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MVP4BStage2GateReportError(f"Report must contain a JSON object: {path}")
    return data


def _stage1_allowed(data: dict[str, Any]) -> bool:
    return bool(data.get("mvp4b_stage2_allowed") is True or data.get("stage2_allowed") is True)


def _depth_split_valid(baseline: dict[str, Any]) -> bool:
    split = _as_dict(baseline.get("split"))
    if split.get("method") != "depth_block_group_split":
        return False
    folds = _as_list(split.get("folds"))
    if len(folds) < 2:
        return False
    for fold in folds:
        fold_data = _as_dict(fold)
        if (_as_int(fold_data.get("train_candidate_count")) or 0) <= 0:
            return False
        if (_as_int(fold_data.get("train_non_candidate_count")) or 0) <= 0:
            return False
        if (_as_int(fold_data.get("validation_candidate_count")) or 0) <= 0:
            return False
        if (_as_int(fold_data.get("validation_non_candidate_count")) or 0) <= 0:
            return False
    return True


def _permutation_passes(baseline: dict[str, Any]) -> bool:
    checks = _as_dict(baseline.get("permutation_check"))
    if not checks:
        return False
    for value in checks.values():
        check = _as_dict(value)
        if check.get("passes_margin") is not True:
            return False
        real = _as_float(check.get("real_balanced_accuracy"))
        permuted = _as_float(check.get("permutation_balanced_accuracy"))
        if real is None or permuted is None or real <= permuted:
            return False
    return True


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4B Stage 2 Gate Report",
        "",
        f"- Version: {report['gate_version']}",
        f"- Generated at: {report['generated_at']}",
        f"- Decision: {report['decision']}",
        f"- MVP-4C allowed: {report['mvp4c_allowed']}",
        f"- Next stage allowed: {report['next_stage_allowed']}",
        "",
        "## Gate Conditions",
        "",
    ]
    for key, value in report["gate_conditions"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Statuses", ""])
    for name, status in report["statuses"].items():
        lines.append(
            f"- {name}: ready={status['ready']}, "
            f"errors={len(status['errors'])}, warnings={len(status['warnings'])}"
        )
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(_message_lines(report["blocking_issues"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report["warnings"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report["not_performed"]))
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


def _decision(blocking: list[str], warnings: list[str]) -> str:
    if blocking:
        return "no_go"
    if warnings:
        return "conditional_go"
    return "go"


def _public_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "ready": bool(status["ready"]),
        "errors": list(status["errors"]),
        "warnings": list(status["warnings"]),
    }


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _ensure_path_within(
    config: dict[str, Any],
    path: Path,
    *,
    key: str,
    action: str,
) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key, ""))).resolve()
    if not str(root):
        raise MVP4BStage2GateReportError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise MVP4BStage2GateReportError(
            f"Refusing to {action} MVP-4B Stage 2 gate path outside data.{key}: {path}"
        ) from exc


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _data_bool(data: dict[str, Any], key: str) -> bool:
    return bool(data.get(key) is True)


if __name__ == "__main__":
    raise SystemExit(main())
