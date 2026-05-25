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
from cement_channel.training.sample_schema import MVP4B_STAGE1_GATE_VERSION  # noqa: E402


class MVP4BStage1GateReportError(RuntimeError):
    """Raised when the MVP-4B Stage 1 gate report cannot be generated safely."""


MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS = 20
MAX_NONFINITE_TRANSFORMED_FRACTION = 0.001
MIN_POSITIVE_SAMPLE_WEIGHT_FRACTION = 0.05
MAX_LARGE_DEPTH_MATCH_ERROR_FRACTION = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MVP-4B Stage 1 gate report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--sample-table-report", default=None)
    parser.add_argument("--preprocessing-diagnostics", default=None)
    parser.add_argument("--mvp4a-gate-report", default=None)
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
        MVP4BStage1GateReportError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4B Stage 1 gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"stage2_allowed={report['mvp4b_stage2_allowed']}."
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
    if not str(reports):
        raise MVP4BStage1GateReportError("data.reports is not configured.")
    return {
        "sample_table_report": Path(
            args.sample_table_report or reports / "baseline_sample_table_report_v001.json"
        ),
        "preprocessing_diagnostics": Path(
            args.preprocessing_diagnostics
            or reports / "feature_preprocessing_diagnostics_v001.json"
        ),
        "mvp4a_gate_report": Path(args.mvp4a_gate_report or reports / "mvp4a_gate_report.json"),
        "output_report_md": Path(
            args.output_report_md or reports / "mvp4b_stage1_gate_report.md"
        ),
        "output_report_json": Path(
            args.output_report_json or reports / "mvp4b_stage1_gate_report.json"
        ),
    }


def _build_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    statuses = {
        "mvp4a_gate": _mvp4a_status(paths["mvp4a_gate_report"]),
        "sample_table": _sample_table_status(paths["sample_table_report"]),
        "preprocessing_diagnostics": _preprocessing_status(
            paths["preprocessing_diagnostics"]
        ),
    }
    blocking: list[str] = []
    warnings: list[str] = []
    for name, status in statuses.items():
        blocking.extend(f"{name}: {message}" for message in status["errors"])
        warnings.extend(f"{name}: {message}" for message in status["warnings"])

    sample_data = statuses["sample_table"]["data"]
    preprocessing_data = statuses["preprocessing_diagnostics"]["data"]
    counts = _as_dict(sample_data.get("counts"))
    excluded = _as_dict(sample_data.get("excluded_counts"))
    total_samples = _as_int(_as_dict(sample_data.get("shape")).get("samples")) or 0
    high_confidence_candidate_count = _as_int(counts.get("high_confidence_candidate_count")) or 0
    high_confidence_non_candidate_count = (
        _as_int(counts.get("high_confidence_non_candidate_count")) or 0
    )
    positive_weight_fraction = _as_float(counts.get("positive_sample_weight_fraction"))
    positive_weight_count = _as_int(counts.get("positive_sample_weight_count")) or 0
    large_depth_error_count = _as_int(excluded.get("exclude_large_depth_match_error")) or 0
    large_depth_error_fraction = (
        large_depth_error_count / total_samples if total_samples > 0 else None
    )
    transformed_features_finite = _transformed_features_finite(sample_data, preprocessing_data)
    sample_weight_valid = _sample_weight_valid(sample_data)
    depth_match_error_policy_applied = "exclude_large_depth_match_error" in excluded
    plus_minus_disagreement_preserved = "plus_minus_disagreement_fraction" in counts
    no_model_training = all(
        [
            _data_bool(sample_data, "no_model_training"),
            _data_bool(preprocessing_data, "no_model_training"),
            _not_performed_contains(sample_data, "model training"),
            _not_performed_contains(preprocessing_data, "model training"),
        ]
    )
    no_final_labels = all(
        [
            _data_bool(sample_data, "no_final_labels"),
            _data_bool(preprocessing_data, "no_final_labels"),
            _not_performed_contains(sample_data, "final label generation"),
            _not_performed_contains(preprocessing_data, "final label generation"),
        ]
    )
    mvp4a_allows_mvp4b = _data_bool(statuses["mvp4a_gate"]["data"], "mvp4b_allowed")

    if high_confidence_candidate_count <= 0:
        blocking.append("sample_table: high-confidence candidate subset is empty.")
    elif high_confidence_candidate_count < MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS:
        blocking.append(
            "sample_table: high-confidence candidate count is below "
            f"{MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS}: {high_confidence_candidate_count}."
        )
    if high_confidence_non_candidate_count <= 0:
        blocking.append("sample_table: high-confidence non-candidate subset is empty.")
    elif high_confidence_non_candidate_count < MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS:
        blocking.append(
            "sample_table: high-confidence non-candidate count is below "
            f"{MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS}: {high_confidence_non_candidate_count}."
        )
    if not transformed_features_finite:
        blocking.append("preprocessing_diagnostics: transformed features are not finite enough.")
    if not sample_weight_valid:
        blocking.append("sample_table: sample_weight is invalid or all zero.")
    if positive_weight_count <= 0:
        blocking.append("sample_table: positive sample_weight count is zero.")
    if positive_weight_fraction is not None and (
        positive_weight_fraction < MIN_POSITIVE_SAMPLE_WEIGHT_FRACTION
    ):
        blocking.append(
            "sample_table: positive sample_weight fraction is below "
            f"{MIN_POSITIVE_SAMPLE_WEIGHT_FRACTION}: {positive_weight_fraction}."
        )
    if not depth_match_error_policy_applied:
        blocking.append("sample_table: depth_match_error exclusion policy was not reported.")
    if (
        large_depth_error_fraction is not None
        and large_depth_error_fraction > MAX_LARGE_DEPTH_MATCH_ERROR_FRACTION
    ):
        blocking.append(
            "sample_table: large depth_match_error fraction exceeds "
            f"{MAX_LARGE_DEPTH_MATCH_ERROR_FRACTION}: {large_depth_error_fraction}."
        )
    if not plus_minus_disagreement_preserved:
        blocking.append("sample_table: plus/minus disagreement audit flag was not preserved.")
    if not no_model_training:
        blocking.append("mvp4b_stage1: reports indicate model training or missing guard flag.")
    if not no_final_labels:
        blocking.append("mvp4b_stage1: reports indicate final labels or missing guard flag.")
    if not mvp4a_allows_mvp4b:
        blocking.append("mvp4a_gate: MVP-4A does not allow MVP-4B.")

    decision = _decision(blocking, warnings)
    stage2_allowed = decision in {"go", "conditional_go"} and not blocking
    return {
        "gate_version": MVP4B_STAGE1_GATE_VERSION,
        "stage": "MVP-4B Stage 1",
        "task": "baseline_sample_table_preprocessing_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "statuses": {key: _public_status(value) for key, value in statuses.items()},
        "thresholds": {
            "min_high_confidence_samples_per_class": MIN_HIGH_CONFIDENCE_SAMPLES_PER_CLASS,
            "max_nonfinite_transformed_fraction": MAX_NONFINITE_TRANSFORMED_FRACTION,
            "min_positive_sample_weight_fraction": MIN_POSITIVE_SAMPLE_WEIGHT_FRACTION,
            "max_large_depth_match_error_fraction": MAX_LARGE_DEPTH_MATCH_ERROR_FRACTION,
        },
        "gate_conditions": {
            "mvp4a_allows_mvp4b": mvp4a_allows_mvp4b,
            "sample_table_built_successfully": statuses["sample_table"]["ready"],
            "high_confidence_candidate_count": high_confidence_candidate_count,
            "high_confidence_non_candidate_count": high_confidence_non_candidate_count,
            "transformed_features_finite": transformed_features_finite,
            "sample_weight_valid": sample_weight_valid,
            "positive_sample_weight_fraction": positive_weight_fraction,
            "depth_match_error_policy_applied": depth_match_error_policy_applied,
            "large_depth_match_error_fraction": large_depth_error_fraction,
            "plus_minus_disagreement_preserved": plus_minus_disagreement_preserved,
            "no_model_training": no_model_training,
            "no_final_labels": no_final_labels,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "mvp4b_stage2_allowed": stage2_allowed,
        "next_stage_allowed": (
            "MVP-4B Stage 2 simple baseline sanity model" if stage2_allowed else None
        ),
        "not_performed": [
            "model training",
            "train/test split",
            "production inference",
            "deep learning",
            "STC",
            "APES",
            "final label generation",
            "MVP-4C",
            "MVP-5",
        ],
    }


def _mvp4a_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=["decision", "mvp4b_allowed", "gate_conditions"],
    )
    data = status["data"]
    if data.get("decision") not in {"go", "conditional_go"}:
        status["errors"].append("MVP-4A decision must be go or conditional_go.")
    if not _data_bool(data, "mvp4b_allowed"):
        status["errors"].append("mvp4b_allowed is not true.")
    return status


def _sample_table_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=[
            "sample_table_version",
            "shape",
            "counts",
            "excluded_counts",
            "transformed_feature_ranges",
            "sample_weight",
            "no_model_training",
            "no_final_labels",
            "not_performed",
        ],
    )
    data = status["data"]
    if data.get("sample_table_version") != "baseline_sample_table_v001":
        status["errors"].append("sample_table_version must be baseline_sample_table_v001.")
    shape = _as_dict(data.get("shape"))
    if (_as_int(shape.get("samples")) or 0) <= 0:
        status["errors"].append("sample count must be positive.")
    if (_as_int(shape.get("features")) or 0) <= 0:
        status["errors"].append("feature count must be positive.")
    if (_as_int(shape.get("transformed_features")) or 0) <= 0:
        status["errors"].append("transformed feature count must be positive.")
    if not _data_bool(data, "no_model_training"):
        status["errors"].append("no_model_training is not true.")
    if not _data_bool(data, "no_final_labels"):
        status["errors"].append("no_final_labels is not true.")
    return status


def _preprocessing_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=[
            "diagnostics_version",
            "figures",
            "nonfinite_counts",
            "sample_weight",
            "depth_match_error",
            "no_model_training",
            "no_final_labels",
            "not_performed",
        ],
    )
    data = status["data"]
    if data.get("diagnostics_version") != "feature_preprocessing_diagnostics_v001":
        status["errors"].append(
            "diagnostics_version must be feature_preprocessing_diagnostics_v001."
        )
    if len(_as_dict(data.get("figures"))) < 5:
        status["warnings"].append("expected 5 MVP-4B preprocessing diagnostic figures.")
    if not _data_bool(data, "no_model_training"):
        status["errors"].append("no_model_training is not true.")
    if not _data_bool(data, "no_final_labels"):
        status["errors"].append("no_final_labels is not true.")
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
        raise MVP4BStage1GateReportError(f"Required report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MVP4BStage1GateReportError(f"Report must contain a JSON object: {path}")
    return data


def _transformed_features_finite(
    sample_table: dict[str, Any],
    diagnostics: dict[str, Any],
) -> bool:
    sample_ranges = _as_dict(sample_table.get("transformed_feature_ranges"))
    if not sample_ranges:
        return False
    for summary in sample_ranges.values():
        finite_ratio = _as_float(_as_dict(summary).get("finite_ratio"))
        if finite_ratio is None:
            return False
        if (1.0 - finite_ratio) > MAX_NONFINITE_TRANSFORMED_FRACTION:
            return False
    nonfinite_counts = _as_dict(diagnostics.get("nonfinite_counts"))
    transformed_counts = [
        _as_dict(value)
        for key, value in nonfinite_counts.items()
        if str(key).startswith("transformed:")
    ]
    if not transformed_counts:
        return False
    for summary in transformed_counts:
        finite_ratio = _as_float(summary.get("finite_ratio"))
        if finite_ratio is None:
            return False
        if (1.0 - finite_ratio) > MAX_NONFINITE_TRANSFORMED_FRACTION:
            return False
    return True


def _sample_weight_valid(sample_table: dict[str, Any]) -> bool:
    sample_weight = _as_dict(sample_table.get("sample_weight"))
    finite_ratio = _as_float(sample_weight.get("finite_ratio"))
    min_value = _as_float(sample_weight.get("min"))
    max_value = _as_float(sample_weight.get("max"))
    if finite_ratio is None or min_value is None or max_value is None:
        return False
    if finite_ratio < 1.0:
        return False
    if min_value < 0.0 or max_value > 1.0:
        return False
    return max_value > 0.0


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4B Stage 1 Gate Report",
        "",
        f"- Version: {report['gate_version']}",
        f"- Generated at: {report['generated_at']}",
        f"- Decision: {report['decision']}",
        f"- Stage 2 allowed: {report['mvp4b_stage2_allowed']}",
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
    data = _as_dict(config.get("data"))
    root = Path(str(data.get(key, ""))).resolve()
    if not str(root):
        raise MVP4BStage1GateReportError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise MVP4BStage1GateReportError(
            f"Refusing to {action} MVP-4B Stage 1 gate path outside data.{key}: {path}"
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


def _not_performed_contains(data: dict[str, Any], phrase: str) -> bool:
    return any(phrase in str(value) for value in _as_list(data.get("not_performed")))


if __name__ == "__main__":
    raise SystemExit(main())
