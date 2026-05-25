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


class MVP4AGateReportError(RuntimeError):
    """Raised when the MVP-4A gate report cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MVP-4A gate report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--label-samples-report", default=None)
    parser.add_argument("--basic-features-report", default=None)
    parser.add_argument("--correlation-report", default=None)
    parser.add_argument("--review-summary", default=None)
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
        MVP4AGateReportError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4A gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"mvp4b_allowed={report['mvp4b_allowed']}."
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
        raise MVP4AGateReportError("data.reports is not configured.")
    return {
        "label_samples_report": Path(
            args.label_samples_report or reports / "xsi_label_samples_report_v001.json"
        ),
        "basic_features_report": Path(
            args.basic_features_report or reports / "xsi_basic_features_report_v001.json"
        ),
        "correlation_report": Path(
            args.correlation_report or reports / "xsi_cast_correlation_report_v001.json"
        ),
        "review_summary": Path(
            args.review_summary
            or reports / "mvp4a_review_v001" / "mvp4a_review_summary_v001.json"
        ),
        "output_report_md": Path(args.output_report_md or reports / "mvp4a_gate_report.md"),
        "output_report_json": Path(args.output_report_json or reports / "mvp4a_gate_report.json"),
    }


def _build_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    statuses = {
        "label_sample_index": _label_sample_status(paths["label_samples_report"]),
        "xsi_basic_features": _basic_feature_status(paths["basic_features_report"]),
        "xsi_cast_correlation": _correlation_status(paths["correlation_report"]),
        "review_figures": _review_status(paths["review_summary"]),
    }
    blocking: list[str] = []
    warnings: list[str] = []
    for name, status in statuses.items():
        blocking.extend(f"{name}: {message}" for message in status["errors"])
        warnings.extend(f"{name}: {message}" for message in status["warnings"])

    correlation = statuses["xsi_cast_correlation"]["data"]
    observations = _as_dict(correlation.get("gate_observations"))
    high_confidence_subset_exists = _data_bool(observations, "high_confidence_subset_exists")
    interpretable_signal = _data_bool(observations, "interpretable_signal_separation")
    low_confidence_policy_respected = _data_bool(observations, "low_confidence_policy_respected")
    no_model_training = all(
        [
            _data_bool(statuses["xsi_basic_features"]["data"], "no_model_training"),
            _data_bool(correlation, "no_model_training"),
            _data_bool(statuses["review_figures"]["data"], "no_model_training"),
        ]
    )
    no_final_labels = all(
        [
            _data_bool(statuses["label_sample_index"]["data"], "no_final_labels"),
            _data_bool(correlation, "no_final_labels"),
            _data_bool(statuses["review_figures"]["data"], "no_final_labels"),
        ]
    )
    if not high_confidence_subset_exists:
        blocking.append("xsi_cast_correlation: high-confidence subset is too small or missing.")
    if not interpretable_signal:
        blocking.append(
            "xsi_cast_correlation: candidate vs non-candidate has no interpretable separation."
        )
    if not low_confidence_policy_respected:
        blocking.append("xsi_cast_correlation: low-confidence policy was not respected.")
    if not no_model_training:
        blocking.append("mvp4a: one or more reports indicate model training or missing guard flag.")
    if not no_final_labels:
        blocking.append("mvp4a: one or more reports indicate final labels or missing guard flag.")

    decision = _decision(blocking, warnings)
    mvp4b_allowed = decision in {"go", "conditional_go"} and not blocking
    return {
        "stage": "MVP-4A",
        "task": "xsi_cast_weak_label_correlation_validation_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "statuses": {key: _public_status(value) for key, value in statuses.items()},
        "gate_conditions": {
            "xsi_basic_features_extracted": statuses["xsi_basic_features"]["ready"],
            "label_sample_index_valid": statuses["label_sample_index"]["ready"],
            "high_confidence_subset_exists": high_confidence_subset_exists,
            "interpretable_signal_separation": interpretable_signal,
            "low_confidence_policy_respected": low_confidence_policy_respected,
            "no_model_training": no_model_training,
            "no_final_labels": no_final_labels,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "mvp4b_allowed": mvp4b_allowed,
        "next_stage_allowed": "MVP-4B" if mvp4b_allowed else None,
        "not_performed": [
            "model training",
            "STC",
            "APES",
            "final label generation",
            "MVP-4B feature engineering",
            "MVP-5 baseline modeling",
        ],
    }


def _label_sample_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=["sample_index_version", "shape", "coverage", "no_final_labels"],
    )
    data = status["data"]
    if data.get("sample_index_version") != "xsi_label_samples_v001":
        status["errors"].append("sample_index_version must be xsi_label_samples_v001.")
    if not _data_bool(data, "no_final_labels"):
        status["errors"].append("no_final_labels is not true.")
    coverage = _as_dict(data.get("coverage"))
    if _as_int(coverage.get("valid_for_non_azimuthal_summary_count")) is None:
        status["errors"].append("valid_for_non_azimuthal_summary_count is missing.")
    if _as_int(coverage.get("valid_for_azimuthal_validation_count")) is None:
        status["errors"].append("valid_for_azimuthal_validation_count is missing.")
    az_count = _as_int(coverage.get("valid_for_azimuthal_validation_count")) or 0
    non_az_count = _as_int(coverage.get("valid_for_non_azimuthal_summary_count")) or 0
    if non_az_count < az_count:
        status["errors"].append("non-azimuthal valid count is smaller than azimuthal valid count.")
    return status


def _basic_feature_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=["feature_version", "summaries", "no_model_training", "no_stc", "no_apes"],
    )
    data = status["data"]
    if data.get("feature_version") != "xsi_basic_features_v001":
        status["errors"].append("feature_version must be xsi_basic_features_v001.")
    if not _data_bool(data, "no_model_training"):
        status["errors"].append("no_model_training is not true.")
    if not _data_bool(data, "no_stc"):
        status["errors"].append("no_stc is not true.")
    if not _data_bool(data, "no_apes"):
        status["errors"].append("no_apes is not true.")
    summaries = _as_dict(data.get("summaries"))
    feature_summary = _as_dict(summaries.get("xsi_basic_features_by_side"))
    finite_ratio = _as_float(feature_summary.get("finite_ratio"))
    if finite_ratio is None:
        status["errors"].append("xsi_basic_features_by_side finite_ratio is missing.")
    elif finite_ratio < 0.50:
        status["errors"].append(
            f"xsi_basic_features_by_side is mostly non-finite: finite_ratio={finite_ratio}."
        )
    elif finite_ratio < 0.95:
        status["warnings"].append(
            f"xsi_basic_features_by_side finite_ratio is below 0.95: {finite_ratio}."
        )
    return status


def _correlation_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=[
            "correlation_version",
            "gate_observations",
            "subset_counts",
            "no_model_training",
            "no_final_labels",
        ],
    )
    data = status["data"]
    if data.get("correlation_version") != "xsi_cast_correlation_v001":
        status["errors"].append("correlation_version must be xsi_cast_correlation_v001.")
    if not _data_bool(data, "no_model_training"):
        status["errors"].append("no_model_training is not true.")
    if not _data_bool(data, "no_final_labels"):
        status["errors"].append("no_final_labels is not true.")
    observations = _as_dict(data.get("gate_observations"))
    if "high_confidence_subset_exists" not in observations:
        status["errors"].append("gate_observations.high_confidence_subset_exists is missing.")
    if "interpretable_signal_separation" not in observations:
        status["errors"].append("gate_observations.interpretable_signal_separation is missing.")
    return status


def _review_status(path: Path) -> dict[str, Any]:
    status = _status_from_report(
        path,
        required_keys=["review_version", "figures", "no_model_training", "no_final_labels"],
    )
    data = status["data"]
    if data.get("review_version") != "mvp4a_review_v001":
        status["errors"].append("review_version must be mvp4a_review_v001.")
    if not _data_bool(data, "no_model_training"):
        status["errors"].append("no_model_training is not true.")
    if not _data_bool(data, "no_final_labels"):
        status["errors"].append("no_final_labels is not true.")
    if len(_as_dict(data.get("figures"))) < 7:
        status["warnings"].append("expected 7 MVP-4A review figures.")
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
        raise MVP4AGateReportError(f"Required report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MVP4AGateReportError(f"Report must contain a JSON object: {path}")
    return data


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4A Gate Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Decision: {report['decision']}",
        f"- MVP-4B allowed: {report['mvp4b_allowed']}",
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
        raise MVP4AGateReportError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise MVP4AGateReportError(
            f"Refusing to {action} MVP-4A gate path outside data.{key}: {path}"
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
