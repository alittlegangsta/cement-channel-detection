from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.io_hdf5 import validate_tiny_hdf5_schema  # noqa: E402
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.data.schema import validate_manifest_basic  # noqa: E402


class MVP1GateReportError(RuntimeError):
    """Raised when the MVP-1 gate report cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MVP-1 gate report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mat-metadata", default=None)
    parser.add_argument("--struct-probe", default=None)
    parser.add_argument("--mapping", default="configs/raw_variable_mapping.yaml")
    parser.add_argument("--small-slice-summary", default=None)
    parser.add_argument("--tiny-hdf5", default=None)
    parser.add_argument("--qc-summary", default=None)
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
        _ensure_report_path_is_safe(paths["output_report_md"])
        _ensure_report_path_is_safe(paths["output_report_json"])
        if not args.dry_run:
            _write_outputs(
                report,
                markdown,
                output_md=paths["output_report_md"],
                output_json=paths["output_report_json"],
                overwrite=args.overwrite,
            )
    except (ManifestBuildError, MVP1GateReportError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-1 gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    data = _as_dict(config.get("data"))
    outputs = _as_dict(config.get("outputs"))
    manifests = Path(str(data.get("manifests", "")))
    reports = Path(str(data.get("reports", "")))
    interim = Path(str(data.get("interim", "")))
    processed = Path(str(data.get("processed", "")))
    return {
        "manifest": Path(
            args.manifest
            or outputs.get("data_manifest_json")
            or manifests / "data_manifest_v001.json"
        ),
        "mat_metadata": Path(args.mat_metadata or manifests / "mat_metadata_v001.json"),
        "struct_probe": Path(args.struct_probe or manifests / "mat_struct_probe_v001.json"),
        "mapping": Path(args.mapping),
        "small_slice_summary": Path(
            args.small_slice_summary or interim / "small_slice_summary_v001.json"
        ),
        "tiny_hdf5": Path(args.tiny_hdf5 or processed / "tiny_aligned_prototype_v001.h5"),
        "qc_summary": Path(args.qc_summary or reports / "qc_mvp1" / "qc_summary_v001.json"),
        "output_report_md": Path(args.output_report_md or reports / "mvp1_gate_report.md"),
        "output_report_json": Path(args.output_report_json or reports / "mvp1_gate_report.json"),
    }


def _build_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    statuses: dict[str, dict[str, Any]] = {}
    blocking: list[str] = []
    warnings: list[str] = []

    statuses["manifest"] = _manifest_status(paths["manifest"])
    statuses["mat_metadata"] = _json_status(paths["mat_metadata"], error_key="errors")
    statuses["struct_probe"] = _struct_probe_status(paths["struct_probe"])
    statuses["raw_variable_mapping"] = _mapping_status(paths["mapping"])
    statuses["small_slice"] = _json_status(paths["small_slice_summary"], error_key="errors")
    statuses["tiny_hdf5"] = _tiny_hdf5_status(paths["tiny_hdf5"])
    statuses["initial_qc"] = _qc_summary_status(paths["qc_summary"])

    for name, status in statuses.items():
        for error in status["errors"]:
            blocking.append(f"{name}: {error}")
        for warning in status["warnings"]:
            warnings.append(f"{name}: {warning}")

    decision = _decision(blocking, warnings)
    return {
        "stage": "MVP-1",
        "task": "mvp1_gate_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "statuses": statuses,
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "next_recommended_stage": (
            "MVP-2" if decision in {"go", "conditional_go"} else "fix_mvp1_blockers"
        ),
        "not_performed": [
            "depth alignment",
            "RelBearing rotation",
            "label generation",
            "feature extraction",
            "model training",
        ],
    }


def _manifest_status(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    validation = validate_manifest_basic(data)
    return {
        "status": "pass" if validation.is_valid else "fail",
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "details": {"path": str(path)},
    }


def _json_status(path: Path, *, error_key: str) -> dict[str, Any]:
    data = _read_json(path)
    errors = [str(error) for error in data.get(error_key, []) if error]
    warnings = [str(warning) for warning in data.get("warnings", []) if warning]
    return {
        "status": "fail" if errors else "warning" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "details": {"path": str(path)},
    }


def _struct_probe_status(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    summary = _as_dict(data.get("summary"))
    errors: list[str] = []
    warnings = [str(warning) for warning in data.get("warnings", []) if warning]
    if int(summary.get("file_error_count", 0)) > 0:
        errors.append(f"struct probe file_error_count={summary.get('file_error_count')}")
    if int(summary.get("can_probe_count", 0)) == 0:
        errors.append("no files were probed successfully")
    return {
        "status": "fail" if errors else "warning" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "details": {"path": str(path), "summary": summary},
    }


def _mapping_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "fail",
            "errors": [f"raw variable mapping does not exist: {path}"],
            "warnings": [],
            "details": {"path": str(path)},
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {
            "status": "fail",
            "errors": [f"raw variable mapping must be a YAML mapping: {path}"],
            "warnings": [],
            "details": {"path": str(path)},
        }
    human_review = _as_dict(data.get("human_review"))
    errors: list[str] = []
    warnings: list[str] = []
    if human_review.get("required") is not False:
        errors.append("raw variable mapping has not been marked human-confirmed")
    for uncertainty in human_review.get("remaining_uncertainties", []) or []:
        warnings.append(str(uncertainty))
    return {
        "status": "fail" if errors else "warning" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "details": {"path": str(path), "mapping_version": data.get("mapping_version")},
    }


def _tiny_hdf5_status(path: Path) -> dict[str, Any]:
    validation = validate_tiny_hdf5_schema(path)
    return {
        "status": "fail" if validation.errors else "warning" if validation.warnings else "pass",
        "errors": validation.errors,
        "warnings": validation.warnings,
        "details": {"path": str(path)},
    }


def _qc_summary_status(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    errors = [str(error) for error in data.get("errors", []) if error]
    warnings = [str(warning) for warning in data.get("warnings", []) if warning]
    if data.get("status") == "failed" and not errors:
        errors.append("QC summary status is failed")
    return {
        "status": "fail" if errors else "warning" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "details": {"path": str(path), "status": data.get("status")},
    }


def _decision(blocking: list[str], warnings: list[str]) -> str:
    if blocking:
        return "no_go"
    if warnings:
        return "conditional_go"
    return "go"


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-1 Gate Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Decision: {report['decision']}",
        f"- Next recommended stage: {report['next_recommended_stage']}",
        "",
        "## Status Summary",
        "",
    ]
    for name, status in report["statuses"].items():
        lines.append(f"- {name}: {status['status']}")
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(_items(report["blocking_issues"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_items(report["warnings"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_items(report["not_performed"]))
    lines.extend(["", "## Go / No-Go", ""])
    if report["decision"] == "go":
        lines.append("- Go to MVP-2.")
    elif report["decision"] == "conditional_go":
        lines.append("- Conditional go to MVP-2 after documenting listed uncertainties.")
    else:
        lines.append("- No-go for MVP-2 until blocking issues are fixed.")
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MVP1GateReportError(f"Required input does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MVP1GateReportError(f"JSON input must contain an object: {path}")
    return data


def _items(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MVP1GateReportError(f"Output already exists: {path}. Pass --overwrite.")


def _ensure_report_path_is_safe(path: Path) -> None:
    project_data_dir = (PROJECT_ROOT / "data").resolve()
    try:
        path.resolve().relative_to(project_data_dir)
    except ValueError:
        return
    raise MVP1GateReportError(f"Refusing to write report inside Git data directory: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
