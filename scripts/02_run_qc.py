from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.data.schema import validate_manifest_basic  # noqa: E402
from cement_channel.qc.cast_qc import run_cast_zc_qc, run_pose_range_qc  # noqa: E402
from cement_channel.qc.xsi_qc import run_xsi_waveform_qc  # noqa: E402


class QCSkeletonError(RuntimeError):
    """Raised when the QC skeleton cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight manifest/schema validation without reading .mat contents."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
        help="Path to paths YAML config.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to data_manifest_v001.json. Defaults to paths config outputs.",
    )
    parser.add_argument(
        "--input-hdf5",
        default=None,
        help="Optional tiny HDF5 prototype path for initial MVP-1 QC.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for qc_skeleton_report.md and qc_skeleton_report.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report to stdout without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing QC skeleton reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        output_dir = _resolve_output_dir(config, args.output_dir)
        _ensure_output_dir_is_safe(output_dir)
        if args.input_hdf5:
            report = _run_hdf5_qc(Path(args.input_hdf5))
            validation = None
            manifest_path = None
        else:
            manifest_path = _resolve_manifest_path(config, args.manifest)
            manifest = _read_manifest_json(manifest_path)
            validation = validate_manifest_basic(manifest)
            report = _build_report(manifest_path, validation)
        if not args.dry_run:
            if args.input_hdf5:
                _write_qc_summary(report, output_dir, overwrite=args.overwrite)
            else:
                _write_reports(report, output_dir, overwrite=args.overwrite)
    except (ManifestBuildError, QCSkeletonError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.input_hdf5:
        print(
            "Initial QC "
            f"status={report['status']}; "
            f"errors: {len(report['errors'])}, warnings: {len(report['warnings'])}."
        )
    else:
        assert validation is not None
        print(
            "Manifest schema validation "
            f"{'passed' if validation.is_valid else 'failed'}; "
            f"errors: {len(validation.errors)}, warnings: {len(validation.warnings)}."
        )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote QC reports to: {output_dir}")
    if args.input_hdf5:
        return 1 if report["errors"] else 0
    assert validation is not None
    return 0 if validation.is_valid else 1


def _resolve_manifest_path(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    outputs = _as_dict(config.get("outputs"))
    if outputs.get("data_manifest_json"):
        return Path(str(outputs["data_manifest_json"]))
    data = _as_dict(config.get("data"))
    manifests_dir = data.get("manifests")
    if manifests_dir:
        return Path(str(manifests_dir)) / "data_manifest_v001.json"
    raise QCSkeletonError("Manifest path is not configured. Pass --manifest.")


def _resolve_output_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    outputs = _as_dict(config.get("outputs"))
    if outputs.get("qc_report_dir"):
        return Path(str(outputs["qc_report_dir"]))
    data = _as_dict(config.get("data"))
    reports_dir = data.get("reports")
    if reports_dir:
        return Path(str(reports_dir)) / "qc"
    raise QCSkeletonError("QC output directory is not configured. Pass --output-dir.")


def _read_manifest_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise QCSkeletonError(f"Manifest JSON does not exist: {path}")
    if not path.is_file():
        raise QCSkeletonError(f"Manifest path is not a file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise QCSkeletonError(f"Manifest JSON must contain an object: {path}")
    return data


def _build_report(manifest_path: Path, validation: Any) -> dict[str, Any]:
    return {
        "stage": "EXP-2",
        "task": "qc_skeleton_manifest_schema_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "status": "passed" if validation.is_valid else "failed",
        "validation": asdict(validation),
        "scope": [
            "manifest top-level fields",
            "well entries",
            "CAST/pose/XSI receiver file inventory presence",
            "XSI receiver count contract",
        ],
        "not_performed": [
            "large .mat content reads",
            "XSI waveform QC",
            "CAST waveform/image QC",
            "depth alignment",
            "RelBearing rotation",
            "label generation",
            "feature extraction",
            "model training",
        ],
    }


def _run_hdf5_qc(input_hdf5: Path) -> dict[str, Any]:
    if not input_hdf5.exists():
        raise QCSkeletonError(f"Input HDF5 does not exist: {input_hdf5}")
    if not input_hdf5.is_file():
        raise QCSkeletonError(f"Input HDF5 path is not a file: {input_hdf5}")
    results: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    with h5py.File(input_hdf5, "r") as h5:
        required = [
            "/aligned/xsi_waveform",
            "/aligned/cast_zc",
            "/pose/inc_deg",
            "/pose/rel_bearing_deg",
        ]
        missing = [path for path in required if path not in h5]
        if missing:
            errors.extend(f"Missing HDF5 dataset: {path}" for path in missing)
        if not missing:
            xsi = run_xsi_waveform_qc(h5["/aligned/xsi_waveform"][()])
            cast = run_cast_zc_qc(h5["/aligned/cast_zc"][()])
            pose = run_pose_range_qc(h5["/pose/inc_deg"][()], h5["/pose/rel_bearing_deg"][()])
            results = {
                "xsi_waveform": xsi.to_dict(),
                "cast_zc": cast.to_dict(),
                "pose": {key: value.to_dict() for key, value in pose.items()},
            }
            for item in [xsi, cast, *pose.values()]:
                errors.extend(item.errors)
                warnings.extend(item.warnings)

    status = "failed" if errors else "passed_with_warnings" if warnings else "passed"
    return {
        "stage": "EXP-2",
        "task": "initial_qc_skeleton_tiny_hdf5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_hdf5": str(input_hdf5),
        "status": status,
        "results": results,
        "warnings": warnings,
        "errors": errors,
        "not_performed": [
            "STC",
            "APES",
            "first-arrival picking",
            "depth alignment",
            "RelBearing rotation",
            "label generation",
            "model training",
        ],
    }


def _write_reports(report: dict[str, Any], output_dir: Path, *, overwrite: bool) -> None:
    json_path = output_dir / "qc_skeleton_report.json"
    markdown_path = output_dir / "qc_skeleton_report.md"
    _ensure_can_write(json_path, overwrite=overwrite)
    _ensure_can_write(markdown_path, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_format_markdown_report(report), encoding="utf-8")


def _write_qc_summary(report: dict[str, Any], output_dir: Path, *, overwrite: bool) -> None:
    json_path = output_dir / "qc_summary_v001.json"
    markdown_path = output_dir / "qc_summary_v001.md"
    _ensure_can_write(json_path, overwrite=overwrite)
    _ensure_can_write(markdown_path, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_format_qc_summary_markdown(report), encoding="utf-8")


def _format_markdown_report(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# QC Skeleton Report",
        "",
        f"- Stage: {report['stage']}",
        f"- Task: {report['task']}",
        f"- Status: {report['status']}",
        f"- Manifest: {report['manifest_path']}",
        f"- Errors: {len(validation['errors'])}",
        f"- Warnings: {len(validation['warnings'])}",
        "",
        "## Errors",
        "",
    ]
    lines.extend(_format_items(validation["errors"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_format_items(validation["warnings"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_format_items(report["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _format_qc_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-1 Initial QC Summary",
        "",
        f"- Stage: {report['stage']}",
        f"- Task: {report['task']}",
        f"- Status: {report['status']}",
        f"- Input HDF5: {report['input_hdf5']}",
        f"- Errors: {len(report['errors'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
        "## Results",
        "",
    ]
    for name, result in report["results"].items():
        if name == "pose":
            continue
        lines.append(
            f"- {name}: shape={result['shape']}, finite_ratio={result['finite_ratio']}, "
            f"min={result['min']}, max={result['max']}"
        )
    for name, result in report["results"].get("pose", {}).items():
        lines.append(
            f"- {name}: shape={result['shape']}, finite_ratio={result['finite_ratio']}, "
            f"min={result['min']}, max={result['max']}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_format_items(report["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_format_items(report["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_format_items(report["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _format_items(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise QCSkeletonError(f"Output already exists: {path}. Pass --overwrite to replace it.")


def _ensure_output_dir_is_safe(output_dir: Path) -> None:
    project_data_dir = (PROJECT_ROOT / "data").resolve()
    resolved_output_dir = output_dir.resolve()
    try:
        resolved_output_dir.relative_to(project_data_dir)
    except ValueError:
        return
    raise QCSkeletonError(f"Refusing to write QC outputs inside Git data directory: {output_dir}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
