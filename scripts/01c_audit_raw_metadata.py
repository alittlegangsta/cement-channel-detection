from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.data.raw_mapping import (  # noqa: E402
    audit_raw_metadata,
    format_mapping_template,
    format_raw_metadata_report,
    load_mat_metadata_json,
)


class RawMetadataAuditError(RuntimeError):
    """Raised when raw metadata audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MAT metadata JSON and generate a raw variable mapping template."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
        help="Path to paths YAML config.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Path to mat_metadata_v001.json.",
    )
    parser.add_argument(
        "--output-report-md",
        default=None,
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--output-report-json",
        default=None,
        help="Output JSON audit report path.",
    )
    parser.add_argument(
        "--output-mapping-template",
        default=None,
        help="Output YAML mapping template path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run audit without writing outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        metadata_json = _resolve_metadata_json(config, args.metadata_json)
        output_report_md = _resolve_report_path(
            config,
            args.output_report_md,
            "raw_metadata_report.md",
        )
        output_report_json = _resolve_report_path(
            config,
            args.output_report_json,
            "raw_metadata_report.json",
        )
        output_mapping_template = _resolve_mapping_template(args.output_mapping_template)
        _ensure_report_path_is_safe(output_report_md)
        _ensure_report_path_is_safe(output_report_json)

        metadata = _read_metadata(metadata_json)
        audit_result = audit_raw_metadata(metadata, metadata_json_path=metadata_json)
        markdown_report = format_raw_metadata_report(audit_result)
        mapping_template = format_mapping_template(
            audit_result,
            well_id=_resolve_well_id(config),
        )
        if not args.dry_run:
            _write_outputs(
                audit_result.to_dict(),
                markdown_report,
                mapping_template,
                output_report_md=output_report_md,
                output_report_json=output_report_json,
                output_mapping_template=output_mapping_template,
                overwrite=args.overwrite,
            )
    except (ManifestBuildError, RawMetadataAuditError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stats = audit_result.statistics
    warning_count = (
        len(audit_result.warnings)
        + len(audit_result.cast_warnings)
        + len(audit_result.pose_warnings)
        + len(audit_result.xsi_warnings)
    )
    print(
        "Raw metadata audit "
        f"status={audit_result.status}; "
        f"total_files={stats['total_files']}; "
        f"warnings={warning_count}; "
        f"errors={len(audit_result.errors)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_report_md}")
        print(f"Wrote JSON report: {output_report_json}")
        print(f"Wrote mapping template: {output_mapping_template}")
    return 0


def _resolve_metadata_json(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    manifests_dir = data.get("manifests")
    if manifests_dir:
        return Path(str(manifests_dir)) / "mat_metadata_v001.json"
    raise RawMetadataAuditError("MAT metadata JSON path is not configured. Pass --metadata-json.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports_dir = data.get("reports")
    if reports_dir:
        return Path(str(reports_dir)) / filename
    return Path(filename)


def _resolve_mapping_template(override: str | None) -> Path:
    if override:
        return Path(override)
    return Path("configs/raw_variable_mapping.example.yaml")


def _resolve_well_id(config: dict[str, Any]) -> str:
    raw_layout = _as_dict(config.get("raw_layout"))
    return str(raw_layout.get("well_id") or "TODO_CONFIRM")


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RawMetadataAuditError(f"MAT metadata JSON does not exist: {path}")
    if not path.is_file():
        raise RawMetadataAuditError(f"MAT metadata JSON path is not a file: {path}")
    return load_mat_metadata_json(path)


def _write_outputs(
    audit_result: dict[str, Any],
    markdown_report: str,
    mapping_template: str,
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_mapping_template: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    _ensure_can_write(output_mapping_template, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_mapping_template.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.write_text(markdown_report, encoding="utf-8")
    output_report_json.write_text(
        json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_mapping_template.write_text(mapping_template, encoding="utf-8")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RawMetadataAuditError(f"Output already exists: {path}. Pass --overwrite.")


def _ensure_report_path_is_safe(path: Path) -> None:
    project_data_dir = (PROJECT_ROOT / "data").resolve()
    try:
        path.resolve().relative_to(project_data_dir)
    except ValueError:
        return
    raise RawMetadataAuditError(f"Refusing to write report inside Git data directory: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
