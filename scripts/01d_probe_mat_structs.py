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
from cement_channel.data.mat_struct_probe import (  # noqa: E402
    format_struct_probe_report,
    load_json_object,
    probe_structs_from_metadata,
)


class StructProbeCliError(RuntimeError):
    """Raised when controlled MAT struct probing cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled probe of MATLAB struct fields.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--max-variables-per-file", type=int, default=3)
    parser.add_argument("--max-field-depth", type=int, default=2)
    parser.add_argument("--max-array-elements-preview", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        manifest_path = _resolve_manifest_path(config, args.manifest)
        metadata_json = _resolve_metadata_json(config, args.metadata_json)
        output_json = _resolve_output_path(
            config,
            args.output_json,
            "manifests",
            "mat_struct_probe_v001.json",
        )
        output_report_md = _resolve_output_path(
            config,
            args.output_report_md,
            "reports",
            "mat_struct_probe_report.md",
        )
        _ensure_data_output_path(output_json)
        _ensure_data_output_path(output_report_md)
        metadata = _read_metadata(metadata_json)
        if manifest_path is not None and not manifest_path.exists():
            raise StructProbeCliError(f"Manifest does not exist: {manifest_path}")
        result = probe_structs_from_metadata(
            metadata,
            metadata_json_path=metadata_json,
            manifest_path=manifest_path,
            max_files=args.max_files,
            max_variables_per_file=args.max_variables_per_file,
            max_field_depth=args.max_field_depth,
            max_array_elements_preview=args.max_array_elements_preview,
        )
        report = format_struct_probe_report(result)
        if not args.dry_run:
            _write_outputs(
                result.to_dict(),
                report,
                output_json=output_json,
                output_report_md=output_report_md,
                overwrite=args.overwrite,
            )
    except (ManifestBuildError, StructProbeCliError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = result.summary
    print(
        "Struct probe "
        f"files={summary['file_count']}; "
        f"can_probe={summary['can_probe_count']}; "
        f"errors={summary['file_error_count']}; "
        f"fields={summary['field_count']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote JSON: {output_json}")
        print(f"Wrote Markdown report: {output_report_md}")
    return 0


def _resolve_manifest_path(config: dict[str, Any], override: str | None) -> Path | None:
    if override:
        return Path(override)
    outputs = _as_dict(config.get("outputs"))
    if outputs.get("data_manifest_json"):
        return Path(str(outputs["data_manifest_json"]))
    return None


def _resolve_metadata_json(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    manifests_dir = data.get("manifests")
    if manifests_dir:
        return Path(str(manifests_dir)) / "mat_metadata_v001.json"
    raise StructProbeCliError("MAT metadata JSON path is not configured. Pass --metadata-json.")


def _resolve_output_path(
    config: dict[str, Any],
    override: str | None,
    data_key: str,
    filename: str,
) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    root = data.get(data_key)
    if root:
        return Path(str(root)) / filename
    return Path(filename)


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StructProbeCliError(f"MAT metadata JSON does not exist: {path}")
    if not path.is_file():
        raise StructProbeCliError(f"MAT metadata JSON path is not a file: {path}")
    return load_json_object(path)


def _write_outputs(
    result: dict[str, Any],
    report: str,
    *,
    output_json: Path,
    output_report_md: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_report_md, overwrite=overwrite)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(report, encoding="utf-8")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise StructProbeCliError(f"Output already exists: {path}. Pass --overwrite.")


def _ensure_data_output_path(path: Path) -> None:
    project_root = PROJECT_ROOT.resolve()
    try:
        path.resolve().relative_to(project_root)
    except ValueError:
        return
    if path.name == "mat_struct_probe_v001.json" or path.name.endswith("_report.md"):
        raise StructProbeCliError(f"Refusing to write probe output inside Git repo: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
