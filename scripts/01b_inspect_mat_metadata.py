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
from cement_channel.data.mat_metadata import inspect_manifest_mat_metadata  # noqa: E402


class MatMetadataInspectionError(RuntimeError):
    """Raised when MAT metadata inspection cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect lightweight MAT file metadata without loading large variables."
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
        "--output-json",
        default=None,
        help="Output path for mat_metadata_v001.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report without writing output files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit inspected files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        manifest_path = _resolve_manifest_path(config, args.manifest)
        output_json = _resolve_output_json(config, args.output_json)
        _ensure_output_path_is_safe(output_json)
        manifest = _read_json(manifest_path)
        metadata = inspect_manifest_mat_metadata(manifest, max_files=args.max_files)
        metadata["manifest_path"] = str(manifest_path)
        metadata["output_json"] = str(output_json)
        if not args.dry_run:
            _write_json(output_json, metadata, overwrite=args.overwrite)
    except (ManifestBuildError, MatMetadataInspectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = metadata["summary"]
    print(
        "Inspected "
        f"{summary['file_count']} MAT file(s); "
        f"can_open: {summary['can_open_count']}; "
        f"errors: {summary['error_file_count']}; "
        f"warnings: {summary['warning_file_count']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote JSON: {output_json}")
    return 0


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
    raise MatMetadataInspectionError("Manifest path is not configured. Pass --manifest.")


def _resolve_output_json(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    manifests_dir = data.get("manifests")
    if manifests_dir:
        return Path(str(manifests_dir)) / "mat_metadata_v001.json"
    return Path("mat_metadata_v001.json")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MatMetadataInspectionError(f"Manifest JSON does not exist: {path}")
    if not path.is_file():
        raise MatMetadataInspectionError(f"Manifest path is not a file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MatMetadataInspectionError(f"Manifest JSON must contain an object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MatMetadataInspectionError(f"Output already exists: {path}. Pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_output_path_is_safe(output_json: Path) -> None:
    project_data_dir = (PROJECT_ROOT / "data").resolve()
    resolved_output = output_json.resolve()
    try:
        resolved_output.relative_to(project_data_dir)
    except ValueError:
        return
    raise MatMetadataInspectionError(
        f"Refusing to write MAT metadata inside Git data directory: {output_json}"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
