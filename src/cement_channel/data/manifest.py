from __future__ import annotations

import csv
import fnmatch
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CSV_FIELDNAMES = [
    "well_id",
    "file_role",
    "receiver_index",
    "path",
    "filename",
    "extension",
    "size_bytes",
    "modified_time",
    "matched_pattern",
]

DEFAULT_MANIFEST_VERSION = "data_manifest_v001"
DEFAULT_SCHEMA_VERSION = "schema_v001"


class ManifestBuildError(RuntimeError):
    """Raised when the raw manifest cannot be built safely."""


@dataclass(frozen=True)
class FileRoleMatch:
    file_role: str
    matched_pattern: str
    receiver_index: int | None = None


@dataclass(frozen=True)
class RawFileRecord:
    well_id: str
    file_role: str
    receiver_index: int | None
    path: str
    filename: str
    extension: str
    size_bytes: int
    modified_time: str
    matched_pattern: str

    def to_csv_row(self) -> dict[str, str | int]:
        row = asdict(self)
        row["receiver_index"] = "" if self.receiver_index is None else self.receiver_index
        return row


@dataclass(frozen=True)
class ManifestWarning:
    code: str
    message: str
    level: str = "warning"
    well_id: str | None = None
    path: str | None = None
    expected: int | None = None
    observed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def load_paths_config(paths_config: Path | str) -> dict[str, Any]:
    config_path = Path(paths_config)
    if not config_path.exists():
        raise ManifestBuildError(f"Paths config does not exist: {config_path}")
    if not config_path.is_file():
        raise ManifestBuildError(f"Paths config is not a file: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        data = _parse_minimal_yaml(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ManifestBuildError(f"Paths config must contain a YAML mapping: {config_path}")
    return data


def parse_receiver_index(filename: str) -> int | None:
    match = re.search(r"XSILMR(\d+)", Path(filename).stem, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def classify_single_well_file(
    file_path: Path | str,
    raw_dir: Path | str,
    layout: dict[str, Any],
) -> FileRoleMatch | None:
    path = Path(file_path)
    root = Path(raw_dir)
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return None

    if len(relative_path.parts) == 1:
        if path.name in _as_str_list(layout.get("cast_files"), ["CAST.mat"]):
            return FileRoleMatch(file_role="cast", matched_pattern=path.name)
        if path.name in _as_str_list(layout.get("pose_files"), []):
            return FileRoleMatch(file_role="pose", matched_pattern=path.name)

    xsi_receiver_dir = str(layout.get("xsi_receiver_dir", "XSILMR"))
    if len(relative_path.parts) >= 2 and relative_path.parts[0] == xsi_receiver_dir:
        for pattern in _as_str_list(layout.get("xsi_receiver_file_patterns"), ["XSILMR*.mat"]):
            if fnmatch.fnmatch(path.name, pattern):
                return FileRoleMatch(
                    file_role="xsi_receiver",
                    receiver_index=parse_receiver_index(path.name),
                    matched_pattern=pattern,
                )

    return None


def build_manifest_from_paths_config(
    paths_config: Path | str,
    *,
    raw_dir_override: Path | str | None = None,
    output_csv: Path | str | None = None,
    output_json: Path | str | None = None,
    max_wells: int | None = None,
) -> dict[str, Any]:
    config_path = Path(paths_config)
    config = load_paths_config(config_path)
    return build_manifest(
        config,
        config_path=config_path,
        raw_dir_override=raw_dir_override,
        output_csv=output_csv,
        output_json=output_json,
        max_wells=max_wells,
    )


def build_manifest(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    raw_dir_override: Path | str | None = None,
    output_csv: Path | str | None = None,
    output_json: Path | str | None = None,
    max_wells: int | None = None,
) -> dict[str, Any]:
    raw_dir = _resolve_raw_dir(config, raw_dir_override)
    if not raw_dir.exists():
        raise ManifestBuildError(f"Raw directory does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise ManifestBuildError(f"Raw path is not a directory: {raw_dir}")

    layout = _as_dict(config.get("raw_layout"))
    organization = str(layout.get("organization", "by_well"))

    if organization == "single_well_flat":
        wells = [_scan_single_well_flat(raw_dir, layout)]
        if max_wells is not None:
            wells = wells[: max(max_wells, 0)]
    elif organization == "by_well":
        wells = _scan_by_well(raw_dir, layout, max_wells=max_wells)
    else:
        raise ManifestBuildError(f"Unsupported raw_layout.organization: {organization}")

    records = [record for well in wells for record in well["files"]]
    warnings = [warning for well in wells for warning in well["warnings"]]
    output_csv_path = _resolve_output_path(
        output_csv,
        config,
        output_key="raw_inventory_csv",
        default_filename="raw_file_inventory.csv",
    )
    output_json_path = _resolve_output_path(
        output_json,
        config,
        output_key="data_manifest_json",
        default_filename="data_manifest_v001.json",
    )
    manifest_well_id = layout.get("well_id") or (
        "D2" if organization == "single_well_flat" else None
    )

    manifest = {
        "manifest_version": DEFAULT_MANIFEST_VERSION,
        "schema_version": str(config.get("schema_version", DEFAULT_SCHEMA_VERSION)),
        "stage": "EXP-1",
        "task": "raw_data_manifest",
        "status": "completed_with_warnings" if warnings else "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": _sha256_file(config_path) if config_path is not None else None,
        "raw_dir": str(raw_dir),
        "raw_layout": {
            "organization": organization,
            "well_id": manifest_well_id,
            "expected_xsi_receiver_files": layout.get("expected_xsi_receiver_files"),
        },
        "outputs": {
            "raw_inventory_csv": str(output_csv_path),
            "data_manifest_json": str(output_json_path),
        },
        "summary": _build_summary(records, warnings, wells),
        "wells": [
            {
                "well_id": well["well_id"],
                "organization": well["organization"],
                "raw_dir": well["raw_dir"],
                "counts": well["counts"],
                "expected_xsi_receiver_files": well.get("expected_xsi_receiver_files"),
                "files": [asdict(record) for record in well["files"]],
                "warnings": [warning.to_dict() for warning in well["warnings"]],
            }
            for well in wells
        ],
        "files": [asdict(record) for record in records],
        "warnings": [warning.to_dict() for warning in warnings],
    }
    return {key: value for key, value in manifest.items() if value is not None}


def write_manifest_outputs(
    manifest: dict[str, Any],
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    output_csv = Path(manifest["outputs"]["raw_inventory_csv"])
    output_json = Path(manifest["outputs"]["data_manifest_json"])
    if dry_run:
        return

    _ensure_can_write(output_csv, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    _write_inventory_csv(output_csv, manifest["files"])
    _write_manifest_json(output_json, manifest)


def _scan_single_well_flat(raw_dir: Path, layout: dict[str, Any]) -> dict[str, Any]:
    well_id = str(layout.get("well_id") or "D2")
    records: list[RawFileRecord] = []
    warnings: list[ManifestWarning] = []

    for filename in _as_str_list(layout.get("cast_files"), ["CAST.mat"]):
        path = raw_dir / filename
        if path.is_file():
            records.append(_make_record(path, well_id, FileRoleMatch("cast", filename)))
        else:
            warnings.append(
                ManifestWarning(
                    code="missing_cast_file",
                    message=f"Configured CAST file is missing: {filename}",
                    well_id=well_id,
                    path=str(path),
                )
            )

    for filename in _as_str_list(layout.get("pose_files"), []):
        path = raw_dir / filename
        if path.is_file():
            records.append(_make_record(path, well_id, FileRoleMatch("pose", filename)))
        else:
            warnings.append(
                ManifestWarning(
                    code="missing_pose_file",
                    message=f"Configured pose file is missing: {filename}",
                    well_id=well_id,
                    path=str(path),
                )
            )

    expected_receiver_count = _optional_int(layout.get("expected_xsi_receiver_files"))
    xsi_receiver_dir = raw_dir / str(layout.get("xsi_receiver_dir", "XSILMR"))
    if xsi_receiver_dir.is_dir():
        receiver_files = _collect_pattern_matches(
            xsi_receiver_dir,
            _as_str_list(layout.get("xsi_receiver_file_patterns"), ["XSILMR*.mat"]),
        )
        for path, matched_pattern in receiver_files:
            records.append(
                _make_record(
                    path,
                    well_id,
                    FileRoleMatch(
                        file_role="xsi_receiver",
                        receiver_index=parse_receiver_index(path.name),
                        matched_pattern=matched_pattern,
                    ),
                )
            )
    else:
        receiver_files = []
        warnings.append(
            ManifestWarning(
                code="missing_xsi_receiver_dir",
                message=f"Configured XSI receiver directory is missing: {xsi_receiver_dir.name}",
                well_id=well_id,
                path=str(xsi_receiver_dir),
            )
        )

    if expected_receiver_count is not None and len(receiver_files) != expected_receiver_count:
        warnings.append(
            ManifestWarning(
                code="xsi_receiver_count_mismatch",
                message=(
                    "XSI receiver file count does not match expected count: "
                    f"expected {expected_receiver_count}, observed {len(receiver_files)}"
                ),
                well_id=well_id,
                path=str(xsi_receiver_dir),
                expected=expected_receiver_count,
                observed=len(receiver_files),
            )
        )

    records = sorted(
        records,
        key=lambda record: (record.file_role, record.receiver_index or 0, record.path),
    )
    return {
        "well_id": well_id,
        "organization": "single_well_flat",
        "raw_dir": str(raw_dir),
        "expected_xsi_receiver_files": expected_receiver_count,
        "files": records,
        "warnings": warnings,
        "counts": _role_counts(records),
    }


def _scan_by_well(
    raw_dir: Path,
    layout: dict[str, Any],
    *,
    max_wells: int | None,
) -> list[dict[str, Any]]:
    well_dir_pattern = str(layout.get("well_dir_pattern", "well_*"))
    well_dirs = sorted(path for path in raw_dir.glob(well_dir_pattern) if path.is_dir())
    if max_wells is not None:
        well_dirs = well_dirs[: max(max_wells, 0)]

    if not well_dirs:
        return [
            {
                "well_id": None,
                "organization": "by_well",
                "raw_dir": str(raw_dir),
                "expected_xsi_receiver_files": None,
                "files": [],
                "warnings": [
                    ManifestWarning(
                        code="missing_well_directories",
                        message=f"No well directories matched pattern: {well_dir_pattern}",
                        path=str(raw_dir),
                    )
                ],
                "counts": {},
            }
        ]

    wells: list[dict[str, Any]] = []
    role_patterns = {
        "cast": _as_str_list(layout.get("cast_file_patterns"), ["*CAST*.mat", "*cast*.mat"]),
        "pose": _as_str_list(layout.get("pose_file_patterns"), ["*pose*.mat"]),
        "xsi": _as_str_list(layout.get("xsi_file_patterns"), ["*XSI*.mat", "*xsi*.mat"]),
    }
    for well_dir in well_dirs:
        records: list[RawFileRecord] = []
        warnings: list[ManifestWarning] = []
        well_id = well_dir.name
        for role, patterns in role_patterns.items():
            matches = _collect_pattern_matches(well_dir, patterns)
            if not matches:
                warnings.append(
                    ManifestWarning(
                        code=f"missing_{role}_file",
                        message=f"No {role} files matched configured patterns.",
                        well_id=well_id,
                        path=str(well_dir),
                    )
                )
            for path, matched_pattern in matches:
                records.append(_make_record(path, well_id, FileRoleMatch(role, matched_pattern)))
        wells.append(
            {
                "well_id": well_id,
                "organization": "by_well",
                "raw_dir": str(well_dir),
                "expected_xsi_receiver_files": None,
                "files": sorted(records, key=lambda record: (record.file_role, record.path)),
                "warnings": warnings,
                "counts": _role_counts(records),
            }
        )
    return wells


def _make_record(path: Path, well_id: str, match: FileRoleMatch) -> RawFileRecord:
    stat = path.stat()
    return RawFileRecord(
        well_id=well_id,
        file_role=match.file_role,
        receiver_index=match.receiver_index,
        path=str(path.resolve()),
        filename=path.name,
        extension=path.suffix,
        size_bytes=stat.st_size,
        modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        matched_pattern=match.matched_pattern,
    )


def _collect_pattern_matches(directory: Path, patterns: list[str]) -> list[tuple[Path, str]]:
    matches: dict[Path, str] = {}
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path not in matches:
                matches[path] = pattern
    return sorted(matches.items(), key=lambda item: item[0].name)


def _build_summary(
    records: list[RawFileRecord],
    warnings: list[ManifestWarning],
    wells: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "well_count": len(wells),
        "file_count": len(records),
        "files_by_role": _role_counts(records),
        "xsi_receiver_file_count": sum(
            1 for record in records if record.file_role == "xsi_receiver"
        ),
        "warning_count": len(warnings),
    }


def _role_counts(records: list[RawFileRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.file_role] = counts.get(record.file_role, 0) + 1
    return counts


def _resolve_raw_dir(config: dict[str, Any], raw_dir_override: Path | str | None) -> Path:
    if raw_dir_override is not None:
        return Path(raw_dir_override)
    data_config = _as_dict(config.get("data"))
    raw_dir = data_config.get("raw")
    if raw_dir is None:
        raise ManifestBuildError("Paths config is missing data.raw")
    return Path(str(raw_dir))


def _resolve_output_path(
    override: Path | str | None,
    config: dict[str, Any],
    *,
    output_key: str,
    default_filename: str,
) -> Path:
    if override is not None:
        return Path(override)
    outputs = _as_dict(config.get("outputs"))
    if outputs.get(output_key):
        return Path(str(outputs[output_key]))
    manifests_dir = _as_dict(config.get("data")).get("manifests")
    if manifests_dir:
        return Path(str(manifests_dir)) / default_filename
    return Path(default_filename)


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ManifestBuildError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_inventory_csv(path: Path, file_records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in file_records:
            row = dict(record)
            row["receiver_index"] = "" if row["receiver_index"] is None else row["receiver_index"]
            writer.writerow({field: row[field] for field in CSV_FIELDNAMES})


def _write_manifest_json(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(manifest, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    lines = _preprocess_yaml_lines(text)
    parsed, next_index = _parse_yaml_block(lines, 0, 0)
    if next_index != len(lines):
        raise ManifestBuildError("Unable to parse paths YAML with minimal parser.")
    if not isinstance(parsed, dict):
        raise ManifestBuildError("Paths YAML root must be a mapping.")
    return parsed


def _preprocess_yaml_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped_comment = _strip_yaml_comment(raw_line).rstrip()
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        lines.append((indent, stripped_comment.strip()))
    return lines


def _parse_yaml_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines) or lines[index][0] < indent:
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ManifestBuildError(f"Unexpected YAML indentation near: {text}")
        if text.startswith("- "):
            break
        if ":" not in text:
            raise ManifestBuildError(f"Expected YAML key-value line near: {text}")

        key, value_text = text.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        if value_text:
            mapping[key] = _parse_yaml_scalar(value_text)
            index += 1
            continue

        next_index = index + 1
        if next_index < len(lines) and lines[next_index][0] > line_indent:
            mapping[key], index = _parse_yaml_block(lines, next_index, lines[next_index][0])
        else:
            mapping[key] = None
            index = next_index
    return mapping, index


def _parse_yaml_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or not text.startswith("- "):
            break

        value_text = text[2:].strip()
        if value_text:
            items.append(_parse_yaml_scalar(value_text))
            index += 1
            continue

        next_index = index + 1
        if next_index < len(lines) and lines[next_index][0] > line_indent:
            value, index = _parse_yaml_block(lines, next_index, lines[next_index][0])
            items.append(value)
        else:
            items.append(None)
            index = next_index
    return items, index


def _parse_yaml_scalar(value: str) -> Any:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    normalized = value.lower()
    if normalized in {"null", "none", "~"}:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _strip_yaml_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]
    return line
