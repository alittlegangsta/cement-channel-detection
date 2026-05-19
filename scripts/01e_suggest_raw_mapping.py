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
from cement_channel.data.mapping_suggester import (  # noqa: E402
    format_mapping_draft_yaml,
    format_mapping_suggestions_report,
    load_struct_probe_json,
    suggest_raw_variable_mapping,
)


class MappingSuggestionCliError(RuntimeError):
    """Raised when raw mapping suggestions cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest a draft raw variable mapping from struct probe metadata."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--struct-probe-json",
        "--probe-json",
        dest="struct_probe_json",
        default=None,
    )
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-draft-yaml", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        struct_probe_json = _resolve_struct_probe_json(config, args.struct_probe_json)
        output_report_md = _resolve_data_output(
            config,
            args.output_report_md,
            "reports",
            "raw_variable_mapping_suggestions.md",
        )
        output_report_json = _resolve_data_output(
            config,
            args.output_report_json,
            "reports",
            "raw_variable_mapping_suggestions.json",
        )
        output_draft_yaml = _resolve_draft_yaml(args.output_draft_yaml)
        _ensure_report_path_is_safe(output_report_md)
        _ensure_report_path_is_safe(output_report_json)
        _ensure_draft_path_is_safe(output_draft_yaml)

        probe = _read_struct_probe(struct_probe_json)
        result = suggest_raw_variable_mapping(
            probe,
            struct_probe_json_path=struct_probe_json,
            well_id=_resolve_well_id(config),
        )
        report = format_mapping_suggestions_report(result)
        draft_yaml = format_mapping_draft_yaml(result)
        if not args.dry_run:
            _write_outputs(
                result.to_dict(),
                report,
                draft_yaml,
                output_report_md=output_report_md,
                output_report_json=output_report_json,
                output_draft_yaml=output_draft_yaml,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        MappingSuggestionCliError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    todo_count = sum(
        1
        for recommendation in result.recommendations.values()
        if recommendation.variable_path == "TODO_CONFIRM"
    )
    print(
        "Raw mapping suggestions "
        f"status={result.status}; "
        f"recommendations={len(result.recommendations)}; "
        f"todo={todo_count}; "
        f"warnings={len(result.warnings)}; "
        f"errors={len(result.errors)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_report_md}")
        print(f"Wrote JSON suggestions: {output_report_json}")
        print(f"Wrote draft mapping YAML: {output_draft_yaml}")
    return 0


def _resolve_struct_probe_json(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    manifests_dir = data.get("manifests")
    if manifests_dir:
        return Path(str(manifests_dir)) / "mat_struct_probe_v001.json"
    raise MappingSuggestionCliError(
        "Struct probe JSON path is not configured. Pass --struct-probe-json."
    )


def _resolve_data_output(
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


def _resolve_draft_yaml(override: str | None) -> Path:
    if override:
        return Path(override)
    return Path("configs/raw_variable_mapping.draft.yaml")


def _resolve_well_id(config: dict[str, Any]) -> str:
    raw_layout = _as_dict(config.get("raw_layout"))
    return str(raw_layout.get("well_id") or "TODO_CONFIRM")


def _read_struct_probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MappingSuggestionCliError(f"Struct probe JSON does not exist: {path}")
    if not path.is_file():
        raise MappingSuggestionCliError(f"Struct probe JSON path is not a file: {path}")
    return load_struct_probe_json(path)


def _write_outputs(
    suggestions: dict[str, Any],
    report: str,
    draft_yaml: str,
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_draft_yaml: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    _ensure_can_write(output_draft_yaml, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_draft_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.write_text(report, encoding="utf-8")
    output_report_json.write_text(
        json.dumps(suggestions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_draft_yaml.write_text(draft_yaml, encoding="utf-8")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MappingSuggestionCliError(f"Output already exists: {path}. Pass --overwrite.")


def _ensure_report_path_is_safe(path: Path) -> None:
    project_data_dir = (PROJECT_ROOT / "data").resolve()
    try:
        path.resolve().relative_to(project_data_dir)
    except ValueError:
        return
    raise MappingSuggestionCliError(f"Refusing to write report inside Git data directory: {path}")


def _ensure_draft_path_is_safe(path: Path) -> None:
    if path.name == "raw_variable_mapping.yaml":
        raise MappingSuggestionCliError(
            "Refusing to generate configs/raw_variable_mapping.yaml directly. "
            "Generate a draft and confirm it manually."
        )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
