from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.depth_audit import (  # noqa: E402
    DEFAULT_MAX_DEPTH_SAMPLES,
    audit_depth_axes,
    read_depth_axes_from_configs,
    write_depth_axis_audit_outputs,
)
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class DepthAxisAuditCliError(RuntimeError):
    """Raised when depth-axis audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CAST, XSI, and pose depth axes.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--mapping", default="configs/raw_variable_mapping.yaml")
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--max-depth-samples", type=int, default=DEFAULT_MAX_DEPTH_SAMPLES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "depth_axis_audit_report.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "depth_axis_audit_report.json",
        )
        _ensure_report_output(config, output_md)
        _ensure_report_output(config, output_json)
        arrays, _config, mapping = read_depth_axes_from_configs(
            args.paths_config,
            args.mapping,
            max_depth_samples=args.max_depth_samples,
        )
        result = audit_depth_axes(
            cast_depth=arrays["cast_depth"],
            xsi_depth_by_receiver=arrays["xsi_depth_by_receiver"],
            pose_depth=arrays["pose_depth"],
            expected_receiver_count=int(
                _as_dict(mapping.get("xsi")).get("expected_receiver_files", 13)
            ),
            depth_unit=_depth_unit(mapping),
        )
        if not args.dry_run:
            write_depth_axis_audit_outputs(
                result,
                output_json=output_json,
                output_md=output_md,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthAxisAuditCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth axis audit "
        f"decision={result.decision}; "
        f"blockers={len(result.no_go_blockers)}; "
        f"warnings={len(result.warnings)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 1 if result.decision == "no_go" else 0


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthAxisAuditCliError("data.reports is not configured; pass an explicit output path.")


def _ensure_report_output(config: dict[str, Any], output_path: Path) -> None:
    data = _as_dict(config.get("data"))
    reports = Path(str(data.get("reports", ""))).resolve()
    if not str(reports):
        raise DepthAxisAuditCliError("data.reports is not configured.")
    try:
        output_path.resolve().relative_to(reports)
    except ValueError as exc:
        raise DepthAxisAuditCliError(
            f"Refusing to write depth-axis audit output outside data.reports: {output_path}"
        ) from exc


def _depth_unit(mapping: dict[str, Any]) -> str:
    units = {
        str(_as_dict(mapping.get("cast")).get("depth_unit", "unknown_to_verify")),
        str(_as_dict(mapping.get("pose")).get("depth_unit", "unknown_to_verify")),
        str(_as_dict(mapping.get("xsi")).get("depth_unit", "unknown_to_verify")),
    }
    if len(units) == 1:
        return units.pop()
    return "mixed_or_unknown:" + ",".join(sorted(units))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
