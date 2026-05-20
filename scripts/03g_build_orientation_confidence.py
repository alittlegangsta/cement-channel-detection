from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.orientation_confidence import (  # noqa: E402
    DEFAULT_I_MIN_DEG,
    DEFAULT_I_STABLE_DEG,
    build_orientation_confidence,
    write_orientation_confidence_outputs,
)
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class OrientationConfidenceCliError(RuntimeError):
    """Raised when orientation confidence cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP-2 orientation confidence masks.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--depth-only-npz", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--i-min-deg", type=float, default=DEFAULT_I_MIN_DEG)
    parser.add_argument("--i-stable-deg", type=float, default=DEFAULT_I_STABLE_DEG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        depth_only_npz = _resolve_interim_path(
            config,
            args.depth_only_npz,
            "depth_only_v001.npz",
        )
        output_npz = _resolve_interim_path(
            config,
            args.output_npz,
            "orientation_confidence_v001.npz",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "orientation_confidence_report.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "orientation_confidence_report.json",
        )
        _ensure_interim_path(config, depth_only_npz, action="read")
        _ensure_interim_path(config, output_npz, action="write")
        _ensure_report_output(config, output_md)
        _ensure_report_output(config, output_json)
        report, arrays = build_orientation_confidence(
            depth_only_npz=depth_only_npz,
            i_min_deg=args.i_min_deg,
            i_stable_deg=args.i_stable_deg,
        )
        if not args.dry_run:
            write_orientation_confidence_outputs(
                report,
                arrays,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        OrientationConfidenceCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Orientation confidence "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"low_inclination_ratio={report.low_inclination_ratio}; "
        f"stable_inclination_ratio={report.stable_inclination_ratio}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote NPZ: {output_npz}")
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise OrientationConfidenceCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise OrientationConfidenceCliError("data.reports is not configured.")


def _ensure_interim_path(config: dict[str, Any], path: Path, *, action: str) -> None:
    _ensure_path_within(config, path, key="interim", action=action)


def _ensure_report_output(config: dict[str, Any], path: Path) -> None:
    _ensure_path_within(config, path, key="reports", action="write")


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
        raise OrientationConfidenceCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise OrientationConfidenceCliError(
            f"Refusing to {action} orientation confidence path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
