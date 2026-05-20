from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.depth_resample import (  # noqa: E402
    build_depth_resample_preview,
    write_depth_resample_preview_outputs,
)
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class DepthResamplePreviewCliError(RuntimeError):
    """Raised when depth resample preview cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP-2 small-slice depth resampling preview.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--depth-only-npz", default=None)
    parser.add_argument("--depth-grid-proposal-json", default=None)
    parser.add_argument("--small-slice-npz", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--max-preview-depth-samples", type=int, default=16)
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
        small_slice_npz = _resolve_interim_path(
            config,
            args.small_slice_npz,
            "small_slice_v001.npz",
        )
        proposal_json = _resolve_report_path(
            config,
            args.depth_grid_proposal_json,
            "depth_grid_proposal.json",
        )
        output_npz = _resolve_interim_path(
            config,
            args.output_npz,
            "depth_resample_preview_v001.npz",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "depth_resample_preview_report.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "depth_resample_preview_report.json",
        )
        _ensure_interim_output(config, output_npz)
        _ensure_report_output(config, output_md)
        _ensure_report_output(config, output_json)
        report, arrays = build_depth_resample_preview(
            depth_only_npz=depth_only_npz,
            depth_grid_proposal_json=proposal_json,
            small_slice_npz=small_slice_npz,
            max_preview_depth_samples=args.max_preview_depth_samples,
        )
        if not args.dry_run:
            write_depth_resample_preview_outputs(
                report,
                arrays,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthResamplePreviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth resample preview "
        f"arrays={len(arrays)}; "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"small_slice_status={report.small_slice.get('status')}."
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
    raise DepthResamplePreviewCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthResamplePreviewCliError("data.reports is not configured.")


def _ensure_interim_output(config: dict[str, Any], output_path: Path) -> None:
    _ensure_path_within(config, output_path, key="interim", action="write")


def _ensure_report_output(config: dict[str, Any], output_path: Path) -> None:
    _ensure_path_within(config, output_path, key="reports", action="write")


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
        raise DepthResamplePreviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthResamplePreviewCliError(
            f"Refusing to {action} depth resample output outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
