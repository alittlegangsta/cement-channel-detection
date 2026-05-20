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
from cement_channel.data.small_slice_reader import (  # noqa: E402
    DepthWindow,
    SmallSliceLimits,
    depth_window_from_center,
    depth_window_from_grid_proposal,
    load_depth_reference_arrays,
    read_small_slice,
    write_small_slice_outputs,
)


class SmallSliceCliError(RuntimeError):
    """Raised when controlled small-slice reading cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a controlled tiny MAT data slice.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--mapping", default="configs/raw_variable_mapping.yaml")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--max-depth-samples", type=int, default=10)
    parser.add_argument("--max-time-samples", type=int, default=32)
    parser.add_argument("--max-receivers", type=int, default=13)
    parser.add_argument("--max-sides", type=int, default=8)
    parser.add_argument("--max-cast-azimuth", type=int, default=180)
    parser.add_argument("--depth-start", type=float, default=None)
    parser.add_argument("--depth-stop", type=float, default=None)
    parser.add_argument("--depth-center", type=float, default=None)
    parser.add_argument("--depth-window-size", type=float, default=2.0)
    parser.add_argument("--depth-only-npz", default=None)
    parser.add_argument("--depth-grid-proposal-json", default=None)
    parser.add_argument(
        "--overlap-targeted",
        action="store_true",
        help="Default to the middle of depth_grid_proposal common overlap.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        mapping = _read_mapping(Path(args.mapping))
        depth_window = _resolve_depth_window(config, args)
        overlap_mode = depth_window is not None or args.overlap_targeted
        output_json = _resolve_output_path(
            config,
            args.output_json,
            "small_slice_overlap_summary_v001.json"
            if overlap_mode
            else "small_slice_summary_v001.json",
        )
        output_npz = (
            _resolve_output_path(
                config,
                args.output_npz,
                "small_slice_overlap_v001.npz" if overlap_mode else "small_slice_v001.npz",
            )
            if args.output_npz is not None or overlap_mode
            else None
        )
        _ensure_interim_output(config, output_json)
        if output_npz is not None:
            _ensure_interim_output(config, output_npz)
        limits = SmallSliceLimits(
            max_depth_samples=args.max_depth_samples,
            max_time_samples=args.max_time_samples,
            max_receivers=args.max_receivers,
            max_sides=args.max_sides,
            max_cast_azimuth=args.max_cast_azimuth,
        )
        depth_reference_arrays = None
        if depth_window is not None:
            depth_reference_arrays = load_depth_reference_arrays(
                _resolve_depth_reference_npz(config, args.depth_only_npz)
            )
        result, arrays = read_small_slice(
            config,
            mapping,
            mapping_path=Path(args.mapping),
            limits=limits,
            depth_window=depth_window,
            depth_reference_arrays=depth_reference_arrays,
        )
        if not args.dry_run:
            write_small_slice_outputs(
                result,
                arrays,
                output_json=output_json,
                output_npz=output_npz,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        SmallSliceCliError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Small slice "
        f"variables={len(result.variables)}; "
        f"arrays={len(arrays)}; "
        f"errors={len(result.errors)}; "
        f"warnings={len(result.warnings)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote JSON summary: {output_json}")
        if output_npz is not None:
            print(f"Wrote NPZ slice: {output_npz}")
    return 1 if result.errors else 0


def _resolve_depth_window(config: dict[str, Any], args: argparse.Namespace) -> DepthWindow | None:
    if args.depth_start is not None or args.depth_stop is not None:
        if args.depth_start is None or args.depth_stop is None:
            raise SmallSliceCliError("--depth-start and --depth-stop must be provided together.")
        return DepthWindow(depth_start=float(args.depth_start), depth_stop=float(args.depth_stop))
    if args.depth_center is not None:
        return depth_window_from_center(
            depth_center=float(args.depth_center),
            depth_window_size=min(float(args.depth_window_size), 2.0),
        )
    if args.overlap_targeted:
        proposal = _resolve_report_path(
            config,
            args.depth_grid_proposal_json,
            "depth_grid_proposal.json",
        )
        return depth_window_from_grid_proposal(
            proposal,
            depth_window_size=min(float(args.depth_window_size), 2.0),
        )
    return None


def _resolve_depth_reference_npz(config: dict[str, Any], override: str | None) -> Path:
    return _resolve_output_path(config, override, "depth_only_v001.npz")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise SmallSliceCliError("data.reports is not configured; pass an explicit report path.")


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SmallSliceCliError(f"Mapping config does not exist: {path}")
    if not path.is_file():
        raise SmallSliceCliError(f"Mapping path is not a file: {path}")
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SmallSliceCliError(f"Mapping config must contain a YAML mapping: {path}")
    return data


def _resolve_output_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise SmallSliceCliError("data.interim is not configured; pass an explicit output path.")


def _ensure_interim_output(config: dict[str, Any], output_path: Path) -> None:
    data = _as_dict(config.get("data"))
    interim = Path(str(data.get("interim", ""))).resolve()
    if not str(interim):
        raise SmallSliceCliError("data.interim is not configured.")
    try:
        output_path.resolve().relative_to(interim)
    except ValueError as exc:
        raise SmallSliceCliError(
            f"Refusing to write small-slice output outside data.interim: {output_path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
