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
    SmallSliceLimits,
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        mapping = _read_mapping(Path(args.mapping))
        output_json = _resolve_output_path(
            config,
            args.output_json,
            "small_slice_summary_v001.json",
        )
        output_npz = (
            _resolve_output_path(config, args.output_npz, "small_slice_v001.npz")
            if args.output_npz is not None
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
        result, arrays = read_small_slice(
            config,
            mapping,
            mapping_path=Path(args.mapping),
            limits=limits,
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
