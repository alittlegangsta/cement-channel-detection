from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.io_hdf5 import build_tiny_hdf5_prototype  # noqa: E402
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class TinyHDF5CliError(RuntimeError):
    """Raised when tiny HDF5 prototype building cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny HDF5 schema prototype.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--small-slice-npz", default=None)
    parser.add_argument("--small-slice-summary", default=None)
    parser.add_argument("--output-hdf5", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        npz_path = _resolve_data_path(
            config,
            args.small_slice_npz,
            "interim",
            "small_slice_v001.npz",
        )
        summary_path = _resolve_data_path(
            config,
            args.small_slice_summary,
            "interim",
            "small_slice_summary_v001.json",
        )
        output_hdf5 = _resolve_data_path(
            config,
            args.output_hdf5,
            "processed",
            "tiny_aligned_prototype_v001.h5",
        )
        _ensure_processed_output(config, output_hdf5)
        if args.dry_run:
            result = None
        else:
            result = build_tiny_hdf5_prototype(
                small_slice_npz=npz_path,
                small_slice_summary=summary_path,
                output_hdf5=output_hdf5,
                overwrite=args.overwrite,
            )
    except (ManifestBuildError, TinyHDF5CliError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("Dry run: no outputs written.")
        return 0
    assert result is not None
    print(
        "Tiny HDF5 prototype "
        f"datasets={len(result.datasets)}; "
        f"errors={len(result.errors)}; "
        f"warnings={len(result.warnings)}."
    )
    print(f"Wrote HDF5: {output_hdf5}")
    return 1 if result.errors else 0


def _resolve_data_path(
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
    raise TinyHDF5CliError(f"data.{data_key} is not configured.")


def _ensure_processed_output(config: dict[str, Any], output_path: Path) -> None:
    data = _as_dict(config.get("data"))
    processed = Path(str(data.get("processed", ""))).resolve()
    if not str(processed):
        raise TinyHDF5CliError("data.processed is not configured.")
    try:
        output_path.resolve().relative_to(processed)
    except ValueError as exc:
        raise TinyHDF5CliError(
            f"Refusing to write tiny HDF5 outside data.processed: {output_path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
