from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import (  # noqa: E402
    ManifestBuildError,
    build_manifest_from_paths_config,
    write_manifest_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight raw data manifest without reading .mat contents."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
        help="Path to paths YAML config.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override data.raw from the paths config.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Override outputs.raw_inventory_csv.",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        default=None,
        help="Override outputs.data_manifest_json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing output files.",
    )
    parser.add_argument(
        "--max-wells",
        type=int,
        default=None,
        help="Limit scanned wells. Useful for future by_well layouts and tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing manifest outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest_from_paths_config(
            args.paths_config,
            raw_dir_override=args.raw_dir,
            output_csv=args.output_csv,
            output_json=args.output_json,
            max_wells=args.max_wells,
        )
        write_manifest_outputs(
            manifest,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except ManifestBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = manifest["summary"]
    print(
        "Scanned "
        f"{summary['file_count']} raw files across {summary['well_count']} well(s); "
        f"warnings: {summary['warning_count']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote CSV: {manifest['outputs']['raw_inventory_csv']}")
        print(f"Wrote JSON: {manifest['outputs']['data_manifest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
