from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.features.mvp4x_stc_apes_pilot import (  # noqa: E402
    DEFAULT_INTERVALS,
    MAX_INTERVALS,
    MIN_INTERVALS,
    default_output_dir,
    run_sa_pilot_from_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVP-4X-SA bounded STC/APES research-only pilot."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--mapping", default="configs/raw_variable_mapping.confirmed.yaml")
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--interval-count", type=int, default=DEFAULT_INTERVALS)
    parser.add_argument("--interval-half-window-ft", type=float, default=2.5)
    parser.add_argument("--chunk-depth-samples", type=int, default=8)
    parser.add_argument("--max-time-samples", type=int, default=1024)
    parser.add_argument("--micro-benchmark-intervals", type=int, default=8)
    parser.add_argument("--receiver-limit", type=int, default=13)
    parser.add_argument("--side-limit", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.interval_count < MIN_INTERVALS or args.interval_count > MAX_INTERVALS:
            raise ValueError(
                f"--interval-count must be between {MIN_INTERVALS} and {MAX_INTERVALS}."
            )
        paths = load_paths_config(args.paths_config)
        snapshot_npz = (
            Path(args.snapshot_npz)
            if args.snapshot_npz
            else Path(str(paths["data"]["interim"])) / "mvp4x_research_snapshot_v001.npz"
        )
        output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(paths)
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X-SA bounded pilot "
                f"interval_count={args.interval_count}; "
                f"chunk_depth_samples={args.chunk_depth_samples}; "
                f"max_time_samples={args.max_time_samples}; "
                f"snapshot_npz={snapshot_npz}; "
                f"output_dir={output_dir}."
            )
            return 0
        report = run_sa_pilot_from_paths(
            paths_config=args.paths_config,
            mapping_path=args.mapping,
            snapshot_npz=snapshot_npz,
            output_dir=output_dir,
            interval_count=args.interval_count,
            interval_half_window_ft=args.interval_half_window_ft,
            chunk_depth_samples=args.chunk_depth_samples,
            max_time_samples=args.max_time_samples,
            micro_benchmark_intervals=args.micro_benchmark_intervals,
            receiver_limit=args.receiver_limit,
            side_limit=args.side_limit,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X-SA pilot completed "
        f"intervals={report.interval_count}; "
        f"skipped={report.skipped_interval_count}; "
        f"runtime_seconds={report.runtime_seconds:.3f}; "
        f"research_only={report.research_only}; "
        f"no_full_well_stc={report.no_full_well_stc}; "
        f"no_full_well_apes={report.no_full_well_apes}."
    )
    print(f"Wrote report JSON: {report.outputs.report_json}")
    print(f"Wrote interval CSV: {report.outputs.interval_csv}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
