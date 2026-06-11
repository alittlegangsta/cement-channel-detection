from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.modeling.mvp4x_max_auto import MaxAutoError, MaxAutoPaths  # noqa: E402
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402
from cement_channel.visualization.mvp4x_cast_heatmap_supplement import (  # noqa: E402
    DEFAULT_HEATMAP_DIR_NAME,
    DEFAULT_REVIEW_DIR_NAME,
    generate_cast_heatmap_supplement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-4X LS/MW CAST 2D heatmap supplement figures."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--selected-intervals", default=None)
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cast-label-input", default=None)
    parser.add_argument("--cast-baseline", default=None)
    parser.add_argument("--connected-mask-npz", default=None)
    parser.add_argument("--depth-half-window-ft", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = MaxAutoPaths.from_config(load_paths_config(args.paths_config))
        review_dir = (
            Path(args.review_dir) if args.review_dir else paths.reports / DEFAULT_REVIEW_DIR_NAME
        )
        selected = (
            Path(args.selected_intervals)
            if args.selected_intervals
            else review_dir / "selected_intervals.csv"
        )
        output_dir = (
            Path(args.output_dir) if args.output_dir else review_dir / DEFAULT_HEATMAP_DIR_NAME
        )
        cast_label_input = (
            Path(args.cast_label_input)
            if args.cast_label_input
            else paths.interim / "cast_label_input_v001.npz"
        )
        cast_baseline = (
            Path(args.cast_baseline)
            if args.cast_baseline
            else paths.interim / "cast_zc_baseline_v001.npz"
        )
        result = generate_cast_heatmap_supplement(
            selected_intervals_csv=selected,
            cast_label_input_npz=cast_label_input,
            cast_baseline_npz=cast_baseline,
            review_dir=review_dir,
            output_dir=output_dir,
            connected_mask_npz=args.connected_mask_npz,
            depth_half_window_ft=args.depth_half_window_ft,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        MaxAutoError,
        PlottingDependencyError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X LS/MW CAST heatmap supplement completed "
        f"heatmaps={result.heatmap_count}; "
        f"connected_traceable={result.connected_mask_traceable_count}; "
        f"warnings={len(result.warnings)}; output_dir={result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
