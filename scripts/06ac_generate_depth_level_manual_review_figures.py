from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.visualization.depth_level_manual_review import (  # noqa: E402
    generate_depth_level_manual_review_figures,
)
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402


class DepthLevelManualReviewFigureCliError(RuntimeError):
    """Raised when depth-level manual review figures cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-4B-R4c+ depth-level manual review figures."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--review-intervals-json", default=None)
    parser.add_argument("--depth-level-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-interval-panels", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        review_json = _resolve_review_file(
            config,
            args.review_intervals_json,
            "review_intervals.json",
        )
        labels_npz = _resolve_interim_path(
            config,
            args.depth_level_labels_npz,
            "depth_level_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            config,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        output_dir = _resolve_review_dir(config, args.output_dir)
        _ensure_path_within(config, review_json, key="reports", action="read")
        _ensure_path_within(config, labels_npz, key="interim", action="read")
        _ensure_path_within(config, features_npz, key="interim", action="read")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        if args.dry_run:
            figure_report = None
        else:
            figure_report = generate_depth_level_manual_review_figures(
                review_intervals_json=review_json,
                depth_level_labels_npz=labels_npz,
                depth_level_features_npz=features_npz,
                output_dir=output_dir,
                overwrite=args.overwrite,
                max_interval_panels=args.max_interval_panels,
                max_points=args.max_points,
            )
    except (
        ManifestBuildError,
        DepthLevelManualReviewFigureCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
        PlottingDependencyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run: depth-level manual review figures would be written under {output_dir}.")
        return 0
    assert figure_report is not None
    print(
        "Depth-level manual review figures "
        f"errors={len(figure_report.errors)}; "
        f"warnings={len(figure_report.warnings)}; "
        f"figures={figure_report.figure_count}; "
        f"cast_panels={figure_report.interval_cast_panel_count}; "
        f"xsi_panels={figure_report.interval_xsi_panel_count}; "
        f"no_final_labels={figure_report.no_final_labels}."
    )
    print(f"Wrote review figure directory: {output_dir}")
    return 1 if figure_report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise DepthLevelManualReviewFigureCliError("data.interim is not configured.")


def _resolve_review_file(
    config: dict[str, Any],
    override: str | None,
    filename: str,
) -> Path:
    if override:
        return Path(override)
    return _resolve_review_dir(config, None) / filename


def _resolve_review_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "depth_level_manual_review_v001"
    raise DepthLevelManualReviewFigureCliError("data.reports is not configured.")


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
        raise DepthLevelManualReviewFigureCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelManualReviewFigureCliError(
            f"Refusing to {action} manual review figure path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
