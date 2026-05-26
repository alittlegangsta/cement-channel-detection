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
from cement_channel.visualization.depth_level_baseline_review import (  # noqa: E402
    generate_depth_level_baseline_review_figures,
)
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402


class DepthLevelBaselineReviewCliError(RuntimeError):
    """Raised when depth-level baseline review figures cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-4B-R4b depth-level baseline review figures."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--baseline-csv", default=None)
    parser.add_argument("--depth-level-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        report_json = _resolve_report_path(
            config,
            args.baseline_report,
            "depth_level_baseline_report_v001.json",
        )
        prediction_csv = _resolve_report_path(
            config,
            args.baseline_csv,
            "depth_level_baseline_report_v001.csv",
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
        _ensure_path_within(config, report_json, key="reports", action="read")
        _ensure_path_within(config, prediction_csv, key="reports", action="read")
        _ensure_path_within(config, labels_npz, key="interim", action="read")
        _ensure_path_within(config, features_npz, key="interim", action="read")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        if args.dry_run:
            review = None
        else:
            review = generate_depth_level_baseline_review_figures(
                baseline_report_json=report_json,
                baseline_csv=prediction_csv,
                depth_level_labels_npz=labels_npz,
                depth_level_features_npz=features_npz,
                output_dir=output_dir,
                overwrite=args.overwrite,
                max_points=args.max_points,
            )
    except (
        ManifestBuildError,
        DepthLevelBaselineReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PlottingDependencyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run: depth-level baseline review figures would be written under {output_dir}.")
        return 0
    assert review is not None
    print(
        "Depth-level baseline review figures "
        f"errors={len(review.errors)}; "
        f"warnings={len(review.warnings)}; "
        f"figures={len(review.figures)}; "
        f"preferred_variant={review.preferred_target_variant}; "
        f"preferred_model={review.preferred_model_type}; "
        f"no_final_labels={review.no_final_labels}."
    )
    print(f"Wrote review directory: {output_dir}")
    return 1 if review.errors else 0


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelBaselineReviewCliError("data.reports is not configured.")


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise DepthLevelBaselineReviewCliError("data.interim is not configured.")


def _resolve_review_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "depth_level_baseline_review_v001"
    raise DepthLevelBaselineReviewCliError("data.reports is not configured.")


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
        raise DepthLevelBaselineReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelBaselineReviewCliError(
            f"Refusing to {action} depth-level baseline review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
