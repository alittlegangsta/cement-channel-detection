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
from cement_channel.visualization.baseline_review import (  # noqa: E402
    generate_baseline_review_figures,
)
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402


class BaselineReviewCliError(RuntimeError):
    """Raised when simple baseline review figures cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP-4B baseline review figures.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--simple-baseline-report", default=None)
    parser.add_argument("--simple-baseline-csv", default=None)
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
            args.simple_baseline_report,
            "simple_baseline_report_v001.json",
        )
        prediction_csv = _resolve_report_path(
            config,
            args.simple_baseline_csv,
            "simple_baseline_v001.csv",
        )
        output_dir = _resolve_review_dir(config, args.output_dir)
        _ensure_path_within(config, report_json, key="reports", action="read")
        _ensure_path_within(config, prediction_csv, key="reports", action="read")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        if args.dry_run:
            review = None
        else:
            review = generate_baseline_review_figures(
                simple_baseline_report_json=report_json,
                simple_baseline_csv=prediction_csv,
                output_dir=output_dir,
                overwrite=args.overwrite,
                max_points=args.max_points,
            )
    except (
        ManifestBuildError,
        BaselineReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PlottingDependencyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run: baseline review figures would be written under {output_dir}.")
        return 0
    assert review is not None
    print(
        "Baseline review figures "
        f"errors={len(review.errors)}; "
        f"warnings={len(review.warnings)}; "
        f"figures={len(review.figures)}; "
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
    raise BaselineReviewCliError("data.reports is not configured.")


def _resolve_review_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "simple_baseline_review_v001"
    raise BaselineReviewCliError("data.reports is not configured.")


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
        raise BaselineReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise BaselineReviewCliError(
            f"Refusing to {action} baseline review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
