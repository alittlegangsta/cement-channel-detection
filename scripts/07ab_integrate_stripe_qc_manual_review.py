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
from cement_channel.qc.stripe_manual_review_integration import (  # noqa: E402
    integrate_stripe_qc_with_manual_review_from_paths,
)


class StripeManualReviewIntegrationCliError(RuntimeError):
    """Raised when stripe QC manual-review integration cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate CAST horizontal stripe QC into MVP-4X manual review pack."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--manual-review-dir", default=None)
    parser.add_argument("--selected-intervals", default=None)
    parser.add_argument("--stripe-inventory", default=None)
    parser.add_argument("--stripe-sensitivity", default=None)
    parser.add_argument("--heatmap-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        reports = _reports_dir(config)
        manual_review_dir = (
            Path(args.manual_review_dir)
            if args.manual_review_dir
            else reports / "mvp4x_label_semantics_manual_review_v001"
        )
        selected = (
            Path(args.selected_intervals)
            if args.selected_intervals
            else manual_review_dir / "selected_intervals.csv"
        )
        heatmap_dir = (
            Path(args.heatmap_dir)
            if args.heatmap_dir
            else manual_review_dir / "interval_cast_heatmaps"
        )
        inventory = (
            Path(args.stripe_inventory)
            if args.stripe_inventory
            else reports / "cast_horizontal_stripe_inventory_v001.csv"
        )
        sensitivity = (
            Path(args.stripe_sensitivity)
            if args.stripe_sensitivity
            else reports / "cast_horizontal_stripe_target_sensitivity_v001.csv"
        )
        for path, action in (
            (manual_review_dir, "write"),
            (selected, "read"),
            (heatmap_dir, "read"),
            (inventory, "read"),
            (sensitivity, "read"),
        ):
            _ensure_within_reports(reports, path, action=action)
        result = integrate_stripe_qc_with_manual_review_from_paths(
            selected_intervals_csv=selected,
            stripe_inventory_csv=inventory,
            stripe_sensitivity_csv=sensitivity,
            manual_review_dir=manual_review_dir,
            heatmap_dir=heatmap_dir,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        StripeManualReviewIntegrationCliError,
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
        "Stripe-aware manual review integration completed "
        f"selected={result.selected_interval_count}; "
        f"actions={result.action_counts}; "
        f"stripe_window={result.stripe_in_heatmap_window_count}; "
        f"warnings={len(result.warnings)}."
    )
    print(f"Wrote CSV: {result.output_csv}")
    print(f"Wrote JSON: {result.output_json}")
    print(f"Wrote summary: {result.summary_md}")
    return 0


def _reports_dir(config: dict[str, Any]) -> Path:
    data = _as_dict(config.get("data"))
    if data.get("reports"):
        return Path(str(data["reports"])).resolve()
    raise StripeManualReviewIntegrationCliError("data.reports is not configured.")


def _ensure_within_reports(reports: Path, path: Path, *, action: str) -> None:
    resolved = path.resolve()
    if action == "write" and resolved == reports:
        return
    try:
        resolved.relative_to(reports)
    except ValueError as exc:
        raise StripeManualReviewIntegrationCliError(
            f"Refusing to {action} path outside data.reports: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
