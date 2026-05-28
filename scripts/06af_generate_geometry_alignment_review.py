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
from cement_channel.visualization.geometry_alignment_review import (  # noqa: E402
    generate_geometry_alignment_review,
)


class GeometryAlignmentReviewCliError(RuntimeError):
    """Raised when geometry alignment review supplement cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometry-aware manual review supplement."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--geometry-config", default="configs/xsi_geometry.example.yaml")
    parser.add_argument("--geometry-depth-labels-npz", default=None)
    parser.add_argument("--geometry-alignment-audit-json", default=None)
    parser.add_argument("--review-intervals-json", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        labels_npz = _resolve_interim_path(
            paths,
            args.geometry_depth_labels_npz,
            "geometry_aware_depth_labels_v001.npz",
        )
        audit_json = _resolve_report_path(
            paths,
            args.geometry_alignment_audit_json,
            "geometry_alignment_audit_v001.json",
        )
        review_json = _resolve_report_path(
            paths,
            args.review_intervals_json,
            "depth_level_manual_review_v001/review_intervals.json",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        output_dir = _resolve_report_path(
            paths,
            args.output_dir,
            "geometry_aware_manual_review_v001",
        )
        _ensure_path_within(paths, labels_npz, key="interim", action="read")
        _ensure_path_within(paths, audit_json, key="reports", action="read")
        _ensure_path_within(paths, review_json, key="reports", action="read")
        _ensure_path_within(paths, features_npz, key="interim", action="read")
        _ensure_path_within(paths, output_dir, key="reports", action="write")
        if args.dry_run:
            print("Dry run: input paths resolved; no supplement outputs written.")
            return 0
        report = generate_geometry_alignment_review(
            geometry_depth_labels_npz=labels_npz,
            geometry_alignment_audit_json=audit_json,
            review_intervals_json=review_json,
            depth_level_features_npz=features_npz,
            output_dir=output_dir,
            overwrite=args.overwrite,
            geometry_config_path=args.geometry_config,
        )
    except (
        ManifestBuildError,
        GeometryAlignmentReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry-aware manual review supplement "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"changed_intervals={report.evidence_category_changed_interval_count}; "
        f"revisit_intervals={report.revisit_interval_count}; "
        f"figures={report.figure_count}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote review supplement: {output_dir}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryAlignmentReviewCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryAlignmentReviewCliError("data.reports is not configured.")


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
        raise GeometryAlignmentReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryAlignmentReviewCliError(
            f"Refusing to {action} geometry review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
