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
from cement_channel.evaluation.geometry_regression_depth_regime_review import (  # noqa: E402
    generate_geometry_regression_depth_regime_review,
)
from cement_channel.visualization.matplotlib_utils import (  # noqa: E402
    PlottingDependencyError,
)


class GeometryRegressionDepthRegimeReviewCliError(RuntimeError):
    """Raised when depth-regime review cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate review-only geometry regression depth-regime stratification audit."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--cast-npz", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--bins-csv", default=None)
    parser.add_argument("--boundaries-csv", default=None)
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--decision-md", default=None)
    parser.add_argument("--decision-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        labels_npz = _resolve_interim_path(
            paths,
            args.regression_labels_npz,
            "geometry_aware_regression_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        cast_npz = _resolve_interim_path(
            paths,
            args.cast_npz,
            "cast_label_input_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_md,
            "geometry_regression_depth_regime_audit_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_json,
            "geometry_regression_depth_regime_audit_v001.json",
        )
        bins_csv = _resolve_report_path(
            paths,
            args.bins_csv,
            "geometry_regression_depth_regime_bins_v001.csv",
        )
        boundaries_csv = _resolve_report_path(
            paths,
            args.boundaries_csv,
            "geometry_regression_depth_regime_boundaries_v001.csv",
        )
        review_dir = _resolve_report_path(
            paths,
            args.review_dir,
            "geometry_regression_depth_regime_review_v001",
        )
        decision_md = _resolve_report_path(
            paths,
            args.decision_md,
            "geometry_regression_depth_regime_decision.md",
        )
        decision_json = _resolve_report_path(
            paths,
            args.decision_json,
            "geometry_regression_depth_regime_decision.json",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (cast_npz, "interim", "read"),
            (output_md, "reports", "write"),
            (output_json, "reports", "write"),
            (bins_csv, "reports", "write"),
            (boundaries_csv, "reports", "write"),
            (review_dir, "reports", "write"),
            (decision_md, "reports", "write"),
            (decision_json, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        report = generate_geometry_regression_depth_regime_review(
            labels_npz=labels_npz,
            features_npz=features_npz,
            cast_npz=cast_npz,
            output_md=output_md,
            output_json=output_json,
            bins_csv=bins_csv,
            boundaries_csv=boundaries_csv,
            review_dir=review_dir,
            decision_md=decision_md,
            decision_json=decision_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        PlottingDependencyError,
        GeometryRegressionDepthRegimeReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression depth-regime review "
        f"folds={len(report.fold_summaries)}; "
        f"candidate_boundaries={len(report.candidate_boundaries)}; "
        f"formal_cv_protocol_changed={report.formal_cv_protocol_changed}; "
        f"no_model_training={report.no_model_training}."
    )
    print(f"Wrote audit Markdown: {output_md}")
    print(f"Wrote audit JSON: {output_json}")
    print(f"Wrote bins CSV: {bins_csv}")
    print(f"Wrote boundaries CSV: {boundaries_csv}")
    print(f"Wrote review dir: {review_dir}")
    print(f"Wrote decision Markdown: {decision_md}")
    print(f"Wrote decision JSON: {decision_json}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryRegressionDepthRegimeReviewCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionDepthRegimeReviewCliError("data.reports is not configured.")


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
        raise GeometryRegressionDepthRegimeReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionDepthRegimeReviewCliError(
            f"Refusing to {action} depth-regime review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
