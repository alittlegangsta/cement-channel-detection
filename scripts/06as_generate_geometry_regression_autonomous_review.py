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
from cement_channel.evaluation.geometry_regression_autonomous_review import (  # noqa: E402
    generate_geometry_regression_autonomous_review,
)
from cement_channel.visualization.matplotlib_utils import (  # noqa: E402
    PlottingDependencyError,
)


class GeometryRegressionAutonomousReviewCliError(RuntimeError):
    """Raised when autonomous review pack generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometry regression autonomous review pack and decision."
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
    parser.add_argument("--cast-qc-json", default=None)
    parser.add_argument("--target-view-json", default=None)
    parser.add_argument("--morphology-json", default=None)
    parser.add_argument("--depth-regime-json", default=None)
    parser.add_argument("--near-far-json", default=None)
    parser.add_argument("--output-dir", default=None)
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
        cast_qc_json = _resolve_report_path(
            paths,
            args.cast_qc_json,
            "cast_zc_physical_qc_v001.json",
        )
        target_view_json = _resolve_report_path(
            paths,
            args.target_view_json,
            "geometry_regression_target_view_audit_v001.json",
        )
        morphology_json = _resolve_report_path(
            paths,
            args.morphology_json,
            "geometry_regression_morphology_audit_v001.json",
        )
        depth_regime_json = _resolve_report_path(
            paths,
            args.depth_regime_json,
            "geometry_regression_depth_regime_audit_v001.json",
        )
        near_far_json = _resolve_report_path(
            paths,
            args.near_far_json,
            "geometry_regression_near_far_divergence_audit_v001.json",
        )
        output_dir = _resolve_report_path(
            paths,
            args.output_dir,
            "geometry_regression_autonomous_review_v001",
        )
        decision_md = _resolve_report_path(
            paths,
            args.decision_md,
            "geometry_regression_autonomous_decision.md",
        )
        decision_json = _resolve_report_path(
            paths,
            args.decision_json,
            "geometry_regression_autonomous_decision.json",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (cast_npz, "interim", "read"),
            (cast_qc_json, "reports", "read"),
            (target_view_json, "reports", "read"),
            (morphology_json, "reports", "read"),
            (depth_regime_json, "reports", "read"),
            (near_far_json, "reports", "read"),
            (output_dir, "reports", "write"),
            (decision_md, "reports", "write"),
            (decision_json, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        report = generate_geometry_regression_autonomous_review(
            labels_npz=labels_npz,
            features_npz=features_npz,
            cast_npz=cast_npz,
            cast_qc_json=cast_qc_json,
            target_view_json=target_view_json,
            morphology_json=morphology_json,
            depth_regime_json=depth_regime_json,
            near_far_json=near_far_json,
            output_dir=output_dir,
            decision_md=decision_md,
            decision_json=decision_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        PlottingDependencyError,
        GeometryRegressionAutonomousReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression autonomous review pack "
        f"figures={len(report.figures)}; "
        f"selected_intervals={report.selected_interval_count}; "
        f"decision={report.decision}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote review dir: {output_dir}")
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
    raise GeometryRegressionAutonomousReviewCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionAutonomousReviewCliError("data.reports is not configured.")


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
        raise GeometryRegressionAutonomousReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionAutonomousReviewCliError(
            f"Refusing to {action} autonomous review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
