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
from cement_channel.evaluation.depth_regime_review_policy import (  # noqa: E402
    DepthRegimeReviewPolicyError,
)
from cement_channel.visualization.geometry_regression_formal_review import (  # noqa: E402
    generate_geometry_regression_formal_review,
)
from cement_channel.visualization.matplotlib_utils import (  # noqa: E402
    PlottingDependencyError,
)


class GeometryRegressionFormalReviewCliError(RuntimeError):
    """Raised when formal review pack generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formal review-only geometry regression manual decision pack."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--review-policy-config",
        default="configs/geometry_regression_depth_regime_review.example.yaml",
    )
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--cast-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--decision-pack-md", default=None)
    parser.add_argument("--decision-pack-json", default=None)
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
        cast_npz = _resolve_interim_path(paths, args.cast_npz, "cast_label_input_v001.npz")
        output_dir = _resolve_report_path(
            paths,
            args.output_dir,
            "geometry_regression_formal_review_v001",
        )
        decision_pack_md = _resolve_report_path(
            paths,
            args.decision_pack_md,
            "geometry_regression_formal_review_decision_pack.md",
        )
        decision_pack_json = _resolve_report_path(
            paths,
            args.decision_pack_json,
            "geometry_regression_formal_review_decision_pack.json",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (cast_npz, "interim", "read"),
            (output_dir, "reports", "write"),
            (decision_pack_md, "reports", "write"),
            (decision_pack_json, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        pack = generate_geometry_regression_formal_review(
            labels_npz=labels_npz,
            features_npz=features_npz,
            cast_npz=cast_npz,
            policy_config=args.review_policy_config,
            output_dir=output_dir,
            decision_pack_md=decision_pack_md,
            decision_pack_json=decision_pack_json,
            overwrite=args.overwrite,
        )
    except (
        DepthRegimeReviewPolicyError,
        ManifestBuildError,
        PlottingDependencyError,
        GeometryRegressionFormalReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression formal review pack "
        f"intervals={pack.selected_manual_review_interval_count}; "
        f"figures={pack.figure_count}; "
        f"no_formal_cv_split_change={pack.no_formal_cv_split_change}; "
        f"no_model_training={pack.no_model_training}."
    )
    print(f"Wrote review dir: {output_dir}")
    print(f"Wrote decision pack Markdown: {decision_pack_md}")
    print(f"Wrote decision pack JSON: {decision_pack_json}")
    return 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryRegressionFormalReviewCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionFormalReviewCliError("data.reports is not configured.")


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
        raise GeometryRegressionFormalReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionFormalReviewCliError(
            f"Refusing to {action} formal review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())

