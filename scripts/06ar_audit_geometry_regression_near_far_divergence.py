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
from cement_channel.evaluation.geometry_regression_near_far_divergence_audit import (  # noqa: E402
    audit_geometry_regression_near_far_divergence_from_paths,
)


class GeometryRegressionNearFarDivergenceCliError(RuntimeError):
    """Raised when near/far divergence audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit geometry regression near/far ratio Pearson/Spearman divergence."
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
    parser.add_argument("--output-csv", default=None)
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
            "geometry_regression_near_far_divergence_audit_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_json,
            "geometry_regression_near_far_divergence_audit_v001.json",
        )
        output_csv = _resolve_report_path(
            paths,
            args.output_csv,
            "geometry_regression_near_far_divergence_audit_v001.csv",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (cast_npz, "interim", "read"),
            (output_md, "reports", "write"),
            (output_json, "reports", "write"),
            (output_csv, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        report = audit_geometry_regression_near_far_divergence_from_paths(
            labels_npz=labels_npz,
            features_npz=features_npz,
            cast_npz=cast_npz,
            output_md=output_md,
            output_json=output_json,
            output_csv=output_csv,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        GeometryRegressionNearFarDivergenceCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    divergence_count = sum(
        1 for row in report.divergence_summary if row["any_divergence_flag"]
    )
    outlier_count = sum(
        int(row["outlier_depth_count"]) for row in report.outlier_overlap_summary
    )
    print(
        "Geometry regression near/far divergence audit "
        f"features={report.audited_feature_names}; "
        f"divergence_rows={divergence_count}; "
        f"outlier_depths={outlier_count}; "
        f"denominator_small_derivable={report.denominator_small_derivable}; "
        f"no_ratio_preprocessing_change={report.no_ratio_preprocessing_change}."
    )
    print(f"Wrote Markdown report: {output_md}")
    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote CSV: {output_csv}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryRegressionNearFarDivergenceCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionNearFarDivergenceCliError("data.reports is not configured.")


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
        raise GeometryRegressionNearFarDivergenceCliError(
            f"data.{key} is not configured."
        )
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionNearFarDivergenceCliError(
            f"Refusing to {action} near/far divergence audit path outside data.{key}: "
            f"{path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
