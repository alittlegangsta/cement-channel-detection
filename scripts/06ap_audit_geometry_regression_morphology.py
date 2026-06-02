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
from cement_channel.evaluation.geometry_regression_morphology_audit import (  # noqa: E402
    audit_geometry_regression_morphology_from_paths,
)


class GeometryRegressionMorphologyCliError(RuntimeError):
    """Raised when morphology audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit existing geometry regression morphology arrays."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
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
        output_md = _resolve_report_path(
            paths,
            args.output_md,
            "geometry_regression_morphology_audit_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_json,
            "geometry_regression_morphology_audit_v001.json",
        )
        output_csv = _resolve_report_path(
            paths,
            args.output_csv,
            "geometry_regression_morphology_audit_v001.csv",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (output_md, "reports", "write"),
            (output_json, "reports", "write"),
            (output_csv, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        report = audit_geometry_regression_morphology_from_paths(
            labels_npz=labels_npz,
            features_npz=features_npz,
            output_md=output_md,
            output_json=output_json,
            output_csv=output_csv,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        GeometryRegressionMorphologyCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression morphology audit "
        f"fold_rows={len(report.fold_summaries)}; "
        f"selected_intervals={len(report.selected_intervals)}; "
        f"errors={len(report.errors)}; "
        f"no_morphology_target_added={report.no_morphology_target_added}."
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
    raise GeometryRegressionMorphologyCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionMorphologyCliError("data.reports is not configured.")


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
        raise GeometryRegressionMorphologyCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionMorphologyCliError(
            f"Refusing to {action} morphology audit path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
