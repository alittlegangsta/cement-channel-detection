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
from cement_channel.visualization.geometry_regression_review import (  # noqa: E402
    generate_geometry_regression_review,
)


class GeometryRegressionReviewCliError(RuntimeError):
    """Raised when geometry regression review cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometry-aware regression manual review supplement."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--regression-audit-json", default=None)
    parser.add_argument("--raw-cast-zc-npz", action="append", default=None)
    parser.add_argument("--cast-baseline-npz", default=None)
    parser.add_argument("--old-binary-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--geometry-config", default="configs/xsi_geometry.example.yaml")
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
            args.regression_labels_npz,
            "geometry_aware_regression_labels_v001.npz",
        )
        audit_json = _resolve_report_path(
            paths,
            args.regression_audit_json,
            "geometry_regression_audit_v001.json",
        )
        raw_candidates = _resolve_raw_candidates(paths, args.raw_cast_zc_npz)
        baseline_npz = _resolve_optional_interim_path(
            paths,
            args.cast_baseline_npz,
            "cast_zc_baseline_v001.npz",
        )
        old_binary_npz = _resolve_optional_interim_path(
            paths,
            args.old_binary_labels_npz,
            "geometry_aware_depth_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        output_dir = _resolve_report_dir(
            paths,
            args.output_dir,
            "geometry_regression_manual_review_v001",
        )
        _ensure_path_within(paths, labels_npz, key="interim", action="read")
        _ensure_path_within(paths, audit_json, key="reports", action="read")
        for candidate in raw_candidates:
            _ensure_path_within(paths, candidate, key="interim", action="read")
        if baseline_npz is not None:
            _ensure_path_within(paths, baseline_npz, key="interim", action="read")
        if old_binary_npz is not None:
            _ensure_path_within(paths, old_binary_npz, key="interim", action="read")
        _ensure_path_within(paths, features_npz, key="interim", action="read")
        _ensure_path_within(paths, output_dir, key="reports", action="write")
        if args.dry_run:
            print("Dry run: inputs and output locations resolved; no review files written.")
            return 0
        report = generate_geometry_regression_review(
            regression_labels_npz=labels_npz,
            regression_audit_json=audit_json,
            raw_cast_zc_npz_candidates=raw_candidates,
            cast_baseline_npz=baseline_npz,
            old_binary_labels_npz=old_binary_npz,
            depth_level_features_npz=features_npz,
            output_dir=output_dir,
            geometry_config_path=args.geometry_config,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        GeometryRegressionReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression review "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"primary_kernel={report.primary_kernel}; "
        f"interval_count={report.interval_count}; "
        f"figure_count={report.figure_count}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote review directory: {output_dir}")
    return 1 if report.errors else 0


def _resolve_raw_candidates(config: dict[str, Any], overrides: list[str] | None) -> list[Path]:
    if overrides:
        return [Path(item) for item in overrides]
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if not interim:
        raise GeometryRegressionReviewCliError("data.interim is not configured.")
    root = Path(str(interim))
    return [root / "cast_label_input_v001.npz", root / "cast_zc_baseline_v001.npz"]


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryRegressionReviewCliError("data.interim is not configured.")


def _resolve_optional_interim_path(
    config: dict[str, Any],
    override: str | None,
    filename: str,
) -> Path | None:
    path = _resolve_interim_path(config, override, filename)
    return path if path.exists() or override is not None else None


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionReviewCliError("data.reports is not configured.")


def _resolve_report_dir(config: dict[str, Any], override: str | None, dirname: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / dirname
    raise GeometryRegressionReviewCliError("data.reports is not configured.")


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
        raise GeometryRegressionReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionReviewCliError(
            f"Refusing to {action} geometry regression review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
