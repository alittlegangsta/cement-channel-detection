from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.xsi_geometry import ReceiverGeometry  # noqa: E402
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.labels.geometry_aware_cast_aggregation import (  # noqa: E402
    GeometryAwareAggregationConfig,
    build_geometry_aware_depth_labels,
    build_geometry_aware_depth_labels_from_paths,
)


class GeometryAwareDepthLabelCliError(RuntimeError):
    """Raised when geometry-aware CAST aggregation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build geometry-aware CAST depth weak-label candidate aggregates."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--geometry-config", default="configs/xsi_geometry.example.yaml")
    parser.add_argument("--cast-weak-label-npz", default=None)
    parser.add_argument("--depth-level-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--interval-padding-ft", type=float, default=0.0)
    parser.add_argument("--nearest-tolerance-ft", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        cast_npz = _resolve_label_path(
            paths,
            args.cast_weak_label_npz,
            "cast_weak_label_candidates_v001.npz",
        )
        labels_npz = _resolve_interim_path(
            paths,
            args.depth_level_labels_npz,
            "depth_level_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        output_npz = _resolve_interim_path(
            paths,
            args.output_npz,
            "geometry_aware_depth_labels_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "geometry_aware_depth_labels_report_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "geometry_aware_depth_labels_report_v001.json",
        )
        _ensure_path_within(paths, cast_npz, key="labels", action="read")
        _ensure_path_within(paths, labels_npz, key="interim", action="read")
        _ensure_path_within(paths, features_npz, key="interim", action="read")
        _ensure_path_within(paths, output_npz, key="interim", action="write")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        config = GeometryAwareAggregationConfig(
            interval_padding_ft=args.interval_padding_ft,
            nearest_tolerance_ft=args.nearest_tolerance_ft,
        )
        if args.dry_run:
            import numpy as np  # noqa: PLC0415

            with np.load(cast_npz, allow_pickle=False) as data:
                cast_arrays = {key: data[key] for key in data.files}
            with np.load(labels_npz, allow_pickle=False) as data:
                label_arrays = {key: data[key] for key in data.files}
            with np.load(features_npz, allow_pickle=False) as data:
                feature_arrays = {key: data[key] for key in data.files}
            _arrays, report = build_geometry_aware_depth_labels(
                cast_arrays=cast_arrays,
                depth_label_arrays=label_arrays,
                feature_arrays=feature_arrays,
                geometry=ReceiverGeometry.from_yaml(args.geometry_config),
                aggregation_config=config,
                inputs={
                    "cast_weak_label_npz": str(cast_npz),
                    "depth_level_labels_npz": str(labels_npz),
                    "depth_level_features_npz": str(features_npz),
                    "geometry_config_path": str(args.geometry_config),
                },
                output_npz=output_npz,
            )
        else:
            report = build_geometry_aware_depth_labels_from_paths(
                cast_weak_label_npz=cast_npz,
                depth_level_labels_npz=labels_npz,
                depth_level_features_npz=features_npz,
                geometry_config_path=args.geometry_config,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
                aggregation_config=config,
            )
    except (
        ManifestBuildError,
        GeometryAwareDepthLabelCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    best = report.mode_sign_summaries[0] if report.mode_sign_summaries else {}
    print(
        "Geometry-aware depth labels "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"mode_sign_count={len(report.mode_sign_summaries)}; "
        f"first_positive_fraction={best.get('positive_fraction')}; "
        f"raw_zc_available={report.raw_zc_available}; "
        f"no_final_labels={report.no_final_labels}."
    )
    if args.dry_run:
        print("Dry run: no NPZ/Markdown/JSON outputs written.")
    else:
        print(f"Wrote NPZ: {output_npz}")
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryAwareDepthLabelCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryAwareDepthLabelCliError("data.reports is not configured.")


def _resolve_label_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    labels = data.get("labels")
    if labels:
        return Path(str(labels)) / filename
    root = data.get("root")
    if root:
        return Path(str(root)) / "labels" / filename
    raise GeometryAwareDepthLabelCliError("data.root or data.labels is required.")


def _ensure_path_within(
    config: dict[str, Any],
    path: Path,
    *,
    key: str,
    action: str,
) -> None:
    data = _as_dict(config.get("data"))
    if key == "labels":
        root = Path(str(data.get("labels", Path(str(data.get("root", ""))) / "labels"))).resolve()
    else:
        root = Path(str(data.get(key, ""))).resolve()
    if not str(root):
        raise GeometryAwareDepthLabelCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryAwareDepthLabelCliError(
            f"Refusing to {action} geometry-aware label path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
