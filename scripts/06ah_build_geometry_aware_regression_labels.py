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
from cement_channel.labels.cast_zc_source import load_controlled_cast_zc_source  # noqa: E402
from cement_channel.labels.geometry_aware_regression_labels import (  # noqa: E402
    build_geometry_aware_regression_labels,
    build_geometry_aware_regression_labels_from_paths,
    load_geometry_aware_regression_config,
)


class GeometryAwareRegressionLabelCliError(RuntimeError):
    """Raised when geometry-aware regression labels cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build geometry-aware continuous CAST regression weak-label candidates."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--label-config",
        default="configs/geometry_aware_regression_labels.example.yaml",
    )
    parser.add_argument("--geometry-config", default="configs/xsi_geometry.example.yaml")
    parser.add_argument("--raw-cast-zc-npz", action="append", default=None)
    parser.add_argument("--cast-baseline-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        raw_candidates = _resolve_raw_candidates(paths, args.raw_cast_zc_npz)
        baseline_npz = _resolve_optional_interim_path(
            paths,
            args.cast_baseline_npz,
            "cast_zc_baseline_v001.npz",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        output_npz = _resolve_interim_path(
            paths,
            args.output_npz,
            "geometry_aware_regression_labels_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "geometry_aware_regression_labels_report_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "geometry_aware_regression_labels_report_v001.json",
        )
        for candidate in raw_candidates:
            _ensure_path_within(paths, candidate, key="interim", action="read")
        if baseline_npz is not None:
            _ensure_path_within(paths, baseline_npz, key="interim", action="read")
        _ensure_path_within(paths, features_npz, key="interim", action="read")
        _ensure_path_within(paths, output_npz, key="interim", action="write")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        if args.dry_run:
            import numpy as np  # noqa: PLC0415

            config = load_geometry_aware_regression_config(args.label_config)
            raw_source = load_controlled_cast_zc_source(
                raw_candidates,
                min_finite_ratio=config.min_raw_zc_finite_ratio,
            )
            with np.load(features_npz, allow_pickle=False) as data:
                feature_arrays = {key: data[key] for key in data.files}
            baseline_arrays = None
            if baseline_npz is not None:
                with np.load(baseline_npz, allow_pickle=False) as data:
                    baseline_arrays = {key: data[key] for key in data.files}
            _arrays, report = build_geometry_aware_regression_labels(
                raw_source=raw_source,
                feature_arrays=feature_arrays,
                geometry=ReceiverGeometry.from_yaml(args.geometry_config),
                config=config,
                baseline_arrays=baseline_arrays,
                inputs=_inputs(args, raw_candidates, baseline_npz, features_npz),
                output_npz=output_npz,
            )
        else:
            report = build_geometry_aware_regression_labels_from_paths(
                raw_cast_zc_npz_candidates=raw_candidates,
                cast_baseline_npz=baseline_npz,
                depth_level_features_npz=features_npz,
                geometry_config_path=args.geometry_config,
                config_path=args.label_config,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        GeometryAwareRegressionLabelCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    first = report.kernel_summaries[0] if report.kernel_summaries else {}
    print(
        "Geometry-aware regression labels "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"kernels={len(report.kernel_summaries)}; "
        f"first_kernel={first.get('geometry_kernel')}; "
        f"first_nonzero_fraction={first.get('nonzero_fraction')}; "
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


def _resolve_raw_candidates(config: dict[str, Any], overrides: list[str] | None) -> list[Path]:
    if overrides:
        return [Path(item) for item in overrides]
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if not interim:
        raise GeometryAwareRegressionLabelCliError("data.interim is not configured.")
    root = Path(str(interim))
    return [root / "cast_label_input_v001.npz", root / "cast_zc_baseline_v001.npz"]


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryAwareRegressionLabelCliError("data.interim is not configured.")


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
    raise GeometryAwareRegressionLabelCliError("data.reports is not configured.")


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
        raise GeometryAwareRegressionLabelCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryAwareRegressionLabelCliError(
            f"Refusing to {action} geometry regression label path outside data.{key}: {path}"
        ) from exc


def _inputs(
    args: argparse.Namespace,
    raw_candidates: list[Path],
    baseline_npz: Path | None,
    features_npz: Path,
) -> dict[str, str]:
    return {
        "raw_cast_zc_npz_candidates": ", ".join(str(path) for path in raw_candidates),
        "cast_baseline_npz": str(baseline_npz or ""),
        "depth_level_features_npz": str(features_npz),
        "geometry_config_path": str(args.geometry_config),
        "config_path": str(args.label_config),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
