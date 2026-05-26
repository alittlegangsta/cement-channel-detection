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
from cement_channel.features.depth_level_features import (  # noqa: E402
    build_depth_level_xsi_feature_table,
    build_depth_level_xsi_features_from_paths,
)


class DepthLevelXsiFeatureCliError(RuntimeError):
    """Raised when depth-level XSI feature generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP-4B-R4 depth-level XSI features.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--basic-features-npz", default=None)
    parser.add_argument("--sample-table-npz", default=None)
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
        basic_npz = _resolve_feature_path(
            paths,
            args.basic_features_npz,
            "xsi_basic_features_v001.npz",
        )
        sample_npz = _resolve_interim_path(
            paths,
            args.sample_table_npz,
            "baseline_sample_table_receiver_enhanced_v001.npz",
        )
        output_npz = _resolve_interim_path(
            paths,
            args.output_npz,
            "depth_level_xsi_features_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "depth_level_xsi_features_report_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "depth_level_xsi_features_report_v001.json",
        )
        _ensure_path_within(paths, basic_npz, key="features", action="read")
        _ensure_path_within(paths, sample_npz, key="interim", action="read")
        _ensure_path_within(paths, output_npz, key="interim", action="write")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        if args.dry_run:
            import numpy as np  # noqa: PLC0415

            with np.load(basic_npz, allow_pickle=False) as basic_data:
                basic_arrays = {key: basic_data[key] for key in basic_data.files}
            with np.load(sample_npz, allow_pickle=False) as sample_data:
                sample_arrays = {key: sample_data[key] for key in sample_data.files}
            _output, report = build_depth_level_xsi_feature_table(
                basic_arrays=basic_arrays,
                sample_arrays=sample_arrays,
                inputs={
                    "basic_features_npz": str(basic_npz),
                    "sample_table_npz": str(sample_npz),
                },
                output_npz=output_npz,
            )
        else:
            report = build_depth_level_xsi_features_from_paths(
                basic_features_npz=basic_npz,
                sample_table_npz=sample_npz,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthLevelXsiFeatureCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth-level XSI features "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"depth={report.depth_count}; "
        f"features={report.depth_feature_count}; "
        f"used_label_information={report.used_label_information_for_feature_construction}; "
        f"no_stc={report.no_stc}; no_apes={report.no_apes}."
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
    raise DepthLevelXsiFeatureCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelXsiFeatureCliError("data.reports is not configured.")


def _resolve_feature_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    features = data.get("features")
    if features:
        return Path(str(features)) / filename
    root = data.get("root")
    if root:
        return Path(str(root)) / "features" / filename
    raise DepthLevelXsiFeatureCliError("data.root or data.features is required.")


def _ensure_path_within(
    config: dict[str, Any],
    path: Path,
    *,
    key: str,
    action: str,
) -> None:
    data = _as_dict(config.get("data"))
    if key == "features":
        root = Path(
            str(data.get("features", Path(str(data.get("root", ""))) / "features"))
        ).resolve()
    else:
        root = Path(str(data.get(key, ""))).resolve()
    if not str(root):
        raise DepthLevelXsiFeatureCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelXsiFeatureCliError(
            f"Refusing to {action} depth-level XSI feature path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
