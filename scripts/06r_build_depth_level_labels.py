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
from cement_channel.labels.depth_level_labels import (  # noqa: E402
    build_depth_level_label_table,
    build_depth_level_labels_from_config,
)
from cement_channel.labels.depth_level_schema import load_depth_level_label_config  # noqa: E402


class DepthLevelLabelCliError(RuntimeError):
    """Raised when depth-level label generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MVP-4B-R4 depth-level CAST weak-label candidates."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--depth-level-config",
        default="configs/depth_level_label.example.yaml",
    )
    parser.add_argument("--cast-weak-label-npz", default=None)
    parser.add_argument("--xsi-label-samples-npz", default=None)
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
        cast_npz = _resolve_label_path(
            paths,
            args.cast_weak_label_npz,
            "cast_weak_label_candidates_v001.npz",
        )
        xsi_npz = _resolve_interim_path(
            paths,
            args.xsi_label_samples_npz,
            "xsi_label_samples_v001.npz",
        )
        sample_table_npz = (
            _resolve_interim_path(
                paths,
                args.sample_table_npz,
                "baseline_sample_table_receiver_enhanced_v001.npz",
            )
            if args.sample_table_npz is not None
            else None
        )
        output_npz = _resolve_interim_path(
            paths,
            args.output_npz,
            "depth_level_labels_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "depth_level_labels_report_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "depth_level_labels_report_v001.json",
        )
        _ensure_path_within(paths, cast_npz, key="labels", action="read")
        _ensure_path_within(paths, xsi_npz, key="interim", action="read")
        if sample_table_npz is not None:
            _ensure_path_within(paths, sample_table_npz, key="interim", action="read")
        _ensure_path_within(paths, output_npz, key="interim", action="write")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        if args.dry_run:
            import numpy as np  # noqa: PLC0415

            with np.load(cast_npz, allow_pickle=False) as cast_data:
                cast_arrays = {key: cast_data[key] for key in cast_data.files}
            with np.load(xsi_npz, allow_pickle=False) as xsi_data:
                xsi_arrays = {key: xsi_data[key] for key in xsi_data.files}
            _output, report = build_depth_level_label_table(
                cast_arrays=cast_arrays,
                xsi_arrays=xsi_arrays,
                config=load_depth_level_label_config(args.depth_level_config),
                inputs={
                    "cast_weak_label_npz": str(cast_npz),
                    "xsi_label_samples_npz": str(xsi_npz),
                    "depth_level_config_path": str(args.depth_level_config),
                    "sample_table_npz": str(sample_table_npz or ""),
                },
                output_npz=output_npz,
            )
        else:
            report = build_depth_level_labels_from_config(
                cast_weak_label_npz=cast_npz,
                xsi_label_samples_npz=xsi_npz,
                sample_table_npz=sample_table_npz,
                depth_level_config_path=args.depth_level_config,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthLevelLabelCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth-level labels "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"positive_fraction={report.positive_fraction}; "
        f"strong_positive={report.strong_positive_count}; "
        f"clear_negative={report.clear_negative_count}; "
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
    raise DepthLevelLabelCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelLabelCliError("data.reports is not configured.")


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
    raise DepthLevelLabelCliError("data.root or data.labels is required for label inputs.")


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
        raise DepthLevelLabelCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelLabelCliError(
            f"Refusing to {action} depth-level label path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
