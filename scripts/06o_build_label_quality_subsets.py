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
from cement_channel.labels.label_quality_subsets import (  # noqa: E402
    build_label_quality_subsets_from_config,
)


class LabelQualitySubsetCliError(RuntimeError):
    """Raised when label-quality subset generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP-4B-R3 label-quality subsets.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--label-quality-config",
        default="configs/mvp4b_label_quality_subsets.example.yaml",
    )
    parser.add_argument("--sample-table-npz", default=None)
    parser.add_argument("--cast-weak-label-npz", default=None)
    parser.add_argument("--output-subsets-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        sample_npz = _resolve_interim_path(
            paths,
            args.sample_table_npz,
            "baseline_sample_table_receiver_enhanced_v001.npz",
        )
        cast_npz = _resolve_label_path(
            paths,
            args.cast_weak_label_npz,
            "cast_weak_label_candidates_v001.npz",
        )
        output_npz = _resolve_interim_path(
            paths,
            args.output_subsets_npz,
            "label_quality_subsets_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "label_quality_subsets_report_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "label_quality_subsets_report_v001.json",
        )
        _ensure_path_within(paths, sample_npz, key="interim", action="read")
        _ensure_path_within(paths, output_npz, key="interim", action="write")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        if args.cast_weak_label_npz is not None:
            _ensure_label_path_within(paths, cast_npz, action="read")
        if args.dry_run:
            import numpy as np  # noqa: PLC0415

            from cement_channel.labels.label_quality_schema import (  # noqa: PLC0415
                load_label_quality_config,
            )
            from cement_channel.labels.label_quality_subsets import (  # noqa: PLC0415
                build_label_quality_subsets,
            )

            with np.load(sample_npz, allow_pickle=False) as sample_data:
                sample_arrays = {key: sample_data[key] for key in sample_data.files}
            _, report = build_label_quality_subsets(
                sample_arrays=sample_arrays,
                config=load_label_quality_config(args.label_quality_config),
                inputs={
                    "sample_table_npz": str(sample_npz),
                    "label_quality_config_path": str(args.label_quality_config),
                    "cast_weak_label_npz": str(cast_npz),
                },
                output_npz=output_npz,
            )
        else:
            report = build_label_quality_subsets_from_config(
                sample_table_npz=sample_npz,
                cast_weak_label_npz=cast_npz,
                label_quality_config_path=args.label_quality_config,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        LabelQualitySubsetCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    quality_positive = report.subset_counts["quality_strong_positive"]["sample_count"]
    quality_negative = report.subset_counts["quality_clear_negative"]["sample_count"]
    print(
        "Label-quality subsets "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"quality_strong_positive={quality_positive}; "
        f"quality_clear_negative={quality_negative}; "
        f"no_final_labels={report.no_final_labels}."
    )
    if args.dry_run:
        print("Dry run: no NPZ/Markdown/JSON outputs written.")
    else:
        print(f"Wrote label-quality subset NPZ: {output_npz}")
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
    raise LabelQualitySubsetCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise LabelQualitySubsetCliError("data.reports is not configured.")


def _resolve_label_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    root = data.get("root")
    if root:
        return Path(str(root)) / "labels" / filename
    raise LabelQualitySubsetCliError("data.root is not configured.")


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
        raise LabelQualitySubsetCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise LabelQualitySubsetCliError(
            f"Refusing to {action} label-quality subset path outside data.{key}: {path}"
        ) from exc


def _ensure_label_path_within(config: dict[str, Any], path: Path, *, action: str) -> None:
    data = _as_dict(config.get("data"))
    root = Path(str(data.get("root", ""))).resolve() / "labels"
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise LabelQualitySubsetCliError(
            f"Refusing to {action} label-quality input outside data.root/labels: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
