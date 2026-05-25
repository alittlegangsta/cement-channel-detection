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
from cement_channel.training.feature_normalization import (  # noqa: E402
    FeatureNormalizationConfig,
    add_normalized_features_from_npz,
)


class FeatureNormalizationCliError(RuntimeError):
    """Raised when MVP-4B-R normalized feature generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add MVP-4B-R side/depth normalized features.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--input-sample-table-npz", default=None)
    parser.add_argument("--output-sample-table-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--rolling-window-samples", type=int, default=21)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        input_npz = _resolve_interim_path(
            paths,
            args.input_sample_table_npz,
            "baseline_sample_table_reweighted_v001.npz",
        )
        output_npz = _resolve_interim_path(
            paths,
            args.output_sample_table_npz,
            "baseline_sample_table_enhanced_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "enhanced_feature_report_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "enhanced_feature_report_v001.json",
        )
        _ensure_path_within(paths, input_npz, key="interim", action="read")
        _ensure_path_within(paths, output_npz, key="interim", action="write")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        config = FeatureNormalizationConfig(rolling_window_samples=args.rolling_window_samples)
        if args.dry_run:
            from cement_channel.training.feature_normalization import (  # noqa: PLC0415
                enhance_sample_table_features,
            )

            arrays = _load_npz(input_npz)
            _, report = enhance_sample_table_features(
                arrays,
                config=config,
                input_npz=input_npz,
                output_npz=output_npz,
            )
        else:
            report = add_normalized_features_from_npz(
                input_npz=input_npz,
                output_npz=output_npz,
                report_md=output_md,
                report_json=output_json,
                config=config,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        FeatureNormalizationCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Enhanced features generated "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"added_features={report.added_feature_count}; "
        f"enhanced_features={report.enhanced_transformed_feature_count}; "
        f"finite_ratio={report.enhanced_transformed_feature_finite_ratio}; "
        f"used_label_information={report.used_label_information_for_features}."
    )
    if args.dry_run:
        print("Dry run: no NPZ/Markdown/JSON outputs written.")
    else:
        print(f"Wrote enhanced sample table: {output_npz}")
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 1 if report.errors else 0


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise FeatureNormalizationCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise FeatureNormalizationCliError("data.reports is not configured.")


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
        raise FeatureNormalizationCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise FeatureNormalizationCliError(
            f"Refusing to {action} normalized feature path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
