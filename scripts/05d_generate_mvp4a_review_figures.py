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
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402
from cement_channel.visualization.mvp4a_review import generate_mvp4a_review_figures  # noqa: E402


class Mvp4aReviewCliError(RuntimeError):
    """Raised when MVP-4A review figures cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP-4A review figures.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--label-samples-npz", default=None)
    parser.add_argument("--features-npz", default=None)
    parser.add_argument("--correlation-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-depth-pixels", type=int, default=1200)
    parser.add_argument("--max-distribution-samples", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        label_npz = _resolve_interim_path(
            config,
            args.label_samples_npz,
            "xsi_label_samples_v001.npz",
        )
        features_npz = _resolve_feature_path(
            config,
            args.features_npz,
            "xsi_basic_features_v001.npz",
        )
        correlation_csv = _resolve_report_path(
            config,
            args.correlation_csv,
            "xsi_cast_correlation_v001.csv",
        )
        output_dir = _resolve_review_dir(config, args.output_dir)
        _ensure_path_within(config, label_npz, key="interim", action="read")
        _ensure_path_within(config, features_npz, key="features", action="read")
        _ensure_path_within(config, correlation_csv, key="reports", action="read")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        if args.dry_run:
            report = None
        else:
            report = generate_mvp4a_review_figures(
                label_samples_npz=label_npz,
                basic_features_npz=features_npz,
                correlation_csv=correlation_csv,
                output_dir=output_dir,
                overwrite=args.overwrite,
                max_depth_pixels=args.max_depth_pixels,
                max_distribution_samples=args.max_distribution_samples,
            )
    except (
        ManifestBuildError,
        Mvp4aReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PlottingDependencyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run: MVP-4A review figures would be written under {output_dir}.")
        return 0
    assert report is not None
    print(
        "MVP-4A review figures "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"figures={len(report.figures)}; "
        f"no_model_training={report.no_model_training}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote review directory: {output_dir}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise Mvp4aReviewCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise Mvp4aReviewCliError("data.reports is not configured.")


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
    raise Mvp4aReviewCliError("data.root or data.features is required for feature inputs.")


def _resolve_review_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "mvp4a_review_v001"
    raise Mvp4aReviewCliError("data.reports is not configured.")


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
        raise Mvp4aReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise Mvp4aReviewCliError(
            f"Refusing to {action} MVP-4A review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
