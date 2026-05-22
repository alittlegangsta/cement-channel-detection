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
from cement_channel.visualization.label_review import generate_label_review_figures  # noqa: E402
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402


class LabelReviewCliError(RuntimeError):
    """Raised when label review figures cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP-3 CAST label review figures.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--input-npz", default=None)
    parser.add_argument("--baseline-npz", default=None)
    parser.add_argument("--weak-label-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-depth-pixels", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        input_npz = _resolve_interim_path(config, args.input_npz, "cast_label_input_v001.npz")
        baseline_npz = _resolve_interim_path(
            config,
            args.baseline_npz,
            "cast_zc_baseline_v001.npz",
        )
        weak_label_npz = _resolve_label_path(
            config,
            args.weak_label_npz,
            "cast_weak_label_candidates_v001.npz",
        )
        output_dir = _resolve_review_dir(config, args.output_dir)
        _ensure_path_within(config, input_npz, key="interim", action="read")
        _ensure_path_within(config, baseline_npz, key="interim", action="read")
        _ensure_path_within(config, weak_label_npz, key="labels", action="read")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        if args.dry_run:
            report = None
        else:
            report = generate_label_review_figures(
                cast_label_input_npz=input_npz,
                cast_baseline_npz=baseline_npz,
                weak_label_npz=weak_label_npz,
                output_dir=output_dir,
                overwrite=args.overwrite,
                max_depth_pixels=args.max_depth_pixels,
            )
    except (
        ManifestBuildError,
        LabelReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PlottingDependencyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run: review figures would be written under {output_dir}.")
        return 0
    assert report is not None
    print(
        "Label review figures "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"figures={len(report.figures)}; "
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
    raise LabelReviewCliError("data.interim is not configured.")


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
    raise LabelReviewCliError("data.root or data.labels is required for label inputs.")


def _resolve_review_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "label_review_v001"
    raise LabelReviewCliError("data.reports is not configured.")


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
        raise LabelReviewCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise LabelReviewCliError(
            f"Refusing to {action} label review path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
