from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.relbearing_calibration import (  # noqa: E402
    build_calibration_report_from_files,
    write_calibration_outputs,
)
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.visualization.relbearing_review import (  # noqa: E402
    write_relbearing_review_figures,
)


class RelBearingCalibrationCliError(RuntimeError):
    """Raised when RelBearing calibration cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-2C RelBearing convention calibration review artifacts."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--depth-resample-overlap-preview-npz", default=None)
    parser.add_argument("--orientation-confidence-npz", default=None)
    parser.add_argument("--small-slice-overlap-npz", default=None)
    parser.add_argument("--relbearing-validation-overlap-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument(
        "--output-config",
        default="configs/alignment.relbearing_calibration.example.yaml",
    )
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--window-depth-samples", type=int, default=3)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--min-valid-windows", type=int, default=5)
    parser.add_argument("--min-orientation-confidence", type=float, default=0.5)
    parser.add_argument("--max-relbearing-jump-deg", type=float, default=45.0)
    parser.add_argument("--min-support-ratio", type=float, default=0.70)
    parser.add_argument("--min-score-gap", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        paths = _resolve_paths(config, args)
        _ensure_interim_input(config, paths["depth_resample_overlap_preview_npz"])
        _ensure_interim_input(config, paths["orientation_confidence_npz"])
        _ensure_interim_input(config, paths["small_slice_overlap_npz"])
        _ensure_report_input(config, paths["relbearing_validation_overlap_json"])
        _ensure_report_output(config, paths["output_report_md"])
        _ensure_report_output(config, paths["output_report_json"])
        _ensure_review_dir(config, paths["review_dir"])
        report, arrays = build_calibration_report_from_files(
            depth_resample_overlap_preview_npz=paths["depth_resample_overlap_preview_npz"],
            orientation_confidence_npz=paths["orientation_confidence_npz"],
            small_slice_overlap_npz=paths["small_slice_overlap_npz"],
            relbearing_validation_report_json=paths["relbearing_validation_overlap_json"],
            window_depth_samples=args.window_depth_samples,
            window_stride=args.window_stride,
            max_windows=args.max_windows,
            min_valid_windows=args.min_valid_windows,
            min_orientation_confidence=args.min_orientation_confidence,
            max_relbearing_jump_deg=args.max_relbearing_jump_deg,
            min_support_ratio=args.min_support_ratio,
            min_score_gap=args.min_score_gap,
        )
        if not args.dry_run:
            figures = write_relbearing_review_figures(
                report,
                arrays,
                output_dir=paths["review_dir"],
                overwrite=args.overwrite,
                max_windows=min(args.max_windows, 3),
            )
            report.figures = figures
            write_calibration_outputs(
                report,
                output_json=paths["output_report_json"],
                output_md=paths["output_report_md"],
                output_config=paths["output_config"],
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        RelBearingCalibrationCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "RelBearing calibration "
        f"decision={report.final_recommendation}; "
        f"valid_windows={report.valid_window_count}; "
        f"best={_best_id(report.best_hypothesis)}; "
        f"warnings={len(report.warnings)}; "
        f"errors={len(report.errors)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
        print(f"Wrote review figures: {paths['review_dir']}")
        print(f"Wrote config example: {paths['output_config']}")
    return 1 if report.errors else 0


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    data = _as_dict(config.get("data"))
    interim = Path(str(data.get("interim", "")))
    reports = Path(str(data.get("reports", "")))
    return {
        "depth_resample_overlap_preview_npz": Path(
            args.depth_resample_overlap_preview_npz
            or interim / "depth_resample_overlap_preview_v001.npz"
        ),
        "orientation_confidence_npz": Path(
            args.orientation_confidence_npz or interim / "orientation_confidence_v001.npz"
        ),
        "small_slice_overlap_npz": Path(
            args.small_slice_overlap_npz or interim / "small_slice_overlap_v001.npz"
        ),
        "relbearing_validation_overlap_json": Path(
            args.relbearing_validation_overlap_json
            or reports / "relbearing_sign_validation_overlap_report.json"
        ),
        "output_report_md": Path(
            args.output_report_md or reports / "relbearing_calibration_report.md"
        ),
        "output_report_json": Path(
            args.output_report_json or reports / "relbearing_calibration_report.json"
        ),
        "output_config": Path(args.output_config),
        "review_dir": Path(args.review_dir or reports / "relbearing_manual_review"),
    }


def _ensure_interim_input(config: dict[str, Any], path: Path) -> None:
    _ensure_path_within(config, path, key="interim", action="read")
    if not path.exists():
        raise RelBearingCalibrationCliError(f"Required interim input does not exist: {path}")


def _ensure_report_input(config: dict[str, Any], path: Path) -> None:
    _ensure_path_within(config, path, key="reports", action="read")
    if not path.exists():
        raise RelBearingCalibrationCliError(f"Required report input does not exist: {path}")


def _ensure_report_output(config: dict[str, Any], path: Path) -> None:
    _ensure_path_within(config, path, key="reports", action="write")


def _ensure_review_dir(config: dict[str, Any], path: Path) -> None:
    _ensure_path_within(config, path, key="reports", action="write")


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
        raise RelBearingCalibrationCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise RelBearingCalibrationCliError(
            f"Refusing to {action} RelBearing calibration path outside data.{key}: {path}"
        ) from exc


def _best_id(best_hypothesis: dict[str, Any] | None) -> str:
    if not best_hypothesis:
        return "none"
    hypothesis = best_hypothesis.get("hypothesis")
    if isinstance(hypothesis, dict):
        return str(hypothesis.get("hypothesis_id", "none"))
    return "none"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
