from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.relbearing_validation import (  # noqa: E402
    validate_relbearing_sign,
    write_relbearing_validation_outputs,
)
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class RelBearingValidationCliError(RuntimeError):
    """Raised when RelBearing validation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RelBearing sign candidates.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--depth-resample-preview-npz", default=None)
    parser.add_argument("--small-slice-summary-json", default=None)
    parser.add_argument("--depth-resample-report-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-config", default="configs/alignment.relbearing.example.yaml")
    parser.add_argument("--random-seed", type=int, default=20260520)
    parser.add_argument("--overlap-targeted", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        preview_npz = _resolve_interim_path(
            config,
            args.depth_resample_preview_npz,
            "depth_resample_overlap_preview_v001.npz"
            if args.overlap_targeted
            else "depth_resample_preview_v001.npz",
        )
        small_slice_summary = _resolve_interim_path(
            config,
            args.small_slice_summary_json,
            "small_slice_overlap_summary_v001.json"
            if args.overlap_targeted
            else "small_slice_summary_v001.json",
        )
        resample_report = _resolve_report_path(
            config,
            args.depth_resample_report_json,
            "depth_resample_overlap_preview_report.json"
            if args.overlap_targeted
            else "depth_resample_preview_report.json",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "relbearing_sign_validation_overlap_report.md"
            if args.overlap_targeted
            else "relbearing_sign_validation_report.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "relbearing_sign_validation_overlap_report.json"
            if args.overlap_targeted
            else "relbearing_sign_validation_report.json",
        )
        output_config = Path(args.output_config)
        _ensure_report_output(config, output_md)
        _ensure_report_output(config, output_json)
        report = validate_relbearing_sign(
            depth_resample_preview_npz=preview_npz,
            small_slice_summary_json=small_slice_summary,
            depth_resample_report_json=resample_report,
            random_seed=args.random_seed,
        )
        if not args.dry_run:
            write_relbearing_validation_outputs(
                report,
                output_json=output_json,
                output_md=output_md,
                output_config=output_config,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        RelBearingValidationCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "RelBearing sign validation "
        f"decision={report.decision}; "
        f"selected={report.selected_convention}; "
        f"confidence={report.confidence}; "
        f"warnings={len(report.warnings)}; "
        f"errors={len(report.errors)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote config example: {output_config}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise RelBearingValidationCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise RelBearingValidationCliError("data.reports is not configured.")


def _ensure_report_output(config: dict[str, Any], output_path: Path) -> None:
    data = _as_dict(config.get("data"))
    reports = Path(str(data.get("reports", ""))).resolve()
    if not str(reports):
        raise RelBearingValidationCliError("data.reports is not configured.")
    try:
        output_path.resolve().relative_to(reports)
    except ValueError as exc:
        raise RelBearingValidationCliError(
            f"Refusing to write RelBearing validation report outside data.reports: {output_path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
