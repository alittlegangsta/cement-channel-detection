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
from cement_channel.training.preprocessing_diagnostics import (  # noqa: E402
    diagnose_feature_preprocessing,
    write_preprocessing_diagnostics_outputs,
)
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402


class PreprocessingDiagnosticsCliError(RuntimeError):
    """Raised when MVP-4B preprocessing diagnostics cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose MVP-4B feature preprocessing.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--sample-table-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        sample_table_npz = _resolve_interim_path(
            config,
            args.sample_table_npz,
            "baseline_sample_table_v001.npz",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "feature_preprocessing_diagnostics_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "feature_preprocessing_diagnostics_v001.json",
        )
        output_dir = _resolve_output_dir(config, args.output_dir)
        _ensure_path_within(config, sample_table_npz, key="interim", action="read")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        if args.dry_run:
            report = None
        else:
            report = diagnose_feature_preprocessing(
                sample_table_npz=sample_table_npz,
                output_dir=output_dir,
                overwrite=args.overwrite,
                max_samples=args.max_samples,
            )
            write_preprocessing_diagnostics_outputs(
                report,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        PreprocessingDiagnosticsCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PlottingDependencyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run: preprocessing diagnostics would be written under {output_dir}.")
        return 0
    assert report is not None
    print(
        "Feature preprocessing diagnostics "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"figures={len(report.figures)}; "
        f"no_model_training={report.no_model_training}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote Markdown report: {output_md}")
    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote diagnostics directory: {output_dir}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise PreprocessingDiagnosticsCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise PreprocessingDiagnosticsCliError("data.reports is not configured.")


def _resolve_output_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "feature_preprocessing_diagnostics_v001"
    raise PreprocessingDiagnosticsCliError("data.reports is not configured.")


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
        raise PreprocessingDiagnosticsCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise PreprocessingDiagnosticsCliError(
            f"Refusing to {action} preprocessing diagnostics path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
