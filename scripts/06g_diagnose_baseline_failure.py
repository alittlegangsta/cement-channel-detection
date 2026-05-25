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
from cement_channel.training.baseline_diagnostics import (  # noqa: E402
    diagnose_baseline_failure_from_config,
    write_baseline_failure_diagnostics_outputs,
)
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402


class BaselineFailureDiagnosticsCliError(RuntimeError):
    """Raised when baseline failure diagnostics cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose MVP-4B simple baseline no-go result.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--baseline-config",
        default="configs/mvp4b_simple_baseline.example.yaml",
    )
    parser.add_argument("--sample-table-npz", default=None)
    parser.add_argument("--simple-baseline-report", default=None)
    parser.add_argument("--simple-baseline-csv", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-dir", default=None)
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
        baseline_report = _resolve_report_path(
            config,
            args.simple_baseline_report,
            "simple_baseline_report_v001.json",
        )
        baseline_csv = _resolve_report_path(
            config,
            args.simple_baseline_csv,
            "simple_baseline_v001.csv",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "baseline_failure_diagnostics_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "baseline_failure_diagnostics_v001.json",
        )
        output_dir = _resolve_output_dir(config, args.output_dir)
        _ensure_path_within(config, sample_table_npz, key="interim", action="read")
        _ensure_path_within(config, baseline_report, key="reports", action="read")
        _ensure_path_within(config, baseline_csv, key="reports", action="read")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        report = diagnose_baseline_failure_from_config(
            sample_table_npz=sample_table_npz,
            simple_baseline_report_json=baseline_report,
            simple_baseline_csv=baseline_csv,
            baseline_config_path=args.baseline_config,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
        if not args.dry_run:
            write_baseline_failure_diagnostics_outputs(
                report,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        BaselineFailureDiagnosticsCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PlottingDependencyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Baseline failure diagnostics "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"no_go_confirmed={report.no_go_confirmed}; "
        f"reason_classes={','.join(report.no_go_reason_classes)}."
    )
    if args.dry_run:
        print("Dry run: no Markdown/JSON outputs written.")
    else:
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
    raise BaselineFailureDiagnosticsCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise BaselineFailureDiagnosticsCliError("data.reports is not configured.")


def _resolve_output_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "baseline_failure_diagnostics_v001"
    raise BaselineFailureDiagnosticsCliError("data.reports is not configured.")


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
        raise BaselineFailureDiagnosticsCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise BaselineFailureDiagnosticsCliError(
            f"Refusing to {action} baseline failure diagnostics path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
