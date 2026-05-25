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
from cement_channel.training.simple_baseline import (  # noqa: E402
    run_simple_baseline_from_config,
    write_simple_baseline_outputs,
)


class SimpleBaselineCliError(RuntimeError):
    """Raised when MVP-4B simple baseline cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP-4B simple baseline sanity model.")
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
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-csv", default=None)
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
            "simple_baseline_report_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "simple_baseline_report_v001.json",
        )
        output_csv = _resolve_report_path(config, args.output_csv, "simple_baseline_v001.csv")
        _ensure_path_within(config, sample_table_npz, key="interim", action="read")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        _ensure_path_within(config, output_csv, key="reports", action="write")
        report, rows = run_simple_baseline_from_config(
            sample_table_npz=sample_table_npz,
            baseline_config_path=args.baseline_config,
        )
        if not args.dry_run:
            write_simple_baseline_outputs(
                report,
                rows,
                output_report_md=output_md,
                output_report_json=output_json,
                output_csv=output_csv,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        SimpleBaselineCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Simple baseline sanity model "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"selected_samples={report.sample_counts['selected_samples']}; "
        f"production_training={report.production_training}; "
        f"no_final_labels={report.no_final_labels}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote CSV report: {output_csv}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise SimpleBaselineCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise SimpleBaselineCliError("data.reports is not configured.")


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
        raise SimpleBaselineCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise SimpleBaselineCliError(
            f"Refusing to {action} simple baseline path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
