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
from cement_channel.labels.threshold_sensitivity import (  # noqa: E402
    run_threshold_sensitivity_from_config,
    write_threshold_sensitivity_outputs,
)


class ThresholdSensitivityCliError(RuntimeError):
    """Raised when threshold sensitivity cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVP-3R CAST weak-label threshold sensitivity.",
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--label-config", default="configs/label.cast_weak_v001.example.yaml")
    parser.add_argument("--input-npz", default=None)
    parser.add_argument("--baseline-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-report-csv", default=None)
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
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "label_threshold_sensitivity_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "label_threshold_sensitivity_v001.json",
        )
        output_csv = _resolve_report_path(
            config,
            args.output_report_csv,
            "label_threshold_sensitivity_v001.csv",
        )
        _ensure_path_within(config, input_npz, key="interim", action="read")
        _ensure_path_within(config, baseline_npz, key="interim", action="read")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        _ensure_path_within(config, output_csv, key="reports", action="write")
        report = run_threshold_sensitivity_from_config(
            cast_label_input_npz=input_npz,
            cast_baseline_npz=baseline_npz,
            label_config_path=args.label_config,
        )
        if not args.dry_run:
            write_threshold_sensitivity_outputs(
                report,
                output_report_md=output_md,
                output_report_json=output_json,
                output_report_csv=output_csv,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        ThresholdSensitivityCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Label threshold sensitivity "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"runs={len(report.results)}; "
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
    raise ThresholdSensitivityCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise ThresholdSensitivityCliError("data.reports is not configured.")


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
        raise ThresholdSensitivityCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ThresholdSensitivityCliError(
            f"Refusing to {action} threshold sensitivity path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
