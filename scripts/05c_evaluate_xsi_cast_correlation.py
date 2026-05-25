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
from cement_channel.evaluation.xsi_cast_correlation import (  # noqa: E402
    evaluate_xsi_cast_correlation_from_config,
    write_xsi_cast_correlation_outputs,
)


class XsiCastCorrelationCliError(RuntimeError):
    """Raised when MVP-4A correlation evaluation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MVP-4A XSI-CAST weak-label correlation.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--correlation-config",
        default="configs/mvp4a_xsi_cast_correlation.example.yaml",
    )
    parser.add_argument("--label-samples-npz", default=None)
    parser.add_argument("--features-npz", default=None)
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
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "xsi_cast_correlation_report_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "xsi_cast_correlation_report_v001.json",
        )
        output_csv = _resolve_report_path(
            config,
            args.output_csv,
            "xsi_cast_correlation_v001.csv",
        )
        _ensure_path_within(config, label_npz, key="interim", action="read")
        _ensure_path_within(config, features_npz, key="features", action="read")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        _ensure_path_within(config, output_csv, key="reports", action="write")
        report, rows = evaluate_xsi_cast_correlation_from_config(
            label_samples_npz=label_npz,
            basic_features_npz=features_npz,
            correlation_config_path=args.correlation_config,
        )
        if not args.dry_run:
            write_xsi_cast_correlation_outputs(
                report,
                rows,
                output_report_md=output_md,
                output_report_json=output_json,
                output_csv=output_csv,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        XsiCastCorrelationCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "XSI-CAST correlation "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"rows={len(rows)}; "
        "high_confidence_exists="
        f"{report.gate_observations['high_confidence_subset_exists']}; "
        "interpretable_signal="
        f"{report.gate_observations['interpretable_signal_separation']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote CSV: {output_csv}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise XsiCastCorrelationCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise XsiCastCorrelationCliError("data.reports is not configured.")


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
    raise XsiCastCorrelationCliError("data.root or data.features is required for feature inputs.")


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
        raise XsiCastCorrelationCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise XsiCastCorrelationCliError(
            f"Refusing to {action} XSI-CAST correlation path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
