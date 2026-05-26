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
from cement_channel.evaluation.depth_level_audit import (  # noqa: E402
    audit_depth_level_separation,
    audit_depth_level_separation_from_paths,
)
from cement_channel.labels.depth_level_schema import load_depth_level_label_config  # noqa: E402


class DepthLevelAuditCliError(RuntimeError):
    """Raised when depth-level separation audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MVP-4B-R4 depth-level XSI/CAST feature separation."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--depth-level-config",
        default="configs/depth_level_label.example.yaml",
    )
    parser.add_argument("--depth-level-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--side-level-audit-report-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-figure-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        label_npz = _resolve_interim_path(
            paths,
            args.depth_level_labels_npz,
            "depth_level_labels_v001.npz",
        )
        feature_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        side_report = _resolve_report_path(
            paths,
            args.side_level_audit_report_json,
            "subset_feature_separation_audit_v001.json",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "depth_level_separation_audit_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "depth_level_separation_audit_v001.json",
        )
        output_csv = _resolve_report_path(
            paths,
            args.output_csv,
            "depth_level_separation_audit_v001.csv",
        )
        output_fig_dir = _resolve_report_path(
            paths,
            args.output_figure_dir,
            "depth_level_separation_audit_v001",
        )
        _ensure_path_within(paths, label_npz, key="interim", action="read")
        _ensure_path_within(paths, feature_npz, key="interim", action="read")
        if side_report.exists():
            _ensure_path_within(paths, side_report, key="reports", action="read")
        for output_path in (output_md, output_json, output_csv, output_fig_dir):
            _ensure_path_within(paths, output_path, key="reports", action="write")
        if args.dry_run:
            import json  # noqa: PLC0415

            import numpy as np  # noqa: PLC0415

            with np.load(label_npz, allow_pickle=False) as label_data:
                label_arrays = {key: label_data[key] for key in label_data.files}
            with np.load(feature_npz, allow_pickle=False) as feature_data:
                feature_arrays = {key: feature_data[key] for key in feature_data.files}
            side_level_report = (
                json.loads(side_report.read_text(encoding="utf-8"))
                if side_report.exists()
                else None
            )
            report, _rows = audit_depth_level_separation(
                label_arrays=label_arrays,
                feature_arrays=feature_arrays,
                config=load_depth_level_label_config(args.depth_level_config),
                side_level_audit_report=side_level_report,
                inputs={
                    "depth_level_labels_npz": str(label_npz),
                    "depth_level_features_npz": str(feature_npz),
                    "depth_level_config_path": str(args.depth_level_config),
                    "side_level_audit_report_json": str(side_report),
                },
                output_csv=output_csv,
                output_figure_dir=output_fig_dir,
            )
        else:
            report = audit_depth_level_separation_from_paths(
                depth_level_labels_npz=label_npz,
                depth_level_features_npz=feature_npz,
                depth_level_config_path=args.depth_level_config,
                side_level_audit_report_json=side_report if side_report.exists() else None,
                output_report_md=output_md,
                output_report_json=output_json,
                output_csv=output_csv,
                output_figure_dir=output_fig_dir,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthLevelAuditCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    comparison = report.depth_vs_side_comparison
    print(
        "Depth-level separation audit "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"depth_best_abs_effect={comparison['depth_level_best_abs_effect_size']}; "
        f"side_best_abs_effect={comparison['side_level_best_abs_effect_size']}; "
        f"enhanced={report.depth_level_separation_enhanced}; "
        f"no_model_training={report.no_model_training}."
    )
    if args.dry_run:
        print("Dry run: no reports, CSV, or figures written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote CSV report: {output_csv}")
        print(f"Wrote review figures: {output_fig_dir}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise DepthLevelAuditCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelAuditCliError("data.reports is not configured.")


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
        raise DepthLevelAuditCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelAuditCliError(
            f"Refusing to {action} depth-level audit path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
