from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.evaluation.geometry_alignment_audit import (  # noqa: E402
    audit_geometry_alignment,
    audit_geometry_alignment_from_paths,
)


class GeometryAlignmentAuditCliError(RuntimeError):
    """Raised when geometry alignment audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit geometry-aware XSI-CAST alignment review targets."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--geometry-config", default="configs/xsi_geometry.example.yaml")
    parser.add_argument("--geometry-depth-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--refinement-report-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        labels_npz = _resolve_interim_path(
            paths,
            args.geometry_depth_labels_npz,
            "geometry_aware_depth_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        refinement_json = _resolve_report_path(
            paths,
            args.refinement_report_json,
            "depth_level_refinement_report_v001.json",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "geometry_alignment_audit_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "geometry_alignment_audit_v001.json",
        )
        output_csv = _resolve_report_path(
            paths,
            args.output_csv,
            "geometry_alignment_audit_v001.csv",
        )
        _ensure_path_within(paths, labels_npz, key="interim", action="read")
        _ensure_path_within(paths, features_npz, key="interim", action="read")
        _ensure_path_within(paths, refinement_json, key="reports", action="read")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        _ensure_path_within(paths, output_csv, key="reports", action="write")
        if args.dry_run:
            with np.load(labels_npz, allow_pickle=False) as data:
                label_arrays = {key: data[key] for key in data.files}
            with np.load(features_npz, allow_pickle=False) as data:
                feature_arrays = {key: data[key] for key in data.files}
            report, _rows = audit_geometry_alignment(
                geometry_arrays=label_arrays,
                feature_arrays=feature_arrays,
                refinement_report=json.loads(refinement_json.read_text(encoding="utf-8")),
                inputs={
                    "geometry_depth_labels_npz": str(labels_npz),
                    "depth_level_features_npz": str(features_npz),
                    "refinement_report_json": str(refinement_json),
                    "geometry_config_path": str(args.geometry_config),
                },
                output_csv=output_csv,
            )
        else:
            report = audit_geometry_alignment_from_paths(
                geometry_depth_labels_npz=labels_npz,
                depth_level_features_npz=features_npz,
                refinement_report_json=refinement_json,
                geometry_config_path=args.geometry_config,
                output_report_md=output_md,
                output_report_json=output_json,
                output_csv=output_csv,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        GeometryAlignmentAuditCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    best = report.best_combo or {}
    print(
        "Geometry alignment audit "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"recommendation={report.recommendation}; "
        f"best_mode={best.get('mode')}; "
        f"best_sign={best.get('sign')}; "
        f"best_margin={best.get('real_minus_permutation_margin')}; "
        f"no_final_labels={report.no_final_labels}."
    )
    if args.dry_run:
        print("Dry run: no Markdown/JSON/CSV outputs written.")
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
    raise GeometryAlignmentAuditCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryAlignmentAuditCliError("data.reports is not configured.")


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
        raise GeometryAlignmentAuditCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryAlignmentAuditCliError(
            f"Refusing to {action} geometry alignment audit path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
