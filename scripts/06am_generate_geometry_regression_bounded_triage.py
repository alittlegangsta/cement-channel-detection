from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.evaluation.geometry_regression_bounded_triage import (  # noqa: E402
    build_bounded_triage,
    write_bounded_triage_outputs,
)


class GeometryRegressionBoundedTriageCliError(RuntimeError):
    """Raised when bounded geometry regression triage cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bounded root-cause triage for geometry regression QA."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--inventory-json", default=None)
    parser.add_argument("--invariants-json", default=None)
    parser.add_argument("--audit-json", default=None)
    parser.add_argument("--feature-correlation-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        labels_npz = _resolve_interim_path(
            paths,
            args.regression_labels_npz,
            "geometry_aware_regression_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        inventory_json = _resolve_report_path(
            paths,
            args.inventory_json,
            "geometry_regression_contract_inventory_v001.json",
        )
        invariants_json = _resolve_report_path(
            paths,
            args.invariants_json,
            "geometry_regression_contract_invariants_v001.json",
        )
        audit_json = _resolve_report_path(
            paths,
            args.audit_json,
            "geometry_regression_audit_v001.json",
        )
        feature_csv = _resolve_report_path(
            paths,
            args.feature_correlation_csv,
            "geometry_regression_feature_correlation_summary_v001.csv",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_md,
            "geometry_regression_bounded_triage_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_json,
            "geometry_regression_bounded_triage_v001.json",
        )
        output_csv = _resolve_report_path(
            paths,
            args.output_csv,
            "geometry_regression_bounded_triage_v001.csv",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (inventory_json, "reports", "read"),
            (invariants_json, "reports", "read"),
            (audit_json, "reports", "read"),
            (feature_csv, "reports", "read"),
            (output_md, "reports", "write"),
            (output_json, "reports", "write"),
            (output_csv, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        label_arrays = _load_npz(labels_npz)
        feature_arrays = _load_npz(features_npz)
        inventory = _load_json(inventory_json)
        invariants = _load_json(invariants_json)
        audit = _load_json(audit_json)
        feature_rows = _load_csv(feature_csv)
        report, rows = build_bounded_triage(
            label_arrays=label_arrays,
            feature_arrays=feature_arrays,
            inventory_report=inventory,
            invariants_report=invariants,
            audit_report=audit,
            feature_correlation_rows=feature_rows,
            inputs={
                "regression_labels_npz": str(labels_npz),
                "depth_level_features_npz": str(features_npz),
                "inventory_json": str(inventory_json),
                "invariants_json": str(invariants_json),
                "audit_json": str(audit_json),
                "feature_correlation_csv": str(feature_csv),
            },
        )
        write_bounded_triage_outputs(
            report,
            rows,
            output_md=output_md,
            output_json=output_json,
            output_csv=output_csv,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        GeometryRegressionBoundedTriageCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression bounded triage "
        f"classification={report.root_cause_classification}; "
        f"errors={len(report.errors)}; "
        f"no_final_labels={report.no_final_labels}."
    )
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
    raise GeometryRegressionBoundedTriageCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionBoundedTriageCliError("data.reports is not configured.")


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
        raise GeometryRegressionBoundedTriageCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionBoundedTriageCliError(
            "Refusing to "
            f"{action} geometry regression bounded triage path outside data.{key}: {path}"
        ) from exc


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
