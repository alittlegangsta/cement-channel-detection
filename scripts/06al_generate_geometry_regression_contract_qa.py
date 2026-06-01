from __future__ import annotations

import argparse
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
from cement_channel.evaluation.geometry_regression_contract import (  # noqa: E402
    build_contract_invariants,
    build_contract_inventory,
    write_contract_invariant_outputs,
    write_contract_inventory_outputs,
)
from cement_channel.labels.geometry_aware_regression_labels import (  # noqa: E402
    build_geometry_aware_regression_label_report_from_arrays,
    format_geometry_aware_regression_markdown,
)


class GeometryRegressionContractQaCliError(RuntimeError):
    """Raised when geometry regression contract QA cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate contract inventory and invariants for geometry regression QA."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--stage9-report-json", default=None)
    parser.add_argument("--stage9-report-md", default=None)
    parser.add_argument("--stage10-report-json", default=None)
    parser.add_argument("--inventory-md", default=None)
    parser.add_argument("--inventory-json", default=None)
    parser.add_argument("--invariants-md", default=None)
    parser.add_argument("--invariants-json", default=None)
    parser.add_argument("--skip-stage9-report-refresh", action="store_true")
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
        stage9_json = _resolve_report_path(
            paths,
            args.stage9_report_json,
            "geometry_aware_regression_labels_report_v001.json",
        )
        stage9_md = _resolve_report_path(
            paths,
            args.stage9_report_md,
            "geometry_aware_regression_labels_report_v001.md",
        )
        stage10_json = _resolve_report_path(
            paths,
            args.stage10_report_json,
            "geometry_regression_audit_v001.json",
        )
        inventory_md = _resolve_report_path(
            paths,
            args.inventory_md,
            "geometry_regression_contract_inventory_v001.md",
        )
        inventory_json = _resolve_report_path(
            paths,
            args.inventory_json,
            "geometry_regression_contract_inventory_v001.json",
        )
        invariants_md = _resolve_report_path(
            paths,
            args.invariants_md,
            "geometry_regression_contract_invariants_v001.md",
        )
        invariants_json = _resolve_report_path(
            paths,
            args.invariants_json,
            "geometry_regression_contract_invariants_v001.json",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (stage9_json, "reports", "read/write"),
            (stage9_md, "reports", "read/write"),
            (stage10_json, "reports", "read"),
            (inventory_md, "reports", "write"),
            (inventory_json, "reports", "write"),
            (invariants_md, "reports", "write"),
            (invariants_json, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        label_arrays = _load_npz(labels_npz)
        feature_arrays = _load_npz(features_npz)
        previous_stage9 = _load_json(stage9_json)
        stage10 = _load_json(stage10_json)
        if not args.skip_stage9_report_refresh:
            _ensure_can_write(stage9_json, overwrite=args.overwrite)
            _ensure_can_write(stage9_md, overwrite=args.overwrite)
            refreshed = build_geometry_aware_regression_label_report_from_arrays(
                arrays=label_arrays,
                previous_report=previous_stage9,
                output_npz=labels_npz,
            )
            stage9_json.write_text(
                json.dumps(refreshed.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            stage9_md.write_text(
                format_geometry_aware_regression_markdown(refreshed),
                encoding="utf-8",
            )
            previous_stage9 = refreshed.to_dict()
        inputs = {
            "regression_labels_npz": str(labels_npz),
            "depth_level_features_npz": str(features_npz),
            "stage9_report_json": str(stage9_json),
            "stage10_report_json": str(stage10_json),
        }
        inventory = build_contract_inventory(
            label_arrays=label_arrays,
            feature_arrays=feature_arrays,
            stage9_report=previous_stage9,
            stage10_report=stage10,
            inputs=inputs,
        )
        invariants = build_contract_invariants(
            label_arrays=label_arrays,
            feature_arrays=feature_arrays,
            stage9_report=previous_stage9,
            stage10_report=stage10,
            inputs=inputs,
        )
        write_contract_inventory_outputs(
            inventory,
            output_md=inventory_md,
            output_json=inventory_json,
            overwrite=args.overwrite,
        )
        write_contract_invariant_outputs(
            invariants,
            output_md=invariants_md,
            output_json=invariants_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        GeometryRegressionContractQaCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Geometry regression contract QA "
        f"inventory_class={inventory.issue_classification}; "
        f"invariants_passed={invariants.passed}; "
        f"errors={len(invariants.errors)}."
    )
    print(f"Wrote inventory Markdown: {inventory_md}")
    print(f"Wrote inventory JSON: {inventory_json}")
    print(f"Wrote invariants Markdown: {invariants_md}")
    print(f"Wrote invariants JSON: {invariants_json}")
    return 0 if invariants.passed else 1


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise GeometryRegressionContractQaCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise GeometryRegressionContractQaCliError("data.reports is not configured.")


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
        raise GeometryRegressionContractQaCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GeometryRegressionContractQaCliError(
            f"Refusing to {action} geometry regression contract QA path outside data.{key}: {path}"
        ) from exc


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
