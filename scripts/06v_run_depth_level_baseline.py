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
from cement_channel.training.depth_level_baseline import (  # noqa: E402
    run_depth_level_baseline,
    run_depth_level_baseline_from_config,
    write_depth_level_baseline_outputs,
)
from cement_channel.training.depth_level_baseline_schema import (  # noqa: E402
    load_depth_level_baseline_config,
)


class DepthLevelBaselineCliError(RuntimeError):
    """Raised when depth-level baseline sanity cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP-4B-R4b depth-level baseline sanity.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--baseline-config",
        default="configs/depth_level_baseline.example.yaml",
    )
    parser.add_argument("--depth-level-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
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
            args.depth_level_labels_npz,
            "depth_level_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            paths,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "depth_level_baseline_report_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "depth_level_baseline_report_v001.json",
        )
        output_csv = _resolve_report_path(
            paths,
            args.output_csv,
            "depth_level_baseline_report_v001.csv",
        )
        _ensure_path_within(paths, labels_npz, key="interim", action="read")
        _ensure_path_within(paths, features_npz, key="interim", action="read")
        _ensure_path_within(paths, output_md, key="reports", action="write")
        _ensure_path_within(paths, output_json, key="reports", action="write")
        _ensure_path_within(paths, output_csv, key="reports", action="write")
        if args.dry_run:
            import numpy as np  # noqa: PLC0415

            with np.load(labels_npz, allow_pickle=False) as label_data:
                label_arrays = {key: label_data[key] for key in label_data.files}
            with np.load(features_npz, allow_pickle=False) as feature_data:
                feature_arrays = {key: feature_data[key] for key in feature_data.files}
            report, rows = run_depth_level_baseline(
                label_arrays=label_arrays,
                feature_arrays=feature_arrays,
                config=load_depth_level_baseline_config(args.baseline_config),
                inputs={
                    "depth_level_labels_npz": str(labels_npz),
                    "depth_level_features_npz": str(features_npz),
                    "baseline_config_path": str(args.baseline_config),
                },
            )
        else:
            report, rows = run_depth_level_baseline_from_config(
                depth_level_labels_npz=labels_npz,
                depth_level_features_npz=features_npz,
                baseline_config_path=args.baseline_config,
            )
            write_depth_level_baseline_outputs(
                report,
                rows,
                output_report_md=output_md,
                output_report_json=output_json,
                output_csv=output_csv,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthLevelBaselineCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    best = report.best_result or {}
    print(
        "Depth-level baseline sanity "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"usable_variants={report.usable_target_variants}; "
        f"best_variant={best.get('target_variant')}; "
        f"best_model={best.get('model_type')}; "
        f"best_margin={best.get('balanced_accuracy_margin')}; "
        f"no_final_labels={report.no_final_labels}."
    )
    if args.dry_run:
        print("Dry run: no reports or CSV written.")
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
    raise DepthLevelBaselineCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelBaselineCliError("data.reports is not configured.")


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
        raise DepthLevelBaselineCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelBaselineCliError(
            f"Refusing to {action} depth-level baseline path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
