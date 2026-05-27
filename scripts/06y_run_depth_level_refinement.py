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
from cement_channel.training.depth_level_refinement import (  # noqa: E402
    run_depth_level_refinement_from_config,
)


class DepthLevelRefinementCliError(RuntimeError):
    """Raised when controlled depth-level refinement cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVP-4B-R4c controlled depth-level refinement ablations."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--refinement-config",
        default="configs/depth_level_refinement.example.yaml",
    )
    parser.add_argument("--depth-level-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--baseline-report", default=None)
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
        labels_npz = _resolve_interim_path(
            config,
            args.depth_level_labels_npz,
            "depth_level_labels_v001.npz",
        )
        features_npz = _resolve_interim_path(
            config,
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        baseline_report = _resolve_report_path(
            config,
            args.baseline_report,
            "depth_level_baseline_report_v001.json",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "depth_level_refinement_report_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "depth_level_refinement_report_v001.json",
        )
        output_csv = _resolve_report_path(
            config,
            args.output_csv,
            "depth_level_refinement_report_v001.csv",
        )
        _ensure_path_within(config, labels_npz, key="interim", action="read")
        _ensure_path_within(config, features_npz, key="interim", action="read")
        _ensure_path_within(config, baseline_report, key="reports", action="read")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        _ensure_path_within(config, output_csv, key="reports", action="write")
        report, rows = run_depth_level_refinement_from_config(
            depth_level_labels_npz=labels_npz,
            depth_level_features_npz=features_npz,
            baseline_report_json=baseline_report,
            refinement_config_path=args.refinement_config,
            output_report_md=None if args.dry_run else output_md,
            output_report_json=None if args.dry_run else output_json,
            output_csv=None if args.dry_run else output_csv,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        DepthLevelRefinementCliError,
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
        "Depth-level refinement "
        f"recommendation={report.recommendation}; "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"passing_scenarios={report.passing_scenario_count}; "
        f"best_feature_group={report.best_feature_group}; "
        f"best_confidence_threshold={best.get('confidence_threshold')}; "
        f"best_margin_mean={best.get('margin_mean')}; "
        f"manual_confirmation_required={report.manual_confirmation_required}; "
        f"prediction_rows={len(rows)}."
    )
    if args.dry_run:
        print("Dry run: no reports or CSV written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote CSV report: {output_csv}")
    return 1 if report.recommendation == "no_go" else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise DepthLevelRefinementCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelRefinementCliError("data.reports is not configured.")


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
        raise DepthLevelRefinementCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelRefinementCliError(
            f"Refusing to {action} depth-level refinement path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
