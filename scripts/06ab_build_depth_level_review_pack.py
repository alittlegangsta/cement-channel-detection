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
from cement_channel.evaluation.depth_level_review_pack import (  # noqa: E402
    build_depth_level_review_pack_from_config,
)


class DepthLevelReviewPackCliError(RuntimeError):
    """Raised when the depth-level manual review pack cannot be built safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MVP-4B-R4c+ depth-level manual review interval pack."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--review-config",
        default="configs/depth_level_manual_review.example.yaml",
    )
    parser.add_argument("--depth-level-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--refinement-report", default=None)
    parser.add_argument("--refinement-gate-report", default=None)
    parser.add_argument("--refinement-csv", default=None)
    parser.add_argument("--cast-weak-label-candidates", default=None)
    parser.add_argument("--cast-label-input", default=None)
    parser.add_argument("--output-dir", default=None)
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
        refinement_report = _resolve_report_path(
            config,
            args.refinement_report,
            "depth_level_refinement_report_v001.json",
        )
        refinement_gate = _resolve_report_path(
            config,
            args.refinement_gate_report,
            "depth_level_refinement_gate_report.json",
        )
        refinement_csv = _resolve_report_path(
            config,
            args.refinement_csv,
            "depth_level_refinement_report_v001.csv",
        )
        cast_candidates = _resolve_optional_root_path(
            config,
            args.cast_weak_label_candidates,
            "labels/cast_weak_label_candidates_v001.npz",
        )
        cast_label_input = _resolve_optional_interim_path(
            config,
            args.cast_label_input,
            "cast_label_input_v001.npz",
        )
        output_dir = _resolve_review_dir(config, args.output_dir)
        _ensure_path_within(config, labels_npz, key="interim", action="read")
        _ensure_path_within(config, features_npz, key="interim", action="read")
        _ensure_path_within(config, refinement_report, key="reports", action="read")
        _ensure_path_within(config, refinement_gate, key="reports", action="read")
        _ensure_path_within(config, refinement_csv, key="reports", action="read")
        _ensure_path_within(config, output_dir, key="reports", action="write")
        _ensure_optional_path_within(config, cast_candidates, key="root", action="read")
        _ensure_optional_path_within(config, cast_label_input, key="interim", action="read")
        report, intervals = build_depth_level_review_pack_from_config(
            depth_level_labels_npz=labels_npz,
            depth_level_features_npz=features_npz,
            refinement_report_json=refinement_report,
            refinement_gate_report_json=refinement_gate,
            refinement_csv=refinement_csv,
            review_config_path=args.review_config,
            cast_weak_label_candidates_npz=cast_candidates,
            cast_label_input_npz=cast_label_input,
            output_dir=None if args.dry_run else output_dir,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        DepthLevelReviewPackCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth-level manual review pack "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"intervals={len(intervals)}; "
        f"source_gate_decision={report.source_gate_decision}; "
        f"no_final_labels={report.no_final_labels}."
    )
    if args.dry_run:
        print(f"Dry run: review pack would be written under {output_dir}.")
    else:
        print(f"Wrote review pack directory: {output_dir}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise DepthLevelReviewPackCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelReviewPackCliError("data.reports is not configured.")


def _resolve_optional_interim_path(
    config: dict[str, Any],
    override: str | None,
    filename: str,
) -> Path | None:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if not interim:
        return None
    return Path(str(interim)) / filename


def _resolve_optional_root_path(
    config: dict[str, Any],
    override: str | None,
    relative_path: str,
) -> Path | None:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    root = data.get("root")
    if not root:
        return None
    return Path(str(root)) / relative_path


def _resolve_review_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / "depth_level_manual_review_v001"
    raise DepthLevelReviewPackCliError("data.reports is not configured.")


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
        raise DepthLevelReviewPackCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelReviewPackCliError(
            f"Refusing to {action} depth-level review pack path outside data.{key}: {path}"
        ) from exc


def _ensure_optional_path_within(
    config: dict[str, Any],
    path: Path | None,
    *,
    key: str,
    action: str,
) -> None:
    if path is None:
        return
    _ensure_path_within(config, path, key=key, action=action)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
