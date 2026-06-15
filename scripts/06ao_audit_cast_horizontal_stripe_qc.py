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
from cement_channel.qc.cast_horizontal_stripe_qc import (  # noqa: E402
    audit_cast_horizontal_stripes_from_paths,
)
from cement_channel.visualization.matplotlib_utils import PlottingDependencyError  # noqa: E402


class CastHorizontalStripeQcCliError(RuntimeError):
    """Raised when CAST horizontal stripe QC cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit near-360-degree horizontal high-Zc stripes in raw CAST Zc."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--cast-npz", default=None)
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--parallel-candidates-npz", default=None)
    parser.add_argument("--raw-cast-mat", default=None)
    parser.add_argument("--manual-review-checklist", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        cast_npz = _resolve_interim_path(config, args.cast_npz, "cast_label_input_v001.npz")
        regression_labels = _resolve_interim_path(
            config,
            args.regression_labels_npz,
            "geometry_aware_regression_labels_v001.npz",
        )
        parallel_candidates = _resolve_interim_path(
            config,
            args.parallel_candidates_npz,
            "mvp4x_parallel_label_candidates_v001.npz",
        )
        reports_dir = _reports_dir(config)
        raw_cast_mat = (
            Path(args.raw_cast_mat)
            if args.raw_cast_mat
            else _data_root(config) / "raw" / "CAST.mat"
        )
        checklist = (
            Path(args.manual_review_checklist)
            if args.manual_review_checklist
            else reports_dir
            / "mvp4x_label_semantics_manual_review_v001"
            / "reviewer_checklist.md"
        )
        for path, key, action in (
            (cast_npz, "interim", "read"),
            (regression_labels, "interim", "read"),
            (parallel_candidates, "interim", "read"),
            (reports_dir, "reports", "write"),
        ):
            _ensure_path_within(config, path, key=key, action=action)
        result = audit_cast_horizontal_stripes_from_paths(
            cast_npz=cast_npz,
            regression_labels_npz=regression_labels,
            parallel_candidates_npz=parallel_candidates,
            raw_cast_mat=raw_cast_mat,
            reports_dir=reports_dir,
            manual_review_checklist=checklist,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        CastHorizontalStripeQcCliError,
        PlottingDependencyError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "CAST horizontal stripe QC completed "
        f"near360_events={result.near360_event_count}; "
        f"partial_events={result.partial_event_count}; "
        f"recommendations={','.join(result.recommendations)}; "
        f"warnings={len(result.warnings)}."
    )
    print(f"Wrote report: {reports_dir / 'cast_horizontal_stripe_qc_report_v001.md'}")
    return 1 if result.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    if data.get("interim"):
        return Path(str(data["interim"])) / filename
    raise CastHorizontalStripeQcCliError("data.interim is not configured.")


def _reports_dir(config: dict[str, Any]) -> Path:
    data = _as_dict(config.get("data"))
    if data.get("reports"):
        return Path(str(data["reports"]))
    raise CastHorizontalStripeQcCliError("data.reports is not configured.")


def _data_root(config: dict[str, Any]) -> Path:
    data = _as_dict(config.get("data"))
    if data.get("root"):
        return Path(str(data["root"]))
    reports = _reports_dir(config)
    return reports.parent


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
        raise CastHorizontalStripeQcCliError(f"data.{key} is not configured.")
    if path == root and action == "write":
        return
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise CastHorizontalStripeQcCliError(
            f"Refusing to {action} CAST horizontal stripe QC path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
