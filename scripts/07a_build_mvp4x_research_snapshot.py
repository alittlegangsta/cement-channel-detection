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
from cement_channel.experiments.research_snapshot import (  # noqa: E402
    build_research_snapshot_from_paths,
)


class ResearchSnapshotCliError(RuntimeError):
    """Raised when the MVP-4X research snapshot cannot be built safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP-4X rapid research snapshot.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--rapid-config",
        default="configs/mvp4x_rapid_exploratory.example.yaml",
    )
    parser.add_argument("--regression-labels-npz", default=None)
    parser.add_argument("--depth-level-features-npz", default=None)
    parser.add_argument("--cast-label-input-npz", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-manifest-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        labels_npz = _resolve_data_path(
            paths,
            "interim",
            args.regression_labels_npz,
            "geometry_aware_regression_labels_v001.npz",
        )
        features_npz = _resolve_data_path(
            paths,
            "interim",
            args.depth_level_features_npz,
            "depth_level_xsi_features_v001.npz",
        )
        cast_npz = _resolve_data_path(
            paths,
            "interim",
            args.cast_label_input_npz,
            "cast_label_input_v001.npz",
        )
        output_npz = _resolve_data_path(
            paths,
            "interim",
            args.output_npz,
            "mvp4x_research_snapshot_v001.npz",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_research_snapshot_report_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_research_snapshot_report_v001.json",
        )
        output_manifest = _resolve_data_path(
            paths,
            "manifests",
            args.output_manifest_json,
            "mvp4x_research_snapshot_v001.json",
        )
        for path, key, action in (
            (labels_npz, "interim", "read"),
            (features_npz, "interim", "read"),
            (cast_npz, "interim", "read"),
            (output_npz, "interim", "write"),
            (output_md, "reports", "write"),
            (output_json, "reports", "write"),
            (output_manifest, "manifests", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        if args.dry_run:
            print(
                "Dry run: would build MVP-4X research snapshot "
                f"from {labels_npz}, {features_npz}, {cast_npz}."
            )
            return 0
        report = build_research_snapshot_from_paths(
            regression_labels_npz=labels_npz,
            depth_level_features_npz=features_npz,
            cast_label_input_npz=cast_npz,
            config_path=args.rapid_config,
            output_npz=output_npz,
            output_report_md=output_md,
            output_report_json=output_json,
            output_manifest_json=output_manifest,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ResearchSnapshotCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4X research snapshot "
        f"samples={report.sample_count}; "
        f"features={report.feature_count}; "
        f"target_kernel={report.target_kernel}; "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"research_only={report.research_only}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote NPZ: {output_npz}")
    print(f"Wrote report JSON: {output_json}")
    print(f"Wrote manifest JSON: {output_manifest}")
    return 1 if report.errors else 0


def _resolve_data_path(
    config: dict[str, Any],
    key: str,
    override: str | None,
    filename: str,
) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    root = data.get(key)
    if root:
        return Path(str(root)) / filename
    raise ResearchSnapshotCliError(f"data.{key} is not configured.")


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
        raise ResearchSnapshotCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ResearchSnapshotCliError(
            f"Refusing to {action} research snapshot path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
