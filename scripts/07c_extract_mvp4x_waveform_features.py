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
from cement_channel.features.mvp4x_waveform_features import (  # noqa: E402
    extract_waveform_features_from_paths,
)


class WaveformFeatureCliError(RuntimeError):
    """Raised when MVP-4X waveform feature extraction cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MVP-4X controlled waveform features.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--mapping", default="configs/raw_variable_mapping.yaml")
    parser.add_argument(
        "--rapid-config",
        default="configs/mvp4x_rapid_exploratory.example.yaml",
    )
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        snapshot_npz = _resolve_data_path(
            paths,
            "interim",
            args.snapshot_npz,
            "mvp4x_research_snapshot_v001.npz",
        )
        output_npz = _resolve_data_path(
            paths,
            "features",
            args.output_npz,
            "mvp4x_waveform_features_v001.npz",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_waveform_feature_report_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_waveform_feature_report_v001.json",
        )
        for path, key, action in (
            (snapshot_npz, "interim", "read"),
            (output_npz, "features", "write"),
            (output_md, "reports", "write"),
            (output_json, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        if args.dry_run:
            print(f"Dry run: would extract MVP-4X waveform features from {snapshot_npz}.")
            return 0
        report = extract_waveform_features_from_paths(
            paths_config=args.paths_config,
            mapping_path=args.mapping,
            snapshot_npz=snapshot_npz,
            rapid_config_path=args.rapid_config,
            output_npz=output_npz,
            output_report_md=output_md,
            output_report_json=output_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        WaveformFeatureCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4X waveform features "
        f"samples={report.sample_count}; "
        f"receivers={report.receiver_count}; "
        f"sides={report.side_count}; "
        f"depth_features={report.depth_feature_count}; "
        f"chunks={report.chunk_count}; "
        f"peak_memory_bytes={report.peak_memory_bytes}; "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"research_only={report.research_only}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote NPZ: {output_npz}")
    print(f"Wrote report JSON: {output_json}")
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
    raise WaveformFeatureCliError(f"data.{key} is not configured.")


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
        raise WaveformFeatureCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise WaveformFeatureCliError(
            f"Refusing to {action} waveform feature path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())

