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
from cement_channel.modeling.mvp4x_baselines import (  # noqa: E402
    run_existing_feature_baselines_from_paths,
)


class ExistingFeatureBaselineCliError(RuntimeError):
    """Raised when MVP-4X existing-feature baselines cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP-4X existing-feature baselines.")
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
    parser.add_argument("--snapshot-npz", default=None)
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
        snapshot_npz = _resolve_data_path(
            paths,
            "interim",
            args.snapshot_npz,
            "mvp4x_research_snapshot_v001.npz",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_existing_feature_baselines_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_existing_feature_baselines_v001.json",
        )
        output_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_csv,
            "mvp4x_existing_feature_baselines_v001.csv",
        )
        for path, key, action in (
            (snapshot_npz, "interim", "read"),
            (output_md, "reports", "write"),
            (output_json, "reports", "write"),
            (output_csv, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        if args.dry_run:
            print(f"Dry run: would run MVP-4X existing-feature baselines from {snapshot_npz}.")
            return 0
        report = run_existing_feature_baselines_from_paths(
            snapshot_npz=snapshot_npz,
            config_path=args.rapid_config,
            output_report_md=output_md,
            output_report_json=output_json,
            output_csv=output_csv,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ExistingFeatureBaselineCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4X existing-feature baselines "
        f"backend={report.model_backend}; "
        f"sklearn_available={report.sklearn_available}; "
        f"baseline_sufficient={report.decision.get('baseline_sufficient')}; "
        f"stage3_recommended={report.decision.get('stage3_recommended')}; "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"research_only={report.research_only}; "
        f"no_final_labels={report.no_final_labels}."
    )
    print(f"Wrote report JSON: {output_json}")
    print(f"Wrote CSV: {output_csv}")
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
    raise ExistingFeatureBaselineCliError(f"data.{key} is not configured.")


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
        raise ExistingFeatureBaselineCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ExistingFeatureBaselineCliError(
            f"Refusing to {action} existing-feature baseline path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())

