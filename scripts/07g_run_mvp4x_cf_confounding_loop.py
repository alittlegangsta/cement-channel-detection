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
from cement_channel.modeling.dependencies import ModelingDependencyError  # noqa: E402
from cement_channel.modeling.mvp4x_confounding import (  # noqa: E402
    ConfoundingResearchError,
    run_confounding_research_from_paths,
)


class ConfoundingLoopCliError(RuntimeError):
    """Raised when the MVP-4X CF loop cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVP-4X confounding disentanglement exploratory loop."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--rapid-config", default="configs/mvp4x_rapid_exploratory.example.yaml")
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--waveform-features-npz", default=None)
    parser.add_argument("--autonomous-decision-json", default=None)
    parser.add_argument("--autonomous-iteration-log-md", default=None)
    parser.add_argument("--output-common-md", default=None)
    parser.add_argument("--output-common-json", default=None)
    parser.add_argument("--output-common-csv", default=None)
    parser.add_argument("--output-nuisance-md", default=None)
    parser.add_argument("--output-nuisance-json", default=None)
    parser.add_argument("--output-nuisance-csv", default=None)
    parser.add_argument("--output-spatial-md", default=None)
    parser.add_argument("--output-spatial-json", default=None)
    parser.add_argument("--output-spatial-csv", default=None)
    parser.add_argument("--output-decision-md", default=None)
    parser.add_argument("--output-decision-json", default=None)
    parser.add_argument("--output-review-dir", default=None)
    parser.add_argument("--output-iteration-log", default=None)
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
        waveform_npz = _resolve_data_path(
            paths,
            "features",
            args.waveform_features_npz,
            "mvp4x_waveform_features_v001.npz",
        )
        autonomous_decision_json = _resolve_data_path(
            paths,
            "reports",
            args.autonomous_decision_json,
            "mvp4x_autonomous_decision.json",
        )
        autonomous_iteration_log = _resolve_data_path(
            paths,
            "reports",
            args.autonomous_iteration_log_md,
            "mvp4x_autonomous_iteration_log.md",
        )
        output_common_md = _resolve_data_path(
            paths,
            "reports",
            args.output_common_md,
            "mvp4x_cf_common_support_v001.md",
        )
        output_common_json = _resolve_data_path(
            paths,
            "reports",
            args.output_common_json,
            "mvp4x_cf_common_support_v001.json",
        )
        output_common_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_common_csv,
            "mvp4x_cf_common_support_v001.csv",
        )
        output_nuisance_md = _resolve_data_path(
            paths,
            "reports",
            args.output_nuisance_md,
            "mvp4x_cf_nuisance_audit_v001.md",
        )
        output_nuisance_json = _resolve_data_path(
            paths,
            "reports",
            args.output_nuisance_json,
            "mvp4x_cf_nuisance_audit_v001.json",
        )
        output_nuisance_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_nuisance_csv,
            "mvp4x_cf_nuisance_audit_v001.csv",
        )
        output_spatial_md = _resolve_data_path(
            paths,
            "reports",
            args.output_spatial_md,
            "mvp4x_cf_spatial_validation_v001.md",
        )
        output_spatial_json = _resolve_data_path(
            paths,
            "reports",
            args.output_spatial_json,
            "mvp4x_cf_spatial_validation_v001.json",
        )
        output_spatial_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_spatial_csv,
            "mvp4x_cf_spatial_validation_v001.csv",
        )
        output_decision_md = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_md,
            "mvp4x_cf_decision.md",
        )
        output_decision_json = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_json,
            "mvp4x_cf_decision.json",
        )
        output_review_dir = _resolve_data_path(
            paths,
            "reports",
            args.output_review_dir,
            "mvp4x_cf_review_v001",
        )
        output_iteration_log = _resolve_data_path(
            paths,
            "reports",
            args.output_iteration_log,
            "mvp4x_cf_iteration_log.md",
        )
        read_paths = (
            (snapshot_npz, "interim"),
            (waveform_npz, "features"),
            (autonomous_decision_json, "reports"),
            (autonomous_iteration_log, "reports"),
        )
        write_paths = (
            output_common_md,
            output_common_json,
            output_common_csv,
            output_nuisance_md,
            output_nuisance_json,
            output_nuisance_csv,
            output_spatial_md,
            output_spatial_json,
            output_spatial_csv,
            output_decision_md,
            output_decision_json,
            output_review_dir,
            output_iteration_log,
        )
        for path, key in read_paths:
            _ensure_path_within(paths, path, key=key, action="read")
        for path in write_paths:
            _ensure_path_within(paths, path, key="reports", action="write")
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X CF loop "
                f"snapshot={snapshot_npz} waveform={waveform_npz}."
            )
            return 0
        outputs = run_confounding_research_from_paths(
            snapshot_npz=snapshot_npz,
            waveform_features_npz=waveform_npz,
            autonomous_decision_json=autonomous_decision_json,
            autonomous_iteration_log_md=autonomous_iteration_log,
            config_path=args.rapid_config,
            output_common_md=output_common_md,
            output_common_json=output_common_json,
            output_common_csv=output_common_csv,
            output_nuisance_md=output_nuisance_md,
            output_nuisance_json=output_nuisance_json,
            output_nuisance_csv=output_nuisance_csv,
            output_spatial_md=output_spatial_md,
            output_spatial_json=output_spatial_json,
            output_spatial_csv=output_spatial_csv,
            output_decision_md=output_decision_md,
            output_decision_json=output_decision_json,
            output_review_dir=output_review_dir,
            output_iteration_log=output_iteration_log,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ModelingDependencyError,
        ConfoundingResearchError,
        ConfoundingLoopCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    best = outputs.spatial_validation.get("best_candidate") or {}
    print(
        "MVP-4X CF loop "
        f"decision={outputs.decision['decision']}; "
        f"best_model={best.get('model')}; "
        f"best_feature_set={best.get('feature_set')}; "
        f"best_target={best.get('target')}; "
        f"research_only={outputs.decision['research_only']}; "
        f"no_final_labels={outputs.decision['no_final_labels']}."
    )
    print(f"Wrote decision JSON: {output_decision_json}")
    print(f"Wrote review dir: {output_review_dir}")
    return 0


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
    raise ConfoundingLoopCliError(f"data.{key} is not configured.")


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
        raise ConfoundingLoopCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ConfoundingLoopCliError(
            f"Refusing to {action} CF-loop path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
