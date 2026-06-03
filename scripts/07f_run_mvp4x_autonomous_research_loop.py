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
from cement_channel.modeling.mvp4x_autonomous import (  # noqa: E402
    AutonomousResearchError,
    run_autonomous_research_from_paths,
)


class AutonomousLoopCliError(RuntimeError):
    """Raised when the MVP-4X autonomous loop cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded MVP-4X autonomous exploratory research loop."
    )
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
    parser.add_argument("--waveform-features-npz", default=None)
    parser.add_argument("--existing-json", default=None)
    parser.add_argument("--enhanced-json", default=None)
    parser.add_argument("--rapid-decision-json", default=None)
    parser.add_argument("--output-decision-md", default=None)
    parser.add_argument("--output-decision-json", default=None)
    parser.add_argument("--output-review-dir", default=None)
    parser.add_argument("--output-iteration-log", default=None)
    parser.add_argument("--output-subgroup-json", default=None)
    parser.add_argument("--output-subgroup-md", default=None)
    parser.add_argument("--output-subgroup-csv", default=None)
    parser.add_argument("--output-matched-json", default=None)
    parser.add_argument("--output-matched-md", default=None)
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
        existing_json = _resolve_data_path(
            paths,
            "reports",
            args.existing_json,
            "mvp4x_existing_feature_baselines_v001.json",
        )
        enhanced_json = _resolve_data_path(
            paths,
            "reports",
            args.enhanced_json,
            "mvp4x_enhanced_feature_baselines_v001.json",
        )
        rapid_json = _resolve_data_path(
            paths,
            "reports",
            args.rapid_decision_json,
            "mvp4x_rapid_decision.json",
        )
        output_decision_md = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_md,
            "mvp4x_autonomous_decision.md",
        )
        output_decision_json = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_json,
            "mvp4x_autonomous_decision.json",
        )
        output_review_dir = _resolve_data_path(
            paths,
            "reports",
            args.output_review_dir,
            "mvp4x_autonomous_review_v001",
        )
        output_iteration_log = _resolve_data_path(
            paths,
            "reports",
            args.output_iteration_log,
            "mvp4x_autonomous_iteration_log.md",
        )
        output_subgroup_json = _resolve_data_path(
            paths,
            "reports",
            args.output_subgroup_json,
            "mvp4x_autonomous_subgroup_audit_v001.json",
        )
        output_subgroup_md = _resolve_data_path(
            paths,
            "reports",
            args.output_subgroup_md,
            "mvp4x_autonomous_subgroup_audit_v001.md",
        )
        output_subgroup_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_subgroup_csv,
            "mvp4x_autonomous_subgroup_audit_v001.csv",
        )
        output_matched_json = _resolve_data_path(
            paths,
            "reports",
            args.output_matched_json,
            "mvp4x_autonomous_matched_control_v001.json",
        )
        output_matched_md = _resolve_data_path(
            paths,
            "reports",
            args.output_matched_md,
            "mvp4x_autonomous_matched_control_v001.md",
        )
        for path, key, action in (
            (snapshot_npz, "interim", "read"),
            (waveform_npz, "features", "read"),
            (existing_json, "reports", "read"),
            (enhanced_json, "reports", "read"),
            (rapid_json, "reports", "read"),
            (output_decision_md, "reports", "write"),
            (output_decision_json, "reports", "write"),
            (output_review_dir, "reports", "write"),
            (output_iteration_log, "reports", "write"),
            (output_subgroup_json, "reports", "write"),
            (output_subgroup_md, "reports", "write"),
            (output_subgroup_csv, "reports", "write"),
            (output_matched_json, "reports", "write"),
            (output_matched_md, "reports", "write"),
        ):
            _ensure_path_within(paths, path, key=key, action=action)
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X autonomous loop "
                f"snapshot={snapshot_npz} waveform={waveform_npz}."
            )
            return 0
        outputs = run_autonomous_research_from_paths(
            snapshot_npz=snapshot_npz,
            waveform_features_npz=waveform_npz,
            existing_json=existing_json,
            enhanced_json=enhanced_json,
            rapid_decision_json=rapid_json,
            config_path=args.rapid_config,
            output_decision_md=output_decision_md,
            output_decision_json=output_decision_json,
            output_review_dir=output_review_dir,
            output_iteration_log=output_iteration_log,
            output_subgroup_json=output_subgroup_json,
            output_subgroup_md=output_subgroup_md,
            output_subgroup_csv=output_subgroup_csv,
            output_matched_json=output_matched_json,
            output_matched_md=output_matched_md,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ModelingDependencyError,
        AutonomousResearchError,
        AutonomousLoopCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    best = outputs.decision.get("best_high_orientation_result") or {}
    print(
        "MVP-4X autonomous loop "
        f"decision={outputs.decision['decision']}; "
        f"best_model={best.get('model')}; "
        f"best_feature_set={best.get('feature_set')}; "
        f"best_target={best.get('target')}; "
        f"research_only={outputs.decision['research_only']}; "
        f"no_final_labels={outputs.decision['no_final_labels']}."
    )
    print(f"Wrote decision JSON: {output_decision_json}")
    print(f"Wrote iteration log: {output_iteration_log}")
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
    raise AutonomousLoopCliError(f"data.{key} is not configured.")


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
        raise AutonomousLoopCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise AutonomousLoopCliError(
            f"Refusing to {action} autonomous-loop path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
