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
from cement_channel.modeling.mvp4x_research_model_artifact import (  # noqa: E402
    ResearchModelArtifactError,
    export_research_model_artifact_from_paths,
)


class ResearchArtifactCliError(RuntimeError):
    """Raised when the MVP-4X research model artifact CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the MVP-4X offline research screening model artifact."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--artifact-config",
        default="configs/mvp4x_screening_baseline.example.yaml",
    )
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--screening-policy-npz", default=None)
    parser.add_argument("--screening-policy-json", default=None)
    parser.add_argument("--scores-json", default=None)
    parser.add_argument("--robustness-json", default=None)
    parser.add_argument("--ranking-json", default=None)
    parser.add_argument("--error-json", default=None)
    parser.add_argument("--output-joblib", default=None)
    parser.add_argument("--output-manifest-json", default=None)
    parser.add_argument("--output-report-md", default=None)
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
        policy_npz = _resolve_data_path(
            paths,
            "interim",
            args.screening_policy_npz,
            "mvp4x_screening_policy_v001.npz",
        )
        policy_json = _resolve_data_path(
            paths,
            "reports",
            args.screening_policy_json,
            "mvp4x_screening_policy_v001.json",
        )
        scores_json = _resolve_data_path(
            paths,
            "reports",
            args.scores_json,
            "mvp4x_screening_scores_oof_v001.json",
        )
        robustness_json = _resolve_data_path(
            paths,
            "reports",
            args.robustness_json,
            "mvp4x_regime_robustness_v001.json",
        )
        ranking_json = _resolve_data_path(
            paths,
            "reports",
            args.ranking_json,
            "mvp4x_ranking_audit_v001.json",
        )
        error_json = _resolve_data_path(
            paths,
            "reports",
            args.error_json,
            "mvp4x_regime_error_analysis_v001.json",
        )
        output_joblib = _resolve_data_path(
            paths,
            "features",
            args.output_joblib,
            "mvp4x_research_screening_model_v001.joblib",
        )
        output_manifest = _resolve_data_path(
            paths,
            "manifests",
            args.output_manifest_json,
            "mvp4x_research_screening_model_v001.json",
        )
        output_report = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_research_screening_model_v001.md",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (policy_npz, "interim"),
            (policy_json, "reports"),
            (scores_json, "reports"),
            (robustness_json, "reports"),
            (ranking_json, "reports"),
            (error_json, "reports"),
            (output_joblib, "features"),
            (output_manifest, "manifests"),
            (output_report, "reports"),
        ):
            _ensure_path_within(paths, path, key=key)
        if args.dry_run:
            print(
                "Dry run: would export MVP-4X research model artifact "
                f"snapshot={snapshot_npz} policy={policy_npz}."
            )
            return 0
        manifest = export_research_model_artifact_from_paths(
            snapshot_npz=snapshot_npz,
            screening_policy_npz=policy_npz,
            screening_policy_json=policy_json,
            scores_json=scores_json,
            robustness_json=robustness_json,
            ranking_json=ranking_json,
            error_json=error_json,
            config_path=args.artifact_config,
            output_joblib=output_joblib,
            output_manifest_json=output_manifest,
            output_report_md=output_report,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ModelingDependencyError,
        ResearchModelArtifactError,
        ResearchArtifactCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X research model artifact "
        f"training_cohort={manifest['training_cohort']}; "
        f"not_validated_for_deployment={manifest['not_validated_for_deployment']}."
    )
    print(f"Wrote artifact: {manifest['artifact_path']}")
    print(f"Wrote manifest: {output_manifest}")
    return 0


def _resolve_data_path(
    config: dict[str, Any],
    key: str,
    override: str | None,
    filename: str,
) -> Path:
    if override:
        return Path(override)
    root = _as_dict(config.get("data")).get(key)
    if root:
        return Path(str(root)) / filename
    raise ResearchArtifactCliError(f"data.{key} is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, key: str) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key))).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ResearchArtifactCliError(f"{resolved} is outside configured data.{key}: {root}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
