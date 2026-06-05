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
from cement_channel.visualization.mvp4x_screening_review import (  # noqa: E402
    ScreeningReviewError,
    generate_screening_review_from_paths,
)


class ScreeningReviewCliError(RuntimeError):
    """Raised when the MVP-4X screening review CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-4X screening consolidation review and decision."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--review-config", default="configs/mvp4x_screening_baseline.example.yaml")
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--scores-npz", default=None)
    parser.add_argument("--scores-csv", default=None)
    parser.add_argument("--scores-json", default=None)
    parser.add_argument("--screening-policy-json", default=None)
    parser.add_argument("--robustness-json", default=None)
    parser.add_argument("--ranking-json", default=None)
    parser.add_argument("--error-json", default=None)
    parser.add_argument("--model-manifest-json", default=None)
    parser.add_argument("--regime-decision-json", default=None)
    parser.add_argument("--output-review-dir", default=None)
    parser.add_argument("--output-policy-comparison-json", default=None)
    parser.add_argument("--output-policy-comparison-csv", default=None)
    parser.add_argument("--output-policy-comparison-md", default=None)
    parser.add_argument("--output-proposal-md", default=None)
    parser.add_argument("--output-proposal-json", default=None)
    parser.add_argument("--output-decision-md", default=None)
    parser.add_argument("--output-decision-json", default=None)
    parser.add_argument("--output-iteration-log", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        snapshot_npz = _resolve_data_path(
            paths, "interim", args.snapshot_npz, "mvp4x_research_snapshot_v001.npz"
        )
        scores_npz = _resolve_data_path(
            paths, "interim", args.scores_npz, "mvp4x_screening_scores_oof_v001.npz"
        )
        scores_csv = _resolve_data_path(
            paths, "reports", args.scores_csv, "mvp4x_screening_scores_oof_v001.csv"
        )
        scores_json = _resolve_data_path(
            paths, "reports", args.scores_json, "mvp4x_screening_scores_oof_v001.json"
        )
        policy_json = _resolve_data_path(
            paths, "reports", args.screening_policy_json, "mvp4x_screening_policy_v001.json"
        )
        robustness_json = _resolve_data_path(
            paths, "reports", args.robustness_json, "mvp4x_regime_robustness_v001.json"
        )
        ranking_json = _resolve_data_path(
            paths, "reports", args.ranking_json, "mvp4x_ranking_audit_v001.json"
        )
        error_json = _resolve_data_path(
            paths, "reports", args.error_json, "mvp4x_regime_error_analysis_v001.json"
        )
        model_manifest = _resolve_data_path(
            paths,
            "manifests",
            args.model_manifest_json,
            "mvp4x_research_screening_model_v001.json",
        )
        regime_decision = _resolve_data_path(
            paths, "reports", args.regime_decision_json, "mvp4x_regime_policy_decision.json"
        )
        review_dir = _resolve_data_path(
            paths,
            "reports",
            args.output_review_dir,
            "mvp4x_screening_consolidation_review_v001",
        )
        comparison_json = _resolve_data_path(
            paths,
            "reports",
            args.output_policy_comparison_json,
            "mvp4x_screening_policy_comparison_v001.json",
        )
        comparison_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_policy_comparison_csv,
            "mvp4x_screening_policy_comparison_v001.csv",
        )
        comparison_md = _resolve_data_path(
            paths,
            "reports",
            args.output_policy_comparison_md,
            "mvp4x_screening_policy_comparison_v001.md",
        )
        proposal_md = _resolve_data_path(
            paths,
            "reports",
            args.output_proposal_md,
            "mvp4x_formal_regime_policy_proposal.md",
        )
        proposal_json = _resolve_data_path(
            paths,
            "reports",
            args.output_proposal_json,
            "mvp4x_formal_regime_policy_proposal.json",
        )
        decision_md = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_md,
            "mvp4x_screening_consolidation_decision.md",
        )
        decision_json = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_json,
            "mvp4x_screening_consolidation_decision.json",
        )
        iteration_log = _resolve_data_path(
            paths,
            "reports",
            args.output_iteration_log,
            "mvp4x_screening_consolidation_iteration_log.md",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (scores_npz, "interim"),
            (scores_csv, "reports"),
            (scores_json, "reports"),
            (policy_json, "reports"),
            (robustness_json, "reports"),
            (ranking_json, "reports"),
            (error_json, "reports"),
            (model_manifest, "manifests"),
            (regime_decision, "reports"),
            (review_dir, "reports"),
            (comparison_json, "reports"),
            (comparison_csv, "reports"),
            (comparison_md, "reports"),
            (proposal_md, "reports"),
            (proposal_json, "reports"),
            (decision_md, "reports"),
            (decision_json, "reports"),
            (iteration_log, "reports"),
        ):
            _ensure_path_within(paths, path, key=key)
        if args.dry_run:
            print("Dry run: would generate MVP-4X screening consolidation review.")
            return 0
        outputs = generate_screening_review_from_paths(
            snapshot_npz=snapshot_npz,
            scores_npz=scores_npz,
            scores_csv=scores_csv,
            scores_json=scores_json,
            screening_policy_json=policy_json,
            robustness_json=robustness_json,
            ranking_json=ranking_json,
            error_json=error_json,
            model_manifest_json=model_manifest,
            regime_decision_json=regime_decision,
            config_path=args.review_config,
            output_review_dir=review_dir,
            output_policy_comparison_json=comparison_json,
            output_policy_comparison_csv=comparison_csv,
            output_policy_comparison_md=comparison_md,
            output_proposal_md=proposal_md,
            output_proposal_json=proposal_json,
            output_decision_md=decision_md,
            output_decision_json=decision_json,
            output_iteration_log=iteration_log,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ScreeningReviewError,
        ScreeningReviewCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X screening consolidation decision "
        f"decision={outputs['decision']['decision']}; "
        f"not_validated_for_deployment={outputs['decision']['not_validated_for_deployment']}."
    )
    print(f"Wrote review dir: {review_dir}")
    print(f"Wrote decision JSON: {decision_json}")
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
    raise ScreeningReviewCliError(f"data.{key} is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, key: str) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key))).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ScreeningReviewCliError(f"{resolved} is outside configured data.{key}: {root}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
