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
from cement_channel.experiments.mvp4x_screening_policy import (  # noqa: E402
    ScreeningPolicyError,
    build_screening_policy_from_paths,
)


class ScreeningPolicyCliError(RuntimeError):
    """Raised when the MVP-4X screening policy CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the MVP-4X research-only screening baseline policy."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--policy-config",
        default="configs/mvp4x_screening_baseline.example.yaml",
    )
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--waveform-features-npz", default=None)
    parser.add_argument("--regime-policy-json", default=None)
    parser.add_argument("--robustness-json", default=None)
    parser.add_argument("--ranking-json", default=None)
    parser.add_argument("--error-json", default=None)
    parser.add_argument("--screening-json", default=None)
    parser.add_argument("--regime-decision-json", default=None)
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
        waveform_npz = _resolve_data_path(
            paths,
            "features",
            args.waveform_features_npz,
            "mvp4x_waveform_features_v001.npz",
        )
        regime_policy_json = _resolve_data_path(
            paths,
            "reports",
            args.regime_policy_json,
            "mvp4x_regime_policy_v001.json",
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
        screening_json = _resolve_data_path(
            paths,
            "reports",
            args.screening_json,
            "mvp4x_screening_baseline_v001.json",
        )
        regime_decision_json = _resolve_data_path(
            paths,
            "reports",
            args.regime_decision_json,
            "mvp4x_regime_policy_decision.json",
        )
        output_npz = _resolve_data_path(
            paths,
            "interim",
            args.output_npz,
            "mvp4x_screening_policy_v001.npz",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_screening_policy_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_screening_policy_v001.json",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (waveform_npz, "features"),
            (regime_policy_json, "reports"),
            (robustness_json, "reports"),
            (ranking_json, "reports"),
            (error_json, "reports"),
            (screening_json, "reports"),
            (regime_decision_json, "reports"),
            (output_npz, "interim"),
            (output_md, "reports"),
            (output_json, "reports"),
        ):
            _ensure_path_within(paths, path, key=key)
        if args.dry_run:
            print(
                "Dry run: would build MVP-4X screening policy "
                f"snapshot={snapshot_npz} screening={screening_json}."
            )
            return 0
        outputs = build_screening_policy_from_paths(
            snapshot_npz=snapshot_npz,
            waveform_features_npz=waveform_npz,
            regime_policy_json=regime_policy_json,
            robustness_json=robustness_json,
            ranking_json=ranking_json,
            error_json=error_json,
            screening_json=screening_json,
            regime_decision_json=regime_decision_json,
            config_path=args.policy_config,
            output_npz=output_npz,
            output_report_md=output_md,
            output_report_json=output_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ScreeningPolicyError,
        ScreeningPolicyCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X screening policy "
        f"cohort_count={len(outputs.cohort_names)}; "
        f"research_only={outputs.policy['research_only']}; "
        f"not_validated_for_deployment={outputs.policy['not_validated_for_deployment']}."
    )
    print(f"Wrote policy NPZ: {output_npz}")
    print(f"Wrote policy JSON: {output_json}")
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
    raise ScreeningPolicyCliError(f"data.{key} is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, key: str) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key))).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ScreeningPolicyCliError(f"{resolved} is outside configured data.{key}: {root}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
