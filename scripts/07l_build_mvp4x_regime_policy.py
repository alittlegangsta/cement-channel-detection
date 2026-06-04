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
from cement_channel.experiments.mvp4x_regime_policy import (  # noqa: E402
    RegimePolicyError,
    build_regime_policy_from_paths,
)


class RegimePolicyCliError(RuntimeError):
    """Raised when the MVP-4X regime policy CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the MVP-4X research-only regime policy."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--policy-config", default="configs/mvp4x_regime_policy.example.yaml")
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--waveform-features-npz", default=None)
    parser.add_argument("--cf-decision-json", default=None)
    parser.add_argument("--cf-spatial-json", default=None)
    parser.add_argument("--stratified-decision-json", default=None)
    parser.add_argument("--stratified-iteration-log", default=None)
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
        cf_decision = _resolve_data_path(
            paths,
            "reports",
            args.cf_decision_json,
            "mvp4x_cf_decision.json",
        )
        cf_spatial = _resolve_data_path(
            paths,
            "reports",
            args.cf_spatial_json,
            "mvp4x_cf_spatial_validation_v001.json",
        )
        strat_decision = _resolve_data_path(
            paths,
            "reports",
            args.stratified_decision_json,
            "mvp4x_stratified_decision.json",
        )
        strat_log = _resolve_data_path(
            paths,
            "reports",
            args.stratified_iteration_log,
            "mvp4x_stratified_iteration_log.md",
        )
        output_npz = _resolve_data_path(
            paths,
            "interim",
            args.output_npz,
            "mvp4x_regime_policy_v001.npz",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_regime_policy_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_regime_policy_v001.json",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (waveform_npz, "features"),
            (cf_decision, "reports"),
            (cf_spatial, "reports"),
            (strat_decision, "reports"),
            (strat_log, "reports"),
            (output_npz, "interim"),
            (output_md, "reports"),
            (output_json, "reports"),
        ):
            _ensure_path_within(paths, path, key=key)
        if args.dry_run:
            print(
                "Dry run: would build MVP-4X regime policy "
                f"snapshot={snapshot_npz} stratified_decision={strat_decision}."
            )
            return 0
        outputs = build_regime_policy_from_paths(
            snapshot_npz=snapshot_npz,
            waveform_features_npz=waveform_npz,
            cf_decision_json=cf_decision,
            cf_spatial_json=cf_spatial,
            stratified_decision_json=strat_decision,
            stratified_iteration_log=strat_log,
            config_path=args.policy_config,
            output_npz=output_npz,
            output_report_md=output_md,
            output_report_json=output_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        RegimePolicyError,
        RegimePolicyCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X regime policy "
        f"cohort_count={len(outputs.cohort_names)}; "
        f"research_only={outputs.policy['research_only']}; "
        f"no_final_labels={outputs.policy['no_final_labels']}."
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
    raise RegimePolicyCliError(f"data.{key} is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, key: str) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key))).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise RegimePolicyCliError(f"{resolved} is outside configured data.{key}: {root}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
