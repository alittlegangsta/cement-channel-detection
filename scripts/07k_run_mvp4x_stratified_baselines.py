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
from cement_channel.modeling.mvp4x_stratified_baselines import (  # noqa: E402
    StratifiedBaselineError,
    run_stratified_baselines_from_paths,
)


class StratifiedBaselineCliError(RuntimeError):
    """Raised when the MVP-4X stratified baseline CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVP-4X research-only stratified classical baselines."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--stratified-config",
        default="configs/mvp4x_stratified_exploratory.example.yaml",
    )
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--waveform-features-npz", default=None)
    parser.add_argument("--design-npz", default=None)
    parser.add_argument("--cf-decision-json", default=None)
    parser.add_argument("--cf-common-support-json", default=None)
    parser.add_argument("--cf-nuisance-json", default=None)
    parser.add_argument("--cf-spatial-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-csv", default=None)
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
        design_npz = _resolve_data_path(
            paths,
            "interim",
            args.design_npz,
            "mvp4x_stratified_design_v001.npz",
        )
        cf_decision = _resolve_data_path(
            paths,
            "reports",
            args.cf_decision_json,
            "mvp4x_cf_decision.json",
        )
        cf_common = _resolve_data_path(
            paths,
            "reports",
            args.cf_common_support_json,
            "mvp4x_cf_common_support_v001.json",
        )
        cf_nuisance = _resolve_data_path(
            paths,
            "reports",
            args.cf_nuisance_json,
            "mvp4x_cf_nuisance_audit_v001.json",
        )
        cf_spatial = _resolve_data_path(
            paths,
            "reports",
            args.cf_spatial_json,
            "mvp4x_cf_spatial_validation_v001.json",
        )
        output_report_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_stratified_baselines_v001.md",
        )
        output_report_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_stratified_baselines_v001.json",
        )
        output_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_csv,
            "mvp4x_stratified_baselines_v001.csv",
        )
        output_decision_md = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_md,
            "mvp4x_stratified_decision.md",
        )
        output_decision_json = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_json,
            "mvp4x_stratified_decision.json",
        )
        output_review_dir = _resolve_data_path(
            paths,
            "reports",
            args.output_review_dir,
            "mvp4x_stratified_review_v001",
        )
        output_iteration_log = _resolve_data_path(
            paths,
            "reports",
            args.output_iteration_log,
            "mvp4x_stratified_iteration_log.md",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (waveform_npz, "features"),
            (design_npz, "interim"),
            (cf_decision, "reports"),
            (cf_common, "reports"),
            (cf_nuisance, "reports"),
            (cf_spatial, "reports"),
            (output_report_md, "reports"),
            (output_report_json, "reports"),
            (output_csv, "reports"),
            (output_decision_md, "reports"),
            (output_decision_json, "reports"),
            (output_review_dir, "reports"),
            (output_iteration_log, "reports"),
        ):
            _ensure_path_within(paths, path, key=key, action="access")
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X stratified baselines "
                f"snapshot={snapshot_npz} waveform={waveform_npz} design={design_npz}."
            )
            return 0
        outputs = run_stratified_baselines_from_paths(
            snapshot_npz=snapshot_npz,
            waveform_features_npz=waveform_npz,
            design_npz=design_npz,
            cf_decision_json=cf_decision,
            cf_common_support_json=cf_common,
            cf_nuisance_json=cf_nuisance,
            cf_spatial_json=cf_spatial,
            config_path=args.stratified_config,
            output_report_md=output_report_md,
            output_report_json=output_report_json,
            output_csv=output_csv,
            output_decision_md=output_decision_md,
            output_decision_json=output_decision_json,
            output_review_dir=output_review_dir,
            output_iteration_log=output_iteration_log,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ModelingDependencyError,
        StratifiedBaselineError,
        StratifiedBaselineCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4X stratified baselines "
        f"decision={outputs.decision['decision']}; "
        f"research_only={outputs.decision['research_only']}; "
        f"no_final_labels={outputs.decision['no_final_labels']}."
    )
    print(f"Wrote report JSON: {output_report_json}")
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
    raise StratifiedBaselineCliError(f"data.{key} is not configured.")


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
        raise StratifiedBaselineCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise StratifiedBaselineCliError(
            f"Refusing to {action} stratified-baseline path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
