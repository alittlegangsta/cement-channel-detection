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
from cement_channel.experiments.mvp4x_stratified_design import (  # noqa: E402
    StratifiedDesignError,
    build_stratified_design_from_paths,
)


class StratifiedDesignCliError(RuntimeError):
    """Raised when the MVP-4X stratified design CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the MVP-4X research-only stratified exploratory design."
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
    parser.add_argument("--cf-decision-json", default=None)
    parser.add_argument("--cf-common-support-json", default=None)
    parser.add_argument("--cf-nuisance-json", default=None)
    parser.add_argument("--cf-spatial-json", default=None)
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
        output_npz = _resolve_data_path(
            paths,
            "interim",
            args.output_npz,
            "mvp4x_stratified_design_v001.npz",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_stratified_design_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_stratified_design_v001.json",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (waveform_npz, "features"),
            (cf_decision, "reports"),
            (cf_common, "reports"),
            (cf_nuisance, "reports"),
            (cf_spatial, "reports"),
            (output_npz, "interim"),
            (output_md, "reports"),
            (output_json, "reports"),
        ):
            _ensure_path_within(paths, path, key=key, action="access")
        if args.dry_run:
            print(
                "Dry run: would build MVP-4X stratified design "
                f"snapshot={snapshot_npz} waveform={waveform_npz}."
            )
            return 0
        outputs = build_stratified_design_from_paths(
            snapshot_npz=snapshot_npz,
            waveform_features_npz=waveform_npz,
            cf_decision_json=cf_decision,
            cf_common_support_json=cf_common,
            cf_nuisance_json=cf_nuisance,
            cf_spatial_json=cf_spatial,
            config_path=args.stratified_config,
            output_npz=output_npz,
            output_report_md=output_md,
            output_report_json=output_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        StratifiedDesignError,
        StratifiedDesignCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4X stratified design "
        f"cohort_count={outputs.design['cohort_count']}; "
        f"research_only={outputs.design['research_only']}; "
        f"no_final_labels={outputs.design['no_final_labels']}."
    )
    print(f"Wrote design NPZ: {output_npz}")
    print(f"Wrote design JSON: {output_json}")
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
    raise StratifiedDesignCliError(f"data.{key} is not configured.")


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
        raise StratifiedDesignCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise StratifiedDesignCliError(
            f"Refusing to {action} stratified-design path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
