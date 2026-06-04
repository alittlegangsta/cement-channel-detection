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
from cement_channel.evaluation.mvp4x_ranking_audit import (  # noqa: E402
    RankingAuditError,
    run_ranking_audit_from_paths,
)


class RankingAuditCliError(RuntimeError):
    """Raised when the MVP-4X ranking audit CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP-4X research-only ranking audit.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--policy-config", default="configs/mvp4x_regime_policy.example.yaml")
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--waveform-features-npz", default=None)
    parser.add_argument("--policy-npz", default=None)
    parser.add_argument("--robustness-json", default=None)
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
        waveform_npz = _resolve_data_path(
            paths,
            "features",
            args.waveform_features_npz,
            "mvp4x_waveform_features_v001.npz",
        )
        policy_npz = _resolve_data_path(
            paths,
            "interim",
            args.policy_npz,
            "mvp4x_regime_policy_v001.npz",
        )
        robustness_json = _resolve_data_path(
            paths,
            "reports",
            args.robustness_json,
            "mvp4x_regime_robustness_v001.json",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_ranking_audit_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_ranking_audit_v001.json",
        )
        output_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_csv,
            "mvp4x_ranking_audit_v001.csv",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (waveform_npz, "features"),
            (policy_npz, "interim"),
            (robustness_json, "reports"),
            (output_md, "reports"),
            (output_json, "reports"),
            (output_csv, "reports"),
        ):
            _ensure_path_within(paths, path, key=key)
        if args.dry_run:
            print(
                "Dry run: would run MVP-4X ranking audit "
                f"policy={policy_npz} robustness={robustness_json}."
            )
            return 0
        outputs = run_ranking_audit_from_paths(
            snapshot_npz=snapshot_npz,
            waveform_features_npz=waveform_npz,
            policy_npz=policy_npz,
            robustness_json=robustness_json,
            config_path=args.policy_config,
            output_report_md=output_md,
            output_report_json=output_json,
            output_csv=output_csv,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        RankingAuditError,
        RankingAuditCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X ranking audit "
        f"row_count={outputs.report['row_count']}; "
        f"stable={outputs.report['summary']['ranking_stable_cohorts']}; "
        f"research_only={outputs.report['research_only']}."
    )
    print(f"Wrote ranking JSON: {output_json}")
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
    raise RankingAuditCliError(f"data.{key} is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, key: str) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key))).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise RankingAuditCliError(f"{resolved} is outside configured data.{key}: {root}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
