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
from cement_channel.modeling.mvp4x_screening_scores import (  # noqa: E402
    ScreeningScoresError,
    generate_screening_scores_from_paths,
)


class ScreeningScoresCliError(RuntimeError):
    """Raised when the MVP-4X screening score CLI cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe MVP-4X out-of-fold screening scores."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--score-config", default="configs/mvp4x_screening_baseline.example.yaml")
    parser.add_argument("--snapshot-npz", default=None)
    parser.add_argument("--screening-policy-npz", default=None)
    parser.add_argument("--screening-policy-json", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-csv", default=None)
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
        output_npz = _resolve_data_path(
            paths,
            "interim",
            args.output_npz,
            "mvp4x_screening_scores_oof_v001.npz",
        )
        output_csv = _resolve_data_path(
            paths,
            "reports",
            args.output_csv,
            "mvp4x_screening_scores_oof_v001.csv",
        )
        output_md = _resolve_data_path(
            paths,
            "reports",
            args.output_report_md,
            "mvp4x_screening_scores_oof_v001.md",
        )
        output_json = _resolve_data_path(
            paths,
            "reports",
            args.output_report_json,
            "mvp4x_screening_scores_oof_v001.json",
        )
        for path, key in (
            (snapshot_npz, "interim"),
            (policy_npz, "interim"),
            (policy_json, "reports"),
            (output_npz, "interim"),
            (output_csv, "reports"),
            (output_md, "reports"),
            (output_json, "reports"),
        ):
            _ensure_path_within(paths, path, key=key)
        if args.dry_run:
            print(
                "Dry run: would generate MVP-4X OOF screening scores "
                f"snapshot={snapshot_npz} policy={policy_npz}."
            )
            return 0
        outputs = generate_screening_scores_from_paths(
            snapshot_npz=snapshot_npz,
            screening_policy_npz=policy_npz,
            screening_policy_json=policy_json,
            config_path=args.score_config,
            output_npz=output_npz,
            output_csv=output_csv,
            output_report_md=output_md,
            output_report_json=output_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ModelingDependencyError,
        ScreeningScoresError,
        ScreeningScoresCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X OOF screening scores "
        f"rows={len(outputs.rows)}; "
        f"supported={outputs.report['sample_counts']['supported_scored']}; "
        f"not_validated_for_deployment={outputs.report['not_validated_for_deployment']}."
    )
    print(f"Wrote scores NPZ: {output_npz}")
    print(f"Wrote scores JSON: {output_json}")
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
    raise ScreeningScoresCliError(f"data.{key} is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, key: str) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key))).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ScreeningScoresCliError(f"{resolved} is outside configured data.{key}: {root}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
