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
from cement_channel.evaluation.subset_feature_audit import (  # noqa: E402
    audit_subset_feature_separation_from_config,
)


class SubsetFeatureAuditCliError(RuntimeError):
    """Raised when subset feature audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit feature separation on MVP-4B-R3 label-quality subsets."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--label-quality-config",
        default="configs/mvp4b_label_quality_subsets.example.yaml",
    )
    parser.add_argument("--sample-table-npz", default=None)
    parser.add_argument("--label-quality-subsets-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-figure-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        sample_npz = _resolve_interim_path(
            paths,
            args.sample_table_npz,
            "baseline_sample_table_receiver_enhanced_v001.npz",
        )
        subsets_npz = _resolve_interim_path(
            paths,
            args.label_quality_subsets_npz,
            "label_quality_subsets_v001.npz",
        )
        output_md = _resolve_report_path(
            paths,
            args.output_report_md,
            "subset_feature_separation_audit_v001.md",
        )
        output_json = _resolve_report_path(
            paths,
            args.output_report_json,
            "subset_feature_separation_audit_v001.json",
        )
        output_csv = _resolve_report_path(
            paths,
            args.output_csv,
            "subset_feature_separation_audit_v001.csv",
        )
        output_fig_dir = _resolve_report_path(
            paths,
            args.output_figure_dir,
            "subset_feature_separation_audit_v001",
        )
        _ensure_path_within(paths, sample_npz, key="interim", action="read")
        _ensure_path_within(paths, subsets_npz, key="interim", action="read")
        for output_path in (output_md, output_json, output_csv, output_fig_dir):
            _ensure_path_within(paths, output_path, key="reports", action="write")

        if args.dry_run:
            import numpy as np  # noqa: PLC0415

            from cement_channel.evaluation.subset_feature_audit import (  # noqa: PLC0415
                audit_subset_feature_separation,
            )
            from cement_channel.labels.label_quality_schema import (  # noqa: PLC0415
                load_label_quality_config,
            )

            with np.load(sample_npz, allow_pickle=False) as sample_data:
                sample_arrays = {key: sample_data[key] for key in sample_data.files}
            with np.load(subsets_npz, allow_pickle=False) as subset_data:
                subset_arrays = {key: subset_data[key] for key in subset_data.files}
            report, _rows = audit_subset_feature_separation(
                sample_arrays=sample_arrays,
                subset_arrays=subset_arrays,
                config=load_label_quality_config(args.label_quality_config),
                inputs={
                    "sample_table_npz": str(sample_npz),
                    "label_quality_subsets_npz": str(subsets_npz),
                    "label_quality_config_path": str(args.label_quality_config),
                },
                output_csv=output_csv,
                output_figure_dir=output_fig_dir,
            )
        else:
            report = audit_subset_feature_separation_from_config(
                sample_table_npz=sample_npz,
                label_quality_subsets_npz=subsets_npz,
                label_quality_config_path=args.label_quality_config,
                output_report_md=output_md,
                output_report_json=output_json,
                output_csv=output_csv,
                output_figure_dir=output_fig_dir,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        SubsetFeatureAuditCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Subset feature audit "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        "quality_best_abs_effect="
        f"{report.signal_enhancement['quality_subset_best_abs_effect_size']}; "
        "quality_minus_all_delta="
        f"{report.signal_enhancement['quality_minus_all_delta']}; "
        f"label_noise_likely={report.label_noise_likely}."
    )
    if args.dry_run:
        print("Dry run: no reports, CSV, or figures written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote CSV report: {output_csv}")
        print(f"Wrote review figures: {output_fig_dir}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise SubsetFeatureAuditCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise SubsetFeatureAuditCliError("data.reports is not configured.")


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
        raise SubsetFeatureAuditCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise SubsetFeatureAuditCliError(
            f"Refusing to {action} subset feature audit path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
