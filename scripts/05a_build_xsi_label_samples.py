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
from cement_channel.evaluation.xsi_label_sampler import (  # noqa: E402
    build_xsi_label_samples_from_config,
    write_xsi_label_sample_outputs,
)


class XsiLabelSampleCliError(RuntimeError):
    """Raised when XSI label sample index generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP-4A XSI label sample index.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--correlation-config",
        default="configs/mvp4a_xsi_cast_correlation.example.yaml",
    )
    parser.add_argument("--label-npz", default=None)
    parser.add_argument("--depth-only-npz", default=None)
    parser.add_argument("--orientation-npz", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        label_npz = _resolve_label_path(
            config,
            args.label_npz,
            "cast_weak_label_candidates_v001.npz",
        )
        depth_npz = _resolve_interim_path(config, args.depth_only_npz, "depth_only_v001.npz")
        orientation_npz = _resolve_interim_path(
            config,
            args.orientation_npz,
            "orientation_confidence_v001.npz",
        )
        output_npz = _resolve_interim_path(
            config,
            args.output_npz,
            "xsi_label_samples_v001.npz",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "xsi_label_samples_report_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "xsi_label_samples_report_v001.json",
        )
        _ensure_path_within(config, label_npz, key="labels", action="read")
        _ensure_path_within(config, depth_npz, key="interim", action="read")
        _ensure_path_within(config, orientation_npz, key="interim", action="read")
        _ensure_path_within(config, output_npz, key="interim", action="write")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        report, arrays = build_xsi_label_samples_from_config(
            label_candidate_npz=label_npz,
            depth_only_npz=depth_npz,
            orientation_confidence_npz=orientation_npz,
            correlation_config_path=args.correlation_config,
        )
        if not args.dry_run:
            write_xsi_label_sample_outputs(
                report,
                arrays,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        XsiLabelSampleCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "XSI label samples "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"depth={report.shape['depth']}; "
        f"side={report.shape['side']}; "
        f"azimuthal_valid={report.coverage['valid_for_azimuthal_validation_count']}; "
        f"no_final_labels={report.no_final_labels}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote NPZ: {output_npz}")
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 1 if report.errors else 0


def _resolve_interim_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    interim = data.get("interim")
    if interim:
        return Path(str(interim)) / filename
    raise XsiLabelSampleCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise XsiLabelSampleCliError("data.reports is not configured.")


def _resolve_label_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    labels = data.get("labels")
    if labels:
        return Path(str(labels)) / filename
    root = data.get("root")
    if root:
        return Path(str(root)) / "labels" / filename
    raise XsiLabelSampleCliError("data.root or data.labels is required for label inputs.")


def _ensure_path_within(
    config: dict[str, Any],
    path: Path,
    *,
    key: str,
    action: str,
) -> None:
    data = _as_dict(config.get("data"))
    if key == "labels":
        root = Path(str(data.get("labels", Path(str(data.get("root", ""))) / "labels"))).resolve()
    else:
        root = Path(str(data.get(key, ""))).resolve()
    if not str(root):
        raise XsiLabelSampleCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise XsiLabelSampleCliError(
            f"Refusing to {action} XSI label sample path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
