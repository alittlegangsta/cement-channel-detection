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
from cement_channel.features.xsi_basic_features import (  # noqa: E402
    extract_xsi_basic_features_from_config,
    write_xsi_basic_feature_outputs,
)


class XsiBasicFeatureCliError(RuntimeError):
    """Raised when MVP-4A XSI basic feature extraction cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MVP-4A XSI basic signal features.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--mapping", default="configs/raw_variable_mapping.yaml")
    parser.add_argument(
        "--correlation-config",
        default="configs/mvp4a_xsi_cast_correlation.example.yaml",
    )
    parser.add_argument("--label-samples-npz", default=None)
    parser.add_argument(
        "--input-waveform-npz",
        default=None,
        help="Optional tiny pre-read waveform NPZ for tests; raw MAT chunking remains the default.",
    )
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--limit-depth", type=int, default=None)
    parser.add_argument("--chunk-depth-samples", type=int, default=None)
    parser.add_argument("--max-time-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        sample_npz = _resolve_interim_path(
            config,
            args.label_samples_npz,
            "xsi_label_samples_v001.npz",
        )
        output_npz = _resolve_feature_path(
            config,
            args.output_npz,
            "xsi_basic_features_v001.npz",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "xsi_basic_features_report_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "xsi_basic_features_report_v001.json",
        )
        _ensure_path_within(config, sample_npz, key="interim", action="read")
        _ensure_path_within(config, output_npz, key="features", action="write")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        input_waveform_npz = Path(args.input_waveform_npz) if args.input_waveform_npz else None
        if input_waveform_npz is not None:
            _ensure_path_within(config, input_waveform_npz, key="interim", action="read")
        report, arrays = extract_xsi_basic_features_from_config(
            paths_config=config,
            mapping_path=Path(args.mapping),
            label_samples_npz=sample_npz,
            correlation_config_path=args.correlation_config,
            input_waveform_npz=input_waveform_npz,
            limit_depth=args.limit_depth,
            chunk_depth_samples=args.chunk_depth_samples,
            max_time_samples=args.max_time_samples,
        )
        if not args.dry_run:
            write_xsi_basic_feature_outputs(
                report,
                arrays,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        XsiBasicFeatureCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "XSI basic features "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"shape={report.shape['xsi_basic_features']}; "
        f"max_chunk_bytes={report.memory_usage['max_observed_chunk_waveform_bytes']}; "
        f"no_model_training={report.no_model_training}."
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
    raise XsiBasicFeatureCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise XsiBasicFeatureCliError("data.reports is not configured.")


def _resolve_feature_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    features = data.get("features")
    if features:
        return Path(str(features)) / filename
    root = data.get("root")
    if root:
        return Path(str(root)) / "features" / filename
    raise XsiBasicFeatureCliError("data.root or data.features is required for feature outputs.")


def _ensure_path_within(
    config: dict[str, Any],
    path: Path,
    *,
    key: str,
    action: str,
) -> None:
    data = _as_dict(config.get("data"))
    if key == "features":
        root = Path(
            str(data.get("features", Path(str(data.get("root", ""))) / "features"))
        ).resolve()
    else:
        root = Path(str(data.get(key, ""))).resolve()
    if not str(root):
        raise XsiBasicFeatureCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise XsiBasicFeatureCliError(
            f"Refusing to {action} XSI basic feature path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
