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
from cement_channel.labels.cast_baseline import (  # noqa: E402
    build_cast_zc_baseline_from_config,
    write_cast_baseline_outputs,
)


class CastBaselineCliError(RuntimeError):
    """Raised when CAST baseline generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adaptive CAST Zc baseline for MVP-3.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--label-config", default="configs/label.cast_weak_v001.example.yaml")
    parser.add_argument("--input-npz", default=None)
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
        input_npz = _resolve_interim_path(config, args.input_npz, "cast_label_input_v001.npz")
        output_npz = _resolve_interim_path(config, args.output_npz, "cast_zc_baseline_v001.npz")
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "cast_zc_baseline_report_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "cast_zc_baseline_report_v001.json",
        )
        _ensure_interim_path(config, input_npz, action="read")
        _ensure_interim_path(config, output_npz, action="write")
        _ensure_report_output(config, output_md)
        _ensure_report_output(config, output_json)
        report, arrays = build_cast_zc_baseline_from_config(
            cast_label_input_npz=input_npz,
            label_config_path=args.label_config,
        )
        if not args.dry_run:
            write_cast_baseline_outputs(
                report,
                arrays,
                output_npz=output_npz,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        CastBaselineCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "CAST baseline "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"method={report.method}; "
        f"window_samples={report.window_samples}; "
        f"valid_ratio={report.baseline_valid_ratio}."
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
    raise CastBaselineCliError("data.interim is not configured.")


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise CastBaselineCliError("data.reports is not configured.")


def _ensure_interim_path(config: dict[str, Any], path: Path, *, action: str) -> None:
    _ensure_path_within(config, path, key="interim", action=action)


def _ensure_report_output(config: dict[str, Any], path: Path) -> None:
    _ensure_path_within(config, path, key="reports", action="write")


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
        raise CastBaselineCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise CastBaselineCliError(
            f"Refusing to {action} CAST baseline path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
