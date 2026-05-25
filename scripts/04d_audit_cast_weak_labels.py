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
from cement_channel.labels.label_audit import (  # noqa: E402
    audit_cast_weak_labels_from_config,
    write_label_audit_outputs,
)


class LabelAuditCliError(RuntimeError):
    """Raised when CAST weak-label audit cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MVP-3 CAST weak-label candidates.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--label-config", default="configs/label.cast_weak_v001.example.yaml")
    parser.add_argument("--input-npz", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        input_npz = _resolve_label_path(
            config,
            args.input_npz,
            "cast_weak_label_candidates_v001.npz",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "cast_weak_label_audit_v001.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "cast_weak_label_audit_v001.json",
        )
        _ensure_path_within(config, input_npz, key="labels", action="read")
        _ensure_path_within(config, output_md, key="reports", action="write")
        _ensure_path_within(config, output_json, key="reports", action="write")
        report = audit_cast_weak_labels_from_config(
            weak_label_npz=input_npz,
            label_config_path=args.label_config,
        )
        if not args.dry_run:
            write_label_audit_outputs(
                report,
                output_report_md=output_md,
                output_report_json=output_json,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        LabelAuditCliError,
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "CAST weak-label audit "
        f"errors={len(report.errors)}; "
        f"warnings={len(report.warnings)}; "
        f"plus_coverage={report.coverage['plus']}; "
        f"disagreement={report.plus_minus_disagreement_rate}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
    return 1 if report.errors else 0


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise LabelAuditCliError("data.reports is not configured.")


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
    raise LabelAuditCliError("data.root or data.labels is required for label inputs.")


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
        raise LabelAuditCliError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise LabelAuditCliError(
            f"Refusing to {action} CAST weak-label audit path outside data.{key}: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
