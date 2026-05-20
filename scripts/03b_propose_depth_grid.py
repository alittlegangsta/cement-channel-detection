from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.depth_grid import (  # noqa: E402
    load_depth_axis_audit,
    propose_depth_grid,
    write_depth_grid_outputs,
)
from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402


class DepthGridProposalCliError(RuntimeError):
    """Raised when depth-grid proposal cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose a canonical MVP-2 depth grid.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--audit-report-json", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--output-config", default="configs/alignment.depth_grid.example.yaml")
    parser.add_argument("--step-strategy", default="coarsest_median_step")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        audit_json = _resolve_report_path(
            config,
            args.audit_report_json,
            "depth_axis_audit_report.json",
        )
        output_md = _resolve_report_path(
            config,
            args.output_report_md,
            "depth_grid_proposal.md",
        )
        output_json = _resolve_report_path(
            config,
            args.output_report_json,
            "depth_grid_proposal.json",
        )
        output_config = Path(args.output_config)
        _ensure_report_input(config, audit_json)
        _ensure_report_output(config, output_md)
        _ensure_report_output(config, output_json)
        audit = load_depth_axis_audit(audit_json)
        proposal = propose_depth_grid(
            audit,
            source_audit_report=audit_json,
            step_strategy=args.step_strategy,
        )
        if not args.dry_run:
            write_depth_grid_outputs(
                proposal,
                output_json=output_json,
                output_md=output_md,
                output_config=output_config,
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthGridProposalCliError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth grid proposal "
        f"decision={proposal.decision}; "
        f"start={proposal.depth_start}; "
        f"stop={proposal.depth_stop}; "
        f"step={proposal.depth_step}; "
        f"warnings={len(proposal.warnings)}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {output_md}")
        print(f"Wrote JSON report: {output_json}")
        print(f"Wrote config example: {output_config}")
    return 1 if proposal.decision == "no_go" else 0


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthGridProposalCliError("data.reports is not configured; pass an explicit path.")


def _ensure_report_input(config: dict[str, Any], input_path: Path) -> None:
    _ensure_reports_path(config, input_path, action="read")
    if not input_path.exists():
        raise DepthGridProposalCliError(f"Depth axis audit report does not exist: {input_path}")


def _ensure_report_output(config: dict[str, Any], output_path: Path) -> None:
    _ensure_reports_path(config, output_path, action="write")


def _ensure_reports_path(config: dict[str, Any], path: Path, *, action: str) -> None:
    data = _as_dict(config.get("data"))
    reports = Path(str(data.get("reports", ""))).resolve()
    if not str(reports):
        raise DepthGridProposalCliError("data.reports is not configured.")
    try:
        path.resolve().relative_to(reports)
    except ValueError as exc:
        raise DepthGridProposalCliError(
            f"Refusing to {action} depth-grid report outside data.reports: {path}"
        ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
