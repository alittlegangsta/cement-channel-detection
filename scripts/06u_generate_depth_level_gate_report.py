from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402
from cement_channel.labels.depth_level_schema import (  # noqa: E402
    DEPTH_LEVEL_GATE_VERSION,
    load_depth_level_label_config,
)


class DepthLevelGateError(RuntimeError):
    """Raised when the depth-level target gate cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP-4B-R4 depth-level target gate.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument(
        "--depth-level-config",
        default="configs/depth_level_label.example.yaml",
    )
    parser.add_argument("--depth-level-label-report", default=None)
    parser.add_argument("--depth-level-feature-report", default=None)
    parser.add_argument("--depth-level-audit-report", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths_config = load_paths_config(args.paths_config)
        paths = _resolve_paths(paths_config, args)
        for key, path in paths.items():
            if key.startswith("output_"):
                _ensure_path_within(paths_config, path, key="reports", action="write")
            else:
                _ensure_path_within(paths_config, path, key="reports", action="read")
        config = load_depth_level_label_config(args.depth_level_config)
        report = build_depth_level_gate_report(paths, config=config)
        markdown = format_depth_level_gate_markdown(report)
        if not args.dry_run:
            _write_outputs(
                report,
                markdown,
                output_md=paths["output_report_md"],
                output_json=paths["output_report_json"],
                overwrite=args.overwrite,
            )
    except (
        ManifestBuildError,
        DepthLevelGateError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth-level target gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        "depth_level_baseline_sanity_allowed="
        f"{report['depth_level_baseline_sanity_allowed']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def build_depth_level_gate_report(
    paths: dict[str, Path],
    *,
    config: Any,
) -> dict[str, Any]:
    labels = _read_json(paths["depth_level_label_report"])
    features = _read_json(paths["depth_level_feature_report"])
    audit = _read_json(paths["depth_level_audit_report"])
    blocking: list[str] = []
    warnings: list[str] = []

    if labels.get("errors"):
        blocking.extend(f"labels: {message}" for message in labels["errors"])
    if features.get("errors"):
        blocking.extend(f"features: {message}" for message in features["errors"])
    if audit.get("errors"):
        blocking.extend(f"audit: {message}" for message in audit["errors"])
    warnings.extend(f"labels: {message}" for message in labels.get("warnings", []))
    warnings.extend(f"features: {message}" for message in features.get("warnings", []))
    warnings.extend(f"audit: {message}" for message in audit.get("warnings", []))

    strong_count = int(labels.get("strong_positive_count", 0))
    clear_count = int(labels.get("clear_negative_count", 0))
    if strong_count < config.gate.min_depth_positive_count:
        blocking.append("depth-level strong-positive subset is empty or below minimum.")
    if clear_count < config.gate.min_depth_negative_count:
        blocking.append("depth-level clear-negative subset is empty or below minimum.")

    review = _as_dict(labels.get("review_band_impact"))
    review_fraction = _as_float(review.get("positive_fraction_in_review_band"))
    if (
        review_fraction is not None
        and review_fraction > config.gate.max_5700_band_positive_fraction
    ):
        blocking.append("depth-level positives are dominated by the ~5700 ft review band.")

    comparison = _as_dict(audit.get("depth_vs_side_comparison"))
    depth_best = _as_float(comparison.get("depth_level_best_abs_effect_size"))
    side_best = _as_float(comparison.get("side_level_best_abs_effect_size"))
    depth_delta = _as_float(comparison.get("depth_minus_side_delta"))
    separation_enhanced = bool(audit.get("depth_level_separation_enhanced"))
    if not separation_enhanced:
        blocking.append("depth-level separation did not clearly improve over side-level audit.")

    no_final_labels = all(
        [
            bool(labels.get("no_final_labels")),
            bool(features.get("no_final_labels")),
            bool(audit.get("no_final_labels")),
        ]
    )
    no_forbidden = all(
        [
            bool(labels.get("no_stc")),
            bool(labels.get("no_apes")),
            bool(labels.get("no_deep_learning")),
            bool(labels.get("no_mvp4c")),
            bool(features.get("no_stc")),
            bool(features.get("no_apes")),
            bool(features.get("no_deep_learning")),
            bool(features.get("no_mvp4c")),
            bool(audit.get("no_stc")),
            bool(audit.get("no_apes")),
            bool(audit.get("no_deep_learning")),
            bool(audit.get("no_mvp4c")),
        ]
    )
    if not no_final_labels:
        blocking.append("guardrails: reports do not preserve no_final_labels.")
    if not no_forbidden:
        blocking.append("guardrails: reports permit STC/APES/deep learning/MVP-4C.")

    decision = "no_go" if blocking else "conditional_go"
    baseline_allowed = decision == "conditional_go" and separation_enhanced
    return {
        "gate_version": DEPTH_LEVEL_GATE_VERSION,
        "stage": "MVP-4B-R4",
        "task": "depth_level_target_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "thresholds": {
            "min_depth_positive_count": config.gate.min_depth_positive_count,
            "min_depth_negative_count": config.gate.min_depth_negative_count,
            "max_5700_band_positive_fraction": config.gate.max_5700_band_positive_fraction,
            "depth_level_improvement_effect_size_delta": (
                config.gate.depth_level_improvement_effect_size_delta
            ),
            "sanity_effect_size_threshold": config.gate.sanity_effect_size_threshold,
        },
        "evidence": {
            "depth_level_positive_fraction": labels.get("positive_fraction"),
            "depth_level_positive_count": labels.get("positive_count"),
            "depth_level_negative_count": labels.get("negative_count"),
            "strong_positive_count": strong_count,
            "clear_negative_count": clear_count,
            "feature_count": features.get("depth_feature_count"),
            "depth_level_best_abs_effect_size": depth_best,
            "side_level_best_abs_effect_size": side_best,
            "depth_minus_side_delta": depth_delta,
            "depth_level_separation_enhanced": separation_enhanced,
            "review_band_impact": review,
            "no_final_labels": no_final_labels,
            "no_stc_apes_deep_learning_mvp4c": no_forbidden,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "depth_level_baseline_sanity_allowed": baseline_allowed,
        "side_level_mvp4c_allowed": False,
        "mvp4c_allowed": False,
        "stc_allowed": False,
        "apes_allowed": False,
        "deep_learning_allowed": False,
        "final_labels_allowed": False,
        "recommendation": (
            "conditional_go: consider a depth-level baseline sanity model only; "
            "do not enter side-level MVP-4C/STC/APES/deep learning"
            if baseline_allowed
            else "no_go: return to manual label review or target-definition review"
        ),
        "not_performed": [
            "final label generation",
            "ground truth claim",
            "model training",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C implementation",
        ],
    }


def format_depth_level_gate_markdown(report: dict[str, Any]) -> str:
    evidence = _as_dict(report.get("evidence"))
    lines = [
        "# MVP-4B-R4 Depth-Level Target Gate Report",
        "",
        "This gate reviews whether the depth-level target definition is a better "
        "sanity target than side-depth weak-label classification. It does not "
        "authorize final labels, STC, APES, deep learning, production modeling, or MVP-4C.",
        "",
        f"- decision: `{report['decision']}`",
        "- depth_level_baseline_sanity_allowed: "
        f"`{report['depth_level_baseline_sanity_allowed']}`",
        f"- mvp4c_allowed: `{report['mvp4c_allowed']}`",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Evidence",
        "",
        f"- depth_level_positive_fraction: {evidence.get('depth_level_positive_fraction')}",
        f"- strong_positive_count: {evidence.get('strong_positive_count')}",
        f"- clear_negative_count: {evidence.get('clear_negative_count')}",
        f"- depth_level_best_abs_effect_size: "
        f"{evidence.get('depth_level_best_abs_effect_size')}",
        f"- side_level_best_abs_effect_size: {evidence.get('side_level_best_abs_effect_size')}",
        f"- depth_minus_side_delta: {evidence.get('depth_minus_side_delta')}",
        f"- depth_level_separation_enhanced: "
        f"`{evidence.get('depth_level_separation_enhanced')}`",
        "",
        "## Blocking Issues",
        "",
    ]
    lines.extend(_message_lines(report["blocking_issues"]))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report["warnings"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    return {
        "depth_level_label_report": _resolve_report_path(
            config,
            args.depth_level_label_report,
            "depth_level_labels_report_v001.json",
        ),
        "depth_level_feature_report": _resolve_report_path(
            config,
            args.depth_level_feature_report,
            "depth_level_xsi_features_report_v001.json",
        ),
        "depth_level_audit_report": _resolve_report_path(
            config,
            args.depth_level_audit_report,
            "depth_level_separation_audit_v001.json",
        ),
        "output_report_md": _resolve_report_path(
            config,
            args.output_report_md,
            "depth_level_gate_report_v001.md",
        ),
        "output_report_json": _resolve_report_path(
            config,
            args.output_report_json,
            "depth_level_gate_report_v001.json",
        ),
    }


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise DepthLevelGateError("data.reports is not configured.")


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
        raise DepthLevelGateError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelGateError(
            f"Refusing to {action} depth-level gate path outside data.{key}: {path}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(
    report: dict[str, Any],
    markdown: str,
    *,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


if __name__ == "__main__":
    raise SystemExit(main())
