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
from cement_channel.training.depth_level_baseline_schema import (  # noqa: E402
    DEPTH_LEVEL_BASELINE_GATE_VERSION,
)


class DepthLevelBaselineGateError(RuntimeError):
    """Raised when the depth-level baseline gate cannot run safely."""


MIN_BASELINE_MARGIN = 0.03
SUPPORTED_GATE_MODELS = {"logistic_regression", "linear_probe"}
SUPPORTED_GATE_VARIANTS = {
    "all_positive_vs_negative",
    "strong_positive_vs_clear_negative",
    "high_confidence_positive_vs_clear_negative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-4B-R4b depth-level baseline gate."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--baseline-review-summary", default=None)
    parser.add_argument("--output-report-md", default=None)
    parser.add_argument("--output-report-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_paths_config(args.paths_config)
        paths = _resolve_paths(config, args)
        for key, path in paths.items():
            if key.startswith("output_"):
                _ensure_path_within(config, path, key="reports", action="write")
            else:
                _ensure_path_within(config, path, key="reports", action="read")
        report = build_depth_level_baseline_gate_report(paths)
        markdown = format_depth_level_baseline_gate_markdown(report)
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
        DepthLevelBaselineGateError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth-level baseline gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        "controlled_depth_level_feature_refinement_allowed="
        f"{report['controlled_depth_level_feature_refinement_allowed']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def build_depth_level_baseline_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    baseline = _read_json(paths["baseline_report"])
    review = _read_json(paths["baseline_review_summary"])
    blocking: list[str] = []
    warnings: list[str] = []

    warnings.extend(f"baseline: {message}" for message in baseline.get("warnings", []))
    warnings.extend(f"review: {message}" for message in review.get("warnings", []))
    if baseline.get("errors"):
        blocking.extend(f"baseline: {message}" for message in baseline["errors"])
    if review.get("errors"):
        blocking.extend(f"review: {message}" for message in review["errors"])
    if baseline.get("report_version") != "depth_level_baseline_v001":
        blocking.append("baseline: unexpected report_version.")
    if review.get("review_version") != "depth_level_baseline_review_v001":
        blocking.append("review: unexpected review_version.")

    best = _as_dict(baseline.get("best_result"))
    if not best:
        blocking.append("baseline: no usable best_result was produced.")
    target_variant = str(best.get("target_variant", ""))
    model_type = str(best.get("model_type", ""))
    if target_variant and target_variant not in SUPPORTED_GATE_VARIANTS:
        blocking.append(f"baseline: unsupported target variant {target_variant}.")
    if model_type and model_type not in SUPPORTED_GATE_MODELS:
        blocking.append(f"baseline: unsupported model type {model_type}.")

    margin = _as_float(best.get("balanced_accuracy_margin"))
    required_margin = max(_as_float(best.get("required_margin")) or 0.0, MIN_BASELINE_MARGIN)
    real_balanced_accuracy = _as_float(best.get("real_balanced_accuracy"))
    permutation_balanced_accuracy = _as_float(best.get("permutation_balanced_accuracy"))
    predicted_positive_rate = _as_float(best.get("predicted_positive_rate"))
    stable_fold_count = int(best.get("stable_fold_count") or 0)
    stable_fold_min_count = int(best.get("stable_fold_min_count") or 0)
    if margin is None or margin < required_margin:
        blocking.append(
            "baseline: real-minus-permutation balanced-accuracy margin is below "
            f"required threshold {required_margin}: {margin}."
        )
    if bool(best.get("permutation_lower_than_real")) is not True:
        blocking.append("baseline: permutation metric is not lower than real-label metric.")
    if bool(best.get("passes_margin")) is not True:
        blocking.append("baseline: best result did not pass the configured margin.")
    if bool(best.get("degenerate_prediction")):
        blocking.append("baseline: best result has degenerate predictions.")
    if bool(best.get("stable_folds_pass")) is not True:
        blocking.append("baseline: best result did not pass stable-fold requirement.")
    if stable_fold_min_count > 0 and stable_fold_count < stable_fold_min_count:
        blocking.append(
            "baseline: stable fold count below requirement "
            f"{stable_fold_min_count}: {stable_fold_count}."
        )

    usable_variants = [
        str(value) for value in baseline.get("usable_target_variants", []) if str(value)
    ]
    if not usable_variants:
        blocking.append("baseline: no target variant is usable.")
    if "high_confidence_positive_vs_clear_negative" in usable_variants:
        warnings.append(
            "baseline: usable evidence is restricted to high-confidence positive vs clear "
            "negative depths; all-positive target should remain audit-only."
        )

    guardrails = {
        "baseline_no_final_labels": _data_bool(baseline, "no_final_labels"),
        "baseline_no_stc": _data_bool(baseline, "no_stc"),
        "baseline_no_apes": _data_bool(baseline, "no_apes"),
        "baseline_no_deep_learning": _data_bool(baseline, "no_deep_learning"),
        "baseline_no_mvp4c": _data_bool(baseline, "no_mvp4c"),
        "review_no_final_labels": _data_bool(review, "no_final_labels"),
        "review_no_stc": _data_bool(review, "no_stc"),
        "review_no_apes": _data_bool(review, "no_apes"),
        "review_no_deep_learning": _data_bool(review, "no_deep_learning"),
        "review_no_mvp4c": _data_bool(review, "no_mvp4c"),
    }
    if not all(guardrails.values()):
        blocking.append("guardrails: baseline or review reports violated forbidden-scope flags.")
    if baseline.get("production_training") is not False:
        blocking.append("guardrails: baseline report indicates production training.")
    if review.get("no_production_model") is not True:
        blocking.append("guardrails: review report does not preserve no_production_model=true.")

    decision = _decision(blocking, warnings)
    controlled_refinement_allowed = decision in {"go", "conditional_go"} and not blocking
    return {
        "gate_version": DEPTH_LEVEL_BASELINE_GATE_VERSION,
        "stage": "MVP-4B-R4b",
        "task": "depth_level_baseline_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "gate_thresholds": {
            "min_real_minus_permutation_balanced_accuracy_margin": required_margin,
            "min_stable_fold_count": stable_fold_min_count,
        },
        "evidence": {
            "usable_target_variants": usable_variants,
            "best_target_variant": target_variant or None,
            "best_model_type": model_type or None,
            "real_balanced_accuracy": real_balanced_accuracy,
            "permutation_balanced_accuracy": permutation_balanced_accuracy,
            "balanced_accuracy_margin": margin,
            "predicted_positive_rate": predicted_positive_rate,
            "stable_fold_count": stable_fold_count,
            "stable_fold_min_count": stable_fold_min_count,
            "permutation_lower_than_real": best.get("permutation_lower_than_real"),
            "passes_margin": best.get("passes_margin"),
            "stable_folds_pass": best.get("stable_folds_pass"),
            "degenerate_prediction": best.get("degenerate_prediction"),
            "review_figure_count": len(_as_dict(review.get("figures"))),
            "guardrails": guardrails,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "controlled_depth_level_feature_refinement_allowed": controlled_refinement_allowed,
        "depth_level_baseline_sanity_allowed": False,
        "side_level_mvp4c_allowed": False,
        "mvp4c_allowed": False,
        "stc_allowed": False,
        "apes_allowed": False,
        "deep_learning_allowed": False,
        "final_labels_allowed": False,
        "recommendation": (
            "controlled depth-level feature refinement may be considered; keep MVP-4C, "
            "STC/APES, deep learning, production claims, and final labels blocked"
            if controlled_refinement_allowed
            else "no_go: prepare a manual label review pack before further feature work"
        ),
        "not_performed": [
            "final label generation",
            "ground truth claim",
            "production model training",
            "model weight export",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C implementation",
        ],
    }


def format_depth_level_baseline_gate_markdown(report: dict[str, Any]) -> str:
    evidence = _as_dict(report.get("evidence"))
    lines = [
        "# MVP-4B-R4b Depth-Level Baseline Gate Report",
        "",
        "This gate reviews a simple depth-level baseline sanity result against CAST "
        "weak-label candidates. It does not authorize final labels, STC, APES, "
        "deep learning, production modeling, or MVP-4C.",
        "",
        "## Decision",
        "",
        f"- decision: `{report['decision']}`",
        "- controlled_depth_level_feature_refinement_allowed: "
        f"`{report['controlled_depth_level_feature_refinement_allowed']}`",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Evidence",
        "",
        f"- usable_target_variants: {evidence.get('usable_target_variants')}",
        f"- best_target_variant: `{evidence.get('best_target_variant')}`",
        f"- best_model_type: `{evidence.get('best_model_type')}`",
        f"- real_balanced_accuracy: {evidence.get('real_balanced_accuracy')}",
        f"- permutation_balanced_accuracy: {evidence.get('permutation_balanced_accuracy')}",
        f"- balanced_accuracy_margin: {evidence.get('balanced_accuracy_margin')}",
        f"- predicted_positive_rate: {evidence.get('predicted_positive_rate')}",
        f"- stable_fold_count: {evidence.get('stable_fold_count')}",
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
    reports = Path(str(_as_dict(config.get("data")).get("reports", "")))
    if not str(reports):
        raise DepthLevelBaselineGateError("data.reports is not configured.")
    return {
        "baseline_report": Path(
            args.baseline_report or reports / "depth_level_baseline_report_v001.json"
        ),
        "baseline_review_summary": Path(
            args.baseline_review_summary
            or reports
            / "depth_level_baseline_review_v001"
            / "depth_level_baseline_review_summary_v001.json"
        ),
        "output_report_md": Path(
            args.output_report_md or reports / "depth_level_baseline_gate_v001.md"
        ),
        "output_report_json": Path(
            args.output_report_json or reports / "depth_level_baseline_gate_v001.json"
        ),
    }


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DepthLevelBaselineGateError(f"Required report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DepthLevelBaselineGateError(f"Report must be a JSON object: {path}")
    return data


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
        raise DepthLevelBaselineGateError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelBaselineGateError(
            f"Refusing to {action} depth-level baseline gate path outside data.{key}: {path}"
        ) from exc


def _decision(blocking: list[str], warnings: list[str]) -> str:
    if blocking:
        return "no_go"
    if warnings:
        return "conditional_go"
    return "go"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _data_bool(data: dict[str, Any], key: str) -> bool:
    return bool(data.get(key)) is True


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


if __name__ == "__main__":
    raise SystemExit(main())
