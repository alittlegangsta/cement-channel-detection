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
from cement_channel.training.depth_level_refinement_schema import (  # noqa: E402
    DEPTH_LEVEL_REFINEMENT_GATE_VERSION,
)


class DepthLevelRefinementGateError(RuntimeError):
    """Raised when depth-level refinement gate cannot run safely."""


MIN_GO_MARGIN = 0.05
MIN_CONDITIONAL_MARGIN = 0.03
MIN_FOLD_FRACTION = 0.66
MIN_PASSING_FEATURE_GROUPS = 2
MIN_PASSING_THRESHOLDS = 2
MIN_PASSING_SPLITS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-4B-R4c depth-level refinement gate."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--refinement-report", default=None)
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--review-summary", default=None)
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
        report = build_depth_level_refinement_gate_report(paths)
        markdown = format_depth_level_refinement_gate_markdown(report)
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
        DepthLevelRefinementGateError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Depth-level refinement gate "
        f"decision={report['decision']}; "
        f"blocking_issues={len(report['blocking_issues'])}; "
        f"warnings={len(report['warnings'])}; "
        f"manual_confirmation_required={report['manual_confirmation_required']}; "
        f"next_branch_requires_human_approval={report['next_branch_requires_human_approval']}."
    )
    if args.dry_run:
        print("Dry run: no outputs written.")
    else:
        print(f"Wrote Markdown report: {paths['output_report_md']}")
        print(f"Wrote JSON report: {paths['output_report_json']}")
    return 1 if report["decision"] == "no_go" else 0


def build_depth_level_refinement_gate_report(paths: dict[str, Path]) -> dict[str, Any]:
    refinement = _read_json(paths["refinement_report"])
    baseline = _read_json(paths["baseline_report"])
    review = _read_json(paths["review_summary"])
    blocking: list[str] = []
    warnings: list[str] = []
    manual_items: list[str] = []

    warnings.extend(f"refinement: {message}" for message in refinement.get("warnings", []))
    warnings.extend(f"review: {message}" for message in review.get("warnings", []))
    if refinement.get("errors"):
        blocking.extend(f"refinement: {message}" for message in refinement["errors"])
    if review.get("errors"):
        blocking.extend(f"review: {message}" for message in review["errors"])
    if refinement.get("report_version") != "depth_level_refinement_v001":
        blocking.append("refinement: unexpected report_version.")
    if baseline.get("report_version") != "depth_level_baseline_v001":
        blocking.append("baseline: unexpected report_version.")
    if review.get("review_version") != "depth_level_refinement_review_v001":
        blocking.append("review: unexpected review_version.")

    best = _as_dict(refinement.get("best_result"))
    robustness = _as_dict(refinement.get("robustness_summary"))
    best_margin = _as_float(best.get("margin_mean"))
    best_permutation = _as_float(best.get("permutation_balanced_accuracy_mean"))
    best_real = _as_float(best.get("balanced_accuracy_mean"))
    predicted_positive_rate = _as_float(best.get("predicted_positive_rate"))
    fold_fraction = _as_float(best.get("folds_above_permutation_fraction"))
    if not best:
        blocking.append("refinement: no best_result was produced.")
    if best_margin is None or best_margin < MIN_CONDITIONAL_MARGIN:
        blocking.append(
            "refinement: best margin does not beat permutation by the conditional "
            f"threshold {MIN_CONDITIONAL_MARGIN}: {best_margin}."
        )
    elif best_margin < MIN_GO_MARGIN:
        manual_items.append(
            "Margin is between conditional and go thresholds; confirm whether it is enough."
        )
    if bool(best.get("degenerate_prediction")):
        blocking.append("refinement: best result has degenerate predictions.")
    if fold_fraction is None or fold_fraction < MIN_FOLD_FRACTION:
        blocking.append(
            "refinement: fold stability below requirement "
            f"{MIN_FOLD_FRACTION}: {fold_fraction}."
        )
    if bool(robustness.get("suspicious_leakage")):
        blocking.append("refinement: suspicious leakage flag is set.")
    if bool(robustness.get("depends_on_5700_band")):
        manual_items.append("Confirm whether the ~5700 ft review band should be retained.")
    if bool(robustness.get("depends_on_single_feature_group")):
        manual_items.append("Confirm reliance on a single feature group.")
    if bool(robustness.get("depends_on_single_confidence_threshold")):
        manual_items.append("Confirm reliance on a single confidence threshold.")
    if bool(robustness.get("depends_on_single_split")):
        manual_items.append("Confirm reliance on one depth-block split setting.")

    passing_groups = _as_list(robustness.get("passing_feature_groups"))
    passing_thresholds = _as_list(robustness.get("passing_confidence_thresholds"))
    passing_splits = _as_list(robustness.get("passing_depth_block_splits"))
    passing_exclude_values = _as_list(robustness.get("passing_exclude_5700_values"))
    if len(passing_groups) < MIN_PASSING_FEATURE_GROUPS:
        manual_items.append("Only a small number of feature groups passed robustness checks.")
    if len(passing_thresholds) < MIN_PASSING_THRESHOLDS:
        manual_items.append("Only a small number of confidence thresholds passed.")
    if len(passing_splits) < MIN_PASSING_SPLITS:
        manual_items.append("Only one depth-block split setting passed.")
    if set(bool(value) for value in passing_exclude_values) != {False, True}:
        manual_items.append("Include/exclude 5700 ft robustness is incomplete.")

    manual_items.extend(
        str(item)
        for item in _as_list(refinement.get("manual_confirmation_items"))
        if str(item)
    )
    manual_items.extend(
        str(item) for item in _as_list(review.get("manual_confirmation_items")) if str(item)
    )
    manual_items = sorted(set(manual_items))

    no_final_labels = all(
        [
            _data_bool(refinement, "no_final_labels"),
            _data_bool(baseline, "no_final_labels"),
            _data_bool(review, "no_final_labels"),
        ]
    )
    no_forbidden = all(
        [
            _data_bool(refinement, "no_stc"),
            _data_bool(refinement, "no_apes"),
            _data_bool(refinement, "no_deep_learning"),
            _data_bool(refinement, "no_mvp4c"),
            _data_bool(baseline, "no_stc"),
            _data_bool(baseline, "no_apes"),
            _data_bool(baseline, "no_deep_learning"),
            _data_bool(baseline, "no_mvp4c"),
            _data_bool(review, "no_stc"),
            _data_bool(review, "no_apes"),
            _data_bool(review, "no_deep_learning"),
            _data_bool(review, "no_mvp4c"),
        ]
    )
    if not no_final_labels:
        blocking.append("guardrails: one or more inputs do not preserve no_final_labels.")
    if not no_forbidden:
        blocking.append(
            "guardrails: one or more inputs permit MVP-4C, STC, APES, or deep learning."
        )
    if refinement.get("production_training") is not False:
        blocking.append("guardrails: refinement report indicates production training.")

    decision = _decision(blocking, manual_items, warnings)
    return {
        "gate_version": DEPTH_LEVEL_REFINEMENT_GATE_VERSION,
        "stage": "MVP-4B-R4c",
        "task": "depth_level_refinement_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(path) for key, path in paths.items() if not key.startswith("output_")},
        "gate_thresholds": {
            "min_go_margin": MIN_GO_MARGIN,
            "min_conditional_margin": MIN_CONDITIONAL_MARGIN,
            "min_folds_above_permutation_fraction": MIN_FOLD_FRACTION,
            "min_passing_feature_groups": MIN_PASSING_FEATURE_GROUPS,
            "min_passing_confidence_thresholds": MIN_PASSING_THRESHOLDS,
            "min_passing_depth_block_splits": MIN_PASSING_SPLITS,
        },
        "evidence": {
            "baseline_best_margin": _as_float(
                _as_dict(baseline.get("best_result")).get("balanced_accuracy_margin")
            ),
            "refinement_recommendation": refinement.get("recommendation"),
            "best_feature_group": best.get("feature_group"),
            "best_confidence_threshold": best.get("confidence_threshold"),
            "best_exclude_5700_band": best.get("exclude_5700_band"),
            "best_n_splits": best.get("n_splits"),
            "best_model_type": best.get("model_type"),
            "real_balanced_accuracy_mean": best_real,
            "permutation_balanced_accuracy_mean": best_permutation,
            "margin_mean": best_margin,
            "margin_std": _as_float(best.get("margin_std")),
            "predicted_positive_rate": predicted_positive_rate,
            "folds_above_permutation_fraction": fold_fraction,
            "passing_scenario_count": refinement.get("passing_scenario_count"),
            "passing_feature_groups": passing_groups,
            "passing_confidence_thresholds": passing_thresholds,
            "passing_depth_block_splits": passing_splits,
            "passing_exclude_5700_values": passing_exclude_values,
            "depends_on_5700_band": robustness.get("depends_on_5700_band"),
            "suspicious_leakage": robustness.get("suspicious_leakage"),
            "review_figure_count": len(_as_dict(review.get("figures"))),
            "no_final_labels": no_final_labels,
            "no_stc_apes_deep_learning_mvp4c": no_forbidden,
        },
        "blocking_issues": blocking,
        "warnings": warnings,
        "decision": decision,
        "manual_confirmation_required": bool(manual_items),
        "manual_confirmation_items": manual_items,
        "next_branch_requires_human_approval": True,
        "depth_level_refinement_supported": decision in {"go", "conditional_go"},
        "mvp4c_allowed": False,
        "stc_allowed": False,
        "apes_allowed": False,
        "deep_learning_allowed": False,
        "final_labels_allowed": False,
        "production_model_allowed": False,
        "recommendation": _recommendation(decision, manual_items),
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


def format_depth_level_refinement_gate_markdown(report: dict[str, Any]) -> str:
    evidence = _as_dict(report.get("evidence"))
    lines = [
        "# MVP-4B-R4c Depth-Level Refinement Gate Report",
        "",
        "This gate reviews controlled depth-level weak-label candidate robustness. "
        "It does not authorize final labels, STC, APES, deep learning, production "
        "modeling, or MVP-4C.",
        "",
        "## Decision",
        "",
        f"- decision: `{report['decision']}`",
        f"- manual_confirmation_required: `{report['manual_confirmation_required']}`",
        "- next_branch_requires_human_approval: "
        f"`{report['next_branch_requires_human_approval']}`",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Evidence",
        "",
        f"- best_feature_group: `{evidence.get('best_feature_group')}`",
        f"- best_confidence_threshold: {evidence.get('best_confidence_threshold')}",
        f"- best_exclude_5700_band: {evidence.get('best_exclude_5700_band')}",
        f"- real_balanced_accuracy_mean: {evidence.get('real_balanced_accuracy_mean')}",
        "- permutation_balanced_accuracy_mean: "
        f"{evidence.get('permutation_balanced_accuracy_mean')}",
        f"- margin_mean: {evidence.get('margin_mean')}",
        f"- predicted_positive_rate: {evidence.get('predicted_positive_rate')}",
        "- folds_above_permutation_fraction: "
        f"{evidence.get('folds_above_permutation_fraction')}",
        f"- passing_feature_groups: {evidence.get('passing_feature_groups')}",
        f"- passing_confidence_thresholds: {evidence.get('passing_confidence_thresholds')}",
        f"- passing_depth_block_splits: {evidence.get('passing_depth_block_splits')}",
        f"- passing_exclude_5700_values: {evidence.get('passing_exclude_5700_values')}",
        "",
        "## Manual Confirmation Items",
        "",
    ]
    lines.extend(_message_lines(report["manual_confirmation_items"]))
    lines.extend(["", "## Blocking Issues", ""])
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
        raise DepthLevelRefinementGateError("data.reports is not configured.")
    return {
        "refinement_report": Path(
            args.refinement_report or reports / "depth_level_refinement_report_v001.json"
        ),
        "baseline_report": Path(
            args.baseline_report or reports / "depth_level_baseline_report_v001.json"
        ),
        "review_summary": Path(
            args.review_summary
            or reports
            / "depth_level_refinement_review_v001"
            / "depth_level_refinement_review_summary_v001.json"
        ),
        "output_report_md": Path(
            args.output_report_md or reports / "depth_level_refinement_gate_report.md"
        ),
        "output_report_json": Path(
            args.output_report_json or reports / "depth_level_refinement_gate_report.json"
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
        raise DepthLevelRefinementGateError(f"Required report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DepthLevelRefinementGateError(f"Report must be a JSON object: {path}")
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
        raise DepthLevelRefinementGateError(f"data.{key} is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DepthLevelRefinementGateError(
            f"Refusing to {action} depth-level refinement gate path outside data.{key}: {path}"
        ) from exc


def _decision(blocking: list[str], manual_items: list[str], warnings: list[str]) -> str:
    if blocking:
        return "no_go"
    if manual_items or warnings:
        return "conditional_go"
    return "go"


def _recommendation(decision: str, manual_items: list[str]) -> str:
    if decision == "no_go":
        return "prepare a manual label review pack before further feature work"
    if decision == "conditional_go":
        return "resolve manual confirmation items before opening any next branch"
    if manual_items:
        return "resolve manual confirmation items before opening any next branch"
    return (
        "generate a decision pack and wait for human approval; do not enter MVP-4C, "
        "STC/APES, deep learning, production modeling, or final-label workflows"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
