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

DECISION_VERSION = "mvp4x_rapid_decision_v001"
DECISION_OPTIONS = {
    "exploratory_signal_detected_continue_classical_modeling",
    "exploratory_signal_weak_request_controlled_time_frequency_review",
    "exploratory_signal_regime_dependent_request_stratified_study",
    "exploratory_signal_target_sensitive_request_target_view_study",
    "exploratory_signal_morphology_sensitive_request_label_redesign",
    "exploratory_signal_insufficient_stop",
    "stop_data_contract_issue",
    "stop_leakage_detected",
}
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}


class RapidDecisionCliError(RuntimeError):
    """Raised when MVP-4X rapid decision generation cannot run safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP-4X rapid decision report.")
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--snapshot-json", default=None)
    parser.add_argument("--existing-json", default=None)
    parser.add_argument("--waveform-json", default=None)
    parser.add_argument("--enhanced-json", default=None)
    parser.add_argument("--iteration-log", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        snapshot_json = _resolve_report_path(
            paths,
            args.snapshot_json,
            "mvp4x_research_snapshot_report_v001.json",
        )
        existing_json = _resolve_report_path(
            paths,
            args.existing_json,
            "mvp4x_existing_feature_baselines_v001.json",
        )
        waveform_json = _resolve_report_path(
            paths,
            args.waveform_json,
            "mvp4x_waveform_feature_report_v001.json",
        )
        enhanced_json = _resolve_report_path(
            paths,
            args.enhanced_json,
            "mvp4x_enhanced_feature_baselines_v001.json",
        )
        iteration_log = _resolve_report_path(
            paths,
            args.iteration_log,
            "mvp4x_rapid_iteration_log.md",
        )
        output_md = _resolve_report_path(paths, args.output_md, "mvp4x_rapid_decision.md")
        output_json = _resolve_report_path(paths, args.output_json, "mvp4x_rapid_decision.json")
        for path in (
            snapshot_json,
            existing_json,
            waveform_json,
            enhanced_json,
            iteration_log,
            output_md,
            output_json,
        ):
            _ensure_path_within(paths, path, action="read/write")
        decision = build_decision(
            snapshot=_read_json(snapshot_json),
            existing=_read_json(existing_json),
            waveform=_read_json(waveform_json),
            enhanced=_read_json(enhanced_json),
            iteration_log_text=iteration_log.read_text(encoding="utf-8")
            if iteration_log.exists()
            else "",
            inputs={
                "snapshot_json": str(snapshot_json),
                "existing_json": str(existing_json),
                "waveform_json": str(waveform_json),
                "enhanced_json": str(enhanced_json),
                "iteration_log": str(iteration_log),
            },
        )
        if decision["decision"] not in DECISION_OPTIONS:
            raise RapidDecisionCliError(f"Unsupported decision: {decision['decision']}")
        _write_decision_outputs(
            decision,
            output_md=output_md,
            output_json=output_json,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        RapidDecisionCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "MVP-4X rapid decision "
        f"decision={decision['decision']}; "
        f"iterations={decision['iteration_count']}; "
        f"research_only={decision['research_only']}; "
        f"no_final_labels={decision['no_final_labels']}."
    )
    print(f"Wrote decision JSON: {output_json}")
    print(f"Wrote decision Markdown: {output_md}")
    return 0


def build_decision(
    *,
    snapshot: dict[str, Any],
    existing: dict[str, Any],
    waveform: dict[str, Any],
    enhanced: dict[str, Any],
    iteration_log_text: str,
    inputs: dict[str, str],
) -> dict[str, Any]:
    errors = _as_list(snapshot.get("errors")) + _as_list(waveform.get("errors")) + _as_list(
        enhanced.get("errors")
    )
    leakage = bool(_as_dict(_as_dict(existing.get("decision")).get("best_result")).get("leakage"))
    leakage = leakage or bool(_as_dict(existing.get("decision")).get("leakage_detected"))
    for report in _as_dict(enhanced.get("sub_reports")).values():
        leakage = leakage or bool(
            _as_dict(_as_dict(report).get("decision")).get("leakage_detected")
        )
    existing_sufficient = bool(_as_dict(existing.get("decision")).get("baseline_sufficient"))
    waveform_triggered = bool(waveform.get("feature_version"))
    enhanced_best = enhanced.get("best_result")
    sklearn_available = bool(existing.get("sklearn_available")) or any(
        bool(_as_dict(report).get("sklearn_available"))
        for report in _as_dict(enhanced.get("sub_reports")).values()
    )
    if errors:
        decision = "stop_data_contract_issue"
    elif leakage:
        decision = "stop_leakage_detected"
    elif existing_sufficient or enhanced_best:
        decision = "exploratory_signal_detected_continue_classical_modeling"
    else:
        decision = "exploratory_signal_insufficient_stop"
    answers = {
        "1_existing_features_sufficient": existing_sufficient,
        "2_waveform_feature_extraction_triggered": waveform_triggered,
        "3_enhanced_waveform_features_improved_results": None
        if not enhanced_best
        else bool(enhanced_best),
        "4_best_model": None if not enhanced_best else enhanced_best.get("model"),
        "5_most_stable_target_view": _stable_target(existing, enhanced),
        "6_largest_feature_groups": _largest_feature_groups(enhanced),
        "7_regime_dependent_signal": "not_evaluated_no_completed_models",
        "8_depends_on_2400_2500_platform": "not_evaluated_no_completed_models",
        "9_depends_on_5680_special_band": "not_evaluated_no_completed_models",
        "10_depends_on_low_orientation_intervals": "not_evaluated_no_completed_models",
        "11_permutation_significantly_below_real": "not_evaluated_no_completed_models",
        "12_stable_folds": [],
        "13_worst_regime": "not_evaluated_no_completed_models",
        "14_morphology_next_stage_value": "audit_only_no_upgrade_without_completed_models",
        "15_stc_apes_deep_learning_discussion": "not_warranted_before_classical_baseline_runs",
        "16_human_confirmations_still_needed": _human_confirmations(sklearn_available),
        "17_research_only_marking_complete": _all_research_marked(
            snapshot,
            existing,
            waveform,
            enhanced,
        ),
    }
    return {
        "decision_version": DECISION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "inputs": inputs,
        "iteration_count": iteration_log_text.count("## iteration_"),
        "answers": answers,
        "quantitative_basis": {
            "snapshot": {
                "sample_count": snapshot.get("sample_count"),
                "feature_count": snapshot.get("feature_count"),
                "target_kernel": snapshot.get("target_kernel"),
            },
            "existing_baseline": {
                "sklearn_available": existing.get("sklearn_available"),
                "model_backend": existing.get("model_backend"),
                "decision": existing.get("decision"),
                "warnings": existing.get("warnings"),
            },
            "waveform_features": {
                "sample_count": waveform.get("sample_count"),
                "receiver_count": waveform.get("receiver_count"),
                "side_count": waveform.get("side_count"),
                "depth_feature_count": waveform.get("depth_feature_count"),
                "chunk_count": waveform.get("chunk_count"),
                "peak_memory_bytes": waveform.get("peak_memory_bytes"),
                "finite_ratio": waveform.get("finite_ratio"),
            },
            "enhanced_baseline": {
                "feature_sets": enhanced.get("feature_sets"),
                "best_result": enhanced.get("best_result"),
                "feature_group_ablation": enhanced.get("feature_group_ablation"),
                "permutation_importance": enhanced.get("permutation_importance"),
                "warnings": enhanced.get("warnings"),
            },
        },
        "next_minimal_recommendation": _next_recommendation(decision, sklearn_available),
        "not_authorized": [
            "production claim",
            "final labels",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "regime-specific model",
        ],
        **RESEARCH_FLAGS,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
    }


def format_decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Rapid Decision",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- decision: `{decision['decision']}`",
        f"- iteration_count: {decision['iteration_count']}",
        f"- next_minimal_recommendation: {decision['next_minimal_recommendation']}",
        "",
        "## Answers",
    ]
    for key, value in decision["answers"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Not Authorized"])
    lines.extend(f"- {item}" for item in decision["not_authorized"])
    lines.append("")
    return "\n".join(lines)


def _stable_target(existing: dict[str, Any], enhanced: dict[str, Any]) -> str | None:
    best = enhanced.get("best_result")
    if best:
        return str(_as_dict(best).get("target"))
    target_summaries = _as_dict(existing.get("target_summaries"))
    if "receiver_p90" in target_summaries:
        return "receiver_p90_primary_view_finite_but_model_stability_not_evaluated"
    return None


def _largest_feature_groups(enhanced: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    combined = _as_dict(_as_dict(enhanced.get("feature_sets")).get("combined_features"))
    groups = _as_dict(combined.get("group_counts"))
    rows = [{"group": key, "count": int(value)} for key, value in groups.items()]
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:limit]


def _human_confirmations(sklearn_available: bool) -> list[str]:
    confirmations = []
    if not sklearn_available:
        confirmations.append(
            "Confirm whether to run in an existing environment with scikit-learn available; "
            "no dependency was added in this research-only run."
        )
    confirmations.append(
        "Confirm no escalation to STC, APES, deep learning, final labels, or production claims."
    )
    return confirmations


def _all_research_marked(*reports: dict[str, Any]) -> bool:
    for report in reports:
        for key, expected in RESEARCH_FLAGS.items():
            if report.get(key) is not expected:
                return False
    return True


def _next_recommendation(decision: str, sklearn_available: bool) -> str:
    if decision == "exploratory_signal_insufficient_stop" and not sklearn_available:
        return (
            "Do not escalate modeling. Rerun Stage 2 and Stage 4 in an approved existing "
            "environment where scikit-learn is available, or explicitly approve dependency "
            "handling."
        )
    if decision == "exploratory_signal_detected_continue_classical_modeling":
        return (
            "Continue classical-model review only; do not advance to STC, APES, or deep "
            "learning."
        )
    if decision == "stop_leakage_detected":
        return "Stop and audit feature leakage before any further modeling."
    if decision == "stop_data_contract_issue":
        return "Stop and fix data/report contract errors before rerunning."
    return "Stop this rapid loop and keep outputs as research-only review artifacts."


def _write_decision_outputs(
    decision: dict[str, Any],
    *,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    for path in (output_md, output_json):
        _ensure_can_write(path, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_decision_markdown(decision), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report JSON does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Report JSON must contain an object: {path}")
    return data


def _resolve_report_path(config: dict[str, Any], override: str | None, filename: str) -> Path:
    if override:
        return Path(override)
    data = _as_dict(config.get("data"))
    reports = data.get("reports")
    if reports:
        return Path(str(reports)) / filename
    raise RapidDecisionCliError("data.reports is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, action: str) -> None:
    data = _as_dict(config.get("data"))
    root = Path(str(data.get("reports", ""))).resolve()
    if not str(root):
        raise RapidDecisionCliError("data.reports is not configured.")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise RapidDecisionCliError(
            f"Refusing to {action} rapid decision path outside data.reports: {path}"
        ) from exc


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
