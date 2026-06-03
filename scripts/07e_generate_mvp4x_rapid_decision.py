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
    existing_best = _as_dict(_as_dict(existing.get("decision")).get("best_result"))
    enhanced_best = _as_dict(enhanced.get("best_result"))
    best_basis = _best_basis(existing_best, enhanced_best)
    sklearn_available = bool(existing.get("sklearn_available")) or any(
        bool(_as_dict(report).get("sklearn_available"))
        for report in _as_dict(enhanced.get("sub_reports")).values()
    )
    has_real_fitted_model = bool(existing_best) or bool(enhanced_best)
    best_sub_report = _best_sub_report(enhanced, enhanced_best)
    best_target_key = str(enhanced_best.get("target_key") or existing_best.get("target_key") or "")
    best_model = str(enhanced_best.get("model") or existing_best.get("model") or "")
    signal_detected = existing_sufficient or _signal_detected(best_basis)
    stratified_signal = _stratified_signal(
        report=best_sub_report,
        target_key=best_target_key,
        model=best_model,
        enhanced=enhanced,
    )
    if errors:
        decision = "stop_data_contract_issue"
    elif leakage:
        decision = "stop_leakage_detected"
    elif signal_detected:
        decision = "exploratory_signal_detected_continue_classical_modeling"
    elif stratified_signal:
        decision = "exploratory_signal_regime_dependent_request_stratified_study"
    else:
        decision = "exploratory_signal_insufficient_stop"
    answers = {
        "f1_1_has_real_fitted_model": has_real_fitted_model,
        "f1_2_best_model": best_basis.get("model"),
        "f1_3_best_feature_set": best_basis.get("feature_set"),
        "f1_4_receiver_p90_stability": _target_stability("receiver_p90", existing, enhanced),
        "f1_5_reference_target_comparison": _reference_target_comparison(existing, enhanced),
        "f1_6_contiguous_3fold": _contiguous_summary(best_sub_report, best_target_key, best_model),
        "f1_7_leave_one_regime_out": _leave_one_summary(
            best_sub_report,
            best_target_key,
            best_model,
        ),
        "f1_8_permutation_below_real": _permutation_answer(best_basis),
        "f1_9_depends_on_2400_2500_platform": _sensitivity_answer(
            best_sub_report,
            best_target_key,
            best_model,
            "exclude_saturation_platform",
        ),
        "f1_10_depends_on_5680_special_band": _sensitivity_answer(
            best_sub_report,
            best_target_key,
            best_model,
            "exclude_5680_special_band",
        ),
        "f1_11_depends_on_low_orientation_intervals": _sensitivity_answer(
            best_sub_report,
            best_target_key,
            best_model,
            "exclude_low_orientation_confidence",
        ),
        "f1_12_waveform_features_improved_results": _waveform_improvement(
            existing_best,
            enhanced_best,
        ),
        "f1_13_top_stable_features": enhanced.get("top_10_stable_features", []),
        "f1_14_worst_regime": _worst_regime(enhanced),
        "f1_15_morphology_next_stage_value": (
            "audit_only_discuss_label_redesign_only_if_error_patterns_support_it"
        ),
        "f1_16_next_minimal_research_action": _next_recommendation(decision, sklearn_available),
        "f1_17_research_only_marking_complete": _all_research_marked(
            snapshot,
            existing,
            waveform,
            enhanced,
        ),
        "1_existing_features_sufficient": existing_sufficient,
        "2_waveform_feature_extraction_triggered": waveform_triggered,
        "2b_waveform_features_reused_without_stage3_rerun": waveform_triggered,
        "3_enhanced_waveform_features_improved_results": _waveform_improvement(
            existing_best,
            enhanced_best,
        ),
        "4_best_model": best_basis.get("model"),
        "4b_best_feature_set": best_basis.get("feature_set"),
        "5_most_stable_target_view": _stable_target(existing, enhanced),
        "6_largest_feature_groups": _largest_feature_groups(enhanced),
        "6b_top_contributing_feature_groups": _top_ablation_groups(enhanced),
        "7_regime_dependent_signal": _regime_signal(enhanced),
        "8_depends_on_2400_2500_platform": _sensitivity_answer(
            best_sub_report,
            best_target_key,
            best_model,
            "exclude_saturation_platform",
        ),
        "9_depends_on_5680_special_band": _sensitivity_answer(
            best_sub_report,
            best_target_key,
            best_model,
            "exclude_5680_special_band",
        ),
        "10_depends_on_low_orientation_intervals": _sensitivity_answer(
            best_sub_report,
            best_target_key,
            best_model,
            "exclude_low_orientation_confidence",
        ),
        "11_permutation_significantly_below_real": _permutation_answer(best_basis),
        "12_stable_folds": _stable_folds(best_sub_report, best_target_key, best_model),
        "13_worst_regime": _worst_regime(enhanced),
        "14_morphology_next_stage_value": "audit_only_not_a_model_feature_in_this_run",
        "15_stc_apes_deep_learning_discussion": _stc_apes_deep_learning_answer(decision),
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
                "top_10_stable_features": enhanced.get("top_10_stable_features"),
                "regime_specific_error_analysis": enhanced.get(
                    "regime_specific_error_analysis"
                ),
                "special_band_error_analysis": enhanced.get("special_band_error_analysis"),
                "low_orientation_error_analysis": enhanced.get(
                    "low_orientation_error_analysis"
                ),
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


def _best_basis(existing_best: dict[str, Any], enhanced_best: dict[str, Any]) -> dict[str, Any]:
    existing_s = _none_safe_float(existing_best.get("spearman"))
    enhanced_s = _none_safe_float(enhanced_best.get("spearman"))
    if enhanced_best and enhanced_s >= existing_s:
        return enhanced_best
    return existing_best


def _best_sub_report(enhanced: dict[str, Any], enhanced_best: dict[str, Any]) -> dict[str, Any]:
    feature_set = enhanced_best.get("feature_set")
    if not feature_set:
        return {}
    return _as_dict(_as_dict(enhanced.get("sub_reports")).get(str(feature_set)))


def _signal_detected(best: dict[str, Any]) -> bool:
    spearman = best.get("spearman")
    margin = best.get("real_minus_permutation_spearman")
    stable_folds = best.get("stable_positive_spearman_folds")
    if spearman is None or margin is None:
        return False
    return float(spearman) > 0.0 and float(margin) > 0.05 and int(stable_folds or 0) >= 2


def _stratified_signal(
    *,
    report: dict[str, Any],
    target_key: str,
    model: str,
    enhanced: dict[str, Any],
) -> bool:
    if not report or not target_key or not model:
        return False
    filters = (
        "exclude_saturation_platform",
        "exclude_5680_special_band",
        "exclude_low_orientation_confidence",
        "exclude_all_special_flags",
    )
    for filter_name in filters:
        answer = _sensitivity_answer(report, target_key, model, filter_name)
        filtered_s = answer.get("filtered_spearman")
        if filtered_s is not None and float(filtered_s) > 0.10:
            return True
    regime = _regime_signal(enhanced)
    return bool(regime.get("regime_dependent"))


def _target_stability(
    target: str,
    existing: dict[str, Any],
    enhanced: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for report in [existing, *_as_dict(enhanced.get("sub_reports")).values()]:
        for target_key, by_model in _as_dict(report.get("contiguous_cv")).items():
            if not str(target_key).startswith(f"{target}:"):
                continue
            for model, summary in _as_dict(by_model).items():
                aggregate = _as_dict(_as_dict(summary).get("aggregate"))
                if aggregate.get("spearman") is None:
                    continue
                candidates.append(
                    {
                        "feature_set": _as_dict(report).get("feature_set_name"),
                        "target_key": target_key,
                        "model": model,
                        "spearman": aggregate.get("spearman"),
                        "stable_positive_spearman_folds": _as_dict(summary).get(
                            "stable_positive_spearman_folds"
                        ),
                    }
                )
    candidates.sort(key=lambda row: _none_safe_float(row.get("spearman")), reverse=True)
    return {"status": "evaluated", "best": candidates[0] if candidates else None}


def _reference_target_comparison(
    existing: dict[str, Any],
    enhanced: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for target in ("receiver_mean", "receiver_max"):
        output[target] = _target_stability(target, existing, enhanced).get("best")
    return output


def _contiguous_summary(
    report: dict[str, Any],
    target_key: str,
    model: str,
) -> dict[str, Any]:
    summary = _as_dict(_as_dict(_as_dict(report.get("contiguous_cv")).get(target_key)).get(model))
    if not summary:
        return {"status": "not_available"}
    return {
        "status": summary.get("status"),
        "fold_count": summary.get("fold_count"),
        "stable_positive_spearman_folds": summary.get("stable_positive_spearman_folds"),
        "aggregate": summary.get("aggregate"),
        "folds": summary.get("folds"),
    }


def _leave_one_summary(
    report: dict[str, Any],
    target_key: str,
    model: str,
) -> dict[str, Any]:
    results = _as_dict(
        _as_dict(_as_dict(report.get("leave_one_regime_out")).get(target_key)).get(model)
    )
    if not results:
        return {"status": "not_available"}
    completed = [
        {**_as_dict(value), "split": key}
        for key, value in results.items()
        if _as_dict(value).get("status") == "completed"
    ]
    completed.sort(key=lambda row: _none_safe_float(row.get("mae")), reverse=True)
    return {
        "status": "completed",
        "splits": results,
        "worst_by_mae": completed[0] if completed else None,
    }


def _permutation_answer(best: dict[str, Any]) -> dict[str, Any]:
    margin = best.get("real_minus_permutation_spearman")
    return {
        "status": "evaluated" if margin is not None else "not_available",
        "real_minus_permutation_spearman": margin,
        "significantly_below_real": None if margin is None else float(margin) > 0.05,
    }


def _sensitivity_answer(
    report: dict[str, Any],
    target_key: str,
    model: str,
    filter_name: str,
) -> dict[str, Any]:
    sensitivity = _as_dict(_as_dict(_as_dict(report.get("sensitivity")).get(target_key)).get(model))
    include = _as_dict(_as_dict(sensitivity.get("include_all")).get("aggregate"))
    filtered = _as_dict(_as_dict(sensitivity.get(filter_name)).get("aggregate"))
    include_s = include.get("spearman")
    filtered_s = filtered.get("spearman")
    if include_s is None or filtered_s is None:
        return {"status": "not_available", "filter": filter_name}
    delta = float(filtered_s) - float(include_s)
    return {
        "status": "evaluated",
        "filter": filter_name,
        "include_all_spearman": include_s,
        "filtered_spearman": filtered_s,
        "filtered_minus_include_spearman": delta,
        "filtered_still_nonpositive": float(filtered_s) <= 0.0,
        "dependency_flag": abs(delta) > 0.05,
    }


def _waveform_improvement(
    existing_best: dict[str, Any],
    enhanced_best: dict[str, Any],
) -> dict[str, Any]:
    existing_s = existing_best.get("spearman")
    enhanced_s = enhanced_best.get("spearman")
    feature_set = enhanced_best.get("feature_set")
    if existing_s is None or enhanced_s is None:
        return {"status": "not_available"}
    delta = float(enhanced_s) - float(existing_s)
    return {
        "status": "evaluated",
        "best_enhanced_feature_set": feature_set,
        "existing_best_spearman": existing_s,
        "enhanced_best_spearman": enhanced_s,
        "enhanced_minus_existing_spearman": delta,
        "waveform_improved": bool(feature_set != "existing_features_only" and delta > 0.0),
    }


def _top_ablation_groups(enhanced: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    results = _as_list(_as_dict(enhanced.get("feature_group_ablation")).get("results"))
    rows = [row for row in results if isinstance(row, dict)]
    rows.sort(key=lambda row: _none_safe_float(row.get("spearman_loss_vs_full")), reverse=True)
    return rows[:limit]


def _regime_signal(enhanced: dict[str, Any]) -> dict[str, Any]:
    analysis = _as_dict(enhanced.get("regime_specific_error_analysis"))
    groups = _as_dict(analysis.get("groups"))
    if not groups:
        return {"status": "not_available"}
    spearman_values = [
        float(_as_dict(value).get("spearman"))
        for value in groups.values()
        if _as_dict(value).get("spearman") is not None
    ]
    spread = None if not spearman_values else max(spearman_values) - min(spearman_values)
    return {
        "status": "evaluated",
        "spearman_spread": spread,
        "regime_dependent": None if spread is None else spread > 0.10,
        "worst_by_mae": analysis.get("worst_by_mae"),
    }


def _stable_folds(report: dict[str, Any], target_key: str, model: str) -> list[dict[str, Any]]:
    folds = _as_list(_contiguous_summary(report, target_key, model).get("folds"))
    return [
        fold
        for fold in folds
        if isinstance(fold, dict)
        and fold.get("spearman") is not None
        and float(fold["spearman"]) > 0.0
    ]


def _worst_regime(enhanced: dict[str, Any]) -> Any:
    return _as_dict(enhanced.get("regime_specific_error_analysis")).get("worst_by_mae")


def _stc_apes_deep_learning_answer(decision: str) -> str:
    if decision == "exploratory_signal_detected_continue_classical_modeling":
        return "not_next_step_continue_classical_feature_review_before_STC_APES_or_deep_learning"
    return "not_warranted_before_stable_classical_signal"


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
    if decision == "exploratory_signal_regime_dependent_request_stratified_study":
        return (
            "Run a bounded stratified audit of low-orientation, special-band, and broad-regime "
            "subgroups; do not train regime-specific models yet."
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


def _none_safe_float(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
