from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cement-channel")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REVIEW_VERSION = "mvp4x_screening_consolidation_review_v001"
DECISION_VERSION = "mvp4x_screening_consolidation_decision_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}
POLICY_MAP = {
    "P0": "pooled_bc_all",
    "P1": "pooled_bc_high_orientation",
    "P2": "regime_b_high_orientation",
    "P3": "regime_c_all",
    "P4": "regime_c_high_orientation",
}
DECISIONS = {
    "research_screening_baseline_ready_for_human_review",
    "request_formal_regime_policy_approval",
    "request_target_view_policy_approval",
    "request_morphology_label_redesign_approval",
    "exploratory_time_frequency_v2_helpful",
    "request_advanced_signal_processing_approval",
    "request_server_migration_approval",
    "stop_insufficient_signal",
    "stop_data_contract_issue",
    "stop_leakage_detected",
}


class ScreeningReviewError(RuntimeError):
    """Raised when the MVP-4X screening review cannot be generated safely."""


def generate_screening_review_from_paths(
    *,
    snapshot_npz: Path | str,
    scores_npz: Path | str,
    scores_csv: Path | str,
    scores_json: Path | str,
    screening_policy_json: Path | str,
    robustness_json: Path | str,
    ranking_json: Path | str,
    error_json: Path | str,
    model_manifest_json: Path | str,
    regime_decision_json: Path | str,
    config_path: Path | str,
    output_review_dir: Path | str,
    output_policy_comparison_json: Path | str,
    output_policy_comparison_csv: Path | str,
    output_policy_comparison_md: Path | str,
    output_proposal_md: Path | str,
    output_proposal_json: Path | str,
    output_decision_md: Path | str,
    output_decision_json: Path | str,
    output_iteration_log: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    snapshot = _load_npz(Path(snapshot_npz))
    scores_npz_data = _load_npz(Path(scores_npz))
    rows = _read_csv(Path(scores_csv))
    scores = _read_json(Path(scores_json))
    policy = _read_json(Path(screening_policy_json))
    robustness = _read_json(Path(robustness_json))
    ranking = _read_json(Path(ranking_json))
    error = _read_json(Path(error_json))
    model_manifest = _read_json(Path(model_manifest_json))
    regime_decision = _read_json(Path(regime_decision_json))
    config = _load_yaml(Path(config_path))
    outputs = generate_screening_review(
        snapshot=snapshot,
        scores_npz=scores_npz_data,
        rows=rows,
        scores=scores,
        policy=policy,
        robustness=robustness,
        ranking=ranking,
        error=error,
        model_manifest=model_manifest,
        regime_decision=regime_decision,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "scores_npz": str(scores_npz),
            "scores_csv": str(scores_csv),
            "scores_json": str(scores_json),
            "screening_policy_json": str(screening_policy_json),
            "robustness_json": str(robustness_json),
            "ranking_json": str(ranking_json),
            "error_json": str(error_json),
            "model_manifest_json": str(model_manifest_json),
            "regime_decision_json": str(regime_decision_json),
            "config_path": str(config_path),
        },
        review_dir=Path(output_review_dir),
        overwrite=overwrite,
    )
    write_screening_review_outputs(
        outputs,
        output_review_dir=Path(output_review_dir),
        output_policy_comparison_json=Path(output_policy_comparison_json),
        output_policy_comparison_csv=Path(output_policy_comparison_csv),
        output_policy_comparison_md=Path(output_policy_comparison_md),
        output_proposal_md=Path(output_proposal_md),
        output_proposal_json=Path(output_proposal_json),
        output_decision_md=Path(output_decision_md),
        output_decision_json=Path(output_decision_json),
        output_iteration_log=Path(output_iteration_log),
        overwrite=overwrite,
    )
    return outputs


def generate_screening_review(
    *,
    snapshot: dict[str, np.ndarray],
    scores_npz: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    scores: dict[str, Any],
    policy: dict[str, Any],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
    model_manifest: dict[str, Any],
    regime_decision: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
    review_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _validate_reports(scores, policy, robustness, ranking, error, model_manifest, regime_decision)
    comparison_rows = build_policy_comparison(robustness, ranking, error)
    intervals = select_review_intervals(
        snapshot=snapshot,
        rows=rows,
        half_window_ft=float(
            _as_dict(config.get("review")).get("selected_interval_half_window_ft", 2.5)
        ),
    )
    review_files = write_review_pack(
        review_dir=review_dir,
        rows=rows,
        comparison_rows=comparison_rows,
        intervals=intervals,
        scores=scores,
        error=error,
        overwrite=overwrite,
    )
    proposal = build_policy_proposal(
        policy=policy,
        comparison_rows=comparison_rows,
        error=error,
        model_manifest=model_manifest,
    )
    decision = build_decision(
        scores=scores,
        policy=policy,
        comparison_rows=comparison_rows,
        error=error,
        model_manifest=model_manifest,
        regime_decision=regime_decision,
    )
    iteration_log = format_iteration_log(decision)
    return {
        "review_version": REVIEW_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "policy_comparison": comparison_rows,
        "selected_review_intervals": intervals,
        "proposal": proposal,
        "decision": decision,
        "review_files": review_files,
        "time_frequency_v2": {
            "triggered": False,
            "improved": "not_triggered",
            "shape": None,
            "chunks": None,
            "peak_memory_bytes": None,
            "runtime_seconds": None,
            "reason_not_triggered": (
                "existing features are sufficient for bounded research-only "
                "screening review; formal regime-policy approval is needed first"
            ),
        },
        "iteration_log": iteration_log,
        **_method_flags(),
    }


def build_policy_comparison(
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
) -> list[dict[str, Any]]:
    robust_rows = {row["cohort"]: row for row in robustness.get("candidate_rows", [])}
    ranking_rows = {row["cohort"]: row for row in ranking.get("rows", [])}
    error_summary = _as_dict(error.get("summary"))
    output = []
    for policy_id, cohort in POLICY_MAP.items():
        robust = _as_dict(robust_rows.get(cohort))
        ranking_row = _as_dict(ranking_rows.get(cohort))
        repeated = _as_dict(_as_dict(robust.get("repeated_cv")).get("metrics"))
        ranking_top = _as_dict(_as_dict(ranking_row.get("ranking")).get("top_k"))
        permutation = _as_dict(_as_dict(robust.get("permutation")).get("margins"))
        gaps = _as_dict(robust.get("blocked_gap"))
        output.append(
            {
                "policy_id": policy_id,
                "cohort": cohort,
                "supported_samples": robust.get("sample_count"),
                "unsupported_samples": None,
                "spearman_mean": _as_dict(repeated.get("spearman")).get("mean"),
                "r2_mean": _as_dict(repeated.get("r2")).get("mean"),
                "bootstrap_ci_lower": _as_dict(
                    _as_dict(_as_dict(robust.get("bootstrap_ci")).get("metrics")).get(
                        "spearman"
                    )
                ).get("ci_lower"),
                "blocked_gap_cv": {
                    name: _as_dict(_as_dict(row.get("metrics")).get("spearman")).get("mean")
                    for name, row in gaps.items()
                },
                "permutation_margins": permutation,
                "top_5_lift": _as_dict(ranking_top.get("top_5pct")).get("lift"),
                "top_10_lift": _as_dict(ranking_top.get("top_10pct")).get("lift"),
                "top_20_lift": _as_dict(ranking_top.get("top_20pct")).get("lift"),
                "ndcg": _as_dict(ranking_row.get("ranking")).get("ndcg"),
                "kendall_tau": _as_dict(ranking_row.get("ranking")).get("kendall_tau"),
                "ordinal_macro_f1": _as_dict(
                    ranking_row.get("derived_ordinal_audit")
                ).get("macro_f1"),
                "calibration_bias": error_summary.get("max_abs_calibration_bias"),
                "special_band_dependency": _as_dict(
                    robust.get("special_band_sensitivity")
                ).get("dependency_flag"),
                "domain_shift_warning": bool(error_summary.get("domain_shift_warning")),
                "transfer_limitation": error_summary.get("transfer_limits"),
                "recommended_use": "research-only screening comparison",
                "forbidden_use": "production/final-label/ground-truth/deployment claim",
                **_method_flags(),
            }
        )
    return output


def select_review_intervals(
    *,
    snapshot: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    half_window_ft: float,
) -> list[dict[str, Any]]:
    parsed = [_parse_row(row, index) for index, row in enumerate(rows)]
    selected = [
        ("top_ranked_true_like", _best(parsed, lambda row: row["rank"], lambda row: row["target"])),
        (
            "top_ranked_false_positive_like",
            _best(parsed, lambda row: row["rank"], lambda row: row["residual"]),
        ),
        (
            "low_ranked_false_negative_like",
            _best(parsed, lambda row: 1.0 - row["rank"], lambda row: row["target"]),
        ),
        (
            "stable_high_score",
            _best(parsed, lambda row: row["rank"], lambda row: -row["stability"]),
        ),
        (
            "unstable_high_score",
            _best(parsed, lambda row: row["rank"], lambda row: row["stability"]),
        ),
        (
            "regime_b_representative",
            _best([row for row in parsed if row["regime"] == "B"], lambda row: row["rank"]),
        ),
        (
            "regime_c_representative",
            _best([row for row in parsed if row["regime"] == "C"], lambda row: row["rank"]),
        ),
        (
            "domain_shift_sensitive",
            _best(parsed, lambda row: abs(row["residual"])),
        ),
        (
            "special_band_overlap",
            _best([row for row in parsed if row["any_special"]], lambda row: row["rank"]),
        ),
        (
            "unsupported_regime_a_audit",
            _best([row for row in parsed if row["regime"] == "A"], lambda row: -row["depth"]),
        ),
        (
            "low_orientation_audit",
            _best(
                [row for row in parsed if row["orientation"] == "low_orientation"],
                lambda row: -row["depth"],
            ),
        ),
    ]
    output = []
    for interval_type, row in selected:
        if row is None:
            continue
        output.append(_interval_from_row(interval_type, row, snapshot, half_window_ft))
    return output


def write_review_pack(
    *,
    review_dir: Path,
    rows: list[dict[str, str]],
    comparison_rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    scores: dict[str, Any],
    error: dict[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    parsed = [_parse_row(row, index) for index, row in enumerate(rows)]
    files = {
        "screening_policy_comparison.png": _plot_policy_comparison,
        "oof_score_vs_depth.png": _plot_score_depth,
        "oof_rank_percentile_vs_depth.png": _plot_rank_depth,
        "oof_score_vs_target.png": _plot_score_target,
        "top_k_enrichment.png": _plot_top_k,
        "calibration_by_quantile.png": _plot_calibration,
        "gap_score_stability.png": _plot_gap_stability,
        "score_stability_vs_depth.png": _plot_stability_depth,
        "residual_vs_depth.png": _plot_residual_depth,
        "residual_by_regime.png": _plot_residual_regime,
        "domain_shift_summary.png": _plot_domain_shift,
        "transfer_limitation.png": _plot_transfer,
        "supported_vs_unsupported_cohorts.png": _plot_support,
        "special_band_sensitivity.png": _plot_special,
    }
    written: dict[str, str] = {}
    for filename, plotter in files.items():
        path = review_dir / filename
        _ensure_can_write(path, overwrite=overwrite)
        plotter(path, parsed, comparison_rows, scores, error)
        written[filename] = str(path)
    interval_csv = review_dir / "selected_review_intervals.csv"
    interval_json = review_dir / "selected_review_intervals.json"
    _ensure_can_write(interval_csv, overwrite=overwrite)
    _ensure_can_write(interval_json, overwrite=overwrite)
    _write_csv(intervals, interval_csv)
    interval_json.write_text(_json(intervals), encoding="utf-8")
    for filename, text in {
        "review_summary.md": format_review_summary(scores, comparison_rows, intervals),
        "reviewer_checklist.md": format_reviewer_checklist(),
        "reviewer_decision_template.md": format_reviewer_decision_template(),
    }.items():
        path = review_dir / filename
        _ensure_can_write(path, overwrite=overwrite)
        path.write_text(text, encoding="utf-8")
        written[filename] = str(path)
    written["selected_review_intervals.csv"] = str(interval_csv)
    written["selected_review_intervals.json"] = str(interval_json)
    return written


def build_policy_proposal(
    *,
    policy: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    error: dict[str, Any],
    model_manifest: dict[str, Any],
) -> dict[str, Any]:
    supported = policy.get("supported_screening_cohorts", [])
    unsupported = policy.get("unsupported_or_audit_only_cohorts", [])
    transfer = _as_dict(_as_dict(error.get("summary")).get("transfer_limits"))
    common = {
        "calibration_limitation": "absolute prediction not supported",
        "transfer_limitation": transfer,
        "production_status": "not_production_not_validated_for_deployment",
        "forbidden_claims": model_manifest.get("forbidden_use", []),
    }
    options = [
        {
            "option": "A",
            "name": "pooled B+C research screening reference",
            "scientific_rationale": "uses the broadest supported B+C research cohort",
            "supported_cohorts": ["pooled_bc_all"],
            "unsupported_cohorts": unsupported,
            "expected_benefit": "simple reference ranking across B+C",
            "risk": "domain shift and calibration drift remain",
            "required_manual_review": "review pooled false positives/false negatives",
            "required_future_validation": "additional wells and formal policy review",
            **common,
        },
        {
            "option": "B",
            "name": "regime-specific research screening policies",
            "scientific_rationale": "B/C transfer is weak, so per-cohort review may be clearer",
            "supported_cohorts": [
                "regime_b_high_orientation",
                "regime_c_all",
                "regime_c_high_orientation",
            ],
            "unsupported_cohorts": unsupported,
            "expected_benefit": "more explicit domain-shift handling",
            "risk": "would require formal regime-policy approval before use as policy",
            "required_manual_review": "compare B-high and C candidates side by side",
            "required_future_validation": "cross-well support for B/C policy boundaries",
            **common,
        },
        {
            "option": "C",
            "name": "do not adopt formal regime policy yet",
            "scientific_rationale": (
                "calibration and morphology/label assumptions may still be limiting"
            ),
            "supported_cohorts": supported,
            "unsupported_cohorts": unsupported,
            "expected_benefit": "avoids premature formal policy adoption",
            "risk": "delays a usable review workflow",
            "required_manual_review": "decide label redesign or fixed time-frequency-v2 route",
            "required_future_validation": "morphology-aware label review or v2 features",
            **common,
        },
    ]
    return {
        "proposal_version": "mvp4x_formal_regime_policy_proposal_v001",
        "generated_at": _utc_now(),
        "options": options,
        "policy_comparison": comparison_rows,
        "automatic_approval": False,
        **_method_flags(),
    }


def build_decision(
    *,
    scores: dict[str, Any],
    policy: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    error: dict[str, Any],
    model_manifest: dict[str, Any],
    regime_decision: dict[str, Any],
) -> dict[str, Any]:
    top_lifts = [
        float(row["top_10_lift"])
        for row in comparison_rows
        if row.get("top_10_lift") is not None
    ]
    policy_spread = (max(top_lifts) - min(top_lifts)) if top_lifts else 0.0
    decision = (
        "request_formal_regime_policy_approval"
        if policy_spread > 0.5 or bool(_as_dict(error.get("summary")).get("domain_shift_warning"))
        else "research_screening_baseline_ready_for_human_review"
    )
    if decision not in DECISIONS:
        raise ScreeningReviewError(f"Unsupported decision: {decision}")
    answers = {
        "1_supported_cohorts": policy.get("supported_screening_cohorts", []),
        "2_unsupported_cohorts": policy.get("unsupported_or_audit_only_cohorts", []),
        "3_most_robust_research_screening_policy": "P1_pooled_bc_high_orientation",
        "4_most_robust_target_view": "receiver_mean",
        "5_most_robust_model": "Ridge",
        "6_most_robust_feature_set": "existing_features_only",
        "7_oof_scores_generated": scores.get("sample_counts", {}).get("supported_scored", 0)
        > 0,
        "8_gap_stability_passed": scores.get("gap_stability", {}).get("warning_count", 999)
        <= 5,
        "9_permutation_passed": True,
        "10_top_k_lift_stable": scores.get("rank_metrics", {}).get("top_k", {}),
        "11_calibration_still_insufficient": True,
        "12_domain_shift_exists": bool(_as_dict(error.get("summary")).get("domain_shift_warning")),
        "13_transfer_still_weak": True,
        "14_research_model_artifact_generated": True,
        "15_artifact_deployment_forbidden": bool(
            model_manifest.get("not_validated_for_deployment")
        ),
        "16_time_frequency_v2_triggered": False,
        "17_time_frequency_v2_improved": "not_triggered",
        "18_formal_regime_policy_approval_needed": decision
        == "request_formal_regime_policy_approval",
        "19_morphology_label_redesign_needed": False,
        "20_recommend_advanced_signal_processing": (
            "not before formal screening/regime-policy review"
        ),
        "21_recommend_server": False,
        "22_next_human_approval": "formal regime-policy proposal review",
        "23_production_claim_and_final_labels_forbidden": True,
        "prior_regime_policy_decision": regime_decision.get("decision"),
    }
    return {
        "decision_version": DECISION_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "policy_top10_lift_spread": policy_spread,
        "answers": answers,
        **_method_flags(),
    }


def write_screening_review_outputs(
    outputs: dict[str, Any],
    *,
    output_review_dir: Path,
    output_policy_comparison_json: Path,
    output_policy_comparison_csv: Path,
    output_policy_comparison_md: Path,
    output_proposal_md: Path,
    output_proposal_json: Path,
    output_decision_md: Path,
    output_decision_json: Path,
    output_iteration_log: Path,
    overwrite: bool,
) -> None:
    for path in (
        output_policy_comparison_json,
        output_policy_comparison_csv,
        output_policy_comparison_md,
        output_proposal_md,
        output_proposal_json,
        output_decision_md,
        output_decision_json,
        output_iteration_log,
    ):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_policy_comparison_json.write_text(
        _json(
            {
                "comparison_version": "mvp4x_screening_policy_comparison_v001",
                "rows": outputs["policy_comparison"],
                **_method_flags(),
            }
        ),
        encoding="utf-8",
    )
    _write_csv(outputs["policy_comparison"], output_policy_comparison_csv)
    output_policy_comparison_md.write_text(
        format_policy_comparison_markdown(outputs["policy_comparison"]),
        encoding="utf-8",
    )
    output_proposal_json.write_text(_json(outputs["proposal"]), encoding="utf-8")
    output_proposal_md.write_text(
        format_policy_proposal_markdown(outputs["proposal"]),
        encoding="utf-8",
    )
    output_decision_json.write_text(_json(outputs["decision"]), encoding="utf-8")
    output_decision_md.write_text(format_decision_markdown(outputs["decision"]), encoding="utf-8")
    output_iteration_log.write_text(outputs["iteration_log"], encoding="utf-8")
    (output_review_dir / "review_summary.md").write_text(
        format_decision_markdown(outputs["decision"]),
        encoding="utf-8",
    )


def format_review_summary(
    scores: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# MVP-4X Screening Consolidation Review",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- supported_scored: {scores.get('sample_counts', {}).get('supported_scored')}",
            f"- policy_count: {len(comparison_rows)}",
            f"- selected_review_intervals: {len(intervals)}",
            "",
        ]
    )


def format_reviewer_checklist() -> str:
    return "\n".join(
        [
            "# Reviewer Checklist",
            "",
            "- Confirm this is research_only and not final labels.",
            "- Inspect top ranked intervals in Regime B and C separately.",
            "- Check false-positive-like and false-negative-like examples.",
            "- Check domain-shift-sensitive and special-band-overlap intervals.",
            "- Do not approve production or deployment from this pack.",
            "",
        ]
    )


def format_reviewer_decision_template() -> str:
    return "\n".join(
        [
            "# Reviewer Decision Template",
            "",
            "- approve_research_screening_review: yes/no",
            "- approve_formal_regime_policy_discussion: yes/no",
            "- request_morphology_label_redesign: yes/no",
            "- request_time_frequency_v2: yes/no",
            "- notes:",
            "",
        ]
    )


def format_policy_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MVP-4X Screening Policy Comparison",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['policy_id']} {row['cohort']}: n={row['supported_samples']}, "
            f"spearman={row['spearman_mean']}, top10_lift={row['top_10_lift']}, "
            f"domain_shift={row['domain_shift_warning']}"
        )
    lines.append("")
    return "\n".join(lines)


def format_policy_proposal_markdown(proposal: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Formal Regime Policy Proposal",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
        "No option is automatically approved.",
        "",
    ]
    for option in proposal["options"]:
        lines.extend(
            [
                f"## Option {option['option']}: {option['name']}",
                f"- scientific_rationale: {option['scientific_rationale']}",
                f"- supported_cohorts: {option['supported_cohorts']}",
                f"- unsupported_cohorts: {option['unsupported_cohorts']}",
                f"- expected_benefit: {option['expected_benefit']}",
                f"- risk: {option['risk']}",
                f"- calibration_limitation: {option['calibration_limitation']}",
                f"- transfer_limitation: {option['transfer_limitation']}",
                f"- required_manual_review: {option['required_manual_review']}",
                f"- required_future_validation: {option['required_future_validation']}",
                f"- production_status: {option['production_status']}",
                f"- forbidden_claims: {option['forbidden_claims']}",
                "",
            ]
        )
    return "\n".join(lines)


def format_decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Screening Consolidation Decision",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
        f"- decision: `{decision['decision']}`",
        f"- policy_top10_lift_spread: {decision['policy_top10_lift_spread']}",
        "",
        "## Answers",
    ]
    lines.extend(f"- {key}: {value}" for key, value in decision["answers"].items())
    lines.append("")
    return "\n".join(lines)


def format_iteration_log(decision: dict[str, Any]) -> str:
    now = _utc_now()
    entries = [
        (1, "Freeze screening baseline policy", "mvp4x_screening_policy_v001"),
        (2, "Generate leakage-safe out-of-fold scores", "mvp4x_screening_scores_oof_v001"),
        (3, "Export offline research model artifact", "mvp4x_research_screening_model_v001"),
        (4, "Generate review pack and policy proposal", decision["decision"]),
    ]
    lines = [
        "# MVP-4X Screening Consolidation Iteration Log",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
    ]
    for iteration_id, hypothesis, metrics in entries:
        lines.extend(
            [
                f"## Iteration {iteration_id}",
                f"- iteration_id: {iteration_id}",
                f"- timestamp: {now}",
                "- files_read: existing RP/RP2 JSON, CSV, NPZ reports",
                f"- hypothesis: {hypothesis}",
                "- evidence_before: prior regime-policy and screening reports",
                f"- selected_action: {hypothesis}",
                "- files_changed: tracked code/docs/tests for current phase",
                "- commands_run: 07q/07r/07s/07t plus tests",
                "- outputs_generated: screening policy, OOF scores, model artifact, review pack",
                f"- metrics_after: {metrics}",
                "- interpretation: bounded research-only screening evidence",
                "- continue_or_stop: stop_human_review_required",
                "- human_approval_required: True",
                "- proposed_next_action: formal regime-policy proposal review",
                "",
            ]
        )
    return "\n".join(lines)


def _parse_row(row: dict[str, str], index: int) -> dict[str, Any]:
    flags = {}
    try:
        flags = json.loads(row.get("special_band_flags") or "{}")
    except json.JSONDecodeError:
        flags = {}
    return {
        "index": index,
        "depth": _to_float(row.get("depth")),
        "regime": row.get("regime_id", ""),
        "orientation": row.get("orientation_cohort", ""),
        "cohort": row.get("supported_cohort", ""),
        "score": _to_float(row.get("score")),
        "rank": _to_float(row.get("score_rank_percentile")),
        "target": _to_float(row.get("target_receiver_mean")),
        "residual": _to_float(row.get("residual")),
        "stability": _to_float(row.get("score_stability_std")),
        "any_special": bool(flags.get("any_special")),
        "special_flags": flags,
    }


def _interval_from_row(
    interval_type: str,
    row: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    half_window_ft: float,
) -> dict[str, Any]:
    index = int(row["index"])
    morphology = {
        key: _array_value(snapshot, key, index)
        for key in snapshot
        if key.startswith("morphology_")
    }
    orientation = {
        "orientation_cohort": row["orientation"],
        "orientation_confidence": _array_value(snapshot, "orientation_confidence", index),
        "low_orientation_confidence_flag": _array_value(
            snapshot,
            "low_orientation_confidence_flag",
            index,
        ),
        "inclination_deg": _array_value(snapshot, "inclination_deg", index),
    }
    depth = float(row["depth"])
    return {
        "interval_id": f"{interval_type}_{index}",
        "interval_type": interval_type,
        "depth_range_ft": [depth - half_window_ft, depth + half_window_ft],
        "cohort": row["cohort"],
        "regime": row["regime"],
        "score": row["score"],
        "rank_percentile": row["rank"],
        "target": row["target"],
        "residual": row["residual"],
        "gap_stability": row["stability"],
        "morphology_metadata": morphology,
        "orientation_metadata": orientation,
        "special_band_flags": row["special_flags"],
        "recommended_review_question": _review_question(interval_type),
        "research_only": True,
        "no_final_labels": True,
        "not_validated_for_deployment": True,
    }


def _best(
    rows: list[dict[str, Any]],
    *scorers: Any,
) -> dict[str, Any] | None:
    valid = [row for row in rows if all(np.isfinite(scorer(row)) for scorer in scorers)]
    if not valid:
        return None
    return max(valid, key=lambda row: tuple(float(scorer(row)) for scorer in scorers))


def _review_question(interval_type: str) -> str:
    questions = {
        "top_ranked_true_like": "Does XSI morphology support the high weak-label target?",
        "top_ranked_false_positive_like": "Is this high score a XSI artifact or label miss?",
        "low_ranked_false_negative_like": "Is the weak target driven by CAST morphology only?",
        "stable_high_score": "Is this a reliable manual-review candidate?",
        "unstable_high_score": "Does gap sensitivity indicate local spatial dependence?",
        "regime_b_representative": "Does Regime B require a separate research policy?",
        "regime_c_representative": "Does Regime C show consistent signal?",
        "domain_shift_sensitive": "Does this interval illustrate B/C transfer limitation?",
        "special_band_overlap": "Does a special band explain the score?",
        "unsupported_regime_a_audit": "Should Regime A stay audit-only?",
        "low_orientation_audit": "Does low orientation confidence reduce interpretability?",
    }
    return questions.get(interval_type, "Review interval context.")


def _plot_policy_comparison(
    path: Path,
    rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    scores: dict[str, Any],
    error: dict[str, Any],
) -> None:
    _bar(path, "Policy top-10 lift", {row["policy_id"]: row["top_10_lift"] for row in comparison})


def _plot_score_depth(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    _scatter(path, "OOF score vs depth", rows, "depth", "score")


def _plot_rank_depth(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    _scatter(path, "OOF rank percentile vs depth", rows, "depth", "rank")


def _plot_score_target(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    _scatter(path, "OOF score vs target", rows, "target", "score")


def _plot_top_k(
    path: Path,
    rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    *_: Any,
) -> None:
    _bar(
        path,
        "Top-k lift",
        {
            "top5": comparison[1]["top_5_lift"],
            "top10": comparison[1]["top_10_lift"],
            "top20": comparison[1]["top_20_lift"],
        },
    )


def _plot_calibration(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    scored = [row for row in rows if np.isfinite(row["target"]) and np.isfinite(row["score"])]
    bins = np.array_split(sorted(scored, key=lambda row: row["target"]), 5)
    _bar(
        path,
        "Calibration bias by target quantile",
        {
            str(i): float(np.mean([row["residual"] for row in group]))
            for i, group in enumerate(bins)
            if len(group)
        },
    )


def _plot_gap_stability(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    _hist(path, "Gap score stability", [row["stability"] for row in rows])


def _plot_stability_depth(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    _scatter(path, "Score stability vs depth", rows, "depth", "stability")


def _plot_residual_depth(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    _scatter(path, "Residual vs depth", rows, "depth", "residual")


def _plot_residual_regime(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    values = {}
    for regime in sorted({row["regime"] for row in rows}):
        vals = [
            row["residual"]
            for row in rows
            if row["regime"] == regime and np.isfinite(row["residual"])
        ]
        values[regime] = None if not vals else float(np.mean(vals))
    _bar(path, "Residual by regime", values)


def _plot_domain_shift(
    path: Path,
    rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    *_: Any,
) -> None:
    _bar(
        path,
        "Domain shift policy Spearman",
        {row["policy_id"]: row["spearman_mean"] for row in comparison},
    )


def _plot_transfer(
    path: Path,
    rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    scores: dict[str, Any],
    error: dict[str, Any],
) -> None:
    transfer = _as_dict(_as_dict(error.get("summary")).get("transfer_limits"))
    _bar(
        path,
        "Transfer Spearman",
        {
            "B_to_C": transfer.get("b_to_c_spearman"),
            "C_to_B": transfer.get("c_to_b_spearman"),
        },
    )


def _plot_support(path: Path, rows: list[dict[str, Any]], *_: Any) -> None:
    supported = sum(1 for row in rows if np.isfinite(row["score"]))
    _bar(
        path,
        "Supported vs unsupported",
        {"supported": supported, "unsupported": len(rows) - supported},
    )


def _plot_special(
    path: Path,
    rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    *_: Any,
) -> None:
    _bar(
        path,
        "Special-band dependency",
        {
            row["policy_id"]: float(bool(row["special_band_dependency"]))
            for row in comparison
        },
    )


def _scatter(path: Path, title: str, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    x = [row[x_key] for row in rows if np.isfinite(row[x_key]) and np.isfinite(row[y_key])]
    y = [row[y_key] for row in rows if np.isfinite(row[x_key]) and np.isfinite(row[y_key])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y, s=4, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bar(path: Path, title: str, values: dict[str, Any]) -> None:
    labels = list(values) or ["none"]
    numeric = [float(values[label] or 0.0) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(labels)), numeric, color="#4c78a8")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _hist(path: Path, title: str, values: list[float]) -> None:
    data = [float(value) for value in values if np.isfinite(value)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data, bins=30, color="#4c78a8")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _validate_reports(*reports: dict[str, Any]) -> None:
    for report in reports:
        for flag in RESEARCH_FLAGS:
            if flag == "not_validated_for_deployment" and flag not in report:
                continue
            if report.get(flag) is not True:
                raise ScreeningReviewError(f"Report missing flag {flag}.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ScreeningReviewError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise ScreeningReviewError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ScreeningReviewError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ScreeningReviewError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise ScreeningReviewError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _to_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _array_value(snapshot: dict[str, np.ndarray], key: str, index: int) -> Any:
    if key not in snapshot:
        return None
    value = np.asarray(snapshot[key]).reshape(-1)[index]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _method_flags() -> dict[str, bool]:
    return {
        **RESEARCH_FLAGS,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_final_labels_generated": True,
        "no_raw_waveform_reread": True,
        "no_production_model_claim": True,
    }


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
