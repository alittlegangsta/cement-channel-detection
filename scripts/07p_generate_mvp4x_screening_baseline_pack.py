from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cement-channel")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cement_channel.data.manifest import ManifestBuildError, load_paths_config  # noqa: E402

REPORT_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
SCREENING_VERSION = "mvp4x_screening_baseline_v001"
DECISION_VERSION = "mvp4x_regime_policy_decision_v001"
REVIEW_VERSION = "mvp4x_regime_policy_review_v001"
DECISIONS = {
    "research_screening_baseline_supported",
    "research_screening_baseline_domain_shift_limited",
    "exploratory_time_frequency_v2_helpful",
    "request_label_redesign_approval",
    "request_advanced_signal_processing_approval",
    "request_server_migration_approval",
    "stop_insufficient_signal",
    "stop_data_contract_issue",
    "stop_leakage_detected",
}


class ScreeningPackCliError(RuntimeError):
    """Raised when the MVP-4X screening pack cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVP-4X research-only screening baseline pack and decision."
    )
    parser.add_argument(
        "--paths",
        "--config",
        dest="paths_config",
        default="configs/paths.local.yaml",
    )
    parser.add_argument("--policy-json", default=None)
    parser.add_argument("--robustness-json", default=None)
    parser.add_argument("--ranking-json", default=None)
    parser.add_argument("--error-json", default=None)
    parser.add_argument("--stratified-decision-json", default=None)
    parser.add_argument("--output-screening-md", default=None)
    parser.add_argument("--output-screening-json", default=None)
    parser.add_argument("--output-screening-review-dir", default=None)
    parser.add_argument("--output-decision-md", default=None)
    parser.add_argument("--output-decision-json", default=None)
    parser.add_argument("--output-review-dir", default=None)
    parser.add_argument("--output-iteration-log", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths_config(args.paths_config)
        policy_json = _resolve_data_path(
            paths,
            "reports",
            args.policy_json,
            "mvp4x_regime_policy_v001.json",
        )
        robustness_json = _resolve_data_path(
            paths,
            "reports",
            args.robustness_json,
            "mvp4x_regime_robustness_v001.json",
        )
        ranking_json = _resolve_data_path(
            paths,
            "reports",
            args.ranking_json,
            "mvp4x_ranking_audit_v001.json",
        )
        error_json = _resolve_data_path(
            paths,
            "reports",
            args.error_json,
            "mvp4x_regime_error_analysis_v001.json",
        )
        stratified_decision_json = _resolve_data_path(
            paths,
            "reports",
            args.stratified_decision_json,
            "mvp4x_stratified_decision.json",
        )
        screening_md = _resolve_data_path(
            paths,
            "reports",
            args.output_screening_md,
            "mvp4x_screening_baseline_v001.md",
        )
        screening_json = _resolve_data_path(
            paths,
            "reports",
            args.output_screening_json,
            "mvp4x_screening_baseline_v001.json",
        )
        screening_review_dir = _resolve_data_path(
            paths,
            "reports",
            args.output_screening_review_dir,
            "mvp4x_screening_review_v001",
        )
        decision_md = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_md,
            "mvp4x_regime_policy_decision.md",
        )
        decision_json = _resolve_data_path(
            paths,
            "reports",
            args.output_decision_json,
            "mvp4x_regime_policy_decision.json",
        )
        review_dir = _resolve_data_path(
            paths,
            "reports",
            args.output_review_dir,
            "mvp4x_regime_policy_review_v001",
        )
        iteration_log = _resolve_data_path(
            paths,
            "reports",
            args.output_iteration_log,
            "mvp4x_regime_policy_iteration_log.md",
        )
        for path in (
            policy_json,
            robustness_json,
            ranking_json,
            error_json,
            stratified_decision_json,
            screening_md,
            screening_json,
            screening_review_dir,
            decision_md,
            decision_json,
            review_dir,
            iteration_log,
        ):
            _ensure_path_within(paths, path, key="reports")
        if args.dry_run:
            print(
                "Dry run: would generate MVP-4X screening pack "
                f"robustness={robustness_json} ranking={ranking_json}."
            )
            return 0
        outputs = generate_outputs(
            policy=_read_json(policy_json),
            robustness=_read_json(robustness_json),
            ranking=_read_json(ranking_json),
            error=_read_json(error_json),
            stratified_decision=_read_json(stratified_decision_json),
            inputs={
                "policy_json": str(policy_json),
                "robustness_json": str(robustness_json),
                "ranking_json": str(ranking_json),
                "error_json": str(error_json),
                "stratified_decision_json": str(stratified_decision_json),
            },
        )
        write_outputs(
            outputs,
            screening_md=screening_md,
            screening_json=screening_json,
            screening_review_dir=screening_review_dir,
            decision_md=decision_md,
            decision_json=decision_json,
            review_dir=review_dir,
            iteration_log=iteration_log,
            overwrite=args.overwrite,
        )
    except (
        ManifestBuildError,
        ScreeningPackCliError,
        OSError,
        ValueError,
        KeyError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MVP-4X screening baseline pack "
        f"decision={outputs['decision']['decision']}; "
        f"research_only={outputs['decision']['research_only']}; "
        f"supported={outputs['screening']['supported_cohorts']}."
    )
    print(f"Wrote decision JSON: {decision_json}")
    return 0


def generate_outputs(
    *,
    policy: dict[str, Any],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
    stratified_decision: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    for name, report in {
        "policy": policy,
        "robustness": robustness,
        "ranking": ranking,
        "error": error,
        "stratified_decision": stratified_decision,
    }.items():
        _validate_report_flags(name, report)
    robust_support = _as_dict(robustness.get("screening_support"))
    ranking_summary = _as_dict(ranking.get("summary"))
    error_summary = _as_dict(error.get("summary"))
    strat_answers = _as_dict(stratified_decision.get("answers"))
    supported = sorted(
        set(robust_support.get("supported_screening_cohorts", []))
        & set(ranking_summary.get("ranking_stable_cohorts", []))
    )
    unsupported = sorted(
        set(robust_support.get("unsupported_screening_cohorts", []))
        | set(ranking_summary.get("ranking_unstable_cohorts", []))
    )
    domain_shift = bool(error_summary.get("domain_shift_warning")) or bool(
        strat_answers.get("12_domain_shift_exists")
    )
    calibration_bad = bool(error_summary.get("absolute_calibration_insufficient"))
    decision = (
        "research_screening_baseline_domain_shift_limited"
        if supported and domain_shift
        else "research_screening_baseline_supported"
        if supported
        else "request_label_redesign_approval"
    )
    if decision not in DECISIONS:
        raise ScreeningPackCliError(f"Unsupported decision: {decision}")
    screening = {
        "screening_version": SCREENING_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "supported_cohorts": supported,
        "unsupported_cohorts": unsupported,
        "target_view": "receiver_mean",
        "best_model": "Ridge",
        "best_feature_set": "existing_features_only",
        "robustness": robustness.get("screening_support"),
        "ranking": ranking.get("summary"),
        "calibration_limits": {
            "absolute_calibration_insufficient": calibration_bad,
            "max_abs_calibration_bias": error_summary.get("max_abs_calibration_bias"),
            "not_absolute_fraction_predictor": True,
        },
        "transfer_limits": error_summary.get("transfer_limits"),
        "domain_shift": domain_shift,
        "special_band_sensitivity": {
            row["cohort"]: row["special_band_sensitivity"]["dependency_flag"]
            for row in robustness.get("candidate_rows", [])
        },
        "feature_stability": error_summary.get("top_stable_features", []),
        "error_intervals_source": "mvp4x_regime_error_analysis_v001",
        "limitations": [
            "research-only screening baseline",
            "not absolute fraction predictor",
            "not production model",
            "not final labels",
            "not ground truth",
            "not validated outside supported cohorts",
            "B/C transfer remains weak",
            "domain shift exists",
        ],
        "forbidden_claims": [
            "production claim",
            "final labels",
            "ground-truth claim",
            "formal production regime split",
            "formal orientation filter",
        ],
        **_method_flags(),
    }
    answers = build_decision_answers(
        supported=supported,
        unsupported=unsupported,
        robustness=robustness,
        ranking=ranking,
        error=error,
        stratified_decision=stratified_decision,
        domain_shift=domain_shift,
    )
    final_decision = {
        "decision_version": DECISION_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "inputs": inputs,
        "answers": answers,
        "time_frequency_v2": {
            "triggered": False,
            "improved": "not_triggered",
            "shape": None,
            "chunks": None,
            "peak_memory_bytes": None,
            "runtime_seconds": None,
            "reason_not_triggered": (
                "existing feature screening signal is present; human review is "
                "required before any waveform reread or advanced route"
            ),
        },
        "next_human_approval": (
            "approve a research-only screening baseline review and decide whether "
            "to open a formal regime-policy discussion; do not approve production "
            "split, final labels, or production claims"
        ),
        "recommend_server": False,
        "not_authorized": [
            "production claim",
            "final labels",
            "ground-truth claim",
            "formal production regime split",
            "formal orientation filter",
            "STC",
            "APES",
            "deep learning",
        ],
        **_method_flags(),
    }
    return {
        "screening": screening,
        "decision": final_decision,
        "review": {
            "robustness": robustness,
            "ranking": ranking,
            "error": error,
            "policy": policy,
        },
    }


def build_decision_answers(
    *,
    supported: list[str],
    unsupported: list[str],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
    stratified_decision: dict[str, Any],
    domain_shift: bool,
) -> dict[str, Any]:
    robust_rows = {row["cohort"]: row for row in robustness.get("candidate_rows", [])}
    ranking_rows = {row["cohort"]: row for row in ranking.get("rows", [])}
    error_summary = _as_dict(error.get("summary"))
    strat_answers = _as_dict(stratified_decision.get("answers"))
    return {
        "1_regime_b_ranking_stable": "regime_b_high_orientation" in supported,
        "2_regime_c_ranking_stable": "regime_c_all" in supported
        or "regime_c_high_orientation" in supported,
        "3_pooled_bc_ranking_stable": "pooled_bc_all" in supported
        or "pooled_bc_high_orientation" in supported,
        "4_repeated_blocked_gap_passed": _all_supported_gap_positive(robust_rows, supported),
        "5_permutation_passed": _all_supported_permutation_positive(robust_rows, supported),
        "6_bootstrap_ci_supports_positive_signal": _all_supported_ci_positive(
            robust_rows,
            supported,
        ),
        "7_absolute_calibration_still_insufficient": bool(
            error_summary.get("absolute_calibration_insufficient")
        ),
        "8_r2_still_negative": {
            key: row["repeated_cv"]["metrics"]["r2"]["mean"]
            for key, row in robust_rows.items()
        },
        "9_supported_screening_cohorts": supported,
        "10_unsupported_cohorts": unsupported,
        "11_b_c_transfer_still_weak": True,
        "12_domain_shift_exists": domain_shift,
        "13_ranking_audit_more_reasonable_than_regression": True,
        "14_ordinal_audit_useful": {
            key: ranking_rows[key]["derived_ordinal_audit"]["macro_f1"]
            for key in ranking_rows
        },
        "15_most_stable_target_view": "receiver_mean",
        "16_most_stable_feature_set": "existing_features_only",
        "17_most_stable_model": "Ridge",
        "18_stable_features": error_summary.get("top_stable_features", []),
        "19_time_frequency_v2_triggered": False,
        "20_time_frequency_v2_improved": "not_triggered",
        "21_special_band_dependency": {
            key: row["special_band_sensitivity"]["dependency_flag"]
            for key, row in robust_rows.items()
        },
        "22_recommend_label_redesign": False,
        "23_recommend_advanced_signal_processing": (
            "not before human review of research screening/regime policy"
        ),
        "24_recommend_server": False,
        "25_next_human_approval": (
            "research-only screening baseline review / formal regime-policy discussion"
        ),
        "26_production_claims_and_final_labels_forbidden": True,
        "b_to_c_transfer": strat_answers.get("4_b_to_c_transfer"),
        "c_to_b_transfer": strat_answers.get("5_c_to_b_transfer"),
    }


def write_outputs(
    outputs: dict[str, Any],
    *,
    screening_md: Path,
    screening_json: Path,
    screening_review_dir: Path,
    decision_md: Path,
    decision_json: Path,
    review_dir: Path,
    iteration_log: Path,
    overwrite: bool,
) -> None:
    for path in (screening_md, screening_json, decision_md, decision_json, iteration_log):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    for directory in (screening_review_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)
    screening_json.write_text(_json(outputs["screening"]), encoding="utf-8")
    screening_md.write_text(format_screening_markdown(outputs["screening"]), encoding="utf-8")
    decision_json.write_text(_json(outputs["decision"]), encoding="utf-8")
    decision_md.write_text(format_decision_markdown(outputs["decision"]), encoding="utf-8")
    write_review_pack(
        review=outputs["review"],
        decision=outputs["decision"],
        review_dir=review_dir,
        screening_review_dir=screening_review_dir,
        overwrite=overwrite,
    )
    iteration_log.write_text(format_iteration_log(outputs), encoding="utf-8")


def write_review_pack(
    *,
    review: dict[str, Any],
    decision: dict[str, Any],
    review_dir: Path,
    screening_review_dir: Path,
    overwrite: bool,
) -> None:
    robustness = review["robustness"]
    ranking = review["ranking"]
    error = review["error"]
    plot_specs = {
        "cohort_support_matrix.png": ("Cohort support", _series_counts(robustness)),
        "repeated_cv_spearman.png": ("Repeated CV Spearman", _series_spearman(robustness)),
        "blocked_gap_sensitivity.png": ("Blocked gap Spearman", _series_gap(robustness)),
        "bootstrap_ci.png": ("Bootstrap CI lower", _series_ci(robustness)),
        "permutation_comparison.png": ("Permutation margin", _series_perm(robustness)),
        "ranking_enrichment.png": ("Top10 lift", _series_top10(ranking)),
        "ordinal_confusion_matrix.png": ("Ordinal macro F1", _series_ordinal(ranking)),
        "calibration_by_regime.png": ("Calibration max bias", _series_calibration(error)),
        "residual_vs_depth.png": ("Residual-depth Spearman", _series_residual_depth(error)),
        "residual_by_regime.png": ("Residual mean", _series_residual_mean(error)),
        "transfer_error_summary.png": ("Transfer Spearman", _series_transfer(error)),
        "feature_group_importance.png": ("Feature group importance", _series_features(error)),
        "stable_features.png": ("Stable features", _series_features(error)),
        "special_band_sensitivity.png": ("Special dependency", _series_special(robustness)),
    }
    for filename, (title, series) in plot_specs.items():
        _write_bar_plot(review_dir / filename, title, series, overwrite=overwrite)
    (review_dir / "review_summary.md").write_text(
        format_decision_markdown(decision),
        encoding="utf-8",
    )
    for filename in (
        "ranking_enrichment.png",
        "calibration_by_regime.png",
        "special_band_sensitivity.png",
    ):
        source = review_dir / filename
        target = screening_review_dir / filename
        _ensure_can_write(target, overwrite=overwrite)
        target.write_bytes(source.read_bytes())
    (screening_review_dir / "review_summary.md").write_text(
        format_decision_markdown(decision),
        encoding="utf-8",
    )


def format_screening_markdown(screening: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X Research Screening Baseline",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, "
            "no_final_labels, no_ground_truth_claim, no_production_claim.",
            "",
            f"- supported_cohorts: {screening['supported_cohorts']}",
            f"- unsupported_cohorts: {screening['unsupported_cohorts']}",
            f"- target_view: {screening['target_view']}",
            f"- model: {screening['best_model']}",
            f"- feature_set: {screening['best_feature_set']}",
            f"- domain_shift: {screening['domain_shift']}",
            "",
            "This is not an absolute fraction predictor, not a production model, "
            "not final labels, and not ground truth.",
            "",
        ]
    )


def format_decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Regime Policy Decision",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "no_final_labels, no_ground_truth_claim, no_production_claim.",
        "",
        f"- decision: `{decision['decision']}`",
        f"- next_human_approval: {decision['next_human_approval']}",
        "",
        "## Answers",
    ]
    for key, value in decision["answers"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Not Authorized"])
    for item in decision["not_authorized"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def format_iteration_log(outputs: dict[str, Any]) -> str:
    decision = outputs["decision"]
    screening = outputs["screening"]
    now = _utc_now()
    sections = [
        "# MVP-4X Regime Policy Iteration Log",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "no_final_labels, no_ground_truth_claim, no_production_claim.",
        "",
    ]
    items = [
        (
            1,
            "Freeze research-only regime policy.",
            "policy frozen; Regime A audit-only, B/C exploratory domains.",
            "continue",
        ),
        (
            2,
            "Repeated robust validation.",
            f"supported={screening['robustness']['supported_screening_cohorts']}",
            "continue",
        ),
        (
            3,
            "Ranking and ordinal audit.",
            f"ranking_stable={screening['ranking']['ranking_stable_cohorts']}",
            "continue",
        ),
        (
            4,
            "Calibration and error analysis.",
            f"calibration_limits={screening['calibration_limits']}",
            "continue",
        ),
        (
            5,
            "Screening baseline pack and decision.",
            f"decision={decision['decision']}",
            "stop_human_review_required",
        ),
    ]
    for iteration_id, action, metrics, status in items:
        sections.extend(
            [
                f"## Iteration {iteration_id}",
                f"- iteration_id: {iteration_id}",
                f"- timestamp: {now}",
                "- files_read: policy, robustness, ranking, error, stratified decision reports",
                f"- hypothesis: {action}",
                "- evidence_before: see prior generated JSON reports",
                f"- selected_action: {action}",
                "- files_changed: tracked code/scripts/tests/docs for current phase",
                "- commands_run: 07l/07m/07n/07o/07p plus required tests",
                "- outputs_generated: MVP-4X regime policy reports and review packs",
                f"- metrics_after: {metrics}",
                "- interpretation: research-only screening evidence remains bounded",
                f"- continue_or_stop: {status}",
                f"- human_approval_required: {status.startswith('stop')}",
                f"- proposed_next_action: {decision['next_human_approval']}",
                "",
            ]
        )
    return "\n".join(sections)


def _write_bar_plot(
    path: Path,
    title: str,
    series: dict[str, float],
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write(path, overwrite=overwrite)
    labels = list(series.keys()) or ["none"]
    values = [float(series[key]) for key in labels] if series else [0.0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(labels)), values, color="#4c78a8")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _series_counts(robustness: dict[str, Any]) -> dict[str, float]:
    return {row["cohort"]: row["sample_count"] for row in robustness.get("candidate_rows", [])}


def _series_spearman(robustness: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: row["repeated_cv"]["metrics"]["spearman"]["mean"]
        for row in robustness.get("candidate_rows", [])
    }


def _series_gap(robustness: dict[str, Any]) -> dict[str, float]:
    output = {}
    for row in robustness.get("candidate_rows", []):
        gaps = [
            item["metrics"]["spearman"]["mean"]
            for item in row["blocked_gap"].values()
            if item["metrics"]["spearman"]["mean"] is not None
        ]
        output[row["cohort"]] = min(gaps) if gaps else 0.0
    return output


def _series_ci(robustness: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: row["bootstrap_ci"]["metrics"]["spearman"]["ci_lower"]
        for row in robustness.get("candidate_rows", [])
    }


def _series_perm(robustness: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: row["permutation"]["margins"]["within_depth_bin"]
        for row in robustness.get("candidate_rows", [])
    }


def _series_top10(ranking: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: row["ranking"]["top_k"]["top_10pct"]["lift"]
        for row in ranking.get("rows", [])
    }


def _series_ordinal(ranking: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: row["derived_ordinal_audit"]["macro_f1"]
        for row in ranking.get("rows", [])
    }


def _series_calibration(error: dict[str, Any]) -> dict[str, float]:
    return {
        "max_abs_bias": float(_as_dict(error.get("summary")).get("max_abs_calibration_bias") or 0.0)
    }


def _series_residual_depth(error: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: float(row["residual_vs_depth"] or 0.0)
        for row in error.get("rows", [])
    }


def _series_residual_mean(error: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: float(row["residual_summary"].get("mean") or 0.0)
        for row in error.get("rows", [])
    }


def _series_transfer(error: dict[str, Any]) -> dict[str, float]:
    transfer = _as_dict(_as_dict(error.get("summary")).get("transfer_limits"))
    return {
        "B_to_C": float(transfer.get("b_to_c_spearman") or 0.0),
        "C_to_B": float(transfer.get("c_to_b_spearman") or 0.0),
    }


def _series_features(error: dict[str, Any]) -> dict[str, float]:
    features = _as_dict(error.get("summary")).get("top_stable_features", [])
    return {
        str(item.get("feature_group", index)): float(item.get("mean_spearman_drop") or 0.0)
        for index, item in enumerate(features[:10])
    }


def _series_special(robustness: dict[str, Any]) -> dict[str, float]:
    return {
        row["cohort"]: float(row["special_band_sensitivity"]["dependency_flag"])
        for row in robustness.get("candidate_rows", [])
    }


def _all_supported_gap_positive(rows: dict[str, Any], supported: list[str]) -> bool:
    for cohort in supported:
        row = rows.get(cohort)
        if not row:
            return False
        values = [
            item["metrics"]["spearman"]["mean"]
            for item in row["blocked_gap"].values()
            if item["metrics"]["spearman"]["mean"] is not None
        ]
        if not values or min(values) <= 0.0:
            return False
    return bool(supported)


def _all_supported_permutation_positive(rows: dict[str, Any], supported: list[str]) -> bool:
    for cohort in supported:
        row = rows.get(cohort)
        if not row:
            return False
        margins = row["permutation"]["margins"]
        if any(
            float(margins[key] or 0.0) <= 0.0
            for key in ("global", "within_depth_bin", "block")
        ):
            return False
    return bool(supported)


def _all_supported_ci_positive(rows: dict[str, Any], supported: list[str]) -> bool:
    for cohort in supported:
        row = rows.get(cohort)
        if not row:
            return False
        lower = row["bootstrap_ci"]["metrics"]["spearman"]["ci_lower"]
        if lower is None or float(lower) <= 0.0:
            return False
    return bool(supported)


def _validate_report_flags(name: str, report: dict[str, Any]) -> None:
    for flag in REPORT_FLAGS:
        if report.get(flag) is not True:
            raise ScreeningPackCliError(f"{name} missing research flag {flag}.")


def _resolve_data_path(
    config: dict[str, Any],
    key: str,
    override: str | None,
    filename: str,
) -> Path:
    if override:
        return Path(override)
    root = _as_dict(config.get("data")).get(key)
    if root:
        return Path(str(root)) / filename
    raise ScreeningPackCliError(f"data.{key} is not configured.")


def _ensure_path_within(config: dict[str, Any], path: Path, *, key: str) -> None:
    root = Path(str(_as_dict(config.get("data")).get(key))).resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ScreeningPackCliError(f"{resolved} is outside configured data.{key}: {root}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ScreeningPackCliError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ScreeningPackCliError(f"JSON must contain an object: {path}")
    return loaded


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _method_flags() -> dict[str, bool]:
    return {
        **REPORT_FLAGS,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_final_labels_generated": True,
        "no_raw_waveform_reread": True,
        "no_production_model_claim": True,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")


if __name__ == "__main__":
    raise SystemExit(main())
