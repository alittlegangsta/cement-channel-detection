from __future__ import annotations

import csv
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from cement_channel.features.mvp4x_time_frequency_features import (
    extract_time_frequency_features_from_paths,
)
from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import (
    _make_model,
    compute_regression_metrics,
    contiguous_depth_folds,
)
from cement_channel.modeling.mvp4x_max_auto import (
    MaxAutoPaths,
    calibration_bins,
    domain_shift_transfer,
    morphology_kernel,
    validate_preflight,
)
from cement_channel.modeling.mvp4x_screening_scores import (
    load_policy_masks,
    rank_percentile,
    score_stability_std,
    top_k_lift,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cement-channel")

NEXT_AUTO_VERSION = "mvp4x_next_auto_v001"
MORPHOLOGY_V003_VERSION = "mvp4x_morphology_candidate_labels_v003"
MORPHOLOGY_AUDIT_VERSION = "mvp4x_morphology_semantics_audit_v003"
FEATURE_SET_COMPARISON_VERSION = "mvp4x_feature_set_v2_comparison"
CONSOLIDATION_VERSION = "mvp4x_next_auto_model_consolidation"
ADVANCED_FEASIBILITY_VERSION = "mvp4x_advanced_methods_feasibility"
NEXT_DECISION_VERSION = "mvp4x_next_auto_decision"
REVIEW_DIR_NAME = "mvp4x_next_auto_review_v001"

SCOPE_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}
METHOD_FLAGS = {
    **SCOPE_FLAGS,
    "no_final_labels_generated": True,
    "no_ground_truth_claim": True,
    "no_production_model_claim": True,
    "no_raw_mat_modified": True,
    "no_stc": True,
    "no_apes": True,
    "no_deep_learning": True,
}
SUPPORTED_COHORTS = (
    "pooled_bc_all",
    "pooled_bc_high_orientation",
    "regime_b_high_orientation",
    "regime_c_all",
    "regime_c_high_orientation",
)
FEATURE_SETS = (
    "existing_features_only",
    "waveform_v1_only",
    "time_frequency_v2_only",
    "waveform_v1_plus_v2",
    "existing_plus_v2",
    "all_combined",
)
TARGETS = ("receiver_mean", "receiver_p90", "receiver_max")
MODELS = (
    "DummyRegressor",
    "Ridge",
    "ElasticNet",
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
)
MODEL_CONFIG = {
    "n_contiguous_folds": 3,
    "ridge_alpha": 1.0,
    "elastic_net_alpha": 0.01,
    "elastic_net_l1_ratio": 0.5,
    "random_forest_n_estimators": 48,
    "random_forest_max_depth": 6,
    "random_forest_n_jobs": 2,
    "hist_gradient_boosting_max_iter": 80,
    "hist_gradient_boosting_max_leaf_nodes": 15,
}


class NextAutoError(RuntimeError):
    """Raised when MVP-4X next-auto cannot continue safely."""


def run_phase(
    paths: MaxAutoPaths,
    *,
    phase: str,
    overwrite: bool,
    paths_config: Path | str = "configs/paths.local.yaml",
    mapping_path: Path | str = "configs/raw_variable_mapping.yaml",
    rapid_config_path: Path | str = "configs/mvp4x_rapid_exploratory.example.yaml",
) -> dict[str, Any]:
    if phase == "preflight":
        return validate_next_preflight(paths)
    if phase == "adr":
        return write_v1_retained_decision(paths, overwrite=overwrite)
    if phase == "morphology":
        return run_morphology_semantics_audit(paths, overwrite=overwrite)
    if phase == "tfv2":
        return run_time_frequency_v2_extraction(
            paths,
            paths_config=paths_config,
            mapping_path=mapping_path,
            rapid_config_path=rapid_config_path,
            overwrite=overwrite,
        )
    if phase == "modeling":
        return run_feature_set_v2_comparison(paths, overwrite=overwrite)
    if phase == "consolidation":
        return run_model_consolidation(paths, overwrite=overwrite)
    if phase == "feasibility":
        return write_advanced_methods_feasibility(paths, overwrite=overwrite)
    if phase == "review":
        return write_next_auto_decision_and_review_pack(paths, overwrite=overwrite)
    if phase == "all":
        outputs = {}
        for item in (
            "preflight",
            "adr",
            "morphology",
            "tfv2",
            "modeling",
            "consolidation",
            "feasibility",
            "review",
        ):
            outputs[item] = run_phase(
                paths,
                phase=item,
                overwrite=overwrite,
                paths_config=paths_config,
                mapping_path=mapping_path,
                rapid_config_path=rapid_config_path,
            )
        return {"phase": "all", "outputs": outputs, **METHOD_FLAGS}
    raise NextAutoError(f"Unknown next-auto phase: {phase}")


def validate_next_preflight(paths: MaxAutoPaths) -> dict[str, Any]:
    paths.ensure_read_write()
    preflight = validate_preflight(paths)
    required = {
        "final_scores_npz": paths.interim / "mvp4x_screening_scores_final_oof_v001.npz",
        "morphology_v2_npz": paths.interim / "mvp4x_morphology_candidate_labels_v002.npz",
        "max_auto_decision": paths.reports / "mvp4x_max_auto_decision.json",
        "max_auto_triage": paths.reports / "mvp4x_max_auto_triage_v001.json",
        "v1_vs_morphology_v2": paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.json",
        "final_scores_json": paths.reports / "mvp4x_screening_scores_final_oof_v001.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise NextAutoError("Missing required next-auto artifact(s): " + ", ".join(missing))
    final = _load_npz(required["final_scores_npz"])
    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    waveform = _load_npz(paths.features / "mvp4x_waveform_features_v001.npz")
    report = _read_json(required["final_scores_json"])
    checks = {
        "total_samples": np.asarray(snapshot["depth"]).shape == (7108,),
        "scored_samples": int(np.count_nonzero(np.isfinite(final["score"]))) == 4738,
        "existing_features": np.asarray(snapshot["xsi_features"]).shape == (7108, 80),
        "waveform_v1_features": np.asarray(waveform["waveform_depth_features"]).shape
        == (7108, 342),
        "combined_features": 80 + 342 == 422,
        "finite_ratios": float(np.isfinite(snapshot["xsi_features"]).mean()) == 1.0
        and float(np.isfinite(waveform["waveform_depth_features"]).mean()) == 1.0,
        "target_kernel": str(np.asarray(snapshot["target_kernel"]).item())
        == "triangular_midpoint_weighted",
        "screening_target": report.get("target") == "receiver_mean",
        "screening_model": report.get("screening_model") == "Ridge",
        "screening_feature_set": report.get("screening_feature_set") == "existing_features_only",
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise NextAutoError("Next-auto preflight failed: " + ", ".join(failed))
    return {
        "phase": "preflight",
        "base_preflight": preflight,
        "required_artifacts": {key: str(path) for key, path in required.items()},
        "checks": checks,
        **METHOD_FLAGS,
    }


def write_v1_retained_decision(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    validate_next_preflight(paths)
    comparison = _read_json(paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.json")
    summary = _as_dict(comparison.get("comparison_summary"))
    decision = {
        "report_version": "mvp4x_v1_retained_morphology_v2_audit_decision_v001",
        "generated_at": _utc_now(),
        "decision": "retain_v1_screening_baseline_and_keep_morphology_v2_audit_only",
        "v1_screening_baseline_retained": True,
        "morphology_v2_replaces_v1": False,
        "morphology_v2_audit_only": True,
        "morphology_explains_false_positive_false_negative_patterns": True,
        "future_label_semantics_requires_parallel_experiment": True,
        "final_labels_forbidden": True,
        "production_claim_forbidden": True,
        "evidence": summary,
        "interpretation": (
            "The current v1 research screening baseline remains the comparison anchor. "
            "Morphology-v2 is useful for audit semantics and error interpretation, but it "
            "did not improve same-protocol ranking and must not replace v1 automatically."
        ),
        **METHOD_FLAGS,
    }
    output_json = paths.reports / "mvp4x_v1_retained_morphology_v2_audit_decision.json"
    output_md = paths.reports / "mvp4x_v1_retained_morphology_v2_audit_decision.md"
    for path in (output_json, output_md):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(_json(decision), encoding="utf-8")
    output_md.write_text(format_v1_retained_decision_markdown(decision), encoding="utf-8")
    append_next_iteration_log(
        paths,
        iteration_id="001",
        files_read=[str(paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.json")],
        hypothesis=(
            "Morphology-v2 explains error modes but does not improve same-protocol ranking, "
            "so v1 should remain the research baseline."
        ),
        evidence_before=summary,
        selected_action="record_v1_retained_morphology_v2_audit_only_decision",
        files_changed=[str(output_json), str(output_md)],
        commands_run=[
            "python scripts/07v_run_mvp4x_next_auto.py --phase adr --overwrite",
        ],
        outputs_generated=[str(output_json), str(output_md)],
        metrics_after={"decision": decision["decision"]},
        interpretation=decision["interpretation"],
        decision=decision["decision"],
        continue_or_stop="continue",
        human_intervention_required=False,
        next_action="bounded_morphology_semantics_audit_v003",
        commit_hash_or_pending_commit="pending_commit_after_tests",
    )
    return {"phase": "adr", "outputs": [str(output_json), str(output_md)], **METHOD_FLAGS}


def format_v1_retained_decision_markdown(report: dict[str, Any]) -> str:
    evidence = _as_dict(report.get("evidence"))
    return "\n".join(
        [
            "# MVP-4X v1 Retained and Morphology-v2 Audit-Only Decision",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- decision: `{report['decision']}`",
            f"- v1_screening_baseline_retained: {report['v1_screening_baseline_retained']}",
            f"- morphology_v2_replaces_v1: {report['morphology_v2_replaces_v1']}",
            f"- morphology_v2_audit_only: {report['morphology_v2_audit_only']}",
            f"- v1_spearman: {evidence.get('v1_spearman')}",
            f"- morphology_v2_spearman: {evidence.get('morphology_v2_spearman')}",
            f"- delta_spearman: {evidence.get('delta_spearman')}",
            f"- v1_top10_lift: {evidence.get('v1_top10_lift')}",
            f"- morphology_v2_top10_lift: {evidence.get('morphology_v2_top10_lift')}",
            f"- delta_top10_lift: {evidence.get('delta_top10_lift')}",
            "",
            report["interpretation"],
            "",
            "No final labels or production claim are authorized.",
            "",
        ]
    )


def run_morphology_semantics_audit(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    validate_next_preflight(paths)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    final = _load_npz(paths.interim / "mvp4x_screening_scores_final_oof_v001.npz")
    morphology_v2 = _load_npz(paths.interim / "mvp4x_morphology_candidate_labels_v002.npz")
    prior = _read_json(paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.json")
    variants = build_morphology_v003_variants(snapshot)
    comparison = compare_morphology_semantics_variants(
        snapshot=snapshot,
        final=final,
        morphology_v2=morphology_v2,
        variants=variants,
        sklearn_modules=sklearn_modules,
        modeling_environment=modeling_environment.to_dict(),
        prior=prior,
    )
    output_npz = paths.interim / f"{MORPHOLOGY_V003_VERSION}.npz"
    output_csv = paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.csv"
    output_md = paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.md"
    output_json = paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.json"
    for path in (output_npz, output_csv, output_md, output_json):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        report_version=np.asarray(MORPHOLOGY_V003_VERSION),
        depth=np.asarray(snapshot["depth"], dtype=np.float32),
        v1_receiver_mean=np.asarray(snapshot["receiver_mean"], dtype=np.float32),
        candidate_fraction=variants["candidate_fraction"].astype(np.float32),
        largest_connected_component_fraction=variants[
            "largest_connected_component_fraction"
        ].astype(np.float32),
        max_azimuth_channel_fraction=variants["max_azimuth_channel_fraction"].astype(np.float32),
        relative_anomaly_fraction=variants["relative_anomaly_fraction"].astype(np.float32),
        combined_channel_fraction=variants["combined_channel_fraction"].astype(np.float32),
        depth_label_confidence=variants["depth_label_confidence"].astype(np.float32),
        continuity_weighted_fraction=variants["continuity_weighted_fraction"].astype(np.float32),
        broad_channel_weighted_fraction=variants["broad_channel_weighted_fraction"].astype(
            np.float32
        ),
        confidence_gated_combined_fraction=variants["confidence_gated_combined_fraction"].astype(
            np.float32
        ),
        variant_names=np.asarray(
            [
                "continuity_weighted_fraction",
                "broad_channel_weighted_fraction",
                "confidence_gated_combined_fraction",
            ]
        ),
        metadata_json=np.asarray(_json(comparison["report"])),
        audit_only=np.asarray(True),
        parallel_comparison_only=np.asarray(True),
        **{key: np.asarray(value) for key, value in METHOD_FLAGS.items()},
    )
    _write_csv(comparison["csv_rows"], output_csv)
    output_json.write_text(_json(comparison["report"]), encoding="utf-8")
    output_md.write_text(
        format_morphology_semantics_markdown(comparison["report"]), encoding="utf-8"
    )
    append_next_iteration_log(
        paths,
        iteration_id="002",
        files_read=[
            str(paths.interim / "mvp4x_research_snapshot_v001.npz"),
            str(paths.interim / "mvp4x_screening_scores_final_oof_v001.npz"),
            str(paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.json"),
        ],
        hypothesis=(
            "A small set of pre-registered morphology semantics may turn explanatory "
            "false-positive/false-negative morphology into a more stable screening target."
        ),
        evidence_before=prior.get("comparison_summary", {}),
        selected_action="generate_three_parallel_morphology_semantics_variants_and_compare",
        files_changed=[str(output_npz), str(output_csv), str(output_md), str(output_json)],
        commands_run=[
            "python scripts/07v_run_mvp4x_next_auto.py --phase morphology --overwrite",
        ],
        outputs_generated=[str(output_npz), str(output_csv), str(output_md), str(output_json)],
        metrics_after=comparison["report"]["decision_summary"],
        interpretation=comparison["report"]["interpretation"],
        decision=comparison["report"]["decision"],
        continue_or_stop="continue",
        human_intervention_required=False,
        next_action="fixed_time_frequency_v2_extraction",
        commit_hash_or_pending_commit="pending_commit_after_tests",
    )
    return {
        "phase": "morphology",
        "outputs": [str(output_npz), str(output_csv), str(output_md), str(output_json)],
        "decision": comparison["report"]["decision"],
        **METHOD_FLAGS,
    }


def build_morphology_v003_variants(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    kernel = morphology_kernel(snapshot)
    candidate_fraction = np.nan_to_num(kernel["candidate_fraction"], nan=0.0).astype(np.float32)
    lcc = np.nan_to_num(kernel["lcc"], nan=0.0).astype(np.float32)
    max_az = np.nan_to_num(kernel["max_azimuth"], nan=0.0).astype(np.float32)
    relative = np.nan_to_num(kernel["relative_drop"], nan=0.0).astype(np.float32)
    confidence = np.clip(
        np.nan_to_num(np.asarray(snapshot["label_confidence"], dtype=np.float32), nan=0.0),
        0.0,
        1.0,
    )
    combined = np.clip(
        0.50 * candidate_fraction + 0.25 * lcc + 0.25 * max_az,
        0.0,
        1.0,
    ).astype(np.float32)
    return {
        "candidate_fraction": candidate_fraction,
        "largest_connected_component_fraction": lcc,
        "max_azimuth_channel_fraction": max_az,
        "relative_anomaly_fraction": relative,
        "combined_channel_fraction": combined,
        "depth_label_confidence": confidence,
        "continuity_weighted_fraction": np.clip(
            0.60 * candidate_fraction + 0.40 * lcc,
            0.0,
            1.0,
        ).astype(np.float32),
        "broad_channel_weighted_fraction": np.clip(
            0.60 * candidate_fraction + 0.40 * max_az,
            0.0,
            1.0,
        ).astype(np.float32),
        "confidence_gated_combined_fraction": np.clip(
            combined * (0.35 + 0.65 * confidence),
            0.0,
            1.0,
        ).astype(np.float32),
    }


def compare_morphology_semantics_variants(
    *,
    snapshot: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    morphology_v2: dict[str, np.ndarray],
    variants: dict[str, np.ndarray],
    sklearn_modules: dict[str, Any],
    modeling_environment: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    X = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    depth = np.asarray(snapshot["depth"], dtype=np.float32)
    supported = np.isfinite(np.asarray(final["score"], dtype=np.float32))
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    targets = {
        "v1_receiver_mean": np.asarray(snapshot["receiver_mean"], dtype=np.float32),
        "morphology_v2_audit_only": np.asarray(
            morphology_v2["morphology_candidate_v2"],
            dtype=np.float32,
        ),
        "continuity_weighted_fraction": variants["continuity_weighted_fraction"],
        "broad_channel_weighted_fraction": variants["broad_channel_weighted_fraction"],
        "confidence_gated_combined_fraction": variants["confidence_gated_combined_fraction"],
    }
    formulas = {
        "continuity_weighted_fraction": (
            "clip(0.60*candidate_fraction + 0.40*largest_connected_component_fraction, 0, 1)"
        ),
        "broad_channel_weighted_fraction": (
            "clip(0.60*candidate_fraction + 0.40*max_azimuth_channel_fraction, 0, 1)"
        ),
        "confidence_gated_combined_fraction": (
            "clip(combined_channel_fraction * (0.35 + 0.65*depth_label_confidence), 0, 1)"
        ),
    }
    summaries: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for target_name, y in targets.items():
        summary = evaluate_ridge_target_protocol(
            X=X,
            y=y,
            depth=depth,
            supported=supported,
            regimes=regimes,
            sklearn_modules=sklearn_modules,
        )
        summaries[target_name] = summary
        csv_rows.append(
            {
                "target": target_name,
                "formula": formulas.get(target_name, "existing target"),
                "sample_count": summary["sample_count"],
                "spearman": _metric(summary["oof_metrics"], "spearman"),
                "mae": _metric(summary["oof_metrics"], "mae"),
                "rmse": _metric(summary["oof_metrics"], "rmse"),
                "r2": _metric(summary["oof_metrics"], "r2"),
                "top5_lift": _top_lift(summary, "top_5pct"),
                "top10_lift": _top_lift(summary, "top_10pct"),
                "top20_lift": _top_lift(summary, "top_20pct"),
                "gap_spearman_std": summary["gap_stability"]["spearman_std"],
                "score_stability_mean": _metric(summary["score_stability"], "mean"),
                "global_permutation_margin": summary["permutation_margins"]["global"],
                "within_depth_bin_permutation_margin": summary["permutation_margins"][
                    "within_depth_bin"
                ],
                "block_permutation_margin": summary["permutation_margins"]["block"],
                "max_abs_calibration_bias": summary["max_abs_calibration_bias"],
                "domain_shift_min_transfer_spearman": summary["domain_shift_min_transfer_spearman"],
                "false_positive_like_count": summary["false_positive_like_count"],
                "false_negative_like_count": summary["false_negative_like_count"],
                "audit_only": True,
                "parallel_comparison_only": True,
                **METHOD_FLAGS,
            }
        )
    v1 = summaries["v1_receiver_mean"]
    recommended = []
    for name in (
        "continuity_weighted_fraction",
        "broad_channel_weighted_fraction",
        "confidence_gated_combined_fraction",
    ):
        row = variant_gate_summary(name, summaries[name], v1)
        summaries[name]["gate_summary"] = row
        if row["passes_all_criteria"]:
            recommended.append(name)
    decision = (
        "recommended_parallel_morphology_candidate_for_human_review"
        if recommended
        else "morphology_candidates_not_helpful_retain_v1"
    )
    interpretation = (
        "At least one pre-registered morphology semantic variant improved same-protocol "
        "ranking without worsening stability/domain-shift gates; keep it for human review "
        "only and do not replace v1 automatically."
        if recommended
        else (
            "No pre-registered morphology semantic variant met the simultaneous improvement "
            "criteria. Retain v1 and stop morphology label-variant exploration."
        )
    )
    report = {
        "report_version": MORPHOLOGY_AUDIT_VERSION,
        "generated_at": _utc_now(),
        "modeling_environment": modeling_environment,
        "feature_set": "existing_features_only",
        "model": "Ridge",
        "sample_mask": "supported_scored_oof_rows_only",
        "variant_formulas": formulas,
        "no_dynamic_threshold_scan": True,
        "no_result_based_reweighting": True,
        "target_distributions": {name: _summary(values) for name, values in targets.items()},
        "target_summaries": summaries,
        "prior_v1_vs_morphology_v2": prior.get("comparison_summary"),
        "decision_summary": {
            "recommended_variants": recommended,
            "v1_spearman": _metric(v1["oof_metrics"], "spearman"),
            "v1_top10_lift": _top_lift(v1, "top_10pct"),
            "criteria": {
                "spearman_improvement_min": 0.03,
                "top10_lift_improvement_min": 0.20,
                "blocked_gap_stability_not_degraded": True,
                "permutation_margins_positive": True,
                "domain_shift_not_obviously_worse": True,
            },
        },
        "decision": decision,
        "interpretation": interpretation,
        "audit_only": True,
        "parallel_comparison_only": True,
        **METHOD_FLAGS,
    }
    return {"report": report, "csv_rows": csv_rows}


def evaluate_ridge_target_protocol(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    supported: np.ndarray,
    regimes: np.ndarray,
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    config = {"n_contiguous_folds": 3, "ridge_alpha": 1.0}
    gap_results = {}
    for gap in (10.0, 25.0, 50.0):
        gap_results[f"gap_{int(gap)}_ft"] = fit_oof_any_model(
            X=X,
            y=y,
            depth=depth,
            sample_mask=supported,
            model_name="Ridge",
            sklearn_modules=sklearn_modules,
            config=config,
            gap_ft=gap,
            random_seed=20240605 + int(gap),
        )
    primary = gap_results["gap_25_ft"]
    prediction = primary["prediction"]
    mask = supported & np.isfinite(prediction) & np.isfinite(y)
    rank = rank_percentile(prediction)
    stability = score_stability_std(
        gap_results["gap_10_ft"]["prediction"],
        gap_results["gap_25_ft"]["prediction"],
        gap_results["gap_50_ft"]["prediction"],
    )
    rng = np.random.default_rng(20240605)
    permutation = {
        "global": permutation_margin(
            X,
            y,
            depth,
            supported,
            "Ridge",
            sklearn_modules,
            config,
            rng.permutation(y),
        ),
        "within_depth_bin": permutation_margin(
            X,
            y,
            depth,
            supported,
            "Ridge",
            sklearn_modules,
            config,
            _permute_within_depth_bins(y, depth, rng),
        ),
        "block": permutation_margin(
            X,
            y,
            depth,
            supported,
            "Ridge",
            sklearn_modules,
            config,
            _permute_blocks(y, depth, rng),
        ),
    }
    transfer = domain_shift_transfer(
        X=X,
        y=y,
        regimes=regimes,
        supported=supported,
        sklearn_modules=sklearn_modules,
        config=config,
    )
    calibration = calibration_bins(y, prediction, mask, n_bins=8)
    residual = prediction - y
    fp_like = mask & (rank >= 0.90) & (y <= np.nanquantile(y[mask], 0.50))
    fn_like = mask & (rank <= 0.50) & (y >= np.nanquantile(y[mask], 0.90))
    spearmans = [
        _metric(result["metrics"], "spearman")
        for result in gap_results.values()
        if _metric(result["metrics"], "spearman") is not None
    ]
    transfer_values = [
        _metric(value, "spearman")
        for value in transfer.values()
        if _metric(value, "spearman") is not None
    ]
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "oof_metrics": compute_regression_metrics(y[mask], prediction[mask]),
        "ranking": top_k_lift(y[mask], rank[mask]),
        "gap_results": {name: result["metrics"] for name, result in gap_results.items()},
        "gap_stability": {
            "spearman_std": None if not spearmans else float(np.std(spearmans)),
            "spearman_min": None if not spearmans else float(np.min(spearmans)),
            "spearman_max": None if not spearmans else float(np.max(spearmans)),
        },
        "score_stability": _summary(stability[mask]),
        "permutation_margins": permutation,
        "domain_shift": transfer,
        "domain_shift_min_transfer_spearman": None
        if not transfer_values
        else float(np.min(transfer_values)),
        "calibration": calibration,
        "max_abs_calibration_bias": max_abs_calibration_bias(calibration),
        "false_positive_like_count": int(np.count_nonzero(fp_like)),
        "false_negative_like_count": int(np.count_nonzero(fn_like)),
        "residual_summary": _summary(residual[mask]),
    }


def variant_gate_summary(
    name: str, variant: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    s = _metric(variant["oof_metrics"], "spearman")
    base_s = _metric(baseline["oof_metrics"], "spearman")
    lift = _top_lift(variant, "top_10pct")
    base_lift = _top_lift(baseline, "top_10pct")
    gap_std = _metric(variant["gap_stability"], "spearman_std")
    base_gap_std = _metric(baseline["gap_stability"], "spearman_std")
    margins = _as_dict(variant.get("permutation_margins"))
    min_transfer = _metric(variant, "domain_shift_min_transfer_spearman")
    base_min_transfer = _metric(baseline, "domain_shift_min_transfer_spearman")
    checks = {
        "spearman_improved": s is not None and base_s is not None and s >= base_s + 0.03,
        "top10_lift_improved": lift is not None
        and base_lift is not None
        and lift >= base_lift + 0.20,
        "blocked_gap_stability_not_degraded": gap_std is not None
        and base_gap_std is not None
        and gap_std <= base_gap_std + 0.01,
        "permutation_margins_positive": all(
            value is not None and float(value) > 0.0 for value in margins.values()
        ),
        "domain_shift_not_obviously_worse": min_transfer is None
        or base_min_transfer is None
        or min_transfer >= base_min_transfer - 0.03,
    }
    return {
        "variant": name,
        "spearman": s,
        "baseline_spearman": base_s,
        "delta_spearman": None if s is None or base_s is None else s - base_s,
        "top10_lift": lift,
        "baseline_top10_lift": base_lift,
        "delta_top10_lift": None if lift is None or base_lift is None else lift - base_lift,
        "checks": checks,
        "passes_all_criteria": all(checks.values()),
    }


def format_morphology_semantics_markdown(report: dict[str, Any]) -> str:
    rows = report["decision_summary"]
    lines = [
        "# MVP-4X Morphology Semantics Audit v003",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, audit_only, "
        "parallel_comparison_only, no_final_labels, no_ground_truth_claim, "
        "no_production_claim, not_validated_for_deployment.",
        "",
        f"- decision: `{report['decision']}`",
        f"- recommended_variants: {rows['recommended_variants']}",
        f"- v1_spearman: {rows['v1_spearman']}",
        f"- v1_top10_lift: {rows['v1_top10_lift']}",
        f"- no_dynamic_threshold_scan: {report['no_dynamic_threshold_scan']}",
        f"- no_result_based_reweighting: {report['no_result_based_reweighting']}",
        "",
        "## Formulas",
    ]
    lines.extend(f"- {name}: `{formula}`" for name, formula in report["variant_formulas"].items())
    lines.extend(["", report["interpretation"], ""])
    return "\n".join(lines)


def run_time_frequency_v2_extraction(
    paths: MaxAutoPaths,
    *,
    paths_config: Path | str,
    mapping_path: Path | str,
    rapid_config_path: Path | str,
    overwrite: bool,
) -> dict[str, Any]:
    validate_next_preflight(paths)
    output_npz = paths.features / "mvp4x_time_frequency_features_v002.npz"
    output_md = paths.reports / "mvp4x_time_frequency_features_v002.md"
    output_json = paths.reports / "mvp4x_time_frequency_features_v002.json"
    report = extract_time_frequency_features_from_paths(
        paths_config=paths_config,
        mapping_path=mapping_path,
        snapshot_npz=paths.interim / "mvp4x_research_snapshot_v001.npz",
        rapid_config_path=rapid_config_path,
        output_npz=output_npz,
        output_report_md=output_md,
        output_report_json=output_json,
        overwrite=overwrite,
    )
    if report.chunk_memory_cap_bytes > 256 * 1024 * 1024:
        raise NextAutoError("TF-v2 memory cap exceeded 256 MB.")
    if report.finite_ratio.get("tf_depth_features") != 1.0:
        raise NextAutoError("TF-v2 depth feature finite ratio is not 1.0.")
    append_next_iteration_log(
        paths,
        iteration_id="003",
        files_read=[
            str(paths.interim / "mvp4x_research_snapshot_v001.npz"),
            str(mapping_path),
            str(rapid_config_path),
        ],
        hypothesis=(
            "A fixed, pre-registered time-frequency feature set can add waveform physics "
            "without dynamic band search, STC, APES, or deep learning."
        ),
        evidence_before={"waveform_v1_features": 342, "local_memory_cap_mb": 256},
        selected_action="extract_fixed_time_frequency_v2_features",
        files_changed=[str(output_npz), str(output_md), str(output_json)],
        commands_run=[
            "python scripts/07v_run_mvp4x_next_auto.py --phase tfv2 --overwrite",
        ],
        outputs_generated=[str(output_npz), str(output_md), str(output_json)],
        metrics_after={
            "shape": [report.sample_count, report.depth_feature_count],
            "chunks": report.chunk_count,
            "peak_memory_bytes": report.peak_memory_bytes,
            "runtime_seconds": report.runtime_seconds,
            "finite_ratio": report.finite_ratio.get("tf_depth_features"),
        },
        interpretation=(
            "TF-v2 extraction completed as a fixed research-only feature artifact with "
            "chunked raw waveform reads and no production or label claim."
        ),
        decision="continue_to_classical_model_v2_comparison",
        continue_or_stop="continue",
        human_intervention_required=False,
        next_action="classical_feature_set_v2_comparison",
        commit_hash_or_pending_commit="pending_commit_after_tests",
    )
    return {
        "phase": "tfv2",
        "outputs": [str(output_npz), str(output_md), str(output_json)],
        "shape": [report.sample_count, report.depth_feature_count],
        "chunks": report.chunk_count,
        "peak_memory_bytes": report.peak_memory_bytes,
        "runtime_seconds": report.runtime_seconds,
        **METHOD_FLAGS,
    }


def run_feature_set_v2_comparison(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    validate_next_preflight(paths)
    tf_path = paths.features / "mvp4x_time_frequency_features_v002.npz"
    if not tf_path.exists():
        raise NextAutoError(f"Missing TF-v2 feature artifact: {tf_path}")
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    waveform = _load_npz(paths.features / "mvp4x_waveform_features_v001.npz")
    tfv2 = _load_npz(tf_path)
    policy = _load_npz(paths.interim / "mvp4x_screening_policy_v001.npz")
    final_report = _read_json(paths.reports / "mvp4x_screening_scores_final_oof_v001.json")
    feature_sets = build_v2_feature_sets(snapshot, waveform, tfv2)
    masks = load_policy_masks(policy)
    matrix = run_classical_model_matrix(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        sklearn_modules=sklearn_modules,
    )
    strict = run_strict_v2_audits(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        matrix_rows=matrix["rows"],
        sklearn_modules=sklearn_modules,
    )
    decision = decide_fixed_time_frequency_v2(matrix, strict, final_report)
    report = {
        "report_version": f"{FEATURE_SET_COMPARISON_VERSION}_v001",
        "generated_at": _utc_now(),
        "modeling_environment": modeling_environment.to_dict(),
        "feature_sets": summarize_v2_feature_sets(feature_sets),
        "targets": list(TARGETS),
        "models": list(MODELS),
        "cohorts": list(SUPPORTED_COHORTS),
        "validation": {
            "full_matrix": "blocked_contiguous_cv_gap_25_ft",
            "strict_audit": [
                "blocked_gap_10_25_50_ft",
                "global_permutation",
                "within_depth_bin_permutation",
                "block_permutation",
                "bootstrap_ci",
                "transfer_analysis",
                "feature_group_ablation",
                "permutation_importance",
            ],
            "train_fold_only_preprocessing": True,
        },
        "matrix_summary": {
            "row_count": len(matrix["rows"]),
            "best_overall": matrix["best_overall"],
            "best_by_feature_set": matrix["best_by_feature_set"],
            "best_by_cohort": matrix["best_by_cohort"],
        },
        "strict_audits": strict,
        "decision_summary": decision,
        "fixed_time_frequency_v2_helpful": decision["fixed_time_frequency_v2_helpful"],
        "interpretation": decision["interpretation"],
        **METHOD_FLAGS,
    }
    output_csv = paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.csv"
    output_md = paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.md"
    output_json = paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.json"
    for path in (output_csv, output_md, output_json):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(matrix["csv_rows"], output_csv)
    output_json.write_text(_json(report), encoding="utf-8")
    output_md.write_text(format_feature_set_comparison_markdown(report), encoding="utf-8")
    append_next_iteration_log(
        paths,
        iteration_id="004",
        files_read=[
            str(paths.interim / "mvp4x_research_snapshot_v001.npz"),
            str(paths.features / "mvp4x_waveform_features_v001.npz"),
            str(tf_path),
            str(paths.interim / "mvp4x_screening_policy_v001.npz"),
        ],
        hypothesis=(
            "Fixed TF-v2 features may improve research-only ranking over existing v1 "
            "features under the same classical blocked-CV protocol."
        ),
        evidence_before=final_report.get("metrics", {}),
        selected_action="run_classical_model_feature_set_v2_comparison",
        files_changed=[str(output_csv), str(output_md), str(output_json)],
        commands_run=[
            "python scripts/07v_run_mvp4x_next_auto.py --phase modeling --overwrite",
        ],
        outputs_generated=[str(output_csv), str(output_md), str(output_json)],
        metrics_after=decision,
        interpretation=decision["interpretation"],
        decision=decision["decision"],
        continue_or_stop="continue",
        human_intervention_required=False,
        next_action="policy_model_consolidation",
        commit_hash_or_pending_commit="pending_commit_after_tests",
    )
    return {
        "phase": "modeling",
        "outputs": [str(output_csv), str(output_md), str(output_json)],
        "decision": decision["decision"],
        **METHOD_FLAGS,
    }


def build_v2_feature_sets(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    tfv2: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    existing_names = np.asarray(snapshot["xsi_feature_names"]).astype(str)
    existing_groups = np.asarray(snapshot["xsi_feature_group"]).astype(str)
    wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    wave_names = np.asarray(waveform["waveform_depth_feature_names"]).astype(str)
    wave_groups = np.asarray(waveform["waveform_depth_feature_group"]).astype(str)
    tf = np.asarray(tfv2["tf_depth_features"], dtype=np.float32)
    tf_names = np.asarray(tfv2["tf_depth_feature_names"]).astype(str)
    tf_groups = np.asarray(tfv2["tf_depth_feature_group"]).astype(str)
    row_count = existing.shape[0]
    for name, matrix in (("waveform_v1", wave), ("tfv2", tf)):
        if matrix.shape[0] != row_count:
            raise NextAutoError(f"{name} feature row count differs from snapshot.")
        if not np.all(np.isfinite(matrix)):
            raise NextAutoError(f"{name} feature matrix is not finite.")
    return {
        "existing_features_only": {
            "matrix": existing,
            "names": existing_names,
            "groups": np.asarray([f"existing:{item}" for item in existing_groups]),
        },
        "waveform_v1_only": {
            "matrix": wave,
            "names": wave_names,
            "groups": np.asarray([f"waveform_v1:{item}" for item in wave_groups]),
        },
        "time_frequency_v2_only": {
            "matrix": tf,
            "names": tf_names,
            "groups": np.asarray([f"time_frequency_v2:{item}" for item in tf_groups]),
        },
        "waveform_v1_plus_v2": {
            "matrix": np.column_stack([wave, tf]).astype(np.float32),
            "names": np.concatenate([wave_names, tf_names]).astype(str),
            "groups": np.concatenate(
                [
                    np.asarray([f"waveform_v1:{item}" for item in wave_groups]),
                    np.asarray([f"time_frequency_v2:{item}" for item in tf_groups]),
                ]
            ).astype(str),
        },
        "existing_plus_v2": {
            "matrix": np.column_stack([existing, tf]).astype(np.float32),
            "names": np.concatenate([existing_names, tf_names]).astype(str),
            "groups": np.concatenate(
                [
                    np.asarray([f"existing:{item}" for item in existing_groups]),
                    np.asarray([f"time_frequency_v2:{item}" for item in tf_groups]),
                ]
            ).astype(str),
        },
        "all_combined": {
            "matrix": np.column_stack([existing, wave, tf]).astype(np.float32),
            "names": np.concatenate([existing_names, wave_names, tf_names]).astype(str),
            "groups": np.concatenate(
                [
                    np.asarray([f"existing:{item}" for item in existing_groups]),
                    np.asarray([f"waveform_v1:{item}" for item in wave_groups]),
                    np.asarray([f"time_frequency_v2:{item}" for item in tf_groups]),
                ]
            ).astype(str),
        },
    }


def run_classical_model_matrix(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for cohort in SUPPORTED_COHORTS:
        if cohort not in masks:
            continue
        for target in TARGETS:
            y = np.asarray(snapshot[target], dtype=np.float32)
            sample_mask = masks[cohort] & np.isfinite(y)
            for feature_set in FEATURE_SETS:
                X = np.asarray(feature_sets[feature_set]["matrix"], dtype=np.float32)
                for model_name in MODELS:
                    print(
                        "MVP-4X next-auto model "
                        f"cohort={cohort} target={target} "
                        f"feature_set={feature_set} model={model_name}",
                        flush=True,
                    )
                    result = fit_oof_any_model(
                        X=X,
                        y=y,
                        depth=depth,
                        sample_mask=sample_mask,
                        model_name=model_name,
                        sklearn_modules=sklearn_modules,
                        config=MODEL_CONFIG,
                        gap_ft=25.0,
                        random_seed=20240605 + len(rows),
                    )
                    prediction = result["prediction"]
                    valid = sample_mask & np.isfinite(prediction)
                    rank = rank_percentile(prediction)
                    aggregate = compute_extended_metrics(y[valid], prediction[valid], rank[valid])
                    calibration = calibration_bins(y, prediction, valid, n_bins=8)
                    by_regime = residual_by_regime(snapshot, y, prediction, valid)
                    row = {
                        "cohort": cohort,
                        "feature_set": feature_set,
                        "target": target,
                        "model": model_name,
                        "sample_count": int(np.count_nonzero(valid)),
                        "metrics": aggregate,
                        "calibration": calibration,
                        "max_abs_calibration_bias": max_abs_calibration_bias(calibration),
                        "residual_by_regime": by_regime,
                        "folds": result["folds"],
                        "bootstrap_ci": bootstrap_spearman_ci(y[valid], prediction[valid]),
                        "supported_cohort": True,
                        **METHOD_FLAGS,
                    }
                    rows.append(row)
                    csv_rows.append(flatten_model_row(row))
    return {
        "status": "completed",
        "rows": rows,
        "csv_rows": csv_rows,
        "best_overall": select_best_model_row(rows),
        "best_by_feature_set": {
            name: select_best_model_row([row for row in rows if row["feature_set"] == name])
            for name in FEATURE_SETS
        },
        "best_by_cohort": {
            name: select_best_model_row([row for row in rows if row["cohort"] == name])
            for name in SUPPORTED_COHORTS
        },
        **METHOD_FLAGS,
    }


def fit_oof_any_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    gap_ft: float,
    random_seed: int,
) -> dict[str, Any]:
    matrix = np.nan_to_num(
        np.asarray(X, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1) & np.all(np.isfinite(matrix), axis=1)
    folds = contiguous_depth_folds(depth, mask, n_folds=int(config.get("n_contiguous_folds", 3)))
    prediction = np.full(y.shape, np.nan, dtype=np.float32)
    fold_id = np.full(y.shape, -1, dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, validation_mask in enumerate(folds):
        validation_mask = validation_mask & mask
        train_mask = mask & ~validation_mask
        if gap_ft > 0.0 and np.any(validation_mask):
            low = float(np.min(depth[validation_mask])) - gap_ft
            high = float(np.max(depth[validation_mask])) + gap_ft
            train_mask = train_mask & ((depth < low) | (depth > high))
        if np.count_nonzero(train_mask) == 0 or np.count_nonzero(validation_mask) == 0:
            continue
        model = _make_model(
            model_name,
            sklearn_modules,
            config,
            random_state=random_seed + fold_index,
        )
        model.fit(matrix[train_mask], y[train_mask])
        pred = np.asarray(model.predict(matrix[validation_mask]), dtype=np.float32)
        prediction[validation_mask] = pred
        fold_id[validation_mask] = fold_index
        fold_rows.append(
            {
                "fold_id": fold_index,
                "train_count": int(np.count_nonzero(train_mask)),
                "validation_count": int(np.count_nonzero(validation_mask)),
                **compute_regression_metrics(y[validation_mask], pred),
            }
        )
    return {
        "prediction": prediction,
        "fold_id": fold_id,
        "folds": fold_rows,
        "metrics": compute_regression_metrics(y[mask], prediction[mask]),
    }


def compute_extended_metrics(
    y_true: np.ndarray, prediction: np.ndarray, rank: np.ndarray
) -> dict[str, Any]:
    metrics = compute_regression_metrics(y_true, prediction)
    metrics["top_k_lift"] = top_k_lift(y_true, rank)
    metrics["ndcg"] = ndcg_score(y_true, prediction)
    metrics["kendall_tau"] = safe_kendall(y_true, prediction)
    metrics["ordinal_macro_f1"] = ordinal_macro_f1(y_true, prediction)
    return metrics


def run_strict_v2_audits(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    matrix_rows: list[dict[str, Any]],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    baseline = select_matching_row(
        matrix_rows,
        feature_set="existing_features_only",
        target="receiver_mean",
        model="Ridge",
        cohort="pooled_bc_all",
    )
    best_overall = select_best_model_row(matrix_rows)
    best_v2 = select_best_model_row(
        [
            row
            for row in matrix_rows
            if row["feature_set"]
            in {"time_frequency_v2_only", "waveform_v1_plus_v2", "existing_plus_v2", "all_combined"}
            and row["target"] == "receiver_mean"
        ]
    )
    candidates = {
        "baseline_existing_ridge_receiver_mean_pooled_bc_all": baseline,
        "best_v2_receiver_mean": best_v2,
        "best_overall": best_overall,
    }
    audits = {}
    for name, row in candidates.items():
        if not row:
            audits[name] = {"status": "skipped_no_candidate"}
            continue
        audits[name] = strict_candidate_audit(
            snapshot=snapshot,
            feature_set=feature_sets[row["feature_set"]],
            masks=masks,
            row=row,
            sklearn_modules=sklearn_modules,
        )
    return {"status": "completed", "candidates": audits, **METHOD_FLAGS}


def strict_candidate_audit(
    *,
    snapshot: dict[str, np.ndarray],
    feature_set: dict[str, Any],
    masks: dict[str, np.ndarray],
    row: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    X = np.asarray(feature_set["matrix"], dtype=np.float32)
    groups = np.asarray(feature_set["groups"]).astype(str)
    names = np.asarray(feature_set["names"]).astype(str)
    y = np.asarray(snapshot[row["target"]], dtype=np.float32)
    depth = np.asarray(snapshot["depth"], dtype=np.float32)
    sample_mask = masks[row["cohort"]] & np.isfinite(y)
    gap_predictions = {}
    gap_metrics = {}
    for gap in (10.0, 25.0, 50.0):
        result = fit_oof_any_model(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=row["model"],
            sklearn_modules=sklearn_modules,
            config=MODEL_CONFIG,
            gap_ft=gap,
            random_seed=20240700 + int(gap),
        )
        gap_predictions[gap] = result["prediction"]
        gap_metrics[f"gap_{int(gap)}_ft"] = result["metrics"]
    primary_pred = gap_predictions[25.0]
    valid = sample_mask & np.isfinite(primary_pred)
    stability = score_stability_std(
        gap_predictions[10.0], gap_predictions[25.0], gap_predictions[50.0]
    )
    rng = np.random.default_rng(20240605)
    permutations = {
        "global": permutation_margin(
            X,
            y,
            depth,
            sample_mask,
            row["model"],
            sklearn_modules,
            MODEL_CONFIG,
            rng.permutation(y),
        ),
        "within_depth_bin": permutation_margin(
            X,
            y,
            depth,
            sample_mask,
            row["model"],
            sklearn_modules,
            MODEL_CONFIG,
            _permute_within_depth_bins(y, depth, rng),
        ),
        "block": permutation_margin(
            X,
            y,
            depth,
            sample_mask,
            row["model"],
            sklearn_modules,
            MODEL_CONFIG,
            _permute_blocks(y, depth, rng),
        ),
    }
    transfer = transfer_for_candidate(
        X=X,
        y=y,
        regimes=np.asarray(snapshot["broad_regime_id"]).astype(str),
        model_name=row["model"],
        sklearn_modules=sklearn_modules,
    )
    group_ablation = feature_group_ablation(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=row["model"],
        groups=groups,
        sklearn_modules=sklearn_modules,
        baseline_spearman=_metric(row["metrics"], "spearman"),
    )
    importance = permutation_importance_summary(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=row["model"],
        feature_names=names,
        feature_groups=groups,
        sklearn_modules=sklearn_modules,
    )
    return {
        "status": "completed",
        "candidate": {
            "cohort": row["cohort"],
            "feature_set": row["feature_set"],
            "target": row["target"],
            "model": row["model"],
        },
        "gap_metrics": gap_metrics,
        "gap_stability": {
            "score_stability": _summary(stability[valid]),
            "spearman_values": {
                key: _metric(value, "spearman") for key, value in gap_metrics.items()
            },
        },
        "permutation_margins": permutations,
        "transfer": transfer,
        "feature_group_ablation": group_ablation,
        "permutation_importance": importance,
        **METHOD_FLAGS,
    }


def permutation_margin(
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    permuted_y: np.ndarray,
) -> float | None:
    real = fit_oof_any_model(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        gap_ft=25.0,
        random_seed=20240605,
    )
    perm = fit_oof_any_model(
        X=X,
        y=np.asarray(permuted_y, dtype=np.float32),
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        gap_ft=25.0,
        random_seed=20240605,
    )
    real_s = _metric(real["metrics"], "spearman")
    perm_s = _metric(perm["metrics"], "spearman")
    return None if real_s is None or perm_s is None else float(real_s - perm_s)


def transfer_for_candidate(
    *,
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    return {
        "B_to_C": fit_eval_once_any(
            X,
            y,
            train_mask=(regimes == "B") & np.isfinite(y),
            validation_mask=(regimes == "C") & np.isfinite(y),
            model_name=model_name,
            sklearn_modules=sklearn_modules,
        ),
        "C_to_B": fit_eval_once_any(
            X,
            y,
            train_mask=(regimes == "C") & np.isfinite(y),
            validation_mask=(regimes == "B") & np.isfinite(y),
            model_name=model_name,
            sklearn_modules=sklearn_modules,
        ),
    }


def fit_eval_once_any(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    if np.count_nonzero(train_mask) < 20 or np.count_nonzero(validation_mask) < 20:
        return {"status": "skipped_insufficient_samples"}
    matrix = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    model = _make_model(model_name, sklearn_modules, MODEL_CONFIG, random_state=20240605)
    model.fit(matrix[train_mask], y[train_mask])
    pred = np.asarray(model.predict(matrix[validation_mask]), dtype=np.float32)
    return {
        "status": "completed",
        "train_count": int(np.count_nonzero(train_mask)),
        "validation_count": int(np.count_nonzero(validation_mask)),
        **compute_regression_metrics(y[validation_mask], pred),
    }


def feature_group_ablation(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    groups: np.ndarray,
    sklearn_modules: dict[str, Any],
    baseline_spearman: float | None,
) -> dict[str, Any]:
    unique = sorted(set(groups.tolist()))
    group_counts = {group: int(np.count_nonzero(groups == group)) for group in unique}
    top_groups = sorted(group_counts, key=group_counts.get, reverse=True)[:8]
    rows = []
    for group in top_groups:
        keep = groups != group
        result = fit_oof_any_model(
            X=X[:, keep],
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=MODEL_CONFIG,
            gap_ft=25.0,
            random_seed=20240800 + len(rows),
        )
        spearman = _metric(result["metrics"], "spearman")
        rows.append(
            {
                "removed_group": group,
                "removed_feature_count": group_counts[group],
                "spearman_after_removal": spearman,
                "spearman_delta_after_minus_baseline": None
                if spearman is None or baseline_spearman is None
                else spearman - baseline_spearman,
            }
        )
    return {"status": "completed", "rows": rows}


def permutation_importance_summary(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    feature_names: np.ndarray,
    feature_groups: np.ndarray,
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    matrix = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    folds = contiguous_depth_folds(depth, sample_mask, n_folds=3)
    validation_mask = folds[-1] & sample_mask
    train_mask = sample_mask & ~validation_mask
    if np.count_nonzero(train_mask) < 20 or np.count_nonzero(validation_mask) < 20:
        return {"status": "skipped_insufficient_samples", "top_features": [], "stable_groups": []}
    model = _make_model(model_name, sklearn_modules, MODEL_CONFIG, random_state=20240900)
    model.fit(matrix[train_mask], y[train_mask])
    baseline_pred = np.asarray(model.predict(matrix[validation_mask]), dtype=np.float32)
    baseline_mae = float(np.mean(np.abs(baseline_pred - y[validation_mask])))
    correlations = []
    for index in range(matrix.shape[1]):
        value = _safe_spearman(matrix[train_mask, index], y[train_mask])
        correlations.append(0.0 if value is None else abs(value))
    candidate_indices = np.argsort(correlations)[-min(30, matrix.shape[1]) :][::-1]
    rng = np.random.default_rng(20240605)
    rows = []
    for index in candidate_indices:
        permuted = matrix[validation_mask].copy()
        permuted[:, index] = rng.permutation(permuted[:, index])
        pred = np.asarray(model.predict(permuted), dtype=np.float32)
        mae = float(np.mean(np.abs(pred - y[validation_mask])))
        rows.append(
            {
                "feature": str(feature_names[index]),
                "feature_group": str(feature_groups[index]),
                "mae_delta": mae - baseline_mae,
                "baseline_mae": baseline_mae,
                "permuted_mae": mae,
            }
        )
    rows.sort(key=lambda item: float(item["mae_delta"]), reverse=True)
    group_scores: dict[str, list[float]] = {}
    for row in rows:
        group_scores.setdefault(str(row["feature_group"]), []).append(float(row["mae_delta"]))
    stable_groups = [
        {
            "feature_group": group,
            "mean_mae_delta": float(np.mean(values)),
            "max_mae_delta": float(np.max(values)),
            "feature_count": len(values),
        }
        for group, values in group_scores.items()
    ]
    stable_groups.sort(key=lambda item: float(item["mean_mae_delta"]), reverse=True)
    return {"status": "completed", "top_features": rows[:30], "stable_groups": stable_groups[:12]}


def decide_fixed_time_frequency_v2(
    matrix: dict[str, Any],
    strict: dict[str, Any],
    final_report: dict[str, Any],
) -> dict[str, Any]:
    baseline = select_matching_row(
        matrix["rows"],
        feature_set="existing_features_only",
        target="receiver_mean",
        model="Ridge",
        cohort="pooled_bc_all",
    )
    best_v2 = select_best_model_row(
        [
            row
            for row in matrix["rows"]
            if row["feature_set"]
            in {"time_frequency_v2_only", "waveform_v1_plus_v2", "existing_plus_v2", "all_combined"}
            and row["target"] == "receiver_mean"
        ]
    )
    if not baseline or not best_v2:
        return {
            "decision": "fixed_time_frequency_v2_not_helpful",
            "fixed_time_frequency_v2_helpful": False,
            "reason": "missing_baseline_or_v2_candidate",
            "interpretation": "No valid baseline/v2 comparison row was available.",
        }
    base_s = _metric(baseline["metrics"], "spearman")
    v2_s = _metric(best_v2["metrics"], "spearman")
    base_lift = _nested_lift_from_metrics(baseline["metrics"], "top_10pct")
    v2_lift = _nested_lift_from_metrics(best_v2["metrics"], "top_10pct")
    baseline_strict = _as_dict(
        _as_dict(strict["candidates"]).get("baseline_existing_ridge_receiver_mean_pooled_bc_all")
    )
    v2_strict = _as_dict(_as_dict(strict["candidates"]).get("best_v2_receiver_mean"))
    margin_values = [
        value
        for value in _as_dict(v2_strict.get("permutation_margins")).values()
        if value is not None
    ]
    base_gap_mean = _metric(
        _as_dict(_as_dict(baseline_strict.get("gap_stability")).get("score_stability")),
        "mean",
    )
    v2_gap_mean = _metric(
        _as_dict(_as_dict(v2_strict.get("gap_stability")).get("score_stability")),
        "mean",
    )
    base_cal = _metric(baseline, "max_abs_calibration_bias")
    v2_cal = _metric(best_v2, "max_abs_calibration_bias")
    base_transfer = min_transfer_spearman(_as_dict(baseline_strict.get("transfer")))
    v2_transfer = min_transfer_spearman(_as_dict(v2_strict.get("transfer")))
    checks = {
        "spearman_improved": v2_s is not None and base_s is not None and v2_s >= base_s + 0.03,
        "top10_lift_improved": v2_lift is not None
        and base_lift is not None
        and v2_lift >= base_lift + 0.20,
        "blocked_gap_stability_not_degraded": v2_gap_mean is None
        or base_gap_mean is None
        or v2_gap_mean <= base_gap_mean + 0.01,
        "permutation_margins_positive": bool(margin_values)
        and all(float(value) > 0.0 for value in margin_values),
        "calibration_not_obviously_worse": v2_cal is None
        or base_cal is None
        or v2_cal <= base_cal + 0.05,
        "domain_shift_not_obviously_worse": v2_transfer is None
        or base_transfer is None
        or v2_transfer >= base_transfer - 0.03,
    }
    helpful = all(checks.values())
    return {
        "decision": "fixed_time_frequency_v2_helpful"
        if helpful
        else "fixed_time_frequency_v2_not_helpful",
        "fixed_time_frequency_v2_helpful": helpful,
        "baseline": flatten_model_row(baseline),
        "best_v2": flatten_model_row(best_v2),
        "delta_spearman": None if base_s is None or v2_s is None else v2_s - base_s,
        "delta_top10_lift": None if base_lift is None or v2_lift is None else v2_lift - base_lift,
        "checks": checks,
        "current_final_oof_reference": final_report.get("metrics", {}),
        "interpretation": (
            "Fixed TF-v2 met the bounded improvement gates."
            if helpful
            else (
                "Fixed TF-v2 did not meet the simultaneous improvement gates. Do not tune "
                "frequency bands dynamically; keep results as research-only evidence."
            )
        ),
    }


def format_feature_set_comparison_markdown(report: dict[str, Any]) -> str:
    decision = report["decision_summary"]
    best = _as_dict(report["matrix_summary"].get("best_overall"))
    return "\n".join(
        [
            "# MVP-4X Feature Set v2 Comparison",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- decision: `{decision['decision']}`",
            f"- fixed_time_frequency_v2_helpful: {decision['fixed_time_frequency_v2_helpful']}",
            f"- row_count: {report['matrix_summary']['row_count']}",
            f"- best_feature_set: {best.get('feature_set')}",
            f"- best_target: {best.get('target')}",
            f"- best_model: {best.get('model')}",
            f"- best_cohort: {best.get('cohort')}",
            f"- best_spearman: {_metric(best.get('metrics', {}), 'spearman')}",
            f"- delta_spearman_vs_existing_ridge: {decision.get('delta_spearman')}",
            f"- delta_top10_lift_vs_existing_ridge: {decision.get('delta_top10_lift')}",
            "",
            report["interpretation"],
            "",
            "No final labels or production model claim are authorized.",
            "",
        ]
    )


def run_model_consolidation(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    morphology = _read_json(paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.json")
    comparison = _read_json(paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.json")
    final_report = _read_json(paths.reports / "mvp4x_screening_scores_final_oof_v001.json")
    target_policy = choose_target_view_policy(
        comparison,
        csv_path=paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.csv",
    )
    best = _as_dict(target_policy.get("selected_non_audit_row"))
    decision = comparison["decision_summary"]
    morphology_recommended = morphology["decision_summary"]["recommended_variants"]
    consolidation = {
        "report_version": f"{CONSOLIDATION_VERSION}_v001",
        "generated_at": _utc_now(),
        "best_label_policy": (
            "morphology_parallel_candidate"
            if morphology_recommended
            else "morphology_candidates_not_helpful"
        ),
        "v1_label_policy_replaced": False,
        "best_target_view": target_policy["best_target_view"],
        "target_view_policy": target_policy,
        "receiver_p90_recommended_research_candidate": target_policy[
            "receiver_p90_recommended_research_candidate"
        ],
        "receiver_max_audit_only": True,
        "best_feature_set": best.get("feature_set"),
        "best_classical_model": best.get("model"),
        "best_supported_cohort_policy": best.get("cohort"),
        "supported_cohorts": list(SUPPORTED_COHORTS),
        "do_not_auto_adopt_orientation_filter": True,
        "do_not_auto_adopt_production_regime_routing": True,
        "negative_results_visible": True,
        "single_cohort_improvement_warning": improvement_single_cohort_warning(comparison),
        "morphology_decision": morphology.get("decision"),
        "feature_set_v2_decision": decision.get("decision"),
        "final_oof_reference": final_report.get("metrics", {}),
        "interpretation": (
            "Consolidation keeps v1 as the baseline label policy. Any morphology or TF-v2 "
            "change remains research-only until human review approves a formal semantics "
            "or feature policy change."
        ),
        **METHOD_FLAGS,
    }
    output_json = paths.reports / f"{CONSOLIDATION_VERSION}.json"
    output_md = paths.reports / f"{CONSOLIDATION_VERSION}.md"
    for path in (output_json, output_md):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(_json(consolidation), encoding="utf-8")
    output_md.write_text(format_consolidation_markdown(consolidation), encoding="utf-8")
    append_next_iteration_log(
        paths,
        iteration_id="005",
        files_read=[
            str(paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.json"),
            str(paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.json"),
            str(paths.reports / "mvp4x_screening_scores_final_oof_v001.json"),
        ],
        hypothesis="The bounded audits can consolidate a research policy without replacing v1.",
        evidence_before={
            "morphology_decision": morphology.get("decision"),
            "feature_set_decision": decision.get("decision"),
        },
        selected_action="write_model_policy_consolidation",
        files_changed=[str(output_json), str(output_md)],
        commands_run=[
            "python scripts/07v_run_mvp4x_next_auto.py --phase consolidation --overwrite",
        ],
        outputs_generated=[str(output_json), str(output_md)],
        metrics_after={
            "best_label_policy": consolidation["best_label_policy"],
            "best_feature_set": consolidation["best_feature_set"],
            "best_model": consolidation["best_classical_model"],
        },
        interpretation=consolidation["interpretation"],
        decision="continue_to_advanced_route_triage",
        continue_or_stop="continue",
        human_intervention_required=False,
        next_action="advanced_methods_feasibility_report",
        commit_hash_or_pending_commit="pending_commit_after_tests",
    )
    return {"phase": "consolidation", "outputs": [str(output_json), str(output_md)], **METHOD_FLAGS}


def write_advanced_methods_feasibility(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    comparison = _read_json(paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.json")
    helpful = bool(comparison.get("fixed_time_frequency_v2_helpful"))
    routes = [
        feasibility_route(
            name="STC pilot",
            rationale=(
                "Slowness-time coherence can test whether casing-wave and formation-wave "
                "modes carry channel evidence unavailable to shallow FFT features."
            ),
            benefit="Medium to high if XSI mode separation is the bottleneck.",
            risk="High runtime and parameter-sensitivity; still weak-label limited.",
            data="Existing chunked XSI waveforms and v1 weak targets.",
            dependencies=(
                "No new dependency preferred for pilot; use NumPy/SciPy implementation "
                "only after approval."
            ),
            runtime="Server recommended for full well; local tiny pilot only after approval.",
            memory="Potentially >256 MB unless tiled by depth/receiver/side/slowness.",
            disk="Several GB for cached slowness-time tensors.",
            local="Not executed; feasible only as bounded pilot.",
            server="Recommended for full pilot.",
            priority="high" if not helpful else "medium",
            stopping=(
                "Stop if permutation/depth-shift controls do not fall or memory exceeds budget."
            ),
        ),
        feasibility_route(
            name="APES pilot",
            rationale=(
                "Adaptive spectral estimation may improve narrowband stability over fixed "
                "FFT bands."
            ),
            benefit="Medium if dominant-frequency instability is noise-sensitive.",
            risk="High complexity and overfitting to weak-label morphology.",
            data="Existing XSI waveforms.",
            dependencies="May require careful custom implementation; no install without approval.",
            runtime="Server recommended.",
            memory="Moderate to high depending covariance windows.",
            disk="GB-scale if cached for all receiver-side-depth cells.",
            local="Not executed.",
            server="Recommended only after STC/FFT audits justify.",
            priority="medium",
            stopping="Stop if APES does not improve strict permutation margins.",
        ),
        feasibility_route(
            name="lightweight deep-learning pilot",
            rationale=(
                "A small XSI-only model could learn interactions across receiver, side, "
                "and time-frequency structure."
            ),
            benefit="Medium, but only after physical baselines establish enough signal.",
            risk="High leakage and weak-label memorization risk.",
            data="Cached HDF5/Zarr/memmap features only; no CAST inputs.",
            dependencies=(
                "Existing torch only if already installed; no new install without approval."
            ),
            runtime="Server GPU recommended; do not use GPU0.",
            memory="A100 40GB adequate for small model.",
            disk="Model/checkpoints/logs outside Git under /home.",
            local="Not executed.",
            server="Recommended if approved.",
            priority="low_to_medium",
            stopping="Stop if random-label or depth-shift controls remain strong.",
        ),
        feasibility_route(
            name="multiwell data collection",
            rationale="Current domain-shift and single-well limits prevent cross-well claims.",
            benefit="High for validation and deployment readiness.",
            risk="Data access and schema harmonization cost.",
            data="Additional wells with XSI, CAST, Inc, RelBearing and metadata.",
            dependencies="None beyond existing pipeline after mapping.",
            runtime="Server recommended for conversion and QC.",
            memory="Similar per well if chunked.",
            disk="Potentially large; external data root/server /home only.",
            local="Not feasible without new data.",
            server="Recommended after data approval.",
            priority="high",
            stopping="Stop if schema cannot be aligned without changing geometry policy.",
        ),
        feasibility_route(
            name="additional physical metadata",
            rationale=(
                "Casing, cement, collar, logging-speed, and material metadata may explain "
                "confounders currently treated as weak-label uncertainty."
            ),
            benefit="Medium to high for calibration and false-positive triage.",
            risk="Incomplete metadata can add bias.",
            data="Casing/cement program, collars, tool settings, logs.",
            dependencies="None.",
            runtime="Low.",
            memory="Low.",
            disk="Low.",
            local="Feasible after data approval.",
            server="Not required.",
            priority="high",
            stopping="Stop if metadata cannot be linked to depth reliably.",
        ),
        feasibility_route(
            name="label-semantics redesign",
            rationale=(
                "Morphology explains some errors but candidate variants did not necessarily "
                "improve ranking; formal semantics may need expert review."
            ),
            benefit="High if current target is the bottleneck.",
            risk="Requires changing weak-label policy; cannot be automatic.",
            data="Existing morphology fields plus expert review intervals.",
            dependencies="None.",
            runtime="Low for design, medium for regeneration.",
            memory="Low.",
            disk="Small audit artifacts.",
            local="Feasible after approval.",
            server="Not required unless regenerating full labels at scale.",
            priority="high",
            stopping="Stop if new semantics fail pre-registered same-protocol audits.",
        ),
    ]
    report = {
        "report_version": f"{ADVANCED_FEASIBILITY_VERSION}_v001",
        "generated_at": _utc_now(),
        "trigger": (
            "classical_fixed_tfv2_insufficient"
            if not helpful
            else "advanced_route_triage_generated_for_future_review"
        ),
        "not_executed": [
            "STC",
            "APES",
            "deep_learning",
            "new_dependency_install",
            "server_migration",
        ],
        "routes": routes,
        "recommendation": (
            "request_stc_apes_pilot_approval"
            if not helpful
            else "continue_human_review_of_fixed_time_frequency_v2"
        ),
        **METHOD_FLAGS,
    }
    output_json = paths.reports / f"{ADVANCED_FEASIBILITY_VERSION}.json"
    output_md = paths.reports / f"{ADVANCED_FEASIBILITY_VERSION}.md"
    for path in (output_json, output_md):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(_json(report), encoding="utf-8")
    output_md.write_text(format_advanced_feasibility_markdown(report), encoding="utf-8")
    append_next_iteration_log(
        paths,
        iteration_id="006",
        files_read=[str(paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.json")],
        hypothesis=(
            "If bounded morphology semantics and fixed TF-v2 remain insufficient, the next "
            "high-value actions require explicit human approval."
        ),
        evidence_before={"fixed_time_frequency_v2_helpful": helpful},
        selected_action="write_advanced_methods_feasibility_without_executing_advanced_routes",
        files_changed=[str(output_json), str(output_md)],
        commands_run=[
            "python scripts/07v_run_mvp4x_next_auto.py --phase feasibility --overwrite",
        ],
        outputs_generated=[str(output_json), str(output_md)],
        metrics_after={"recommendation": report["recommendation"]},
        interpretation="Advanced methods are triaged but not executed.",
        decision="continue_to_final_review_pack",
        continue_or_stop="continue",
        human_intervention_required=False,
        next_action="next_auto_decision_and_review_pack",
        commit_hash_or_pending_commit="pending_commit_after_tests",
    )
    return {"phase": "feasibility", "outputs": [str(output_json), str(output_md)], **METHOD_FLAGS}


def write_next_auto_decision_and_review_pack(
    paths: MaxAutoPaths, *, overwrite: bool
) -> dict[str, Any]:
    morphology = _read_json(paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.json")
    feature = _read_json(paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.json")
    consolidation = _read_json(paths.reports / f"{CONSOLIDATION_VERSION}.json")
    feasibility = _read_json(paths.reports / f"{ADVANCED_FEASIBILITY_VERSION}.json")
    final_scores = _read_json(paths.reports / "mvp4x_screening_scores_final_oof_v001.json")
    decision = choose_final_decision(morphology, feature, feasibility)
    review_dir = paths.reports / REVIEW_DIR_NAME
    if review_dir.exists() and overwrite:
        shutil.rmtree(review_dir)
    if review_dir.exists() and not overwrite:
        raise FileExistsError(f"Review directory exists: {review_dir}")
    review_dir.mkdir(parents=True, exist_ok=True)
    review_files = write_next_review_pack(
        paths=paths,
        review_dir=review_dir,
        morphology=morphology,
        feature=feature,
        consolidation=consolidation,
        final_scores=final_scores,
    )
    report = {
        "report_version": f"{NEXT_DECISION_VERSION}_v001",
        "generated_at": _utc_now(),
        "decision": decision,
        "autonomous_iterations_completed": 7,
        "best_label_policy": consolidation["best_label_policy"],
        "best_target_view": consolidation["best_target_view"],
        "best_feature_set": consolidation["best_feature_set"],
        "best_model": consolidation["best_classical_model"],
        "supported_cohorts": consolidation["supported_cohorts"],
        "morphology_variants_improved": bool(
            morphology["decision_summary"]["recommended_variants"]
        ),
        "fixed_time_frequency_v2_extracted": True,
        "fixed_time_frequency_v2_helpful": bool(feature["fixed_time_frequency_v2_helpful"]),
        "advanced_feasibility_report": str(paths.reports / f"{ADVANCED_FEASIBILITY_VERSION}.json"),
        "review_dir": str(review_dir),
        "review_files": review_files,
        "next_human_approval_required": next_human_approval(decision),
        "recommend_server": decision
        in {
            "request_stc_apes_pilot_approval",
            "request_deep_learning_pilot_approval",
            "request_server_migration_approval",
            "request_multiwell_data_approval",
        },
        "production_claim_and_final_labels_forbidden": True,
        "final_labels_still_forbidden": True,
        "production_claim_still_forbidden": True,
        "source_reports": {
            "morphology": str(paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.json"),
            "feature_set": str(paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.json"),
            "consolidation": str(paths.reports / f"{CONSOLIDATION_VERSION}.json"),
            "feasibility": str(paths.reports / f"{ADVANCED_FEASIBILITY_VERSION}.json"),
            "final_oof": str(paths.reports / "mvp4x_screening_scores_final_oof_v001.json"),
        },
        **METHOD_FLAGS,
    }
    output_json = paths.reports / f"{NEXT_DECISION_VERSION}.json"
    output_md = paths.reports / f"{NEXT_DECISION_VERSION}.md"
    for path in (output_json, output_md):
        _ensure_can_write(path, overwrite=overwrite)
    output_json.write_text(_json(report), encoding="utf-8")
    output_md.write_text(format_next_decision_markdown(report), encoding="utf-8")
    append_next_iteration_log(
        paths,
        iteration_id="007",
        files_read=list(report["source_reports"].values()),
        hypothesis="All bounded local actions are now complete; only true approval gates remain.",
        evidence_before={
            "morphology_decision": morphology["decision"],
            "feature_decision": feature["decision_summary"]["decision"],
            "feasibility_recommendation": feasibility["recommendation"],
        },
        selected_action="write_final_next_auto_decision_and_review_pack",
        files_changed=[str(output_json), str(output_md), str(review_dir)],
        commands_run=[
            "python scripts/07v_run_mvp4x_next_auto.py --phase review --overwrite",
        ],
        outputs_generated=[str(output_json), str(output_md), *review_files.values()],
        metrics_after={"decision": decision},
        interpretation=(
            "Bounded autonomous loop completed without final labels or production claims."
        ),
        decision=decision,
        continue_or_stop="stop_true_human_approval_required",
        human_intervention_required=True,
        next_action=report["next_human_approval_required"],
        commit_hash_or_pending_commit="pending_commit_after_tests",
    )
    return {
        "phase": "review",
        "decision": decision,
        "outputs": [str(output_json), str(output_md), str(review_dir)],
        **METHOD_FLAGS,
    }


def write_next_review_pack(
    *,
    paths: MaxAutoPaths,
    review_dir: Path,
    morphology: dict[str, Any],
    feature: dict[str, Any],
    consolidation: dict[str, Any],
    final_scores: dict[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: dict[str, str] = {}
    summary = review_dir / "review_summary.md"
    summary.write_text(
        format_review_summary(morphology, feature, consolidation, final_scores),
        encoding="utf-8",
    )
    outputs["review_summary"] = str(summary)
    checklist = review_dir / "reviewer_checklist.md"
    checklist.write_text(format_reviewer_checklist(), encoding="utf-8")
    outputs["reviewer_checklist"] = str(checklist)
    final_npz = _load_npz(paths.interim / "mvp4x_screening_scores_final_oof_v001.npz")
    final_csv = paths.reports / "mvp4x_screening_scores_final_oof_v001.csv"
    interval_csv = paths.reports / "mvp4x_screening_review_intervals_v001.csv"
    write_interval_exports(final_npz, final_csv, interval_csv, review_dir, outputs)

    morph_csv = _read_csv(paths.reports / f"{MORPHOLOGY_AUDIT_VERSION}.csv")
    feature_csv = _read_csv(paths.reports / f"{FEATURE_SET_COMPARISON_VERSION}.csv")
    plot_bar(
        review_dir / "v1_vs_morphology_variants.png",
        [row["target"] for row in morph_csv],
        [_float(row.get("spearman")) for row in morph_csv],
        "Spearman",
        plt,
    )
    outputs["v1_vs_morphology_variants"] = str(review_dir / "v1_vs_morphology_variants.png")
    best_by_feature = {}
    for row in feature_csv:
        name = row["feature_set"]
        current = best_by_feature.get(name)
        if current is None or _float(row.get("spearman")) > _float(current.get("spearman")):
            best_by_feature[name] = row
    plot_bar(
        review_dir / "feature_set_v2_comparison.png",
        list(best_by_feature),
        [_float(row.get("spearman")) for row in best_by_feature.values()],
        "Best Spearman",
        plt,
    )
    outputs["feature_set_v2_comparison"] = str(review_dir / "feature_set_v2_comparison.png")
    plot_bar(
        review_dir / "top_k_lift_comparison.png",
        list(best_by_feature),
        [_float(row.get("top10_lift")) for row in best_by_feature.values()],
        "Top-10 lift",
        plt,
    )
    outputs["top_k_lift_comparison"] = str(review_dir / "top_k_lift_comparison.png")
    strict = _as_dict(feature.get("strict_audits")).get("candidates", {})
    write_strict_metric_plot(
        review_dir / "blocked_gap_comparison.png",
        strict,
        "gap_stability",
        plt,
    )
    outputs["blocked_gap_comparison"] = str(review_dir / "blocked_gap_comparison.png")
    write_strict_metric_plot(
        review_dir / "permutation_margin_comparison.png",
        strict,
        "permutation_margins",
        plt,
    )
    outputs["permutation_margin_comparison"] = str(review_dir / "permutation_margin_comparison.png")
    plot_bar(
        review_dir / "calibration_comparison.png",
        list(best_by_feature),
        [_float(row.get("max_abs_calibration_bias")) for row in best_by_feature.values()],
        "Max abs calibration bias",
        plt,
    )
    outputs["calibration_comparison"] = str(review_dir / "calibration_comparison.png")
    write_domain_shift_plot(review_dir / "domain_shift_comparison.png", strict, plt)
    outputs["domain_shift_comparison"] = str(review_dir / "domain_shift_comparison.png")
    write_stable_features_plot(review_dir / "stable_features.png", strict, plt)
    outputs["stable_features"] = str(review_dir / "stable_features.png")
    return outputs


def write_interval_exports(
    final_npz: dict[str, np.ndarray],
    final_csv: Path,
    interval_csv: Path,
    review_dir: Path,
    outputs: dict[str, str],
) -> None:
    score = np.asarray(final_npz["score"], dtype=np.float32)
    target = np.asarray(final_npz["target_receiver_mean"], dtype=np.float32)
    rank = np.asarray(final_npz["global_rank_percentile"], dtype=np.float32)
    depth = np.asarray(final_npz["depth"], dtype=np.float32)
    supported = np.isfinite(score)
    fp = supported & (rank >= 0.90) & (target <= np.nanquantile(target[supported], 0.50))
    fn = supported & (rank <= 0.50) & (target >= np.nanquantile(target[supported], 0.90))
    top = supported & (rank >= 0.95)
    for name, mask in (
        ("false_positive_like_intervals.csv", fp),
        ("false_negative_like_intervals.csv", fn),
        ("top_ranked_intervals.csv", top),
    ):
        rows = []
        indices = np.flatnonzero(mask)
        if name == "false_negative_like_intervals.csv":
            order = indices[np.argsort((target - score)[indices])[::-1]]
        else:
            order = indices[np.argsort(rank[indices])[::-1]]
        for index in order[:50]:
            rows.append(
                {
                    "sample_index": int(index),
                    "depth": float(depth[index]),
                    "score": float(score[index]),
                    "target_receiver_mean": float(target[index]),
                    "rank": float(rank[index]),
                    **METHOD_FLAGS,
                }
            )
        path = review_dir / name
        _write_csv(rows, path)
        outputs[name.removesuffix(".csv")] = str(path)
    for source, output_name in (
        (final_csv, "source_final_scores_csv"),
        (interval_csv, "source_review_intervals_csv"),
    ):
        if source.exists():
            outputs[output_name] = str(source)


def choose_final_decision(
    morphology: dict[str, Any],
    feature: dict[str, Any],
    feasibility: dict[str, Any],
) -> str:
    if morphology["decision_summary"]["recommended_variants"]:
        return "morphology_parallel_candidate_recommended_for_human_review"
    if bool(feature.get("fixed_time_frequency_v2_helpful")):
        return "fixed_time_frequency_v2_helpful"
    if feasibility.get("recommendation") == "request_stc_apes_pilot_approval":
        return "request_stc_apes_pilot_approval"
    return "stop_insufficient_signal"


def next_human_approval(decision: str) -> str:
    mapping = {
        "morphology_parallel_candidate_recommended_for_human_review": (
            "Approve or reject formal review of the recommended parallel morphology candidate; "
            "do not replace v1 automatically."
        ),
        "fixed_time_frequency_v2_helpful": (
            "Approve whether fixed TF-v2 may become a formal research feature policy candidate."
        ),
        "request_stc_apes_pilot_approval": (
            "Approve or reject a bounded STC/APES pilot; no STC/APES has been executed."
        ),
        "stop_insufficient_signal": (
            "Review insufficient-signal outcome and decide whether to add data or redesign labels."
        ),
    }
    return mapping.get(decision, "Human review required before any policy change.")


def format_next_decision_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X Next Auto Decision",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- decision: `{report['decision']}`",
            f"- autonomous_iterations_completed: {report['autonomous_iterations_completed']}",
            f"- best_label_policy: {report['best_label_policy']}",
            f"- best_target_view: {report['best_target_view']}",
            f"- best_feature_set: {report['best_feature_set']}",
            f"- best_model: {report['best_model']}",
            f"- morphology_variants_improved: {report['morphology_variants_improved']}",
            f"- fixed_time_frequency_v2_extracted: {report['fixed_time_frequency_v2_extracted']}",
            f"- fixed_time_frequency_v2_helpful: {report['fixed_time_frequency_v2_helpful']}",
            f"- recommend_server: {report['recommend_server']}",
            f"- next_human_approval_required: {report['next_human_approval_required']}",
            "",
            "Final labels and production claims remain forbidden.",
            "",
        ]
    )


def feasibility_route(
    *,
    name: str,
    rationale: str,
    benefit: str,
    risk: str,
    data: str,
    dependencies: str,
    runtime: str,
    memory: str,
    disk: str,
    local: str,
    server: str,
    priority: str,
    stopping: str,
) -> dict[str, Any]:
    return {
        "route": name,
        "scientific_rationale": rationale,
        "expected_benefit": benefit,
        "main_risk": risk,
        "required_data": data,
        "required_dependencies": dependencies,
        "expected_runtime": runtime,
        "expected_memory": memory,
        "expected_disk": disk,
        "local_feasibility": local,
        "server_recommendation": server,
        "priority": priority,
        "stopping_criteria": stopping,
    }


def format_advanced_feasibility_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Advanced Methods Feasibility",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
        f"- recommendation: `{report['recommendation']}`",
        f"- not_executed: {report['not_executed']}",
        "",
    ]
    for route in report["routes"]:
        lines.extend(
            [
                f"## {route['route']}",
                f"- priority: {route['priority']}",
                f"- scientific_rationale: {route['scientific_rationale']}",
                f"- expected_benefit: {route['expected_benefit']}",
                f"- main_risk: {route['main_risk']}",
                f"- server_recommendation: {route['server_recommendation']}",
                f"- stopping_criteria: {route['stopping_criteria']}",
                "",
            ]
        )
    return "\n".join(lines)


def format_consolidation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X Next Auto Model Consolidation",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- best_label_policy: {report['best_label_policy']}",
            f"- v1_label_policy_replaced: {report['v1_label_policy_replaced']}",
            f"- best_target_view: {report['best_target_view']}",
            f"- best_feature_set: {report['best_feature_set']}",
            f"- best_classical_model: {report['best_classical_model']}",
            f"- best_supported_cohort_policy: {report['best_supported_cohort_policy']}",
            f"- single_cohort_improvement_warning: {report['single_cohort_improvement_warning']}",
            "",
            report["interpretation"],
            "",
        ]
    )


def choose_target_view_policy(comparison: dict[str, Any], *, csv_path: Path) -> dict[str, Any]:
    best_overall = _as_dict(_as_dict(comparison.get("matrix_summary")).get("best_overall"))
    csv_rows = _read_csv(csv_path)
    non_audit = [row for row in csv_rows if row.get("target") in {"receiver_mean", "receiver_p90"}]
    mean_best = max(
        [row for row in non_audit if row.get("target") == "receiver_mean"],
        key=lambda row: _float(row.get("spearman")),
    )
    p90_best = max(
        [row for row in non_audit if row.get("target") == "receiver_p90"],
        key=lambda row: _float(row.get("spearman")),
    )
    mean_s = _float(mean_best.get("spearman"))
    p90_s = _float(p90_best.get("spearman"))
    mean_lift = _float(mean_best.get("top10_lift"))
    p90_lift = _float(p90_best.get("top10_lift"))
    p90_candidate = bool(p90_s >= mean_s + 0.03 or p90_lift >= mean_lift + 0.20)
    selected = p90_best if p90_candidate else mean_best
    return {
        "best_target_view": "receiver_p90" if p90_candidate else "receiver_mean",
        "receiver_p90_recommended_research_candidate": p90_candidate,
        "receiver_max_audit_only": True,
        "receiver_max_not_selected_for_policy": True,
        "audit_only_best_overall": best_overall
        if best_overall.get("target") == "receiver_max"
        else None,
        "receiver_mean_best": mean_best,
        "receiver_p90_best": p90_best,
        "selected_non_audit_row": selected,
    }


def format_review_summary(
    morphology: dict[str, Any],
    feature: dict[str, Any],
    consolidation: dict[str, Any],
    final_scores: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# MVP-4X Next Auto Review Summary",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- morphology_decision: `{morphology['decision']}`",
            f"- feature_set_v2_decision: `{feature['decision_summary']['decision']}`",
            f"- best_label_policy: `{consolidation['best_label_policy']}`",
            f"- best_target_view: `{consolidation['best_target_view']}`",
            f"- best_feature_set: `{consolidation['best_feature_set']}`",
            f"- best_model: `{consolidation['best_classical_model']}`",
            f"- current_oof_top_k: `{final_scores.get('metrics', {}).get('top_k_lift')}`",
            "",
            "No final labels or production claims are authorized by this pack.",
            "",
        ]
    )


def format_reviewer_checklist() -> str:
    return "\n".join(
        [
            "# Reviewer Checklist",
            "",
            "- Confirm every artifact is research_only and exploratory_only.",
            "- Confirm no final labels were generated.",
            "- Confirm no ground-truth or production claim is made.",
            "- Inspect morphology variants before any label semantics change.",
            "- Inspect TF-v2 improvements across cohorts, not only a single cohort.",
            "- Reject any route that requires STC, APES, deep learning, server migration, "
            "new dependencies, or new data unless explicitly approved.",
            "- Keep XSI-only leakage boundary intact; do not use CAST fields as model inputs.",
            "",
        ]
    )


def append_next_iteration_log(
    paths: MaxAutoPaths,
    *,
    iteration_id: str,
    files_read: list[str],
    hypothesis: str,
    evidence_before: dict[str, Any],
    selected_action: str,
    files_changed: list[str],
    commands_run: list[str],
    outputs_generated: list[str],
    metrics_after: dict[str, Any],
    interpretation: str,
    decision: str,
    continue_or_stop: str,
    human_intervention_required: bool,
    next_action: str,
    commit_hash_or_pending_commit: str,
) -> None:
    path = paths.reports / "mvp4x_next_auto_iteration_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# MVP-4X Next Auto Iteration Log\n\n"
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.\n\n",
            encoding="utf-8",
        )
    entry = [
        f"## iteration_id: {iteration_id}",
        f"- timestamp: {_utc_now()}",
        f"- files_read: `{json.dumps(files_read, ensure_ascii=False)}`",
        f"- hypothesis: {hypothesis}",
        f"- evidence_before: `{_json(evidence_before)}`",
        f"- selected_action: {selected_action}",
        f"- files_changed: `{json.dumps(files_changed, ensure_ascii=False)}`",
        f"- commands_run: `{json.dumps(commands_run, ensure_ascii=False)}`",
        f"- outputs_generated: `{json.dumps(outputs_generated, ensure_ascii=False)}`",
        f"- metrics_after: `{_json(metrics_after)}`",
        f"- interpretation: {interpretation}",
        f"- decision: {decision}",
        f"- continue_or_stop: {continue_or_stop}",
        f"- human_intervention_required: {human_intervention_required}",
        f"- next_action: {next_action}",
        f"- commit_hash_or_pending_commit: {commit_hash_or_pending_commit}",
        "- scope_flags: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))


def flatten_model_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(row.get("metrics"))
    return {
        "cohort": row.get("cohort"),
        "feature_set": row.get("feature_set"),
        "target": row.get("target"),
        "model": row.get("model"),
        "sample_count": row.get("sample_count"),
        "spearman": _metric(metrics, "spearman"),
        "mae": _metric(metrics, "mae"),
        "rmse": _metric(metrics, "rmse"),
        "r2": _metric(metrics, "r2"),
        "kendall_tau": _metric(metrics, "kendall_tau"),
        "ndcg": _metric(metrics, "ndcg"),
        "ordinal_macro_f1": _metric(metrics, "ordinal_macro_f1"),
        "top5_lift": _nested_lift_from_metrics(metrics, "top_5pct"),
        "top10_lift": _nested_lift_from_metrics(metrics, "top_10pct"),
        "top20_lift": _nested_lift_from_metrics(metrics, "top_20pct"),
        "max_abs_calibration_bias": row.get("max_abs_calibration_bias"),
        **METHOD_FLAGS,
    }


def select_best_model_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("model") != "DummyRegressor"
        and _metric(row.get("metrics", {}), "spearman") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(_metric(row["metrics"], "spearman") or -999.0))


def select_matching_row(
    rows: list[dict[str, Any]],
    *,
    feature_set: str,
    target: str,
    model: str,
    cohort: str,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("feature_set") == feature_set
            and row.get("target") == target
            and row.get("model") == model
            and row.get("cohort") == cohort
        ):
            return row
    return None


def summarize_v2_feature_sets(feature_sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for name, feature_set in feature_sets.items():
        matrix = np.asarray(feature_set["matrix"], dtype=np.float32)
        groups = np.asarray(feature_set["groups"]).astype(str)
        unique, counts = np.unique(groups, return_counts=True)
        output[name] = {
            "sample_count": int(matrix.shape[0]),
            "feature_count": int(matrix.shape[1]),
            "finite_ratio": float(np.isfinite(matrix).mean()),
            "group_counts": {
                str(group): int(count) for group, count in zip(unique, counts, strict=True)
            },
        }
    return output


def residual_by_regime(
    snapshot: dict[str, np.ndarray],
    y: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    residual = prediction - y
    return {
        regime: _summary(residual[valid & (regimes == regime)])
        for regime in sorted(set(regimes.tolist()))
    }


def bootstrap_spearman_ci(
    y_true: np.ndarray,
    prediction: np.ndarray,
    *,
    repeats: int = 80,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(prediction, dtype=np.float32)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size < 20:
        return {"status": "skipped_too_few_samples"}
    rng = np.random.default_rng(20240605)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, y.size, size=y.size)
        value = _safe_spearman(y[indices], p[indices])
        if value is not None:
            values.append(value)
    if not values:
        return {"status": "skipped_degenerate"}
    return {
        "status": "completed",
        "repeats": repeats,
        "spearman_p05": float(np.quantile(values, 0.05)),
        "spearman_p50": float(np.quantile(values, 0.50)),
        "spearman_p95": float(np.quantile(values, 0.95)),
    }


def ndcg_score(y_true: np.ndarray, prediction: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size == 0:
        return None
    order = np.argsort(p)[::-1]
    ideal = np.argsort(y)[::-1]
    discount = 1.0 / np.log2(np.arange(2, y.size + 2))
    dcg = float(np.sum(y[order] * discount))
    idcg = float(np.sum(y[ideal] * discount))
    return None if idcg <= 0.0 else dcg / idcg


def ordinal_macro_f1(y_true: np.ndarray, prediction: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size < 3 or np.unique(y).size < 3:
        return None
    q1, q2 = np.quantile(y, [1.0 / 3.0, 2.0 / 3.0])
    truth = np.digitize(y, [q1, q2])
    pred = np.digitize(p, [q1, q2])
    scores = []
    for label in (0, 1, 2):
        tp = np.count_nonzero((truth == label) & (pred == label))
        fp = np.count_nonzero((truth != label) & (pred == label))
        fn = np.count_nonzero((truth == label) & (pred != label))
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if tp + fn == 0 else tp / (tp + fn)
        if precision is None or recall is None or precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(float(2.0 * precision * recall / (precision + recall)))
    return float(np.mean(scores))


def safe_kendall(y_true: np.ndarray, prediction: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    if np.count_nonzero(mask) < 2:
        return None
    value = kendalltau(y[mask], p[mask]).statistic
    return None if not np.isfinite(value) else float(value)


def max_abs_calibration_bias(rows: list[dict[str, Any]]) -> float | None:
    values = [
        abs(float(row["bias_score_minus_target"]))
        for row in rows
        if row.get("bias_score_minus_target") is not None
    ]
    return None if not values else float(max(values))


def _permute_within_depth_bins(
    y: np.ndarray,
    depth: np.ndarray,
    rng: np.random.Generator,
    *,
    bin_count: int = 10,
) -> np.ndarray:
    output = np.asarray(y, dtype=np.float32).copy()
    edges = np.quantile(depth[np.isfinite(depth)], np.linspace(0.0, 1.0, bin_count + 1))
    for index in range(bin_count):
        if index == bin_count - 1:
            mask = (depth >= edges[index]) & (depth <= edges[index + 1])
        else:
            mask = (depth >= edges[index]) & (depth < edges[index + 1])
        selected = np.flatnonzero(mask & np.isfinite(output))
        if selected.size > 1:
            output[selected] = rng.permutation(output[selected])
    return output


def _permute_blocks(
    y: np.ndarray,
    depth: np.ndarray,
    rng: np.random.Generator,
    *,
    block_count: int = 12,
) -> np.ndarray:
    values = np.asarray(y, dtype=np.float32).copy()
    order = np.argsort(depth)
    blocks = [block for block in np.array_split(order, block_count) if block.size]
    shuffled = blocks.copy()
    rng.shuffle(shuffled)
    output = values.copy()
    for source, dest in zip(blocks, shuffled, strict=True):
        count = min(source.size, dest.size)
        output[source[:count]] = values[dest[:count]]
    return output


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 2:
        return None
    if np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return None
    value = spearmanr(left[mask], right[mask]).statistic
    return None if not np.isfinite(value) else float(value)


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0, "finite_ratio": 0.0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "finite_ratio": float(finite.size / max(array.size, 1)),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _nested_lift_from_metrics(metrics: dict[str, Any], key: str) -> float | None:
    return _top_lift({"ranking": _as_dict(metrics.get("top_k_lift"))}, key)


def _top_lift(summary: dict[str, Any], key: str) -> float | None:
    return _metric(
        _as_dict(_as_dict(_as_dict(summary.get("ranking")).get("top_k")).get(key)),
        "lift",
    )


def _metric(row: Any, key: str) -> float | None:
    if not isinstance(row, dict):
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def min_transfer_spearman(transfer: dict[str, Any]) -> float | None:
    values = [
        _metric(_as_dict(value), "spearman")
        for value in transfer.values()
        if _metric(_as_dict(value), "spearman") is not None
    ]
    return None if not values else float(min(values))


def improvement_single_cohort_warning(comparison: dict[str, Any]) -> bool:
    best_by_feature = _as_dict(
        _as_dict(comparison.get("matrix_summary")).get("best_by_feature_set")
    )
    cohorts = [
        _as_dict(row).get("cohort")
        for name, row in best_by_feature.items()
        if name != "existing_features_only" and row
    ]
    return len(set(cohorts)) <= 1 if cohorts else False


def plot_bar(path: Path, labels: list[str], values: list[float], ylabel: str, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4))
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_strict_metric_plot(path: Path, strict: dict[str, Any], field: str, plt: Any) -> None:
    labels = []
    values = []
    for name, audit in strict.items():
        audit = _as_dict(audit)
        if audit.get("status") != "completed":
            continue
        labels.append(name.replace("_", "\n"))
        if field == "permutation_margins":
            margins = [value for value in _as_dict(audit.get(field)).values() if value is not None]
            values.append(float(np.min(margins)) if margins else float("nan"))
        else:
            values.append(
                _metric(_as_dict(_as_dict(audit.get(field)).get("score_stability")), "mean")
                or float("nan")
            )
    plot_bar(path, labels, values, field, plt)


def write_domain_shift_plot(path: Path, strict: dict[str, Any], plt: Any) -> None:
    labels = []
    values = []
    for name, audit in strict.items():
        audit = _as_dict(audit)
        if audit.get("status") != "completed":
            continue
        labels.append(name.replace("_", "\n"))
        values.append(min_transfer_spearman(_as_dict(audit.get("transfer"))) or float("nan"))
    plot_bar(path, labels, values, "min transfer Spearman", plt)


def write_stable_features_plot(path: Path, strict: dict[str, Any], plt: Any) -> None:
    best = _as_dict(strict.get("best_overall")) or _as_dict(strict.get("best_v2_receiver_mean"))
    if not best:
        best = next(
            (
                _as_dict(value)
                for value in strict.values()
                if _as_dict(value).get("status") == "completed"
            ),
            {},
        )
    groups = _as_dict(_as_dict(best.get("permutation_importance")).get("stable_groups"))
    if isinstance(groups, dict):
        rows = list(groups.values())
    else:
        rows = _as_dict(best.get("permutation_importance")).get("stable_groups", [])
    labels = [str(row.get("feature_group")) for row in rows[:12]]
    values = [_float(row.get("mean_mae_delta")) for row in rows[:12]]
    if not labels:
        labels = ["no_completed_importance"]
        values = [0.0]
    plot_bar(path, labels, values, "mean MAE delta", plt)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise NextAutoError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise NextAutoError(f"Missing JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise NextAutoError(f"JSON must be an object: {path}")
    return data


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json(data: Any) -> str:
    return (
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
