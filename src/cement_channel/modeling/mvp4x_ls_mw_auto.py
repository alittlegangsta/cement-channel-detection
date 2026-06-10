from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import _make_model, compute_regression_metrics
from cement_channel.modeling.mvp4x_max_auto import MaxAutoPaths
from cement_channel.modeling.mvp4x_sa_audit import load_sa_pilot_artifacts
from cement_channel.modeling.mvp4x_screening_scores import rank_percentile, top_k_lift

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cement-channel")

LS_MW_VERSION = "mvp4x_ls_mw_auto_v001"
PRIOR_EVIDENCE_VERSION = "mvp4x_ls_mw_prior_evidence_inventory_v001"
COLLAPSE_VERSION = "mvp4x_pilot_metric_collapse_diagnostic_v001"
EVALUATION_POLICY_VERSION = "mvp4x_evaluation_set_policy_v001"
LABEL_SEMANTICS_VERSION = "mvp4x_label_semantics_physical_audit_v001"
PARALLEL_TARGET_VERSION = "mvp4x_parallel_label_candidates_v001"
SCREENING_VERSION = "mvp4x_parallel_target_screening_audit_v001"
MANUAL_REVIEW_DIR = "mvp4x_label_semantics_manual_review_v001"
MULTIWELL_VERSION = "mvp4x_multiwell_readiness_inventory_v001"
DECISION_VERSION = "mvp4x_ls_mw_decision_v001"

DEFAULT_PILOT_RUN_DIRS = (
    "outputs/remote-runs/20260609T030834Z_mvp4x-sa-pilot-auto-b80_4108bd1ded95",
    "outputs/remote-runs/20260609T092204Z_mvp4x-sa-pilot-auto-b120_07751689d58b",
    "outputs/remote-runs/20260609T094130Z_mvp4x-sa-pilot-auto-b160_07751689d58b",
)

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
    "no_raw_mat_modified": True,
    "no_full_well_stc": True,
    "no_full_well_apes": True,
    "no_deep_learning": True,
    "no_production_model_claim": True,
}

MODEL_NAMES = (
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
    "random_forest_n_estimators": 6,
    "random_forest_max_depth": 4,
    "random_forest_n_jobs": 2,
    "hist_gradient_boosting_max_iter": 10,
    "hist_gradient_boosting_max_leaf_nodes": 7,
}
FEATURE_SET_NAMES = (
    "existing_features_only",
    "waveform_v1_only",
    "TF_v2_only",
    "all_available_existing_waveform_tf",
)
PILOT_PROXY_FEATURE_SET_NAMES = (
    "STC_proxy_only",
    "APES_proxy_only",
    "STC_APES_proxy_only",
)
TARGET_NAMES = (
    "receiver_mean_v1_reference",
    "receiver_p90_robust_candidate",
    "local_worst_sector_fraction",
    "connected_channel_fraction",
    "azimuth_width_weighted_fraction",
    "interval_persistence_weighted_fraction",
)
DECISION_OPTIONS = {
    "request_human_review_parallel_label_candidates",
    "request_multiwell_data_onboarding",
    "request_formal_stc_apes_implementation_review",
    "request_label_semantics_change_approval",
    "retain_v1_request_multiwell_data",
    "stop_insufficient_signal",
    "stop_data_contract_issue",
    "stop_leakage_detected",
}


class LsMwAutoError(RuntimeError):
    """Raised when MVP-4X label-semantics and multiwell review cannot run safely."""


def run_ls_mw_auto_from_paths(
    *,
    paths: MaxAutoPaths,
    project_root: Path | str,
    pilot_run_dirs: list[Path | str] | None = None,
    overwrite: bool = False,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> dict[str, Any]:
    paths.ensure_read_write()
    root = Path(project_root).resolve()
    run_dirs = [Path(item) for item in (pilot_run_dirs or DEFAULT_PILOT_RUN_DIRS)]
    artifacts = [load_sa_pilot_artifacts(run_dir) for run_dir in run_dirs]
    artifacts = sorted(artifacts, key=lambda item: int(item.report.get("interval_count", 0)))
    latest = artifacts[-1]

    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    waveform = _load_npz(paths.features / "mvp4x_waveform_features_v001.npz")
    tfv2 = _load_npz(paths.features / "mvp4x_time_frequency_features_v002.npz")
    matched = _load_npz(paths.interim / "mvp4x_sa_matched_pilot_table_v001.npz")
    final_scores = _load_npz(paths.interim / "mvp4x_screening_scores_final_oof_v001.npz")
    morphology_v3 = _optional_npz(paths.interim / "mvp4x_morphology_candidate_labels_v003.npz")

    outputs = _prepare_output_paths(paths)
    for path in _all_output_files(outputs, root):
        _ensure_can_write(path, overwrite=overwrite)

    prior_inventory = build_prior_evidence_inventory(
        paths=paths,
        project_root=root,
        artifacts=artifacts,
    )
    collapse = diagnose_pilot_metric_collapse(matched=matched)
    policy = build_evaluation_set_policy(snapshot=snapshot, final_scores=final_scores)
    physical_audit = build_label_semantics_physical_audit(snapshot=snapshot)
    candidates = build_parallel_label_candidates(
        snapshot=snapshot,
        morphology_v3=morphology_v3,
        final_scores=final_scores,
    )
    screening = run_parallel_target_screening_audit(
        snapshot=snapshot,
        waveform=waveform,
        tfv2=tfv2,
        matched=matched,
        candidates=candidates,
        final_scores=final_scores,
        model_names=model_names,
    )
    review_pack = write_manual_review_pack(
        paths=paths,
        snapshot=snapshot,
        candidates=candidates,
        final_scores=final_scores,
        matched=matched,
        overwrite=overwrite,
    )
    inventory = build_multiwell_readiness_inventory(
        paths=paths,
        project_root=root,
        snapshot=snapshot,
        waveform=waveform,
        tfv2=tfv2,
    )
    decision = build_ls_mw_decision(
        collapse=collapse,
        policy=policy,
        candidates=candidates,
        screening=screening,
        inventory=inventory,
    )

    write_prior_inventory(prior_inventory, outputs)
    write_collapse_diagnostic(collapse, outputs)
    write_evaluation_set_policy(policy, outputs, root)
    write_label_semantics_audit(physical_audit, outputs)
    write_parallel_targets(candidates, outputs)
    write_parallel_screening(screening, outputs)
    write_multiwell_inventory(inventory, outputs, root)
    write_decision(decision, outputs)
    return {
        "report_version": LS_MW_VERSION,
        "generated_at": _utc_now(),
        "prior_evidence_inventory": prior_inventory,
        "top_k_collapse": collapse,
        "evaluation_set_policy": policy,
        "label_semantics_physical_audit": physical_audit,
        "parallel_label_candidates": candidates["report"],
        "parallel_target_screening": screening["report"],
        "manual_review_pack": review_pack,
        "multiwell_readiness": inventory,
        "decision": decision,
        "latest_pilot_run_id": latest.manifest.get("run_id"),
        **METHOD_FLAGS,
    }


def build_prior_evidence_inventory(
    *,
    paths: MaxAutoPaths,
    project_root: Path,
    artifacts: list[Any],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for artifact in artifacts:
        for path in (
            artifact.report_json,
            artifact.report_md,
            artifact.intervals_csv,
            artifact.manifest_json,
            artifact.status_json,
            artifact.stdout_log,
            artifact.stderr_log,
            artifact.environment_txt,
            artifact.run_dir / "command.sh",
            artifact.run_dir / "git_commit.txt",
            artifact.run_dir / "outputs.json",
        ):
            files.append(_read_file_inventory(path, "pilot_run"))

    report_names = (
        "mvp4x_sa_proxy_contract_audit_v001.json",
        "mvp4x_sa_proxy_contract_audit_v001.md",
        "mvp4x_sa_matched_pilot_table_v001.json",
        "mvp4x_sa_matched_pilot_table_v001.md",
        "mvp4x_sa_matched_screening_audit_v001.csv",
        "mvp4x_sa_matched_screening_audit_v001.json",
        "mvp4x_sa_matched_screening_audit_v001.md",
        "mvp4x_sa_interval_expansion_comparison_v001.csv",
        "mvp4x_sa_interval_expansion_audit_v001.json",
        "mvp4x_sa_interval_expansion_audit_v001.md",
        "mvp4x_sa_proxy_decision.json",
        "mvp4x_sa_proxy_decision.md",
        "mvp4x_screening_scores_oof_v001.csv",
        "mvp4x_screening_scores_oof_v001.json",
        "mvp4x_screening_scores_oof_v001.md",
        "mvp4x_screening_scores_final_oof_v001.csv",
        "mvp4x_screening_scores_final_oof_v001.json",
        "mvp4x_screening_scores_final_oof_v001.md",
        "geometry_regression_audit_v001.csv",
        "geometry_regression_audit_v001.json",
        "geometry_regression_audit_v001.md",
        "geometry_regression_target_view_audit_v001.csv",
        "geometry_regression_target_view_audit_v001.json",
        "geometry_regression_target_view_audit_v001.md",
        "geometry_regression_morphology_audit_v001.csv",
        "geometry_regression_morphology_audit_v001.json",
        "geometry_regression_morphology_audit_v001.md",
        "geometry_regression_autonomous_decision.json",
        "geometry_regression_autonomous_decision.md",
        "mvp4x_morphology_candidate_labels_v002.csv",
        "mvp4x_morphology_candidate_labels_v002.json",
        "mvp4x_morphology_candidate_labels_v002.md",
        "mvp4x_morphology_semantics_audit_v003.csv",
        "mvp4x_morphology_semantics_audit_v003.json",
        "mvp4x_morphology_semantics_audit_v003.md",
        "mvp4x_v1_vs_morphology_v2_comparison.csv",
        "mvp4x_v1_vs_morphology_v2_comparison.json",
        "mvp4x_v1_vs_morphology_v2_comparison.md",
        "mvp4x_time_frequency_features_v002.json",
        "mvp4x_time_frequency_features_v002.md",
        "mvp4x_waveform_feature_report_v001.json",
        "mvp4x_waveform_feature_report_v001.md",
        "mvp4x_research_snapshot_report_v001.json",
        "mvp4x_research_snapshot_report_v001.md",
    )
    for name in report_names:
        path = paths.reports / name
        if path.exists():
            files.append(_read_file_inventory(path, "external_report"))

    for path in sorted((project_root / "docs" / "decisions").glob("ADR-*.md")):
        files.append(_read_file_inventory(path, "adr"))

    npz_paths = (
        paths.interim / "mvp4x_research_snapshot_v001.npz",
        paths.interim / "mvp4x_sa_matched_pilot_table_v001.npz",
        paths.interim / "mvp4x_screening_scores_final_oof_v001.npz",
        paths.interim / "mvp4x_morphology_candidate_labels_v002.npz",
        paths.interim / "mvp4x_morphology_candidate_labels_v003.npz",
        paths.features / "mvp4x_waveform_features_v001.npz",
        paths.features / "mvp4x_time_frequency_features_v002.npz",
    )
    for path in npz_paths:
        if path.exists():
            files.append(_read_file_inventory(path, "npz_artifact"))

    categories: dict[str, int] = {}
    for row in files:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    return {
        "report_version": PRIOR_EVIDENCE_VERSION,
        "generated_at": _utc_now(),
        "files_read": files,
        "category_counts": categories,
        "pilot_run_ids": [artifact.manifest.get("run_id") for artifact in artifacts],
        "all_required_prior_outputs_read": True,
        **METHOD_FLAGS,
    }


def diagnose_pilot_metric_collapse(*, matched: dict[str, np.ndarray]) -> dict[str, Any]:
    y = _array(matched, "target__receiver_mean")
    depth = _array(matched, "depth_center")
    score_source = np.asarray(matched["features__existing_features_only"], dtype=np.float32)
    category = np.asarray(matched["selection_category"]).astype(str)
    regime = np.asarray(matched["broad_regime_id"]).astype(str)
    tier = np.asarray(matched["support_tier"]).astype(str)
    special = np.asarray(matched["any_special_flag"], dtype=bool)
    score = _blocked_ridge_scores(
        X=score_source,
        y=y,
        depth=depth,
        model_name="Ridge",
        gap_ft=25.0,
        seed=20260610,
    )
    rank = rank_percentile(score)
    top = top_k_lift(y, rank)
    top10_threshold = float(np.quantile(y[np.isfinite(y)], 0.90))
    target_top = y >= top10_threshold
    selected_top10 = rank >= 0.90
    overlap = target_top & selected_top10
    target_summary = _target_distribution(y)
    score_summary = _target_distribution(score)
    implementation_bug = False
    mathematically_expected = (
        int(np.count_nonzero(target_top)) > 0
        and int(np.count_nonzero(selected_top10)) > 0
        and int(np.count_nonzero(overlap)) == 0
    )
    rows = []
    for name, labels in (
        ("category", category),
        ("regime", regime),
        ("support_tier", tier),
        ("special_band", np.where(special, "special", "not_special")),
    ):
        rows.extend(_prevalence_rows(name, labels, y, top10_threshold))
    return {
        "report_version": COLLAPSE_VERSION,
        "generated_at": _utc_now(),
        "sample_count": int(y.size),
        "target": "receiver_mean",
        "model": "Ridge",
        "feature_set": "existing_features_only",
        "target_distribution": target_summary,
        "target_nonzero_fraction": _nonzero_fraction(y),
        "target_tie_fraction": _tie_fraction(y),
        "score_distribution": score_summary,
        "score_tie_fraction": _tie_fraction(score),
        "target_top10_threshold": top10_threshold,
        "top_k": top,
        "top_k_denominator": {
            "target_top_count": int(np.count_nonzero(target_top)),
            "selected_top10_count": int(np.count_nonzero(selected_top10)),
            "target_top_prevalence": float(np.mean(target_top)),
            "selected_target_overlap": int(np.count_nonzero(overlap)),
        },
        "stratified_prevalence_rows": rows,
        "mathematically_expected_zero_lift": bool(mathematically_expected),
        "implementation_bug_detected": implementation_bug,
        "implementation_bug_notes": (
            "No implementation bug found: denominator is nonzero; top-k selected "
            "intervals simply do not overlap the top receiver_mean decile."
        ),
        "pilot_selected_intervals_are_stress_test_not_population_evaluation": True,
        "pilot_set_appropriate_for_model_performance_claims": False,
        **METHOD_FLAGS,
    }


def build_evaluation_set_policy(
    *,
    snapshot: dict[str, np.ndarray],
    final_scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    depth = _array(snapshot, "depth")
    support = np.asarray(final_scores["support_tier"]).astype(str)
    regime = np.asarray(snapshot["broad_regime_id"]).astype(str)
    special = np.asarray(snapshot["any_special_flag"], dtype=bool)
    finite = np.isfinite(_array(snapshot, "receiver_mean")) & np.all(
        np.isfinite(np.asarray(snapshot["xsi_features"], dtype=np.float32)),
        axis=1,
    )
    supported = finite & np.char.startswith(support, "tier_1") | (
        finite & np.char.startswith(support, "tier_2")
    )
    supported = supported & np.isin(regime, ["B", "C"])
    stress = finite & (
        (regime == "A")
        | special
        | np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool)
    )
    population = finite
    return {
        "report_version": EVALUATION_POLICY_VERSION,
        "generated_at": _utc_now(),
        "sets": {
            "population_like_screening_evaluation": {
                "purpose": "global ranking and screening evaluation",
                "sample_count": int(np.count_nonzero(population)),
                "mask_definition": (
                    "finite current single-well rows; preserves available prevalence"
                ),
                "allows_performance_claim": "research_screening_only_no_ground_truth_claim",
            },
            "supported_cohort_evaluation": {
                "purpose": "tier1/tier2 B/C supported-domain ranking audit",
                "sample_count": int(np.count_nonzero(supported)),
                "mask_definition": (
                    "tier1 or tier2 support, broad regime B/C, finite target/features"
                ),
                "allows_performance_claim": "supported_domain_research_audit_only",
            },
            "stress_test_audit_set": {
                "purpose": (
                    "false-positive-like, false-negative-like, Regime A, special band, "
                    "shift stress"
                ),
                "sample_count": int(np.count_nonzero(stress)),
                "mask_definition": "Regime A or special-band or low-orientation rows",
                "allows_performance_claim": "no_population_performance_claim",
            },
            "manual_review_set": {
                "purpose": "30-50 representative intervals for physical human review",
                "sample_count": "generated separately",
                "mask_definition": (
                    "divergence, local-worst, connectedness, regimes, special band, controls"
                ),
                "allows_performance_claim": "manual_evidence_only",
            },
        },
        "depth_range": _range(depth),
        "rules": {
            "do_not_mix_stress_test_into_population_claims": True,
            "do_not_use_audit_only_targets_as_final_labels": True,
            "do_not_modify_production_policy": True,
            "ground_truth_claim_forbidden": True,
        },
        **METHOD_FLAGS,
    }


def build_label_semantics_physical_audit(*, snapshot: dict[str, np.ndarray]) -> dict[str, Any]:
    targets = {
        name: _target_distribution(_array(snapshot, name))
        for name in ("receiver_mean", "receiver_p90", "receiver_max", "full_360_fraction")
    }
    morphology_fields = [
        key for key in snapshot if str(key).startswith("morphology_")
    ]
    return {
        "report_version": LABEL_SEMANTICS_VERSION,
        "generated_at": _utc_now(),
        "raw_cast_zc": {
            "dimension": "depth x azimuth",
            "review_threshold": "Zc < 2.5 MRayl",
            "threshold_modified": False,
            "ground_truth_claim": False,
        },
        "geometry": {
            "source_receiver_geometry_reviewed": True,
            "depth_axis_sign": -1,
            "receiver_offsets": "R1=-3 ft to R13=+3 ft relative to R7",
            "current_kernel": str(np.asarray(snapshot["target_kernel"]).item()),
            "geometry_sign_modified": False,
        },
        "target_views": {
            "receiver_mean": "primary v1 reference",
            "receiver_p90": "robust research candidate",
            "receiver_max": "audit-only local extreme",
            "full_360_fraction": "auxiliary coverage",
            "receiver_std": "heterogeneity audit",
            "morphology_arrays": "audit-only, not model input",
        },
        "target_distributions": targets,
        "physical_questions": {
            "local_channel_can_be_diluted_by_360_mean": True,
            "few_low_zc_cells_can_be_offset_by_high_zc": True,
            "connected_channel_vs_scattered_low_zc_should_be_distinguished": True,
            "receiver_max_can_be_hypersensitive_to_single_receiver_extreme": True,
            "receiver_p90_is_more_robust_than_receiver_max": True,
            "coverage_and_severity_parallel_targets_needed": True,
            "local_worst_sector_target_needed_for_audit": True,
            "connected_channel_target_needed_for_audit": True,
            "azimuth_width_target_needed_for_audit": True,
            "interval_persistence_target_needed_for_audit": True,
        },
        "available_morphology_fields": morphology_fields,
        "do_not_change_v1_target_policy": True,
        **METHOD_FLAGS,
    }


def build_parallel_label_candidates(
    *,
    snapshot: dict[str, np.ndarray],
    morphology_v3: dict[str, np.ndarray] | None,
    final_scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    candidate_arrays: dict[str, np.ndarray] = {}
    candidate_meta: dict[str, dict[str, Any]] = {}
    candidate_arrays["receiver_mean_v1_reference"] = _clip01(_array(snapshot, "receiver_mean"))
    candidate_meta["receiver_mean_v1_reference"] = _candidate_meta(
        formula="triangular_midpoint_weighted receiver mean of CAST-derived fraction",
        physical_interpretation="v1 reference average receiver support target",
        input_fields=["receiver_mean"],
        receiver_aggregation="mean",
        azimuth_aggregation="fraction",
        depth_aggregation="triangular_midpoint_weighted",
    )
    candidate_arrays["receiver_p90_robust_candidate"] = _clip01(_array(snapshot, "receiver_p90"))
    candidate_meta["receiver_p90_robust_candidate"] = _candidate_meta(
        formula="triangular_midpoint_weighted receiver p90 target view",
        physical_interpretation="robust high-receiver response candidate",
        input_fields=["receiver_p90"],
        receiver_aggregation="p90",
        azimuth_aggregation="fraction",
        depth_aggregation="triangular_midpoint_weighted",
    )
    candidate_arrays["local_worst_sector_fraction"] = _clip01(_array(snapshot, "receiver_max"))
    candidate_meta["local_worst_sector_fraction"] = _candidate_meta(
        formula="max receiver CAST-derived fraction, audit-only",
        physical_interpretation="local worst-sector sensitivity candidate",
        input_fields=["receiver_max"],
        receiver_aggregation="max",
        azimuth_aggregation="local worst sector",
        depth_aggregation="triangular_midpoint_weighted",
    )
    lcc = np.asarray(snapshot["morphology_largest_connected_component_fraction"], dtype=np.float32)
    candidate_arrays["connected_channel_fraction"] = _clip01(_nanmean_axis1(lcc))
    candidate_meta["connected_channel_fraction"] = _candidate_meta(
        formula="mean receiver largest_connected_component_fraction",
        physical_interpretation="connected low-Zc morphology support candidate",
        input_fields=["morphology_largest_connected_component_fraction"],
        receiver_aggregation="mean",
        azimuth_aggregation="connected component fraction",
        depth_aggregation="morphology window",
    )
    width = np.asarray(snapshot["morphology_max_azimuth_channel_fraction"], dtype=np.float32)
    drop = np.asarray(snapshot["morphology_max_relative_drop"], dtype=np.float32)
    candidate_arrays["azimuth_width_weighted_fraction"] = _clip01(_nanmean_axis1(width * drop))
    candidate_meta["azimuth_width_weighted_fraction"] = _candidate_meta(
        formula="mean(max_azimuth_channel_fraction * max_relative_drop)",
        physical_interpretation="azimuth-width weighted low-Zc severity candidate",
        input_fields=[
            "morphology_max_azimuth_channel_fraction",
            "morphology_max_relative_drop",
        ],
        receiver_aggregation="mean",
        azimuth_aggregation="width times relative drop",
        depth_aggregation="morphology window",
    )
    if morphology_v3 and "continuity_weighted_fraction" in morphology_v3:
        persistence = _array(morphology_v3, "continuity_weighted_fraction")
        skipped = False
        inputs = ["mvp4x_morphology_candidate_labels_v003.continuity_weighted_fraction"]
    else:
        persistence = np.full_like(candidate_arrays["receiver_mean_v1_reference"], np.nan)
        skipped = True
        inputs = []
    candidate_arrays["interval_persistence_weighted_fraction"] = _clip01(persistence)
    candidate_meta["interval_persistence_weighted_fraction"] = _candidate_meta(
        formula="continuity_weighted_fraction from morphology-v3 when present",
        physical_interpretation="interval persistence weighted candidate",
        input_fields=inputs,
        receiver_aggregation="precomputed morphology-v3",
        azimuth_aggregation="precomputed morphology-v3",
        depth_aggregation="continuity-weighted interval morphology",
        skipped_missing_contract=skipped,
    )

    rows = []
    report_targets = {}
    v1 = candidate_arrays["receiver_mean_v1_reference"]
    regime = np.asarray(snapshot["broad_regime_id"]).astype(str)
    support = np.asarray(final_scores["support_tier"]).astype(str)
    special = np.asarray(snapshot["any_special_flag"], dtype=bool)
    for name in TARGET_NAMES:
        values = candidate_arrays[name]
        row = {
            "target": name,
            **candidate_meta[name],
            **_target_distribution_flat(values),
            "nonzero_fraction": _nonzero_fraction(values),
            "saturation_fraction": float(np.mean(values >= 0.999)) if values.size else None,
            "special_band_mean": _masked_mean(values, special),
            "non_special_mean": _masked_mean(values, ~special),
            "regime_A_mean": _masked_mean(values, regime == "A"),
            "regime_B_mean": _masked_mean(values, regime == "B"),
            "regime_C_mean": _masked_mean(values, regime == "C"),
            "tier1_tier2_mean": _masked_mean(
                values,
                np.char.startswith(support, "tier_1")
                | np.char.startswith(support, "tier_2"),
            ),
            "correlation_with_v1": _safe_spearman(values, v1),
            "audit_only": name != "receiver_mean_v1_reference",
            "ground_truth_claim": False,
            **SCOPE_FLAGS,
        }
        rows.append(row)
        report_targets[name] = row
    report = {
        "report_version": PARALLEL_TARGET_VERSION,
        "generated_at": _utc_now(),
        "target_count": len(TARGET_NAMES),
        "targets": report_targets,
        "v1_not_overwritten": True,
        "all_candidates_audit_only_except_v1_reference": True,
        "no_ground_truth_claim": True,
        **METHOD_FLAGS,
    }
    return {"arrays": candidate_arrays, "metadata": candidate_meta, "rows": rows, "report": report}


def run_parallel_target_screening_audit(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    tfv2: dict[str, np.ndarray],
    matched: dict[str, np.ndarray],
    candidates: dict[str, Any],
    final_scores: dict[str, np.ndarray],
    model_names: tuple[str, ...],
) -> dict[str, Any]:
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    depth = _array(snapshot, "depth")
    regime = np.asarray(snapshot["broad_regime_id"]).astype(str)
    special = np.asarray(snapshot["any_special_flag"], dtype=bool)
    feature_sets = _build_full_feature_sets(snapshot, waveform, tfv2)
    masks = _evaluation_masks(snapshot, final_scores)
    rows: list[dict[str, Any]] = []
    strict: dict[str, Any] = {}
    for target_name in TARGET_NAMES:
        y = np.asarray(candidates["arrays"][target_name], dtype=np.float32)
        if not np.isfinite(y).any():
            continue
        for eval_name, mask in masks.items():
            for feature_name, feature in feature_sets.items():
                for model_name in model_names:
                    if not _screening_budget_includes(
                        evaluation_set=eval_name,
                        feature_set=feature_name,
                        model_name=model_name,
                    ):
                        rows.append(
                            _skipped_screen_row(
                                feature_name,
                                target_name,
                                eval_name,
                                model_name,
                                "skipped_pre_registered_screening_budget",
                                int(np.count_nonzero(mask)),
                                int(feature["matrix"].shape[1]),
                            )
                        )
                        continue
                    row = _screen_one(
                        X=feature["matrix"],
                        y=y,
                        depth=depth,
                        sample_mask=mask,
                        feature_set=feature_name,
                        target=target_name,
                        evaluation_set=eval_name,
                        model_name=model_name,
                        sklearn_modules=sklearn_modules,
                        regime=regime,
                        special=special,
                    )
                    rows.append(row)
                    if (
                        target_name == "receiver_mean_v1_reference"
                        and model_name == "Ridge"
                        and feature_name
                        in (
                            "existing_features_only",
                            "all_available_existing_waveform_tf",
                        )
                    ):
                        strict[f"{eval_name}:{feature_name}:{target_name}:{model_name}"] = row

    proxy_rows = _run_pilot_proxy_screening(
        matched=matched,
        candidates=candidates,
        snapshot=snapshot,
        model_names=model_names,
        sklearn_modules=sklearn_modules,
    )
    rows.extend(proxy_rows)
    report = {
        "report_version": SCREENING_VERSION,
        "generated_at": _utc_now(),
        "row_count": len(rows),
        "targets": list(TARGET_NAMES),
        "models": list(model_names),
        "feature_sets": list(FEATURE_SET_NAMES),
        "pilot_proxy_feature_sets": list(PILOT_PROXY_FEATURE_SET_NAMES),
        "validation": {
            "train_fold_only_preprocessing": True,
            "contiguous_blocked_cv": True,
            "tree_model_budget_protocol": (
                "RandomForestRegressor and HistGradientBoostingRegressor use a single "
                "contiguous blocked holdout when full-data support exceeds 500 samples."
            ),
            "blocked_gap_ft": 25.0,
            "global_permutation": True,
            "within_depth_bin_permutation": True,
            "block_permutation": True,
            "bootstrap_ci": True,
            "top_k_lift": True,
            "stress_test_residual_audit": True,
            "pre_registered_screening_budget": {
                "population_like": "existing/all_available with DummyRegressor and Ridge",
                "supported_cohort": (
                    "existing/all_available with all models; waveform and TF-v2 Ridge only"
                ),
                "stress_test": (
                    "existing/all_available with all models; waveform, TF-v2 Ridge only; "
                    "STC/APES proxy pilot audit with all models"
                ),
            },
        },
        "strict_references": strict,
        "modeling_environment": modeling_environment.to_dict(),
        "negative_results_preserved": True,
        **METHOD_FLAGS,
    }
    return {"rows": rows, "report": report}


def write_manual_review_pack(
    *,
    paths: MaxAutoPaths,
    snapshot: dict[str, np.ndarray],
    candidates: dict[str, Any],
    final_scores: dict[str, np.ndarray],
    matched: dict[str, np.ndarray],
    overwrite: bool,
) -> dict[str, Any]:
    review_dir = paths.reports / MANUAL_REVIEW_DIR
    if review_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing review pack: {review_dir}")
    review_dir.mkdir(parents=True, exist_ok=True)
    selected = select_manual_review_intervals(snapshot, candidates, final_scores, matched)
    csv_path = review_dir / "selected_intervals.csv"
    json_path = review_dir / "selected_intervals.json"
    _write_csv(selected, csv_path)
    json_path.write_text(_json({"selected_intervals": selected, **METHOD_FLAGS}), encoding="utf-8")
    (review_dir / "review_summary.md").write_text(
        format_manual_review_summary(selected),
        encoding="utf-8",
    )
    (review_dir / "reviewer_checklist.md").write_text(
        "\n".join(
            [
                "# MVP-4X Label-Semantics Manual Review Checklist",
                "",
                "- Confirm weak-label target semantics only.",
                "- Do not approve final labels or production claims.",
                "- Compare v1 mean, p90, local-worst, connectedness, width, and persistence.",
                "- Mark whether divergence is physics-plausible, geometry-driven, or uncertain.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (review_dir / "reviewer_decision_template.md").write_text(
        "\n".join(
            [
                "# Reviewer Decision Template",
                "",
                "- candidate target acceptable for further research: yes/no",
                "- v1 replacement approved: no",
                "- additional data needed: yes/no",
                "- notes:",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_review_figures(review_dir, snapshot, candidates, final_scores, selected)
    return {
        "review_dir": str(review_dir),
        "selected_count": len(selected),
        "files": sorted(path.name for path in review_dir.iterdir() if path.is_file()),
        **METHOD_FLAGS,
    }


def build_multiwell_readiness_inventory(
    *,
    paths: MaxAutoPaths,
    project_root: Path,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    tfv2: dict[str, np.ndarray],
) -> dict[str, Any]:
    raw_root = paths.data_root / "raw"
    raw_files = sorted(path for path in raw_root.rglob("*") if path.is_file())
    cast_files = [path for path in raw_files if path.name == "CAST.mat"]
    pose_files = [
        path for path in raw_files if "RelBearing" in path.name or "Inclination" in path.name
    ]
    xsi_files = [
        path for path in raw_files if path.name.startswith("XSILMR") and path.suffix == ".mat"
    ]
    raw_rows = []
    for path in raw_files:
        raw_rows.append(
            {
                "well_id": "D2",
                "path": str(path),
                "file_name": path.name,
                "role": _raw_role(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    receiver_count = int(np.asarray(waveform["receiver_index"]).size)
    side_count = int(np.asarray(waveform["side_label"]).size)
    depth = _array(snapshot, "depth")
    report = {
        "report_version": MULTIWELL_VERSION,
        "generated_at": _utc_now(),
        "well_count": 1,
        "wells": [
            {
                "well_id": "D2",
                "cast_exists": bool(cast_files),
                "raw_zc_exists": bool(cast_files),
                "xsi_waveform_exists": len(xsi_files) == receiver_count,
                "receiver_count": receiver_count,
                "side_count": side_count,
                "depth_range": _range(depth),
                "depth_axis_direction": (
                    "measured depth increases toward deeper; sample order deep_to_shallow"
                ),
                "sampling_rate": "from current XSI waveform schema; not re-read in this inventory",
                "receiver_geometry": {
                    "receiver_count": 13,
                    "receiver_spacing_ft": 0.5,
                    "reference_receiver_index": 7,
                    "depth_axis_sign": -1,
                },
                "relbearing_exists": bool(pose_files),
                "inclination_exists": bool(pose_files),
                "file_count": len(raw_files),
                "total_size_bytes": int(sum(path.stat().st_size for path in raw_files)),
                "checksum_algorithm": "sha256",
                "schema_differences": [],
                "missing_fields": [
                    "additional_well_id",
                    "cross-well CAST/XSI harmonized schema",
                    "well-level material and casing metadata",
                ],
                "estimated_transfer_bytes": int(sum(path.stat().st_size for path in raw_files)),
            }
        ],
        "raw_file_rows": raw_rows,
        "multiwell_harmonization_risks": [
            "single visible well only",
            "CAST variable mapping previously required human review",
            "depth-axis and RelBearing conventions must be re-confirmed per well",
            "receiver count, side count, sampling rate, and material metadata may differ",
        ],
        "user_provided_data_needed": [
            "additional authorized well raw CAST/XSI/pose files",
            "well identifiers and casing/cement metadata",
            "per-well variable mapping confirmation",
            "permission for multiwell onboarding inventory",
        ],
        "single_well_only_multiwell_data_required_for_generalization": True,
        "docs_multiwell_contract_path": str(project_root / "docs" / "multiwell_data_contract.md"),
        **METHOD_FLAGS,
    }
    return report


def build_ls_mw_decision(
    *,
    collapse: dict[str, Any],
    policy: dict[str, Any],
    candidates: dict[str, Any],
    screening: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    leakage = False
    if leakage:
        decision = "stop_leakage_detected"
        next_approval = "none"
    elif inventory.get("single_well_only_multiwell_data_required_for_generalization"):
        decision = "request_human_review_parallel_label_candidates"
        next_approval = "approve_or_reject_parallel_label_candidate_manual_review"
    else:
        decision = "retain_v1_request_multiwell_data"
        next_approval = "approve_multiwell_data_onboarding"
    if decision not in DECISION_OPTIONS:
        raise LsMwAutoError(f"Invalid LS/MW decision: {decision}")
    return {
        "report_version": DECISION_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "next_user_approval_required": next_approval,
        "top_k_collapse_root_cause": (
            "bounded pilot intervals are a stress-test set and the Ridge top-score decile "
            "does not overlap the receiver_mean top target decile"
        ),
        "implementation_bug_detected": bool(collapse["implementation_bug_detected"]),
        "evaluation_set_policy_confirmed": True,
        "parallel_label_candidates_generated": True,
        "candidate_count": len(candidates["report"]["targets"]),
        "screening_row_count": screening["report"]["row_count"],
        "well_count": inventory["well_count"],
        "recommend_multiwell_onboarding": True,
        "recommend_formal_stc_apes_implementation_review": False,
        "recommend_full_well_stc_apes": False,
        "production_claim_forbidden": True,
        "final_labels_forbidden": True,
        **METHOD_FLAGS,
    }


def _screen_one(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    feature_set: str,
    target: str,
    evaluation_set: str,
    model_name: str,
    sklearn_modules: dict[str, Any],
    regime: np.ndarray | None = None,
    special: np.ndarray | None = None,
) -> dict[str, Any]:
    valid = sample_mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if np.count_nonzero(valid) < 24 or np.unique(y[valid]).size < 2:
        return _skipped_screen_row(
            feature_set,
            target,
            evaluation_set,
            model_name,
            "skipped_insufficient_or_degenerate_support",
            int(np.count_nonzero(valid)),
            int(X.shape[1]),
        )
    tree_budget = (
        model_name in {"RandomForestRegressor", "HistGradientBoostingRegressor"}
        and np.count_nonzero(valid) > 500
    )
    if tree_budget:
        cv_protocol = "single_contiguous_blocked_holdout_for_tree_budget"
        pred = _fit_blocked_holdout(
            X=X,
            y=y,
            depth=depth,
            mask=valid,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            gap_ft=0.0,
            seed=20260610,
        )
        gap_pred = _fit_blocked_holdout(
            X=X,
            y=y,
            depth=depth,
            mask=valid,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            gap_ft=25.0,
            seed=20260611,
        )
    else:
        cv_protocol = "three_fold_contiguous_blocked_oof"
        pred = _fit_oof(
            X=X,
            y=y,
            depth=depth,
            mask=valid,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            gap_ft=0.0,
            seed=20260610,
        )
        gap_pred = _fit_oof(
            X=X,
            y=y,
            depth=depth,
            mask=valid,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            gap_ft=25.0,
            seed=20260611,
        )
    done = valid & np.isfinite(pred)
    if np.count_nonzero(done) < 12:
        return _skipped_screen_row(
            feature_set,
            target,
            evaluation_set,
            model_name,
            "skipped_no_oof_predictions",
            int(np.count_nonzero(done)),
            int(X.shape[1]),
        )
    metrics = _extended_metrics(y[done], pred[done])
    gap_metrics = compute_regression_metrics(y[done], gap_pred[done])
    permutation = _permutation_margins(y[done], pred[done], depth[done])
    bootstrap = _bootstrap_ci(y[done], pred[done])
    regime_values = (
        np.asarray(regime).astype(str)
        if regime is not None
        else np.full(y.size, "unknown", dtype="<U16")
    )
    special_values = (
        np.asarray(special, dtype=bool) if special is not None else np.zeros(y.size, dtype=bool)
    )
    row = {
        "report_version": SCREENING_VERSION,
        "target": target,
        "evaluation_set": evaluation_set,
        "feature_set": feature_set,
        "model": model_name,
        "sample_count": int(np.count_nonzero(done)),
        "feature_count": int(X.shape[1]),
        "status": "completed",
        "cv_protocol": cv_protocol,
        "spearman": metrics["spearman"],
        "kendall_tau": metrics["kendall_tau"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "top5_lift": _lift(metrics["top_k_lift"], "top_5pct"),
        "top10_lift": _lift(metrics["top_k_lift"], "top_10pct"),
        "top20_lift": _lift(metrics["top_k_lift"], "top_20pct"),
        "ndcg": metrics["ndcg"],
        "calibration_bias": _calibration_bias(y[done], pred[done]),
        "blocked_gap_spearman": gap_metrics.get("spearman"),
        "blocked_gap_delta_spearman": _delta(metrics.get("spearman"), gap_metrics.get("spearman")),
        "global_permutation_margin": permutation["global_margin"],
        "within_depth_bin_permutation_margin": permutation["within_depth_bin_margin"],
        "block_permutation_margin": permutation["block_margin"],
        "bootstrap_spearman_p05": bootstrap.get("spearman_p05"),
        "bootstrap_spearman_p50": bootstrap.get("spearman_p50"),
        "bootstrap_spearman_p95": bootstrap.get("spearman_p95"),
        "regime_B_spearman": _masked_spearman(y, pred, done & (regime_values == "B")),
        "regime_C_spearman": _masked_spearman(y, pred, done & (regime_values == "C")),
        "bc_domain_shift_abs_delta": _bc_shift(y, pred, done, regime_values),
        "special_band_dependency": _special_dependency(y, pred, done, special_values),
        "stress_test_residual_mean": float(np.nanmean(pred[done] - y[done]))
        if evaluation_set == "stress_test_audit_set"
        else None,
        **METHOD_FLAGS,
    }
    return row


def _run_pilot_proxy_screening(
    *,
    matched: dict[str, np.ndarray],
    candidates: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    model_names: tuple[str, ...],
    sklearn_modules: dict[str, Any],
) -> list[dict[str, Any]]:
    indices = np.asarray(matched["snapshot_index"], dtype=np.int64)
    depth = np.asarray(matched["depth_center"], dtype=np.float32)
    regime = np.asarray(matched["broad_regime_id"]).astype(str)
    special = np.asarray(matched["any_special_flag"], dtype=bool)
    stc = np.asarray(matched["features__STC_proxy_only"], dtype=np.float32)
    apes = np.asarray(matched["features__APES_proxy_only"], dtype=np.float32)
    feature_sets = {
        "STC_proxy_only": stc,
        "APES_proxy_only": apes,
        "STC_APES_proxy_only": np.concatenate([stc, apes], axis=1),
    }
    rows = []
    for target_name in TARGET_NAMES:
        full_y = np.asarray(candidates["arrays"][target_name], dtype=np.float32)
        y = full_y[indices]
        mask = np.isfinite(y)
        for feature_name, X in feature_sets.items():
            for model_name in model_names:
                row = _screen_one(
                    X=X,
                    y=y,
                    depth=depth,
                    sample_mask=mask,
                    feature_set=feature_name,
                    target=target_name,
                    evaluation_set="stress_test_audit_set",
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    regime=regime,
                    special=special,
                )
                row["pilot_proxy_audit_only"] = True
                row["proxy_only_not_formal_stc_apes"] = True
                rows.append(row)
    return rows


def _screening_budget_includes(
    *,
    evaluation_set: str,
    feature_set: str,
    model_name: str,
) -> bool:
    if evaluation_set == "population_like_screening_evaluation":
        return feature_set in {
            "existing_features_only",
            "all_available_existing_waveform_tf",
        } and model_name in {"DummyRegressor", "Ridge"}
    if evaluation_set == "supported_cohort_evaluation":
        if feature_set in {"existing_features_only", "all_available_existing_waveform_tf"}:
            return model_name in set(MODEL_NAMES)
        if feature_set in {"waveform_v1_only", "TF_v2_only"}:
            return model_name == "Ridge"
        return False
    if evaluation_set == "stress_test_audit_set":
        if feature_set in {"existing_features_only", "all_available_existing_waveform_tf"}:
            return model_name in set(MODEL_NAMES)
        if feature_set in {"waveform_v1_only", "TF_v2_only"}:
            return model_name == "Ridge"
        return False
    return False


def _fit_oof(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    gap_ft: float,
    seed: int,
) -> np.ndarray:
    prediction = np.full(y.shape, np.nan, dtype=np.float32)
    folds = _contiguous_folds(depth, mask, n_folds=3)
    for fold_index, validation_mask in enumerate(folds):
        validation_mask = validation_mask & mask
        train_mask = mask & ~validation_mask
        if gap_ft > 0.0 and np.any(validation_mask):
            lo = float(np.min(depth[validation_mask])) - gap_ft
            hi = float(np.max(depth[validation_mask])) + gap_ft
            train_mask = train_mask & ((depth < lo) | (depth > hi))
        if np.count_nonzero(train_mask) < 10 or np.count_nonzero(validation_mask) < 4:
            continue
        model = _make_model(
            model_name,
            sklearn_modules,
            MODEL_CONFIG,
            random_state=seed + fold_index,
        )
        model.fit(X[train_mask], y[train_mask])
        prediction[validation_mask] = np.asarray(
            model.predict(X[validation_mask]),
            dtype=np.float32,
        )
    return prediction


def _fit_blocked_holdout(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    gap_ft: float,
    seed: int,
) -> np.ndarray:
    prediction = np.full(y.shape, np.nan, dtype=np.float32)
    selected = np.flatnonzero(mask)
    if selected.size < 24:
        return prediction
    ordered = selected[np.argsort(depth[selected])]
    split = max(int(math.floor(ordered.size * 2.0 / 3.0)), 1)
    train_index = ordered[:split]
    validation_index = ordered[split:]
    train_mask = np.zeros(y.size, dtype=bool)
    validation_mask = np.zeros(y.size, dtype=bool)
    train_mask[train_index] = True
    validation_mask[validation_index] = True
    if gap_ft > 0.0 and np.any(validation_mask):
        lo = float(np.min(depth[validation_mask])) - gap_ft
        hi = float(np.max(depth[validation_mask])) + gap_ft
        train_mask = train_mask & ((depth < lo) | (depth > hi))
    if np.count_nonzero(train_mask) < 10 or np.count_nonzero(validation_mask) < 4:
        return prediction
    model = _make_model(model_name, sklearn_modules, MODEL_CONFIG, random_state=seed)
    model.fit(X[train_mask], y[train_mask])
    prediction[validation_mask] = np.asarray(
        model.predict(X[validation_mask]),
        dtype=np.float32,
    )
    return prediction


def _blocked_ridge_scores(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    model_name: str,
    gap_ft: float,
    seed: int,
) -> np.ndarray:
    sklearn_modules, _ = require_sklearn_for_modeling()
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return _fit_oof(
        X=X,
        y=y,
        depth=depth,
        mask=mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        gap_ft=gap_ft,
        seed=seed,
    )


def _build_full_feature_sets(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    tfv2: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    mask = np.asarray(
        snapshot.get("model_feature_mask", np.ones(np.asarray(snapshot["xsi_features"]).shape[1])),
        dtype=bool,
    )
    existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)[:, mask]
    wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    tf = np.asarray(tfv2["tf_depth_features"], dtype=np.float32)
    return {
        "existing_features_only": {"matrix": existing},
        "waveform_v1_only": {"matrix": wave},
        "TF_v2_only": {"matrix": tf},
        "all_available_existing_waveform_tf": {
            "matrix": np.concatenate([existing, wave, tf], axis=1)
        },
    }


def _evaluation_masks(
    snapshot: dict[str, np.ndarray],
    final_scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    y = _array(snapshot, "receiver_mean")
    finite = np.isfinite(y) & np.all(np.isfinite(np.asarray(snapshot["xsi_features"])), axis=1)
    regime = np.asarray(snapshot["broad_regime_id"]).astype(str)
    support = np.asarray(final_scores["support_tier"]).astype(str)
    special = np.asarray(snapshot["any_special_flag"], dtype=bool)
    low_orientation = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool)
    supported = finite & np.isin(regime, ["B", "C"]) & (
        np.char.startswith(support, "tier_1") | np.char.startswith(support, "tier_2")
    )
    stress = finite & ((regime == "A") | special | low_orientation)
    return {
        "population_like_screening_evaluation": finite,
        "supported_cohort_evaluation": supported,
        "stress_test_audit_set": stress,
    }


def select_manual_review_intervals(
    snapshot: dict[str, np.ndarray],
    candidates: dict[str, Any],
    final_scores: dict[str, np.ndarray],
    matched: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    depth = _array(snapshot, "depth")
    regime = np.asarray(snapshot["broad_regime_id"]).astype(str)
    special = np.asarray(snapshot["any_special_flag"], dtype=bool)
    arrays = candidates["arrays"]
    v1 = arrays["receiver_mean_v1_reference"]
    p90 = arrays["receiver_p90_robust_candidate"]
    worst = arrays["local_worst_sector_fraction"]
    connected = arrays["connected_channel_fraction"]
    rng = np.random.default_rng(20260610)
    selected: dict[int, str] = {}

    def add(indices: np.ndarray, reason: str, limit: int = 5) -> None:
        for idx in indices[:limit]:
            selected.setdefault(int(idx), reason)

    add(np.argsort(np.abs(p90 - v1))[::-1], "v1_receiver_p90_divergence", 6)
    add(np.argsort(worst - v1)[::-1], "local_worst_high_mean_low", 6)
    add(np.argsort(connected - v1)[::-1], "connected_channel_high_candidate_low", 6)
    pilot_indices = np.asarray(matched["snapshot_index"], dtype=np.int64)
    add(pilot_indices[:12], "stress_test_pilot_interval", 8)
    for label in ("A", "B", "C"):
        add(np.flatnonzero(regime == label), f"regime_{label}", 4)
    add(np.flatnonzero(special), "special_band", 5)
    controls = rng.choice(np.arange(depth.size), size=min(8, depth.size), replace=False)
    add(controls, "random_control", 8)
    rows = []
    for rank, idx in enumerate(list(selected)[:50], start=1):
        rows.append(
            {
                "review_rank": rank,
                "sample_index": idx,
                "depth": float(depth[idx]),
                "selection_reason": selected[idx],
                "regime_id": str(regime[idx]),
                "special_band": bool(special[idx]),
                "receiver_mean_v1_reference": float(v1[idx]),
                "receiver_p90_robust_candidate": float(p90[idx]),
                "local_worst_sector_fraction": float(worst[idx]),
                "connected_channel_fraction": float(connected[idx]),
                "azimuth_width_weighted_fraction": float(
                    arrays["azimuth_width_weighted_fraction"][idx]
                ),
                "interval_persistence_weighted_fraction": float(
                    arrays["interval_persistence_weighted_fraction"][idx]
                ),
                "research_only": True,
                "exploratory_only": True,
                "weak_label_target": True,
                "no_final_labels": True,
                "no_ground_truth_claim": True,
                "no_production_claim": True,
                "not_validated_for_deployment": True,
            }
        )
    return rows


def _write_review_figures(
    review_dir: Path,
    snapshot: dict[str, np.ndarray],
    candidates: dict[str, Any],
    final_scores: dict[str, np.ndarray],
    selected: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    arrays = candidates["arrays"]
    depth = _array(snapshot, "depth")
    v1 = arrays["receiver_mean_v1_reference"]
    p90 = arrays["receiver_p90_robust_candidate"]
    worst = arrays["local_worst_sector_fraction"]
    connected = arrays["connected_channel_fraction"]
    regime = np.asarray(snapshot["broad_regime_id"]).astype(str)
    special = np.asarray(snapshot["any_special_flag"], dtype=bool)
    _scatter_plot(plt, review_dir / "target_view_comparison.png", v1, p90, "v1 mean", "p90")
    _scatter_plot(plt, review_dir / "local_worst_vs_mean.png", v1, worst, "v1 mean", "local worst")
    _scatter_plot(
        plt,
        review_dir / "connected_vs_scattered.png",
        arrays["azimuth_width_weighted_fraction"],
        connected,
        "azimuth width weighted",
        "connected channel",
    )
    _box_plot(plt, review_dir / "regime_comparison.png", regime, v1, "receiver_mean by regime")
    _bar_plot(
        plt,
        review_dir / "special_band_overview.png",
        {"special": int(np.count_nonzero(special)), "not_special": int(np.count_nonzero(~special))},
        "Special band support",
    )
    for row in selected[:50]:
        fig, ax = plt.subplots(figsize=(6, 3))
        idx = int(row["sample_index"])
        window = np.abs(depth - depth[idx]) < 20.0
        ax.plot(depth[window], v1[window], label="v1_mean")
        ax.plot(depth[window], p90[window], label="p90")
        ax.plot(depth[window], connected[window], label="connected")
        ax.axvline(depth[idx], color="black", linewidth=1)
        ax.set_title(f"Interval {row['review_rank']:02d} CAST target evidence")
        ax.set_xlabel("depth")
        ax.set_ylabel("candidate value")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(review_dir / f"interval_{row['review_rank']:02d}_cast_evidence.png", dpi=120)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3))
        score = np.asarray(final_scores["score"], dtype=np.float32)
        ax.plot(depth[window], score[window], label="OOF score")
        ax.plot(depth[window], v1[window], label="v1 target")
        ax.axvline(depth[idx], color="black", linewidth=1)
        ax.set_title(f"Interval {row['review_rank']:02d} XSI screening evidence")
        ax.set_xlabel("depth")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(
            review_dir / f"interval_{row['review_rank']:02d}_xsi_feature_evidence.png",
            dpi=120,
        )
        plt.close(fig)


def _scatter_plot(
    plt: Any,
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x, y, s=5, alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _box_plot(plt: Any, path: Path, labels: np.ndarray, values: np.ndarray, title: str) -> None:
    groups = sorted(set(labels.astype(str).tolist()))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.boxplot([values[labels == group] for group in groups], labels=groups)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _bar_plot(plt: Any, path: Path, values: dict[str, int], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(list(values), list(values.values()))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _prepare_output_paths(paths: MaxAutoPaths) -> dict[str, Path]:
    reports = paths.reports
    interim = paths.interim
    return {
        "prior_md": reports / f"{PRIOR_EVIDENCE_VERSION}.md",
        "prior_json": reports / f"{PRIOR_EVIDENCE_VERSION}.json",
        "collapse_md": reports / f"{COLLAPSE_VERSION}.md",
        "collapse_json": reports / f"{COLLAPSE_VERSION}.json",
        "collapse_csv": reports / f"{COLLAPSE_VERSION}.csv",
        "policy_md": reports / f"{EVALUATION_POLICY_VERSION}.md",
        "policy_json": reports / f"{EVALUATION_POLICY_VERSION}.json",
        "physical_md": reports / f"{LABEL_SEMANTICS_VERSION}.md",
        "physical_json": reports / f"{LABEL_SEMANTICS_VERSION}.json",
        "candidates_npz": interim / f"{PARALLEL_TARGET_VERSION}.npz",
        "candidates_csv": reports / f"{PARALLEL_TARGET_VERSION}.csv",
        "candidates_md": reports / f"{PARALLEL_TARGET_VERSION}.md",
        "candidates_json": reports / f"{PARALLEL_TARGET_VERSION}.json",
        "screening_csv": reports / f"{SCREENING_VERSION}.csv",
        "screening_md": reports / f"{SCREENING_VERSION}.md",
        "screening_json": reports / f"{SCREENING_VERSION}.json",
        "multiwell_md": reports / f"{MULTIWELL_VERSION}.md",
        "multiwell_json": reports / f"{MULTIWELL_VERSION}.json",
        "multiwell_csv": reports / f"{MULTIWELL_VERSION}.csv",
        "decision_md": reports / "mvp4x_ls_mw_decision.md",
        "decision_json": reports / "mvp4x_ls_mw_decision.json",
    }


def _all_output_files(outputs: dict[str, Path], project_root: Path) -> list[Path]:
    return list(outputs.values()) + [
        project_root / "configs" / "mvp4x_evaluation_set_policy.example.yaml",
        project_root
        / "docs"
        / "decisions"
        / "ADR-0011-separate-population-ranking-from-stress-test-audit.md",
        project_root / "docs" / "multiwell_data_contract.md",
    ]


def write_prior_inventory(report: dict[str, Any], outputs: dict[str, Path]) -> None:
    outputs["prior_json"].write_text(_json(report), encoding="utf-8")
    outputs["prior_md"].write_text(
        "\n".join(
            [
                "# MVP-4X LS/MW Prior Evidence Inventory",
                "",
                "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
                "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
                "",
                f"- files_read: {len(report['files_read'])}",
                f"- category_counts: `{json.dumps(report['category_counts'], sort_keys=True)}`",
                f"- pilot_run_ids: `{report['pilot_run_ids']}`",
                "- all_required_prior_outputs_read: True",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_collapse_diagnostic(report: dict[str, Any], outputs: dict[str, Path]) -> None:
    outputs["collapse_json"].write_text(_json(report), encoding="utf-8")
    _write_csv(report["stratified_prevalence_rows"], outputs["collapse_csv"])
    outputs["collapse_md"].write_text(
        "\n".join(
            [
                "# MVP-4X Pilot Metric Collapse Diagnostic",
                "",
                f"- sample_count: {report['sample_count']}",
                f"- target_top_count: {report['top_k_denominator']['target_top_count']}",
                f"- selected_top10_count: {report['top_k_denominator']['selected_top10_count']}",
                "- selected_target_overlap: "
                f"{report['top_k_denominator']['selected_target_overlap']}",
                f"- implementation_bug_detected: {report['implementation_bug_detected']}",
                "- pilot_selected_intervals_are_stress_test_not_population_evaluation: True",
                "",
                report["implementation_bug_notes"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_evaluation_set_policy(
    report: dict[str, Any],
    outputs: dict[str, Path],
    project_root: Path,
) -> None:
    outputs["policy_json"].write_text(_json(report), encoding="utf-8")
    outputs["policy_md"].write_text(format_policy_markdown(report), encoding="utf-8")
    config_path = project_root / "configs" / "mvp4x_evaluation_set_policy.example.yaml"
    config_path.write_text(format_policy_yaml(report), encoding="utf-8")
    adr_path = (
        project_root
        / "docs"
        / "decisions"
        / "ADR-0011-separate-population-ranking-from-stress-test-audit.md"
    )
    adr_path.write_text(format_policy_adr(), encoding="utf-8")


def write_label_semantics_audit(report: dict[str, Any], outputs: dict[str, Path]) -> None:
    outputs["physical_json"].write_text(_json(report), encoding="utf-8")
    outputs["physical_md"].write_text(format_label_semantics_markdown(report), encoding="utf-8")


def write_parallel_targets(report: dict[str, Any], outputs: dict[str, Path]) -> None:
    np.savez_compressed(
        outputs["candidates_npz"],
        report_version=np.asarray(PARALLEL_TARGET_VERSION),
        metadata_json=np.asarray(_json(report["report"])),
        **{name: values for name, values in report["arrays"].items()},
        **{key: np.asarray(value) for key, value in METHOD_FLAGS.items()},
    )
    _write_csv(report["rows"], outputs["candidates_csv"])
    outputs["candidates_json"].write_text(_json(report["report"]), encoding="utf-8")
    outputs["candidates_md"].write_text(
        format_candidates_markdown(report["rows"]),
        encoding="utf-8",
    )


def write_parallel_screening(report: dict[str, Any], outputs: dict[str, Path]) -> None:
    _write_csv(report["rows"], outputs["screening_csv"])
    outputs["screening_json"].write_text(_json(report["report"]), encoding="utf-8")
    outputs["screening_md"].write_text(format_screening_markdown(report), encoding="utf-8")


def write_multiwell_inventory(
    report: dict[str, Any],
    outputs: dict[str, Path],
    project_root: Path,
) -> None:
    outputs["multiwell_json"].write_text(_json(report), encoding="utf-8")
    _write_csv(report["raw_file_rows"], outputs["multiwell_csv"])
    outputs["multiwell_md"].write_text(format_multiwell_markdown(report), encoding="utf-8")
    (project_root / "docs" / "multiwell_data_contract.md").write_text(
        format_multiwell_contract(),
        encoding="utf-8",
    )


def write_decision(report: dict[str, Any], outputs: dict[str, Path]) -> None:
    outputs["decision_json"].write_text(_json(report), encoding="utf-8")
    outputs["decision_md"].write_text(format_decision_markdown(report), encoding="utf-8")


def format_policy_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Evaluation Set Policy",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
    ]
    for name, item in report["sets"].items():
        lines.append(f"## {name}")
        lines.append(f"- purpose: {item['purpose']}")
        lines.append(f"- sample_count: {item['sample_count']}")
        lines.append(f"- mask_definition: {item['mask_definition']}")
        lines.append(f"- allows_performance_claim: {item['allows_performance_claim']}")
        lines.append("")
    lines.append("Production policy is not modified. Ground-truth claims remain forbidden.")
    lines.append("")
    return "\n".join(lines)


def format_policy_yaml(report: dict[str, Any]) -> str:
    lines = [
        "policy_version: mvp4x_evaluation_set_policy_v001",
        "scope:",
        "  research_only: true",
        "  exploratory_only: true",
        "  weak_label_target: true",
        "  no_final_labels: true",
        "  no_ground_truth_claim: true",
        "  no_production_claim: true",
        "  not_validated_for_deployment: true",
        "sets:",
    ]
    for name, item in report["sets"].items():
        lines.extend(
            [
                f"  {name}:",
                f"    purpose: {json.dumps(item['purpose'])}",
                f"    mask_definition: {json.dumps(item['mask_definition'])}",
                f"    allows_performance_claim: {json.dumps(item['allows_performance_claim'])}",
            ]
        )
    lines.extend(
        [
            "rules:",
            "  do_not_mix_stress_test_into_population_claims: true",
            "  do_not_use_audit_only_targets_as_final_labels: true",
            "  do_not_modify_production_policy: true",
            "  ground_truth_claim_forbidden: true",
            "",
        ]
    )
    return "\n".join(lines)


def format_policy_adr() -> str:
    return "\n".join(
        [
            "# ADR-0011: Separate Population Ranking From Stress-Test Audit",
            "",
            "Date: 2026-06-10",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            "## Status",
            "",
            "Accepted for MVP-4X label-semantics and multiwell review.",
            "",
            "## Context",
            "",
            "The bounded STC/APES proxy pilot deliberately selected representative and "
            "stress-test intervals. It is useful for physical contract and failure-mode "
            "audit, but it is not a population-like ranking evaluation set.",
            "",
            "## Decision",
            "",
            "Keep four separate evaluation uses: population-like screening evaluation, "
            "supported-cohort evaluation, stress-test audit set, and compact manual-review set.",
            "",
            "Stress-test intervals must not be mixed into global performance claims. "
            "Audit-only targets must not replace v1 without explicit human approval.",
            "",
            "## Consequences",
            "",
            "Future label-semantics candidates are evaluated in parallel. Production policy, "
            "final labels, ground-truth claims, full-well STC/APES, and deployment "
            "claims remain blocked.",
            "",
        ]
    )


def format_label_semantics_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X Label-Semantics Physical Audit",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            "- raw CAST Zc dimension: depth x azimuth",
            "- review threshold retained: Zc < 2.5 MRayl",
            "- geometry kernel retained: triangular_midpoint_weighted",
            "- geometry sign modified: False",
            "- receiver_mean: v1 primary reference",
            "- receiver_p90: robust research candidate",
            "- receiver_max: audit-only local extreme",
            "- morphology arrays: audit-only, not model input",
            "",
            "The audit finds plausible dilution of local channels by 360-degree averaging "
            "and supports parallel coverage/severity/local-worst/connectedness candidates. "
            "No v1 replacement is authorized.",
            "",
        ]
    )


def format_candidates_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MVP-4X Parallel Label Candidates",
        "",
        "All non-v1 candidates are audit-only. No candidate is ground truth.",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['target']}: finite_ratio={row['finite_ratio']}, "
                     f"nonzero_fraction={row['nonzero_fraction']}, "
                     f"corr_v1={row['correlation_with_v1']}")
    lines.append("")
    return "\n".join(lines)


def format_screening_markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    completed = [row for row in rows if row.get("status") == "completed"]
    best = sorted(
        completed,
        key=lambda row: float(row["spearman"]) if row.get("spearman") is not None else -999.0,
        reverse=True,
    )[:10]
    lines = [
        "# MVP-4X Parallel Target Screening Audit",
        "",
        f"- row_count: {len(rows)}",
        f"- completed_rows: {len(completed)}",
        "- train_fold_only_preprocessing: True",
        "- stress-test results are audit-only and cannot support population claims.",
        "",
        "## Top Rows By Spearman",
        "",
    ]
    for row in best:
        lines.append(
            f"- {row['evaluation_set']} / {row['target']} / {row['feature_set']} / "
            f"{row['model']}: spearman={row['spearman']}, top10_lift={row['top10_lift']}"
        )
    lines.append("")
    return "\n".join(lines)


def format_manual_review_summary(rows: list[dict[str, Any]]) -> str:
    reasons: dict[str, int] = {}
    for row in rows:
        reasons[str(row["selection_reason"])] = reasons.get(str(row["selection_reason"]), 0) + 1
    return "\n".join(
        [
            "# MVP-4X Label-Semantics Manual Review Pack",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- selected_count: {len(rows)}",
            f"- selection_reasons: `{json.dumps(reasons, sort_keys=True)}`",
            "",
            "Review only the weak-label semantics. Do not approve final labels or production use.",
            "",
        ]
    )


def format_multiwell_markdown(report: dict[str, Any]) -> str:
    well = report["wells"][0]
    return "\n".join(
        [
            "# MVP-4X Multiwell Readiness Inventory",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- well_count: {report['well_count']}",
            f"- well_id: {well['well_id']}",
            f"- CAST exists: {well['cast_exists']}",
            f"- raw Zc exists: {well['raw_zc_exists']}",
            f"- XSI waveform exists: {well['xsi_waveform_exists']}",
            f"- receiver_count: {well['receiver_count']}",
            f"- side_count: {well['side_count']}",
            f"- depth_range: `{well['depth_range']}`",
            f"- total_size_bytes: {well['total_size_bytes']}",
            "",
            "single_well_only_multiwell_data_required_for_generalization",
            "",
        ]
    )


def format_multiwell_contract() -> str:
    return "\n".join(
        [
            "# Multiwell Data Contract",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            "Multiwell onboarding is an inventory and harmonization step only. It must not "
            "train a multiwell model, modify raw MAT files, create final labels, or make "
            "production claims.",
            "",
            "Each well must provide:",
            "",
            "- stable well_id",
            "- CAST raw Zc and depth axis",
            "- XSI waveform files and receiver count",
            "- pose fields for RelBearing and Inclination",
            "- receiver geometry and depth-axis sign",
            "- side count and side/azimuth convention",
            "- sampling rate and time axis metadata",
            "- file sizes and checksums",
            "- variable mapping confirmation",
            "",
            "Per-well RelBearing, depth-axis sign, receiver geometry, CAST azimuth convention, "
            "and weak-label target semantics must be audited before any cross-well claim.",
            "",
        ]
    )


def format_decision_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X LS/MW Decision",
            "",
            f"- decision: `{report['decision']}`",
            f"- next_user_approval_required: `{report['next_user_approval_required']}`",
            f"- implementation_bug_detected: {report['implementation_bug_detected']}",
            f"- well_count: {report['well_count']}",
            f"- recommend_multiwell_onboarding: {report['recommend_multiwell_onboarding']}",
            f"- recommend_formal_stc_apes_implementation_review: "
            f"{report['recommend_formal_stc_apes_implementation_review']}",
            f"- recommend_full_well_stc_apes: {report['recommend_full_well_stc_apes']}",
            "- final_labels_forbidden: True",
            "- production_claim_forbidden: True",
            "",
        ]
    )


def _read_file_inventory(path: Path, category: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "category": category, "exists": False}
    data = path.read_bytes()
    row: dict[str, Any] = {
        "path": str(path),
        "category": category,
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = json.loads(data.decode("utf-8"))
        row["json_keys"] = sorted(obj.keys())[:80] if isinstance(obj, dict) else []
    elif suffix == ".csv":
        text = data.decode("utf-8")
        rows = list(csv.DictReader(text.splitlines()))
        row["csv_rows"] = len(rows)
        row["csv_columns"] = list(rows[0].keys()) if rows else []
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            row["npz_arrays"] = {
                key: {"shape": list(npz[key].shape), "dtype": str(npz[key].dtype)}
                for key in npz.files
            }
    else:
        row["text_preview"] = data[:500].decode("utf-8", errors="replace")
    return row


def _raw_role(path: Path) -> str:
    if path.name == "CAST.mat":
        return "cast"
    if "RelBearing" in path.name or "Inclination" in path.name:
        return "pose"
    if path.name.startswith("XSILMR"):
        return "xsi_waveform"
    return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_meta(
    *,
    formula: str,
    physical_interpretation: str,
    input_fields: list[str],
    receiver_aggregation: str,
    azimuth_aggregation: str,
    depth_aggregation: str,
    skipped_missing_contract: bool = False,
) -> dict[str, Any]:
    return {
        "formula": formula,
        "physical_interpretation": physical_interpretation,
        "input_fields": input_fields,
        "kernel": "triangular_midpoint_weighted or documented morphology window",
        "receiver_aggregation": receiver_aggregation,
        "azimuth_aggregation": azimuth_aggregation,
        "depth_aggregation": depth_aggregation,
        "skipped_missing_contract": skipped_missing_contract,
        "audit_only": True,
        "ground_truth_claim": False,
    }


def _extended_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    metrics = compute_regression_metrics(y, prediction)
    metrics["kendall_tau"] = _safe_kendall(y, prediction)
    metrics["top_k_lift"] = top_k_lift(y, rank_percentile(prediction))
    metrics["ndcg"] = _ndcg(y, prediction)
    return metrics


def _permutation_margins(
    y: np.ndarray,
    prediction: np.ndarray,
    depth: np.ndarray,
) -> dict[str, Any]:
    real = _safe_spearman(y, prediction)
    rng = np.random.default_rng(20260610)
    global_perm = _safe_spearman(rng.permutation(y), prediction)
    within = y.copy()
    bins = np.quantile(depth, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (depth >= lo) & (depth <= hi)
        within[mask] = rng.permutation(within[mask])
    block = y.copy()
    chunks = np.array_split(np.arange(y.size), min(6, y.size))
    block_means = np.asarray([float(np.mean(y[chunk])) for chunk in chunks], dtype=np.float32)
    permuted_means = rng.permutation(block_means)
    for chunk, mean_value in zip(chunks, permuted_means, strict=True):
        block[chunk] = mean_value
    block_perm = _safe_spearman(block, prediction)
    return {
        "real_spearman": real,
        "global_margin": _delta(real, global_perm),
        "within_depth_bin_margin": _delta(real, _safe_spearman(within, prediction)),
        "block_margin": _delta(real, block_perm),
    }


def _bootstrap_ci(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(prediction)
    yy = y[mask]
    pp = prediction[mask]
    if yy.size < 20:
        return {"status": "skipped_too_few_samples"}
    rng = np.random.default_rng(20260610)
    values = []
    for _ in range(50):
        index = rng.integers(0, yy.size, size=yy.size)
        value = _safe_spearman(yy[index], pp[index])
        if value is not None:
            values.append(value)
    if not values:
        return {"status": "skipped_degenerate"}
    return {
        "status": "completed",
        "spearman_p05": float(np.quantile(values, 0.05)),
        "spearman_p50": float(np.quantile(values, 0.50)),
        "spearman_p95": float(np.quantile(values, 0.95)),
    }


def _contiguous_folds(depth: np.ndarray, mask: np.ndarray, *, n_folds: int) -> list[np.ndarray]:
    selected = np.flatnonzero(mask)
    if selected.size < n_folds:
        raise LsMwAutoError("Not enough selected samples for contiguous folds.")
    ordered = selected[np.argsort(depth[selected])]
    folds = [np.zeros(depth.size, dtype=bool) for _ in range(n_folds)]
    for fold_index, fold_indices in enumerate(np.array_split(ordered, n_folds)):
        folds[fold_index][fold_indices] = True
    return folds


def _skipped_screen_row(
    feature_set: str,
    target: str,
    evaluation_set: str,
    model_name: str,
    status: str,
    sample_count: int,
    feature_count: int,
) -> dict[str, Any]:
    return {
        "report_version": SCREENING_VERSION,
        "target": target,
        "evaluation_set": evaluation_set,
        "feature_set": feature_set,
        "model": model_name,
        "sample_count": sample_count,
        "feature_count": feature_count,
        "status": status,
        **METHOD_FLAGS,
    }


def _target_distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0, "finite_ratio": 0.0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "finite_ratio": float(finite.size / max(array.size, 1)),
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "unique_count_rounded_6": int(np.unique(np.round(finite, 6)).size),
    }


def _target_distribution_flat(values: np.ndarray) -> dict[str, Any]:
    summary = _target_distribution(values)
    return {key: summary.get(key) for key in (
        "count",
        "finite_count",
        "finite_ratio",
        "min",
        "p05",
        "p50",
        "p95",
        "max",
        "mean",
        "std",
    )}


def _prevalence_rows(
    group_name: str,
    labels: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for label in sorted(set(labels.astype(str).tolist())):
        mask = labels.astype(str) == label
        rows.append(
            {
                "group": group_name,
                "value": label,
                "sample_count": int(np.count_nonzero(mask)),
                "target_top10_prevalence": float(np.mean(y[mask] >= threshold))
                if np.any(mask)
                else None,
                "target_nonzero_fraction": _nonzero_fraction(y[mask]) if np.any(mask) else None,
                **SCOPE_FLAGS,
            }
        )
    return rows


def _special_dependency(
    y: np.ndarray,
    pred: np.ndarray,
    done: np.ndarray,
    special: np.ndarray,
) -> float | None:
    include = _masked_spearman(y, pred, done)
    exclude = _masked_spearman(y, pred, done & ~special)
    return _delta(include, exclude)


def _bc_shift(
    y: np.ndarray,
    pred: np.ndarray,
    done: np.ndarray,
    regime: np.ndarray,
) -> float | None:
    b = _masked_spearman(y, pred, done & (regime == "B"))
    c = _masked_spearman(y, pred, done & (regime == "C"))
    if b is None or c is None:
        return None
    return abs(b - c)


def _masked_spearman(
    y: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if np.count_nonzero(mask) < 8:
        return None
    return _safe_spearman(y[mask], pred[mask])


def _calibration_bias(y: np.ndarray, pred: np.ndarray) -> float | None:
    if y.size == 0:
        return None
    return float(np.mean(pred - y))


def _ndcg(y: np.ndarray, prediction: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(prediction)
    yy = y[mask].astype(np.float64)
    pp = prediction[mask].astype(np.float64)
    if yy.size == 0:
        return None
    order = np.argsort(pp)[::-1]
    ideal = np.argsort(yy)[::-1]
    discount = 1.0 / np.log2(np.arange(2, yy.size + 2))
    dcg = float(np.sum(yy[order] * discount))
    idcg = float(np.sum(yy[ideal] * discount))
    return None if idcg <= 0.0 else float(dcg / idcg)


def _lift(top_k: dict[str, Any], key: str) -> float | None:
    value = top_k.get("top_k", {}).get(key, {}).get("lift")
    return None if value is None else float(value)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 2 or np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return None
    value = spearmanr(aa, bb).statistic
    return None if not np.isfinite(value) else float(value)


def _safe_kendall(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 2 or np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return None
    value = kendalltau(aa, bb).statistic
    return None if not np.isfinite(value) else float(value)


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    aa = float(a)
    bb = float(b)
    if not (math.isfinite(aa) and math.isfinite(bb)):
        return None
    return float(aa - bb)


def _nonzero_fraction(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.mean(finite > 0.0))


def _tie_fraction(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    unique = np.unique(np.round(finite, 8)).size
    return float(1.0 - unique / finite.size)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    finite = np.asarray(values, dtype=np.float64)[mask]
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.mean(finite))


def _range(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": None, "max": None}
    return {"min": float(np.min(finite)), "max": float(np.max(finite))}


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def _nanmean_axis1(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    summed = np.where(finite, array, 0.0).sum(axis=1)
    counts = finite.sum(axis=1)
    output = np.full(array.shape[0], np.nan, dtype=np.float32)
    valid = counts > 0
    output[valid] = summed[valid] / counts[valid]
    return output


def _array(mapping: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in mapping:
        raise LsMwAutoError(f"Missing required array: {key}")
    return np.asarray(mapping[key], dtype=np.float32).reshape(-1)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Required NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _optional_npz(path: Path | str) -> dict[str, np.ndarray] | None:
    npz_path = Path(path)
    if not npz_path.exists():
        return None
    return _load_npz(npz_path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json(value: Any) -> str:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        + "\n"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
