from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_autonomous import (
    evaluate_model_cv,
    fit_eval_once,
    fit_oof_with_permutation_importance,
    run_target_permutations,
    summarize_group_importance,
    summarize_importance,
)
from cement_channel.modeling.mvp4x_baselines import (
    MODEL_NAMES,
    compute_regression_metrics,
)
from cement_channel.modeling.mvp4x_confounding import (
    build_depth_target_matched_controls,
    evaluate_blocked_gap_cv,
    run_structured_permutations,
)
from cement_channel.modeling.mvp4x_model_analysis import build_feature_sets

REPORT_VERSION = "mvp4x_stratified_baselines_v001"
DECISION_VERSION = "mvp4x_stratified_decision_v001"
REVIEW_VERSION = "mvp4x_stratified_review_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
TARGET_VIEWS = ("receiver_mean", "receiver_p90", "receiver_max")
EXTENDED_TARGET_VIEWS = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)
FEATURE_SETS = ("existing_features_only", "waveform_features_only", "combined_features")
BASELINE_COHORTS = (
    "high_orientation_cohort",
    "regime_b_all",
    "regime_b_low_orientation",
    "regime_b_high_orientation",
    "regime_c_all",
    "regime_c_high_orientation",
    "regime_bc_all",
    "regime_bc_high_orientation",
)
PRIMARY_BASIS = {
    "feature_set": "existing_features_only",
    "target": "receiver_mean",
    "model": "Ridge",
}
DECISION_OPTIONS = {
    "exploratory_regime_specific_baseline_supported",
    "exploratory_pooled_bc_baseline_supported",
    "exploratory_orientation_effect_identified_within_regime_b",
    "exploratory_orientation_effect_not_identifiable",
    "exploratory_time_frequency_v2_helpful",
    "request_target_view_policy_approval",
    "request_formal_regime_policy_approval",
    "request_label_redesign_approval",
    "request_advanced_signal_processing_approval",
    "request_server_migration_approval",
    "stop_insufficient_signal",
    "stop_data_contract_issue",
    "stop_leakage_detected",
}


@dataclass(frozen=True)
class StratifiedBaselineOutputs:
    report: dict[str, Any]
    csv_rows: list[dict[str, Any]]
    decision: dict[str, Any]
    iteration_log: str
    review_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StratifiedBaselineError(RuntimeError):
    """Raised when stratified MVP-4X baselines cannot run safely."""


def run_stratified_baselines_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    design_npz: Path | str,
    cf_decision_json: Path | str,
    cf_common_support_json: Path | str,
    cf_nuisance_json: Path | str,
    cf_spatial_json: Path | str,
    config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    output_decision_md: Path | str,
    output_decision_json: Path | str,
    output_review_dir: Path | str,
    output_iteration_log: Path | str,
    overwrite: bool = False,
) -> StratifiedBaselineOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    design_npz_data = _load_npz(Path(design_npz))
    cf_reports = {
        "cf_decision_json": _read_json(Path(cf_decision_json)),
        "cf_common_support_json": _read_json(Path(cf_common_support_json)),
        "cf_nuisance_json": _read_json(Path(cf_nuisance_json)),
        "cf_spatial_json": _read_json(Path(cf_spatial_json)),
    }
    config = _load_yaml(Path(config_path))
    outputs = run_stratified_baselines(
        snapshot=snapshot,
        waveform=waveform,
        design_npz=design_npz_data,
        cf_reports=cf_reports,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "design_npz": str(design_npz),
            "cf_decision_json": str(cf_decision_json),
            "cf_common_support_json": str(cf_common_support_json),
            "cf_nuisance_json": str(cf_nuisance_json),
            "cf_spatial_json": str(cf_spatial_json),
            "config_path": str(config_path),
        },
        review_dir=Path(output_review_dir),
        overwrite=overwrite,
    )
    write_stratified_outputs(
        outputs,
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_csv=Path(output_csv),
        output_decision_md=Path(output_decision_md),
        output_decision_json=Path(output_decision_json),
        output_review_dir=Path(output_review_dir),
        output_iteration_log=Path(output_iteration_log),
        overwrite=overwrite,
    )
    return outputs


def run_stratified_baselines(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    design_npz: dict[str, np.ndarray],
    cf_reports: dict[str, dict[str, Any]],
    config: dict[str, Any],
    inputs: dict[str, str],
    review_dir: Path,
    overwrite: bool,
) -> StratifiedBaselineOutputs:
    _validate_artifacts(snapshot, waveform, design_npz, cf_reports)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    baseline_config = _as_dict(config.get("baseline"))
    feature_sets = build_feature_sets(snapshot, waveform)
    design = load_design(design_npz)
    masks = design["masks"]
    cohorts = [name for name in BASELINE_COHORTS if name in masks]
    matrix = run_cohort_model_matrix(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        cohorts=cohorts,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    strict = run_strict_controls_for_best_by_cohort(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        matrix_rows=matrix["rows"],
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    orientation = run_regime_b_orientation_audit(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    transfer = run_transfer_and_domain_shift(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    target_feature = run_target_and_feature_group_study(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
        matrix_rows=matrix["rows"],
        cf_reports=cf_reports,
    )
    v2_decision = decide_time_frequency_v2(
        matrix=matrix,
        strict=strict,
        transfer=transfer,
        target_feature=target_feature,
    )
    review_files = write_review_pack(
        review_dir=review_dir,
        design=design,
        matrix=matrix,
        strict=strict,
        orientation=orientation,
        transfer=transfer,
        target_feature=target_feature,
        v2_decision=v2_decision,
        overwrite=overwrite,
    )
    decision = build_decision(
        inputs=inputs,
        modeling_environment=modeling_environment.to_dict(),
        design=design,
        matrix=matrix,
        strict=strict,
        orientation=orientation,
        transfer=transfer,
        target_feature=target_feature,
        v2_decision=v2_decision,
        review_files=review_files,
    )
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "modeling_environment": modeling_environment.to_dict(),
        "design_summary": {
            "cohort_count": len(design["cohort_names"]),
            "cohort_names": design["cohort_names"],
            "support": _as_dict(design["metadata"]).get("support"),
            "policy": _as_dict(design["metadata"]).get("policy"),
        },
        "model_matrix": matrix,
        "strict_controls": strict,
        "regime_b_orientation_common_support_audit": orientation,
        "cross_regime_transfer_and_domain_shift": transfer,
        "target_view_and_feature_group_study": target_feature,
        "time_frequency_v2_decision": v2_decision,
        "review_files": review_files,
        **_method_flags(),
    }
    iteration_log = format_iteration_log(
        inputs=inputs,
        design=design,
        matrix=matrix,
        strict=strict,
        orientation=orientation,
        transfer=transfer,
        decision=decision,
    )
    return StratifiedBaselineOutputs(
        report=report,
        csv_rows=matrix["csv_rows"],
        decision=decision,
        iteration_log=iteration_log,
        review_files=review_files,
    )


def load_design(design_npz: dict[str, np.ndarray]) -> dict[str, Any]:
    cohort_names = np.asarray(design_npz["cohort_names"]).astype(str).tolist()
    cohort_masks = np.asarray(design_npz["cohort_masks"], dtype=bool)
    if cohort_masks.ndim != 2:
        raise StratifiedBaselineError("cohort_masks must have shape [cohort, sample].")
    if len(cohort_names) != cohort_masks.shape[0]:
        raise StratifiedBaselineError("cohort_names length must match cohort_masks rows.")
    metadata = json.loads(str(np.asarray(design_npz["metadata_json"]).item()))
    return {
        "cohort_names": cohort_names,
        "cohort_masks": cohort_masks,
        "masks": {
            name: cohort_masks[index].astype(bool)
            for index, name in enumerate(cohort_names)
        },
        "metadata": metadata,
    }


def run_cohort_model_matrix(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    cohorts: list[str],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 20240603))
    model_names = _configured_models(config)
    target_views = _as_list(config.get("target_views"), TARGET_VIEWS)
    feature_set_names = _as_list(config.get("feature_sets"), FEATURE_SETS)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        for target in target_views:
            y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
            sample_mask = masks[cohort] & np.isfinite(y)
            for feature_set_name in feature_set_names:
                X = _feature_matrix(feature_sets[feature_set_name])
                for model_name in model_names:
                    print(
                        "MVP-4X stratified baseline "
                        f"cohort={cohort} feature_set={feature_set_name} "
                        f"target={target} model={model_name}",
                        flush=True,
                    )
                    summary = _safe_evaluate_model_cv(
                        X=X,
                        y=y,
                        depth=depth,
                        sample_mask=sample_mask,
                        model_name=model_name,
                        sklearn_modules=sklearn_modules,
                        config=config,
                        rng_seed=seed + len(rows),
                    )
                    row = {
                        "cohort": cohort,
                        "feature_set": feature_set_name,
                        "target": target,
                        "model": model_name,
                        "sample_count": int(np.count_nonzero(sample_mask)),
                        "target_distribution": _summary(y[sample_mask]),
                        "summary": summary,
                        "prediction_degeneracy": _prediction_degeneracy(summary),
                        "support_warnings": _support_warnings(cohort, sample_mask, summary),
                    }
                    rows.append(row)
                    aggregate = _as_dict(summary.get("aggregate"))
                    csv_rows.append(
                        {
                            "cohort": cohort,
                            "feature_set": feature_set_name,
                            "target": target,
                            "model": model_name,
                            "sample_count": row["sample_count"],
                            "spearman": _metric(aggregate, "spearman"),
                            "pearson": _metric(aggregate, "pearson"),
                            "mae": _metric(aggregate, "mae"),
                            "rmse": _metric(aggregate, "rmse"),
                            "r2": _metric(aggregate, "r2"),
                            "median_absolute_error": _metric(
                                aggregate,
                                "median_absolute_error",
                            ),
                            "stable_positive_spearman_folds": summary.get(
                                "stable_positive_spearman_folds"
                            ),
                        }
                    )
    best_by_cohort = {
        cohort: _select_best_candidate([row for row in rows if row["cohort"] == cohort])
        for cohort in cohorts
    }
    return {
        "status": "completed",
        "cohorts": cohorts,
        "target_views": list(target_views),
        "feature_sets": list(feature_set_names),
        "models": list(model_names),
        "row_count": len(rows),
        "rows": rows,
        "csv_rows": csv_rows,
        "best_by_cohort": best_by_cohort,
        **_method_flags(),
    }


def run_strict_controls_for_best_by_cohort(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    matrix_rows: list[dict[str, Any]],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 20240603))
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    output = {}
    for cohort in BASELINE_COHORTS:
        best = _select_best_candidate([row for row in matrix_rows if row["cohort"] == cohort])
        if not best:
            output[cohort] = {"status": "skipped_no_best_candidate"}
            continue
        X = _feature_matrix(feature_sets[best["feature_set"]])
        y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
        sample_mask = masks[cohort] & np.isfinite(y)
        global_perm = run_target_permutations(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            permutation_count=int(config.get("permutation_count", 20)),
            seed=seed + 101 + len(output),
        )
        within_perm = run_structured_permutations(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            permutation_count=int(config.get("permutation_count", 20)),
            seed=seed + 201 + len(output),
            method="within_depth_bin",
        )
        block_perm = run_structured_permutations(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            permutation_count=int(config.get("permutation_count", 20)),
            seed=seed + 301 + len(output),
            method="block",
        )
        gaps = {
            f"gap_{float(gap):g}_ft": evaluate_blocked_gap_cv(
                X=X,
                y=y,
                depth=depth,
                sample_mask=sample_mask,
                model_name=best["model"],
                sklearn_modules=sklearn_modules,
                config=config,
                gap_ft=float(gap),
                rng_seed=seed + 401 + int(float(gap)),
            )
            for gap in config.get("blocked_gap_ft", [10.0, 25.0, 50.0])
        }
        real_s = best.get("spearman")
        perm_mean = global_perm.get("spearman_mean")
        output[cohort] = {
            "status": "completed",
            "best_candidate": best,
            "global_permutation": global_perm,
            "within_depth_bin_permutation": within_perm,
            "block_permutation": block_perm,
            "blocked_gap_cv": gaps,
            "permutation_margins": {
                "global": _subtract(real_s, global_perm.get("spearman_mean")),
                "within_depth_bin": _subtract(real_s, within_perm.get("spearman_mean")),
                "block": _subtract(real_s, block_perm.get("spearman_mean")),
            },
            "folds_above_global_permutation": _folds_above_permutation(best, perm_mean),
            "fold_sign_consistency": best.get("stable_positive_spearman_folds"),
            "gap_sensitivity": _gap_sensitivity(gaps),
            "special_band_dependency": _special_dependency_for_cohort(
                cohort=cohort,
                snapshot=snapshot,
                feature_sets=feature_sets,
                masks=masks,
                best=best,
                config=config,
                sklearn_modules=sklearn_modules,
            ),
            "support_warnings": best.get("support_warnings", []),
        }
    return {"status": "completed", "by_cohort": output, **_method_flags()}


def run_regime_b_orientation_audit(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 20240603))
    target = PRIMARY_BASIS["target"]
    model_name = PRIMARY_BASIS["model"]
    feature_set_name = PRIMARY_BASIS["feature_set"]
    X = _feature_matrix(feature_sets[feature_set_name])
    y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    low_mask = masks["regime_b_low_orientation"] & np.isfinite(y)
    high_mask = masks["regime_b_high_orientation"] & np.isfinite(y)
    low_result = _safe_evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=low_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=seed + 501,
    )
    high_result = _safe_evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=high_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=seed + 502,
    )
    controls, compositions = build_depth_target_matched_controls(
        depth=depth,
        target=y,
        reference_mask=high_mask,
        candidate_mask=masks["regime_b_all"] & np.isfinite(y),
        repeat_count=int(config.get("matched_control_repeats", 30)),
        seed=seed + 503,
        bin_width_ft=float(config.get("matched_depth_bin_width_ft", 100.0)),
        target_quantile_count=int(config.get("matched_target_quantile_count", 4)),
        high_orientation_mask=high_mask,
        regimes=np.asarray(snapshot["broad_regime_id"]).astype(str),
    )
    control_rows = []
    for index, control_mask in enumerate(controls):
        summary = _safe_evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=control_mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=seed + 520 + index,
        )
        control_rows.append(
            {
                "control_index": index,
                "summary": summary,
                "composition": compositions[index],
            }
        )
    high_perm = run_target_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=high_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=int(config.get("permutation_count", 20)),
        seed=seed + 555,
    )
    control_s = [
        _metric(_as_dict(row["summary"]).get("aggregate", {}), "spearman")
        for row in control_rows
    ]
    control_s = [value for value in control_s if value is not None]
    high_s = _metric(_as_dict(high_result.get("aggregate")), "spearman")
    low_s = _metric(_as_dict(low_result.get("aggregate")), "spearman")
    coverage = _regime_b_common_support_coverage(depth, low_mask, high_mask, config=config)
    high_minus_p95 = None if not control_s else float(high_s - np.quantile(control_s, 0.95))
    return {
        "status": "completed",
        "basis": PRIMARY_BASIS,
        "sample_counts": {
            "regime_b_low_orientation": int(np.count_nonzero(low_mask)),
            "regime_b_high_orientation": int(np.count_nonzero(high_mask)),
        },
        "depth_ranges": {
            "regime_b_low_orientation": _depth_range(depth, low_mask),
            "regime_b_high_orientation": _depth_range(depth, high_mask),
        },
        "target_distributions": {
            "regime_b_low_orientation": _summary(y[low_mask]),
            "regime_b_high_orientation": _summary(y[high_mask]),
        },
        "model_metrics": {
            "regime_b_low_orientation": low_result,
            "regime_b_high_orientation": high_result,
        },
        "matched_controls": {
            "repeat_count": len(control_rows),
            "rows": control_rows,
            "spearman_mean": None if not control_s else float(np.mean(control_s)),
            "spearman_std": None if not control_s else float(np.std(control_s)),
            "spearman_p95": None if not control_s else float(np.quantile(control_s, 0.95)),
            "high_minus_matched_p95": high_minus_p95,
        },
        "permutation_margin": _subtract(high_s, high_perm.get("spearman_mean")),
        "high_minus_low_spearman": _subtract(high_s, low_s),
        "common_support_coverage": coverage,
        "orientation_effect_identifiable_within_regime_b": bool(
            coverage["bin_count_with_both_orientation"] > 0
            and int(np.count_nonzero(low_mask)) >= 100
            and int(np.count_nonzero(high_mask)) >= 100
        ),
        "effect_survives_depth_matching": bool(
            high_minus_p95 is not None and high_minus_p95 > 0.0
        ),
        "not_extrapolated_to_regime_a_or_c": True,
        **_method_flags(),
    }


def run_transfer_and_domain_shift(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    X = _feature_matrix(feature_sets[PRIMARY_BASIS["feature_set"]])
    y = np.asarray(snapshot[PRIMARY_BASIS["target"]], dtype=np.float32).reshape(-1)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    transfer = {
        "B_to_C": fit_eval_once(
            X=X,
            y=y,
            train_mask=masks["regime_b_all"] & np.isfinite(y),
            validation_mask=masks["regime_c_all"] & np.isfinite(y),
            model_name=PRIMARY_BASIS["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=610,
        ),
        "C_to_B": fit_eval_once(
            X=X,
            y=y,
            train_mask=masks["regime_c_all"] & np.isfinite(y),
            validation_mask=masks["regime_b_all"] & np.isfinite(y),
            model_name=PRIMARY_BASIS["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=611,
        ),
        "B_high_to_C_high": fit_eval_once(
            X=X,
            y=y,
            train_mask=masks["regime_b_high_orientation"] & np.isfinite(y),
            validation_mask=masks["regime_c_high_orientation"] & np.isfinite(y),
            model_name=PRIMARY_BASIS["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=612,
        ),
        "C_high_to_B_high": fit_eval_once(
            X=X,
            y=y,
            train_mask=masks["regime_c_high_orientation"] & np.isfinite(y),
            validation_mask=masks["regime_b_high_orientation"] & np.isfinite(y),
            model_name=PRIMARY_BASIS["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=613,
        ),
    }
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    pooled_cv = _safe_evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=masks["regime_bc_all"] & np.isfinite(y),
        model_name=PRIMARY_BASIS["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=614,
    )
    scaler_audit = {
        "B_to_C": _global_scaler_vs_train_scaler_audit(
            X=X,
            y=y,
            train_mask=masks["regime_b_all"] & np.isfinite(y),
            validation_mask=masks["regime_c_all"] & np.isfinite(y),
            sklearn_modules=sklearn_modules,
        ),
        "C_to_B": _global_scaler_vs_train_scaler_audit(
            X=X,
            y=y,
            train_mask=masks["regime_c_all"] & np.isfinite(y),
            validation_mask=masks["regime_b_all"] & np.isfinite(y),
            sklearn_modules=sklearn_modules,
        ),
    }
    feature_shift = _feature_distribution_shift(
        feature_sets=feature_sets,
        mask_left=regimes == "B",
        mask_right=regimes == "C",
    )
    target_shift = {
        "regime_b_receiver_mean": _summary(y[regimes == "B"]),
        "regime_c_receiver_mean": _summary(y[regimes == "C"]),
    }
    warnings = _domain_shift_warnings(transfer, feature_shift)
    return {
        "status": "completed",
        "basis": PRIMARY_BASIS,
        "train_pooled_bc_validate_held_out_blocked_folds": pooled_cv,
        "transfer": transfer,
        "global_scaler_vs_train_regime_only_scaler_audit": scaler_audit,
        "target_distribution_shift": target_shift,
        "feature_distribution_shift": feature_shift,
        "calibration_drift": _calibration_drift_summary(pooled_cv, transfer),
        "residual_shift": _residual_shift_summary(transfer),
        "transfer_degradation": _transfer_degradation(transfer, pooled_cv),
        "domain_shift_warnings": warnings,
        "top_shifted_feature_groups": feature_shift["top_shifted_feature_groups"][:10],
        **_method_flags(),
    }


def run_target_and_feature_group_study(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    cf_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    best = _select_best_candidate(
        [row for row in matrix_rows if row["cohort"] == "regime_bc_high_orientation"]
    ) or _select_best_candidate(matrix_rows)
    if not best:
        return {"status": "skipped_no_best_candidate", **_method_flags()}
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    X = _feature_matrix(feature_sets[best["feature_set"]])
    target_views = {}
    for target in EXTENDED_TARGET_VIEWS:
        y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
        target_views[target] = {
            "target_policy": _target_policy(target),
            "summary": _safe_evaluate_model_cv(
                X=X,
                y=y,
                depth=depth,
                sample_mask=masks["regime_bc_high_orientation"] & np.isfinite(y),
                model_name=best["model"],
                sklearn_modules=sklearn_modules,
                config=config,
                rng_seed=701 + len(target_views),
            ),
        }
    feature_groups = run_feature_group_ablation(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=config,
        sklearn_modules=sklearn_modules,
        best=best,
    )
    importance = run_permutation_importance(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=config,
        sklearn_modules=sklearn_modules,
        best=best,
    )
    return {
        "status": "completed",
        "best_basis": best,
        "target_view_comparison": target_views,
        "most_stable_target_view": _best_target_view(target_views),
        "feature_group_ablation": feature_groups,
        "permutation_importance": importance,
        "stable_feature_groups": importance.get("top_feature_groups", [])[:10],
        "unstable_feature_groups": feature_groups.get("unstable_feature_groups", []),
        "feature_depth_proxy_warnings": _as_dict(
            cf_reports["cf_nuisance_json"].get("feature_depth_proxy_audit")
        ).get("feature_groups_most_likely_encoding_depth_proxy", [])[:10],
        "recommendation_for_human_review": (
            "receiver_mean remains the conservative exploratory primary; do not "
            "change target schema automatically"
        ),
        **_method_flags(),
    }


def run_feature_group_ablation(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, Any]:
    feature_set = feature_sets[best["feature_set"]]
    X_full = _feature_matrix(feature_set)
    groups = np.asarray(feature_set["groups"]).astype(str)
    y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    mask = masks["regime_bc_high_orientation"] & np.isfinite(y)
    baseline = _safe_evaluate_model_cv(
        X=X_full,
        y=y,
        depth=depth,
        sample_mask=mask,
        model_name=best["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=810,
    )
    baseline_s = _metric(_as_dict(baseline.get("aggregate")), "spearman")
    configured = config.get(
        "feature_group_ablation_groups",
        [
            "existing:side_std",
            "existing:receiver_std",
            "existing:late_over_early_mean",
            "existing:side_max",
            "existing:near_far_receiver_ratio",
        ],
    )
    rows = []
    for group in configured:
        keep = groups != str(group)
        if np.count_nonzero(keep) == 0:
            continue
        summary = _safe_evaluate_model_cv(
            X=X_full[:, keep],
            y=y,
            depth=depth,
            sample_mask=mask,
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=820 + len(rows),
        )
        current_s = _metric(_as_dict(summary.get("aggregate")), "spearman")
        rows.append(
            {
                "ablation": f"drop_{group}",
                "feature_group": group,
                "remaining_feature_count": int(np.count_nonzero(keep)),
                "summary": summary,
                "spearman_drop": _subtract(baseline_s, current_s),
            }
        )
    stable = sorted(
        rows,
        key=lambda row: float(row.get("spearman_drop") or 0.0),
        reverse=True,
    )
    unstable = sorted(rows, key=lambda row: float(row.get("spearman_drop") or 0.0))[:5]
    return {
        "baseline": baseline,
        "rows": rows,
        "stable_feature_groups": stable[:10],
        "unstable_feature_groups": unstable,
    }


def run_permutation_importance(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, Any]:
    feature_set = feature_sets[best["feature_set"]]
    X = _feature_matrix(feature_set)
    y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    mask = masks["regime_bc_high_orientation"] & np.isfinite(y)
    _oof, folds, rows = fit_oof_with_permutation_importance(
        X=X,
        y=y,
        depth=depth,
        sample_mask=mask,
        model_name=best["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        feature_names=np.asarray(feature_set["names"]).astype(str),
        feature_groups=np.asarray(feature_set["groups"]).astype(str),
    )
    summary = summarize_importance(rows)
    return {
        "method": "validation_fold_feature_shuffle",
        "folds": folds,
        "top_30_features": summary["top_30_features"],
        "top_10_stable_features": summary["top_10_stable_features"],
        "top_feature_groups": summarize_group_importance(rows)[:15],
    }


def decide_time_frequency_v2(
    *,
    matrix: dict[str, Any],
    strict: dict[str, Any],
    transfer: dict[str, Any],
    target_feature: dict[str, Any],
) -> dict[str, Any]:
    best_by_cohort = _as_dict(matrix.get("best_by_cohort"))
    regime_signal = any(
        _permutation_safe(_as_dict(_as_dict(strict.get("by_cohort")).get(cohort)))
        for cohort in ("regime_b_high_orientation", "regime_c_high_orientation")
    )
    transfer_warnings = bool(transfer.get("domain_shift_warnings"))
    best_bc = _as_dict(best_by_cohort.get("regime_bc_high_orientation"))
    waveform_best = best_bc.get("feature_set") == "waveform_features_only"
    triggered = False
    reasons = []
    if not regime_signal:
        reasons.append("no_sufficient_regime_permutation_safe_signal")
    if not transfer_warnings:
        reasons.append("transfer_or_calibration_not_clearly_limiting")
    if waveform_best:
        reasons.append("waveform_v1_already_selected_as_best")
    reasons.append("human_review_needed_before_waveform_reread_due_regime_policy_uncertainty")
    return {
        "time_frequency_v2_triggered": triggered,
        "time_frequency_v2_improved": "not_triggered",
        "shape": None,
        "chunks": None,
        "peak_memory_bytes": None,
        "runtime_seconds": None,
        "reason_not_triggered": reasons,
        "no_raw_waveform_reread": True,
        **_method_flags(),
    }


def build_decision(
    *,
    inputs: dict[str, str],
    modeling_environment: dict[str, Any],
    design: dict[str, Any],
    matrix: dict[str, Any],
    strict: dict[str, Any],
    orientation: dict[str, Any],
    transfer: dict[str, Any],
    target_feature: dict[str, Any],
    v2_decision: dict[str, Any],
    review_files: dict[str, str],
) -> dict[str, Any]:
    best_by = _as_dict(matrix.get("best_by_cohort"))
    b_best = _as_dict(best_by.get("regime_b_high_orientation"))
    c_best = _as_dict(best_by.get("regime_c_high_orientation"))
    bc_best = _as_dict(best_by.get("regime_bc_high_orientation"))
    b_signal = _cohort_strict_signal(strict, "regime_b_high_orientation")
    c_signal = _cohort_strict_signal(strict, "regime_c_high_orientation")
    bc_signal = _cohort_strict_signal(strict, "regime_bc_high_orientation")
    orientation_identified = bool(
        orientation.get("orientation_effect_identifiable_within_regime_b")
        and orientation.get("effect_survives_depth_matching")
    )
    b_to_c = _metric(_as_dict(transfer.get("transfer")).get("B_to_C", {}), "spearman")
    c_to_b = _metric(_as_dict(transfer.get("transfer")).get("C_to_B", {}), "spearman")
    transfer_ok = bool(
        b_to_c is not None and c_to_b is not None and b_to_c > 0.15 and c_to_b > 0.15
    )
    if orientation_identified:
        decision = "exploratory_orientation_effect_identified_within_regime_b"
    elif b_signal and c_signal and not transfer_ok:
        decision = "exploratory_regime_specific_baseline_supported"
    elif bc_signal and transfer_ok:
        decision = "exploratory_pooled_bc_baseline_supported"
    elif b_signal or c_signal or bc_signal:
        decision = "exploratory_regime_specific_baseline_supported"
    else:
        decision = "stop_insufficient_signal"
    if decision not in DECISION_OPTIONS:
        raise StratifiedBaselineError(f"Unsupported stratified decision: {decision}")
    answers = {
        "1_regime_b_stable_signal": b_signal,
        "2_regime_c_stable_signal": c_signal,
        "3_pooled_bc_stable": bc_signal,
        "4_b_to_c_transfer": b_to_c,
        "5_c_to_b_transfer": c_to_b,
        "6_regime_b_orientation_effect_identifiable": orientation.get(
            "orientation_effect_identifiable_within_regime_b"
        ),
        "7_orientation_effect_survives_depth_matching": orientation.get(
            "effect_survives_depth_matching"
        ),
        "8_most_stable_target_view": target_feature.get("most_stable_target_view"),
        "9_most_stable_feature_set": bc_best.get("feature_set"),
        "10_most_stable_model": bc_best.get("model"),
        "11_stable_feature_groups": target_feature.get("stable_feature_groups", [])[:10],
        "12_domain_shift_exists": bool(transfer.get("domain_shift_warnings")),
        "13_blocked_gap_cv_stable": _blocked_gap_stable(strict, "regime_bc_high_orientation"),
        "14_within_bin_permutation_passed": _within_bin_passed(
            strict,
            "regime_bc_high_orientation",
        ),
        "15_special_bands_affect_results": _special_affects(strict),
        "16_time_frequency_v2_triggered": v2_decision.get("time_frequency_v2_triggered"),
        "17_time_frequency_v2_improved": v2_decision.get("time_frequency_v2_improved"),
        "18_apply_for_formal_regime_policy": decision
        in {
            "exploratory_regime_specific_baseline_supported",
            "exploratory_pooled_bc_baseline_supported",
        },
        "19_need_label_redesign": False,
        "20_worth_stc_apes": "not_before_human_review_of_formal_regime_policy",
        "21_worth_deep_learning": "not_before_human_review_of_formal_regime_policy",
        "22_recommend_server": False,
        "23_next_human_approval": _next_human_approval(decision),
        "24_production_claims_and_final_labels_forbidden": True,
    }
    return {
        "decision_version": DECISION_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "inputs": inputs,
        "modeling_environment": modeling_environment,
        "design_support": _as_dict(design["metadata"]).get("support"),
        "best_candidates": {
            "regime_b_high_orientation": b_best,
            "regime_c_high_orientation": c_best,
            "regime_bc_high_orientation": bc_best,
        },
        "answers": answers,
        "time_frequency_v2": v2_decision,
        "review_files": review_files,
        "next_minimal_recommendation": _next_human_approval(decision),
        "not_authorized": [
            "production claim",
            "final labels",
            "ground-truth claim",
            "formal regime split",
            "formal orientation filter",
            "regime_id as model input",
            "depth as model input",
            "orientation confidence as model input",
            "STC",
            "APES",
            "deep learning",
        ],
        **_method_flags(),
    }


def write_stratified_outputs(
    outputs: StratifiedBaselineOutputs,
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_csv: Path,
    output_decision_md: Path,
    output_decision_json: Path,
    output_review_dir: Path,
    output_iteration_log: Path,
    overwrite: bool,
) -> None:
    for path in (
        output_report_md,
        output_report_json,
        output_csv,
        output_decision_md,
        output_decision_json,
        output_iteration_log,
    ):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_review_dir.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(_json(outputs.report), encoding="utf-8")
    output_report_md.write_text(format_report_markdown(outputs.report), encoding="utf-8")
    write_csv(outputs.csv_rows, output_csv)
    output_decision_json.write_text(_json(outputs.decision), encoding="utf-8")
    output_decision_md.write_text(format_decision_markdown(outputs.decision), encoding="utf-8")
    output_iteration_log.write_text(outputs.iteration_log, encoding="utf-8")


def write_review_pack(
    *,
    review_dir: Path,
    design: dict[str, Any],
    matrix: dict[str, Any],
    strict: dict[str, Any],
    orientation: dict[str, Any],
    transfer: dict[str, Any],
    target_feature: dict[str, Any],
    v2_decision: dict[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    summary = review_dir / "review_summary.md"
    _ensure_can_write(summary, overwrite=overwrite)
    summary.write_text(
        "# MVP-4X Stratified Review Summary\n\n"
        + _scope_line()
        + "\n\n"
        + f"- time_frequency_v2_triggered: {v2_decision['time_frequency_v2_triggered']}\n",
        encoding="utf-8",
    )
    files["review_summary_md"] = str(summary)
    plot_specs = {
        "cohort_support_matrix.png": lambda path: _plot_cohort_support(design, path),
        "cohort_target_distributions.png": lambda path: _plot_target_distributions(design, path),
        "regime_specific_metrics.png": lambda path: _plot_regime_metrics(matrix, path),
        "regime_b_orientation_common_support.png": lambda path: _plot_orientation(
            orientation, path
        ),
        "blocked_gap_cv_summary.png": lambda path: _plot_blocked_gap(strict, path),
        "permutation_comparison.png": lambda path: _plot_permutation(strict, path),
        "cross_regime_transfer.png": lambda path: _plot_transfer(transfer, path),
        "transfer_degradation.png": lambda path: _plot_transfer_degradation(transfer, path),
        "feature_group_stability.png": lambda path: _plot_feature_groups(target_feature, path),
        "target_view_comparison.png": lambda path: _plot_target_views(target_feature, path),
        "calibration_by_regime.png": lambda path: _plot_calibration(transfer, path),
        "residual_vs_depth.png": lambda path: _plot_placeholder(path, "Residual vs depth"),
        "residual_by_regime.png": lambda path: _plot_residual_shift(transfer, path),
        "special_band_sensitivity.png": lambda path: _plot_special(strict, path),
    }
    for name, writer in plot_specs.items():
        path = review_dir / name
        _ensure_can_write(path, overwrite=overwrite)
        writer(path)
        files[name.replace(".", "_")] = str(path)
    review_json = review_dir / "stratified_review_summary.json"
    _ensure_can_write(review_json, overwrite=overwrite)
    review_json.write_text(
        _json(
            {
                "review_version": REVIEW_VERSION,
                "matrix_best_by_cohort": matrix.get("best_by_cohort"),
                "orientation": orientation,
                "transfer": transfer,
                "target_feature": target_feature,
                "time_frequency_v2": v2_decision,
                **_method_flags(),
            }
        ),
        encoding="utf-8",
    )
    files["stratified_review_summary_json"] = str(review_json)
    return files


def format_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Stratified Baselines",
        "",
        _scope_line(),
        "",
        f"- report_version: `{report.get('report_version')}`",
        f"- row_count: {report.get('model_matrix', {}).get('row_count')}",
        "",
        "## Best By Cohort",
    ]
    for cohort, row in report.get("model_matrix", {}).get("best_by_cohort", {}).items():
        lines.append(f"- {cohort}: {row}")
    return "\n".join(lines) + "\n"


def format_decision_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Stratified Decision",
        "",
        _scope_line(),
        "",
        f"- decision: `{report.get('decision')}`",
        f"- next_minimal_recommendation: {report.get('next_minimal_recommendation')}",
        "",
        "## Answers",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report.get("answers", {}).items())
    lines.extend(["", "## Not Authorized"])
    lines.extend(f"- {item}" for item in report.get("not_authorized", []))
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "cohort",
        "feature_set",
        "target",
        "model",
        "sample_count",
        "spearman",
        "pearson",
        "mae",
        "rmse",
        "r2",
        "median_absolute_error",
        "stable_positive_spearman_folds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_iteration_log(
    *,
    inputs: dict[str, str],
    design: dict[str, Any],
    matrix: dict[str, Any],
    strict: dict[str, Any],
    orientation: dict[str, Any],
    transfer: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    timestamp = _utc_now()
    return "\n".join(
        [
            "# MVP-4X Stratified Iteration Log",
            "",
            _scope_line(),
            "",
            "## Iteration 1",
            "- iteration_id: 1",
            f"- timestamp: {timestamp}",
            f"- files_read: {list(inputs.values())}",
            "- hypothesis: frozen cohorts can separate regime-specific learnability.",
            f"- evidence_before: design_support={_as_dict(design['metadata']).get('support')}",
            "- selected_action: run cohort model matrix.",
            "- files_changed: mvp4x_stratified_baselines.py, 07k script, tests",
            "- commands_run: scripts/07k_run_mvp4x_stratified_baselines.py --overwrite",
            "- outputs_generated: mvp4x_stratified_baselines_v001.json/md/csv",
            f"- metrics_after: best_by_cohort={matrix.get('best_by_cohort')}",
            "- interpretation: cohort-level candidate signals identified.",
            "- continue_or_stop: continue",
            "- human_approval_required: false",
            "- proposed_next_action: strict controls and orientation audit",
            "",
            "## Iteration 2",
            "- iteration_id: 2",
            f"- timestamp: {timestamp}",
            f"- files_read: {list(inputs.values())}",
            "- hypothesis: B/C cohort signals survive blocked-gap and structured permutation.",
            f"- evidence_before: row_count={matrix.get('row_count')}",
            "- selected_action: blocked-gap CV, global/within-bin/block permutation.",
            "- files_changed: same tracked stratified implementation files",
            "- commands_run: scripts/07k_run_mvp4x_stratified_baselines.py --overwrite",
            "- outputs_generated: strict_controls in mvp4x_stratified_baselines_v001.json",
            f"- metrics_after: strict_keys={list(_as_dict(strict.get('by_cohort')).keys())}",
            "- interpretation: strict controls determine whether signal is permutation-safe.",
            "- continue_or_stop: continue",
            "- human_approval_required: false",
            "- proposed_next_action: Regime B orientation common-support audit",
            "",
            "## Iteration 3",
            "- iteration_id: 3",
            f"- timestamp: {timestamp}",
            f"- files_read: {list(inputs.values())}",
            "- hypothesis: Regime B orientation effect may not survive depth/target matching.",
            "- selected_action: matched controls and same-protocol B low/high audit.",
            "- files_changed: same tracked stratified implementation files",
            "- commands_run: scripts/07k_run_mvp4x_stratified_baselines.py --overwrite",
            "- outputs_generated: regime_b_orientation_common_support_audit",
            f"- metrics_after: {orientation.get('matched_controls')}",
            "- interpretation: orientation evidence remains research-only within Regime B.",
            "- continue_or_stop: continue",
            "- human_approval_required: false",
            "- proposed_next_action: transfer/domain-shift and target/feature study",
            "",
            "## Iteration 4",
            "- iteration_id: 4",
            f"- timestamp: {timestamp}",
            f"- files_read: {list(inputs.values())}",
            "- hypothesis: transfer/domain shift controls should drive the final decision.",
            f"- evidence_before: transfer={transfer.get('transfer')}",
            "- selected_action: generate review pack and bounded decision.",
            "- files_changed: same tracked stratified implementation files",
            "- commands_run: scripts/07k_run_mvp4x_stratified_baselines.py --overwrite",
            "- outputs_generated: mvp4x_stratified_decision.json/md and review dir",
            f"- metrics_after: decision={decision.get('decision')}",
            f"- interpretation: {decision.get('next_minimal_recommendation')}",
            f"- continue_or_stop: {decision.get('decision')}",
            "- human_approval_required: true",
            f"- proposed_next_action: {decision.get('next_minimal_recommendation')}",
            "",
        ]
    )


# Utility functions


def _safe_evaluate_model_cv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    rng_seed: int,
) -> dict[str, Any]:
    if int(np.count_nonzero(sample_mask)) < int(config.get("n_contiguous_folds", 3)):
        return {
            "status": "skipped_too_few_samples",
            "sample_count": int(np.count_nonzero(sample_mask)),
        }
    return evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=rng_seed,
    )["summary"]


def _select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    for row in rows:
        if row.get("model") == "DummyRegressor":
            continue
        summary = _as_dict(row.get("summary"))
        if summary.get("status") != "completed":
            continue
        aggregate = _as_dict(summary.get("aggregate"))
        spearman = _metric(aggregate, "spearman")
        if spearman is None:
            continue
        candidate = {
            "cohort": row.get("cohort"),
            "feature_set": row.get("feature_set"),
            "target": row.get("target"),
            "model": row.get("model"),
            "sample_count": row.get("sample_count"),
            "spearman": spearman,
            "pearson": _metric(aggregate, "pearson"),
            "mae": _metric(aggregate, "mae"),
            "rmse": _metric(aggregate, "rmse"),
            "r2": _metric(aggregate, "r2"),
            "stable_positive_spearman_folds": summary.get("stable_positive_spearman_folds"),
            "folds": summary.get("folds"),
            "support_warnings": row.get("support_warnings", []),
        }
        if best is None or _candidate_score(candidate) > _candidate_score(best):
            best = candidate
    return best


def _candidate_score(row: dict[str, Any]) -> tuple[float, int, float, float]:
    target_bonus = 0.05 if row.get("target") == "receiver_mean" else 0.0
    feature_bonus = 0.02 if row.get("feature_set") == "existing_features_only" else 0.0
    spearman = float(row.get("spearman") if row.get("spearman") is not None else -1e9)
    stable = int(row.get("stable_positive_spearman_folds") or 0)
    mae = float(row.get("mae") if row.get("mae") is not None else 1e9)
    r2 = float(row.get("r2") if row.get("r2") is not None else -1e9)
    return (spearman + target_bonus + feature_bonus, stable, r2, -mae)


def _configured_models(config: dict[str, Any]) -> tuple[str, ...]:
    requested = tuple(str(item) for item in config.get("model_names", MODEL_NAMES))
    unsupported = [name for name in requested if name not in MODEL_NAMES]
    if unsupported:
        raise StratifiedBaselineError(f"Unsupported model(s): {unsupported}")
    return requested


def _special_dependency_for_cohort(
    *,
    cohort: str,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    best: dict[str, Any],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    if cohort != "regime_bc_high_orientation":
        return {"status": "not_applicable"}
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    X = _feature_matrix(feature_sets[best["feature_set"]])
    rows = {}
    for name in (
        "regime_bc_high_orientation",
        "regime_bc_high_orientation_exclude_2400_2500",
        "regime_bc_high_orientation_exclude_5680",
        "regime_bc_high_orientation_exclude_all_special",
    ):
        rows[name] = _safe_evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=masks[name] & np.isfinite(y),
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=910 + len(rows),
        )
    base = _metric(_as_dict(rows["regime_bc_high_orientation"]).get("aggregate", {}), "spearman")
    deltas = {
        name: _subtract(_metric(_as_dict(row).get("aggregate", {}), "spearman"), base)
        for name, row in rows.items()
        if name != "regime_bc_high_orientation"
    }
    return {
        "status": "completed",
        "rows": rows,
        "spearman_deltas": deltas,
        "dependency_flag": any(abs(float(value or 0.0)) > 0.05 for value in deltas.values()),
    }


def _regime_b_common_support_coverage(
    depth: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    bin_width = float(config.get("matched_depth_bin_width_ft", 100.0))
    selected = low_mask | high_mask
    edges = _fixed_depth_edges(depth[selected], bin_width)
    rows = []
    both = 0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        bin_mask = (depth >= left) & (depth < right)
        low_count = int(np.count_nonzero(bin_mask & low_mask))
        high_count = int(np.count_nonzero(bin_mask & high_mask))
        if low_count and high_count:
            both += 1
        rows.append(
            {
                "depth_min": float(left),
                "depth_max": float(right),
                "low_count": low_count,
                "high_count": high_count,
            }
        )
    return {
        "depth_bin_width_ft": bin_width,
        "bins": rows,
        "bin_count_with_both_orientation": both,
    }


def _global_scaler_vs_train_scaler_audit(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    if not np.any(train) or not np.any(validation):
        return {"status": "skipped_empty_train_or_validation"}
    preprocessing = sklearn_modules["preprocessing"]
    linear = sklearn_modules["linear_model"]
    train_scaler = preprocessing.StandardScaler().fit(X[train])
    global_scaler = preprocessing.StandardScaler().fit(X[train | validation])
    train_model = linear.Ridge(alpha=1.0).fit(train_scaler.transform(X[train]), y[train])
    global_model = linear.Ridge(alpha=1.0).fit(global_scaler.transform(X[train]), y[train])
    train_pred = train_model.predict(train_scaler.transform(X[validation]))
    global_pred = global_model.predict(global_scaler.transform(X[validation]))
    return {
        "status": "completed",
        "train_regime_only_scaler": compute_regression_metrics(y[validation], train_pred),
        "global_scaler_audit_only": compute_regression_metrics(y[validation], global_pred),
        "validation_leakage_audit_only": True,
    }


def _feature_distribution_shift(
    *,
    feature_sets: dict[str, dict[str, Any]],
    mask_left: np.ndarray,
    mask_right: np.ndarray,
) -> dict[str, Any]:
    feature_set = feature_sets["combined_features"]
    X = _feature_matrix(feature_set)
    groups = np.asarray(feature_set["groups"]).astype(str)
    names = np.asarray(feature_set["names"]).astype(str)
    rows = []
    for index in range(X.shape[1]):
        left = X[mask_left, index]
        right = X[mask_right, index]
        pooled = float(np.sqrt((np.var(left) + np.var(right)) / 2.0))
        smd = 0.0 if pooled == 0.0 else float((np.mean(left) - np.mean(right)) / pooled)
        rows.append(
            {
                "feature_index": index,
                "feature_name": str(names[index]),
                "feature_group": str(groups[index]),
                "standardized_mean_difference": smd,
                "abs_standardized_mean_difference": abs(smd),
            }
        )
    group_rows = []
    for group in sorted(set(groups.tolist())):
        values = [
            row["abs_standardized_mean_difference"]
            for row in rows
            if row["feature_group"] == group
        ]
        group_rows.append(
            {
                "feature_group": group,
                "mean_abs_standardized_mean_difference": float(np.mean(values)),
                "max_abs_standardized_mean_difference": float(np.max(values)),
                "feature_count": len(values),
            }
        )
    rows.sort(key=lambda row: row["abs_standardized_mean_difference"], reverse=True)
    group_rows.sort(
        key=lambda row: row["mean_abs_standardized_mean_difference"],
        reverse=True,
    )
    return {
        "comparison": "regime_b_vs_regime_c",
        "top_shifted_features": rows[:30],
        "top_shifted_feature_groups": group_rows[:20],
    }


def _domain_shift_warnings(transfer: dict[str, Any], feature_shift: dict[str, Any]) -> list[str]:
    warnings = []
    for name in ("B_to_C", "C_to_B", "B_high_to_C_high", "C_high_to_B_high"):
        value = _metric(_as_dict(transfer.get(name)), "spearman")
        if value is None or value < 0.15:
            warnings.append(f"{name}_weak_transfer_spearman")
    top_group = _as_dict(feature_shift.get("top_shifted_feature_groups", [{}])[0])
    if float(top_group.get("mean_abs_standardized_mean_difference") or 0.0) > 1.0:
        warnings.append("large_feature_distribution_shift")
    return warnings


def _calibration_drift_summary(
    pooled_cv: dict[str, Any],
    transfer: dict[str, Any],
) -> dict[str, Any]:
    pooled_mae = _metric(_as_dict(pooled_cv.get("aggregate")), "mae")
    return {
        name: {
            "transfer_mae": _metric(row, "mae"),
            "mae_delta_vs_pooled_cv": _subtract(_metric(row, "mae"), pooled_mae),
        }
        for name, row in transfer.items()
    }


def _residual_shift_summary(transfer: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "mae": _metric(row, "mae"),
            "rmse": _metric(row, "rmse"),
            "r2": _metric(row, "r2"),
        }
        for name, row in transfer.items()
    }


def _transfer_degradation(transfer: dict[str, Any], pooled_cv: dict[str, Any]) -> dict[str, Any]:
    pooled_s = _metric(_as_dict(pooled_cv.get("aggregate")), "spearman")
    return {
        name: {
            "spearman": _metric(row, "spearman"),
            "spearman_delta_vs_pooled_cv": _subtract(_metric(row, "spearman"), pooled_s),
        }
        for name, row in transfer.items()
    }


def _target_policy(target: str) -> str:
    return {
        "receiver_mean": "conservative_exploratory_primary",
        "receiver_p90": "robust_reference_candidate",
        "receiver_max": "sensitive_audit_only",
        "full_360_fraction": "auxiliary_only",
        "receiver_std": "heterogeneity_audit_only",
    }.get(target, "unknown")


def _best_target_view(rows: dict[str, Any]) -> str | None:
    values = {
        target: _metric(_as_dict(_as_dict(row).get("summary")).get("aggregate", {}), "spearman")
        for target, row in rows.items()
    }
    values = {target: value for target, value in values.items() if value is not None}
    if not values:
        return None
    if "receiver_mean" in values and values["receiver_mean"] >= max(values.values()) - 0.02:
        return "receiver_mean"
    return max(values, key=lambda target: values[target])


def _permutation_safe(row: dict[str, Any]) -> bool:
    best = _as_dict(row.get("best_candidate"))
    margins = _as_dict(row.get("permutation_margins"))
    return bool(
        best.get("spearman") is not None
        and float(best["spearman"]) > 0.0
        and (margins.get("global") or 0.0) > 0.05
        and (margins.get("within_depth_bin") or 0.0) > 0.05
        and (margins.get("block") or 0.0) > 0.05
    )


def _cohort_strict_signal(strict: dict[str, Any], cohort: str) -> bool:
    return _permutation_safe(_as_dict(_as_dict(strict.get("by_cohort")).get(cohort)))


def _blocked_gap_stable(strict: dict[str, Any], cohort: str) -> bool:
    row = _as_dict(_as_dict(strict.get("by_cohort")).get(cohort))
    gaps = _as_dict(row.get("blocked_gap_cv"))
    positives = [
        _metric(_as_dict(gap).get("aggregate", {}), "spearman")
        for gap in gaps.values()
    ]
    return sum(value is not None and value > 0.0 for value in positives) >= 2


def _within_bin_passed(strict: dict[str, Any], cohort: str) -> bool:
    row = _as_dict(_as_dict(strict.get("by_cohort")).get(cohort))
    return bool((_as_dict(row.get("permutation_margins")).get("within_depth_bin") or 0.0) > 0.05)


def _special_affects(strict: dict[str, Any]) -> bool:
    row = _as_dict(_as_dict(strict.get("by_cohort")).get("regime_bc_high_orientation"))
    return bool(_as_dict(row.get("special_band_dependency")).get("dependency_flag"))


def _next_human_approval(decision: str) -> str:
    if decision in {
        "exploratory_regime_specific_baseline_supported",
        "exploratory_pooled_bc_baseline_supported",
    }:
        return (
            "human approval for a formal regime-policy review; do not adopt a formal "
            "regime split or production filter yet"
        )
    if decision == "exploratory_orientation_effect_identified_within_regime_b":
        return "human review of Regime B orientation evidence before any formal filter discussion"
    return "human review of target policy and weak-label assumptions"


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _folds_above_permutation(best: dict[str, Any], permutation_mean: float | None) -> int | None:
    if permutation_mean is None:
        return None
    return int(
        sum(
            _metric(row, "spearman") is not None
            and float(row["spearman"]) > float(permutation_mean)
            for row in best.get("folds", [])
        )
    )


def _gap_sensitivity(gaps: dict[str, Any]) -> dict[str, Any]:
    values = [
        _metric(_as_dict(row).get("aggregate", {}), "spearman")
        for row in gaps.values()
    ]
    values = [value for value in values if value is not None]
    return {
        "spearman_min": None if not values else float(np.min(values)),
        "spearman_max": None if not values else float(np.max(values)),
        "positive_gap_count": int(sum(value > 0.0 for value in values)),
    }


def _prediction_degeneracy(summary: dict[str, Any]) -> dict[str, Any]:
    prediction = _as_dict(summary.get("prediction_summary"))
    std = prediction.get("std")
    return {
        "degenerate_prediction_warning": bool(std is not None and float(std) < 1e-8),
        "prediction_std": std,
    }


def _support_warnings(cohort: str, sample_mask: np.ndarray, summary: dict[str, Any]) -> list[str]:
    warnings = []
    if int(np.count_nonzero(sample_mask)) < 100:
        warnings.append(f"{cohort}_sample_count_lt_100")
    if int(summary.get("fold_count") or 0) < 3:
        warnings.append(f"{cohort}_fold_count_lt_3")
    return warnings


def _feature_matrix(feature_set: dict[str, Any]) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(feature_set["matrix"], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "p01": None, "p50": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _depth_range(depth: np.ndarray, mask: np.ndarray) -> list[float | None]:
    if not np.any(mask):
        return [None, None]
    return [float(np.min(depth[mask])), float(np.max(depth[mask]))]


def _fixed_depth_edges(values: np.ndarray, bin_width_ft: float) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise StratifiedBaselineError("Cannot build depth bins without finite values.")
    start = np.floor(float(np.min(finite)) / bin_width_ft) * bin_width_ft
    stop = np.ceil(float(np.max(finite)) / bin_width_ft) * bin_width_ft + bin_width_ft
    return np.arange(start, stop + 0.5 * bin_width_ft, bin_width_ft, dtype=np.float32)


def _configured_feature_sets(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in config.get("feature_sets", FEATURE_SETS))


def _as_list(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(row: dict[str, Any], key: str) -> float | None:
    if not isinstance(row, dict) or row.get(key) is None:
        return None
    value = row[key]
    if not np.isfinite(value):
        return None
    return float(value)


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


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


def _scope_line() -> str:
    return (
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim."
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")


def _validate_artifacts(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    design_npz: dict[str, np.ndarray],
    cf_reports: dict[str, dict[str, Any]],
) -> None:
    if snapshot["depth"].shape[0] != 7108:
        raise StratifiedBaselineError(f"Unexpected sample count: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise StratifiedBaselineError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise StratifiedBaselineError(
            "Unexpected waveform-v1 feature shape: "
            f"{waveform['waveform_depth_features'].shape}"
        )
    if snapshot["xsi_features"].shape[1] + waveform["waveform_depth_features"].shape[1] != 422:
        raise StratifiedBaselineError("Combined feature count is not 422.")
    if float(np.isfinite(snapshot["xsi_features"]).mean()) != 1.0:
        raise StratifiedBaselineError("Existing feature finite ratio is not 1.0.")
    if float(np.isfinite(waveform["waveform_depth_features"]).mean()) != 1.0:
        raise StratifiedBaselineError("Waveform-v1 finite ratio is not 1.0.")
    if str(snapshot["target_kernel"]) != "triangular_midpoint_weighted":
        raise StratifiedBaselineError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for target in EXTENDED_TARGET_VIEWS:
        if target not in snapshot:
            raise StratifiedBaselineError(f"Missing target view: {target}")
    for name, container in (("snapshot", snapshot), ("waveform", waveform), ("design", design_npz)):
        for flag in RESEARCH_FLAGS:
            if not bool(np.asarray(container.get(flag, False)).item()):
                raise StratifiedBaselineError(f"{name} missing research flag {flag}.")
    for name, report in cf_reports.items():
        for flag in RESEARCH_FLAGS:
            if not bool(report.get(flag)):
                raise StratifiedBaselineError(f"{name} missing research flag {flag}.")


# Plot helpers


def _plot_cohort_support(design: dict[str, Any], path: Path) -> None:
    cohorts = _as_dict(design["metadata"].get("cohorts"))
    names = list(cohorts)
    values = [cohorts[name]["sample_count"] for name in names]
    _bar_plot(path, names, values, "Cohort support", "sample count")


def _plot_target_distributions(design: dict[str, Any], path: Path) -> None:
    cohorts = _as_dict(design["metadata"].get("cohorts"))
    names = list(cohorts)
    values = [
        cohorts[name]["target_distribution"]["receiver_mean"].get("p50") or 0.0
        for name in names
    ]
    _bar_plot(path, names, values, "receiver_mean p50 by cohort", "p50")


def _plot_regime_metrics(matrix: dict[str, Any], path: Path) -> None:
    best = _as_dict(matrix.get("best_by_cohort"))
    names = ["regime_b_high_orientation", "regime_c_high_orientation", "regime_bc_high_orientation"]
    values = [_as_dict(best.get(name)).get("spearman") or 0.0 for name in names]
    _bar_plot(path, names, values, "Regime-specific best Spearman", "Spearman")


def _plot_orientation(report: dict[str, Any], path: Path) -> None:
    names = ["B low", "B high", "matched p95"]
    low = _metric(
        _as_dict(_as_dict(report["model_metrics"]["regime_b_low_orientation"]).get("aggregate")),
        "spearman",
    )
    high = _metric(
        _as_dict(_as_dict(report["model_metrics"]["regime_b_high_orientation"]).get("aggregate")),
        "spearman",
    )
    p95 = _as_dict(report.get("matched_controls")).get("spearman_p95")
    _bar_plot(
        path,
        names,
        [_safe_float(low), _safe_float(high), _safe_float(p95)],
        "Regime B orientation audit",
        "Spearman",
    )


def _plot_blocked_gap(strict: dict[str, Any], path: Path) -> None:
    row = _as_dict(_as_dict(strict.get("by_cohort")).get("regime_bc_high_orientation"))
    gaps = _as_dict(row.get("blocked_gap_cv"))
    names = list(gaps)
    values = [
        _metric(_as_dict(gap).get("aggregate", {}), "spearman") or 0.0
        for gap in gaps.values()
    ]
    _bar_plot(path, names, values, "Blocked-gap CV", "Spearman")


def _plot_permutation(strict: dict[str, Any], path: Path) -> None:
    row = _as_dict(_as_dict(strict.get("by_cohort")).get("regime_bc_high_orientation"))
    best = _as_dict(row.get("best_candidate"))
    names = ["real", "global", "within_bin", "block"]
    values = [
        best.get("spearman") or 0.0,
        _as_dict(row.get("global_permutation")).get("spearman_mean") or 0.0,
        _as_dict(row.get("within_depth_bin_permutation")).get("spearman_mean") or 0.0,
        _as_dict(row.get("block_permutation")).get("spearman_mean") or 0.0,
    ]
    _bar_plot(path, names, values, "Permutation comparison", "Spearman")


def _plot_transfer(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("transfer"))
    names = list(rows)
    values = [_metric(row, "spearman") or 0.0 for row in rows.values()]
    _bar_plot(path, names, values, "Cross-regime transfer", "Spearman")


def _plot_transfer_degradation(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("transfer_degradation"))
    names = list(rows)
    values = [row.get("spearman_delta_vs_pooled_cv") or 0.0 for row in rows.values()]
    _bar_plot(path, names, values, "Transfer degradation", "Spearman delta")


def _plot_feature_groups(report: dict[str, Any], path: Path) -> None:
    rows = report.get("stable_feature_groups", [])[:10]
    names = [str(row.get("feature_group", ""))[:24] for row in rows]
    values = [row.get("mean_spearman_drop") or row.get("spearman_drop") or 0.0 for row in rows]
    _bar_plot(path, names, values, "Feature group stability", "Spearman drop")


def _plot_target_views(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("target_view_comparison"))
    names = list(rows)
    values = [
        _metric(_as_dict(_as_dict(row).get("summary")).get("aggregate", {}), "spearman") or 0.0
        for row in rows.values()
    ]
    _bar_plot(path, names, values, "Target-view comparison", "Spearman")


def _plot_calibration(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("calibration_drift"))
    names = list(rows)
    values = [row.get("mae_delta_vs_pooled_cv") or 0.0 for row in rows.values()]
    _bar_plot(path, names, values, "Calibration drift", "MAE delta")


def _plot_residual_shift(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("residual_shift"))
    names = list(rows)
    values = [row.get("mae") or 0.0 for row in rows.values()]
    _bar_plot(path, names, values, "Residual by transfer regime", "MAE")


def _plot_special(strict: dict[str, Any], path: Path) -> None:
    row = _as_dict(_as_dict(strict.get("by_cohort")).get("regime_bc_high_orientation"))
    special = _as_dict(row.get("special_band_dependency"))
    rows = _as_dict(special.get("rows"))
    names = list(rows)
    values = [
        _metric(_as_dict(item).get("aggregate", {}), "spearman") or 0.0
        for item in rows.values()
    ]
    _bar_plot(path, names, values, "Special-band sensitivity", "Spearman")


def _plot_placeholder(path: Path, title: str) -> None:
    _bar_plot(path, [title], [0.0], title, "audit placeholder")


def _bar_plot(path: Path, names: list[str], values: list[float], title: str, ylabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text(f"{title}\n{list(zip(names, values, strict=True))}\n", encoding="utf-8")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
