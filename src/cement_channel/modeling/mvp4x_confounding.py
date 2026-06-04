from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

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
    contiguous_depth_folds,
)
from cement_channel.modeling.mvp4x_model_analysis import build_feature_sets

CF_REPORT_VERSION = "mvp4x_cf_decision_v001"
COMMON_SUPPORT_VERSION = "mvp4x_cf_common_support_v001"
NUISANCE_AUDIT_VERSION = "mvp4x_cf_nuisance_audit_v001"
SPATIAL_VALIDATION_VERSION = "mvp4x_cf_spatial_validation_v001"
REVIEW_VERSION = "mvp4x_cf_review_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
DECISION_OPTIONS = {
    "exploratory_signal_survives_confounding_controls",
    "exploratory_regime_specific_signal_only",
    "request_label_or_physical_assumption_review",
    "exploratory_time_frequency_v2_helpful",
    "request_advanced_signal_processing_approval",
    "request_server_migration_approval",
    "stop_insufficient_signal",
    "stop_data_contract_issue",
    "stop_leakage_detected",
}
TARGET_VIEWS = ("receiver_mean", "receiver_p90", "receiver_max")
ALL_TARGET_VIEWS = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)
STRICT_GAPS_FT = (10.0, 25.0, 50.0)
PRIMARY_CANDIDATE = {
    "feature_set": "existing_features_only",
    "target": "receiver_mean",
    "model": "Ridge",
}


@dataclass(frozen=True)
class ConfoundingOutputs:
    common_support: dict[str, Any]
    nuisance_audit: dict[str, Any]
    spatial_validation: dict[str, Any]
    decision: dict[str, Any]
    iteration_log: str
    review_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfoundingResearchError(RuntimeError):
    """Raised when the MVP-4X confounding loop cannot run safely."""


def run_confounding_research_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    autonomous_decision_json: Path | str,
    autonomous_iteration_log_md: Path | str,
    config_path: Path | str,
    output_common_md: Path | str,
    output_common_json: Path | str,
    output_common_csv: Path | str,
    output_nuisance_md: Path | str,
    output_nuisance_json: Path | str,
    output_nuisance_csv: Path | str,
    output_spatial_md: Path | str,
    output_spatial_json: Path | str,
    output_spatial_csv: Path | str,
    output_decision_md: Path | str,
    output_decision_json: Path | str,
    output_review_dir: Path | str,
    output_iteration_log: Path | str,
    overwrite: bool = False,
) -> ConfoundingOutputs:
    config = _load_yaml(Path(config_path))
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    autonomous_decision = _read_json(Path(autonomous_decision_json))
    autonomous_log = Path(autonomous_iteration_log_md).read_text(encoding="utf-8")
    outputs = run_confounding_research(
        snapshot=snapshot,
        waveform=waveform,
        autonomous_decision=autonomous_decision,
        autonomous_iteration_log=autonomous_log,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "autonomous_decision_json": str(autonomous_decision_json),
            "autonomous_iteration_log_md": str(autonomous_iteration_log_md),
            "config_path": str(config_path),
        },
        review_dir=Path(output_review_dir),
        overwrite=overwrite,
    )
    write_confounding_outputs(
        outputs,
        output_common_md=Path(output_common_md),
        output_common_json=Path(output_common_json),
        output_common_csv=Path(output_common_csv),
        output_nuisance_md=Path(output_nuisance_md),
        output_nuisance_json=Path(output_nuisance_json),
        output_nuisance_csv=Path(output_nuisance_csv),
        output_spatial_md=Path(output_spatial_md),
        output_spatial_json=Path(output_spatial_json),
        output_spatial_csv=Path(output_spatial_csv),
        output_decision_md=Path(output_decision_md),
        output_decision_json=Path(output_decision_json),
        output_review_dir=Path(output_review_dir),
        output_iteration_log=Path(output_iteration_log),
        overwrite=overwrite,
    )
    return outputs


def run_confounding_research(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    autonomous_decision: dict[str, Any],
    autonomous_iteration_log: str,
    config: dict[str, Any],
    inputs: dict[str, str],
    review_dir: Path,
    overwrite: bool,
) -> ConfoundingOutputs:
    _validate_reusable_artifacts(
        snapshot=snapshot,
        waveform=waveform,
        autonomous_decision=autonomous_decision,
        autonomous_iteration_log=autonomous_iteration_log,
    )
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    baseline_config = _as_dict(config.get("baseline"))
    feature_sets = build_feature_sets(snapshot, waveform)
    masks = build_cf_masks(snapshot)
    common_support = build_common_support_audit(
        snapshot=snapshot,
        waveform=waveform,
        masks=masks,
        config=baseline_config,
    )
    nuisance_audit = run_nuisance_audit(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    spatial_validation = run_spatial_validation(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    stratified = run_stratified_exploratory_baseline(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
        spatial_validation=spatial_validation,
    )
    feature_review, review_files = run_cf_feature_review(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
        spatial_validation=spatial_validation,
        review_dir=review_dir,
        overwrite=overwrite,
    )
    decision = build_cf_decision(
        inputs=inputs,
        modeling_environment=modeling_environment.to_dict(),
        common_support=common_support,
        nuisance_audit=nuisance_audit,
        spatial_validation=spatial_validation,
        stratified=stratified,
        feature_review=feature_review,
        review_files=review_files,
    )
    review_files = write_cf_review_pack(
        review_dir=review_dir,
        snapshot=snapshot,
        masks=masks,
        common_support=common_support,
        nuisance_audit=nuisance_audit,
        spatial_validation=spatial_validation,
        stratified=stratified,
        feature_review=feature_review,
        decision=decision,
        overwrite=overwrite,
    )
    decision["review_files"] = review_files
    iteration_log = format_cf_iteration_log(
        inputs=inputs,
        common_support=common_support,
        nuisance_audit=nuisance_audit,
        spatial_validation=spatial_validation,
        decision=decision,
    )
    return ConfoundingOutputs(
        common_support=common_support,
        nuisance_audit=nuisance_audit,
        spatial_validation=spatial_validation,
        decision=decision,
        iteration_log=iteration_log,
        review_files=review_files,
    )


def build_cf_masks(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    all_samples = np.ones(depth.size, dtype=bool)
    low = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(-1)
    high = all_samples & ~low
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str).reshape(-1)
    saturation = np.asarray(snapshot["saturation_platform_flag"], dtype=bool).reshape(-1)
    special_5680 = np.asarray(snapshot["special_5680_flag"], dtype=bool).reshape(-1)
    any_special = np.asarray(snapshot["any_special_flag"], dtype=bool).reshape(-1)
    masks = {
        "all_samples": all_samples,
        "low_orientation": low,
        "high_orientation": high,
        "depth_transition_4000_4400": (depth >= 4000.0) & (depth <= 4400.0),
        "depth_transition_4100_4300": (depth >= 4100.0) & (depth <= 4300.0),
        "high_orientation_exclude_2400_2500": high & ~saturation,
        "high_orientation_exclude_5680": high & ~special_5680,
        "high_orientation_exclude_all_special": high & ~any_special,
    }
    for regime in ("A", "B", "C"):
        regime_mask = regimes == regime
        key = regime.lower()
        masks[f"regime_{key}"] = regime_mask
        masks[f"regime_{key}_low_orientation"] = regime_mask & low
        masks[f"regime_{key}_high_orientation"] = regime_mask & high
    return masks


def build_common_support_audit(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    orientation = np.asarray(snapshot["orientation_confidence"], dtype=np.float32).reshape(-1)
    groups = {}
    for name, mask in masks.items():
        valid = np.asarray(mask, dtype=bool).reshape(-1)
        groups[name] = {
            "sample_count": int(np.count_nonzero(valid)),
            "sample_fraction": float(np.count_nonzero(valid) / max(depth.size, 1)),
            "depth_range": _depth_range(depth, valid),
            "target_distributions": {
                target: _summary(np.asarray(snapshot[target], dtype=np.float32)[valid])
                for target in ALL_TARGET_VIEWS
            },
            "feature_distributions": {
                "existing_mean_abs": _summary(np.mean(np.abs(existing[valid]), axis=1)),
                "waveform_v1_mean_abs": _summary(np.mean(np.abs(wave[valid]), axis=1)),
            },
            "orientation_confidence_distribution": _summary(orientation[valid]),
            "regime_overlaps": _overlap_counts(valid, masks, ("regime_a", "regime_b", "regime_c")),
            "special_band_overlaps": {
                "saturation_platform_2400_2500": int(
                    np.count_nonzero(
                        valid & np.asarray(snapshot["saturation_platform_flag"], dtype=bool)
                    )
                ),
                "special_band_5680": int(
                    np.count_nonzero(valid & np.asarray(snapshot["special_5680_flag"], dtype=bool))
                ),
                "any_special_flag": int(
                    np.count_nonzero(valid & np.asarray(snapshot["any_special_flag"], dtype=bool))
                ),
            },
            "finite_ratio": {
                "existing_features": _finite_ratio(existing[valid]),
                "waveform_v1_features": _finite_ratio(wave[valid]),
            },
        }
    common_bins = _orientation_common_depth_bins(
        depth=depth,
        low_mask=masks["low_orientation"],
        high_mask=masks["high_orientation"],
        regime_ids=np.asarray(snapshot["broad_regime_id"]).astype(str),
        bin_width_ft=float(config.get("common_support_depth_bin_width_ft", 50.0)),
        min_samples_per_orientation=int(config.get("common_support_min_bin_samples", 20)),
    )
    support = {
        "regime_a_high_orientation_count": groups["regime_a_high_orientation"]["sample_count"],
        "regime_c_low_orientation_count": groups["regime_c_low_orientation"]["sample_count"],
        "regime_b_low_orientation_count": groups["regime_b_low_orientation"]["sample_count"],
        "regime_b_high_orientation_count": groups["regime_b_high_orientation"]["sample_count"],
        "common_support_depth_bins": common_bins,
    }
    regime_a_lacks_high = support["regime_a_high_orientation_count"] == 0
    regime_c_lacks_low = support["regime_c_low_orientation_count"] == 0
    regime_b_overlap = support["regime_b_low_orientation_count"] >= int(
        config.get("common_support_min_group_samples", 100)
    ) and support["regime_b_high_orientation_count"] >= int(
        config.get("common_support_min_group_samples", 100)
    )
    common_window_available = any(
        row["low_count"] >= 20 and row["high_count"] >= 20
        for row in common_bins["bins_with_both_orientation"]
    )
    not_identifiable = regime_a_lacks_high or regime_c_lacks_low or not common_window_available
    answers = {
        "1_regime_a_completely_lacks_high_orientation_support": regime_a_lacks_high,
        "2_regime_c_completely_lacks_low_orientation_support": regime_c_lacks_low,
        "3_regime_b_has_low_high_orientation_overlap": regime_b_overlap,
        "4_common_support_depth_window_orientation_comparison_available": common_window_available,
        "5_orientation_effect_statistically_not_identifiable": not_identifiable,
        "6_cohort_only_conclusions": [
            "full_well_low_vs_high_orientation_comparison",
            "all_sample_vs_high_orientation_metric_change",
            "high_orientation_deeper_interval_signal",
        ],
    }
    warnings = []
    if regime_a_lacks_high:
        warnings.append("regime_a_high_orientation_zero_support")
    if regime_c_lacks_low:
        warnings.append("regime_c_low_orientation_zero_support")
    if not common_window_available:
        warnings.append("no_sufficient_low_high_common_depth_bin_support")
    if not_identifiable:
        warnings.append("orientation_effect_not_identifiable_from_current_well")
    return {
        "report_version": COMMON_SUPPORT_VERSION,
        "generated_at": _utc_now(),
        "orientation_threshold_source": (
            "snapshot.low_orientation_confidence_flag from existing approved artifact; "
            "threshold not modified"
        ),
        "groups": groups,
        "common_support": support,
        "answers": answers,
        "identifiability_warning": warnings,
        "orientation_effect_not_identifiable_from_current_well": not_identifiable,
        **_method_flags(),
    }


def run_nuisance_audit(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 20240603))
    repeats = int(config.get("cf_matched_control_repeats", 30))
    target = PRIMARY_CANDIDATE["target"]
    model_name = PRIMARY_CANDIDATE["model"]
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
    high_mask = masks["high_orientation"] & np.isfinite(y)
    feature_matrices = _build_audit_feature_matrices(snapshot, feature_sets)
    matched_controls, composition_rows = build_depth_target_matched_controls(
        depth=depth,
        target=y,
        reference_mask=high_mask,
        candidate_mask=masks["all_samples"] & np.isfinite(y),
        repeat_count=repeats,
        seed=seed + 31,
        bin_width_ft=float(config.get("cf_matched_depth_bin_width_ft", 100.0)),
        target_quantile_count=int(config.get("cf_matched_target_quantile_count", 4)),
        high_orientation_mask=masks["high_orientation"],
        regimes=np.asarray(snapshot["broad_regime_id"]).astype(str),
    )
    X_primary = _finite_matrix(feature_sets[PRIMARY_CANDIDATE["feature_set"]]["matrix"])
    high_result = evaluate_model_cv(
        X=X_primary,
        y=y,
        depth=depth,
        sample_mask=high_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=seed + 41,
    )
    control_results = []
    for index, control_mask in enumerate(matched_controls):
        result = evaluate_model_cv(
            X=X_primary,
            y=y,
            depth=depth,
            sample_mask=control_mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=seed + 50 + index,
        )
        control_results.append(
            {
                "control_index": index,
                "summary": result["summary"],
                "composition": composition_rows[index],
            }
        )
    nuisance_rows: dict[str, Any] = {}
    for cohort_name in ("all_samples", "high_orientation", "regime_b_high_orientation"):
        cohort_mask = masks[cohort_name] & np.isfinite(y)
        nuisance_rows[cohort_name] = {}
        for matrix_name, matrix in feature_matrices.items():
            if matrix_name.startswith("xsi_plus") and cohort_name == "all_samples":
                production_candidate = False
            else:
                production_candidate = matrix_name.startswith("xsi_")
            result = evaluate_model_cv(
                X=matrix,
                y=y,
                depth=depth,
                sample_mask=cohort_mask,
                model_name=model_name,
                sklearn_modules=sklearn_modules,
                config=config,
                rng_seed=seed + len(nuisance_rows[cohort_name]),
            )
            nuisance_rows[cohort_name][matrix_name] = {
                "summary": result["summary"],
                "audit_only": not production_candidate or "nuisance" in matrix_name,
                "candidate_production_feature_set": False,
            }
    feature_proxy = run_feature_depth_proxy_audit(
        snapshot=snapshot,
        feature_sets=feature_sets,
    )
    prediction_depth = run_prediction_depth_audit(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=config,
        sklearn_modules=sklearn_modules,
    )
    permutation = run_target_permutations(
        X=X_primary,
        y=y,
        depth=depth,
        sample_mask=high_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=int(config.get("permutation_count", 20)),
        seed=seed + 600,
    )
    spearman_controls = [
        _metric(row["summary"]["aggregate"], "spearman") for row in control_results
    ]
    spearman_controls = [value for value in spearman_controls if value is not None]
    high_s = _metric(high_result["summary"]["aggregate"], "spearman")
    nuisance_comparison = _summarize_nuisance_comparison(nuisance_rows)
    return {
        "report_version": NUISANCE_AUDIT_VERSION,
        "generated_at": _utc_now(),
        "candidate": PRIMARY_CANDIDATE,
        "matched_controls": {
            "method": "fixed_depth_bins_with_target_quantile_strata",
            "repeat_count": repeats,
            "rows": control_results,
            "spearman_mean": None if not spearman_controls else float(np.mean(spearman_controls)),
            "spearman_std": None if not spearman_controls else float(np.std(spearman_controls)),
            "spearman_p95": None
            if not spearman_controls
            else float(np.quantile(spearman_controls, 0.95)),
            "high_minus_control_p95_spearman": None
            if high_s is None or not spearman_controls
            else float(high_s - np.quantile(spearman_controls, 0.95)),
        },
        "high_orientation_candidate": high_result["summary"],
        "target_permutation": permutation,
        "nuisance_baselines": nuisance_rows,
        "xsi_vs_nuisance": nuisance_comparison,
        "feature_depth_proxy_audit": feature_proxy,
        "prediction_depth_audit": prediction_depth,
        "warnings": _nuisance_warnings(control_results, nuisance_comparison),
        **_method_flags(),
        "nuisance_fields_audit_only": True,
    }


def build_depth_target_matched_controls(
    *,
    depth: np.ndarray,
    target: np.ndarray,
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    repeat_count: int,
    seed: int,
    bin_width_ft: float,
    target_quantile_count: int,
    high_orientation_mask: np.ndarray,
    regimes: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    values = np.asarray(depth, dtype=np.float32).reshape(-1)
    y = np.asarray(target, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference_mask, dtype=bool).reshape(-1)
    candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    if np.count_nonzero(reference) == 0:
        raise ConfoundingResearchError("Cannot match controls without reference samples.")
    edges = _fixed_depth_edges(values[reference], bin_width_ft)
    rng = np.random.default_rng(seed)
    masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for repeat in range(repeat_count):
        selected: list[int] = []
        replacement_bins = 0
        empty_bins = 0
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            ref_bin = reference & (values >= left) & (values < right) & np.isfinite(y)
            if not np.any(ref_bin):
                continue
            ref_values = y[ref_bin]
            quantiles = _safe_quantiles(ref_values, target_quantile_count)
            for q_left, q_right in zip(quantiles[:-1], quantiles[1:], strict=True):
                ref_stratum = ref_bin & (y >= q_left) & (y <= q_right)
                need = int(np.count_nonzero(ref_stratum))
                if need == 0:
                    continue
                cand = np.flatnonzero(
                    candidate
                    & (values >= left)
                    & (values < right)
                    & np.isfinite(y)
                    & (y >= q_left)
                    & (y <= q_right)
                )
                if cand.size == 0:
                    empty_bins += 1
                    cand = np.flatnonzero(
                        candidate & (values >= left) & (values < right) & np.isfinite(y)
                    )
                replace = cand.size < need
                if replace:
                    replacement_bins += 1
                if cand.size:
                    picks = rng.choice(cand, size=need, replace=replace)
                    selected.extend(np.asarray(picks, dtype=int).tolist())
        mask = np.zeros(values.size, dtype=bool)
        if selected:
            mask[np.asarray(selected, dtype=int)] = True
        masks.append(mask)
        rows.append(
            {
                "repeat": repeat,
                "sample_count": int(np.count_nonzero(mask)),
                "high_orientation_fraction": float(
                    np.count_nonzero(mask & high_orientation_mask) / max(np.count_nonzero(mask), 1)
                ),
                "regime_counts": {
                    regime: int(np.count_nonzero(mask & (regimes == regime)))
                    for regime in ("A", "B", "C")
                },
                "replacement_strata_count": replacement_bins,
                "empty_target_strata_fallback_count": empty_bins,
            }
        )
    return masks, rows


def run_spatial_validation(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 20240603))
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    rows = []
    for feature_set_name, feature_set in feature_sets.items():
        X = _finite_matrix(feature_set["matrix"])
        for target in TARGET_VIEWS:
            y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
            sample_mask = masks["high_orientation"] & np.isfinite(y)
            for model_name in MODEL_NAMES:
                result = evaluate_model_cv(
                    X=X,
                    y=y,
                    depth=depth,
                    sample_mask=sample_mask,
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    config=config,
                    rng_seed=seed + len(rows),
                )
                rows.append(
                    {
                        "feature_set": feature_set_name,
                        "target": target,
                        "model": model_name,
                        "cohort": "high_orientation",
                        "summary": result["summary"],
                    }
                )
    best = _select_best_spatial_candidate(rows)
    if best is None:
        return {
            "report_version": SPATIAL_VALIDATION_VERSION,
            "generated_at": _utc_now(),
            "contiguous_cv_rows": rows,
            "best_candidate": None,
            "decision_support": {"signal_survives_strict_controls": False},
            **_method_flags(),
        }
    X_best = _finite_matrix(feature_sets[best["feature_set"]]["matrix"])
    y_best = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    sample_mask = masks["high_orientation"] & np.isfinite(y_best)
    strict = run_strict_protocols(
        X=X_best,
        y=y_best,
        depth=depth,
        sample_mask=sample_mask,
        model_name=best["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        seed=seed + 700,
    )
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    transfer = {
        "B_to_C": fit_eval_once(
            X=X_best,
            y=y_best,
            train_mask=sample_mask & (regimes == "B"),
            validation_mask=sample_mask & (regimes == "C"),
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=seed + 801,
        ),
        "C_to_B": fit_eval_once(
            X=X_best,
            y=y_best,
            train_mask=sample_mask & (regimes == "C"),
            validation_mask=sample_mask & (regimes == "B"),
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=seed + 802,
        ),
    }
    leave_one = {
        f"train_not_{regime}_validate_{regime}": fit_eval_once(
            X=X_best,
            y=y_best,
            train_mask=sample_mask & (regimes != regime),
            validation_mask=sample_mask & (regimes == regime),
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=seed + 820 + index,
        )
        for index, regime in enumerate(("A", "B", "C"))
    }
    special = run_exclusion_sensitivity(
        X=X_best,
        y=y_best,
        depth=depth,
        masks=masks,
        model_name=best["model"],
        sklearn_modules=sklearn_modules,
        config=config,
    )
    regime_b_common = run_regime_b_common_support_evaluation(
        X=X_best,
        y=y_best,
        depth=depth,
        masks=masks,
        model_name=best["model"],
        sklearn_modules=sklearn_modules,
        config=config,
    )
    target_comparison = run_target_comparison(
        snapshot=snapshot,
        X=X_best,
        depth=depth,
        sample_mask=sample_mask,
        model_name=best["model"],
        sklearn_modules=sklearn_modules,
        config=config,
    )
    support = _spatial_decision_support(best, strict, transfer, special, leave_one)
    return {
        "report_version": SPATIAL_VALIDATION_VERSION,
        "generated_at": _utc_now(),
        "protocol_scope": "research_only_supplementary_not_formal_cv_change",
        "contiguous_cv_rows": rows,
        "best_candidate": best,
        "blocked_gap_cv": strict["blocked_gap_cv"],
        "permutation": strict["permutation"],
        "leave_one_regime_out": leave_one,
        "cross_regime_transfer": transfer,
        "regime_b_common_support_evaluation": regime_b_common,
        "exclusion_sensitivity": special,
        "target_view_comparison": target_comparison,
        "decision_support": support,
        "spatial_leakage_warnings": _spatial_leakage_warnings(strict, transfer),
        **_method_flags(),
    }


def run_strict_protocols(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    real = evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=seed,
    )
    blocked = {
        f"gap_{gap:g}_ft": evaluate_blocked_gap_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            gap_ft=gap,
            rng_seed=seed + int(gap),
        )
        for gap in STRICT_GAPS_FT
    }
    count = int(config.get("permutation_count", 20))
    global_perm = run_target_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=count,
        seed=seed + 101,
    )
    within_bin = run_structured_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=count,
        seed=seed + 202,
        method="within_depth_bin",
    )
    block_perm = run_structured_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=count,
        seed=seed + 303,
        method="block",
    )
    real_s = _metric(real["summary"]["aggregate"], "spearman")
    return {
        "contiguous": real["summary"],
        "blocked_gap_cv": blocked,
        "permutation": {
            "global": global_perm,
            "within_depth_bin": within_bin,
            "block": block_perm,
            "real_spearman": real_s,
            "global_margin": _subtract(real_s, global_perm.get("spearman_mean")),
            "within_depth_bin_margin": _subtract(real_s, within_bin.get("spearman_mean")),
            "block_margin": _subtract(real_s, block_perm.get("spearman_mean")),
        },
    }


def evaluate_blocked_gap_cv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    gap_ft: float,
    rng_seed: int,
) -> dict[str, Any]:
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1) & np.all(np.isfinite(X), axis=1)
    folds = contiguous_depth_folds(depth, mask, n_folds=int(config.get("n_contiguous_folds", 3)))
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    fold_rows = []
    for fold_index, validation_mask in enumerate(folds):
        validation = validation_mask & mask
        if not np.any(validation):
            continue
        v_min = float(np.min(depth[validation]))
        v_max = float(np.max(depth[validation]))
        train = mask & ~validation & ((depth < v_min - gap_ft) | (depth > v_max + gap_ft))
        if np.count_nonzero(train) == 0:
            fold_rows.append(
                {
                    "fold": fold_index,
                    "status": "skipped_empty_train_after_gap",
                    "validation_count": int(np.count_nonzero(validation)),
                }
            )
            continue
        model = _make_model_from_existing(
            model_name,
            sklearn_modules,
            config,
            random_state=rng_seed + fold_index,
        )
        model.fit(X[train], y[train])
        pred = np.asarray(model.predict(X[validation]), dtype=np.float32)
        oof[validation] = pred
        fold_rows.append(
            {
                "fold": fold_index,
                "status": "completed",
                "train_count": int(np.count_nonzero(train)),
                "validation_count": int(np.count_nonzero(validation)),
                "gap_ft": gap_ft,
                "validation_depth_min": v_min,
                "validation_depth_max": v_max,
                **compute_regression_metrics(y[validation], pred),
            }
        )
    aggregate = compute_regression_metrics(y[mask], oof[mask])
    return {
        "status": "completed",
        "gap_ft": gap_ft,
        "sample_count": int(np.count_nonzero(mask)),
        "fold_count": int(sum(row.get("status") == "completed" for row in fold_rows)),
        "stable_positive_spearman_folds": int(
            sum(
                row.get("spearman") is not None and float(row["spearman"]) > 0.0
                for row in fold_rows
            )
        ),
        "aggregate": aggregate,
        "folds": fold_rows,
    }


def run_structured_permutations(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    permutation_count: int,
    seed: int,
    method: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    selected = np.flatnonzero(sample_mask)
    metrics = []
    for index in range(permutation_count):
        y_perm = y.copy()
        if method == "within_depth_bin":
            y_perm[selected] = _permute_within_depth_bins(
                values=y[selected],
                depth=depth[selected],
                rng=rng,
                bin_width_ft=float(config.get("cf_permutation_depth_bin_width_ft", 100.0)),
            )
        elif method == "block":
            y_perm[selected] = _permute_contiguous_blocks(y[selected], depth[selected], rng=rng)
        else:
            raise ValueError(f"Unknown structured permutation method: {method}")
        result = evaluate_model_cv(
            X=X,
            y=y_perm,
            depth=depth,
            sample_mask=sample_mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=seed + index,
        )
        row = dict(result["summary"]["aggregate"])
        row["permutation_index"] = index
        metrics.append(row)
    spearman_values = [row["spearman"] for row in metrics if row.get("spearman") is not None]
    return {
        "method": method,
        "permutation_count": permutation_count,
        "spearman_mean": None if not spearman_values else float(np.mean(spearman_values)),
        "spearman_std": None if not spearman_values else float(np.std(spearman_values)),
        "spearman_p95": None if not spearman_values else float(np.quantile(spearman_values, 0.95)),
        "spearman_max": None if not spearman_values else float(np.max(spearman_values)),
        "metrics": metrics,
    }


def run_stratified_exploratory_baseline(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    spatial_validation: dict[str, Any],
) -> dict[str, Any]:
    best = _as_dict(spatial_validation.get("best_candidate"))
    if not best:
        return {"status": "skipped_no_spatial_best", **_method_flags()}
    X = _finite_matrix(feature_sets[best["feature_set"]]["matrix"])
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    rows: dict[str, Any] = {}
    for regime in ("B", "C"):
        for target in ("receiver_mean", "receiver_p90"):
            y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
            mask = masks[f"regime_{regime.lower()}_high_orientation"] & np.isfinite(y)
            rows[f"regime_{regime}_{target}"] = evaluate_model_cv(
                X=X,
                y=y,
                depth=depth,
                sample_mask=mask,
                model_name=best["model"],
                sklearn_modules=sklearn_modules,
                config=config,
                rng_seed=2300 + len(rows),
            )["summary"]
    return {
        "status": "completed",
        "regime_specific_exploratory_only": True,
        "not_formal_regime_split": True,
        "regime_a_high_orientation_support": int(
            np.count_nonzero(masks["regime_a_high_orientation"])
        ),
        "best_basis": best,
        "within_regime_high_orientation": rows,
        "cross_regime_transfer": spatial_validation.get("cross_regime_transfer"),
        **_method_flags(),
    }


def run_cf_feature_review(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    spatial_validation: dict[str, Any],
    review_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    best = _as_dict(spatial_validation.get("best_candidate"))
    if not best:
        return {"status": "skipped_no_spatial_best", **_method_flags()}, {}
    feature_set = feature_sets[best["feature_set"]]
    X = _finite_matrix(feature_set["matrix"])
    y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    mask = masks["high_orientation"] & np.isfinite(y)
    oof, folds, importance_rows = fit_oof_with_permutation_importance(
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
    importance = summarize_importance(importance_rows)
    group_importance = summarize_group_importance(importance_rows)
    summary = {
        "status": "completed",
        "method": "validation_fold_feature_shuffle_high_orientation",
        "best_basis": best,
        "folds": folds,
        "aggregate": compute_regression_metrics(y[mask], oof[mask]),
        "top_30_features": importance["top_30_features"],
        "top_10_stable_features": importance["top_10_stable_features"],
        "top_feature_groups": group_importance[:15],
        "oof_prediction_summary": _summary(oof[mask]),
        **_method_flags(),
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / "feature_review_summary.json"
    _ensure_can_write(path, overwrite=overwrite)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary, {"feature_review_summary_json": str(path)}


def build_cf_decision(
    *,
    inputs: dict[str, str],
    modeling_environment: dict[str, Any],
    common_support: dict[str, Any],
    nuisance_audit: dict[str, Any],
    spatial_validation: dict[str, Any],
    stratified: dict[str, Any],
    feature_review: dict[str, Any],
    review_files: dict[str, str],
) -> dict[str, Any]:
    best = _as_dict(spatial_validation.get("best_candidate"))
    support = _as_dict(spatial_validation.get("decision_support"))
    not_identifiable = bool(
        common_support.get("orientation_effect_not_identifiable_from_current_well")
    )
    nuisance_by_cohort = _as_dict(nuisance_audit.get("xsi_vs_nuisance"))
    nuisance = _as_dict(nuisance_by_cohort.get("high_orientation"))
    strict_survives = bool(support.get("signal_survives_strict_controls"))
    b_to_c = _metric(
        _as_dict(spatial_validation.get("cross_regime_transfer")).get("B_to_C", {}),
        "spearman",
    )
    c_to_b = _metric(
        _as_dict(spatial_validation.get("cross_regime_transfer")).get("C_to_B", {}),
        "spearman",
    )
    if not best or not strict_survives:
        decision = "request_label_or_physical_assumption_review"
    elif not_identifiable:
        decision = "exploratory_regime_specific_signal_only"
    elif bool(nuisance.get("xsi_increment_over_best_nuisance_positive")):
        decision = "exploratory_signal_survives_confounding_controls"
    else:
        decision = "request_label_or_physical_assumption_review"
    if decision not in DECISION_OPTIONS:
        raise ConfoundingResearchError(f"Unsupported CF decision: {decision}")
    answers = {
        "1_orientation_effect_identifiable": not not_identifiable,
        "2_high_orientation_improvement_mainly_cohort_effect": not_identifiable,
        "3_xsi_only_exceeds_nuisance_only": nuisance.get("xsi_only_exceeds_best_nuisance"),
        "4_xsi_increment_after_depth_control": nuisance.get(
            "xsi_increment_over_best_nuisance_positive"
        ),
        "5_blocked_gap_cv_still_stable": support.get("blocked_gap_stable"),
        "6_within_depth_bin_permutation_margin_positive": support.get(
            "within_depth_bin_margin_positive"
        ),
        "7_b_to_c_transfer": b_to_c,
        "8_c_to_b_transfer": c_to_b,
        "9_most_stable_target_view": _best_target_from_spatial(spatial_validation),
        "10_most_stable_feature_set": best.get("feature_set"),
        "11_most_stable_model": best.get("model"),
        "12_time_frequency_v2_triggered": False,
        "13_time_frequency_v2_improved": "not_triggered",
        "14_depends_on_special_interval": support.get("special_band_dependency"),
        "15_spatial_leakage_risk": spatial_validation.get("spatial_leakage_warnings"),
        "16_worth_advanced_signal_processing": ("not_without_human_review_of_regime_confounding"),
        "17_local_resources_sufficient": True,
        "18_recommend_server": False,
        "19_next_human_approval": _next_human_approval(decision),
        "20_production_claims_and_final_labels_forbidden": True,
    }
    return {
        "decision_version": CF_REPORT_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "inputs": inputs,
        "modeling_environment": modeling_environment,
        "artifact_reuse": {
            "snapshot_reused": True,
            "waveform_v1_reused": True,
            "time_frequency_v2_triggered": False,
            "raw_waveform_reread": False,
        },
        "common_support_summary": _compact_common_support(common_support),
        "nuisance_summary": {
            "xsi_vs_nuisance": nuisance_by_cohort,
            "xsi_vs_nuisance_high_orientation": nuisance,
            "matched_controls": _compact_matched(nuisance_audit),
        },
        "spatial_validation_summary": _compact_spatial(spatial_validation),
        "stratified_exploratory_baseline": stratified,
        "feature_review": {
            "top_10_stable_features": feature_review.get("top_10_stable_features", []),
            "top_feature_groups": feature_review.get("top_feature_groups", []),
        },
        "review_files": review_files,
        "answers": answers,
        "next_minimal_recommendation": _next_human_approval(decision),
        "not_authorized": [
            "production claim",
            "final labels",
            "ground-truth claim",
            "formal orientation filter",
            "formal regime split",
            "formal CV protocol change",
            "STC",
            "APES",
            "deep learning",
        ],
        **_method_flags(),
    }


def write_confounding_outputs(
    outputs: ConfoundingOutputs,
    *,
    output_common_md: Path,
    output_common_json: Path,
    output_common_csv: Path,
    output_nuisance_md: Path,
    output_nuisance_json: Path,
    output_nuisance_csv: Path,
    output_spatial_md: Path,
    output_spatial_json: Path,
    output_spatial_csv: Path,
    output_decision_md: Path,
    output_decision_json: Path,
    output_review_dir: Path,
    output_iteration_log: Path,
    overwrite: bool,
) -> None:
    paths = (
        output_common_md,
        output_common_json,
        output_common_csv,
        output_nuisance_md,
        output_nuisance_json,
        output_nuisance_csv,
        output_spatial_md,
        output_spatial_json,
        output_spatial_csv,
        output_decision_md,
        output_decision_json,
        output_iteration_log,
    )
    for path in paths:
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_review_dir.mkdir(parents=True, exist_ok=True)
    output_common_json.write_text(_json(outputs.common_support), encoding="utf-8")
    output_common_md.write_text(format_common_markdown(outputs.common_support), encoding="utf-8")
    write_common_csv(outputs.common_support, output_common_csv)
    output_nuisance_json.write_text(_json(outputs.nuisance_audit), encoding="utf-8")
    output_nuisance_md.write_text(
        format_nuisance_markdown(outputs.nuisance_audit), encoding="utf-8"
    )
    write_nuisance_csv(outputs.nuisance_audit, output_nuisance_csv)
    output_spatial_json.write_text(_json(outputs.spatial_validation), encoding="utf-8")
    output_spatial_md.write_text(
        format_spatial_markdown(outputs.spatial_validation), encoding="utf-8"
    )
    write_spatial_csv(outputs.spatial_validation, output_spatial_csv)
    output_decision_json.write_text(_json(outputs.decision), encoding="utf-8")
    output_decision_md.write_text(format_decision_markdown(outputs.decision), encoding="utf-8")
    output_iteration_log.write_text(outputs.iteration_log, encoding="utf-8")


# Helper and formatting functions


def _validate_reusable_artifacts(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    autonomous_decision: dict[str, Any],
    autonomous_iteration_log: str,
) -> None:
    missing_targets = [target for target in ALL_TARGET_VIEWS if target not in snapshot]
    if missing_targets:
        raise ConfoundingResearchError(f"Missing target views: {missing_targets}")
    if snapshot["depth"].shape[0] != 7108:
        raise ConfoundingResearchError(f"Unexpected sample count: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise ConfoundingResearchError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise ConfoundingResearchError(
            f"Unexpected waveform-v1 feature shape: {waveform['waveform_depth_features'].shape}"
        )
    if snapshot["xsi_features"].shape[1] + waveform["waveform_depth_features"].shape[1] != 422:
        raise ConfoundingResearchError("Combined feature count is not 422.")
    if _finite_ratio(snapshot["xsi_features"]) != 1.0:
        raise ConfoundingResearchError("Existing feature finite ratio is not 1.0.")
    if _finite_ratio(waveform["waveform_depth_features"]) != 1.0:
        raise ConfoundingResearchError("Waveform-v1 finite ratio is not 1.0.")
    if str(snapshot["target_kernel"]) != "triangular_midpoint_weighted":
        raise ConfoundingResearchError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for container_name, container in (
        ("snapshot", snapshot),
        ("waveform", waveform),
        ("autonomous_decision", autonomous_decision),
    ):
        for flag in RESEARCH_FLAGS:
            if not bool(np.asarray(container.get(flag, False)).item()):
                raise ConfoundingResearchError(f"{container_name} missing flag {flag}.")
    if "no_final_labels" not in autonomous_iteration_log:
        raise ConfoundingResearchError("Autonomous iteration log missing research-only scope text.")


def _build_audit_feature_matrices(
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    orientation = np.asarray(snapshot["orientation_confidence"], dtype=np.float32).reshape(-1)
    inclination = np.asarray(snapshot.get("inclination_deg", np.full(depth.shape, np.nan)))
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    regime_matrix = np.column_stack(
        [(regimes == regime).astype(float) for regime in ("A", "B", "C")]
    )
    depth_matrix = _zscore_matrix(depth.reshape(-1, 1))
    orientation_matrix = _zscore_matrix(
        np.nan_to_num(orientation, nan=np.nanmedian(orientation)).reshape(-1, 1)
    )
    inclination_matrix = _zscore_matrix(
        np.nan_to_num(inclination, nan=np.nanmedian(inclination)).reshape(-1, 1)
    )
    nuisance = {
        "nuisance_depth_only": depth_matrix,
        "nuisance_orientation_confidence_only": orientation_matrix,
        "nuisance_inclination_only": inclination_matrix,
        "nuisance_regime_id_only": regime_matrix.astype(np.float32),
        "nuisance_depth_plus_orientation": np.column_stack([depth_matrix, orientation_matrix]),
        "nuisance_depth_plus_regime": np.column_stack([depth_matrix, regime_matrix]),
    }
    existing = _finite_matrix(feature_sets["existing_features_only"]["matrix"])
    wave = _finite_matrix(feature_sets["waveform_features_only"]["matrix"])
    combined = _finite_matrix(feature_sets["combined_features"]["matrix"])
    return {
        "xsi_existing_features_only": existing,
        "xsi_waveform_features_only": wave,
        "xsi_combined_features": combined,
        **nuisance,
        "xsi_plus_nuisance_existing_depth_orientation_audit_only": np.column_stack(
            [existing, depth_matrix, orientation_matrix]
        ).astype(np.float32),
        "xsi_plus_nuisance_combined_depth_regime_audit_only": np.column_stack(
            [combined, depth_matrix, regime_matrix]
        ).astype(np.float32),
    }


def run_feature_depth_proxy_audit(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    feature_set = feature_sets["combined_features"]
    X = _finite_matrix(feature_set["matrix"])
    names = np.asarray(feature_set["names"]).astype(str)
    groups = np.asarray(feature_set["groups"]).astype(str)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    orientation = np.asarray(snapshot["orientation_confidence"], dtype=np.float32).reshape(-1)
    regimes = np.asarray(snapshot["broad_regime_code"], dtype=np.float32).reshape(-1)
    target = np.asarray(snapshot["receiver_mean"], dtype=np.float32).reshape(-1)
    rows = []
    for index in range(X.shape[1]):
        values = X[:, index]
        depth_corr = _safe_spearman(values, depth)
        orientation_corr = _safe_spearman(values, orientation)
        regime_corr = _safe_spearman(values, regimes)
        residual_corr = _partial_depth_residual_spearman(values, target, depth)
        rows.append(
            {
                "feature_index": index,
                "feature_name": str(names[index]),
                "feature_group": str(groups[index]),
                "abs_spearman_depth": _abs_or_none(depth_corr),
                "spearman_depth": depth_corr,
                "abs_spearman_orientation_confidence": _abs_or_none(orientation_corr),
                "spearman_orientation_confidence": orientation_corr,
                "abs_spearman_regime_code": _abs_or_none(regime_corr),
                "spearman_regime_code": regime_corr,
                "depth_residual_target_spearman": residual_corr,
            }
        )
    return {
        "top_abs_correlation_with_depth": _top_by(rows, "abs_spearman_depth", 30),
        "top_abs_correlation_with_orientation_confidence": _top_by(
            rows, "abs_spearman_orientation_confidence", 30
        ),
        "top_abs_correlation_with_regime": _top_by(rows, "abs_spearman_regime_code", 30),
        "feature_groups_most_likely_encoding_depth_proxy": _group_proxy_summary(rows),
        "stable_features_after_controlling_depth_audit_only": _top_by(
            rows, "depth_residual_target_spearman", 30
        ),
    }


def run_prediction_depth_audit(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    X = _finite_matrix(feature_sets[PRIMARY_CANDIDATE["feature_set"]]["matrix"])
    y = np.asarray(snapshot[PRIMARY_CANDIDATE["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    orientation = np.asarray(snapshot["orientation_confidence"], dtype=np.float32).reshape(-1)
    mask = masks["high_orientation"] & np.isfinite(y)
    result = evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=mask,
        model_name=PRIMARY_CANDIDATE["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=3400,
    )
    pred = np.asarray(result["oof_prediction"], dtype=np.float32)
    residual = pred - y
    return {
        "basis": PRIMARY_CANDIDATE,
        "metrics": result["summary"]["aggregate"],
        "predicted_vs_depth_spearman": _safe_spearman(pred[mask], depth[mask]),
        "residual_vs_depth_spearman": _safe_spearman(residual[mask], depth[mask]),
        "target_vs_depth_spearman": _safe_spearman(y[mask], depth[mask]),
        "predicted_vs_orientation_spearman": _safe_spearman(pred[mask], orientation[mask]),
        "residual_vs_orientation_spearman": _safe_spearman(residual[mask], orientation[mask]),
    }


def run_exclusion_sensitivity(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    filters = (
        "high_orientation",
        "high_orientation_exclude_2400_2500",
        "high_orientation_exclude_5680",
        "high_orientation_exclude_all_special",
    )
    rows = {}
    for index, name in enumerate(filters):
        mask = masks[name] & np.isfinite(y)
        rows[name] = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=4100 + index,
        )["summary"]
    base = _metric(rows["high_orientation"]["aggregate"], "spearman")
    for _name, row in rows.items():
        row["spearman_delta_vs_high_orientation"] = _subtract(
            _metric(row["aggregate"], "spearman"), base
        )
    return rows


def run_regime_b_common_support_evaluation(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for name in ("regime_b_low_orientation", "regime_b_high_orientation"):
        mask = masks[name] & np.isfinite(y)
        if np.count_nonzero(mask) < int(config.get("n_contiguous_folds", 3)):
            output[name] = {"status": "skipped_too_few_samples", "sample_count": int(mask.sum())}
        else:
            output[name] = evaluate_model_cv(
                X=X,
                y=y,
                depth=depth,
                sample_mask=mask,
                model_name=model_name,
                sklearn_modules=sklearn_modules,
                config=config,
                rng_seed=4300 + len(output),
            )["summary"]
    return output


def run_target_comparison(
    *,
    snapshot: dict[str, np.ndarray],
    X: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = {}
    for target in TARGET_VIEWS:
        y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
        rows[target] = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask & np.isfinite(y),
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=4400 + len(rows),
        )["summary"]
    return rows


def _spatial_decision_support(
    best: dict[str, Any],
    strict: dict[str, Any],
    transfer: dict[str, Any],
    special: dict[str, Any],
    leave_one: dict[str, Any],
) -> dict[str, Any]:
    real_s = _metric(best, "spearman")
    perm = _as_dict(strict.get("permutation"))
    blocked = _as_dict(strict.get("blocked_gap_cv"))
    blocked_s = [
        _metric(_as_dict(row).get("aggregate", {}), "spearman") for row in blocked.values()
    ]
    blocked_positive = [value for value in blocked_s if value is not None and value > 0.0]
    within_margin = perm.get("within_depth_bin_margin")
    block_margin = perm.get("block_margin")
    global_margin = perm.get("global_margin")
    transfer_values = [_metric(row, "spearman") for row in transfer.values()]
    special_delta = [
        abs(float(row.get("spearman_delta_vs_high_orientation") or 0.0))
        for row in special.values()
        if _as_dict(row).get("aggregate") is not None
    ]
    return {
        "real_spearman": real_s,
        "global_permutation_margin_positive": global_margin is not None
        and float(global_margin) > 0.05,
        "within_depth_bin_margin_positive": within_margin is not None
        and float(within_margin) > 0.05,
        "block_permutation_margin_positive": block_margin is not None
        and float(block_margin) > 0.05,
        "blocked_gap_stable": len(blocked_positive) >= 2,
        "fold_sign_consistency": best.get("stable_positive_spearman_folds"),
        "folds_above_permutation": best.get("stable_positive_spearman_folds"),
        "cross_regime_transfer_positive_count": int(
            sum(value is not None and value > 0.0 for value in transfer_values)
        ),
        "leave_one_regime_out": leave_one,
        "special_band_dependency": bool(special_delta and max(special_delta) > 0.05),
        "signal_survives_strict_controls": bool(
            real_s is not None
            and real_s > 0.0
            and global_margin is not None
            and float(global_margin) > 0.05
            and within_margin is not None
            and float(within_margin) > 0.05
            and block_margin is not None
            and float(block_margin) > 0.05
            and len(blocked_positive) >= 2
        ),
    }


def _select_best_spatial_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    for row in rows:
        if row["model"] == "DummyRegressor":
            continue
        aggregate = _as_dict(row["summary"].get("aggregate"))
        spearman = aggregate.get("spearman")
        if spearman is None:
            continue
        candidate = {
            "feature_set": row["feature_set"],
            "target": row["target"],
            "model": row["model"],
            "cohort": row["cohort"],
            "sample_count": row["summary"].get("sample_count"),
            "spearman": float(spearman),
            "mae": aggregate.get("mae"),
            "r2": aggregate.get("r2"),
            "pearson": aggregate.get("pearson"),
            "stable_positive_spearman_folds": row["summary"].get("stable_positive_spearman_folds"),
            "folds": row["summary"].get("folds"),
        }
        if best is None or _candidate_score(candidate) > _candidate_score(best):
            best = candidate
    return best


def _candidate_score(row: dict[str, Any]) -> tuple[float, int, float, float]:
    spearman = float(row.get("spearman") if row.get("spearman") is not None else -1e9)
    stable = int(row.get("stable_positive_spearman_folds") or 0)
    r2 = float(row.get("r2") if row.get("r2") is not None else -1e9)
    mae = float(row.get("mae") if row.get("mae") is not None else 1e9)
    target_bonus = 0.05 if row.get("target") == "receiver_mean" else 0.0
    return (spearman + target_bonus, stable, r2, -mae)


def _make_model_from_existing(
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    *,
    random_state: int,
) -> Any:
    from cement_channel.modeling.mvp4x_baselines import _make_model

    return _make_model(model_name, sklearn_modules, config, random_state=random_state)


def _permute_within_depth_bins(
    *,
    values: np.ndarray,
    depth: np.ndarray,
    rng: np.random.Generator,
    bin_width_ft: float,
) -> np.ndarray:
    output = values.copy()
    edges = _fixed_depth_edges(depth, bin_width_ft)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (depth >= left) & (depth < right)
        if np.count_nonzero(mask) > 1:
            output[mask] = rng.permutation(output[mask])
    return output


def _permute_contiguous_blocks(
    values: np.ndarray,
    depth: np.ndarray,
    *,
    rng: np.random.Generator,
    block_count: int = 12,
) -> np.ndarray:
    order = np.argsort(depth)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    ordered_values = values[order]
    chunks = np.array_split(ordered_values, min(block_count, max(1, ordered_values.size)))
    permuted = rng.permutation(len(chunks))
    output_ordered = np.concatenate([chunks[index] for index in permuted])
    return output_ordered[inverse]


def _fixed_depth_edges(values: np.ndarray, bin_width_ft: float) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ConfoundingResearchError("Cannot build depth bins without finite depths.")
    start = np.floor(float(np.min(finite)) / bin_width_ft) * bin_width_ft
    stop = np.ceil(float(np.max(finite)) / bin_width_ft) * bin_width_ft + bin_width_ft
    return np.arange(start, stop + 0.5 * bin_width_ft, bin_width_ft, dtype=np.float32)


def _safe_quantiles(values: np.ndarray, count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    qs = np.quantile(finite, np.linspace(0.0, 1.0, max(2, count + 1)))
    qs[0] -= 1e-8
    qs[-1] += 1e-8
    return np.asarray(qs, dtype=np.float32)


def _orientation_common_depth_bins(
    *,
    depth: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    regime_ids: np.ndarray,
    bin_width_ft: float,
    min_samples_per_orientation: int,
) -> dict[str, Any]:
    edges = _fixed_depth_edges(depth, bin_width_ft)
    rows = []
    both = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        bin_mask = (depth >= left) & (depth < right)
        low_count = int(np.count_nonzero(bin_mask & low_mask))
        high_count = int(np.count_nonzero(bin_mask & high_mask))
        regime_counts = {
            regime: int(np.count_nonzero(bin_mask & (regime_ids == regime)))
            for regime in ("A", "B", "C")
        }
        row = {
            "depth_min": float(left),
            "depth_max": float(right),
            "low_count": low_count,
            "high_count": high_count,
            "regime_counts": regime_counts,
        }
        rows.append(row)
        if low_count >= min_samples_per_orientation and high_count >= min_samples_per_orientation:
            both.append(row)
    return {
        "bin_width_ft": bin_width_ft,
        "min_samples_per_orientation": min_samples_per_orientation,
        "bins": rows,
        "bins_with_both_orientation": both,
        "bin_count_with_both_orientation": len(both),
    }


def _summarize_nuisance_comparison(rows: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for cohort, matrices in rows.items():
        xsi = {
            name: _metric(_as_dict(row["summary"]).get("aggregate", {}), "spearman")
            for name, row in matrices.items()
            if name.startswith("xsi_") and "plus_nuisance" not in name
        }
        nuisance = {
            name: _metric(_as_dict(row["summary"]).get("aggregate", {}), "spearman")
            for name, row in matrices.items()
            if name.startswith("nuisance_")
        }
        xsi_plus = {
            name: _metric(_as_dict(row["summary"]).get("aggregate", {}), "spearman")
            for name, row in matrices.items()
            if name.startswith("xsi_plus")
        }
        best_xsi = _best_metric_item(xsi)
        best_nuisance = _best_metric_item(nuisance)
        best_plus = _best_metric_item(xsi_plus)
        output[cohort] = {
            "best_xsi_only": best_xsi,
            "best_nuisance_only": best_nuisance,
            "best_xsi_plus_nuisance_audit_only": best_plus,
            "xsi_only_exceeds_best_nuisance": _metric_greater(best_xsi, best_nuisance),
            "xsi_increment_over_best_nuisance_spearman": _subtract(
                _item_value(best_xsi), _item_value(best_nuisance)
            ),
            "xsi_increment_over_best_nuisance_positive": bool(
                (_subtract(_item_value(best_xsi), _item_value(best_nuisance)) or -1.0) > 0.0
            ),
        }
    return output


def _best_metric_item(values: dict[str, float | None]) -> dict[str, Any] | None:
    finite = [(name, value) for name, value in values.items() if value is not None]
    if not finite:
        return None
    name, value = max(finite, key=lambda item: float(item[1]))
    return {"name": name, "spearman": float(value)}


def _metric_greater(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool | None:
    if left is None or right is None:
        return None
    return bool(float(left["spearman"]) > float(right["spearman"]))


def _item_value(item: dict[str, Any] | None) -> float | None:
    if item is None or item.get("spearman") is None:
        return None
    return float(item["spearman"])


def _nuisance_warnings(
    control_results: list[dict[str, Any]],
    nuisance_comparison: dict[str, Any],
) -> list[str]:
    warnings = []
    high_fractions = [row["composition"]["high_orientation_fraction"] for row in control_results]
    if high_fractions and float(np.mean(high_fractions)) > 0.80:
        warnings.append("matched_controls_dominated_by_high_orientation_samples")
    high = _as_dict(nuisance_comparison.get("high_orientation"))
    if not high.get("xsi_only_exceeds_best_nuisance"):
        warnings.append("xsi_not_above_best_nuisance_in_high_orientation")
    return warnings


def _spatial_leakage_warnings(
    strict: dict[str, Any],
    transfer: dict[str, Any],
) -> list[str]:
    warnings = []
    perm = _as_dict(strict.get("permutation"))
    if (perm.get("within_depth_bin_margin") or 0.0) <= 0.05:
        warnings.append("within_depth_bin_permutation_margin_not_strong")
    if (perm.get("block_margin") or 0.0) <= 0.05:
        warnings.append("block_permutation_margin_not_strong")
    if any(
        _as_dict(row).get("status") == "skipped_empty_train_or_validation"
        for row in transfer.values()
    ):
        warnings.append("cross_regime_transfer_has_empty_support")
    return warnings


def _best_target_from_spatial(spatial: dict[str, Any]) -> str | None:
    target_rows = _as_dict(spatial.get("target_view_comparison"))
    if target_rows:
        values = {
            target: _metric(_as_dict(row).get("aggregate", {}), "spearman")
            for target, row in target_rows.items()
        }
        values = {target: value for target, value in values.items() if value is not None}
        if values:
            return max(values, key=lambda key: values[key])
    rows = spatial.get("contiguous_cv_rows", [])
    by_target: dict[str, list[float]] = {}
    for row in rows:
        if row.get("model") == "DummyRegressor":
            continue
        value = _metric(_as_dict(row.get("summary")).get("aggregate", {}), "spearman")
        if value is not None:
            by_target.setdefault(str(row.get("target")), []).append(float(value))
    if not by_target:
        return None
    return max(by_target, key=lambda key: np.mean(by_target[key]))


def _next_human_approval(decision: str) -> str:
    if decision == "exploratory_regime_specific_signal_only":
        return (
            "human review of a formal stratified study design; do not formally adopt "
            "regime split or orientation filter yet"
        )
    if decision == "exploratory_signal_survives_confounding_controls":
        return "human approval for limited advanced signal-processing discussion"
    return "human review of weak-label design and physical assumptions"


def _compact_common_support(report: dict[str, Any]) -> dict[str, Any]:
    groups = _as_dict(report.get("groups"))
    keep = (
        "all_samples",
        "low_orientation",
        "high_orientation",
        "regime_a_high_orientation",
        "regime_b_low_orientation",
        "regime_b_high_orientation",
        "regime_c_low_orientation",
        "regime_c_high_orientation",
    )
    return {
        "counts": {name: groups.get(name, {}).get("sample_count") for name in keep},
        "depth_ranges": {name: groups.get(name, {}).get("depth_range") for name in keep},
        "answers": report.get("answers"),
        "orientation_effect_not_identifiable_from_current_well": report.get(
            "orientation_effect_not_identifiable_from_current_well"
        ),
        "identifiability_warning": report.get("identifiability_warning"),
    }


def _compact_matched(report: dict[str, Any]) -> dict[str, Any]:
    matched = _as_dict(report.get("matched_controls"))
    return {
        "spearman_mean": matched.get("spearman_mean"),
        "spearman_std": matched.get("spearman_std"),
        "spearman_p95": matched.get("spearman_p95"),
        "high_minus_control_p95_spearman": matched.get("high_minus_control_p95_spearman"),
        "warnings": report.get("warnings"),
    }


def _compact_spatial(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_candidate": report.get("best_candidate"),
        "decision_support": report.get("decision_support"),
        "spatial_leakage_warnings": report.get("spatial_leakage_warnings"),
        "cross_regime_transfer": report.get("cross_regime_transfer"),
    }


def write_cf_review_pack(
    *,
    review_dir: Path,
    snapshot: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    common_support: dict[str, Any],
    nuisance_audit: dict[str, Any],
    spatial_validation: dict[str, Any],
    stratified: dict[str, Any],
    feature_review: dict[str, Any],
    decision: dict[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    summary_md = review_dir / "review_summary.md"
    _ensure_can_write(summary_md, overwrite=overwrite)
    summary_md.write_text(format_review_summary(decision), encoding="utf-8")
    files["review_summary_md"] = str(summary_md)
    plot_specs = {
        "subgroup_support_matrix.png": lambda path: _plot_subgroup_support(common_support, path),
        "orientation_vs_depth.png": lambda path: _plot_orientation_vs_depth(snapshot, masks, path),
        "target_vs_depth.png": lambda path: _plot_target_vs_depth(snapshot, masks, path),
        "nuisance_baseline_comparison.png": lambda path: _plot_nuisance(nuisance_audit, path),
        "feature_depth_proxy_summary.png": lambda path: _plot_feature_proxy(nuisance_audit, path),
        "blocked_gap_cv_summary.png": lambda path: _plot_blocked_gap(spatial_validation, path),
        "global_vs_within_bin_permutation.png": lambda path: _plot_permutation(
            spatial_validation, path
        ),
        "cross_regime_transfer.png": lambda path: _plot_transfer(spatial_validation, path),
        "target_view_comparison.png": lambda path: _plot_target_comparison(
            spatial_validation, path
        ),
        "residual_vs_depth.png": lambda path: _plot_residuals(
            snapshot, masks, spatial_validation, "depth", path
        ),
        "residual_vs_orientation.png": lambda path: _plot_residuals(
            snapshot, masks, spatial_validation, "orientation", path
        ),
        "special_band_sensitivity.png": lambda path: _plot_special(spatial_validation, path),
    }
    for name, writer in plot_specs.items():
        path = review_dir / name
        _ensure_can_write(path, overwrite=overwrite)
        writer(path)
        files[name.replace(".", "_")] = str(path)
    summary_json = review_dir / "cf_review_summary.json"
    _ensure_can_write(summary_json, overwrite=overwrite)
    summary_json.write_text(
        _json(
            {
                "review_version": REVIEW_VERSION,
                "decision": decision.get("decision"),
                "common_support": _compact_common_support(common_support),
                "nuisance": _compact_matched(nuisance_audit),
                "spatial": _compact_spatial(spatial_validation),
                "stratified": stratified,
                "feature_review": feature_review,
                **_method_flags(),
            }
        ),
        encoding="utf-8",
    )
    files["cf_review_summary_json"] = str(summary_json)
    return files


def format_common_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X CF Common-Support Audit",
        "",
        _scope_line(),
        "",
        f"- orientation_effect_not_identifiable_from_current_well: "
        f"{report.get('orientation_effect_not_identifiable_from_current_well')}",
        f"- identifiability_warning: {report.get('identifiability_warning')}",
        "",
        "## Answers",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report.get("answers", {}).items())
    lines.extend(["", "## Counts"])
    for name, row in _as_dict(report.get("groups")).items():
        lines.append(f"- {name}: n={row.get('sample_count')}, depth={row.get('depth_range')}")
    return "\n".join(lines) + "\n"


def format_nuisance_markdown(report: dict[str, Any]) -> str:
    matched = _as_dict(report.get("matched_controls"))
    lines = [
        "# MVP-4X CF Nuisance Audit",
        "",
        _scope_line(),
        "",
        f"- candidate: {report.get('candidate')}",
        f"- matched_control_spearman_mean: {matched.get('spearman_mean')}",
        f"- matched_control_spearman_p95: {matched.get('spearman_p95')}",
        f"- high_minus_control_p95_spearman: {matched.get('high_minus_control_p95_spearman')}",
        f"- warnings: {report.get('warnings')}",
        "",
        "## XSI vs Nuisance",
    ]
    for cohort, row in _as_dict(report.get("xsi_vs_nuisance")).items():
        lines.append(f"- {cohort}: {row}")
    return "\n".join(lines) + "\n"


def format_spatial_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X CF Spatial Validation",
        "",
        _scope_line(),
        "",
        f"- best_candidate: {report.get('best_candidate')}",
        f"- decision_support: {report.get('decision_support')}",
        f"- spatial_leakage_warnings: {report.get('spatial_leakage_warnings')}",
        "",
        "## Cross-Regime Transfer",
    ]
    for name, row in _as_dict(report.get("cross_regime_transfer")).items():
        lines.append(f"- {name}: {row}")
    return "\n".join(lines) + "\n"


def format_decision_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X CF Decision",
        "",
        _scope_line(),
        "",
        f"- decision: `{report.get('decision')}`",
        f"- next_minimal_recommendation: {report.get('next_minimal_recommendation')}",
        "",
        "## Answers",
    ]
    lines.extend(f"- {key}: {value}" for key, value in _as_dict(report.get("answers")).items())
    lines.extend(["", "## Not Authorized"])
    lines.extend(f"- {item}" for item in report.get("not_authorized", []))
    return "\n".join(lines) + "\n"


def format_review_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X CF Review Summary",
            "",
            _scope_line(),
            "",
            f"- decision: {report.get('decision')}",
            f"- next_minimal_recommendation: {report.get('next_minimal_recommendation')}",
            f"- production_claims_and_final_labels_forbidden: "
            f"{_as_dict(report.get('answers')).get('20_production_claims_and_final_labels_forbidden')}",
            "",
        ]
    )


def write_common_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = ["group", "sample_count", "depth_min", "depth_max", "orientation_p50"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, row in _as_dict(report.get("groups")).items():
            depth_range = row.get("depth_range") or [None, None]
            orientation = _as_dict(row.get("orientation_confidence_distribution"))
            writer.writerow(
                {
                    "group": group,
                    "sample_count": row.get("sample_count"),
                    "depth_min": depth_range[0],
                    "depth_max": depth_range[1],
                    "orientation_p50": orientation.get("p50"),
                }
            )


def write_nuisance_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = ["cohort", "feature_set", "spearman", "mae", "r2"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cohort, rows in _as_dict(report.get("nuisance_baselines")).items():
            for feature_set, row in rows.items():
                aggregate = _as_dict(_as_dict(row).get("summary")).get("aggregate", {})
                writer.writerow(
                    {
                        "cohort": cohort,
                        "feature_set": feature_set,
                        "spearman": _metric(aggregate, "spearman"),
                        "mae": _metric(aggregate, "mae"),
                        "r2": _metric(aggregate, "r2"),
                    }
                )


def write_spatial_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = ["feature_set", "target", "model", "cohort", "spearman", "mae", "r2"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("contiguous_cv_rows", []):
            aggregate = _as_dict(_as_dict(row.get("summary")).get("aggregate"))
            writer.writerow(
                {
                    "feature_set": row.get("feature_set"),
                    "target": row.get("target"),
                    "model": row.get("model"),
                    "cohort": row.get("cohort"),
                    "spearman": _metric(aggregate, "spearman"),
                    "mae": _metric(aggregate, "mae"),
                    "r2": _metric(aggregate, "r2"),
                }
            )


def format_cf_iteration_log(
    *,
    inputs: dict[str, str],
    common_support: dict[str, Any],
    nuisance_audit: dict[str, Any],
    spatial_validation: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    timestamp = _utc_now()
    files = list(inputs.values())
    common_counts = _compact_common_support(common_support).get("counts")
    matched = _compact_matched(nuisance_audit)
    best = spatial_validation.get("best_candidate")
    command_line = "scripts/07g_run_mvp4x_cf_confounding_loop.py --overwrite"
    return "\n".join(
        [
            "# MVP-4X CF Iteration Log",
            "",
            _scope_line(),
            "",
            "## Iteration 1",
            "- iteration_id: 1",
            f"- timestamp: {timestamp}",
            f"- files_read: {files}",
            "- hypothesis: orientation improvement may be a depth/regime cohort effect.",
            "- evidence_before: high-orientation signal was permutation-safe but confounded.",
            "- selected_action: common-support and identifiability audit.",
            "- files_changed: src/cement_channel/modeling/mvp4x_confounding.py, "
            "scripts/07g_run_mvp4x_cf_confounding_loop.py, tests",
            f"- commands_run: {command_line}",
            "- outputs_generated: mvp4x_cf_common_support_v001.md/json/csv",
            f"- metrics_after: {common_counts}",
            "- interpretation: orientation identifiability is determined by support matrix.",
            "- continue_or_stop: continue",
            "- human_approval_required: false",
            "- proposed_next_action: stricter depth-matched and nuisance-control audit",
            "",
            "## Iteration 2",
            "- iteration_id: 2",
            f"- timestamp: {timestamp}",
            f"- files_read: {files}",
            "- hypothesis: XSI signal may be explained by depth/regime nuisance proxies.",
            f"- evidence_before: common_support={common_counts}",
            "- selected_action: matched controls, nuisance baselines, feature-depth proxy audit.",
            "- files_changed: same tracked CF implementation files",
            f"- commands_run: {command_line}",
            "- outputs_generated: mvp4x_cf_nuisance_audit_v001.md/json/csv",
            f"- metrics_after: {matched}",
            "- interpretation: nuisance-only comparisons quantify residual XSI value.",
            "- continue_or_stop: continue",
            "- human_approval_required: false",
            "- proposed_next_action: blocked-gap and structured-permutation validation",
            "",
            "## Iteration 3",
            "- iteration_id: 3",
            f"- timestamp: {timestamp}",
            f"- files_read: {files}",
            "- hypothesis: spatial autocorrelation may inflate contiguous CV stability.",
            f"- evidence_before: matched_controls={matched}",
            "- selected_action: contiguous matrix, blocked-gap CV, within-bin/block/global "
            "permutation, B/C transfer.",
            "- files_changed: same tracked CF implementation files",
            f"- commands_run: {command_line}",
            "- outputs_generated: mvp4x_cf_spatial_validation_v001.md/json/csv",
            f"- metrics_after: best={best}",
            "- interpretation: stricter spatial validation informs branch decision.",
            "- continue_or_stop: continue",
            "- human_approval_required: false",
            "- proposed_next_action: review pack and bounded decision",
            "",
            "## Iteration 4",
            "- iteration_id: 4",
            f"- timestamp: {timestamp}",
            f"- files_read: {files}",
            "- hypothesis: a bounded CF decision can stop without formal strategy changes.",
            f"- evidence_before: spatial_support={spatial_validation.get('decision_support')}",
            "- selected_action: generate review pack and decision.",
            "- files_changed: same tracked CF implementation files",
            f"- commands_run: {command_line}",
            "- outputs_generated: mvp4x_cf_decision.md/json and mvp4x_cf_review_v001/",
            f"- metrics_after: decision={decision.get('decision')}",
            f"- interpretation: {decision.get('next_minimal_recommendation')}",
            f"- continue_or_stop: {decision.get('decision')}",
            "- human_approval_required: true",
            f"- proposed_next_action: {decision.get('next_minimal_recommendation')}",
            "",
        ]
    )


def _plot_subgroup_support(report: dict[str, Any], path: Path) -> None:
    groups = _as_dict(report.get("groups"))
    names = [
        "low_orientation",
        "high_orientation",
        "regime_a_high_orientation",
        "regime_b_low_orientation",
        "regime_b_high_orientation",
        "regime_c_low_orientation",
        "regime_c_high_orientation",
    ]
    values = [groups.get(name, {}).get("sample_count", 0) for name in names]
    _bar_plot(path, names, values, "Subgroup support", "sample count")


def _plot_orientation_vs_depth(
    snapshot: dict[str, np.ndarray], masks: dict[str, np.ndarray], path: Path
) -> None:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    orientation = np.asarray(snapshot["orientation_confidence"], dtype=np.float32).reshape(-1)
    _scatter_plot(
        path, depth, orientation, masks["high_orientation"], "Depth", "Orientation confidence"
    )


def _plot_target_vs_depth(
    snapshot: dict[str, np.ndarray], masks: dict[str, np.ndarray], path: Path
) -> None:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    target = np.asarray(snapshot["receiver_mean"], dtype=np.float32).reshape(-1)
    _scatter_plot(path, depth, target, masks["high_orientation"], "Depth", "receiver_mean")


def _plot_nuisance(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(_as_dict(report.get("nuisance_baselines")).get("high_orientation"))
    names = []
    values = []
    for name, row in rows.items():
        value = _metric(_as_dict(_as_dict(row).get("summary")).get("aggregate", {}), "spearman")
        if value is not None:
            names.append(name.replace("nuisance_", "").replace("xsi_", "xsi:")[:24])
            values.append(float(value))
    _bar_plot(path, names, values, "High-orientation nuisance audit", "Spearman")


def _plot_feature_proxy(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("feature_depth_proxy_audit")).get(
        "top_abs_correlation_with_depth", []
    )[:15]
    names = [str(row["feature_name"])[:24] for row in rows]
    values = [float(row["abs_spearman_depth"] or 0.0) for row in rows]
    _bar_plot(path, names, values, "Feature-depth proxy audit", "|Spearman(depth)|")


def _plot_blocked_gap(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("blocked_gap_cv"))
    names = list(rows)
    values = [
        _metric(_as_dict(row).get("aggregate", {}), "spearman") or 0.0 for row in rows.values()
    ]
    _bar_plot(path, names, values, "Blocked-gap CV", "Spearman")


def _plot_permutation(report: dict[str, Any], path: Path) -> None:
    perm = _as_dict(report.get("permutation"))
    names = ["real", "global", "within_bin", "block"]
    values = [
        perm.get("real_spearman") or 0.0,
        _as_dict(perm.get("global")).get("spearman_mean") or 0.0,
        _as_dict(perm.get("within_depth_bin")).get("spearman_mean") or 0.0,
        _as_dict(perm.get("block")).get("spearman_mean") or 0.0,
    ]
    _bar_plot(path, names, values, "Real vs structured permutations", "Spearman")


def _plot_transfer(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("cross_regime_transfer"))
    names = list(rows)
    values = [_metric(row, "spearman") or 0.0 for row in rows.values()]
    _bar_plot(path, names, values, "Cross-regime transfer", "Spearman")


def _plot_target_comparison(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("target_view_comparison"))
    names = list(rows)
    values = [
        _metric(_as_dict(row).get("aggregate", {}), "spearman") or 0.0 for row in rows.values()
    ]
    _bar_plot(path, names, values, "Target-view comparison", "Spearman")


def _plot_residuals(
    snapshot: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    spatial_validation: dict[str, Any],
    x_axis: str,
    path: Path,
) -> None:
    best = _as_dict(spatial_validation.get("best_candidate"))
    if not best:
        _bar_plot(path, ["no_best"], [0.0], "Residual audit", "value")
        return
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    x = depth
    xlabel = "Depth"
    if x_axis == "orientation":
        x = np.asarray(snapshot["orientation_confidence"], dtype=np.float32).reshape(-1)
        xlabel = "Orientation confidence"
    target = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    mask = masks["high_orientation"] & np.isfinite(target)
    residual_proxy = target - np.nanmean(target[mask])
    _scatter_plot(path, x, residual_proxy, mask, xlabel, "centered target residual proxy")


def _plot_special(report: dict[str, Any], path: Path) -> None:
    rows = _as_dict(report.get("exclusion_sensitivity"))
    names = list(rows)
    values = [
        _metric(_as_dict(row).get("aggregate", {}), "spearman") or 0.0 for row in rows.values()
    ]
    _bar_plot(path, names, values, "Special-band sensitivity", "Spearman")


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


def _scatter_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text(f"{xlabel},{ylabel}\n", encoding="utf-8")
        return
    valid = mask & np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x[valid], y[valid], s=4, alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def _overlap_counts(
    mask: np.ndarray, masks: dict[str, np.ndarray], keys: tuple[str, ...]
) -> dict[str, int]:
    return {key: int(np.count_nonzero(mask & masks[key])) for key in keys}


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return 1.0
    return float(np.isfinite(arr).mean())


def _finite_matrix(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _zscore_matrix(values: np.ndarray) -> np.ndarray:
    arr = _finite_matrix(values)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return ((arr - mean) / std).astype(np.float32)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return None
    if np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return None
    value = spearmanr(left[mask], right[mask]).statistic
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _partial_depth_residual_spearman(
    feature: np.ndarray, target: np.ndarray, depth: np.ndarray
) -> float | None:
    mask = np.isfinite(feature) & np.isfinite(target) & np.isfinite(depth)
    if np.count_nonzero(mask) < 5:
        return None
    design = np.column_stack(
        [
            np.ones(np.count_nonzero(mask)),
            depth[mask],
            depth[mask] ** 2,
        ]
    )
    feature_coef, *_ = np.linalg.lstsq(design, feature[mask], rcond=None)
    target_coef, *_ = np.linalg.lstsq(design, target[mask], rcond=None)
    feature_resid = feature[mask] - design @ feature_coef
    target_resid = target[mask] - design @ target_coef
    return _safe_spearman(feature_resid, target_resid)


def _top_by(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get(key) is not None]
    valid.sort(key=lambda row: abs(float(row[key])), reverse=True)
    return valid[:limit]


def _group_proxy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[float]] = {}
    for row in rows:
        value = row.get("abs_spearman_depth")
        if value is not None:
            by_group.setdefault(str(row["feature_group"]), []).append(float(value))
    output = [
        {
            "feature_group": group,
            "mean_abs_spearman_depth": float(np.mean(values)),
            "max_abs_spearman_depth": float(np.max(values)),
            "feature_count": len(values),
        }
        for group, values in by_group.items()
    ]
    output.sort(key=lambda row: row["mean_abs_spearman_depth"], reverse=True)
    return output[:20]


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


def _abs_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(float(value))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
