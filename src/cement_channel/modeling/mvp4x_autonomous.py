from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import (
    MODEL_NAMES,
    _make_model,
    _target_summary,
    compute_regression_metrics,
    contiguous_depth_folds,
)
from cement_channel.modeling.mvp4x_model_analysis import build_feature_sets

AUTONOMOUS_REPORT_VERSION = "mvp4x_autonomous_decision_v001"
SUBGROUP_AUDIT_VERSION = "mvp4x_autonomous_subgroup_audit_v001"
MATCHED_CONTROL_VERSION = "mvp4x_autonomous_matched_control_v001"
REVIEW_VERSION = "mvp4x_autonomous_review_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
DECISION_OPTIONS = {
    "exploratory_high_orientation_signal_detected",
    "exploratory_regime_dependent_signal_detected",
    "exploratory_target_view_sensitive_request_human_decision",
    "exploratory_time_frequency_features_helpful",
    "request_advanced_signal_processing_approval",
    "request_label_redesign_approval",
    "request_server_migration_approval",
    "stop_insufficient_signal",
    "stop_data_contract_issue",
    "stop_leakage_detected",
}
PRIMARY_MODEL_CANDIDATE = {
    "feature_set": "waveform_features_only",
    "target": "receiver_mean",
    "model": "Ridge",
}
TARGET_VIEWS_EXTENDED = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)


@dataclass(frozen=True)
class AutonomousOutputs:
    decision: dict[str, Any]
    subgroup_audit: dict[str, Any]
    matched_control: dict[str, Any]
    iteration_log: str
    review_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutonomousResearchError(RuntimeError):
    """Raised when the autonomous MVP-4X loop cannot run safely."""


def run_autonomous_research_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    existing_json: Path | str,
    enhanced_json: Path | str,
    rapid_decision_json: Path | str,
    config_path: Path | str,
    output_decision_md: Path | str,
    output_decision_json: Path | str,
    output_review_dir: Path | str,
    output_iteration_log: Path | str,
    output_subgroup_json: Path | str,
    output_subgroup_md: Path | str,
    output_subgroup_csv: Path | str,
    output_matched_json: Path | str,
    output_matched_md: Path | str,
    overwrite: bool = False,
) -> AutonomousOutputs:
    config = _load_yaml(config_path)
    snapshot = _load_npz(snapshot_npz)
    waveform = _load_npz(waveform_features_npz)
    existing = _read_json(existing_json)
    enhanced = _read_json(enhanced_json)
    rapid = _read_json(rapid_decision_json)
    outputs = run_autonomous_research(
        snapshot=snapshot,
        waveform=waveform,
        existing_report=existing,
        enhanced_report=enhanced,
        rapid_decision=rapid,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "existing_json": str(existing_json),
            "enhanced_json": str(enhanced_json),
            "rapid_decision_json": str(rapid_decision_json),
            "config_path": str(config_path),
        },
        review_dir=Path(output_review_dir),
        overwrite=overwrite,
    )
    write_autonomous_outputs(
        outputs,
        output_decision_md=Path(output_decision_md),
        output_decision_json=Path(output_decision_json),
        output_review_dir=Path(output_review_dir),
        output_iteration_log=Path(output_iteration_log),
        output_subgroup_json=Path(output_subgroup_json),
        output_subgroup_md=Path(output_subgroup_md),
        output_subgroup_csv=Path(output_subgroup_csv),
        output_matched_json=Path(output_matched_json),
        output_matched_md=Path(output_matched_md),
        overwrite=overwrite,
    )
    return outputs


def run_autonomous_research(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    existing_report: dict[str, Any],
    enhanced_report: dict[str, Any],
    rapid_decision: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
    review_dir: Path,
    overwrite: bool,
) -> AutonomousOutputs:
    _validate_reusable_artifacts(
        snapshot, waveform, existing_report, enhanced_report, rapid_decision
    )
    baseline_config = _as_dict(config.get("baseline"))
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    feature_sets = build_feature_sets(snapshot, waveform)
    masks = build_autonomous_masks(snapshot)
    subgroup_audit = build_subgroup_audit(snapshot=snapshot, waveform=waveform, masks=masks)
    all_sample_baseline = _extract_prior_best(enhanced_report, existing_report)
    matched_control = run_matched_control_study(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    high_orientation = run_high_orientation_baselines(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
    )
    best_high = _select_best_high_orientation_result(high_orientation)
    regime_study = run_regime_study(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
        best=best_high,
    )
    target_study = run_target_view_study(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
        best=best_high,
    )
    special_sensitivity = run_special_sensitivity_study(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
        best=best_high,
    )
    feature_importance, review_files = run_autonomous_feature_review(
        snapshot=snapshot,
        feature_sets=feature_sets,
        masks=masks,
        config=baseline_config,
        sklearn_modules=sklearn_modules,
        best=best_high,
        review_dir=review_dir,
        overwrite=overwrite,
    )
    decision = build_autonomous_decision(
        inputs=inputs,
        modeling_environment=modeling_environment.to_dict(),
        all_sample_baseline=all_sample_baseline,
        subgroup_audit=subgroup_audit,
        matched_control=matched_control,
        high_orientation=high_orientation,
        best_high=best_high,
        regime_study=regime_study,
        target_study=target_study,
        special_sensitivity=special_sensitivity,
        feature_importance=feature_importance,
        review_files=review_files,
    )
    iteration_log = format_iteration_log(
        inputs=inputs,
        all_sample_baseline=all_sample_baseline,
        subgroup_audit=subgroup_audit,
        matched_control=matched_control,
        high_orientation=high_orientation,
        decision=decision,
    )
    return AutonomousOutputs(
        decision=decision,
        subgroup_audit=subgroup_audit,
        matched_control=matched_control,
        iteration_log=iteration_log,
        review_files=review_files,
    )


def build_autonomous_masks(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    all_samples = np.ones(depth.size, dtype=bool)
    low_orientation = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(
        -1
    )
    high_orientation = all_samples & ~low_orientation
    orientation_confidence = np.asarray(
        snapshot.get("orientation_confidence", np.nan), dtype=np.float32
    )
    orientation_valid = np.isfinite(orientation_confidence.reshape(-1))
    saturation = np.asarray(snapshot["saturation_platform_flag"], dtype=bool).reshape(-1)
    transition_2582 = np.asarray(snapshot["transition_2582_flag"], dtype=bool).reshape(-1)
    transition_4219 = np.asarray(snapshot["transition_4219_flag"], dtype=bool).reshape(-1)
    special_5680 = np.asarray(snapshot["special_5680_flag"], dtype=bool).reshape(-1)
    any_special = np.asarray(snapshot["any_special_flag"], dtype=bool).reshape(-1)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    masks = {
        "all_samples": all_samples,
        "low_orientation": low_orientation,
        "high_orientation": high_orientation,
        "orientation_valid": orientation_valid,
        "saturation_platform_2400_2500": saturation,
        "transition_2582": transition_2582,
        "transition_4219": transition_4219,
        "special_band_5680": special_5680,
        "any_special_flag": any_special,
        "high_orientation_only": high_orientation,
        "high_orientation_exclude_2400_2500": high_orientation & ~saturation,
        "high_orientation_exclude_5680": high_orientation & ~special_5680,
        "high_orientation_exclude_all_special": high_orientation & ~any_special,
    }
    for regime in ("A", "B", "C"):
        regime_mask = regimes == regime
        masks[f"regime_{regime.lower()}"] = regime_mask
        masks[f"regime_{regime.lower()}_high_orientation"] = regime_mask & high_orientation
    return masks


def build_subgroup_audit(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    X_existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    X_wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    groups: dict[str, Any] = {}
    target_fields = TARGET_VIEWS_EXTENDED
    for name, mask in masks.items():
        valid = np.asarray(mask, dtype=bool).reshape(-1)
        count = int(np.count_nonzero(valid))
        if count:
            depth_range = [float(np.min(depth[valid])), float(np.max(depth[valid]))]
        else:
            depth_range = [None, None]
        groups[name] = {
            "sample_count": count,
            "sample_fraction": float(count / max(depth.size, 1)),
            "depth_range": depth_range,
            "target_distribution": {
                target: _target_summary(np.asarray(snapshot[target], dtype=np.float32)[valid])
                for target in target_fields
                if target in snapshot
            },
            "overlap_counts": {
                other: int(np.count_nonzero(valid & other_mask))
                for other, other_mask in masks.items()
                if other != name
            },
            "finite_ratio": {
                "existing_features": _finite_ratio(X_existing[valid]),
                "waveform_depth_features": _finite_ratio(X_wave[valid]),
            },
            "feature_distribution_summary": {
                "existing_feature_mean_abs": _summary_or_empty(
                    np.mean(np.abs(X_existing[valid]), axis=1)
                ),
                "waveform_feature_mean_abs": _summary_or_empty(
                    np.mean(np.abs(X_wave[valid]), axis=1)
                ),
            },
            "morphology_audit_metadata": _morphology_summary(snapshot, valid),
            "special_band_overlap": {
                "saturation_platform_2400_2500": int(
                    np.count_nonzero(valid & masks["saturation_platform_2400_2500"])
                ),
                "transition_2582": int(np.count_nonzero(valid & masks["transition_2582"])),
                "transition_4219": int(np.count_nonzero(valid & masks["transition_4219"])),
                "special_band_5680": int(np.count_nonzero(valid & masks["special_band_5680"])),
                "any_special_flag": int(np.count_nonzero(valid & masks["any_special_flag"])),
            },
        }
    return {
        "report_version": SUBGROUP_AUDIT_VERSION,
        "generated_at": _utc_now(),
        "orientation_threshold_source": (
            "snapshot.low_orientation_confidence_flag generated by prior approved config; "
            "threshold not modified in this run"
        ),
        "groups": groups,
        **RESEARCH_FLAGS,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
    }


def run_matched_control_study(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 20240603))
    repeat_count = int(_as_dict(config.get("autonomous", {})).get("matched_control_repeats", 20))
    permutation_count = int(config.get("permutation_count", 20))
    candidate = PRIMARY_MODEL_CANDIDATE
    X = _feature_matrix(feature_sets[candidate["feature_set"]])
    y = np.asarray(snapshot[candidate["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    high_mask = masks["high_orientation_only"] & np.isfinite(y)
    high_result = evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=high_mask,
        model_name=candidate["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        rng_seed=seed,
    )
    controls = []
    control_masks = build_depth_matched_controls(
        depth=depth,
        reference_mask=high_mask,
        candidate_mask=masks["all_samples"] & np.isfinite(y),
        repeat_count=repeat_count,
        seed=seed + 101,
    )
    for index, control_mask in enumerate(control_masks):
        result = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=control_mask,
            model_name=candidate["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=seed + 200 + index,
        )
        controls.append(
            {
                "control_index": index,
                "sample_count": int(np.count_nonzero(control_mask)),
                "high_orientation_fraction": float(
                    np.count_nonzero(control_mask & high_mask)
                    / max(np.count_nonzero(control_mask), 1)
                ),
                "aggregate": result["summary"]["aggregate"],
                "stable_positive_spearman_folds": result["summary"][
                    "stable_positive_spearman_folds"
                ],
            }
        )
    permutation = run_target_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=high_mask,
        model_name=candidate["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=permutation_count,
        seed=seed + 303,
    )
    high_s = _metric(high_result["summary"]["aggregate"], "spearman")
    control_s = [_metric(row["aggregate"], "spearman") for row in controls]
    control_s = [value for value in control_s if value is not None]
    p95 = None if not control_s else float(np.quantile(control_s, 0.95))
    mean = None if not control_s else float(np.mean(control_s))
    std = None if not control_s else float(np.std(control_s))
    perm_mean = permutation["spearman_mean"]
    return {
        "report_version": MATCHED_CONTROL_VERSION,
        "generated_at": _utc_now(),
        "candidate": candidate,
        "repeat_count": repeat_count,
        "matching_method": "depth_quantile_bin_sampling_fixed_seed",
        "high_orientation": high_result["summary"],
        "matched_controls": {
            "rows": controls,
            "spearman_mean": mean,
            "spearman_std": std,
            "spearman_p95": p95,
        },
        "permutation": permutation,
        "improvement_margin": {
            "high_minus_control_mean_spearman": _none_subtract(high_s, mean),
            "high_minus_control_p95_spearman": _none_subtract(high_s, p95),
            "high_minus_permutation_mean_spearman": _none_subtract(high_s, perm_mean),
        },
        "fold_sign_consistency": high_result["summary"]["stable_positive_spearman_folds"],
        "support_warnings": _matched_support_warnings(high_mask, controls),
        **RESEARCH_FLAGS,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
    }


def build_depth_matched_controls(
    *,
    depth: np.ndarray,
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    repeat_count: int,
    seed: int,
    bin_count: int = 20,
) -> list[np.ndarray]:
    values = np.asarray(depth, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference_mask, dtype=bool).reshape(-1)
    candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    if np.count_nonzero(reference) == 0:
        raise AutonomousResearchError(
            "Cannot build matched controls without high-orientation samples."
        )
    quantiles = np.quantile(values[reference], np.linspace(0.0, 1.0, bin_count + 1))
    quantiles[0] = min(quantiles[0], float(np.min(values[candidate])))
    quantiles[-1] = max(quantiles[-1], float(np.max(values[candidate]))) + 1e-3
    rng = np.random.default_rng(seed)
    controls: list[np.ndarray] = []
    for _repeat in range(repeat_count):
        selected: list[int] = []
        used: set[int] = set()
        for bin_index in range(bin_count):
            ref_bin = (
                reference & (values >= quantiles[bin_index]) & (values < quantiles[bin_index + 1])
            )
            need = int(np.count_nonzero(ref_bin))
            if need == 0:
                continue
            cand_bin = np.flatnonzero(
                candidate & (values >= quantiles[bin_index]) & (values < quantiles[bin_index + 1])
            )
            cand_bin = np.asarray([idx for idx in cand_bin.tolist() if idx not in used], dtype=int)
            if cand_bin.size < need:
                cand_bin = np.flatnonzero(
                    candidate
                    & (values >= quantiles[bin_index])
                    & (values < quantiles[bin_index + 1])
                )
            replace = cand_bin.size < need
            picks = rng.choice(cand_bin, size=need, replace=replace)
            for pick in np.asarray(picks, dtype=int).tolist():
                selected.append(pick)
                used.add(pick)
        mask = np.zeros(values.size, dtype=bool)
        mask[np.asarray(selected, dtype=int)] = True
        controls.append(mask)
    return controls


def run_high_orientation_baselines(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 20240603))
    permutation_count = int(config.get("permutation_count", 20))
    target_views = tuple(
        _as_list(config.get("target_views"), ["receiver_p90", "receiver_mean", "receiver_max"])
    )
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    for feature_set_name, feature_set in feature_sets.items():
        X = _feature_matrix(feature_set)
        for target in target_views:
            y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
            sample_mask = masks["high_orientation_only"] & np.isfinite(y)
            for model_name in MODEL_NAMES:
                print(
                    "MVP-4X autonomous high-orientation "
                    f"feature_set={feature_set_name} target={target} model={model_name}",
                    flush=True,
                )
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
                        "sample_count": int(np.count_nonzero(sample_mask)),
                        "summary": result["summary"],
                    }
                )
    candidates = _top_rows_for_permutation(rows, limit=10)
    for row in candidates:
        X = _feature_matrix(feature_sets[row["feature_set"]])
        y = np.asarray(snapshot[row["target"]], dtype=np.float32).reshape(-1)
        sample_mask = masks["high_orientation_only"] & np.isfinite(y)
        row["permutation"] = run_target_permutations(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=row["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            permutation_count=permutation_count,
            seed=seed + 707 + candidates.index(row),
        )
        row["real_minus_permutation_spearman"] = _none_subtract(
            _metric(row["summary"]["aggregate"], "spearman"),
            row["permutation"].get("spearman_mean"),
        )
    return {
        "status": "completed",
        "cohort": "high_orientation_only",
        "target_views": list(target_views),
        "feature_sets": list(feature_sets),
        "models": list(MODEL_NAMES),
        "rows": rows,
        "permutation_evaluated_for_top_candidate_count": len(candidates),
        **RESEARCH_FLAGS,
    }


def run_regime_study(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    if not best:
        return {"status": "skipped_no_high_orientation_best", **RESEARCH_FLAGS}
    X = _feature_matrix(feature_sets[best["feature_set"]])
    y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    rows: dict[str, Any] = {}
    for regime in ("A", "B", "C"):
        mask = masks[f"regime_{regime.lower()}_high_orientation"] & np.isfinite(y)
        if np.count_nonzero(mask) < int(config.get("n_contiguous_folds", 3)):
            rows[regime] = {"status": "skipped_too_few_samples", "sample_count": int(mask.sum())}
            continue
        result = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=mask,
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=901 + ord(regime),
        )
        rows[regime] = result["summary"]
    leave_one: dict[str, Any] = {}
    high = masks["high_orientation_only"] & np.isfinite(y)
    for regime in ("A", "B", "C"):
        validation_mask = high & (regimes == regime)
        train_mask = high & (regimes != regime)
        leave_one[f"train_not_{regime}_validate_{regime}"] = fit_eval_once(
            X=X,
            y=y,
            train_mask=train_mask,
            validation_mask=validation_mask,
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            random_state=1001 + ord(regime),
        )
    transfer: dict[str, Any] = {}
    for train_regime in ("A", "B", "C"):
        for validate_regime in ("A", "B", "C"):
            if train_regime == validate_regime:
                continue
            transfer[f"{train_regime}_to_{validate_regime}"] = fit_eval_once(
                X=X,
                y=y,
                train_mask=high & (regimes == train_regime),
                validation_mask=high & (regimes == validate_regime),
                model_name=best["model"],
                sklearn_modules=sklearn_modules,
                config=config,
                random_state=1101 + ord(train_regime) + ord(validate_regime),
            )
    return {
        "status": "completed",
        "best_basis": best,
        "within_regime_cv": rows,
        "leave_one_regime_out": leave_one,
        "cross_regime_transfer": transfer,
        "domain_shift_warnings": _domain_shift_warnings(rows, leave_one, transfer),
        **RESEARCH_FLAGS,
    }


def run_target_view_study(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    if not best:
        return {"status": "skipped_no_high_orientation_best", **RESEARCH_FLAGS}
    feature_set_name = best["feature_set"]
    model_name = best["model"]
    X = _feature_matrix(feature_sets[feature_set_name])
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    rows: dict[str, Any] = {}
    for target in TARGET_VIEWS_EXTENDED:
        y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
        mask = masks["high_orientation_only"] & np.isfinite(y)
        result = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=1201 + len(rows),
        )
        rows[target] = {
            "target_policy": _target_policy(target),
            "summary": result["summary"],
        }
    recommendation = _target_view_recommendation(rows)
    return {
        "status": "completed",
        "feature_set": feature_set_name,
        "model": model_name,
        "target_views": rows,
        "recommendation_for_human_review": recommendation,
        **RESEARCH_FLAGS,
    }


def run_special_sensitivity_study(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    if not best:
        return {"status": "skipped_no_high_orientation_best", **RESEARCH_FLAGS}
    X = _feature_matrix(feature_sets[best["feature_set"]])
    y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    filters = (
        "high_orientation_only",
        "high_orientation_exclude_2400_2500",
        "high_orientation_exclude_5680",
        "high_orientation_exclude_all_special",
        "low_orientation",
    )
    rows: dict[str, Any] = {}
    for name in filters:
        mask = masks[name] & np.isfinite(y)
        if np.count_nonzero(mask) < int(config.get("n_contiguous_folds", 3)):
            rows[name] = {"status": "skipped_too_few_samples", "sample_count": int(mask.sum())}
            continue
        result = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=mask,
            model_name=best["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=1301 + len(rows),
        )
        rows[name] = result["summary"]
    base_s = _metric(_as_dict(rows.get("high_orientation_only")).get("aggregate", {}), "spearman")
    for name, row in rows.items():
        if name == "high_orientation_only" or _as_dict(row).get("aggregate") is None:
            continue
        row["spearman_delta_vs_high_orientation"] = _none_subtract(
            _metric(row["aggregate"], "spearman"),
            base_s,
        )
    return {"status": "completed", "best_basis": best, "filters": rows, **RESEARCH_FLAGS}


def run_autonomous_feature_review(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    best: dict[str, Any] | None,
    review_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not best:
        return {"status": "skipped_no_high_orientation_best", **RESEARCH_FLAGS}, {}
    feature_set = feature_sets[best["feature_set"]]
    X = _feature_matrix(feature_set)
    y = np.asarray(snapshot[best["target"]], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    mask = masks["high_orientation_only"] & np.isfinite(y)
    oof, fold_rows, importance_rows = fit_oof_with_permutation_importance(
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
    review_files = write_review_files(
        review_dir=review_dir,
        y_true=y,
        y_pred=oof,
        depth=depth,
        sample_mask=mask,
        subgroup_audit={
            "best_basis": best,
            "folds": fold_rows,
            "aggregate_metrics": compute_regression_metrics(y[mask], oof[mask]),
            "top_10_stable_features": importance["top_10_stable_features"],
            "top_feature_groups": group_importance[:10],
            **RESEARCH_FLAGS,
        },
        overwrite=overwrite,
    )
    return (
        {
            "status": "completed",
            "method": "validation_fold_feature_shuffle_high_orientation",
            "best_basis": best,
            "folds": fold_rows,
            "top_30_features": importance["top_30_features"],
            "top_10_stable_features": importance["top_10_stable_features"],
            "top_feature_groups": group_importance[:10],
            **RESEARCH_FLAGS,
        },
        review_files,
    )


def evaluate_model_cv(
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
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1) & np.all(np.isfinite(X), axis=1)
    folds = contiguous_depth_folds(
        depth,
        mask,
        n_folds=int(config.get("n_contiguous_folds", 3)),
    )
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    for fold_index, validation_mask in enumerate(folds):
        train_mask = mask & ~validation_mask
        validation_mask = mask & validation_mask
        if np.count_nonzero(train_mask) == 0 or np.count_nonzero(validation_mask) == 0:
            continue
        model = _make_model(
            model_name,
            sklearn_modules,
            config,
            random_state=rng_seed + fold_index,
        )
        model.fit(X[train_mask], y[train_mask])
        prediction = np.asarray(model.predict(X[validation_mask]), dtype=np.float32)
        oof[validation_mask] = prediction
        fold_metrics.append(
            {
                "fold": fold_index,
                "validation_count": int(np.count_nonzero(validation_mask)),
                "validation_depth_min": float(np.min(depth[validation_mask])),
                "validation_depth_max": float(np.max(depth[validation_mask])),
                **compute_regression_metrics(y[validation_mask], prediction),
            }
        )
    aggregate = compute_regression_metrics(y[mask], oof[mask])
    stable_folds = int(
        np.count_nonzero(
            [
                (metrics.get("spearman") is not None) and (float(metrics["spearman"]) > 0.0)
                for metrics in fold_metrics
            ]
        )
    )
    return {
        "summary": {
            "status": "completed",
            "sample_count": int(np.count_nonzero(mask)),
            "fold_count": len(fold_metrics),
            "stable_positive_spearman_folds": stable_folds,
            "aggregate": aggregate,
            "folds": fold_metrics,
            "prediction_summary": _target_summary(oof[mask]),
            "calibration_by_target_quantile": _calibration_by_target_quantile(y[mask], oof[mask]),
        },
        "oof_prediction": oof,
    }


def run_target_permutations(
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
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    selected = np.flatnonzero(sample_mask)
    y_selected = y[selected].copy()
    metrics: list[dict[str, Any]] = []
    for index in range(permutation_count):
        y_perm = y.copy()
        y_perm[selected] = rng.permutation(y_selected)
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
    spearman_values = [float(row["spearman"]) for row in metrics if row.get("spearman") is not None]
    return {
        "permutation_count": permutation_count,
        "spearman_mean": None if not spearman_values else float(np.mean(spearman_values)),
        "spearman_std": None if not spearman_values else float(np.std(spearman_values)),
        "spearman_p95": None if not spearman_values else float(np.quantile(spearman_values, 0.95)),
        "spearman_max": None if not spearman_values else float(np.max(spearman_values)),
        "metrics": metrics,
    }


def fit_eval_once(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    train = np.asarray(train_mask, dtype=bool).reshape(-1) & np.all(np.isfinite(X), axis=1)
    validation = np.asarray(validation_mask, dtype=bool).reshape(-1) & np.all(
        np.isfinite(X), axis=1
    )
    if np.count_nonzero(train) == 0 or np.count_nonzero(validation) == 0:
        return {"status": "skipped_empty_train_or_validation"}
    model = _make_model(model_name, sklearn_modules, config, random_state=random_state)
    model.fit(X[train], y[train])
    prediction = np.asarray(model.predict(X[validation]), dtype=np.float32)
    return {
        "status": "completed",
        "train_count": int(np.count_nonzero(train)),
        "validation_count": int(np.count_nonzero(validation)),
        **compute_regression_metrics(y[validation], prediction),
    }


def fit_oof_with_permutation_importance(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    feature_names: np.ndarray,
    feature_groups: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1) & np.all(np.isfinite(X), axis=1)
    folds = contiguous_depth_folds(
        depth,
        mask,
        n_folds=int(config.get("n_contiguous_folds", 3)),
    )
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for fold_index, validation_mask in enumerate(folds):
        train_mask = mask & ~validation_mask
        validation_mask = mask & validation_mask
        model = _make_model(model_name, sklearn_modules, config, random_state=1701 + fold_index)
        model.fit(X[train_mask], y[train_mask])
        X_validation = X[validation_mask]
        y_validation = y[validation_mask]
        prediction = np.asarray(model.predict(X_validation), dtype=np.float32)
        oof[validation_mask] = prediction
        metrics = compute_regression_metrics(y_validation, prediction)
        fold_rows.append(
            {
                "fold": fold_index,
                "validation_count": int(np.count_nonzero(validation_mask)),
                **metrics,
            }
        )
        baseline_s = _metric(metrics, "spearman") or 0.0
        rng = np.random.default_rng(1801 + fold_index)
        for feature_index, feature_name in enumerate(feature_names.tolist()):
            permuted = X_validation.copy()
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            permuted_prediction = np.asarray(model.predict(permuted), dtype=np.float32)
            permuted_metrics = compute_regression_metrics(y_validation, permuted_prediction)
            permuted_s = _metric(permuted_metrics, "spearman")
            importance_rows.append(
                {
                    "fold": fold_index,
                    "feature_index": int(feature_index),
                    "feature_name": str(feature_name),
                    "feature_group": str(feature_groups[feature_index]),
                    "spearman_drop": None if permuted_s is None else float(baseline_s - permuted_s),
                }
            )
    return oof, fold_rows, importance_rows


def summarize_importance(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_feature: dict[int, list[dict[str, Any]]] = {}
    fold_top: dict[int, set[int]] = {}
    for row in rows:
        by_feature.setdefault(int(row["feature_index"]), []).append(row)
    for fold in sorted({int(row["fold"]) for row in rows}):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        fold_rows.sort(key=lambda row: _float_or_neg_inf(row.get("spearman_drop")), reverse=True)
        fold_top[fold] = {int(row["feature_index"]) for row in fold_rows[:30]}
    summaries: list[dict[str, Any]] = []
    for feature_index, feature_rows in by_feature.items():
        drops = [
            float(row["spearman_drop"])
            for row in feature_rows
            if row.get("spearman_drop") is not None
        ]
        summaries.append(
            {
                "feature_index": feature_index,
                "feature_name": str(feature_rows[0]["feature_name"]),
                "feature_group": str(feature_rows[0]["feature_group"]),
                "mean_spearman_drop": None if not drops else float(np.mean(drops)),
                "std_spearman_drop": None if not drops else float(np.std(drops)),
                "fold_count": len(feature_rows),
                "top_30_fold_count": int(sum(feature_index in top for top in fold_top.values())),
            }
        )
    summaries.sort(
        key=lambda row: (
            _float_or_neg_inf(row.get("mean_spearman_drop")),
            int(row.get("top_30_fold_count") or 0),
        ),
        reverse=True,
    )
    stable = [
        row
        for row in summaries
        if int(row.get("top_30_fold_count") or 0) >= min(2, max(len(fold_top), 1))
    ]
    if len(stable) < 10:
        stable = summaries[:10]
    return {
        "top_30_features": summaries[:30],
        "top_10_stable_features": stable[:10],
    }


def summarize_group_importance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[float]] = {}
    for row in rows:
        if row.get("spearman_drop") is None:
            continue
        by_group.setdefault(str(row["feature_group"]), []).append(float(row["spearman_drop"]))
    summaries = [
        {
            "feature_group": group,
            "mean_spearman_drop": float(np.mean(values)),
            "max_spearman_drop": float(np.max(values)),
            "feature_fold_count": len(values),
        }
        for group, values in by_group.items()
    ]
    summaries.sort(key=lambda row: row["mean_spearman_drop"], reverse=True)
    return summaries


def build_autonomous_decision(
    *,
    inputs: dict[str, str],
    modeling_environment: dict[str, Any],
    all_sample_baseline: dict[str, Any],
    subgroup_audit: dict[str, Any],
    matched_control: dict[str, Any],
    high_orientation: dict[str, Any],
    best_high: dict[str, Any] | None,
    regime_study: dict[str, Any],
    target_study: dict[str, Any],
    special_sensitivity: dict[str, Any],
    feature_importance: dict[str, Any],
    review_files: dict[str, str],
) -> dict[str, Any]:
    high_signal = _high_orientation_signal_detected(best_high, matched_control, special_sensitivity)
    regime_dependent = _regime_dependent(regime_study)
    target_sensitive = _target_sensitive(target_study)
    depth_regime_confounded = _orientation_depth_regime_confounded(
        matched_control,
        regime_study,
    )
    permutation_safe = _best_is_permutation_safe(best_high)
    if not best_high:
        decision = "stop_insufficient_signal"
    elif permutation_safe and depth_regime_confounded:
        decision = "exploratory_regime_dependent_signal_detected"
    elif high_signal and regime_dependent:
        decision = "exploratory_regime_dependent_signal_detected"
    elif high_signal:
        decision = "exploratory_high_orientation_signal_detected"
    elif target_sensitive:
        decision = "exploratory_target_view_sensitive_request_human_decision"
    elif _label_redesign_needed(best_high, matched_control, high_orientation):
        decision = "request_label_redesign_approval"
    else:
        decision = "stop_insufficient_signal"
    if decision not in DECISION_OPTIONS:
        raise AutonomousResearchError(f"Unsupported autonomous decision: {decision}")
    answers = {
        "1_permutation_safe_signal": _permutation_safe_answer(best_high, matched_control),
        "2_signal_only_high_orientation_subset": _high_orientation_only_answer(
            all_sample_baseline,
            best_high,
            special_sensitivity,
        ),
        "3_high_orientation_exceeds_matched_controls": _matched_answer(matched_control),
        "4_depends_on_2400_2500_platform": _sensitivity_answer(
            special_sensitivity,
            "high_orientation_exclude_2400_2500",
        ),
        "5_depends_on_5680_band": _sensitivity_answer(
            special_sensitivity,
            "high_orientation_exclude_5680",
        ),
        "6_depends_on_single_regime": _single_regime_answer(regime_study),
        "7_most_stable_target_view": _target_view_recommendation(
            _as_dict(target_study.get("target_views"))
        ),
        "8_most_stable_feature_set": None if not best_high else best_high.get("feature_set"),
        "9_most_stable_classical_model": None if not best_high else best_high.get("model"),
        "10_waveform_v1_improved": all_sample_baseline.get("waveform_improved"),
        "11_time_frequency_v2_triggered": False,
        "12_time_frequency_v2_improved": "not_triggered",
        "13_important_feature_groups": feature_importance.get("top_feature_groups", [])[:10],
        "14_stable_folds": None if not best_high else best_high.get("folds"),
        "15_weakest_regimes": _weakest_regime_answer(regime_study),
        "16_cross_regime_domain_shift": regime_study.get("domain_shift_warnings"),
        "16b_orientation_depth_regime_confounding": depth_regime_confounded,
        "17_morphology_label_redesign_value": (
            "audit_only_discuss_if_subgroup_errors_align_with_morphology_metadata"
        ),
        "18_needs_human_scientific_route_confirmation": decision
        in {
            "exploratory_regime_dependent_signal_detected",
            "exploratory_target_view_sensitive_request_human_decision",
            "request_label_redesign_approval",
        },
        "19_local_resources_sufficient": True,
        "20_recommend_server_migration": False,
        "21_production_claims_and_final_labels_still_forbidden": True,
    }
    return {
        "decision_version": AUTONOMOUS_REPORT_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "inputs": inputs,
        "modeling_environment": modeling_environment,
        "artifact_reuse": {
            "snapshot_reused": True,
            "waveform_v1_reused": True,
            "time_frequency_v2_extracted": False,
            "raw_waveform_reread": False,
        },
        "all_sample_baseline": all_sample_baseline,
        "best_high_orientation_result": best_high,
        "matched_control_summary": matched_control,
        "high_orientation_baselines": high_orientation,
        "regime_study": regime_study,
        "target_view_study": target_study,
        "special_band_sensitivity": special_sensitivity,
        "feature_importance": feature_importance,
        "subgroup_audit_summary": _compact_subgroup_counts(subgroup_audit),
        "review_dir": review_files,
        "answers": answers,
        "next_minimal_recommendation": _next_minimal_recommendation(decision),
        "not_authorized": [
            "production claim",
            "final labels",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "regime-specific model",
            "formal low-orientation exclusion rule",
        ],
        **RESEARCH_FLAGS,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
    }


def write_autonomous_outputs(
    outputs: AutonomousOutputs,
    *,
    output_decision_md: Path,
    output_decision_json: Path,
    output_review_dir: Path,
    output_iteration_log: Path,
    output_subgroup_json: Path,
    output_subgroup_md: Path,
    output_subgroup_csv: Path,
    output_matched_json: Path,
    output_matched_md: Path,
    overwrite: bool,
) -> None:
    for path in (
        output_decision_md,
        output_decision_json,
        output_iteration_log,
        output_subgroup_json,
        output_subgroup_md,
        output_subgroup_csv,
        output_matched_json,
        output_matched_md,
    ):
        _ensure_can_write(path, overwrite=overwrite)
    output_review_dir.mkdir(parents=True, exist_ok=True)
    output_decision_md.parent.mkdir(parents=True, exist_ok=True)
    output_decision_json.write_text(
        json.dumps(outputs.decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_decision_md.write_text(format_decision_markdown(outputs.decision), encoding="utf-8")
    output_iteration_log.write_text(outputs.iteration_log, encoding="utf-8")
    output_subgroup_json.write_text(
        json.dumps(outputs.subgroup_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_subgroup_md.write_text(
        format_subgroup_markdown(outputs.subgroup_audit),
        encoding="utf-8",
    )
    write_subgroup_csv(outputs.subgroup_audit, output_subgroup_csv)
    output_matched_json.write_text(
        json.dumps(outputs.matched_control, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_matched_md.write_text(
        format_matched_markdown(outputs.matched_control),
        encoding="utf-8",
    )


def write_review_files(
    *,
    review_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    subgroup_audit: dict[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    summary_json = review_dir / "autonomous_review_summary.json"
    summary_md = review_dir / "autonomous_review_summary.md"
    for path in (summary_json, summary_md):
        _ensure_can_write(path, overwrite=overwrite)
    summary_json.write_text(
        json.dumps(
            {"review_version": REVIEW_VERSION, **subgroup_audit}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(
        "\n".join(
            [
                "# MVP-4X Autonomous Review",
                "",
                "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
                "no_ground_truth_claim, no_production_claim.",
                "",
                f"- best_basis: {subgroup_audit.get('best_basis')}",
                f"- top_stable_features: {len(subgroup_audit.get('top_10_stable_features', []))}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    plot_files = _write_plots(
        review_dir=review_dir,
        y_true=y_true,
        y_pred=y_pred,
        depth=depth,
        sample_mask=sample_mask,
        overwrite=overwrite,
    )
    return {
        "autonomous_review_summary_json": str(summary_json),
        "autonomous_review_summary_md": str(summary_md),
        **plot_files,
    }


def format_decision_markdown(report: dict[str, Any]) -> str:
    best = _as_dict(report.get("best_high_orientation_result"))
    lines = [
        "# MVP-4X Autonomous Decision",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- best_model: {best.get('model')}",
        f"- best_feature_set: {best.get('feature_set')}",
        f"- best_target: {best.get('target')}",
        f"- best_spearman: {best.get('spearman')}",
        f"- real_minus_permutation_spearman: {best.get('real_minus_permutation_spearman')}",
        f"- next_minimal_recommendation: {report.get('next_minimal_recommendation')}",
        "",
        "## Answers",
    ]
    for key, value in _as_dict(report.get("answers")).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Not Authorized"])
    lines.extend(f"- {item}" for item in report.get("not_authorized", []))
    return "\n".join(lines) + "\n"


def format_subgroup_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Autonomous Subgroup Audit",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- orientation_threshold_source: {report.get('orientation_threshold_source')}",
        "",
        "## Counts",
    ]
    for name, row in _as_dict(report.get("groups")).items():
        lines.append(
            f"- {name}: n={row.get('sample_count')}, fraction={row.get('sample_fraction')}"
        )
    return "\n".join(lines) + "\n"


def format_matched_markdown(report: dict[str, Any]) -> str:
    high_s = _metric(_as_dict(report.get("high_orientation", {})).get("aggregate", {}), "spearman")
    control = _as_dict(report.get("matched_controls"))
    margins = _as_dict(report.get("improvement_margin"))
    permutation_mean = _as_dict(report.get("permutation")).get("spearman_mean")
    high_minus_control_p95 = margins.get("high_minus_control_p95_spearman")
    high_minus_permutation = margins.get("high_minus_permutation_mean_spearman")
    return "\n".join(
        [
            "# MVP-4X Autonomous Matched-Control Study",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim.",
            "",
            f"- candidate: {report.get('candidate')}",
            f"- high_orientation_spearman: {high_s}",
            f"- matched_control_spearman_mean: {control.get('spearman_mean')}",
            f"- matched_control_spearman_p95: {control.get('spearman_p95')}",
            f"- permutation_spearman_mean: {permutation_mean}",
            f"- high_minus_control_p95_spearman: {high_minus_control_p95}",
            f"- high_minus_permutation_mean_spearman: {high_minus_permutation}",
            "",
        ]
    )


def write_subgroup_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "group",
        "sample_count",
        "sample_fraction",
        "depth_min",
        "depth_max",
        "existing_finite_ratio",
        "waveform_finite_ratio",
        "saturation_overlap",
        "special_5680_overlap",
        "any_special_overlap",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, row in _as_dict(report.get("groups")).items():
            depth_range = row.get("depth_range") or [None, None]
            finite = _as_dict(row.get("finite_ratio"))
            special = _as_dict(row.get("special_band_overlap"))
            writer.writerow(
                {
                    "group": group,
                    "sample_count": row.get("sample_count"),
                    "sample_fraction": row.get("sample_fraction"),
                    "depth_min": depth_range[0],
                    "depth_max": depth_range[1],
                    "existing_finite_ratio": finite.get("existing_features"),
                    "waveform_finite_ratio": finite.get("waveform_depth_features"),
                    "saturation_overlap": special.get("saturation_platform_2400_2500"),
                    "special_5680_overlap": special.get("special_band_5680"),
                    "any_special_overlap": special.get("any_special_flag"),
                }
            )


def format_iteration_log(
    *,
    inputs: dict[str, str],
    all_sample_baseline: dict[str, Any],
    subgroup_audit: dict[str, Any],
    matched_control: dict[str, Any],
    high_orientation: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    timestamp = _utc_now()
    high_count = _as_dict(_as_dict(subgroup_audit.get("groups")).get("high_orientation")).get(
        "sample_count"
    )
    low_count = _as_dict(_as_dict(subgroup_audit.get("groups")).get("low_orientation")).get(
        "sample_count"
    )
    best = _as_dict(decision.get("best_high_orientation_result"))
    command_line = "scripts/07f_run_mvp4x_autonomous_research_loop.py --overwrite"
    subgroup_action = (
        "- selected_action: build subgroup masks and audit sample/target/feature "
        "distributions."
    )
    subgroup_metrics = (
        f"- metrics_after: high_orientation_count={high_count}, "
        f"low_orientation_count={low_count}"
    )
    matched_action = (
        "- selected_action: run fixed-seed matched controls and target permutation for "
        "Ridge / waveform_features_only / receiver_mean."
    )
    high_action = (
        "- selected_action: run bounded high-orientation model/feature/target comparison, "
        "regime study, target-view study, sensitivity, and feature review."
    )
    high_count_text = (
        "- evidence_before: high_orientation_candidate_count="
        f"{len(high_orientation.get('rows', []))}"
    )
    return "\n".join(
        [
            "# MVP-4X Autonomous Iteration Log",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim.",
            "",
            "## Iteration 1",
            "- iteration_id: 1",
            f"- timestamp: {timestamp}",
            f"- files_read: {list(inputs.values())}",
            "- hypothesis: low-orientation intervals are a major confounder in all-sample metrics.",
            f"- evidence_before: {all_sample_baseline}",
            subgroup_action,
            "- tracked_files_changed: pending autonomous module/script/tests commit",
            f"- commands_run: {command_line}",
            "- outputs_generated: mvp4x_autonomous_subgroup_audit_v001.json/md/csv",
            subgroup_metrics,
            "- interpretation: subgroup support is available for bounded cohort analysis.",
            "- decision: continue_to_matched_control",
            "- reason_to_continue_or_stop: high-orientation sensitivity was the largest "
            "prior effect.",
            "- whether_human_approval_required: false",
            "- proposed_next_action: depth-matched high-orientation control evaluation",
            "",
            "## Iteration 2",
            "- iteration_id: 2",
            f"- timestamp: {timestamp}",
            f"- files_read: {list(inputs.values())}",
            "- hypothesis: high-orientation improvement exceeds depth-matched sample-count "
            "controls.",
            f"- evidence_before: prior_all_sample_baseline={all_sample_baseline}",
            matched_action,
            "- tracked_files_changed: pending autonomous module/script/tests commit",
            f"- commands_run: {command_line}",
            "- outputs_generated: mvp4x_autonomous_matched_control_v001.json/md",
            f"- metrics_after: {matched_control.get('improvement_margin')}",
            "- interpretation: matched-control evidence determines whether sample-count "
            "reduction explains the signal.",
            "- decision: continue_to_high_orientation_baseline",
            "- reason_to_continue_or_stop: compare all classical models, feature sets, "
            "and target views within bounded cohort.",
            "- whether_human_approval_required: false",
            "- proposed_next_action: high-orientation classical baseline and regime/target audit",
            "",
            "## Iteration 3",
            "- iteration_id: 3",
            f"- timestamp: {timestamp}",
            f"- files_read: {list(inputs.values())}",
            "- hypothesis: any learnable signal is high-orientation and possibly "
            "regime/target-view dependent.",
            high_count_text,
            high_action,
            "- tracked_files_changed: pending autonomous module/script/tests commit",
            f"- commands_run: {command_line}",
            "- outputs_generated: mvp4x_autonomous_decision.json/md and "
            "mvp4x_autonomous_review_v001/",
            f"- metrics_after: best={best}",
            f"- interpretation: decision={decision.get('decision')}",
            f"- decision: {decision.get('decision')}",
            f"- reason_to_continue_or_stop: {decision.get('next_minimal_recommendation')}",
            "- whether_human_approval_required: "
            f"{_as_dict(decision.get('answers')).get('18_needs_human_scientific_route_confirmation')}",
            f"- proposed_next_action: {decision.get('next_minimal_recommendation')}",
            "",
        ]
    )


def _validate_reusable_artifacts(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    existing_report: dict[str, Any],
    enhanced_report: dict[str, Any],
    rapid_decision: dict[str, Any],
) -> None:
    required_targets = {
        "receiver_mean",
        "receiver_p90",
        "receiver_max",
        "full_360_fraction",
        "receiver_std",
    }
    missing_targets = sorted(target for target in required_targets if target not in snapshot)
    if missing_targets:
        raise AutonomousResearchError(f"Missing target views: {missing_targets}")
    if snapshot["depth"].shape[0] != 7108:
        raise AutonomousResearchError(
            f"Unexpected snapshot sample count: {snapshot['depth'].shape}"
        )
    if snapshot["xsi_features"].shape != (7108, 80):
        raise AutonomousResearchError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise AutonomousResearchError(
            f"Unexpected waveform feature shape: {waveform['waveform_depth_features'].shape}"
        )
    if snapshot["xsi_features"].shape[1] + waveform["waveform_depth_features"].shape[1] != 422:
        raise AutonomousResearchError("Combined feature count is not 422.")
    if float(np.isfinite(snapshot["xsi_features"]).mean()) != 1.0:
        raise AutonomousResearchError("Existing feature finite ratio is not 1.0.")
    if float(np.isfinite(waveform["waveform_depth_features"]).mean()) != 1.0:
        raise AutonomousResearchError("Waveform feature finite ratio is not 1.0.")
    if str(snapshot["target_kernel"]) != "triangular_midpoint_weighted":
        raise AutonomousResearchError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    if not all(bool(existing_report.get(flag)) for flag in RESEARCH_FLAGS):
        raise AutonomousResearchError("Existing baseline report is missing research-only flags.")
    if not all(bool(enhanced_report.get(flag)) for flag in RESEARCH_FLAGS):
        raise AutonomousResearchError("Enhanced baseline report is missing research-only flags.")
    if not all(bool(rapid_decision.get(flag)) for flag in RESEARCH_FLAGS):
        raise AutonomousResearchError("Rapid decision report is missing research-only flags.")


def _feature_matrix(feature_set: dict[str, Any]) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(feature_set["matrix"], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _extract_prior_best(
    enhanced_report: dict[str, Any],
    existing_report: dict[str, Any],
) -> dict[str, Any]:
    existing_best = _as_dict(_as_dict(existing_report.get("decision")).get("best_result"))
    enhanced_best = _as_dict(enhanced_report.get("best_result"))
    existing_s = existing_best.get("spearman")
    enhanced_s = enhanced_best.get("spearman")
    return {
        "existing_best": existing_best,
        "enhanced_best": enhanced_best,
        "waveform_improved": None
        if existing_s is None or enhanced_s is None
        else float(enhanced_s) > float(existing_s),
        "enhanced_minus_existing_spearman": None
        if existing_s is None or enhanced_s is None
        else float(enhanced_s) - float(existing_s),
    }


def _select_best_high_orientation_result(report: dict[str, Any]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in report.get("rows", []):
        if row.get("model") == "DummyRegressor":
            continue
        aggregate = _as_dict(_as_dict(row.get("summary")).get("aggregate"))
        spearman = aggregate.get("spearman")
        if spearman is None:
            continue
        margin = row.get("real_minus_permutation_spearman")
        candidate = {
            "feature_set": row.get("feature_set"),
            "target": row.get("target"),
            "model": row.get("model"),
            "sample_count": row.get("sample_count"),
            "spearman": float(spearman),
            "mae": aggregate.get("mae"),
            "r2": aggregate.get("r2"),
            "pearson": aggregate.get("pearson"),
            "stable_positive_spearman_folds": _as_dict(row.get("summary")).get(
                "stable_positive_spearman_folds"
            ),
            "folds": _as_dict(row.get("summary")).get("folds"),
            "permutation": row.get("permutation"),
            "real_minus_permutation_spearman": margin,
        }
        if best is None or _candidate_score(candidate) > _candidate_score(best):
            best = candidate
    return best


def _candidate_score(row: dict[str, Any]) -> tuple[float, float, int, float]:
    spearman = _float_or_neg_inf(row.get("spearman"))
    margin = _float_or_neg_inf(row.get("real_minus_permutation_spearman"))
    stable = int(row.get("stable_positive_spearman_folds") or 0)
    mae = _float_or_pos_inf(row.get("mae"))
    return (spearman, margin, stable, -mae)


def _top_rows_for_permutation(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("model") != "DummyRegressor"
        and _metric(_as_dict(row.get("summary")).get("aggregate", {}), "spearman") is not None
    ]
    candidates.sort(
        key=lambda row: (
            _metric(_as_dict(row.get("summary")).get("aggregate", {}), "spearman") or float("-inf")
        ),
        reverse=True,
    )
    return candidates[:limit]


def _high_orientation_signal_detected(
    best: dict[str, Any] | None,
    matched_control: dict[str, Any],
    special_sensitivity: dict[str, Any],
) -> bool:
    if not best:
        return False
    spearman = _metric(best, "spearman")
    margin = best.get("real_minus_permutation_spearman")
    matched_margin = _as_dict(matched_control.get("improvement_margin")).get(
        "high_minus_control_p95_spearman"
    )
    stable = int(best.get("stable_positive_spearman_folds") or 0)
    special_ok = not bool(
        _sensitivity_answer(special_sensitivity, "high_orientation_exclude_5680").get(
            "dependency_flag"
        )
    )
    return bool(
        spearman is not None
        and spearman > 0.0
        and margin is not None
        and float(margin) > 0.05
        and matched_margin is not None
        and float(matched_margin) > 0.0
        and stable >= 2
        and special_ok
    )


def _best_is_permutation_safe(best: dict[str, Any] | None) -> bool:
    if not best:
        return False
    spearman = _metric(best, "spearman")
    margin = best.get("real_minus_permutation_spearman")
    stable = int(best.get("stable_positive_spearman_folds") or 0)
    return bool(
        spearman is not None
        and spearman > 0.0
        and margin is not None
        and float(margin) > 0.05
        and stable >= 2
    )


def _orientation_depth_regime_confounded(
    matched_control: dict[str, Any],
    regime_study: dict[str, Any],
) -> bool:
    support_warnings = set(str(item) for item in matched_control.get("support_warnings", []))
    domain_warnings = set(str(item) for item in regime_study.get("domain_shift_warnings", []))
    return bool(
        "matched_controls_dominated_by_high_orientation_samples" in support_warnings
        or "regime_a_high_orientation_zero_support" in domain_warnings
        or "regime_a_high_orientation_too_few_samples" in domain_warnings
    )


def _regime_dependent(regime_study: dict[str, Any]) -> bool:
    warnings = regime_study.get("domain_shift_warnings", [])
    return bool(warnings)


def _target_sensitive(target_study: dict[str, Any]) -> bool:
    target_views = _as_dict(target_study.get("target_views"))
    values = [
        _metric(_as_dict(_as_dict(row).get("summary")).get("aggregate", {}), "spearman")
        for row in target_views.values()
    ]
    finite = [value for value in values if value is not None]
    return bool(finite and (max(finite) - min(finite)) > 0.20)


def _label_redesign_needed(
    best: dict[str, Any] | None,
    matched_control: dict[str, Any],
    high_orientation: dict[str, Any],
) -> bool:
    if best is None:
        return True
    high_s = _metric(best, "spearman")
    high_minus_perm = _as_dict(matched_control.get("improvement_margin")).get(
        "high_minus_permutation_mean_spearman"
    )
    positive_rows = [
        row
        for row in high_orientation.get("rows", [])
        if (_metric(_as_dict(row.get("summary")).get("aggregate", {}), "spearman") or 0.0) > 0.0
    ]
    return bool(
        (high_s is None or high_s <= 0.0)
        and (high_minus_perm is None or high_minus_perm <= 0.0)
        and not positive_rows
    )


def _domain_shift_warnings(
    within_regime: dict[str, Any],
    leave_one: dict[str, Any],
    transfer: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    regime_a = _as_dict(within_regime.get("A"))
    if int(regime_a.get("sample_count") or 0) == 0:
        warnings.append("regime_a_high_orientation_zero_support")
    elif regime_a.get("status") == "skipped_too_few_samples":
        warnings.append("regime_a_high_orientation_too_few_samples")
    within_s = [
        _metric(_as_dict(row).get("aggregate", {}), "spearman")
        for row in within_regime.values()
        if _as_dict(row).get("status") == "completed"
    ]
    within_finite = [value for value in within_s if value is not None]
    if within_finite and (max(within_finite) - min(within_finite)) > 0.20:
        warnings.append("within_regime_spearman_spread_gt_0.20")
    leave_s = [
        _metric(row, "spearman")
        for row in leave_one.values()
        if _as_dict(row).get("status") == "completed"
    ]
    leave_finite = [value for value in leave_s if value is not None]
    if leave_finite and min(leave_finite) < 0.0:
        warnings.append("leave_one_regime_negative_spearman")
    transfer_s = [
        _metric(row, "spearman")
        for row in transfer.values()
        if _as_dict(row).get("status") == "completed"
    ]
    transfer_finite = [value for value in transfer_s if value is not None]
    if transfer_finite and min(transfer_finite) < 0.0:
        warnings.append("cross_regime_transfer_negative_spearman")
    return warnings


def _target_view_recommendation(target_views: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for target, row in target_views.items():
        summary = _as_dict(row.get("summary") if "summary" in row else row)
        aggregate = _as_dict(summary.get("aggregate"))
        spearman = aggregate.get("spearman")
        if spearman is None:
            continue
        candidates.append(
            {
                "target": target,
                "target_policy": row.get("target_policy"),
                "spearman": float(spearman),
                "mae": aggregate.get("mae"),
                "stable_positive_spearman_folds": summary.get("stable_positive_spearman_folds"),
            }
        )
    candidates.sort(
        key=lambda row: (
            _float_or_neg_inf(row.get("spearman")),
            int(row.get("stable_positive_spearman_folds") or 0),
            -_float_or_pos_inf(row.get("mae")),
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    return {
        "recommended_for_human_review": None if best is None else best["target"],
        "best": best,
        "note": (
            "receiver_max/full_360_fraction/receiver_std remain audit or auxiliary views; "
            "this does not change label semantics."
        ),
    }


def _target_policy(target: str) -> str:
    return {
        "receiver_mean": "conservative_reference",
        "receiver_p90": "robust_candidate",
        "receiver_max": "sensitive_audit_only",
        "full_360_fraction": "auxiliary_coverage_only",
        "receiver_std": "heterogeneity_audit_only",
    }.get(target, "unknown")


def _permutation_safe_answer(
    best: dict[str, Any] | None,
    matched_control: dict[str, Any],
) -> dict[str, Any]:
    if not best:
        return {"status": "no_best_result", "permutation_safe": False}
    margin = best.get("real_minus_permutation_spearman")
    matched_margin = _as_dict(matched_control.get("improvement_margin")).get(
        "high_minus_permutation_mean_spearman"
    )
    return {
        "status": "evaluated",
        "best_real_minus_permutation_spearman": margin,
        "primary_candidate_high_minus_permutation_mean_spearman": matched_margin,
        "permutation_safe": bool(margin is not None and float(margin) > 0.05),
    }


def _high_orientation_only_answer(
    all_sample_baseline: dict[str, Any],
    best: dict[str, Any] | None,
    special_sensitivity: dict[str, Any],
) -> dict[str, Any]:
    all_best = _as_dict(all_sample_baseline.get("enhanced_best"))
    all_s = all_best.get("spearman")
    high_s = None if not best else best.get("spearman")
    low_row = _as_dict(_as_dict(special_sensitivity.get("filters")).get("low_orientation"))
    low_s = _metric(_as_dict(low_row).get("aggregate", {}), "spearman")
    return {
        "all_sample_best_spearman": all_s,
        "high_orientation_best_spearman": high_s,
        "low_orientation_spearman_same_best": low_s,
        "high_minus_all_spearman": _none_subtract(high_s, all_s),
        "high_minus_low_spearman": _none_subtract(high_s, low_s),
    }


def _matched_answer(matched_control: dict[str, Any]) -> dict[str, Any]:
    margins = _as_dict(matched_control.get("improvement_margin"))
    return {
        "high_minus_control_mean_spearman": margins.get("high_minus_control_mean_spearman"),
        "high_minus_control_p95_spearman": margins.get("high_minus_control_p95_spearman"),
        "exceeds_matched_control_p95": bool(
            margins.get("high_minus_control_p95_spearman") is not None
            and float(margins["high_minus_control_p95_spearman"]) > 0.0
        ),
    }


def _sensitivity_answer(study: dict[str, Any], filter_name: str) -> dict[str, Any]:
    rows = _as_dict(study.get("filters"))
    high = _as_dict(rows.get("high_orientation_only"))
    filt = _as_dict(rows.get(filter_name))
    high_s = _metric(_as_dict(high.get("aggregate")), "spearman")
    filt_s = _metric(_as_dict(filt.get("aggregate")), "spearman")
    delta = _none_subtract(filt_s, high_s)
    return {
        "filter": filter_name,
        "high_orientation_spearman": high_s,
        "filtered_spearman": filt_s,
        "filtered_minus_high_spearman": delta,
        "dependency_flag": bool(delta is not None and abs(float(delta)) > 0.10),
    }


def _single_regime_answer(regime_study: dict[str, Any]) -> dict[str, Any]:
    rows = _as_dict(regime_study.get("within_regime_cv"))
    values = {
        regime: _metric(_as_dict(row).get("aggregate", {}), "spearman")
        for regime, row in rows.items()
    }
    finite = {regime: value for regime, value in values.items() if value is not None}
    if not finite:
        return {"status": "not_evaluable", "depends_on_single_regime": False}
    positive = [regime for regime, value in finite.items() if value > 0.0]
    return {
        "within_regime_spearman": finite,
        "positive_regimes": positive,
        "depends_on_single_regime": len(positive) == 1,
    }


def _weakest_regime_answer(regime_study: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for regime, row in _as_dict(regime_study.get("within_regime_cv")).items():
        aggregate = _as_dict(_as_dict(row).get("aggregate"))
        if aggregate.get("mae") is None:
            continue
        rows.append(
            {
                "regime": regime,
                "mae": aggregate.get("mae"),
                "spearman": aggregate.get("spearman"),
                "sample_count": _as_dict(row).get("sample_count"),
            }
        )
    rows.sort(key=lambda row: float(row["mae"]), reverse=True)
    return rows


def _compact_subgroup_counts(subgroup_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "sample_count": row.get("sample_count"),
            "sample_fraction": row.get("sample_fraction"),
            "depth_range": row.get("depth_range"),
        }
        for name, row in _as_dict(subgroup_audit.get("groups")).items()
    }


def _next_minimal_recommendation(decision: str) -> str:
    if decision == "exploratory_high_orientation_signal_detected":
        return (
            "Ask for human approval before treating high-orientation as a formal cohort "
            "policy; continue bounded classical validation only."
        )
    if decision == "exploratory_regime_dependent_signal_detected":
        return (
            "Ask for human approval to design a formal stratified study; do not train "
            "regime-specific models yet."
        )
    if decision == "exploratory_target_view_sensitive_request_human_decision":
        return (
            "Ask humans to choose whether receiver_mean, receiver_p90, or audit views "
            "should guide the next weak-label study."
        )
    if decision == "request_label_redesign_approval":
        return (
            "Return to weak-label and morphology review before adding higher-cost signal "
            "processing."
        )
    return (
        "Stop this autonomous loop; do not escalate to STC, APES, deep learning, "
        "final labels, or production claims."
    )


def _matched_support_warnings(high_mask: np.ndarray, controls: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if np.count_nonzero(high_mask) < 100:
        warnings.append("high_orientation_support_below_100")
    fractions = [float(row.get("high_orientation_fraction") or 0.0) for row in controls]
    if fractions and float(np.mean(fractions)) > 0.80:
        warnings.append("matched_controls_dominated_by_high_orientation_samples")
    return warnings


def _morphology_summary(snapshot: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    fields = [
        key
        for key in snapshot
        if key.startswith("morphology_")
        and np.issubdtype(np.asarray(snapshot[key]).dtype, np.number)
    ]
    return {field: _summary_or_empty(np.asarray(snapshot[field])[mask]) for field in fields}


def _finite_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    return float(np.isfinite(values).mean())


def _summary_or_empty(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0, "finite_ratio": 0.0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "finite_ratio": float(finite.size / max(array.size, 1)),
        "min": float(np.min(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _calibration_by_target_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantile_count: int = 5,
) -> list[dict[str, Any]]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    if np.count_nonzero(mask) < quantile_count:
        return []
    true = true[mask]
    pred = pred[mask]
    order = np.argsort(true)
    rows = []
    for index, indices in enumerate(np.array_split(order, quantile_count)):
        rows.append(
            {
                "quantile_bin": index,
                "count": int(indices.size),
                "target_mean": float(np.mean(true[indices])),
                "prediction_mean": float(np.mean(pred[indices])),
                "bias": float(np.mean(pred[indices] - true[indices])),
                "mae": float(np.mean(np.abs(pred[indices] - true[indices]))),
            }
        )
    return rows


def _write_plots(
    *,
    review_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    overwrite: bool,
) -> dict[str, str]:
    files = {
        "predicted_vs_target_png": review_dir / "predicted_vs_target_high_orientation.png",
        "residual_vs_depth_png": review_dir / "residual_vs_depth_high_orientation.png",
        "calibration_png": review_dir / "calibration_high_orientation.png",
    }
    for path in files.values():
        _ensure_can_write(path, overwrite=overwrite)
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/cement_channel_matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"plot_status": "skipped_matplotlib_unavailable"}
    mask = sample_mask & np.isfinite(y_pred)
    true = y_true[mask]
    pred = y_pred[mask]
    depth_values = depth[mask]
    residual = pred - true

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true, pred, s=7, alpha=0.35)
    bounds = [
        float(np.min(np.concatenate([true, pred]))),
        float(np.max(np.concatenate([true, pred]))),
    ]
    ax.plot(bounds, bounds, color="black", linewidth=1)
    ax.set_xlabel("weak label target")
    ax.set_ylabel("prediction")
    ax.set_title("High-orientation predicted vs target")
    fig.tight_layout()
    fig.savefig(files["predicted_vs_target_png"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(depth_values, residual, s=7, alpha=0.35)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("depth ft")
    ax.set_ylabel("residual")
    ax.set_title("High-orientation residual vs depth")
    fig.tight_layout()
    fig.savefig(files["residual_vs_depth_png"], dpi=150)
    plt.close(fig)

    calibration = _calibration_by_target_quantile(true, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [row["target_mean"] for row in calibration],
        [row["prediction_mean"] for row in calibration],
        marker="o",
    )
    ax.set_xlabel("target quantile mean")
    ax.set_ylabel("prediction mean")
    ax.set_title("High-orientation calibration")
    fig.tight_layout()
    fig.savefig(files["calibration_png"], dpi=150)
    plt.close(fig)
    return {name: str(path) for name, path in files.items()}


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Required NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_yaml(path: Path | str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def _read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return default


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = _as_dict(row).get(key)
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def _none_subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        output = float(left) - float(right)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def _float_or_neg_inf(value: Any) -> float:
    metric = _metric({"value": value}, "value")
    return float("-inf") if metric is None else metric


def _float_or_pos_inf(value: Any) -> float:
    metric = _metric({"value": value}, "value")
    return float("inf") if metric is None else metric
