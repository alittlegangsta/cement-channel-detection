from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_autonomous import evaluate_model_cv, run_target_permutations
from cement_channel.modeling.mvp4x_baselines import compute_regression_metrics
from cement_channel.modeling.mvp4x_confounding import (
    evaluate_blocked_gap_cv,
    run_structured_permutations,
)
from cement_channel.modeling.mvp4x_model_analysis import build_feature_sets
from cement_channel.modeling.mvp4x_stratified_baselines import _feature_matrix

REPORT_VERSION = "mvp4x_regime_robustness_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
DEFAULT_COHORTS = (
    "regime_b_all",
    "regime_b_high_orientation",
    "regime_c_all",
    "regime_c_high_orientation",
    "pooled_bc_all",
    "pooled_bc_high_orientation",
)
STRATIFIED_COHORT_ALIAS = {
    "pooled_bc_all": "regime_bc_all",
    "pooled_bc_high_orientation": "regime_bc_high_orientation",
}


class RegimeRobustnessError(RuntimeError):
    """Raised when MVP-4X regime robustness cannot run safely."""


@dataclass(frozen=True)
class RegimeRobustnessOutputs:
    report: dict[str, Any]
    csv_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_regime_robustness_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    policy_npz: Path | str,
    stratified_baselines_json: Path | str,
    config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> RegimeRobustnessOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    policy_npz_data = _load_npz(Path(policy_npz))
    stratified_baselines = _read_json(Path(stratified_baselines_json))
    config = _load_yaml(Path(config_path))
    outputs = run_regime_robustness(
        snapshot=snapshot,
        waveform=waveform,
        policy_npz=policy_npz_data,
        stratified_baselines=stratified_baselines,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "policy_npz": str(policy_npz),
            "stratified_baselines_json": str(stratified_baselines_json),
            "config_path": str(config_path),
        },
    )
    write_regime_robustness_outputs(
        outputs,
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return outputs


def run_regime_robustness(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    stratified_baselines: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
) -> RegimeRobustnessOutputs:
    _validate_artifacts(snapshot, waveform, policy_npz, stratified_baselines)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    robustness_config = _as_dict(config.get("robustness"))
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    policy = load_policy(policy_npz)
    feature_sets = build_feature_sets(snapshot, waveform)
    candidates = select_robustness_candidates(
        stratified_baselines=stratified_baselines,
        requested_cohorts=_as_list(robustness_config.get("candidate_cohorts"), DEFAULT_COHORTS),
    )
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for cohort, candidate in candidates.items():
        print(
            "MVP-4X regime robustness "
            f"cohort={cohort} feature_set={candidate['feature_set']} "
            f"target={candidate['target']} model={candidate['model']}",
            flush=True,
        )
        result = evaluate_candidate_robustness(
            snapshot=snapshot,
            feature_sets=feature_sets,
            depth=depth,
            masks=policy["masks"],
            cohort=cohort,
            candidate=candidate,
            config=robustness_config,
            sklearn_modules=sklearn_modules,
        )
        rows.append(result)
        csv_rows.append(_csv_row(result))
    screening_support = classify_screening_support(rows)
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "modeling_environment": modeling_environment.to_dict(),
        "candidate_count": len(rows),
        "candidate_rows": rows,
        "screening_support": screening_support,
        "validation_protocol": {
            "repeated_seeds": _as_int_list(
                robustness_config.get("repeated_seeds"),
                [11, 23, 37, 53, 71],
            ),
            "blocked_gap_ft": _as_float_list(
                robustness_config.get("blocked_gap_ft"),
                [10.0, 25.0, 50.0],
            ),
            "permutation_count": int(robustness_config.get("permutation_count", 20)),
            "bootstrap_repeats": int(robustness_config.get("bootstrap_repeats", 200)),
            "train_fold_only_preprocessing": True,
            "leakage_checks": {
                "depth_not_model_input": True,
                "regime_id_not_model_input": True,
                "orientation_confidence_not_model_input": True,
                "cast_derived_fields_not_model_input": True,
            },
        },
        **_method_flags(),
    }
    return RegimeRobustnessOutputs(report=report, csv_rows=csv_rows)


def load_policy(policy_npz: dict[str, np.ndarray]) -> dict[str, Any]:
    names = np.asarray(policy_npz["cohort_names"]).astype(str).tolist()
    masks = np.asarray(policy_npz["cohort_masks"], dtype=bool)
    if masks.ndim != 2 or masks.shape[0] != len(names):
        raise RegimeRobustnessError("policy cohort masks must have shape [cohort, sample].")
    metadata = json.loads(str(np.asarray(policy_npz["metadata_json"]).item()))
    return {
        "cohort_names": names,
        "masks": {name: masks[index] for index, name in enumerate(names)},
        "metadata": metadata,
    }


def select_robustness_candidates(
    *,
    stratified_baselines: dict[str, Any],
    requested_cohorts: list[str],
) -> dict[str, dict[str, Any]]:
    best_by_cohort = _as_dict(
        _as_dict(stratified_baselines.get("model_matrix")).get("best_by_cohort")
    )
    selected: dict[str, dict[str, Any]] = {}
    for cohort in requested_cohorts:
        source = STRATIFIED_COHORT_ALIAS.get(cohort, cohort)
        best = _as_dict(best_by_cohort.get(source))
        if not best:
            continue
        selected[cohort] = {
            "cohort": cohort,
            "source_cohort": source,
            "feature_set": best["feature_set"],
            "target": best["target"],
            "model": best["model"],
            "prior_spearman": best.get("spearman"),
            "prior_mae": best.get("mae"),
            "prior_r2": best.get("r2"),
            "prior_stable_positive_spearman_folds": best.get("stable_positive_spearman_folds"),
        }
    return selected


def evaluate_candidate_robustness(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    cohort: str,
    candidate: dict[str, Any],
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    seeds = _as_int_list(config.get("repeated_seeds"), [11, 23, 37, 53, 71])
    gap_values = _as_float_list(config.get("blocked_gap_ft"), [10.0, 25.0, 50.0])
    X = _feature_matrix(feature_sets[candidate["feature_set"]])
    y = np.asarray(snapshot[candidate["target"]], dtype=np.float32).reshape(-1)
    base_mask = masks[cohort] & np.isfinite(y)
    seed_rows = []
    oof_by_seed = []
    for seed in seeds:
        cv = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=base_mask,
            model_name=candidate["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=int(seed),
        )
        seed_rows.append({"seed": int(seed), "summary": cv["summary"]})
        oof_by_seed.append(np.asarray(cv["oof_prediction"], dtype=np.float32))
    repeated = summarize_repeated_cv(seed_rows)
    first_oof = oof_by_seed[0]
    bootstrap = bootstrap_metric_ci(
        y_true=y[base_mask],
        y_pred=first_oof[base_mask],
        repeats=int(config.get("bootstrap_repeats", 200)),
        seed=int(seeds[0]) + 700,
        ci_quantiles=_as_float_list(config.get("bootstrap_ci"), [0.025, 0.975]),
    )
    global_perm = run_target_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=base_mask,
        model_name=candidate["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=int(config.get("permutation_count", 20)),
        seed=int(seeds[0]) + 800,
    )
    within_perm = run_structured_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=base_mask,
        model_name=candidate["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=int(config.get("permutation_count", 20)),
        seed=int(seeds[0]) + 900,
        method="within_depth_bin",
    )
    block_perm = run_structured_permutations(
        X=X,
        y=y,
        depth=depth,
        sample_mask=base_mask,
        model_name=candidate["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        permutation_count=int(config.get("permutation_count", 20)),
        seed=int(seeds[0]) + 1000,
        method="block",
    )
    gap_rows = {
        f"gap_{float(gap):g}_ft": summarize_repeated_gap(
            X=X,
            y=y,
            depth=depth,
            sample_mask=base_mask,
            model_name=candidate["model"],
            sklearn_modules=sklearn_modules,
            config=config,
            gap_ft=float(gap),
            seeds=seeds,
        )
        for gap in gap_values
    }
    special = evaluate_special_sensitivity(
        X=X,
        y=y,
        depth=depth,
        base_mask=base_mask,
        snapshot=snapshot,
        model_name=candidate["model"],
        sklearn_modules=sklearn_modules,
        config=config,
        seed=int(seeds[0]) + 1100,
    )
    spearman_mean = repeated["metrics"]["spearman"]["mean"]
    return {
        "status": "completed",
        "cohort": cohort,
        "candidate": candidate,
        "sample_count": int(np.count_nonzero(base_mask)),
        "target_distribution": _summary(y[base_mask]),
        "repeated_cv": repeated,
        "bootstrap_ci": bootstrap,
        "permutation": {
            "global": global_perm,
            "within_depth_bin": within_perm,
            "block": block_perm,
            "margins": {
                "global": _subtract(spearman_mean, global_perm.get("spearman_mean")),
                "within_depth_bin": _subtract(spearman_mean, within_perm.get("spearman_mean")),
                "block": _subtract(spearman_mean, block_perm.get("spearman_mean")),
            },
        },
        "blocked_gap": gap_rows,
        "special_band_sensitivity": special,
        "fold_sign_consistency": repeated["fold_sign_consistency"],
        "folds_above_permutation": folds_above_permutation(
            seed_rows,
            global_perm.get("spearman_mean"),
        ),
        "seeds_above_permutation": seeds_above_permutation(
            seed_rows,
            global_perm.get("spearman_p95"),
        ),
        "support_warnings": support_warnings(cohort, base_mask),
        "degeneracy_warnings": degeneracy_warnings(seed_rows),
        "calibration_drift": calibration_drift(seed_rows),
        "residual_drift": residual_drift(y[base_mask], first_oof[base_mask]),
        **_method_flags(),
    }


def summarize_repeated_cv(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        name: summarize_values(
            [
                _metric(_as_dict(_as_dict(row["summary"]).get("aggregate")), name)
                for row in seed_rows
            ]
        )
        for name in ("spearman", "pearson", "mae", "rmse", "r2", "median_absolute_error")
    }
    fold_count = sum(int(_as_dict(row["summary"]).get("fold_count") or 0) for row in seed_rows)
    positive = sum(
        int(_as_dict(row["summary"]).get("stable_positive_spearman_folds") or 0)
        for row in seed_rows
    )
    return {
        "seed_count": len(seed_rows),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "metrics": metrics,
        "fold_sign_consistency": {
            "positive_spearman_folds": positive,
            "total_folds": fold_count,
            "fraction": None if fold_count == 0 else float(positive / fold_count),
        },
        "seed_rows": seed_rows,
    }


def summarize_repeated_gap(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    gap_ft: float,
    seeds: list[int],
) -> dict[str, Any]:
    rows = [
        {
            "seed": int(seed),
            "summary": evaluate_blocked_gap_cv(
                X=X,
                y=y,
                depth=depth,
                sample_mask=sample_mask,
                model_name=model_name,
                sklearn_modules=sklearn_modules,
                config=config,
                gap_ft=gap_ft,
                rng_seed=int(seed) + int(gap_ft * 10),
            ),
        }
        for seed in seeds
    ]
    metrics = {
        name: summarize_values(
            [_metric(_as_dict(_as_dict(row["summary"]).get("aggregate")), name) for row in rows]
        )
        for name in ("spearman", "mae", "rmse", "r2", "median_absolute_error")
    }
    return {
        "gap_ft": float(gap_ft),
        "seed_count": len(rows),
        "metrics": metrics,
        "rows": rows,
    }


def bootstrap_metric_ci(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    repeats: int,
    seed: int,
    ci_quantiles: list[float],
) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true[mask], dtype=np.float32)
    y_pred = np.asarray(y_pred[mask], dtype=np.float32)
    if y_true.size == 0:
        return {"status": "skipped_no_samples"}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        key: [] for key in ("spearman", "mae", "rmse", "r2", "median_absolute_error")
    }
    for _ in range(max(1, repeats)):
        idx = rng.integers(0, y_true.size, size=y_true.size)
        metrics = compute_regression_metrics(y_true[idx], y_pred[idx])
        for key in values:
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                values[key].append(float(value))
    lower_q, upper_q = ci_quantiles
    return {
        "status": "completed",
        "sample_count": int(y_true.size),
        "repeats": int(max(1, repeats)),
        "ci_quantiles": [float(lower_q), float(upper_q)],
        "metrics": {
            key: {
                "mean": None if not vals else float(np.mean(vals)),
                "std": None if not vals else float(np.std(vals)),
                "ci_lower": None if not vals else float(np.quantile(vals, lower_q)),
                "ci_upper": None if not vals else float(np.quantile(vals, upper_q)),
            }
            for key, vals in values.items()
        },
    }


def evaluate_special_sensitivity(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    base_mask: np.ndarray,
    snapshot: dict[str, np.ndarray],
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    filters = {
        "include_all": base_mask,
        "exclude_2400_2500": base_mask
        & ~np.asarray(snapshot["saturation_platform_flag"], dtype=bool),
        "exclude_5680": base_mask & ~np.asarray(snapshot["special_5680_flag"], dtype=bool),
        "exclude_all_special": base_mask & ~np.asarray(snapshot["any_special_flag"], dtype=bool),
    }
    rows = {}
    for name, mask in filters.items():
        if np.count_nonzero(mask) < 30:
            rows[name] = {
                "status": "skipped_low_support",
                "sample_count": int(np.count_nonzero(mask)),
            }
            continue
        result = evaluate_model_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=mask & np.isfinite(y),
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=seed + len(rows),
        )
        rows[name] = result["summary"]
    base = _metric(_as_dict(rows.get("include_all", {})).get("aggregate", {}), "spearman")
    deltas = {
        name: _subtract(_metric(_as_dict(row).get("aggregate", {}), "spearman"), base)
        for name, row in rows.items()
        if name != "include_all"
    }
    return {
        "status": "completed",
        "rows": rows,
        "spearman_deltas": deltas,
        "dependency_flag": any(abs(float(value or 0.0)) > 0.05 for value in deltas.values()),
    }


def classify_screening_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = []
    unsupported = []
    for row in rows:
        metrics = row["repeated_cv"]["metrics"]
        spearman = metrics["spearman"]["mean"]
        ci_lower = row["bootstrap_ci"]["metrics"]["spearman"]["ci_lower"]
        global_margin = row["permutation"]["margins"]["global"]
        within_margin = row["permutation"]["margins"]["within_depth_bin"]
        fold_fraction = row["fold_sign_consistency"]["fraction"]
        special_dep = row["special_band_sensitivity"]["dependency_flag"]
        ok = (
            spearman is not None
            and spearman > 0.15
            and ci_lower is not None
            and ci_lower > 0.0
            and global_margin is not None
            and global_margin > 0.05
            and within_margin is not None
            and within_margin > 0.0
            and fold_fraction is not None
            and fold_fraction >= 2 / 3
            and not special_dep
        )
        (supported if ok else unsupported).append(row["cohort"])
    return {
        "supported_screening_cohorts": supported,
        "unsupported_screening_cohorts": unsupported,
        "research_only_screening": True,
        "not_absolute_fraction_prediction": True,
    }


def write_regime_robustness_outputs(
    outputs: RegimeRobustnessOutputs,
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    for path in (output_report_md, output_report_json, output_csv):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(_json(outputs.report), encoding="utf-8")
    output_report_md.write_text(format_robustness_markdown(outputs.report), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(outputs.csv_rows[0].keys()) if outputs.csv_rows else ["status"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs.csv_rows)


def format_robustness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Regime Robustness",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "no_final_labels, no_ground_truth_claim, no_production_claim.",
        "",
        f"- report_version: `{report['report_version']}`",
        f"- candidate_count: {report['candidate_count']}",
        "",
        "## Candidate Summary",
    ]
    for row in report["candidate_rows"]:
        metrics = row["repeated_cv"]["metrics"]
        lines.append(
            f"- {row['cohort']}: model={row['candidate']['model']}, "
            f"feature_set={row['candidate']['feature_set']}, "
            f"target={row['candidate']['target']}, "
            f"spearman_mean={metrics['spearman']['mean']}, "
            f"bootstrap_spearman_ci={row['bootstrap_ci']['metrics']['spearman']}"
        )
    lines.extend(
        [
            "",
            "## Screening Support",
            f"- supported: {report['screening_support']['supported_screening_cohorts']}",
            f"- unsupported: {report['screening_support']['unsupported_screening_cohorts']}",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_values(values: list[Any]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def folds_above_permutation(
    seed_rows: list[dict[str, Any]],
    permutation_mean: Any,
) -> dict[str, Any]:
    if permutation_mean is None:
        return {"count": 0, "total": 0, "fraction": None}
    total = 0
    count = 0
    for row in seed_rows:
        for fold in _as_list(_as_dict(row["summary"]).get("folds"), []):
            value = _metric(fold, "spearman")
            if value is None:
                continue
            total += 1
            count += int(float(value) > float(permutation_mean))
    return {"count": count, "total": total, "fraction": None if total == 0 else count / total}


def seeds_above_permutation(
    seed_rows: list[dict[str, Any]],
    permutation_p95: Any,
) -> dict[str, Any]:
    if permutation_p95 is None:
        return {"count": 0, "total": len(seed_rows), "fraction": None}
    count = 0
    for row in seed_rows:
        value = _metric(_as_dict(_as_dict(row["summary"]).get("aggregate")), "spearman")
        count += int(value is not None and float(value) > float(permutation_p95))
    return {"count": count, "total": len(seed_rows), "fraction": count / max(1, len(seed_rows))}


def calibration_drift(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_abs_biases = []
    for row in seed_rows:
        bins = _as_list(_as_dict(row["summary"]).get("calibration_by_target_quantile"), [])
        biases = [abs(float(item["bias"])) for item in bins if item.get("bias") is not None]
        if biases:
            max_abs_biases.append(max(biases))
    return {"max_abs_quantile_bias": summarize_values(max_abs_biases)}


def residual_drift(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    residual = np.asarray(y_pred[mask] - y_true[mask], dtype=np.float32)
    if residual.size == 0:
        return {"status": "skipped_no_samples"}
    return {
        "status": "completed",
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_p01": float(np.quantile(residual, 0.01)),
        "residual_p99": float(np.quantile(residual, 0.99)),
    }


def degeneracy_warnings(seed_rows: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for row in seed_rows:
        pred = _as_dict(_as_dict(row["summary"]).get("prediction_summary"))
        if pred.get("std") is not None and float(pred["std"]) < 1e-6:
            warnings.append(f"seed_{row['seed']}_near_constant_prediction")
    return warnings


def support_warnings(cohort: str, mask: np.ndarray) -> list[str]:
    warnings = []
    count = int(np.count_nonzero(mask))
    if count < 100:
        warnings.append("low_sample_support")
    if cohort == "regime_b_high_orientation" and count < 1000:
        warnings.append("regime_b_high_orientation_support_is_limited")
    return warnings


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["repeated_cv"]["metrics"]
    bootstrap = row["bootstrap_ci"]["metrics"]
    return {
        "cohort": row["cohort"],
        "feature_set": row["candidate"]["feature_set"],
        "target": row["candidate"]["target"],
        "model": row["candidate"]["model"],
        "sample_count": row["sample_count"],
        "spearman_mean": metrics["spearman"]["mean"],
        "spearman_std": metrics["spearman"]["std"],
        "spearman_ci_lower": bootstrap["spearman"]["ci_lower"],
        "spearman_ci_upper": bootstrap["spearman"]["ci_upper"],
        "mae_mean": metrics["mae"]["mean"],
        "rmse_mean": metrics["rmse"]["mean"],
        "r2_mean": metrics["r2"]["mean"],
        "global_permutation_margin": row["permutation"]["margins"]["global"],
        "within_depth_bin_permutation_margin": row["permutation"]["margins"]["within_depth_bin"],
        "block_permutation_margin": row["permutation"]["margins"]["block"],
        "fold_sign_fraction": row["fold_sign_consistency"]["fraction"],
        "seeds_above_permutation_fraction": row["seeds_above_permutation"]["fraction"],
        "special_dependency_flag": row["special_band_sensitivity"]["dependency_flag"],
    }


def _validate_artifacts(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    stratified_baselines: dict[str, Any],
) -> None:
    if snapshot["depth"].shape != (7108,):
        raise RegimeRobustnessError(f"Unexpected sample shape: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise RegimeRobustnessError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise RegimeRobustnessError(
            f"Unexpected waveform feature shape: {waveform['waveform_depth_features'].shape}"
        )
    if snapshot["xsi_features"].shape[1] + waveform["waveform_depth_features"].shape[1] != 422:
        raise RegimeRobustnessError("Combined feature count is not 422.")
    if _finite_ratio(snapshot["xsi_features"]) != 1.0:
        raise RegimeRobustnessError("Existing feature finite ratio is not 1.0.")
    if _finite_ratio(waveform["waveform_depth_features"]) != 1.0:
        raise RegimeRobustnessError("Waveform feature finite ratio is not 1.0.")
    if str(snapshot["target_kernel"]) != "triangular_midpoint_weighted":
        raise RegimeRobustnessError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for flag in RESEARCH_FLAGS:
        if not bool(np.asarray(snapshot.get(flag, False)).item()):
            raise RegimeRobustnessError(f"snapshot missing research flag {flag}.")
        if not bool(np.asarray(waveform.get(flag, False)).item()):
            raise RegimeRobustnessError(f"waveform missing research flag {flag}.")
        if not bool(np.asarray(policy_npz.get(flag, False)).item()):
            raise RegimeRobustnessError(f"policy missing research flag {flag}.")
        if stratified_baselines.get(flag) is not True:
            raise RegimeRobustnessError(f"stratified baselines missing research flag {flag}.")
    if "model_matrix" not in stratified_baselines:
        raise RegimeRobustnessError("stratified baseline report missing model_matrix.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise RegimeRobustnessError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegimeRobustnessError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RegimeRobustnessError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegimeRobustnessError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    return [int(item) for item in _as_list(value, default)]


def _as_float_list(value: Any, default: list[float]) -> list[float]:
    return [float(item) for item in _as_list(value, default)]


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p50": float(np.quantile(finite, 0.5)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _subtract(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
