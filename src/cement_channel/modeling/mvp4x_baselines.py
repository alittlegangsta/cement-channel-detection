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

BASELINE_REPORT_VERSION = "mvp4x_existing_feature_baselines_v001"
BASELINE_CSV_VERSION = "mvp4x_existing_feature_baselines_csv_v001"
MODEL_NAMES = (
    "DummyRegressor",
    "Ridge",
    "ElasticNet",
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
)
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
TARGET_DEFAULTS = ("receiver_p90", "receiver_mean", "receiver_max")
SENSITIVITY_FILTERS = (
    "include_all",
    "exclude_saturation_platform",
    "exclude_5680_special_band",
    "exclude_low_orientation_confidence",
    "exclude_all_special_flags",
)


@dataclass(frozen=True)
class Mvp4xBaselineReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    feature_set_name: str
    allowed_scope: str
    model_backend: str
    sklearn_available: bool
    modeling_environment: dict[str, Any]
    model_random_seed: int
    cv_protocol: dict[str, Any]
    permutation_protocol: dict[str, Any]
    target_views: list[str]
    sample_count: int
    feature_count: int
    model_feature_names: list[str]
    model_summaries: dict[str, dict[str, Any]]
    target_summaries: dict[str, dict[str, Any]]
    contiguous_cv: dict[str, Any]
    leave_one_regime_out: dict[str, Any]
    regime_transfer: dict[str, Any]
    sensitivity: dict[str, Any]
    permutation: dict[str, Any]
    derived_binary_audit: dict[str, Any]
    decision: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    research_only: bool
    exploratory_only: bool
    weak_label_target: bool
    no_final_labels: bool
    no_ground_truth_claim: bool
    no_production_claim: bool
    production_training: bool
    derived_binary_audit_only: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_existing_feature_baselines_from_paths(
    *,
    snapshot_npz: Path | str,
    config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> Mvp4xBaselineReport:
    config = _load_yaml(config_path)
    snapshot = _load_npz(snapshot_npz)
    report, rows = run_mvp4x_baselines(
        snapshot=snapshot,
        config=config,
        inputs={"snapshot_npz": str(snapshot_npz), "config_path": str(config_path)},
        feature_set_name="existing_features_only",
        feature_matrix_key="xsi_features",
        feature_name_key="xsi_feature_names",
    )
    write_mvp4x_baseline_outputs(
        report,
        rows,
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def run_mvp4x_baselines(
    *,
    snapshot: dict[str, np.ndarray],
    config: dict[str, Any],
    inputs: dict[str, str],
    feature_set_name: str,
    feature_matrix_key: str,
    feature_name_key: str,
) -> tuple[Mvp4xBaselineReport, list[dict[str, Any]]]:
    warnings: list[str] = []
    errors: list[str] = []
    _validate_research_flags(snapshot, errors)
    baseline_config = _as_dict(config.get("baseline"))
    random_seed = int(baseline_config.get("random_seed", 20240603))
    rng = np.random.default_rng(random_seed)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    sklearn_available = True

    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    feature_matrix = np.asarray(snapshot[feature_matrix_key], dtype=np.float32)
    feature_names = np.asarray(snapshot[feature_name_key]).astype(str)
    if feature_matrix.ndim != 2:
        raise ValueError(f"{feature_matrix_key} must have shape [sample, feature].")
    if feature_matrix.shape[0] != depth.size:
        raise ValueError("feature row count must match depth count.")
    if feature_names.size != feature_matrix.shape[1]:
        raise ValueError("feature name count must match feature column count.")
    if feature_set_name != "existing_features_only":
        model_feature_mask = np.ones(feature_names.size, dtype=bool)
    else:
        model_feature_mask = np.asarray(
            snapshot.get("model_feature_mask", np.ones(feature_names.size, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if model_feature_mask.size != feature_names.size:
            raise ValueError("model_feature_mask length must match feature names.")
    model_feature_names = feature_names[model_feature_mask].tolist()
    X = np.nan_to_num(
        feature_matrix[:, model_feature_mask],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    if not np.all(np.isfinite(X)):
        errors.append("Selected model feature matrix is not finite after fill.")

    target_views = tuple(_as_list(baseline_config.get("target_views"), TARGET_DEFAULTS))
    target_summaries = {
        target: _target_summary(np.asarray(snapshot[target], dtype=np.float32))
        for target in target_views
        if target in snapshot
    }
    missing_targets = [target for target in target_views if target not in snapshot]
    if missing_targets:
        errors.append("Missing target view(s): " + ", ".join(missing_targets))

    rows: list[dict[str, Any]] = []
    model_summaries: dict[str, dict[str, Any]] = {}
    contiguous: dict[str, Any] = {}
    leave_one: dict[str, Any] = {}
    transfer: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    permutation: dict[str, Any] = {}
    derived_binary: dict[str, Any] = {
        "derived_binary_audit_only": True,
        "thresholds": _as_float_list(
            baseline_config.get("derived_binary_thresholds"),
            [0.01, 0.05, 0.10, 0.20],
        ),
        "results": {},
    }

    if not errors:
        for target in target_views:
            y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
            base_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            target_key = f"{target}:{feature_set_name}"
            contiguous[target_key] = {}
            leave_one[target_key] = {}
            transfer[target_key] = {}
            sensitivity[target_key] = {}
            permutation[target_key] = {}
            derived_binary["results"][target_key] = {}
            for model_name in MODEL_NAMES:
                print(
                    "MVP-4X fitted baseline "
                    f"feature_set={feature_set_name} target={target} model={model_name}",
                    flush=True,
                )
                cv_result = _run_contiguous_cv(
                    X=X,
                    y=y,
                    depth=depth,
                    sample_mask=base_mask,
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    config=baseline_config,
                    rng_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
                contiguous[target_key][model_name] = cv_result["summary"]
                for row in cv_result["rows"]:
                    row["feature_set"] = feature_set_name
                    row["target"] = target
                    rows.append(row)
                model_summaries[f"{target_key}:{model_name}"] = cv_result["summary"]
                leave_one[target_key][model_name] = _run_leave_one_regime_out(
                    X=X,
                    y=y,
                    regimes=np.asarray(snapshot["broad_regime_id"]).astype(str),
                    sample_mask=base_mask,
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    config=baseline_config,
                )
                transfer[target_key][model_name] = _run_regime_transfer(
                    X=X,
                    y=y,
                    regimes=np.asarray(snapshot["broad_regime_id"]).astype(str),
                    sample_mask=base_mask,
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    config=baseline_config,
                )
                sensitivity[target_key][model_name] = _run_sensitivity_filters(
                    X=X,
                    y=y,
                    depth=depth,
                    snapshot=snapshot,
                    sample_mask=base_mask,
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    config=baseline_config,
                )
                permutation[target_key][model_name] = _run_target_permutations(
                    X=X,
                    y=y,
                    depth=depth,
                    sample_mask=base_mask,
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    config=baseline_config,
                    rng=rng,
                )
                derived_binary["results"][target_key][model_name] = _derived_binary_audit(
                    y_true=y,
                    y_pred=np.asarray(cv_result["oof_prediction"], dtype=np.float32),
                    sample_mask=base_mask,
                    thresholds=derived_binary["thresholds"],
                    sklearn_modules=sklearn_modules,
                )
    else:
        for target in target_views:
            for model_name in MODEL_NAMES:
                key = f"{target}:{feature_set_name}:{model_name}"
                model_summaries[key] = {
                    "status": "skipped_errors",
                    "reason": errors,
                }
                rows.append(
                    {
                        "csv_version": BASELINE_CSV_VERSION,
                        "feature_set": feature_set_name,
                        "target": target,
                        "model": model_name,
                        "evaluation": "skipped",
                        "fold": "",
                        "status": model_summaries[key]["status"],
                    }
                )

    decision = _baseline_decision(
        contiguous=contiguous,
        leave_one=leave_one,
        sensitivity=sensitivity,
        permutation=permutation,
        sklearn_available=sklearn_available,
        errors=errors,
        config=baseline_config,
    )
    report = Mvp4xBaselineReport(
        report_version=BASELINE_REPORT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        feature_set_name=feature_set_name,
        allowed_scope="research_only_exploratory_weak_label_regression",
        model_backend="scikit_learn",
        sklearn_available=True,
        modeling_environment=modeling_environment.to_dict(),
        model_random_seed=random_seed,
        cv_protocol={
            "method": "contiguous_depth_block_cv",
            "n_folds": int(baseline_config.get("n_contiguous_folds", 3)),
            "depth_sorted": True,
            "metadata_only_fields": [
                "depth",
                "broad_regime_id",
                "special-band flags",
                "label_confidence",
                "orientation_confidence",
                "inclination",
                "morphology arrays",
                "CAST-derived fields",
            ],
        },
        permutation_protocol={
            "enabled": True,
            "target_permutation_count": int(baseline_config.get("permutation_count", 20)),
            "seed": random_seed,
            "same_cv_protocol": True,
        },
        target_views=list(target_views),
        sample_count=int(depth.size),
        feature_count=int(X.shape[1]),
        model_feature_names=model_feature_names,
        model_summaries=model_summaries,
        target_summaries=target_summaries,
        contiguous_cv=contiguous,
        leave_one_regime_out=leave_one,
        regime_transfer=transfer,
        sensitivity=sensitivity,
        permutation=permutation,
        derived_binary_audit=derived_binary,
        decision=decision,
        warnings=warnings,
        errors=errors,
        research_only=True,
        exploratory_only=True,
        weak_label_target=True,
        no_final_labels=True,
        no_ground_truth_claim=True,
        no_production_claim=True,
        production_training=False,
        derived_binary_audit_only=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
    )
    return report, rows


def write_mvp4x_baseline_outputs(
    report: Mvp4xBaselineReport,
    rows: list[dict[str, Any]],
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    for path in (output_report_md, output_report_json, output_csv):
        _ensure_can_write(path, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_mvp4x_baseline_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_mvp4x_baseline_markdown(report: Mvp4xBaselineReport) -> str:
    best = report.decision.get("best_result") or {}
    lines = [
        "# MVP-4X Existing-Feature Baselines",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- report_version: `{report.report_version}`",
        f"- feature_set_name: `{report.feature_set_name}`",
        f"- model_backend: `{report.model_backend}`",
        f"- sklearn_available: {report.sklearn_available}",
        f"- sample_count: {report.sample_count}",
        f"- feature_count: {report.feature_count}",
        f"- baseline_sufficient: {report.decision.get('baseline_sufficient')}",
        f"- stage3_recommended: {report.decision.get('stage3_recommended')}",
        f"- best_target: {best.get('target')}",
        f"- best_model: {best.get('model')}",
        f"- best_spearman: {best.get('spearman')}",
        f"- warnings: {len(report.warnings)}",
        f"- errors: {len(report.errors)}",
        "",
        "Derived binary metrics are derived_binary_audit_only and are not final labels.",
    ]
    lines.extend(
        [
            f"- python_version: {report.modeling_environment.get('python_version')}",
            f"- sklearn_version: {report.modeling_environment.get('sklearn_version')}",
            f"- numpy_version: {report.modeling_environment.get('numpy_version')}",
            f"- scipy_version: {report.modeling_environment.get('scipy_version')}",
            f"- model_random_seed: {report.model_random_seed}",
            f"- cv_protocol: {report.cv_protocol}",
            f"- permutation_protocol: {report.permutation_protocol}",
        ]
    )
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report.warnings)
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in report.errors)
    return "\n".join(lines) + "\n"


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    if np.count_nonzero(mask) == 0:
        return _empty_metrics()
    true = true[mask]
    pred = pred[mask]
    residual = pred - true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    median_abs_error = float(np.median(np.abs(residual)))
    denom = float(np.sum((true - np.mean(true)) ** 2))
    r2 = None if denom <= 0.0 else float(1.0 - np.sum(residual**2) / denom)
    pearson = _pearson(true, pred)
    spearman = _spearman(true, pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "pearson": pearson,
        "median_absolute_error": median_abs_error,
    }


def contiguous_depth_folds(
    depth: np.ndarray,
    sample_mask: np.ndarray,
    n_folds: int,
) -> list[np.ndarray]:
    values = np.asarray(depth, dtype=np.float32).reshape(-1)
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    selected = np.flatnonzero(mask)
    if selected.size < n_folds:
        raise ValueError("Not enough selected samples for contiguous folds.")
    ordered = selected[np.argsort(values[selected])]
    folds = [np.zeros(values.size, dtype=bool) for _ in range(n_folds)]
    for fold_index, fold_indices in enumerate(np.array_split(ordered, n_folds)):
        folds[fold_index][fold_indices] = True
    return folds


def _run_contiguous_cv(
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
    folds = contiguous_depth_folds(
        depth,
        sample_mask,
        n_folds=int(config.get("n_contiguous_folds", 3)),
    )
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_index, validation_mask in enumerate(folds):
        train_mask = sample_mask & ~validation_mask
        validation_mask = sample_mask & validation_mask
        if np.count_nonzero(train_mask) == 0 or np.count_nonzero(validation_mask) == 0:
            continue
        model = _make_model(model_name, sklearn_modules, config, random_state=rng_seed + fold_index)
        model.fit(X[train_mask], y[train_mask])
        pred = np.asarray(model.predict(X[validation_mask]), dtype=np.float32)
        oof[validation_mask] = pred
        metrics = compute_regression_metrics(y[validation_mask], pred)
        fold_row = {
            "fold": fold_index,
            "validation_count": int(np.count_nonzero(validation_mask)),
            "validation_depth_min": float(np.min(depth[validation_mask])),
            "validation_depth_max": float(np.max(depth[validation_mask])),
            **metrics,
        }
        fold_metrics.append(fold_row)
        rows.append(
            {
                "csv_version": BASELINE_CSV_VERSION,
                "feature_set": "",
                "target": "",
                "model": model_name,
                "evaluation": "contiguous_cv",
                "fold": fold_index,
                "status": "completed",
                **_csv_metrics(metrics),
            }
        )
    aggregate = compute_regression_metrics(y[sample_mask], oof[sample_mask])
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
            "fold_count": len(fold_metrics),
            "stable_positive_spearman_folds": stable_folds,
            "aggregate": aggregate,
            "folds": fold_metrics,
            "prediction_summary": _target_summary(oof[sample_mask]),
            "calibration_by_target_quantile": _calibration_by_target_quantile(
                y[sample_mask],
                oof[sample_mask],
            ),
        },
        "rows": rows,
        "oof_prediction": oof,
    }


def _run_leave_one_regime_out(
    *,
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for regime in sorted(set(regimes[sample_mask].tolist())):
        validation_mask = sample_mask & (regimes == regime)
        train_mask = sample_mask & (regimes != regime)
        results[f"train_not_{regime}_validate_{regime}"] = _fit_eval_once(
            X,
            y,
            train_mask,
            validation_mask,
            model_name,
            sklearn_modules,
            config,
            random_state=17,
        )
    return results


def _run_regime_transfer(
    *,
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for regime in sorted(set(regimes[sample_mask].tolist())):
        train_mask = sample_mask & (regimes == regime)
        validation_mask = sample_mask & (regimes != regime)
        results[f"{regime}_to_other_regimes"] = _fit_eval_once(
            X,
            y,
            train_mask,
            validation_mask,
            model_name,
            sklearn_modules,
            config,
            random_state=23,
        )
    return results


def _run_sensitivity_filters(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    snapshot: dict[str, np.ndarray],
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in SENSITIVITY_FILTERS:
        mask = sample_mask & _sensitivity_mask(snapshot, name)
        if np.count_nonzero(mask) < int(config.get("n_contiguous_folds", 3)):
            results[name] = {"status": "skipped_too_few_samples", "sample_count": int(mask.sum())}
            continue
        results[name] = _run_contiguous_cv(
            X=X,
            y=y,
            depth=depth,
            sample_mask=mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=101,
        )["summary"]
    return results


def _run_target_permutations(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    count = int(config.get("permutation_count", 20))
    metrics: list[dict[str, float | None]] = []
    y_selected = y[sample_mask].copy()
    selected_indices = np.flatnonzero(sample_mask)
    for permutation_index in range(count):
        y_perm = y.copy()
        y_perm[selected_indices] = rng.permutation(y_selected)
        result = _run_contiguous_cv(
            X=X,
            y=y_perm,
            depth=depth,
            sample_mask=sample_mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
            rng_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        metric = result["summary"]["aggregate"]
        metric["permutation_index"] = permutation_index
        metrics.append(metric)
    return {
        "permutation_count": count,
        "spearman_mean": _metric_mean(metrics, "spearman"),
        "spearman_std": _metric_std(metrics, "spearman"),
        "spearman_max": _metric_max(metrics, "spearman"),
        "r2_mean": _metric_mean(metrics, "r2"),
        "metrics": metrics,
    }


def _derived_binary_audit(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_mask: np.ndarray,
    thresholds: list[float],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    metrics_module = sklearn_modules.get("metrics")
    output: dict[str, Any] = {"derived_binary_audit_only": True}
    for threshold in thresholds:
        mask = sample_mask & np.isfinite(y_pred)
        truth = y_true[mask] > threshold
        pred = y_pred[mask] > threshold
        if truth.size == 0 or len(set(truth.tolist())) < 2:
            output[str(threshold)] = {"status": "skipped_single_class_or_empty"}
            continue
        tp = int(np.count_nonzero(truth & pred))
        tn = int(np.count_nonzero(~truth & ~pred))
        fp = int(np.count_nonzero(~truth & pred))
        fn = int(np.count_nonzero(truth & ~pred))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = None if precision is None or recall is None or precision + recall == 0 else (
            2.0 * precision * recall / (precision + recall)
        )
        pr_auc = None
        if metrics_module is not None:
            pr_auc = float(metrics_module.average_precision_score(truth.astype(int), y_pred[mask]))
        output[str(threshold)] = {
            "balanced_accuracy": None
            if recall is None or specificity is None
            else (recall + specificity) / 2.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
        }
    return output


def _fit_eval_once(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    *,
    random_state: int,
) -> dict[str, Any]:
    if np.count_nonzero(train_mask) == 0 or np.count_nonzero(validation_mask) == 0:
        return {"status": "skipped_empty_train_or_validation"}
    model = _make_model(model_name, sklearn_modules, config, random_state=random_state)
    model.fit(X[train_mask], y[train_mask])
    pred = np.asarray(model.predict(X[validation_mask]), dtype=np.float32)
    return {
        "status": "completed",
        "train_count": int(np.count_nonzero(train_mask)),
        "validation_count": int(np.count_nonzero(validation_mask)),
        **compute_regression_metrics(y[validation_mask], pred),
    }


def _make_model(
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    *,
    random_state: int,
) -> Any:
    pipeline = sklearn_modules["pipeline"]
    preprocessing = sklearn_modules["preprocessing"]
    impute = sklearn_modules["impute"]
    linear = sklearn_modules["linear_model"]
    dummy = sklearn_modules["dummy"]
    ensemble = sklearn_modules["ensemble"]
    steps: list[Any] = [impute.SimpleImputer(strategy="median")]
    if model_name == "DummyRegressor":
        model = dummy.DummyRegressor(strategy="mean")
    elif model_name == "Ridge":
        steps.append(preprocessing.StandardScaler())
        model = linear.Ridge(alpha=float(config.get("ridge_alpha", 1.0)))
    elif model_name == "ElasticNet":
        steps.append(preprocessing.StandardScaler())
        model = linear.ElasticNet(
            alpha=float(config.get("elastic_net_alpha", 0.01)),
            l1_ratio=float(config.get("elastic_net_l1_ratio", 0.5)),
            max_iter=3000,
            random_state=random_state,
        )
    elif model_name == "RandomForestRegressor":
        model = ensemble.RandomForestRegressor(
            n_estimators=int(config.get("random_forest_n_estimators", 80)),
            max_depth=int(config.get("random_forest_max_depth", 6)),
            random_state=random_state,
            n_jobs=int(config.get("random_forest_n_jobs", -1)),
        )
    elif model_name == "HistGradientBoostingRegressor":
        model = ensemble.HistGradientBoostingRegressor(
            max_iter=int(config.get("hist_gradient_boosting_max_iter", 120)),
            max_leaf_nodes=int(config.get("hist_gradient_boosting_max_leaf_nodes", 15)),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    steps.append(model)
    return pipeline.Pipeline([(f"step_{index}", step) for index, step in enumerate(steps)])


def _sensitivity_mask(snapshot: dict[str, np.ndarray], name: str) -> np.ndarray:
    depth = np.asarray(snapshot["depth"]).reshape(-1)
    mask = np.ones(depth.size, dtype=bool)
    if name == "include_all":
        return mask
    if name == "exclude_saturation_platform":
        return mask & ~np.asarray(snapshot.get("saturation_platform_flag", False), dtype=bool)
    if name == "exclude_5680_special_band":
        return mask & ~np.asarray(snapshot.get("special_5680_flag", False), dtype=bool)
    if name == "exclude_low_orientation_confidence":
        return mask & ~np.asarray(
            snapshot.get("low_orientation_confidence_flag", False),
            dtype=bool,
        )
    if name == "exclude_all_special_flags":
        return mask & ~np.asarray(snapshot.get("any_special_flag", False), dtype=bool)
    raise ValueError(f"Unsupported sensitivity filter: {name}")


def _baseline_decision(
    *,
    contiguous: dict[str, Any],
    leave_one: dict[str, Any],
    sensitivity: dict[str, Any],
    permutation: dict[str, Any],
    sklearn_available: bool,
    errors: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    if errors:
        return {
            "baseline_sufficient": False,
            "stage3_recommended": False,
            "decision_reason": "data_or_feature_error",
            "best_result": None,
            "leakage_detected": True,
        }
    if not sklearn_available:
        return {
            "baseline_sufficient": False,
            "stage3_recommended": True,
            "decision_reason": "sklearn_unavailable_all_models_skipped",
            "best_result": None,
            "leakage_detected": False,
        }
    best = _best_contiguous_result(contiguous)
    if best is None:
        return {
            "baseline_sufficient": False,
            "stage3_recommended": True,
            "decision_reason": "no_completed_model_result",
            "best_result": None,
            "leakage_detected": False,
        }
    perm = permutation.get(best["target_key"], {}).get(best["model"], {})
    spearman = best.get("spearman")
    perm_mean = perm.get("spearman_mean")
    margin = None if spearman is None or perm_mean is None else float(spearman - perm_mean)
    stable_folds = int(best.get("stable_positive_spearman_folds") or 0)
    min_margin = float(config.get("sufficient_spearman_margin", 0.05))
    min_folds = int(config.get("sufficient_min_stable_folds", 2))
    sufficient = (
        margin is not None
        and margin >= min_margin
        and stable_folds >= min_folds
        and _regime_not_fully_degenerate(
            leave_one.get(best["target_key"], {}).get(best["model"], {})
        )
        and _sensitivity_not_special_only(
            sensitivity.get(best["target_key"], {}).get(best["model"], {})
        )
    )
    return {
        "baseline_sufficient": bool(sufficient),
        "stage3_recommended": not sufficient,
        "decision_reason": "criteria_met" if sufficient else "weak_or_unstable_existing_features",
        "best_result": {**best, "real_minus_permutation_spearman": margin},
        "leakage_detected": False,
    }


def _best_contiguous_result(contiguous: dict[str, Any]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for target_key, by_model in contiguous.items():
        for model, summary in by_model.items():
            aggregate = _as_dict(summary.get("aggregate"))
            spearman = aggregate.get("spearman")
            if spearman is None or model == "DummyRegressor":
                continue
            candidate = {
                "target_key": target_key,
                "target": target_key.split(":", 1)[0],
                "model": model,
                "spearman": float(spearman),
                "r2": aggregate.get("r2"),
                "mae": aggregate.get("mae"),
                "stable_positive_spearman_folds": summary.get(
                    "stable_positive_spearman_folds"
                ),
            }
            if best is None or candidate["spearman"] > float(best["spearman"]):
                best = candidate
    return best


def _regime_not_fully_degenerate(results: dict[str, Any]) -> bool:
    completed = [row for row in results.values() if _as_dict(row).get("status") == "completed"]
    if not completed:
        return False
    return any((_as_dict(row).get("spearman") or 0.0) > 0.0 for row in completed)


def _sensitivity_not_special_only(results: dict[str, Any]) -> bool:
    include = _as_dict(results.get("include_all")).get("aggregate", {})
    exclude_special = _as_dict(results.get("exclude_all_special_flags")).get("aggregate", {})
    include_s = _as_dict(include).get("spearman")
    exclude_s = _as_dict(exclude_special).get("spearman")
    if include_s is None or exclude_s is None:
        return False
    return float(exclude_s) > 0.0 and float(exclude_s) >= (0.25 * float(include_s))


def _target_summary(values: np.ndarray) -> dict[str, float | int | None]:
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
        "p50": float(np.quantile(finite, 0.50)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _calibration_by_target_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantile_count: int = 5,
) -> list[dict[str, float | int | None]]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    if np.count_nonzero(mask) < quantile_count:
        return []
    true = true[mask]
    pred = pred[mask]
    order = np.argsort(true)
    rows: list[dict[str, float | int | None]] = []
    for index, indices in enumerate(np.array_split(order, quantile_count)):
        if indices.size == 0:
            continue
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


def _pearson(true: np.ndarray, pred: np.ndarray) -> float | None:
    if true.size < 2 or np.std(true) == 0.0 or np.std(pred) == 0.0:
        return None
    return float(np.corrcoef(true, pred)[0, 1])


def _spearman(true: np.ndarray, pred: np.ndarray) -> float | None:
    if true.size < 2 or np.std(true) == 0.0 or np.std(pred) == 0.0:
        return None
    value = spearmanr(true, pred).statistic
    return None if not np.isfinite(value) else float(value)


def _empty_metrics() -> dict[str, float | None]:
    return {
        "mae": None,
        "rmse": None,
        "r2": None,
        "spearman": None,
        "pearson": None,
        "median_absolute_error": None,
    }


def _metric_mean(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in metrics if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _metric_std(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in metrics if row.get(key) is not None]
    return None if not values else float(np.std(values))


def _metric_max(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in metrics if row.get(key) is not None]
    return None if not values else float(np.max(values))


def _csv_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in _empty_metrics()}


def _safe_div(num: int, den: int) -> float | None:
    return None if den == 0 else float(num / den)


def _validate_research_flags(snapshot: dict[str, np.ndarray], errors: list[str]) -> None:
    for key, expected in RESEARCH_FLAGS.items():
        value = snapshot.get(key)
        if value is None or bool(np.asarray(value).reshape(())) != expected:
            errors.append(f"snapshot flag {key} must be {expected}.")


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


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "csv_version",
        "feature_set",
        "target",
        "model",
        "evaluation",
        "fold",
        "status",
        "mae",
        "rmse",
        "r2",
        "spearman",
        "pearson",
        "median_absolute_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any, default: tuple[str, ...] | None = None) -> list[Any]:
    if isinstance(value, list):
        return value
    return list(default or ())


def _as_float_list(value: Any, default: list[float]) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return default
