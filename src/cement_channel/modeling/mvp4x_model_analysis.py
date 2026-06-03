from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import (
    BASELINE_CSV_VERSION,
    _make_model,
    compute_regression_metrics,
    contiguous_depth_folds,
    run_mvp4x_baselines,
)

ENHANCED_REPORT_VERSION = "mvp4x_enhanced_feature_baselines_v001"
MODEL_REVIEW_VERSION = "mvp4x_model_review_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}


@dataclass(frozen=True)
class EnhancedBaselineReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    feature_sets: dict[str, dict[str, Any]]
    sub_reports: dict[str, dict[str, Any]]
    best_result: dict[str, Any] | None
    feature_group_ablation: dict[str, Any]
    permutation_importance: dict[str, Any]
    top_30_features: list[dict[str, Any]]
    top_10_stable_features: list[dict[str, Any]]
    regime_specific_error_analysis: dict[str, Any]
    special_band_error_analysis: dict[str, Any]
    low_orientation_error_analysis: dict[str, Any]
    review_dir: str
    review_files: dict[str, str]
    warnings: list[str]
    errors: list[str]
    research_only: bool
    exploratory_only: bool
    weak_label_target: bool
    no_final_labels: bool
    no_ground_truth_claim: bool
    no_production_claim: bool
    production_training: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_enhanced_feature_baselines_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    output_review_dir: Path | str,
    overwrite: bool = False,
) -> EnhancedBaselineReport:
    config = _load_yaml(config_path)
    snapshot = _load_npz(snapshot_npz)
    waveform = _load_npz(waveform_features_npz)
    report, rows, review_files = run_enhanced_feature_baselines(
        snapshot=snapshot,
        waveform=waveform,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "config_path": str(config_path),
        },
        review_dir=Path(output_review_dir),
        overwrite=overwrite,
    )
    write_enhanced_baseline_outputs(
        report,
        rows,
        review_files=review_files,
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def run_enhanced_feature_baselines(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    config: dict[str, Any],
    inputs: dict[str, str],
    review_dir: Path,
    overwrite: bool,
) -> tuple[EnhancedBaselineReport, list[dict[str, Any]], dict[str, str]]:
    warnings: list[str] = []
    errors: list[str] = []
    feature_sets = build_feature_sets(snapshot, waveform)
    rows: list[dict[str, Any]] = []
    sub_reports: dict[str, dict[str, Any]] = {}
    for feature_set_name, feature_set in feature_sets.items():
        augmented = dict(snapshot)
        augmented["active_features"] = feature_set["matrix"]
        augmented["active_feature_names"] = feature_set["names"]
        report, set_rows = run_mvp4x_baselines(
            snapshot=augmented,
            config=config,
            inputs=inputs,
            feature_set_name=feature_set_name,
            feature_matrix_key="active_features",
            feature_name_key="active_feature_names",
        )
        sub_reports[feature_set_name] = report.to_dict()
        warnings.extend(f"{feature_set_name}: {warning}" for warning in report.warnings)
        errors.extend(f"{feature_set_name}: {error}" for error in report.errors)
        rows.extend(set_rows)
    best = _best_enhanced_result(sub_reports)
    feature_group_summary = summarize_feature_groups(feature_sets)
    analysis = compute_fitted_model_analysis(
        snapshot=snapshot,
        feature_sets=feature_sets,
        config=config,
        best=best,
        review_dir=review_dir,
        overwrite=overwrite,
    )
    review_files = write_model_review_dir(
        review_dir=review_dir,
        feature_group_summary=feature_group_summary,
        analysis=analysis,
        generated_files=_as_dict(analysis.get("review_files")),
        overwrite=overwrite,
    )
    report = EnhancedBaselineReport(
        report_version=ENHANCED_REPORT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        feature_sets=feature_group_summary,
        sub_reports=sub_reports,
        best_result=best,
        feature_group_ablation=analysis["feature_group_ablation"],
        permutation_importance=analysis["permutation_importance"],
        top_30_features=analysis["top_30_features"],
        top_10_stable_features=analysis["top_10_stable_features"],
        regime_specific_error_analysis=analysis["regime_specific_error_analysis"],
        special_band_error_analysis=analysis["special_band_error_analysis"],
        low_orientation_error_analysis=analysis["low_orientation_error_analysis"],
        review_dir=str(review_dir),
        review_files=review_files,
        warnings=warnings,
        errors=errors,
        research_only=True,
        exploratory_only=True,
        weak_label_target=True,
        no_final_labels=True,
        no_ground_truth_claim=True,
        no_production_claim=True,
        production_training=False,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
    )
    return report, rows, review_files


def build_feature_sets(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    existing_names = np.asarray(snapshot["xsi_feature_names"]).astype(str)
    existing_groups = np.asarray(snapshot["xsi_feature_group"]).astype(str)
    wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    wave_names = np.asarray(waveform["waveform_depth_feature_names"]).astype(str)
    wave_groups = np.asarray(waveform["waveform_depth_feature_group"]).astype(str)
    if existing.shape[0] != wave.shape[0]:
        raise ValueError("Existing and waveform feature row counts differ.")
    return {
        "existing_features_only": {
            "matrix": existing,
            "names": existing_names,
            "groups": np.asarray([f"existing:{item}" for item in existing_groups]),
        },
        "waveform_features_only": {
            "matrix": wave,
            "names": wave_names,
            "groups": np.asarray([f"waveform:{item}" for item in wave_groups]),
        },
        "combined_features": {
            "matrix": np.column_stack([existing, wave]).astype(np.float32),
            "names": np.concatenate([existing_names, wave_names]).astype(str),
            "groups": np.concatenate(
                [
                    np.asarray([f"existing:{item}" for item in existing_groups]),
                    np.asarray([f"waveform:{item}" for item in wave_groups]),
                ]
            ).astype(str),
        },
    }


def summarize_feature_groups(feature_sets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, feature_set in feature_sets.items():
        groups = np.asarray(feature_set["groups"]).astype(str)
        unique, counts = np.unique(groups, return_counts=True)
        output[name] = {
            "feature_count": int(np.asarray(feature_set["matrix"]).shape[1]),
            "sample_count": int(np.asarray(feature_set["matrix"]).shape[0]),
            "group_counts": {
                str(group): int(count) for group, count in zip(unique, counts, strict=True)
            },
            "finite_ratio": float(np.isfinite(np.asarray(feature_set["matrix"])).mean()),
        }
    return output


def compute_fitted_model_analysis(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    config: dict[str, Any],
    best: dict[str, Any] | None,
    review_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if not best:
        return _analysis_placeholders({})
    feature_set_name = str(best.get("feature_set"))
    target = str(best.get("target"))
    model_name = str(best.get("model"))
    if feature_set_name not in feature_sets:
        skipped = {
            "status": "skipped_best_feature_set_missing",
            "feature_set": feature_set_name,
        }
        return _skipped_analysis(skipped)
    if target not in snapshot:
        skipped = {"status": "skipped_best_target_missing", "target": target}
        return _skipped_analysis(skipped)

    baseline_config = _as_dict(config.get("baseline"))
    sklearn_modules, _environment = require_sklearn_for_modeling()
    feature_set = feature_sets[feature_set_name]
    raw_X = np.asarray(feature_set["matrix"], dtype=np.float32)
    X = np.nan_to_num(raw_X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.asarray(snapshot[target], dtype=np.float32).reshape(-1)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    feature_names = np.asarray(feature_set["names"]).astype(str)
    feature_groups = np.asarray(feature_set["groups"]).astype(str)
    sample_mask = np.isfinite(y) & np.all(np.isfinite(raw_X), axis=1)

    oof, fold_summaries, fold_importance = _fit_oof_with_permutation_importance(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=baseline_config,
        feature_names=feature_names,
        feature_groups=feature_groups,
    )
    full_metrics = compute_regression_metrics(y[sample_mask], oof[sample_mask])
    importance = _summarize_permutation_importance(fold_importance)
    ablation = _run_feature_group_ablation(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        config=baseline_config,
        feature_groups=feature_groups,
        full_metrics=full_metrics,
    )
    regime_analysis = _label_error_analysis(
        y_true=y,
        y_pred=oof,
        labels=np.asarray(snapshot["broad_regime_id"]).astype(str),
        sample_mask=sample_mask,
    )
    special_analysis = _flag_error_analysis(
        y_true=y,
        y_pred=oof,
        sample_mask=sample_mask,
        flags={
            "saturation_platform_flag": np.asarray(
                snapshot.get("saturation_platform_flag", False),
                dtype=bool,
            ),
            "special_5680_flag": np.asarray(snapshot.get("special_5680_flag", False), dtype=bool),
            "any_special_flag": np.asarray(snapshot.get("any_special_flag", False), dtype=bool),
        },
    )
    low_orientation_analysis = _flag_error_analysis(
        y_true=y,
        y_pred=oof,
        sample_mask=sample_mask,
        flags={
            "low_orientation_confidence_flag": np.asarray(
                snapshot.get("low_orientation_confidence_flag", False),
                dtype=bool,
            ),
        },
    )
    review_files = _write_review_plots(
        review_dir=review_dir,
        y_true=y,
        y_pred=oof,
        depth=depth,
        sample_mask=sample_mask,
        overwrite=overwrite,
    )
    return {
        "feature_group_ablation": ablation,
        "permutation_importance": {
            "status": "completed",
            "method": "validation_fold_feature_shuffle",
            "scoring": "spearman_drop",
            "feature_set": feature_set_name,
            "target": target,
            "model": model_name,
            "folds": fold_summaries,
            "aggregate_metrics": full_metrics,
            "feature_count": int(feature_names.size),
        },
        "top_30_features": importance["top_30_features"],
        "top_10_stable_features": importance["top_10_stable_features"],
        "regime_specific_error_analysis": regime_analysis,
        "special_band_error_analysis": special_analysis,
        "low_orientation_error_analysis": low_orientation_analysis,
        "review_files": review_files,
    }


def _fit_oof_with_permutation_importance(
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
    folds = contiguous_depth_folds(
        depth,
        sample_mask,
        n_folds=int(config.get("n_contiguous_folds", 3)),
    )
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, Any]] = []
    fold_importance: list[dict[str, Any]] = []
    for fold_index, validation_mask in enumerate(folds):
        train_mask = sample_mask & ~validation_mask
        validation_mask = sample_mask & validation_mask
        if np.count_nonzero(train_mask) == 0 or np.count_nonzero(validation_mask) == 0:
            continue
        model = _make_model(
            model_name,
            sklearn_modules,
            config,
            random_state=31 + fold_index,
        )
        model.fit(X[train_mask], y[train_mask])
        X_validation = X[validation_mask]
        y_validation = y[validation_mask]
        prediction = np.asarray(model.predict(X_validation), dtype=np.float32)
        oof[validation_mask] = prediction
        metrics = compute_regression_metrics(y_validation, prediction)
        fold_summaries.append(
            {
                "fold": fold_index,
                "validation_count": int(np.count_nonzero(validation_mask)),
                **metrics,
            }
        )
        fold_importance.extend(
            _fold_permutation_importance(
                model=model,
                X_validation=X_validation,
                y_validation=y_validation,
                baseline_spearman=metrics.get("spearman"),
                feature_names=feature_names,
                feature_groups=feature_groups,
                fold_index=fold_index,
                seed=20240603 + fold_index,
            )
        )
    return oof, fold_summaries, fold_importance


def _fold_permutation_importance(
    *,
    model: Any,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    baseline_spearman: float | None,
    feature_names: np.ndarray,
    feature_groups: np.ndarray,
    fold_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    if baseline_spearman is None:
        baseline = 0.0
    else:
        baseline = float(baseline_spearman)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for feature_index, (feature_name, feature_group) in enumerate(
        zip(feature_names.tolist(), feature_groups.tolist(), strict=True)
    ):
        permuted = X_validation.copy()
        permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
        prediction = np.asarray(model.predict(permuted), dtype=np.float32)
        metrics = compute_regression_metrics(y_validation, prediction)
        permuted_spearman = metrics.get("spearman")
        drop = None if permuted_spearman is None else baseline - float(permuted_spearman)
        rows.append(
            {
                "fold": fold_index,
                "feature_index": int(feature_index),
                "feature_name": str(feature_name),
                "feature_group": str(feature_group),
                "spearman_drop": drop,
            }
        )
    return rows


def _summarize_permutation_importance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_feature: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_feature.setdefault(int(row["feature_index"]), []).append(row)
    fold_top: dict[int, set[int]] = {}
    for fold in sorted({int(row["fold"]) for row in rows}):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        fold_rows.sort(key=lambda row: _none_safe_float(row.get("spearman_drop")), reverse=True)
        fold_top[fold] = {int(row["feature_index"]) for row in fold_rows[:30]}

    summaries: list[dict[str, Any]] = []
    for feature_index, feature_rows in by_feature.items():
        drops = [
            float(row["spearman_drop"])
            for row in feature_rows
            if row.get("spearman_drop") is not None
        ]
        if not drops:
            mean_drop = None
            std_drop = None
        else:
            mean_drop = float(np.mean(drops))
            std_drop = float(np.std(drops))
        summaries.append(
            {
                "feature_index": feature_index,
                "feature_name": str(feature_rows[0]["feature_name"]),
                "feature_group": str(feature_rows[0]["feature_group"]),
                "mean_spearman_drop": mean_drop,
                "std_spearman_drop": std_drop,
                "fold_count": len(feature_rows),
                "top_30_fold_count": int(
                    sum(feature_index in top_features for top_features in fold_top.values())
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            _none_safe_float(row.get("mean_spearman_drop")),
            int(row.get("top_30_fold_count") or 0),
        ),
        reverse=True,
    )
    stable = [
        row
        for row in summaries
        if int(row.get("top_30_fold_count") or 0) >= min(2, max(1, len(fold_top)))
    ]
    if len(stable) < 10:
        stable = summaries[:10]
    return {
        "top_30_features": summaries[:30],
        "top_10_stable_features": stable[:10],
    }


def _run_feature_group_ablation(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
    feature_groups: np.ndarray,
    full_metrics: dict[str, Any],
) -> dict[str, Any]:
    groups = sorted(set(feature_groups.tolist()))
    full_spearman = full_metrics.get("spearman")
    results: list[dict[str, Any]] = []
    for group in groups:
        keep = feature_groups != group
        if np.count_nonzero(keep) == 0:
            continue
        oof = _fit_oof_predictions(
            X=X[:, keep],
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            config=config,
        )
        metrics = compute_regression_metrics(y[sample_mask], oof[sample_mask])
        group_spearman = metrics.get("spearman")
        loss = None
        if full_spearman is not None and group_spearman is not None:
            loss = float(full_spearman) - float(group_spearman)
        results.append(
            {
                "dropped_group": str(group),
                "remaining_feature_count": int(np.count_nonzero(keep)),
                "spearman_without_group": group_spearman,
                "spearman_loss_vs_full": loss,
                "mae_without_group": metrics.get("mae"),
            }
        )
    results.sort(key=lambda row: _none_safe_float(row.get("spearman_loss_vs_full")), reverse=True)
    return {
        "status": "completed",
        "method": "drop_one_feature_group_contiguous_cv",
        "full_model_metrics": full_metrics,
        "results": results,
    }


def _fit_oof_predictions(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> np.ndarray:
    folds = contiguous_depth_folds(
        depth,
        sample_mask,
        n_folds=int(config.get("n_contiguous_folds", 3)),
    )
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    for fold_index, validation_mask in enumerate(folds):
        train_mask = sample_mask & ~validation_mask
        validation_mask = sample_mask & validation_mask
        if np.count_nonzero(train_mask) == 0 or np.count_nonzero(validation_mask) == 0:
            continue
        model = _make_model(
            model_name,
            sklearn_modules,
            config,
            random_state=71 + fold_index,
        )
        model.fit(X[train_mask], y[train_mask])
        oof[validation_mask] = np.asarray(model.predict(X[validation_mask]), dtype=np.float32)
    return oof


def _label_error_analysis(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    sample_mask: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {"status": "completed", "groups": {}}
    mask = sample_mask & np.isfinite(y_pred)
    for label in sorted(set(labels[mask].tolist())):
        group_mask = mask & (labels == label)
        output["groups"][str(label)] = {
            "sample_count": int(np.count_nonzero(group_mask)),
            **compute_regression_metrics(y_true[group_mask], y_pred[group_mask]),
        }
    completed = [
        {"label": label, **metrics}
        for label, metrics in output["groups"].items()
        if metrics.get("mae") is not None
    ]
    completed.sort(key=lambda row: float(row["mae"]), reverse=True)
    output["worst_by_mae"] = completed[0] if completed else None
    return output


def _flag_error_analysis(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_mask: np.ndarray,
    flags: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {"status": "completed", "flags": {}}
    finite_mask = sample_mask & np.isfinite(y_pred)
    for name, values in flags.items():
        flag = np.asarray(values, dtype=bool).reshape(-1)
        flagged = finite_mask & flag
        unflagged = finite_mask & ~flag
        flagged_metrics = compute_regression_metrics(y_true[flagged], y_pred[flagged])
        unflagged_metrics = compute_regression_metrics(y_true[unflagged], y_pred[unflagged])
        flagged_s = flagged_metrics.get("spearman")
        unflagged_s = unflagged_metrics.get("spearman")
        delta = None if flagged_s is None or unflagged_s is None else (
            float(flagged_s) - float(unflagged_s)
        )
        output["flags"][name] = {
            "flagged_sample_count": int(np.count_nonzero(flagged)),
            "unflagged_sample_count": int(np.count_nonzero(unflagged)),
            "flagged_metrics": flagged_metrics,
            "unflagged_metrics": unflagged_metrics,
            "spearman_delta_flagged_minus_unflagged": delta,
        }
    return output


def _write_review_plots(
    *,
    review_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    overwrite: bool,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "predicted_vs_target_png": review_dir / "predicted_vs_target.png",
        "residual_vs_depth_png": review_dir / "residual_vs_depth.png",
        "calibration_by_target_quantile_png": review_dir / "calibration_by_target_quantile.png",
    }
    for path in files.values():
        _ensure_can_write(path, overwrite=overwrite)
    try:
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
    min_value = float(np.nanmin(np.concatenate([true, pred])))
    max_value = float(np.nanmax(np.concatenate([true, pred])))
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
    ax.set_xlabel("weak label target")
    ax.set_ylabel("prediction")
    ax.set_title("Predicted vs target")
    fig.tight_layout()
    fig.savefig(files["predicted_vs_target_png"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(depth_values, residual, s=7, alpha=0.35)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("depth ft")
    ax.set_ylabel("residual")
    ax.set_title("Residual vs depth")
    fig.tight_layout()
    fig.savefig(files["residual_vs_depth_png"], dpi=150)
    plt.close(fig)

    calibration = _calibration_rows(true, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [row["target_mean"] for row in calibration],
        [row["prediction_mean"] for row in calibration],
        marker="o",
    )
    ax.set_xlabel("target quantile mean")
    ax.set_ylabel("prediction mean")
    ax.set_title("Calibration by target quantile")
    fig.tight_layout()
    fig.savefig(files["calibration_by_target_quantile_png"], dpi=150)
    plt.close(fig)
    return {key: str(path) for key, path in files.items()}


def _calibration_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantile_count: int = 5,
) -> list[dict[str, float | int]]:
    order = np.argsort(y_true)
    rows: list[dict[str, float | int]] = []
    for index, indices in enumerate(np.array_split(order, quantile_count)):
        if indices.size == 0:
            continue
        rows.append(
            {
                "quantile_bin": index,
                "target_mean": float(np.mean(y_true[indices])),
                "prediction_mean": float(np.mean(y_pred[indices])),
            }
        )
    return rows


def write_model_review_dir(
    *,
    review_dir: Path,
    feature_group_summary: dict[str, Any],
    analysis: dict[str, Any],
    generated_files: dict[str, str],
    overwrite: bool,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    summary_json = review_dir / "feature_set_summary.json"
    summary_md = review_dir / "model_review_summary.md"
    for path in (summary_json, summary_md):
        _ensure_can_write(path, overwrite=overwrite)
    summary_json.write_text(
        json.dumps(
            {
                "review_version": MODEL_REVIEW_VERSION,
                "feature_sets": feature_group_summary,
                "analysis": analysis,
                "review_files": generated_files,
                **RESEARCH_FLAGS,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(
        "\n".join(
            [
                "# MVP-4X Model Review",
                "",
                "Scope: research_only, exploratory_only, weak_label_target, "
                "no_final_labels, no_ground_truth_claim, no_production_claim.",
                "",
                f"- feature_group_ablation: {analysis['feature_group_ablation'].get('status')}",
                f"- permutation_importance: {analysis['permutation_importance'].get('status')}",
                f"- top_stable_features: {len(analysis['top_10_stable_features'])}",
                "",
                "## Review Files",
                *[f"- {name}: {path}" for name, path in sorted(generated_files.items())],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "feature_set_summary_json": str(summary_json),
        "model_review_summary_md": str(summary_md),
        **generated_files,
    }


def write_enhanced_baseline_outputs(
    report: EnhancedBaselineReport,
    rows: list[dict[str, Any]],
    *,
    review_files: dict[str, str],
    output_report_md: Path,
    output_report_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    del review_files
    for path in (output_report_md, output_report_json, output_csv):
        _ensure_can_write(path, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_enhanced_baseline_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_enhanced_baseline_markdown(report: EnhancedBaselineReport) -> str:
    lines = [
        "# MVP-4X Enhanced-Feature Baselines",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- report_version: `{report.report_version}`",
        f"- best_result: {report.best_result}",
        f"- feature_sets: {', '.join(report.feature_sets)}",
        f"- warnings: {len(report.warnings)}",
        f"- errors: {len(report.errors)}",
        "",
        "Derived binary metrics, if present, are derived_binary_audit_only.",
    ]
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report.warnings[:20])
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in report.errors)
    return "\n".join(lines) + "\n"


def _best_enhanced_result(sub_reports: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for feature_set, report in sub_reports.items():
        result = _as_dict(_as_dict(report.get("decision")).get("best_result"))
        if not result:
            continue
        candidate = {**result, "feature_set": feature_set}
        spearman = candidate.get("spearman")
        if spearman is None:
            continue
        if best is None or float(spearman) > float(best["spearman"]):
            best = candidate
    return best


def _analysis_placeholders(sub_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    any_fitted = any(
        _as_dict(report.get("decision")).get("best_result") for report in sub_reports.values()
    )
    if any_fitted:
        status = "not_computed_in_current_lightweight_analysis"
        reason = "model fitting completed but fold-level feature attribution is not yet implemented"
    else:
        status = "skipped_no_completed_models"
        reason = "scikit-learn unavailable or all models skipped"
    skipped = {"status": status, "reason": reason}
    return {
        "feature_group_ablation": skipped,
        "permutation_importance": skipped,
        "top_30_features": [],
        "top_10_stable_features": [],
        "regime_specific_error_analysis": skipped,
        "special_band_error_analysis": skipped,
        "low_orientation_error_analysis": skipped,
        "review_files": {},
    }


def _skipped_analysis(skipped: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_group_ablation": skipped,
        "permutation_importance": skipped,
        "top_30_features": [],
        "top_10_stable_features": [],
        "regime_specific_error_analysis": skipped,
        "special_band_error_analysis": skipped,
        "low_orientation_error_analysis": skipped,
        "review_files": {},
    }


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
    normalized = []
    for row in rows:
        normalized.append({"csv_version": row.get("csv_version", BASELINE_CSV_VERSION), **row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)


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


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _none_safe_float(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return output if np.isfinite(output) else float("-inf")
