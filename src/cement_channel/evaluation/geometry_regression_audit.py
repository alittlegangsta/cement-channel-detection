from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

GEOMETRY_REGRESSION_AUDIT_VERSION = "geometry_regression_audit_v001"
DEFAULT_KERNELS = (
    "r7_reference_point",
    "midpoint_window",
    "uniform_source_receiver_interval",
    "triangular_midpoint_weighted",
)


@dataclass(frozen=True)
class GeometryRegressionAuditConfig:
    target_field: str = "receiver_max"
    fold_count: int = 3
    permutation_repeats: int = 11
    min_abs_spearman_margin: float = 0.03
    review_band_min_ft: float = 5680.0
    review_band_max_ft: float = 5720.0


@dataclass(frozen=True)
class GeometryRegressionAuditReport:
    audit_version: str
    generated_at: str
    inputs: dict[str, str]
    output_csv: str
    target_field: str
    kernel_summaries: list[dict[str, Any]]
    kernel_sensitivity: list[dict[str, Any]]
    best_kernel: str | None
    recommendation: str
    recommendation_reason: str
    warnings: list[str]
    errors: list[str]
    no_model_training: bool
    no_model_weights: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_geometry_regression_from_paths(
    *,
    regression_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    regression_config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
    config: GeometryRegressionAuditConfig | None = None,
) -> GeometryRegressionAuditReport:
    label_arrays = _load_npz(regression_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    report, rows = audit_geometry_regression(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        config=config or GeometryRegressionAuditConfig(),
        inputs={
            "regression_labels_npz": str(regression_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "regression_config_path": str(regression_config_path),
        },
        output_csv=Path(output_csv),
    )
    write_geometry_regression_audit_outputs(
        report,
        rows,
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def audit_geometry_regression(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    config: GeometryRegressionAuditConfig | None = None,
    inputs: dict[str, str] | None = None,
    output_csv: Path | None = None,
) -> tuple[GeometryRegressionAuditReport, list[dict[str, Any]]]:
    cfg = config or GeometryRegressionAuditConfig()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(label_arrays, feature_arrays, cfg, errors)
    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    if depth.size != feature_depth.size:
        raise ValueError("Regression label and feature depth counts differ.")
    if not np.allclose(depth, feature_depth, atol=1e-3):
        warnings.append("Regression label and feature depths differ; using label order.")
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    finite_rows = np.all(np.isfinite(features), axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    leakage_warnings = _feature_leakage_warnings(feature_names)
    warnings.extend(leakage_warnings)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    kernels = label_arrays["geometry_kernel"].astype(str)
    targets = np.asarray(label_arrays[cfg.target_field], dtype=np.float32)
    for kernel_index, kernel in enumerate(kernels):
        target = targets[kernel_index]
        selected = finite_rows & np.isfinite(target)
        summary, feature_rows = _audit_kernel(
            kernel=kernel,
            kernel_index=kernel_index,
            depth=depth,
            features=features,
            feature_names=feature_names,
            target=target,
            selected=selected,
            config=cfg,
            leakage_warnings=leakage_warnings,
        )
        summaries.append(summary)
        rows.extend(feature_rows)
    sensitivity = _kernel_sensitivity(summaries)
    best = _best_kernel(summaries)
    recommendation, reason = _recommendation(summaries, best, config=cfg)
    if recommendation == "stop":
        errors.append(reason)
    report = GeometryRegressionAuditReport(
        audit_version=GEOMETRY_REGRESSION_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_csv=str(output_csv) if output_csv else "",
        target_field=cfg.target_field,
        kernel_summaries=summaries,
        kernel_sensitivity=sensitivity,
        best_kernel=None if best is None else str(best["geometry_kernel"]),
        recommendation=recommendation,
        recommendation_reason=reason,
        warnings=warnings,
        errors=errors,
        no_model_training=True,
        no_model_weights=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "formal model training",
            "hyperparameter tuning",
            "model weight export",
            "production metric claim",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, rows


def write_geometry_regression_audit_outputs(
    report: GeometryRegressionAuditReport,
    rows: list[dict[str, Any]],
    *,
    output_md: Path,
    output_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_csv, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_geometry_regression_audit_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_geometry_regression_audit_markdown(report: GeometryRegressionAuditReport) -> str:
    lines = [
        "# Geometry-Aware Regression Alignment Audit",
        "",
        "This is a sanity audit for continuous CAST weak-label candidates. It is "
        "not model performance, does not export weights, and does not authorize "
        "MVP-4C, STC/APES, deep learning, production use, or final labels.",
        "",
        f"- recommendation: `{report.recommendation}`",
        f"- recommendation_reason: {report.recommendation_reason}",
        f"- best_kernel: `{report.best_kernel}`",
        f"- target_field: `{report.target_field}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Kernel Summaries",
        "",
    ]
    for row in report.kernel_summaries:
        lines.append(
            "- "
            f"{row['geometry_kernel']}: sample_count={row['sample_count']}, "
            f"best_feature={row['best_feature_name']}, "
            f"spearman={row['spearman_correlation']}, "
            f"permutation_spearman={row['permutation_spearman_correlation']}, "
            f"margin={row['real_minus_permutation_margin']}, "
            f"cv_mae={row['cross_validated_mae']}, "
            f"cv_r2={row['cross_validated_r2_sanity']}, "
            f"perm_cv_r2={row['permutation_probe_r2']}, "
            f"fold_stability={row['fold_stability']}, "
            f"depends_on_5700_band={row['depends_on_5700_band']}"
        )
    lines.extend(["", "## Kernel Sensitivity", ""])
    lines.extend(_dict_message_lines(report.kernel_sensitivity))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _audit_kernel(
    *,
    kernel: str,
    kernel_index: int,
    depth: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    config: GeometryRegressionAuditConfig,
    leakage_warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_count = int(np.count_nonzero(selected))
    base = {
        "geometry_kernel": kernel,
        "kernel_index": kernel_index,
        "sample_count": sample_count,
        "target_distribution": _numeric_distribution(target[selected]),
        "zero_fraction": _fraction(target[selected] <= 0.0),
        "nonzero_fraction": _fraction(target[selected] > 0.0),
        "depends_on_5700_band": _depends_on_review_band(
            depth[selected],
            target[selected],
            config,
        ),
        "leakage_warnings": leakage_warnings,
    }
    if sample_count < max(config.fold_count, 3) or np.nanstd(target[selected]) <= 1e-8:
        return (
            {
                **base,
                "status": "skipped_constant_or_too_small",
                "best_feature_name": None,
                "best_feature_index": None,
                "spearman_correlation": None,
                "pearson_correlation": None,
                "permutation_spearman_correlation": None,
                "permutation_probe_r2": None,
                "real_minus_permutation_margin": None,
                "simple_linear_sanity_probe": {},
                "cross_validated_mae": None,
                "cross_validated_r2_sanity": None,
                "fold_stability": 0.0,
                "sample_support_collapse": True,
            },
            [],
        )
    feature_rows = _feature_correlation_rows(
        kernel=kernel,
        features=features[selected],
        feature_names=feature_names,
        target=target[selected],
    )
    best = max(feature_rows, key=lambda row: abs(float(row["spearman_correlation"] or 0.0)))
    probe = _linear_probe_with_permutation(
        features=features[selected],
        target=target[selected],
        depth=depth[selected],
        config=config,
        seed=_stable_seed(kernel),
    )
    permutation_spearman = _permutation_spearman(
        values=features[selected, int(best["feature_index"])],
        target=target[selected],
        repeats=config.permutation_repeats,
        seed=_stable_seed(kernel) + 41,
    )
    spearman = _as_float(best["spearman_correlation"])
    margin = None if spearman is None else abs(spearman) - abs(permutation_spearman)
    return (
        {
            **base,
            "status": "runnable",
            "best_feature_name": best["feature_name"],
            "best_feature_index": int(best["feature_index"]),
            "spearman_correlation": spearman,
            "pearson_correlation": _as_float(best["pearson_correlation"]),
            "permutation_spearman_correlation": permutation_spearman,
            "permutation_probe_r2": probe["permutation_r2"],
            "real_minus_permutation_margin": margin,
            "simple_linear_sanity_probe": probe,
            "cross_validated_mae": probe["mae"],
            "cross_validated_r2_sanity": probe["r2"],
            "fold_stability": probe["fold_stability"],
            "sample_support_collapse": False,
        },
        feature_rows,
    )


def _feature_correlation_rows(
    *,
    kernel: str,
    features: np.ndarray,
    feature_names: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names.astype(str)):
        values = features[:, index]
        rows.append(
            {
                "geometry_kernel": kernel,
                "feature_index": index,
                "feature_name": name,
                "pearson_correlation": _pearson(values, target),
                "spearman_correlation": _spearman(values, target),
            }
        )
    return rows


def _linear_probe_with_permutation(
    *,
    features: np.ndarray,
    target: np.ndarray,
    depth: np.ndarray,
    config: GeometryRegressionAuditConfig,
    seed: int,
) -> dict[str, Any]:
    prediction, fold_metrics = _cross_validated_linear_predictions(
        features,
        target,
        depth,
        fold_count=config.fold_count,
    )
    mae = _mae(target, prediction)
    r2 = _r2(target, prediction)
    rng = np.random.default_rng(seed)
    permutation_mae: list[float] = []
    permutation_r2: list[float] = []
    folds_above = 0
    valid_folds = 0
    for _ in range(config.permutation_repeats):
        permuted = target.copy()
        rng.shuffle(permuted)
        perm_prediction, perm_fold_metrics = _cross_validated_linear_predictions(
            features,
            permuted,
            depth,
            fold_count=config.fold_count,
        )
        perm_mae = _mae(permuted, perm_prediction)
        perm_r2 = _r2(permuted, perm_prediction)
        if perm_mae is not None:
            permutation_mae.append(perm_mae)
        if perm_r2 is not None:
            permutation_r2.append(perm_r2)
        for real_fold, perm_fold in zip(fold_metrics, perm_fold_metrics, strict=False):
            if real_fold["mae"] is None or perm_fold["mae"] is None:
                continue
            valid_folds += 1
            if real_fold["mae"] < perm_fold["mae"]:
                folds_above += 1
    permutation_mae_mean = None if not permutation_mae else float(np.mean(permutation_mae))
    permutation_r2_mean = None if not permutation_r2 else float(np.mean(permutation_r2))
    return {
        "mae": mae,
        "r2": r2,
        "permutation_mae": permutation_mae_mean,
        "permutation_r2": permutation_r2_mean,
        "real_minus_permutation_r2": (
            None if r2 is None or permutation_r2_mean is None else r2 - permutation_r2_mean
        ),
        "folds_real_mae_better_than_permutation": folds_above,
        "valid_fold_comparisons": valid_folds,
        "fold_stability": 0.0 if valid_folds == 0 else folds_above / valid_folds,
        "fold_metrics": fold_metrics,
    }


def _cross_validated_linear_predictions(
    features: np.ndarray,
    target: np.ndarray,
    depth: np.ndarray,
    *,
    fold_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prediction = np.full(target.shape, np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    order = np.argsort(depth)
    splits = np.array_split(order, fold_count)
    for fold_index, test_index in enumerate(splits):
        train_index = np.setdiff1d(np.arange(target.size), test_index, assume_unique=False)
        if test_index.size == 0 or train_index.size < 2:
            fold_metrics.append({"fold": fold_index, "mae": None, "r2": None})
            continue
        train_x, test_x = _standardize_train_test(features[train_index], features[test_index])
        train_design = np.column_stack([np.ones(train_x.shape[0]), train_x])
        test_design = np.column_stack([np.ones(test_x.shape[0]), test_x])
        coef, *_ = np.linalg.lstsq(train_design, target[train_index], rcond=None)
        prediction[test_index] = test_design @ coef
        fold_metrics.append(
            {
                "fold": fold_index,
                "mae": _mae(target[test_index], prediction[test_index]),
                "r2": _r2(target[test_index], prediction[test_index]),
            }
        )
    fallback = float(np.nanmean(target))
    prediction = np.where(np.isfinite(prediction), prediction, fallback).astype(np.float32)
    return prediction, fold_metrics


def _standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return (train - mean) / std, (test - mean) / std


def _kernel_sensitivity(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r7 = _summary_for(summaries, "r7_reference_point")
    if r7 is None:
        return []
    r7_margin = _as_float(r7.get("real_minus_permutation_margin"))
    r7_nonzero = _as_float(r7.get("nonzero_fraction"))
    rows = []
    for summary in summaries:
        margin = _as_float(summary.get("real_minus_permutation_margin"))
        nonzero = _as_float(summary.get("nonzero_fraction"))
        rows.append(
            {
                "geometry_kernel": summary["geometry_kernel"],
                "margin_delta_vs_r7": (
                    None if margin is None or r7_margin is None else margin - r7_margin
                ),
                "nonzero_fraction_delta_vs_r7": (
                    None if nonzero is None or r7_nonzero is None else nonzero - r7_nonzero
                ),
            }
        )
    return rows


def _recommendation(
    summaries: list[dict[str, Any]],
    best: dict[str, Any] | None,
    *,
    config: GeometryRegressionAuditConfig,
) -> tuple[str, str]:
    runnable = [row for row in summaries if row["status"] == "runnable"]
    if not runnable:
        return "stop", "No geometry regression kernel had enough non-constant samples."
    stable = [
        row
        for row in runnable
        if (_as_float(row.get("real_minus_permutation_margin")) or -1.0)
        >= config.min_abs_spearman_margin
        and (_as_float(row.get("fold_stability")) or 0.0) >= 0.5
    ]
    if not stable:
        return "stop", "All geometry regression kernels are unstable versus permutation."
    r7 = _summary_for(summaries, "r7_reference_point")
    r7_margin = None if r7 is None else _as_float(r7.get("real_minus_permutation_margin"))
    for candidate in ("triangular_midpoint_weighted", "midpoint_window"):
        row = _summary_for(summaries, candidate)
        margin = None if row is None else _as_float(row.get("real_minus_permutation_margin"))
        if row in stable and margin is not None and r7_margin is not None and margin > r7_margin:
            return (
                "human_review_primary_candidate",
                f"{candidate} is stable and improves over the R7 reference sanity baseline.",
            )
    uniform = _summary_for(summaries, "uniform_source_receiver_interval")
    if uniform is not None:
        distribution = _as_dict(uniform.get("target_distribution"))
        p90 = _as_float(distribution.get("p90"))
        median = _as_float(distribution.get("median"))
        if p90 is not None and median is not None and abs(p90 - median) < 1e-6:
            return (
                "r7_or_midpoint_review",
                "Uniform interval distribution is overly concentrated; keep it audit-only.",
            )
    if best is None:
        return "stop", "No best stable kernel could be selected."
    return (
        "human_review_required",
        f"{best['geometry_kernel']} has the best sanity margin; wait for manual review.",
    )


def _best_kernel(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    runnable = [
        row
        for row in summaries
        if row["status"] == "runnable"
        and _as_float(row.get("real_minus_permutation_margin")) is not None
    ]
    if not runnable:
        return None
    return max(runnable, key=lambda row: float(row["real_minus_permutation_margin"]))


def _summary_for(summaries: list[dict[str, Any]], kernel: str) -> dict[str, Any] | None:
    for row in summaries:
        if row["geometry_kernel"] == kernel:
            return row
    return None


def _permutation_spearman(
    *,
    values: np.ndarray,
    target: np.ndarray,
    repeats: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        permuted = target.copy()
        rng.shuffle(permuted)
        value = _spearman(values, permuted)
        if value is not None:
            scores.append(abs(value))
    return 0.0 if not scores else float(np.mean(scores))


def _pearson(values: np.ndarray, target: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return None
    x = x[finite]
    y = y[finite]
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(values: np.ndarray, target: np.ndarray) -> float | None:
    x = _rank(values)
    y = _rank(target)
    return _pearson(x, y)


def _rank(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    return ranks


def _mae(target: np.ndarray, prediction: np.ndarray) -> float | None:
    finite = np.isfinite(target) & np.isfinite(prediction)
    if not np.any(finite):
        return None
    return float(np.mean(np.abs(target[finite] - prediction[finite])))


def _r2(target: np.ndarray, prediction: np.ndarray) -> float | None:
    finite = np.isfinite(target) & np.isfinite(prediction)
    if np.count_nonzero(finite) < 2:
        return None
    y = target[finite]
    pred = prediction[finite]
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return None
    return float(1.0 - np.sum((y - pred) ** 2) / total)


def _numeric_distribution(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _depends_on_review_band(
    depth: np.ndarray,
    target: np.ndarray,
    config: GeometryRegressionAuditConfig,
) -> bool:
    positive = target > 0.0
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        return False
    review = (depth >= config.review_band_min_ft) & (depth <= config.review_band_max_ft)
    return bool(np.count_nonzero(positive & review) / positive_count > 0.5)


def _validate_guardrails(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    config: GeometryRegressionAuditConfig,
    errors: list[str],
) -> None:
    required_labels = ("depth", "geometry_kernel", config.target_field, "no_final_labels")
    required_features = ("depth", "depth_level_xsi_features", "depth_level_xsi_feature_names")
    missing_labels = [key for key in required_labels if key not in label_arrays]
    missing_features = [key for key in required_features if key not in feature_arrays]
    if missing_labels:
        raise KeyError("Regression label NPZ missing field(s): " + ", ".join(missing_labels))
    if missing_features:
        raise KeyError("Depth-level feature NPZ missing field(s): " + ", ".join(missing_features))
    if not bool(np.asarray(label_arrays.get("no_final_labels", False))):
        errors.append("Regression labels must preserve no_final_labels=true.")
    if not bool(np.asarray(feature_arrays.get("no_final_labels", False))):
        errors.append("Depth-level features must preserve no_final_labels=true.")


def _feature_leakage_warnings(feature_names: np.ndarray) -> list[str]:
    suspicious = [
        name
        for name in feature_names.astype(str)
        if any(token in name.lower() for token in ("cast", "label", "zc"))
    ]
    if not suspicious:
        return []
    return ["Potential leakage feature names found: " + ", ".join(suspicious[:10])]


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "geometry_kernel",
        "feature_index",
        "feature_name",
        "pearson_correlation",
        "spearman_correlation",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stable_seed(text: str) -> int:
    total = 123
    for char in text:
        total = (total * 131 + ord(char)) % (2**32 - 1)
    return int(total)


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _dict_message_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row}" for row in rows]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
