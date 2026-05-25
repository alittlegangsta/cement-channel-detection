from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.sample_schema import MVP4B_PREPROCESSING_DIAGNOSTICS_VERSION
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

DIAGNOSTIC_FILENAMES = {
    "feature_hist_raw_vs_log": "01_feature_hist_raw_vs_log.png",
    "feature_hist_scaled": "02_feature_hist_scaled.png",
    "candidate_vs_non_candidate_by_feature": "03_candidate_vs_non_candidate_by_feature.png",
    "sample_weight_distribution": "04_sample_weight_distribution.png",
    "depth_match_error_distribution": "05_depth_match_error_distribution.png",
}


@dataclass(frozen=True)
class FeaturePreprocessingDiagnosticsReport:
    diagnostics_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figures: dict[str, str]
    sample_count: int
    feature_names: list[str]
    transformed_feature_names: list[str]
    raw_vs_log: dict[str, dict[str, float | int | None]]
    scaled: dict[str, dict[str, float | int | None]]
    outliers: dict[str, dict[str, float | int | None]]
    nonfinite_counts: dict[str, dict[str, int | float | None]]
    standardized_differences: dict[str, dict[str, float | int | None]]
    sample_weight: dict[str, float | None]
    depth_match_error: dict[str, float | None]
    no_model_training: bool
    no_final_labels: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_feature_preprocessing(
    *,
    sample_table_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_samples: int = 20000,
) -> FeaturePreprocessingDiagnosticsReport:
    arrays = _load_npz(sample_table_npz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    no_model_training = bool(np.asarray(arrays.get("no_model_training", False)).reshape(()))
    no_final_labels = bool(np.asarray(arrays.get("no_final_labels", False)).reshape(()))
    if not no_model_training:
        errors.append("Sample table does not set no_model_training=true.")
    if not no_final_labels:
        errors.append("Sample table does not set no_final_labels=true.")
    features = np.asarray(arrays["features"], dtype=np.float32)
    transformed = np.asarray(arrays["transformed_features"], dtype=np.float32)
    feature_names = np.asarray(arrays["feature_names"]).astype(str).tolist()
    transformed_names = np.asarray(arrays["transformed_feature_names"]).astype(str).tolist()
    transform_stats = _load_transform_stats(arrays)
    presence = np.asarray(arrays["label_presence_plus"], dtype=np.int8)
    valid_azimuthal = np.asarray(arrays["valid_for_azimuthal_validation"], dtype=bool)
    disagreement = np.asarray(arrays["plus_minus_disagreement"], dtype=bool)
    sample_weight = np.asarray(arrays["sample_weight"], dtype=np.float32)
    depth_match_error = np.asarray(arrays["depth_match_error"], dtype=np.float32)
    figures = {key: output / filename for key, filename in DIAGNOSTIC_FILENAMES.items()}

    _save_raw_vs_log_hist(
        features,
        transformed,
        feature_names,
        transformed_names,
        figures["feature_hist_raw_vs_log"],
        max_samples=max_samples,
        overwrite=overwrite,
    )
    _save_scaled_hist(
        transformed,
        transformed_names,
        figures["feature_hist_scaled"],
        max_samples=max_samples,
        overwrite=overwrite,
    )
    standardized = standardized_differences_by_subset(
        transformed=transformed,
        transformed_names=transformed_names,
        presence=presence,
        valid_azimuthal=valid_azimuthal,
        disagreement=disagreement,
    )
    _save_standardized_difference_bar(
        standardized,
        figures["candidate_vs_non_candidate_by_feature"],
        overwrite=overwrite,
    )
    _save_histogram(
        sample_weight,
        figures["sample_weight_distribution"],
        title="Sample Weight Distribution",
        xlabel="sample_weight",
        overwrite=overwrite,
    )
    _save_histogram(
        depth_match_error,
        figures["depth_match_error_distribution"],
        title="Depth Match Error Distribution",
        xlabel="depth_match_error",
        overwrite=overwrite,
    )
    raw_vs_log = _raw_vs_log_summary(features, transformed, feature_names, transformed_names)
    scaled = _scaled_summary(transformed, transformed_names)
    outliers = _outlier_summary(transform_stats)
    nonfinite = _nonfinite_summary(features, transformed, feature_names, transformed_names)
    report = FeaturePreprocessingDiagnosticsReport(
        diagnostics_version=MVP4B_PREPROCESSING_DIAGNOSTICS_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={"sample_table_npz": str(sample_table_npz)},
        output_dir=str(output),
        figures={key: str(path) for key, path in figures.items()},
        sample_count=int(features.shape[0]),
        feature_names=feature_names,
        transformed_feature_names=transformed_names,
        raw_vs_log=raw_vs_log,
        scaled=scaled,
        outliers=outliers,
        nonfinite_counts=nonfinite,
        standardized_differences=standardized,
        sample_weight=_numeric_summary(sample_weight),
        depth_match_error=_numeric_summary(depth_match_error),
        no_model_training=no_model_training,
        no_final_labels=no_final_labels,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "model training",
            "train/test split",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "final label generation",
            "MVP-4C",
            "MVP-5",
        ],
    )
    return report


def write_preprocessing_diagnostics_outputs(
    report: FeaturePreprocessingDiagnosticsReport,
    *,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_preprocessing_diagnostics_markdown(report), encoding="utf-8")


def format_preprocessing_diagnostics_markdown(
    report: FeaturePreprocessingDiagnosticsReport,
) -> str:
    data = report.to_dict()
    lines = [
        "# MVP-4B Feature Preprocessing Diagnostics",
        "",
        f"- Version: {data['diagnostics_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Samples: {data['sample_count']}",
        f"- Output dir: {data['output_dir']}",
        f"- No model training: {data['no_model_training']}",
        f"- No final labels: {data['no_final_labels']}",
        "",
        "## Sample Weight",
        "",
    ]
    for key, value in data["sample_weight"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Depth Match Error", ""])
    for key, value in data["depth_match_error"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Outliers", ""])
    for feature, summary in data["outliers"].items():
        lines.append(
            f"- {feature}: clipped_before={summary['clipped_count_before']}, "
            f"clipped_after={summary['clipped_count_after']}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def standardized_differences_by_subset(
    *,
    transformed: np.ndarray,
    transformed_names: list[str],
    presence: np.ndarray,
    valid_azimuthal: np.ndarray,
    disagreement: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    subsets = {
        "all_known": presence >= 0,
        "high_confidence": valid_azimuthal & (presence >= 0),
        "low_confidence": (~valid_azimuthal) & (presence >= 0),
        "plus_minus_disagreement": disagreement & (presence >= 0),
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for subset_name, subset_mask in subsets.items():
        for feature_index, feature_name in enumerate(transformed_names):
            values = transformed[:, feature_index]
            candidate = subset_mask & (presence == 1) & np.isfinite(values)
            non_candidate = subset_mask & (presence == 0) & np.isfinite(values)
            key = f"{subset_name}:{feature_name}"
            result[key] = {
                "candidate_count": int(np.count_nonzero(candidate)),
                "non_candidate_count": int(np.count_nonzero(non_candidate)),
                "standardized_difference": _standardized_difference(
                    values[candidate],
                    values[non_candidate],
                ),
            }
    return result


def _raw_vs_log_summary(
    features: np.ndarray,
    transformed: np.ndarray,
    feature_names: list[str],
    transformed_names: list[str],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for feature_index, feature_name in enumerate(feature_names):
        log_name = f"log1p_{feature_name}"
        log_index = transformed_names.index(log_name) if log_name in transformed_names else None
        result[feature_name] = {
            "raw_finite_ratio": _finite_ratio(features[:, feature_index]),
            "raw_min": _numeric_summary(features[:, feature_index])["min"],
            "raw_max": _numeric_summary(features[:, feature_index])["max"],
            "log_finite_ratio": (
                _finite_ratio(transformed[:, log_index]) if log_index is not None else None
            ),
            "log_min": (
                _numeric_summary(transformed[:, log_index])["min"]
                if log_index is not None
                else None
            ),
            "log_max": (
                _numeric_summary(transformed[:, log_index])["max"]
                if log_index is not None
                else None
            ),
        }
    return result


def _scaled_summary(
    transformed: np.ndarray,
    transformed_names: list[str],
) -> dict[str, dict[str, float | int | None]]:
    return {
        name: _numeric_summary(transformed[:, index])
        for index, name in enumerate(transformed_names)
        if name.startswith("robust_scaled_")
    }


def _outlier_summary(
    transform_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for feature, stats in transform_stats.items():
        result[feature] = {
            "clipped_count_before": int(stats.get("clipped_count") or 0),
            "clipped_count_after": 0,
            "clip_low": _float_or_none(stats.get("clip_low")),
            "clip_high": _float_or_none(stats.get("clip_high")),
        }
    return result


def _nonfinite_summary(
    features: np.ndarray,
    transformed: np.ndarray,
    feature_names: list[str],
    transformed_names: list[str],
) -> dict[str, dict[str, int | float | None]]:
    result: dict[str, dict[str, int | float | None]] = {}
    for index, name in enumerate(feature_names):
        result[f"raw:{name}"] = _nonfinite_counts(features[:, index])
    for index, name in enumerate(transformed_names):
        result[f"transformed:{name}"] = _nonfinite_counts(transformed[:, index])
    return result


def _save_raw_vs_log_hist(
    features: np.ndarray,
    transformed: np.ndarray,
    feature_names: list[str],
    transformed_names: list[str],
    output_path: Path,
    *,
    max_samples: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    feature_name = (
        "late_over_early_ratio" if "late_over_early_ratio" in feature_names else feature_names[0]
    )
    feature_index = feature_names.index(feature_name)
    log_name = f"log1p_{feature_name}"
    log_index = transformed_names.index(log_name) if log_name in transformed_names else 0
    raw = _sample(features[:, feature_index], max_samples)
    log_values = _sample(transformed[:, log_index], max_samples)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(raw, bins=50, alpha=0.55, label=f"raw {feature_name}", color="tab:blue")
    ax.hist(log_values, bins=50, alpha=0.55, label=log_name, color="tab:orange")
    ax.set_title("Raw vs log1p distribution")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_scaled_hist(
    transformed: np.ndarray,
    transformed_names: list[str],
    output_path: Path,
    *,
    max_samples: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    scaled_names = [name for name in transformed_names if name.startswith("robust_scaled_")]
    target_name = (
        "robust_scaled_late_over_early_ratio"
        if "robust_scaled_late_over_early_ratio" in scaled_names
        else scaled_names[0]
    )
    values = _sample(transformed[:, transformed_names.index(target_name)], max_samples)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(values, bins=60, color="tab:green", alpha=0.75)
    ax.set_title("Robust scaled distribution")
    ax.set_xlabel(target_name)
    ax.set_ylabel("Count")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_standardized_difference_bar(
    standardized: dict[str, dict[str, float | int | None]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    rows = [
        (key.split(":", 1)[1], value["standardized_difference"])
        for key, value in standardized.items()
        if key.startswith("high_confidence:") and value["standardized_difference"] is not None
    ]
    if not rows:
        rows = [("no_data", 0.0)]
    names = [row[0] for row in rows]
    values = [float(row[1]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color="tab:purple")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("High-confidence candidate vs non-candidate standardized difference")
    ax.set_xticks(np.arange(len(values)), labels=names, rotation=35, ha="right")
    ax.set_ylabel("standardized difference")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_histogram(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    sampled = _sample(values, max_samples=50000)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(sampled, bins=60, color="tab:blue", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    save_figure(fig, output_path, overwrite=overwrite)


def _standardized_difference(candidate: np.ndarray, non_candidate: np.ndarray) -> float | None:
    if candidate.size < 2 or non_candidate.size < 2:
        return None
    candidate_std = float(np.var(candidate, ddof=1))
    non_candidate_std = float(np.var(non_candidate, ddof=1))
    pooled = np.sqrt((candidate_std + non_candidate_std) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0.0:
        return None
    return float((np.mean(candidate) - np.mean(non_candidate)) / pooled)


def _numeric_summary(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {"finite_ratio": None, "min": None, "max": None, "mean": None, "median": None}
    finite = np.isfinite(array)
    finite_ratio = float(np.mean(finite))
    if not np.any(finite):
        return {
            "finite_ratio": finite_ratio,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    finite_values = array[finite]
    return {
        "finite_ratio": finite_ratio,
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def _nonfinite_counts(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    return {
        "total": int(array.size),
        "nonfinite": int(np.count_nonzero(~finite)),
        "finite_ratio": float(np.mean(finite)) if array.size else None,
    }


def _finite_ratio(values: np.ndarray) -> float | None:
    array = np.asarray(values)
    if array.size == 0:
        return None
    return float(np.mean(np.isfinite(array)))


def _sample(values: np.ndarray, max_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.array([np.nan], dtype=np.float32)
    if array.size <= max_samples:
        return array
    indices = np.linspace(0, array.size - 1, num=max_samples).astype(int)
    return array[indices]


def _load_transform_stats(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    if "transform_stats_json" not in arrays:
        return {}
    return json.loads(str(np.asarray(arrays["transform_stats_json"]).reshape(())))


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None
