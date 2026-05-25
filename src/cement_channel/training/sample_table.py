from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.sample_schema import (
    MVP4B_SAMPLE_TABLE_VERSION,
    SampleTableConfig,
    load_sample_table_config,
    transformed_feature_names,
)


@dataclass(frozen=True)
class BaselineSampleTableReport:
    sample_table_version: str
    generated_at: str
    inputs: dict[str, str]
    shape: dict[str, int]
    feature_names: list[str]
    transformed_feature_names: list[str]
    counts: dict[str, int | float | None]
    excluded_counts: dict[str, int]
    feature_finite_ratio: dict[str, float | None]
    raw_feature_ranges: dict[str, dict[str, float | None]]
    transformed_feature_ranges: dict[str, dict[str, float | None]]
    label_distribution: dict[str, int]
    severity_distribution: dict[str, int]
    depth_match_error: dict[str, float | None]
    sample_weight: dict[str, float | None]
    transform_stats: dict[str, dict[str, float | int | None]]
    no_model_training: bool
    no_final_labels: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_baseline_sample_table_from_config(
    *,
    label_samples_npz: Path | str,
    basic_features_npz: Path | str,
    sample_config_path: Path | str,
) -> tuple[BaselineSampleTableReport, dict[str, np.ndarray]]:
    return build_baseline_sample_table(
        label_samples_npz=label_samples_npz,
        basic_features_npz=basic_features_npz,
        sample_config=load_sample_table_config(sample_config_path),
        sample_config_path=sample_config_path,
    )


def build_baseline_sample_table(
    *,
    label_samples_npz: Path | str,
    basic_features_npz: Path | str,
    sample_config: SampleTableConfig,
    sample_config_path: Path | str | None = None,
) -> tuple[BaselineSampleTableReport, dict[str, np.ndarray]]:
    label_arrays = _load_npz(label_samples_npz)
    feature_arrays = _load_npz(basic_features_npz)
    arrays, stats = build_baseline_sample_table_from_arrays(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        sample_config=sample_config,
    )
    report = BaselineSampleTableReport(
        sample_table_version=MVP4B_SAMPLE_TABLE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "label_samples_npz": str(label_samples_npz),
            "basic_features_npz": str(basic_features_npz),
            "sample_config_path": str(sample_config_path) if sample_config_path else "",
        },
        shape={
            "samples": int(arrays["sample_id"].shape[0]),
            "features": int(arrays["features"].shape[1]),
            "transformed_features": int(arrays["transformed_features"].shape[1]),
        },
        feature_names=arrays["feature_names"].astype(str).tolist(),
        transformed_feature_names=arrays["transformed_feature_names"].astype(str).tolist(),
        counts=stats["counts"],
        excluded_counts=stats["excluded_counts"],
        feature_finite_ratio=stats["feature_finite_ratio"],
        raw_feature_ranges=stats["raw_feature_ranges"],
        transformed_feature_ranges=stats["transformed_feature_ranges"],
        label_distribution=stats["label_distribution"],
        severity_distribution=stats["severity_distribution"],
        depth_match_error=stats["depth_match_error"],
        sample_weight=stats["sample_weight"],
        transform_stats=stats["transform_stats"],
        no_model_training=True,
        no_final_labels=True,
        warnings=stats["warnings"],
        errors=stats["errors"],
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
    return report, arrays


def build_baseline_sample_table_from_arrays(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    sample_config: SampleTableConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    warnings: list[str] = []
    errors: list[str] = []
    labels = _label_arrays(label_arrays)
    feature_names = np.asarray(feature_arrays["feature_names"]).astype(str).tolist()
    feature_cube = np.asarray(feature_arrays["xsi_basic_features_by_side"], dtype=np.float32)
    if feature_cube.ndim != 3:
        raise ValueError("xsi_basic_features_by_side must have shape [depth, side, feature].")
    requested_feature_indices = _feature_indices(feature_names, sample_config.feature_names)
    if labels["presence_plus"].shape != feature_cube.shape[:2]:
        errors.append(
            "label/feature shape mismatch: "
            f"labels={labels['presence_plus'].shape}, features={feature_cube.shape[:2]}."
        )
    depth_count = min(labels["presence_plus"].shape[0], feature_cube.shape[0])
    side_count = min(labels["presence_plus"].shape[1], feature_cube.shape[1])
    feature_cube = feature_cube[:depth_count, :side_count, requested_feature_indices]
    labels = {
        key: value[:depth_count, :side_count] if value.ndim == 2 else value[:depth_count]
        for key, value in labels.items()
    }
    no_final_labels = bool(np.asarray(label_arrays.get("no_final_labels", False)).reshape(()))
    no_model_training = bool(np.asarray(feature_arrays.get("no_model_training", False)).reshape(()))
    if not no_final_labels:
        errors.append("label sample artifact does not set no_final_labels=true.")
    if not no_model_training:
        errors.append("feature artifact does not set no_model_training=true.")

    sample_count = depth_count * side_count
    depth = np.repeat(labels["depth"], side_count).astype(np.float32)
    side_index = np.tile(np.arange(side_count, dtype=np.int16), depth_count)
    side_azimuth = np.tile(labels["side_azimuth"], depth_count).astype(np.float32)
    features = feature_cube.reshape(sample_count, len(requested_feature_indices)).astype(np.float32)
    raw_feature_names = tuple(sample_config.feature_names)
    transformed, transform_stats = transform_features(features, raw_feature_names, sample_config)
    presence_plus = labels["presence_plus"].reshape(-1).astype(np.int8)
    severity_plus = labels["severity_plus"].reshape(-1).astype(np.int8)
    confidence_plus = labels["confidence_plus"].reshape(-1).astype(np.float32)
    presence_minus = labels["presence_minus"].reshape(-1).astype(np.int8)
    disagreement = labels["disagreement"].reshape(-1).astype(bool)
    orientation_confidence = labels["orientation_confidence"].reshape(-1).astype(np.float32)
    valid_azimuthal = labels["valid_azimuthal"].reshape(-1).astype(bool)
    valid_non_azimuthal = labels["valid_non_azimuthal"].reshape(-1).astype(bool)
    depth_match_error = np.repeat(labels["depth_match_error"], side_count).astype(np.float32)
    feature_finite = np.all(np.isfinite(features), axis=1)
    transformed_finite = np.all(np.isfinite(transformed), axis=1)
    large_depth_error = _large_depth_error(depth_match_error, sample_config)
    exclude_nonfinite = ~(feature_finite & transformed_finite)
    exclude_large_depth_error = sample_config.exclude_large_depth_match_error & large_depth_error
    sample_weight = compute_sample_weight(
        label_confidence=confidence_plus,
        valid_for_azimuthal=valid_azimuthal,
        plus_minus_disagreement=disagreement,
        large_depth_error=large_depth_error,
        feature_valid=feature_finite & transformed_finite,
        sample_config=sample_config,
    )
    azimuthal_sample_weight = np.where(valid_azimuthal, sample_weight, 0.0).astype(np.float32)
    included_for_azimuthal = (azimuthal_sample_weight > 0.0) & (presence_plus >= 0)
    arrays = {
        "sample_id": np.arange(sample_count, dtype=np.int64),
        "depth": depth,
        "side_index": side_index,
        "side_azimuth_deg": side_azimuth,
        "label_presence_plus": presence_plus,
        "label_severity_plus": severity_plus,
        "label_confidence_plus": confidence_plus,
        "label_presence_minus_audit": presence_minus,
        "plus_minus_disagreement": disagreement,
        "orientation_confidence": orientation_confidence,
        "valid_for_azimuthal_validation": valid_azimuthal,
        "valid_for_non_azimuthal_summary": valid_non_azimuthal,
        "depth_match_error": depth_match_error,
        "sample_weight": sample_weight,
        "azimuthal_sample_weight": azimuthal_sample_weight,
        "included_for_azimuthal_baseline": included_for_azimuthal,
        "audit_flag_plus_minus_disagreement": disagreement,
        "exclude_nonfinite_feature": exclude_nonfinite,
        "exclude_large_depth_match_error": exclude_large_depth_error,
        "features": features,
        "feature_names": np.asarray(raw_feature_names),
        "transformed_features": transformed.astype(np.float32),
        "transformed_feature_names": np.asarray(transformed_feature_names(sample_config)),
        "transform_stats_json": np.asarray(json.dumps(transform_stats, ensure_ascii=False)),
        "no_model_training": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
    }
    metadata = {
        "sample_table_version": MVP4B_SAMPLE_TABLE_VERSION,
        "input_features": sample_config.input_features,
        "input_labels": sample_config.input_labels,
        "primary_label": sample_config.primary_label,
        "audit_label": sample_config.audit_label,
        "no_model_training": True,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    stats = _stats(
        arrays,
        transform_stats=transform_stats,
        warnings=warnings,
        errors=errors,
    )
    return arrays, stats


def transform_features(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    config: SampleTableConfig,
) -> tuple[np.ndarray, dict[str, dict[str, float | int | None]]]:
    values = np.asarray(features, dtype=np.float32)
    transformed_parts: list[np.ndarray] = []
    stats: dict[str, dict[str, float | int | None]] = {}
    log_values = np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    if config.log1p:
        transformed_parts.append(log_values)
    scaled = np.empty_like(log_values, dtype=np.float32)
    for feature_index, feature_name in enumerate(feature_names):
        source = log_values[:, feature_index]
        finite = source[np.isfinite(source)]
        if finite.size == 0:
            q_low = q_high = median = iqr = None
            scaled[:, feature_index] = np.nan
            clipped_count = int(source.size)
        else:
            q_low, q_high = np.quantile(finite, config.clip_quantiles)
            clipped = np.clip(source, q_low, q_high)
            median = float(np.median(clipped[np.isfinite(clipped)]))
            q25, q75 = np.quantile(clipped[np.isfinite(clipped)], [0.25, 0.75])
            iqr = float(q75 - q25)
            scale = iqr if iqr > 0.0 else 1.0
            scaled[:, feature_index] = ((clipped - median) / scale).astype(np.float32)
            clipped_count = int(np.count_nonzero((source < q_low) | (source > q_high)))
        stats[feature_name] = {
            "clip_low": _float_or_none(q_low),
            "clip_high": _float_or_none(q_high),
            "median": _float_or_none(median),
            "iqr": _float_or_none(iqr),
            "clipped_count": clipped_count,
            "finite_count": int(finite.size),
        }
    if config.robust_scaling:
        transformed_parts.append(scaled)
    if not transformed_parts:
        return np.empty((values.shape[0], 0), dtype=np.float32), stats
    return np.concatenate(transformed_parts, axis=1).astype(np.float32), stats


def compute_sample_weight(
    *,
    label_confidence: np.ndarray,
    valid_for_azimuthal: np.ndarray,
    plus_minus_disagreement: np.ndarray,
    large_depth_error: np.ndarray,
    feature_valid: np.ndarray,
    sample_config: SampleTableConfig,
) -> np.ndarray:
    weight = np.asarray(label_confidence, dtype=np.float32).copy()
    weight = np.where(valid_for_azimuthal, weight, 0.0)
    weight = np.where(
        plus_minus_disagreement,
        weight * np.float32(sample_config.plus_minus_disagreement_weight_multiplier),
        weight,
    )
    weight = np.where(
        large_depth_error,
        weight * np.float32(sample_config.depth_mismatch_weight_multiplier),
        weight,
    )
    weight = np.where(feature_valid, weight, 0.0)
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def write_baseline_sample_table_outputs(
    report: BaselineSampleTableReport,
    arrays: dict[str, np.ndarray],
    *,
    output_npz: Path,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_baseline_sample_table_markdown(report), encoding="utf-8")


def format_baseline_sample_table_markdown(report: BaselineSampleTableReport) -> str:
    data = report.to_dict()
    lines = [
        "# MVP-4B Baseline Sample Table Report",
        "",
        f"- Version: {data['sample_table_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Samples: {data['shape']['samples']}",
        f"- Features: {data['shape']['features']}",
        f"- Transformed features: {data['shape']['transformed_features']}",
        f"- No model training: {data['no_model_training']}",
        f"- No final labels: {data['no_final_labels']}",
        "",
        "## Counts",
        "",
    ]
    for key, value in data["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Excluded Counts", ""])
    for key, value in data["excluded_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Label Distribution", ""])
    for key, value in data["label_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Severity Distribution", ""])
    for key, value in data["severity_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Sample Weight", ""])
    for key, value in data["sample_weight"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Depth Match Error", ""])
    for key, value in data["depth_match_error"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _label_arrays(label_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = np.asarray(label_arrays["xsi_depth"], dtype=np.float32)
    side_azimuth = np.asarray(label_arrays["xsi_side_azimuth_deg"], dtype=np.float32)
    presence = np.asarray(label_arrays["label_presence_plus"], dtype=np.int8)
    depth_match = np.asarray(
        label_arrays.get("cast_depth_mismatch", np.zeros(depth.shape, dtype=np.float32)),
        dtype=np.float32,
    )
    return {
        "depth": depth,
        "side_azimuth": side_azimuth,
        "presence_plus": presence,
        "severity_plus": np.asarray(label_arrays["label_severity_plus"], dtype=np.int8),
        "confidence_plus": np.asarray(label_arrays["label_confidence_plus"], dtype=np.float32),
        "presence_minus": np.asarray(label_arrays["label_presence_minus_audit"], dtype=np.int8),
        "disagreement": np.asarray(label_arrays["plus_minus_disagreement"], dtype=bool),
        "orientation_confidence": np.asarray(
            label_arrays["orientation_confidence"],
            dtype=np.float32,
        ),
        "valid_azimuthal": np.asarray(
            label_arrays["valid_for_azimuthal_validation"],
            dtype=bool,
        ),
        "valid_non_azimuthal": np.asarray(
            label_arrays["valid_for_non_azimuthal_summary"],
            dtype=bool,
        ),
        "depth_match_error": depth_match,
    }


def _feature_indices(feature_names: list[str], requested: tuple[str, ...]) -> list[int]:
    missing = [feature for feature in requested if feature not in feature_names]
    if missing:
        raise ValueError("Missing requested feature(s): " + ", ".join(missing))
    return [feature_names.index(feature) for feature in requested]


def _large_depth_error(values: np.ndarray, config: SampleTableConfig) -> np.ndarray:
    if config.max_depth_match_error_ft is None:
        return np.zeros(values.shape, dtype=bool)
    return np.isfinite(values) & (np.abs(values) > config.max_depth_match_error_ft)


def _stats(
    arrays: dict[str, np.ndarray],
    *,
    transform_stats: dict[str, dict[str, float | int | None]],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    presence = arrays["label_presence_plus"]
    severity = arrays["label_severity_plus"]
    valid_az = arrays["valid_for_azimuthal_validation"]
    sample_weight = arrays["sample_weight"]
    positive_weight = sample_weight > 0.0
    candidate = presence == 1
    non_candidate = presence == 0
    transformed = arrays["transformed_features"]
    features = arrays["features"]
    counts = {
        "total_samples": int(presence.size),
        "candidate_count": int(np.count_nonzero(candidate)),
        "non_candidate_count": int(np.count_nonzero(non_candidate)),
        "unknown_count": int(np.count_nonzero(presence < 0)),
        "high_confidence_candidate_count": int(np.count_nonzero(valid_az & candidate)),
        "high_confidence_non_candidate_count": int(np.count_nonzero(valid_az & non_candidate)),
        "positive_sample_weight_count": int(np.count_nonzero(positive_weight)),
        "positive_sample_weight_fraction": (
            float(np.mean(positive_weight)) if positive_weight.size else None
        ),
        "plus_minus_disagreement_fraction": (
            float(np.mean(arrays["plus_minus_disagreement"]))
            if arrays["plus_minus_disagreement"].size
            else None
        ),
    }
    return {
        "counts": counts,
        "excluded_counts": {
            "exclude_nonfinite_feature": int(np.count_nonzero(arrays["exclude_nonfinite_feature"])),
            "exclude_large_depth_match_error": int(
                np.count_nonzero(arrays["exclude_large_depth_match_error"])
            ),
            "zero_sample_weight": int(np.count_nonzero(sample_weight <= 0.0)),
        },
        "feature_finite_ratio": _per_feature_finite_ratio(features, arrays["feature_names"]),
        "raw_feature_ranges": _feature_ranges(features, arrays["feature_names"]),
        "transformed_feature_ranges": _feature_ranges(
            transformed,
            arrays["transformed_feature_names"],
        ),
        "label_distribution": _code_counts(presence),
        "severity_distribution": _code_counts(severity),
        "depth_match_error": _numeric_summary(arrays["depth_match_error"]),
        "sample_weight": _numeric_summary(sample_weight),
        "transform_stats": transform_stats,
        "warnings": warnings,
        "errors": errors,
    }


def _per_feature_finite_ratio(values: np.ndarray, names: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for index, name in enumerate(names.astype(str).tolist()):
        column = values[:, index]
        result[name] = float(np.mean(np.isfinite(column))) if column.size else None
    return result


def _feature_ranges(values: np.ndarray, names: np.ndarray) -> dict[str, dict[str, float | None]]:
    return {
        name: _numeric_summary(values[:, index])
        for index, name in enumerate(names.astype(str).tolist())
    }


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


def _code_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}


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
