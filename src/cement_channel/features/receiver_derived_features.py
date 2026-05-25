from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.features.receiver_feature_schema import (
    MVP4B_RECEIVER_FEATURE_REPORT_VERSION,
    MVP4B_RECEIVER_FEATURE_VERSION,
    ReceiverFeatureConfig,
    load_receiver_feature_config,
    receiver_group_zero_based,
    receiver_offsets_ft,
    receiver_source_distances_ft,
)


@dataclass(frozen=True)
class ReceiverDerivedFeatureReport:
    report_version: str
    feature_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    sample_count: int
    receiver_count: int
    source_feature_count: int
    raw_receiver_feature_count: int
    transformed_receiver_feature_count: int
    output_transformed_feature_count: int
    depth_match: dict[str, float | int]
    finite_ratio: dict[str, float]
    raw_feature_ranges: dict[str, dict[str, float | None]]
    transformed_feature_ranges: dict[str, dict[str, float | None]]
    top_standardized_differences: list[dict[str, float | str | int | None]]
    used_label_information_for_feature_construction: bool
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_receiver_derived_features_from_config(
    *,
    basic_features_npz: Path | str,
    sample_table_npz: Path | str,
    receiver_config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
) -> ReceiverDerivedFeatureReport:
    config = load_receiver_feature_config(receiver_config_path)
    basic_arrays = _load_npz(basic_features_npz)
    sample_arrays = _load_npz(sample_table_npz)
    updated, report = build_receiver_derived_feature_table(
        basic_arrays=basic_arrays,
        sample_arrays=sample_arrays,
        config=config,
        inputs={
            "basic_features_npz": str(basic_features_npz),
            "sample_table_npz": str(sample_table_npz),
            "receiver_config_path": str(receiver_config_path),
        },
        output_npz=Path(output_npz),
    )
    write_receiver_derived_feature_table(updated, Path(output_npz), overwrite=overwrite)
    write_receiver_derived_feature_report(
        report,
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return report


def build_receiver_derived_feature_table(
    *,
    basic_arrays: dict[str, np.ndarray],
    sample_arrays: dict[str, np.ndarray],
    config: ReceiverFeatureConfig,
    inputs: dict[str, str] | None = None,
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], ReceiverDerivedFeatureReport]:
    basic_features = np.asarray(basic_arrays["xsi_basic_features"], dtype=np.float32)
    basic_depth = np.asarray(basic_arrays["xsi_depth"], dtype=np.float32).reshape(-1)
    basic_feature_names = np.asarray(basic_arrays["feature_names"]).astype(str)
    sample_depth = np.asarray(sample_arrays["depth"], dtype=np.float32).reshape(-1)
    side_index = np.asarray(sample_arrays["side_index"], dtype=np.int16).reshape(-1)
    errors = _validate_inputs(
        basic_features=basic_features,
        basic_depth=basic_depth,
        basic_feature_names=basic_feature_names,
        sample_depth=sample_depth,
        side_index=side_index,
        config=config,
    )
    if errors:
        raise ValueError("; ".join(errors))
    depth_indices, depth_errors = match_sample_depths_to_basic(sample_depth, basic_depth)
    receiver_profiles = basic_features[depth_indices, :, side_index, :]
    raw_receiver, raw_names = compute_receiver_derived_features(
        receiver_profiles,
        basic_feature_names,
        config=config,
    )
    transformed_receiver, transformed_names, transform_stats = transform_receiver_features(
        raw_receiver,
        raw_names,
        config=config,
    )
    existing_transformed = np.asarray(sample_arrays["transformed_features"], dtype=np.float32)
    existing_names = np.asarray(sample_arrays["transformed_feature_names"]).astype(str)
    output_transformed = np.column_stack([existing_transformed, transformed_receiver]).astype(
        np.float32
    )
    output_names = np.asarray([*existing_names.tolist(), *transformed_names])
    updated = dict(sample_arrays)
    updated["receiver_features_added"] = raw_receiver.astype(np.float32)
    updated["receiver_feature_names_added"] = np.asarray(raw_names)
    updated["receiver_transformed_features_added"] = transformed_receiver.astype(np.float32)
    updated["receiver_transformed_feature_names_added"] = np.asarray(transformed_names)
    updated["receiver_feature_version"] = np.asarray(MVP4B_RECEIVER_FEATURE_VERSION)
    updated["receiver_feature_metadata_json"] = np.asarray(
        json.dumps(
            {
                "config_version": config.config_version,
                "used_label_information_for_feature_construction": False,
                "transform_stats": transform_stats,
            },
            sort_keys=True,
        )
    )
    updated["transformed_features"] = output_transformed
    updated["transformed_feature_names"] = output_names
    updated["no_model_training"] = np.asarray(True)
    updated["no_final_labels"] = np.asarray(True)
    updated["no_stc"] = np.asarray(True)
    updated["no_apes"] = np.asarray(True)

    finite_ratio = {
        "raw_receiver_features": _finite_ratio(raw_receiver),
        "transformed_receiver_features": _finite_ratio(transformed_receiver),
        "output_transformed_features": _finite_ratio(output_transformed),
    }
    errors = []
    if finite_ratio["transformed_receiver_features"] < 1.0:
        errors.append("Receiver-derived transformed features contain non-finite values.")
    if finite_ratio["output_transformed_features"] < 1.0:
        errors.append("Output transformed_features contain non-finite values.")
    report = ReceiverDerivedFeatureReport(
        report_version=MVP4B_RECEIVER_FEATURE_REPORT_VERSION,
        feature_version=MVP4B_RECEIVER_FEATURE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_npz=str(output_npz) if output_npz else "",
        sample_count=int(sample_depth.size),
        receiver_count=int(basic_features.shape[1]),
        source_feature_count=int(basic_features.shape[3]),
        raw_receiver_feature_count=int(raw_receiver.shape[1]),
        transformed_receiver_feature_count=int(transformed_receiver.shape[1]),
        output_transformed_feature_count=int(output_transformed.shape[1]),
        depth_match={
            "max_abs_error": float(np.max(depth_errors)) if depth_errors.size else 0.0,
            "median_abs_error": float(np.median(depth_errors)) if depth_errors.size else 0.0,
            "matched_unique_depth_indices": int(np.unique(depth_indices).size),
        },
        finite_ratio=finite_ratio,
        raw_feature_ranges=_feature_ranges(raw_receiver, raw_names),
        transformed_feature_ranges=_feature_ranges(transformed_receiver, transformed_names),
        top_standardized_differences=top_standardized_differences(
            transformed_receiver,
            transformed_names,
            sample_arrays,
        ),
        used_label_information_for_feature_construction=False,
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=[],
        errors=errors,
        not_performed=[
            "raw XSI waveform reading",
            "label-derived feature construction",
            "final label generation",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
            "production model training",
        ],
    )
    return updated, report


def compute_receiver_derived_features(
    receiver_profiles: np.ndarray,
    feature_names: np.ndarray,
    *,
    config: ReceiverFeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    profiles = _finite_or_zero(np.asarray(receiver_profiles, dtype=np.float32))
    if profiles.ndim != 3:
        raise ValueError("receiver_profiles must have shape [sample, receiver, feature].")
    names = feature_names.astype(str).tolist()
    offsets = receiver_offsets_ft(config)
    distances = receiver_source_distances_ft(config)
    near = receiver_group_zero_based(config, "near")
    mid = receiver_group_zero_based(config, "mid")
    far = receiver_group_zero_based(config, "far")
    mean = np.mean(profiles, axis=1)
    std = np.std(profiles, axis=1)
    slope = _linear_slope(profiles, offsets)
    near_mean = np.mean(profiles[:, near, :], axis=1)
    mid_mean = np.mean(profiles[:, mid, :], axis=1)
    far_mean = np.mean(profiles[:, far, :], axis=1)
    far_minus_near = far_mean - near_mean
    far_over_near = far_mean / np.maximum(np.abs(near_mean), config.epsilon)
    peak_position = _receiver_peak_position(profiles)
    energy_decay_slope = _linear_slope(np.log1p(np.maximum(profiles, 0.0)), distances)
    consistency = std / np.maximum(np.abs(mean), config.epsilon)
    normalized = (profiles - mean[:, None, :]) / np.maximum(std[:, None, :], config.epsilon)
    norm_near = np.mean(normalized[:, near, :], axis=1)
    norm_mid = np.mean(normalized[:, mid, :], axis=1)
    norm_far = np.mean(normalized[:, far, :], axis=1)
    norm_far_minus_near = norm_far - norm_near
    blocks = [
        ("receiver_mean", mean),
        ("receiver_std", std),
        ("receiver_slope", slope),
        ("near_receiver_mean", near_mean),
        ("mid_receiver_mean", mid_mean),
        ("far_receiver_mean", far_mean),
        ("far_minus_near", far_minus_near),
        ("far_over_near", far_over_near),
        ("receiver_peak_position", peak_position),
        ("receiver_energy_decay_slope", energy_decay_slope),
        ("receiver_consistency_cv", consistency),
        ("receiver_norm_near_mean", norm_near),
        ("receiver_norm_mid_mean", norm_mid),
        ("receiver_norm_far_mean", norm_far),
        ("receiver_norm_far_minus_near", norm_far_minus_near),
    ]
    feature_matrix = np.column_stack([values for _, values in blocks]).astype(np.float32)
    output_names: list[str] = []
    for prefix, values in blocks:
        if values.shape[1] != len(names):
            raise ValueError(f"Receiver feature block has unexpected width: {prefix}")
        output_names.extend(f"{prefix}_{name}" for name in names)
    return _finite_or_zero(feature_matrix), output_names


def transform_receiver_features(
    raw_features: np.ndarray,
    raw_feature_names: list[str],
    *,
    config: ReceiverFeatureConfig,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    raw = _finite_or_zero(raw_features)
    clipped, clip_stats = _clip_by_quantile(raw, config.clip_quantiles)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    nonnegative = np.min(clipped, axis=0) >= 0.0
    if config.log1p_positive_features and np.any(nonnegative):
        blocks.append(np.log1p(clipped[:, nonnegative]).astype(np.float32))
        names.extend(
            f"log1p_receiver_{name}"
            for name, keep in zip(raw_feature_names, nonnegative, strict=True)
            if keep
        )
    if config.robust_scaling:
        scaled, robust_stats = _robust_scale(clipped, epsilon=config.epsilon)
        blocks.append(scaled)
        names.extend(f"robust_scaled_receiver_{name}" for name in raw_feature_names)
    else:
        robust_stats = {}
    if not blocks:
        blocks.append(clipped)
        names.extend(f"receiver_{name}" for name in raw_feature_names)
    transformed = _finite_or_zero(np.column_stack(blocks))
    return (
        transformed,
        names,
        {
            "clip_stats": clip_stats,
            "robust_stats": robust_stats,
            "log1p_positive_feature_count": int(np.count_nonzero(nonnegative)),
        },
    )


def match_sample_depths_to_basic(
    sample_depth: np.ndarray,
    basic_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(basic_depth, kind="mergesort")
    sorted_depth = basic_depth[order]
    positions = np.searchsorted(sorted_depth, sample_depth)
    positions = np.clip(positions, 0, sorted_depth.size - 1)
    previous = np.clip(positions - 1, 0, sorted_depth.size - 1)
    choose_previous = (
        np.abs(sorted_depth[previous] - sample_depth)
        < np.abs(sorted_depth[positions] - sample_depth)
    )
    sorted_indices = np.where(choose_previous, previous, positions)
    depth_indices = order[sorted_indices].astype(np.int32)
    errors = np.abs(basic_depth[depth_indices] - sample_depth).astype(np.float32)
    return depth_indices, errors


def top_standardized_differences(
    features: np.ndarray,
    feature_names: list[str],
    sample_arrays: dict[str, np.ndarray],
    *,
    top_n: int = 20,
) -> list[dict[str, float | str | int | None]]:
    labels = np.asarray(sample_arrays.get("label_presence_plus", []), dtype=np.int8).reshape(-1)
    weights = np.asarray(sample_arrays.get("sample_weight", np.ones(labels.size)), dtype=np.float32)
    if labels.size != features.shape[0] or weights.size != features.shape[0]:
        return []
    valid = np.isin(labels, [0, 1]) & np.isfinite(weights) & (weights > 0.0)
    positive = valid & (labels == 1)
    negative = valid & (labels == 0)
    if not np.any(positive) or not np.any(negative):
        return []
    rows: list[dict[str, float | str | int | None]] = []
    for index, name in enumerate(feature_names):
        pos_values = features[positive, index]
        neg_values = features[negative, index]
        pos_mean = float(np.mean(pos_values))
        neg_mean = float(np.mean(neg_values))
        pooled = float(np.sqrt(0.5 * (np.var(pos_values) + np.var(neg_values))))
        effect = None if pooled <= 0.0 else (pos_mean - neg_mean) / pooled
        rows.append(
            {
                "feature_name": name,
                "standardized_difference": effect,
                "candidate_mean": pos_mean,
                "non_candidate_mean": neg_mean,
                "candidate_count": int(np.count_nonzero(positive)),
                "non_candidate_count": int(np.count_nonzero(negative)),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(float(row["standardized_difference"] or 0.0)),
        reverse=True,
    )[:top_n]


def write_receiver_derived_feature_table(
    arrays: dict[str, np.ndarray],
    output_npz: Path,
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def write_receiver_derived_feature_report(
    report: ReceiverDerivedFeatureReport,
    *,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_receiver_derived_feature_markdown(report), encoding="utf-8")


def format_receiver_derived_feature_markdown(report: ReceiverDerivedFeatureReport) -> str:
    lines = [
        "# MVP-4B-R2 Receiver-Derived Feature Report",
        "",
        "Receiver-derived features were built from existing XSI basic features, "
        "not from raw waveform and not from weak-label fields.",
        "",
        "## Summary",
        "",
        f"- feature_version: `{report.feature_version}`",
        f"- sample_count: {report.sample_count}",
        f"- receiver_count: {report.receiver_count}",
        f"- raw_receiver_feature_count: {report.raw_receiver_feature_count}",
        f"- transformed_receiver_feature_count: {report.transformed_receiver_feature_count}",
        f"- output_transformed_feature_count: {report.output_transformed_feature_count}",
        f"- transformed_receiver_finite_ratio: "
        f"{report.finite_ratio['transformed_receiver_features']}",
        f"- no_final_labels: `{report.no_final_labels}`",
        f"- no_stc: `{report.no_stc}`",
        f"- no_apes: `{report.no_apes}`",
        "",
        "## Top Standardized Differences",
        "",
    ]
    for row in report.top_standardized_differences[:15]:
        lines.append(
            "- "
            f"{row['feature_name']}: standardized_difference="
            f"{row['standardized_difference']}"
        )
    if not report.top_standardized_differences:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _linear_slope(values: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float32).reshape(1, -1, 1)
    centered_x = x - np.mean(x)
    denom = float(np.sum(centered_x**2))
    if denom <= 0.0:
        return np.zeros((values.shape[0], values.shape[2]), dtype=np.float32)
    y_mean = np.mean(values, axis=1, keepdims=True)
    return (np.sum((values - y_mean) * centered_x, axis=1) / denom).astype(np.float32)


def _receiver_peak_position(values: np.ndarray) -> np.ndarray:
    indices = np.argmax(values, axis=1).astype(np.float32) + 1.0
    return indices.astype(np.float32)


def _clip_by_quantile(
    values: np.ndarray,
    clip_quantiles: tuple[float, float],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    low_q, high_q = clip_quantiles
    low = np.quantile(values, low_q, axis=0)
    high = np.quantile(values, high_q, axis=0)
    clipped = np.clip(values, low[None, :], high[None, :]).astype(np.float32)
    return clipped, {"low": low.astype(float).tolist(), "high": high.astype(float).tolist()}


def _robust_scale(
    values: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    median = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    iqr = q75 - q25
    std = np.std(values, axis=0)
    scale = np.where(iqr > epsilon, iqr, np.maximum(std, epsilon))
    scaled = ((values - median[None, :]) / scale[None, :]).astype(np.float32)
    return (
        _finite_or_zero(scaled),
        {
            "median": median.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
        },
    )


def _validate_inputs(
    *,
    basic_features: np.ndarray,
    basic_depth: np.ndarray,
    basic_feature_names: np.ndarray,
    sample_depth: np.ndarray,
    side_index: np.ndarray,
    config: ReceiverFeatureConfig,
) -> list[str]:
    errors: list[str] = []
    if basic_features.ndim != 4:
        errors.append("xsi_basic_features must have shape [depth, receiver, side, feature].")
    else:
        if basic_features.shape[0] != basic_depth.size:
            errors.append("xsi_depth length must match xsi_basic_features depth dimension.")
        if basic_features.shape[1] != config.geometry.receiver_count:
            errors.append("xsi_basic_features receiver dimension must be 13.")
        if basic_features.shape[3] != basic_feature_names.size:
            errors.append("feature_names length must match xsi_basic_features feature dimension.")
    if sample_depth.size != side_index.size:
        errors.append("sample depth and side_index lengths must match.")
    if np.any(side_index < 0):
        errors.append("side_index must be non-negative.")
    if basic_features.ndim == 4 and np.any(side_index >= basic_features.shape[2]):
        errors.append("side_index contains values outside the basic feature side dimension.")
    missing = [
        name for name in config.source_feature_names if name not in basic_feature_names.tolist()
    ]
    if missing:
        errors.append(
            "Basic feature NPZ missing configured source feature(s): " + ", ".join(missing)
        )
    if not np.all(np.isfinite(basic_depth)):
        errors.append("xsi_depth must be finite.")
    if not np.all(np.isfinite(sample_depth)):
        errors.append("sample depth must be finite.")
    return errors


def _feature_ranges(
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, dict[str, float | None]]:
    ranges: dict[str, dict[str, float | None]] = {}
    for index, name in enumerate(feature_names):
        values = features[:, index]
        finite = np.isfinite(values)
        if not np.any(finite):
            ranges[name] = {"min": None, "median": None, "max": None}
            continue
        selected = values[finite]
        ranges[name] = {
            "min": float(np.min(selected)),
            "median": float(np.median(selected)),
            "max": float(np.max(selected)),
        }
    return ranges


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _finite_ratio(values: np.ndarray) -> float:
    return float(np.count_nonzero(np.isfinite(values)) / values.size) if values.size else 0.0


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
