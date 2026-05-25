from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ENHANCED_FEATURE_VERSION = "mvp4b_enhanced_features_v001"


@dataclass(frozen=True)
class FeatureNormalizationConfig:
    rolling_window_samples: int = 21
    epsilon: float = 1.0e-6

    def validate(self) -> None:
        if self.rolling_window_samples <= 1:
            raise ValueError("rolling_window_samples must be greater than 1.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")


@dataclass(frozen=True)
class EnhancedFeatureReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    config: dict[str, Any]
    sample_count: int
    raw_feature_count: int
    original_transformed_feature_count: int
    added_feature_count: int
    enhanced_transformed_feature_count: int
    raw_feature_finite_ratio: float
    original_transformed_feature_finite_ratio: float
    added_feature_finite_ratio: float
    enhanced_transformed_feature_finite_ratio: float
    added_feature_ranges: dict[str, dict[str, float | None]]
    skipped_features: list[str]
    used_label_information_for_features: bool
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


def add_normalized_features_from_npz(
    *,
    input_npz: Path | str,
    output_npz: Path | str,
    report_md: Path | str,
    report_json: Path | str,
    config: FeatureNormalizationConfig | None = None,
    overwrite: bool = False,
) -> EnhancedFeatureReport:
    arrays = _load_npz(input_npz)
    updated, report = enhance_sample_table_features(
        arrays,
        config=config or FeatureNormalizationConfig(),
        input_npz=Path(input_npz),
        output_npz=Path(output_npz),
    )
    write_enhanced_sample_table(updated, Path(output_npz), overwrite=overwrite)
    write_enhanced_feature_report(
        report,
        output_md=Path(report_md),
        output_json=Path(report_json),
        overwrite=overwrite,
    )
    return report


def enhance_sample_table_features(
    arrays: dict[str, np.ndarray],
    *,
    config: FeatureNormalizationConfig,
    input_npz: Path | None = None,
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], EnhancedFeatureReport]:
    config.validate()
    depth = _required_array(arrays, "depth").astype(np.float32).reshape(-1)
    side_index = _required_array(arrays, "side_index").astype(np.int16).reshape(-1)
    raw_features = _required_array(arrays, "features").astype(np.float32)
    raw_feature_names = _required_array(arrays, "feature_names").astype(str)
    transformed = _required_array(arrays, "transformed_features").astype(np.float32)
    transformed_names = _required_array(arrays, "transformed_feature_names").astype(str)
    errors = _shape_errors(
        depth=depth,
        side_index=side_index,
        raw_features=raw_features,
        transformed=transformed,
        raw_feature_names=raw_feature_names,
        transformed_names=transformed_names,
    )
    if errors:
        raise ValueError("; ".join(errors))
    warnings: list[str] = []
    skipped_features = ["receiver_aggregated_side_normalized_metrics"]
    warnings.append(
        "receiver-level features were not present in the sample table; "
        "receiver-aggregated side-normalized metrics were skipped."
    )

    depth_z = per_depth_side_zscore(
        depth,
        raw_features,
        epsilon=config.epsilon,
    )
    depth_rank = per_depth_side_rank(depth, raw_features)
    depth_residual = per_depth_median_residual(depth, raw_features)
    rolling_z = per_side_depth_rolling_zscore(
        depth,
        side_index,
        raw_features,
        window_samples=config.rolling_window_samples,
        epsilon=config.epsilon,
    )
    derived, derived_names = derived_energy_features(
        raw_features,
        raw_feature_names,
        epsilon=config.epsilon,
    )
    added_blocks = [
        depth_z,
        depth_rank,
        depth_residual,
        rolling_z,
        derived,
    ]
    added_names = [
        *[f"per_depth_side_z_{name}" for name in raw_feature_names.tolist()],
        *[f"per_depth_side_rank_{name}" for name in raw_feature_names.tolist()],
        *[f"depth_median_residual_{name}" for name in raw_feature_names.tolist()],
        *[f"per_side_rolling_z_{name}" for name in raw_feature_names.tolist()],
        *derived_names,
    ]
    added_features = np.column_stack(added_blocks).astype(np.float32)
    added_features = _finite_or_zero(added_features)
    enhanced = np.column_stack([transformed, added_features]).astype(np.float32)
    enhanced = _finite_or_zero(enhanced)
    enhanced_names = np.asarray([*transformed_names.tolist(), *added_names])

    updated = dict(arrays)
    updated["transformed_features_original"] = transformed.astype(np.float32)
    updated["transformed_feature_names_original"] = transformed_names
    updated["base_transformed_feature_count"] = np.asarray(transformed.shape[1], dtype=np.int32)
    updated["enhanced_feature_names_added"] = np.asarray(added_names)
    updated["enhanced_features_added"] = added_features.astype(np.float32)
    updated["transformed_features"] = enhanced
    updated["transformed_feature_names"] = enhanced_names
    updated["feature_normalization_version"] = np.asarray(ENHANCED_FEATURE_VERSION)
    updated["feature_normalization_metadata_json"] = np.asarray(
        json.dumps(
            {
                "rolling_window_samples": config.rolling_window_samples,
                "epsilon": config.epsilon,
                "used_label_information_for_features": False,
                "skipped_features": skipped_features,
            },
            sort_keys=True,
        )
    )
    updated["no_final_labels"] = np.asarray(True)
    updated["no_stc"] = np.asarray(True)
    updated["no_apes"] = np.asarray(True)

    report = EnhancedFeatureReport(
        report_version=ENHANCED_FEATURE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={"input_npz": str(input_npz) if input_npz else ""},
        output_npz=str(output_npz) if output_npz else "",
        config=asdict(config),
        sample_count=int(depth.size),
        raw_feature_count=int(raw_features.shape[1]),
        original_transformed_feature_count=int(transformed.shape[1]),
        added_feature_count=int(added_features.shape[1]),
        enhanced_transformed_feature_count=int(enhanced.shape[1]),
        raw_feature_finite_ratio=_finite_ratio(raw_features),
        original_transformed_feature_finite_ratio=_finite_ratio(transformed),
        added_feature_finite_ratio=_finite_ratio(added_features),
        enhanced_transformed_feature_finite_ratio=_finite_ratio(enhanced),
        added_feature_ranges=_feature_ranges(added_features, added_names),
        skipped_features=skipped_features,
        used_label_information_for_features=False,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=[],
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


def per_depth_side_zscore(
    depth: np.ndarray,
    features: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    output = np.zeros(features.shape, dtype=np.float32)
    for indices in _group_indices(depth):
        values = features[indices].astype(np.float32)
        center = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        output[indices] = np.divide(
            values - center,
            np.maximum(scale, epsilon),
            out=np.zeros_like(values, dtype=np.float32),
            where=scale[None, :] >= epsilon,
        )
    return output


def per_depth_side_rank(depth: np.ndarray, features: np.ndarray) -> np.ndarray:
    output = np.zeros(features.shape, dtype=np.float32)
    for indices in _group_indices(depth):
        values = features[indices]
        if indices.size == 1:
            output[indices] = 0.5
            continue
        for feature_index in range(features.shape[1]):
            order = np.argsort(values[:, feature_index], kind="mergesort")
            ranks = np.empty(indices.size, dtype=np.float32)
            ranks[order] = np.arange(indices.size, dtype=np.float32)
            output[indices, feature_index] = ranks / float(max(indices.size - 1, 1))
    return output


def per_depth_median_residual(depth: np.ndarray, features: np.ndarray) -> np.ndarray:
    output = np.zeros(features.shape, dtype=np.float32)
    for indices in _group_indices(depth):
        values = features[indices].astype(np.float32)
        output[indices] = values - np.median(values, axis=0)
    return output


def per_side_depth_rolling_zscore(
    depth: np.ndarray,
    side_index: np.ndarray,
    features: np.ndarray,
    *,
    window_samples: int,
    epsilon: float,
) -> np.ndarray:
    output = np.zeros(features.shape, dtype=np.float32)
    half_window = max(window_samples // 2, 1)
    for side in np.unique(side_index):
        side_rows = np.flatnonzero(side_index == side)
        order = np.argsort(depth[side_rows], kind="mergesort")
        ordered_rows = side_rows[order]
        values = features[ordered_rows].astype(np.float32)
        for row_position, row_index in enumerate(ordered_rows):
            start = max(0, row_position - half_window)
            stop = min(values.shape[0], row_position + half_window + 1)
            window = values[start:stop]
            center = np.mean(window, axis=0)
            scale = np.std(window, axis=0)
            output[row_index] = np.divide(
                values[row_position] - center,
                np.maximum(scale, epsilon),
                out=np.zeros(features.shape[1], dtype=np.float32),
                where=scale >= epsilon,
            )
    return output


def derived_energy_features(
    features: np.ndarray,
    feature_names: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, list[str]]:
    names = feature_names.astype(str).tolist()
    late_ratio = _feature_or_none(features, names, "late_over_early_ratio")
    early = _feature_or_none(features, names, "early_energy")
    late = _feature_or_none(features, names, "late_energy")
    if late_ratio is None and early is not None and late is not None:
        late_ratio = late / np.maximum(np.abs(early), epsilon)
    if late_ratio is None:
        late_ratio = np.zeros(features.shape[0], dtype=np.float32)
    log_ratio = np.log1p(np.maximum(late_ratio, 0.0))
    if early is None or late is None:
        normalized_delta = np.zeros(features.shape[0], dtype=np.float32)
    else:
        normalized_delta = (late - early) / np.maximum(np.abs(late) + np.abs(early), epsilon)
    return (
        np.column_stack([log_ratio, normalized_delta]).astype(np.float32),
        ["log_late_over_early_ratio", "normalized_late_minus_early"],
    )


def write_enhanced_sample_table(
    arrays: dict[str, np.ndarray],
    output_npz: Path,
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def write_enhanced_feature_report(
    report: EnhancedFeatureReport,
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
    output_md.write_text(format_enhanced_feature_markdown(report), encoding="utf-8")


def format_enhanced_feature_markdown(report: EnhancedFeatureReport) -> str:
    lines = [
        "# MVP-4B-R Enhanced Feature Report",
        "",
        "This report audits side/depth normalized features derived only from the "
        "existing sample table and basic XSI features. No labels were used to "
        "construct features.",
        "",
        "## Scope",
        "",
        f"- report_version: `{report.report_version}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        f"- no_stc: `{report.no_stc}`",
        f"- no_apes: `{report.no_apes}`",
        f"- no_deep_learning: `{report.no_deep_learning}`",
        f"- no_mvp4c: `{report.no_mvp4c}`",
        "",
        "## Summary",
        "",
        f"- sample_count: {report.sample_count}",
        f"- raw_feature_count: {report.raw_feature_count}",
        f"- original_transformed_feature_count: {report.original_transformed_feature_count}",
        f"- added_feature_count: {report.added_feature_count}",
        f"- enhanced_transformed_feature_count: {report.enhanced_transformed_feature_count}",
        f"- raw_feature_finite_ratio: {report.raw_feature_finite_ratio}",
        f"- original_transformed_feature_finite_ratio: "
        f"{report.original_transformed_feature_finite_ratio}",
        f"- added_feature_finite_ratio: {report.added_feature_finite_ratio}",
        f"- enhanced_transformed_feature_finite_ratio: "
        f"{report.enhanced_transformed_feature_finite_ratio}",
        f"- used_label_information_for_features: "
        f"{report.used_label_information_for_features}",
        "",
        "## Skipped Features",
        "",
    ]
    lines.extend(f"- {item}" for item in report.skipped_features)
    if not report.skipped_features:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.warnings)
    if not report.warnings:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.errors)
    if not report.errors:
        lines.append("- none")
    lines.extend(["", "## Not Performed", ""])
    lines.extend(f"- {item}" for item in report.not_performed)
    lines.append("")
    return "\n".join(lines)


def _group_indices(values: np.ndarray) -> list[np.ndarray]:
    unique_values = np.unique(values)
    return [np.flatnonzero(values == value) for value in unique_values]


def _feature_or_none(
    features: np.ndarray,
    names: list[str],
    target_name: str,
) -> np.ndarray | None:
    if target_name not in names:
        return None
    return features[:, names.index(target_name)].astype(np.float32)


def _feature_ranges(
    features: np.ndarray,
    names: list[str],
) -> dict[str, dict[str, float | None]]:
    ranges: dict[str, dict[str, float | None]] = {}
    for index, name in enumerate(names):
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


def _required_array(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise KeyError(f"Sample table is missing required field: {name}")
    return np.asarray(arrays[name])


def _shape_errors(
    *,
    depth: np.ndarray,
    side_index: np.ndarray,
    raw_features: np.ndarray,
    transformed: np.ndarray,
    raw_feature_names: np.ndarray,
    transformed_names: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    sample_count = depth.shape[0]
    if side_index.shape[0] != sample_count:
        errors.append("side_index length does not match depth length.")
    if raw_features.ndim != 2:
        errors.append("features must be a 2D array.")
    elif raw_features.shape[0] != sample_count:
        errors.append("features row count does not match depth length.")
    elif raw_feature_names.shape[0] != raw_features.shape[1]:
        errors.append("feature_names length does not match features column count.")
    if transformed.ndim != 2:
        errors.append("transformed_features must be a 2D array.")
    elif transformed.shape[0] != sample_count:
        errors.append("transformed_features row count does not match depth length.")
    elif transformed_names.shape[0] != transformed.shape[1]:
        errors.append(
            "transformed_feature_names length does not match transformed_features column count."
        )
    if not np.all(np.isfinite(depth)):
        errors.append("depth must be finite.")
    return errors


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
