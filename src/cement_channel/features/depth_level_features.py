from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.labels.depth_level_schema import (
    DEPTH_LEVEL_XSI_FEATURE_REPORT_VERSION,
    DEPTH_LEVEL_XSI_FEATURE_VERSION,
)


@dataclass(frozen=True)
class DepthLevelXsiFeatureReport:
    report_version: str
    feature_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    depth_count: int
    source_receiver_count: int
    source_side_count: int
    source_feature_count: int
    depth_feature_count: int
    feature_group_counts: dict[str, int]
    finite_ratio: dict[str, float]
    source_feature_names: list[str]
    depth_match: dict[str, float | int | None]
    used_label_information_for_feature_construction: bool
    high_side_sector_summaries_audit_only: bool
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


def build_depth_level_xsi_features_from_paths(
    *,
    basic_features_npz: Path | str,
    sample_table_npz: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
) -> DepthLevelXsiFeatureReport:
    basic_arrays = _load_npz(basic_features_npz)
    sample_arrays = _load_npz(sample_table_npz)
    output, report = build_depth_level_xsi_feature_table(
        basic_arrays=basic_arrays,
        sample_arrays=sample_arrays,
        inputs={
            "basic_features_npz": str(basic_features_npz),
            "sample_table_npz": str(sample_table_npz),
        },
        output_npz=Path(output_npz),
    )
    write_depth_level_xsi_feature_outputs(
        output,
        report,
        output_npz=Path(output_npz),
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return report


def build_depth_level_xsi_feature_table(
    *,
    basic_arrays: dict[str, np.ndarray],
    sample_arrays: dict[str, np.ndarray] | None = None,
    inputs: dict[str, str] | None = None,
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], DepthLevelXsiFeatureReport]:
    errors: list[str] = []
    warnings: list[str] = []
    features = np.asarray(basic_arrays["xsi_basic_features"], dtype=np.float32)
    depth = np.asarray(basic_arrays["xsi_depth"], dtype=np.float32).reshape(-1)
    feature_names = np.asarray(basic_arrays["feature_names"]).astype(str)
    _validate_basic_inputs(features, depth, feature_names, basic_arrays, errors)
    matrix, names, groups = compute_depth_level_xsi_features(features, feature_names)
    finite_before = _finite_ratio(matrix)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth_match = _depth_match_summary(depth, sample_arrays)
    output = {
        "depth": depth.astype(np.float32),
        "depth_level_xsi_features": matrix.astype(np.float32),
        "depth_level_xsi_feature_names": np.asarray(names),
        "source_feature_names": feature_names.astype(str),
        "feature_group_names": np.asarray(list(groups.keys())),
        "feature_group_counts_json": np.asarray(json.dumps(groups, sort_keys=True)),
        "depth_level_xsi_feature_version": np.asarray(DEPTH_LEVEL_XSI_FEATURE_VERSION),
        "depth_level_xsi_feature_metadata_json": np.asarray(
            json.dumps(
                {
                    "used_label_information_for_feature_construction": False,
                    "high_side_sector_summaries_audit_only": True,
                    "no_final_labels": True,
                    "no_stc": True,
                    "no_apes": True,
                },
                sort_keys=True,
            )
        ),
        "no_model_training": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "no_mvp4c": np.asarray(True),
    }
    report = DepthLevelXsiFeatureReport(
        report_version=DEPTH_LEVEL_XSI_FEATURE_REPORT_VERSION,
        feature_version=DEPTH_LEVEL_XSI_FEATURE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_npz=str(output_npz) if output_npz else "",
        depth_count=int(features.shape[0]),
        source_receiver_count=int(features.shape[1]),
        source_side_count=int(features.shape[2]),
        source_feature_count=int(features.shape[3]),
        depth_feature_count=int(matrix.shape[1]),
        feature_group_counts=groups,
        finite_ratio={
            "source_xsi_basic_features": _finite_ratio(features),
            "depth_level_xsi_features_before_fill": finite_before,
            "depth_level_xsi_features": _finite_ratio(matrix),
        },
        source_feature_names=feature_names.tolist(),
        depth_match=depth_match,
        used_label_information_for_feature_construction=False,
        high_side_sector_summaries_audit_only=True,
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "label-derived feature construction",
            "CAST feature input",
            "final label generation",
            "model training",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return output, report


def compute_depth_level_xsi_features(
    features: np.ndarray,
    feature_names: np.ndarray,
    *,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError("features must have shape [depth, receiver, side, feature].")
    names = feature_names.astype(str).tolist()
    by_side = np.nanmean(array, axis=1)
    by_receiver = np.nanmean(array, axis=2)
    side_mean = np.nanmean(by_side, axis=1)
    side_max = np.nanmax(by_side, axis=1)
    side_std = np.nanstd(by_side, axis=1)
    receiver_mean = np.nanmean(by_receiver, axis=1)
    receiver_max = np.nanmax(by_receiver, axis=1)
    receiver_std = np.nanstd(by_receiver, axis=1)
    side_center = side_mean[:, None, :]
    side_scale = np.maximum(side_std[:, None, :], epsilon)
    max_side_anomaly = np.nanmax(np.abs((by_side - side_center) / side_scale), axis=1)
    side_contrast = np.nanmax(by_side, axis=1) - np.nanmin(by_side, axis=1)
    blocks: list[tuple[str, np.ndarray, list[str]]] = [
        ("side_mean", side_mean, _names("side_mean", names)),
        ("side_max", side_max, _names("side_max", names)),
        ("side_std", side_std, _names("side_std", names)),
        ("receiver_mean", receiver_mean, _names("receiver_mean", names)),
        ("receiver_max", receiver_max, _names("receiver_max", names)),
        ("receiver_std", receiver_std, _names("receiver_std", names)),
        ("max_side_anomaly", max_side_anomaly, _names("max_side_anomaly", names)),
        ("side_contrast", side_contrast, _names("side_contrast", names)),
    ]
    late_indices = [index for index, name in enumerate(names) if "late_over_early" in name]
    if late_indices:
        late = by_side[:, :, late_indices]
        late_names = [names[index] for index in late_indices]
        blocks.extend(
            [
                (
                    "late_over_early_mean",
                    np.nanmean(late, axis=1),
                    _names("late_over_early_side_mean", late_names),
                ),
                (
                    "late_over_early_max",
                    np.nanmax(late, axis=1),
                    _names("late_over_early_side_max", late_names),
                ),
            ]
        )
    near_far, near_far_names = _near_far_ratio_features(array, names, epsilon=epsilon)
    blocks.append(("near_far_receiver_ratio", near_far, near_far_names))
    high_side, high_names = _high_side_audit_features(by_side, names)
    blocks.append(("high_side_sector_audit", high_side, high_names))
    matrix = np.column_stack([values for _, values, _ in blocks]).astype(np.float32)
    output_names = [name for _, _, block_names in blocks for name in block_names]
    groups = {group_name: int(values.shape[1]) for group_name, values, _ in blocks}
    return matrix, output_names, groups


def write_depth_level_xsi_feature_outputs(
    arrays: dict[str, np.ndarray],
    report: DepthLevelXsiFeatureReport,
    *,
    output_npz: Path,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_depth_level_xsi_feature_markdown(report), encoding="utf-8")


def format_depth_level_xsi_feature_markdown(report: DepthLevelXsiFeatureReport) -> str:
    lines = [
        "# MVP-4B-R4 Depth-Level XSI Feature Report",
        "",
        "This table aggregates existing XSI basic features by depth. It does not use "
        "label fields, CAST feature inputs, STC, APES, or deep learning.",
        "",
        f"- feature_version: `{report.feature_version}`",
        f"- depth_count: {report.depth_count}",
        f"- depth_feature_count: {report.depth_feature_count}",
        "- used_label_information_for_feature_construction: "
        f"`{report.used_label_information_for_feature_construction}`",
        "- high_side_sector_summaries_audit_only: "
        f"`{report.high_side_sector_summaries_audit_only}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        f"- no_stc: `{report.no_stc}`",
        f"- no_apes: `{report.no_apes}`",
        "",
        "## Feature Group Counts",
        "",
    ]
    lines.extend(_dict_lines(report.feature_group_counts))
    lines.extend(["", "## Finite Ratio", ""])
    lines.extend(_dict_lines(report.finite_ratio))
    lines.extend(["", "## Depth Match", ""])
    lines.extend(_dict_lines(report.depth_match))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _near_far_ratio_features(
    features: np.ndarray,
    feature_names: list[str],
    *,
    epsilon: float,
) -> tuple[np.ndarray, list[str]]:
    receiver_count = features.shape[1]
    near = np.arange(0, min(4, receiver_count))
    far_start = max(receiver_count - 4, 0)
    far = np.arange(far_start, receiver_count)
    near_side = np.nanmean(features[:, near, :, :], axis=1)
    far_side = np.nanmean(features[:, far, :, :], axis=1)
    ratio_side = far_side / np.maximum(np.abs(near_side), epsilon)
    ratio_mean = np.nanmean(ratio_side, axis=1)
    ratio_max = np.nanmax(ratio_side, axis=1)
    ratio_std = np.nanstd(ratio_side, axis=1)
    values = np.column_stack([ratio_mean, ratio_max, ratio_std]).astype(np.float32)
    names = [
        *[f"near_far_ratio_mean_{name}" for name in feature_names],
        *[f"near_far_ratio_max_{name}" for name in feature_names],
        *[f"near_far_ratio_std_{name}" for name in feature_names],
    ]
    return values, names


def _high_side_audit_features(
    by_side: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    high_side = by_side[:, 0, :]
    side_mean = np.nanmean(by_side, axis=1)
    values = np.column_stack([high_side, high_side - side_mean]).astype(np.float32)
    names = [
        *[f"high_side_audit_{name}" for name in feature_names],
        *[f"high_side_minus_side_mean_audit_{name}" for name in feature_names],
    ]
    return values, names


def _names(prefix: str, feature_names: list[str]) -> list[str]:
    return [f"{prefix}_{name}" for name in feature_names]


def _validate_basic_inputs(
    features: np.ndarray,
    depth: np.ndarray,
    feature_names: np.ndarray,
    arrays: dict[str, np.ndarray],
    errors: list[str],
) -> None:
    if features.ndim != 4:
        raise ValueError("xsi_basic_features must have shape [depth, receiver, side, feature].")
    if depth.size != features.shape[0]:
        raise ValueError("xsi_depth length must match feature depth dimension.")
    if feature_names.size != features.shape[3]:
        raise ValueError("feature_names length must match feature dimension.")
    if "no_stc" in arrays and not bool(np.asarray(arrays["no_stc"]).reshape(())):
        errors.append("xsi_basic_features input must preserve no_stc=true.")
    if "no_apes" in arrays and not bool(np.asarray(arrays["no_apes"]).reshape(())):
        errors.append("xsi_basic_features input must preserve no_apes=true.")


def _depth_match_summary(
    depth: np.ndarray,
    sample_arrays: dict[str, np.ndarray] | None,
) -> dict[str, float | int | None]:
    if not sample_arrays or "depth" not in sample_arrays:
        return {
            "sample_table_unique_depth_count": None,
            "matched_depth_count": None,
            "max_abs_depth_difference_ft": None,
            "median_abs_depth_difference_ft": None,
        }
    sample_depth = np.unique(np.asarray(sample_arrays["depth"], dtype=np.float32).reshape(-1))
    if sample_depth.size == 0 or depth.size == 0:
        return {
            "sample_table_unique_depth_count": int(sample_depth.size),
            "matched_depth_count": 0,
            "max_abs_depth_difference_ft": None,
            "median_abs_depth_difference_ft": None,
        }
    count = min(depth.size, sample_depth.size)
    diff = np.abs(depth[:count] - sample_depth[:count])
    return {
        "sample_table_unique_depth_count": int(sample_depth.size),
        "matched_depth_count": int(count),
        "max_abs_depth_difference_ft": float(np.max(diff)),
        "median_abs_depth_difference_ft": float(np.median(diff)),
    }


def _finite_ratio(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(array)) / array.size)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
