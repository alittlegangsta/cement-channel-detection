from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as connected_label

from cement_channel.labels.label_quality_schema import (
    MVP4B_LABEL_QUALITY_SUBSET_VERSION,
    LabelQualityConfig,
    ReviewIntervalConfig,
    active_review_intervals,
    load_label_quality_config,
)


@dataclass(frozen=True)
class LabelQualitySubsetReport:
    subset_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    sample_count: int
    subset_counts: dict[str, dict[str, int | float | None]]
    excluded_counts_by_reason: dict[str, int]
    plus_minus_disagreement_removal_impact: dict[str, int | float | None]
    review_interval_exclusion_impact: dict[str, dict[str, int | float | str | None]]
    high_confidence_orientation_coverage: dict[str, dict[str, int | float | None]]
    connected_object_summary: dict[str, int | float | None]
    clear_negative_definition: dict[str, Any]
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


def build_label_quality_subsets_from_config(
    *,
    sample_table_npz: Path | str,
    label_quality_config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    cast_weak_label_npz: Path | str | None = None,
    overwrite: bool = False,
) -> LabelQualitySubsetReport:
    config = load_label_quality_config(label_quality_config_path)
    sample_arrays = _load_npz(sample_table_npz)
    updated, report = build_label_quality_subsets(
        sample_arrays=sample_arrays,
        config=config,
        inputs={
            "sample_table_npz": str(sample_table_npz),
            "label_quality_config_path": str(label_quality_config_path),
            "cast_weak_label_npz": str(cast_weak_label_npz or ""),
        },
        output_npz=Path(output_npz),
    )
    write_label_quality_subset_table(updated, Path(output_npz), overwrite=overwrite)
    write_label_quality_subset_report(
        report,
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return report


def build_label_quality_subsets(
    *,
    sample_arrays: dict[str, np.ndarray],
    config: LabelQualityConfig,
    inputs: dict[str, str] | None = None,
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], LabelQualitySubsetReport]:
    arrays = _required_sample_arrays(sample_arrays)
    depth = arrays["depth"]
    label = arrays["label_presence_plus"].astype(np.int8)
    severity = arrays["label_severity_plus"].astype(np.int8)
    confidence = arrays["label_confidence_plus"].astype(np.float32)
    disagreement = arrays["plus_minus_disagreement"].astype(bool)
    orientation = arrays["orientation_confidence"].astype(np.float32)
    depth_error = arrays["depth_match_error"].astype(np.float32)

    errors = _shape_errors(arrays)
    if errors:
        raise ValueError("; ".join(errors))

    review_mask = review_interval_mask(depth, active_review_intervals(config))
    depth_ok_strong = np.isfinite(depth_error) & (
        depth_error <= config.strong_positive.max_depth_match_error_ft
    )
    depth_ok_negative = np.isfinite(depth_error) & (
        depth_error <= config.clear_negative.max_depth_match_error_ft
    )
    disagreement_free = ~disagreement
    strong_positive = (
        (label == config.strong_positive.label_presence_plus)
        & (severity >= config.strong_positive.min_severity)
        & (confidence >= config.strong_positive.min_label_confidence)
        & disagreement_free
        & depth_ok_strong
    )
    clear_negative_confident = (
        (label == config.clear_negative.label_presence_plus)
        & (confidence >= config.clear_negative.min_label_confidence)
    )
    clear_negative_local_normal = (
        (label == config.clear_negative.label_presence_plus)
        & (confidence >= config.clear_negative.min_local_normal_confidence)
    )
    if config.clear_negative.local_normal_requires_severity_none:
        clear_negative_local_normal &= severity == 0
    clear_negative = (
        (clear_negative_confident | clear_negative_local_normal)
        & disagreement_free
        & depth_ok_negative
    )
    high_orientation = np.isfinite(orientation) & (
        orientation >= config.high_confidence_orientation.min_orientation_confidence
    )
    connected_mask, connected_object_id, connected_summary = connected_object_candidate_mask(
        depth=depth,
        side_index=arrays["side_index"].astype(np.int16),
        candidate_mask=label == 1,
        min_area_samples=config.connected_object_only.min_area_samples,
        min_depth_length_ft=config.connected_object_only.min_depth_length_ft,
        circular_side_connectivity=config.connected_object_only.circular_side_connectivity,
    )
    quality_strong_positive = strong_positive & high_orientation & connected_mask & ~review_mask
    quality_clear_negative = clear_negative & high_orientation & ~review_mask
    strong_clear_quality = quality_strong_positive | quality_clear_negative

    subset_masks = {
        "strong_positive": strong_positive,
        "clear_negative": clear_negative,
        "disagreement_free": disagreement_free,
        "high_confidence_orientation": high_orientation,
        "connected_object_only": connected_mask,
        "review_exclusion": review_mask,
        "quality_strong_positive": quality_strong_positive,
        "quality_clear_negative": quality_clear_negative,
        "strong_clear_quality": strong_clear_quality,
    }
    warnings: list[str] = []
    report_errors = _subset_size_errors(
        subset_masks,
        label,
        min_samples_per_class=config.quality_policy.min_subset_samples_per_class,
    )
    if np.any(review_mask):
        warnings.append("review exclusion interval around ~5700 ft removed samples.")

    updated = _subset_output_arrays(
        sample_arrays,
        subset_masks,
        connected_object_id,
        config=config,
    )
    report = LabelQualitySubsetReport(
        subset_version=MVP4B_LABEL_QUALITY_SUBSET_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_npz=str(output_npz) if output_npz else "",
        sample_count=int(label.size),
        subset_counts={
            name: subset_count_summary(mask, label)
            for name, mask in subset_masks.items()
        },
        excluded_counts_by_reason={
            "plus_minus_disagreement": int(np.count_nonzero(disagreement)),
            "large_depth_match_error": int(np.count_nonzero(~depth_ok_negative)),
            "below_strong_positive_severity": int(
                np.count_nonzero((label == 1) & (severity < config.strong_positive.min_severity))
            ),
            "review_interval": int(np.count_nonzero(review_mask)),
            "low_orientation_confidence": int(np.count_nonzero(~high_orientation)),
            "candidate_not_connected_object": int(
                np.count_nonzero((label == 1) & ~connected_mask)
            ),
        },
        plus_minus_disagreement_removal_impact=disagreement_removal_impact(label, disagreement),
        review_interval_exclusion_impact=review_exclusion_impact(
            label,
            strong_positive=strong_positive,
            clear_negative=clear_negative,
            review_mask=review_mask,
            intervals=active_review_intervals(config),
        ),
        high_confidence_orientation_coverage=orientation_coverage(
            label,
            orientation,
            thresholds=config.quality_policy.high_confidence_orientation_thresholds,
        ),
        connected_object_summary=connected_summary,
        clear_negative_definition={
            "min_label_confidence": config.clear_negative.min_label_confidence,
            "allow_local_cast_normal": config.clear_negative.allow_local_cast_normal,
            "local_normal_requires_severity_none": (
                config.clear_negative.local_normal_requires_severity_none
            ),
            "min_local_normal_confidence": config.clear_negative.min_local_normal_confidence,
        },
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=report_errors,
        not_performed=[
            "final label generation",
            "ground truth claim",
            "model training",
            "production inference",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return updated, report


def review_interval_mask(
    depth: np.ndarray,
    intervals: tuple[ReviewIntervalConfig, ...],
) -> np.ndarray:
    mask = np.zeros(depth.size, dtype=bool)
    for interval in intervals:
        mask |= (depth >= interval.depth_min_ft) & (depth <= interval.depth_max_ft)
    return mask


def connected_object_candidate_mask(
    *,
    depth: np.ndarray,
    side_index: np.ndarray,
    candidate_mask: np.ndarray,
    min_area_samples: int,
    min_depth_length_ft: float,
    circular_side_connectivity: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | None]]:
    grid, unique_depth, unique_side, row_index, col_index = _depth_side_grid(
        depth,
        side_index,
        candidate_mask,
    )
    if grid.size == 0 or not np.any(grid):
        return (
            np.zeros(candidate_mask.size, dtype=bool),
            np.zeros(candidate_mask.size, dtype=np.int32),
            {
                "component_count": 0,
                "kept_component_count": 0,
                "candidate_samples": int(np.count_nonzero(candidate_mask)),
                "kept_candidate_samples": 0,
                "min_area_samples": int(min_area_samples),
                "min_depth_length_ft": float(min_depth_length_ft),
            },
        )
    labels = _circular_connected_labels(grid) if circular_side_connectivity else _plain_labels(grid)
    roots = np.unique(labels[labels > 0])
    keep_roots: set[int] = set()
    object_id_grid = np.zeros(labels.shape, dtype=np.int32)
    depth_step = _median_step(unique_depth)
    for output_id, root in enumerate(roots, start=1):
        rows, cols = np.where(labels == root)
        if rows.size == 0:
            continue
        area = int(rows.size)
        depth_length = (float(rows.max() - rows.min()) + 1.0) * depth_step
        if area >= min_area_samples and depth_length >= min_depth_length_ft:
            keep_roots.add(int(root))
            object_id_grid[rows, cols] = output_id
    kept_grid = np.isin(labels, list(keep_roots))
    kept_samples = kept_grid[row_index, col_index] & candidate_mask
    object_ids = object_id_grid[row_index, col_index].astype(np.int32)
    object_ids = np.where(kept_samples, object_ids, 0).astype(np.int32)
    return (
        kept_samples.astype(bool),
        object_ids,
        {
            "component_count": int(roots.size),
            "kept_component_count": int(len(keep_roots)),
            "candidate_samples": int(np.count_nonzero(candidate_mask)),
            "kept_candidate_samples": int(np.count_nonzero(kept_samples)),
            "min_area_samples": int(min_area_samples),
            "min_depth_length_ft": float(min_depth_length_ft),
            "unique_depth_count": int(unique_depth.size),
            "unique_side_count": int(unique_side.size),
        },
    )


def subset_count_summary(mask: np.ndarray, label: np.ndarray) -> dict[str, int | float | None]:
    selected = np.asarray(mask, dtype=bool)
    total = int(np.count_nonzero(selected))
    candidate = int(np.count_nonzero(selected & (label == 1)))
    non_candidate = int(np.count_nonzero(selected & (label == 0)))
    return {
        "sample_count": total,
        "candidate_count": candidate,
        "non_candidate_count": non_candidate,
        "candidate_fraction": None if total == 0 else candidate / total,
    }


def disagreement_removal_impact(
    label: np.ndarray,
    disagreement: np.ndarray,
) -> dict[str, int | float | None]:
    valid = np.isin(label, [0, 1])
    kept = valid & ~disagreement
    removed = valid & disagreement
    before_candidate = int(np.count_nonzero(valid & (label == 1)))
    after_candidate = int(np.count_nonzero(kept & (label == 1)))
    before_total = int(np.count_nonzero(valid))
    after_total = int(np.count_nonzero(kept))
    return {
        "removed_total": int(np.count_nonzero(removed)),
        "removed_candidate": int(np.count_nonzero(removed & (label == 1))),
        "removed_non_candidate": int(np.count_nonzero(removed & (label == 0))),
        "candidate_fraction_before": None
        if before_total == 0
        else before_candidate / before_total,
        "candidate_fraction_after": None if after_total == 0 else after_candidate / after_total,
    }


def review_exclusion_impact(
    label: np.ndarray,
    *,
    strong_positive: np.ndarray,
    clear_negative: np.ndarray,
    review_mask: np.ndarray,
    intervals: tuple[ReviewIntervalConfig, ...],
) -> dict[str, dict[str, int | float | str | None]]:
    if not intervals:
        return {}
    impact: dict[str, dict[str, int | float | str | None]] = {}
    for interval in intervals:
        mask = review_mask
        total = int(np.count_nonzero(mask))
        impact[interval.name] = {
            "depth_min_ft": interval.depth_min_ft,
            "depth_max_ft": interval.depth_max_ft,
            "reason": interval.reason,
            "removed_total": total,
            "removed_candidate": int(np.count_nonzero(mask & (label == 1))),
            "removed_non_candidate": int(np.count_nonzero(mask & (label == 0))),
            "removed_strong_positive": int(np.count_nonzero(mask & strong_positive)),
            "removed_clear_negative": int(np.count_nonzero(mask & clear_negative)),
            "removed_fraction": None if label.size == 0 else total / label.size,
        }
    return impact


def orientation_coverage(
    label: np.ndarray,
    orientation: np.ndarray,
    *,
    thresholds: tuple[float, ...],
) -> dict[str, dict[str, int | float | None]]:
    coverage: dict[str, dict[str, int | float | None]] = {}
    finite = np.isfinite(orientation)
    for threshold in thresholds:
        mask = finite & (orientation >= threshold)
        key = f"orientation_ge_{threshold:g}"
        coverage[key] = subset_count_summary(mask, label)
    return coverage


def write_label_quality_subset_table(
    arrays: dict[str, np.ndarray],
    output_npz: Path,
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def write_label_quality_subset_report(
    report: LabelQualitySubsetReport,
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
    output_md.write_text(format_label_quality_subset_markdown(report), encoding="utf-8")


def format_label_quality_subset_markdown(report: LabelQualitySubsetReport) -> str:
    lines = [
        "# MVP-4B-R3 Label-Quality Subsets Report",
        "",
        "These subsets are diagnostic weak-label quality masks, not final labels.",
        "",
        f"- subset_version: `{report.subset_version}`",
        f"- sample_count: {report.sample_count}",
        f"- no_final_labels: `{report.no_final_labels}`",
        f"- no_stc: `{report.no_stc}`",
        f"- no_apes: `{report.no_apes}`",
        "",
        "## Subset Counts",
        "",
    ]
    for name, summary in report.subset_counts.items():
        lines.append(
            "- "
            f"{name}: samples={summary['sample_count']}, "
            f"candidate={summary['candidate_count']}, "
            f"non_candidate={summary['non_candidate_count']}, "
            f"candidate_fraction={summary['candidate_fraction']}"
        )
    lines.extend(["", "## Excluded Counts By Reason", ""])
    for key, value in report.excluded_counts_by_reason.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Review Interval Impact", ""])
    for name, value in report.review_interval_exclusion_impact.items():
        lines.append(
            "- "
            f"{name}: removed_total={value['removed_total']}, "
            f"removed_candidate={value['removed_candidate']}, "
            f"removed_clear_negative={value['removed_clear_negative']}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _subset_output_arrays(
    sample_arrays: dict[str, np.ndarray],
    subset_masks: dict[str, np.ndarray],
    connected_object_id: np.ndarray,
    *,
    config: LabelQualityConfig,
) -> dict[str, np.ndarray]:
    selected_keys = (
        "sample_id",
        "depth",
        "side_index",
        "side_azimuth_deg",
        "label_presence_plus",
        "label_severity_plus",
        "label_confidence_plus",
        "label_presence_minus_audit",
        "plus_minus_disagreement",
        "orientation_confidence",
        "depth_match_error",
    )
    output = {key: sample_arrays[key] for key in selected_keys if key in sample_arrays}
    for name, mask in subset_masks.items():
        output[f"{name}_mask"] = mask.astype(bool)
    output["connected_object_id"] = connected_object_id.astype(np.int32)
    output["subset_mask_names"] = np.asarray([f"{name}_mask" for name in subset_masks])
    output["label_quality_subset_version"] = np.asarray(MVP4B_LABEL_QUALITY_SUBSET_VERSION)
    output["label_quality_metadata_json"] = np.asarray(
        json.dumps(
            {
                "config_version": config.config_version,
                "no_final_labels": True,
                "weak_label_quality_subsets_only": True,
            },
            sort_keys=True,
        )
    )
    output["no_model_training"] = np.asarray(True)
    output["no_final_labels"] = np.asarray(True)
    output["no_stc"] = np.asarray(True)
    output["no_apes"] = np.asarray(True)
    output["no_deep_learning"] = np.asarray(True)
    output["no_mvp4c"] = np.asarray(True)
    return output


def _subset_size_errors(
    subset_masks: dict[str, np.ndarray],
    label: np.ndarray,
    *,
    min_samples_per_class: int,
) -> list[str]:
    errors: list[str] = []
    strong_count = int(np.count_nonzero(subset_masks["quality_strong_positive"]))
    clear_count = int(np.count_nonzero(subset_masks["quality_clear_negative"]))
    if strong_count < min_samples_per_class:
        errors.append(
            "quality_strong_positive subset too small: "
            f"{strong_count} < {min_samples_per_class}."
        )
    if clear_count < min_samples_per_class:
        errors.append(
            "quality_clear_negative subset too small: "
            f"{clear_count} < {min_samples_per_class}."
        )
    combined = subset_masks["strong_clear_quality"]
    if not np.any(combined & (label == 1)) or not np.any(combined & (label == 0)):
        errors.append("strong_clear_quality subset is single-class.")
    return errors


def _required_sample_arrays(sample_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = (
        "depth",
        "side_index",
        "label_presence_plus",
        "label_severity_plus",
        "label_confidence_plus",
        "plus_minus_disagreement",
        "orientation_confidence",
        "depth_match_error",
    )
    missing = [key for key in required if key not in sample_arrays]
    if missing:
        raise KeyError("sample table missing required field(s): " + ", ".join(missing))
    return {key: np.asarray(sample_arrays[key]).reshape(-1) for key in required}


def _shape_errors(arrays: dict[str, np.ndarray]) -> list[str]:
    sizes = {key: value.size for key, value in arrays.items()}
    expected = next(iter(sizes.values()))
    return [
        f"{key} length {size} does not match expected {expected}."
        for key, size in sizes.items()
        if size != expected
    ]


def _depth_side_grid(
    depth: np.ndarray,
    side_index: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_depth, row_index = np.unique(depth, return_inverse=True)
    unique_side, col_index = np.unique(side_index, return_inverse=True)
    grid = np.zeros((unique_depth.size, unique_side.size), dtype=bool)
    grid[row_index, col_index] = values.astype(bool)
    return grid, unique_depth, unique_side, row_index.astype(np.int32), col_index.astype(np.int32)


def _plain_labels(grid: np.ndarray) -> np.ndarray:
    labels, _count = connected_label(grid, structure=np.ones((3, 3), dtype=bool))
    return labels.astype(np.int32)


def _circular_connected_labels(grid: np.ndarray) -> np.ndarray:
    side_count = grid.shape[1]
    tiled = np.concatenate([grid, grid, grid], axis=1)
    labels, _count = connected_label(tiled, structure=np.ones((3, 3), dtype=bool))
    center = labels[:, side_count : 2 * side_count].astype(np.int32)
    component_ids = np.unique(center[center > 0])
    parent = {int(component_id): int(component_id) for component_id in component_ids}
    for row in range(center.shape[0]):
        for delta in (-1, 0, 1):
            other_row = row + delta
            if other_row < 0 or other_row >= center.shape[0]:
                continue
            left_id = int(center[row, 0])
            right_id = int(center[other_row, -1])
            if left_id > 0 and right_id > 0:
                _union(parent, left_id, right_id)
    output = np.zeros_like(center, dtype=np.int32)
    for component_id in component_ids:
        root = _find(parent, int(component_id))
        output[center == component_id] = root
    return output


def _find(parent: dict[int, int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: dict[int, int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _median_step(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 1.0
    diffs = np.abs(np.diff(array))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


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
