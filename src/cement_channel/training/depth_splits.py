from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthFoldSummary:
    fold_index: int
    validation_blocks: list[int]
    train_count: int
    validation_count: int
    train_candidate_count: int
    train_non_candidate_count: int
    validation_candidate_count: int
    validation_non_candidate_count: int
    train_depth_min: float | None
    train_depth_max: float | None
    validation_depth_min: float | None
    validation_depth_max: float | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthFoldSplit:
    fold_index: int
    validation_blocks: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    summary: DepthFoldSummary


@dataclass(frozen=True)
class DepthSplitPlan:
    depth_block_ids: np.ndarray
    folds: list[DepthFoldSplit]
    block_size_ft: float
    min_gap_ft: float
    warnings: list[str]
    errors: list[str]

    def summaries(self) -> list[dict[str, Any]]:
        return [fold.summary.to_dict() for fold in self.folds]


def build_depth_block_ids(
    depth: np.ndarray,
    *,
    block_size_ft: float | None,
    n_splits: int,
) -> tuple[np.ndarray, float]:
    depth_values = _finite_depth(depth)
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1.")
    if block_size_ft is None:
        depth_range = float(np.max(depth_values) - np.min(depth_values))
        block_size = max(depth_range / max(n_splits * 4, 1), 1.0)
    else:
        block_size = float(block_size_ft)
    if block_size <= 0.0:
        raise ValueError("block_size_ft must be positive.")
    origin = float(np.min(depth_values))
    block_ids = np.floor((np.asarray(depth, dtype=np.float32) - origin) / block_size).astype(
        np.int32
    )
    return block_ids, block_size


def make_depth_block_splits(
    *,
    depth: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    min_gap_ft: float,
    block_size_ft: float | None = None,
    min_samples_per_class: int = 1,
) -> DepthSplitPlan:
    depth_values = _finite_depth(depth)
    label_values = np.asarray(labels, dtype=np.int8).reshape(-1)
    if label_values.shape[0] != depth_values.shape[0]:
        raise ValueError("depth and labels must have the same length.")
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1.")
    if min_gap_ft < 0.0:
        raise ValueError("min_gap_ft must be non-negative.")
    block_ids, resolved_block_size = build_depth_block_ids(
        depth_values,
        block_size_ft=block_size_ft,
        n_splits=n_splits,
    )
    unique_blocks = np.unique(block_ids)
    if unique_blocks.size < n_splits:
        raise ValueError(
            f"Need at least {n_splits} depth blocks, observed {unique_blocks.size}."
        )

    folds: list[DepthFoldSplit] = []
    warnings: list[str] = []
    errors: list[str] = []
    for fold_index, validation_blocks in enumerate(np.array_split(unique_blocks, n_splits)):
        validation_mask = np.isin(block_ids, validation_blocks)
        val_depth = depth_values[validation_mask]
        if val_depth.size == 0:
            raise ValueError(f"Fold {fold_index} has no validation samples.")
        train_mask = ~validation_mask
        if min_gap_ft > 0.0:
            gap_mask = (
                (depth_values >= float(np.min(val_depth)) - min_gap_ft)
                & (depth_values <= float(np.max(val_depth)) + min_gap_ft)
            )
            train_mask &= ~gap_mask
        train_mask &= np.isin(label_values, [0, 1])
        validation_mask &= np.isin(label_values, [0, 1])
        summary = _fold_summary(
            fold_index=fold_index,
            validation_blocks=validation_blocks,
            depth=depth_values,
            labels=label_values,
            train_mask=train_mask,
            validation_mask=validation_mask,
            min_samples_per_class=min_samples_per_class,
        )
        warnings.extend(
            f"fold_{fold_index}: {message}" for message in summary.warnings
        )
        folds.append(
            DepthFoldSplit(
                fold_index=fold_index,
                validation_blocks=validation_blocks.astype(np.int32),
                train_mask=train_mask,
                validation_mask=validation_mask,
                summary=summary,
            )
        )
    return DepthSplitPlan(
        depth_block_ids=block_ids,
        folds=folds,
        block_size_ft=resolved_block_size,
        min_gap_ft=float(min_gap_ft),
        warnings=warnings,
        errors=errors,
    )


def _fold_summary(
    *,
    fold_index: int,
    validation_blocks: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    min_samples_per_class: int,
) -> DepthFoldSummary:
    train_candidate = int(np.count_nonzero(train_mask & (labels == 1)))
    train_non_candidate = int(np.count_nonzero(train_mask & (labels == 0)))
    validation_candidate = int(np.count_nonzero(validation_mask & (labels == 1)))
    validation_non_candidate = int(np.count_nonzero(validation_mask & (labels == 0)))
    warnings: list[str] = []
    for split_name, candidate_count, non_candidate_count in (
        ("train", train_candidate, train_non_candidate),
        ("validation", validation_candidate, validation_non_candidate),
    ):
        if candidate_count < min_samples_per_class:
            warnings.append(
                f"{split_name} candidate count below {min_samples_per_class}: "
                f"{candidate_count}."
            )
        if non_candidate_count < min_samples_per_class:
            warnings.append(
                f"{split_name} non-candidate count below {min_samples_per_class}: "
                f"{non_candidate_count}."
            )
    return DepthFoldSummary(
        fold_index=fold_index,
        validation_blocks=[int(value) for value in validation_blocks.tolist()],
        train_count=int(np.count_nonzero(train_mask)),
        validation_count=int(np.count_nonzero(validation_mask)),
        train_candidate_count=train_candidate,
        train_non_candidate_count=train_non_candidate,
        validation_candidate_count=validation_candidate,
        validation_non_candidate_count=validation_non_candidate,
        train_depth_min=_masked_min(depth, train_mask),
        train_depth_max=_masked_max(depth, train_mask),
        validation_depth_min=_masked_min(depth, validation_mask),
        validation_depth_max=_masked_max(depth, validation_mask),
        warnings=warnings,
    )


def _finite_depth(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("depth must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("depth must be finite.")
    return values


def _masked_min(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(np.min(selected)) if selected.size else None


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(np.max(selected)) if selected.size else None
