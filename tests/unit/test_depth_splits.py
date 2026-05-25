from __future__ import annotations

import numpy as np
import pytest

from cement_channel.training.depth_splits import (
    build_depth_block_ids,
    make_depth_block_splits,
)


def test_build_depth_block_ids_groups_nearby_depths() -> None:
    depth = np.array([100.0, 100.5, 101.0, 125.0, 126.0], dtype=np.float32)

    block_ids, block_size = build_depth_block_ids(
        depth,
        block_size_ft=10.0,
        n_splits=3,
    )

    assert block_size == 10.0
    assert block_ids[:3].tolist() == [0, 0, 0]
    assert block_ids[-2:].tolist() == [2, 2]


def test_make_depth_block_splits_keeps_blocks_out_of_both_train_and_validation() -> None:
    depth_unique = np.arange(0.0, 120.0, 5.0, dtype=np.float32)
    depth = np.repeat(depth_unique, 2)
    labels = np.tile(np.array([0, 1], dtype=np.int8), depth_unique.size)

    plan = make_depth_block_splits(
        depth=depth,
        labels=labels,
        n_splits=3,
        min_gap_ft=2.0,
        block_size_ft=20.0,
        min_samples_per_class=2,
    )

    assert len(plan.folds) == 3
    for fold in plan.folds:
        assert not np.any(fold.train_mask & fold.validation_mask)
        train_blocks = set(plan.depth_block_ids[fold.train_mask].tolist())
        validation_blocks = set(plan.depth_block_ids[fold.validation_mask].tolist())
        assert train_blocks.isdisjoint(validation_blocks)
        assert fold.summary.validation_candidate_count >= 2
        assert fold.summary.validation_non_candidate_count >= 2


def test_make_depth_block_splits_reports_class_count_warnings() -> None:
    depth = np.arange(0.0, 60.0, 5.0, dtype=np.float32)
    labels = np.zeros(depth.shape, dtype=np.int8)
    labels[0] = 1

    plan = make_depth_block_splits(
        depth=depth,
        labels=labels,
        n_splits=3,
        min_gap_ft=0.0,
        block_size_ft=10.0,
        min_samples_per_class=2,
    )

    assert plan.warnings
    assert any("candidate count below" in warning for warning in plan.warnings)


def test_make_depth_block_splits_rejects_too_few_blocks() -> None:
    depth = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int8)

    with pytest.raises(ValueError, match="Need at least"):
        make_depth_block_splits(
            depth=depth,
            labels=labels,
            n_splits=5,
            min_gap_ft=0.0,
            block_size_ft=100.0,
        )


def test_make_depth_block_splits_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        make_depth_block_splits(
            depth=np.array([1.0, 2.0], dtype=np.float32),
            labels=np.array([0], dtype=np.int8),
            n_splits=3,
            min_gap_ft=0.0,
            block_size_ft=1.0,
        )
