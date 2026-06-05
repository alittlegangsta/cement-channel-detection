from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_screening_scores import (
    rank_percentile,
    score_stability_std,
    select_primary_scores,
    top_k_lift,
)


def test_rank_percentile_puts_largest_score_at_one() -> None:
    scores = np.asarray([0.2, np.nan, 0.1, 0.4], dtype=np.float32)
    rank = rank_percentile(scores)

    assert np.isnan(rank[1])
    assert rank[3] == 1.0
    assert rank[2] == 0.0


def test_select_primary_scores_uses_policy_priority() -> None:
    matrix = np.full((3, 5), np.nan, dtype=np.float32)
    folds = np.full((3, 5), -1, dtype=np.int16)
    matrix[0, 0] = 0.1
    matrix[1, 0] = 0.2
    matrix[1, 1] = 0.5
    matrix[2, 2] = 0.3
    folds[matrix == matrix] = 1

    selected_policy, score, fold_id = select_primary_scores(matrix, folds)

    assert selected_policy.tolist() == ["P0", "P1", "P2"]
    assert np.allclose(score, [0.1, 0.5, 0.3])
    assert fold_id.tolist() == [1, 1, 1]


def test_top_k_lift_reports_top_fractions() -> None:
    y = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    rank = np.linspace(0.0, 1.0, 100, dtype=np.float32)

    metrics = top_k_lift(y, rank)

    assert metrics["status"] == "completed"
    assert metrics["top_k"]["top_10pct"]["lift"] is not None
    assert metrics["top_k"]["top_10pct"]["lift"] > 1.0


def test_score_stability_std_leaves_all_nan_columns_nan() -> None:
    stability = score_stability_std(
        np.asarray([1.0, np.nan], dtype=np.float32),
        np.asarray([2.0, np.nan], dtype=np.float32),
        np.asarray([3.0, np.nan], dtype=np.float32),
    )

    assert np.isfinite(stability[0])
    assert np.isnan(stability[1])
