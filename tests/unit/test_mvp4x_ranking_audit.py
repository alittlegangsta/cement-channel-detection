from __future__ import annotations

import numpy as np

from cement_channel.evaluation.mvp4x_ranking_audit import (
    ndcg_score,
    ordinal_audit,
    pairwise_concordance,
    top_k_metrics,
)


def test_top_k_metrics_identifies_enrichment() -> None:
    y = np.arange(20, dtype=np.float32)
    pred = y.copy()

    result = top_k_metrics(y, pred, 0.10)

    assert result["precision"] == 1.0
    assert result["lift"] > 1.0


def test_pairwise_concordance_perfect_ordering() -> None:
    y = np.arange(20, dtype=np.float32)
    pred = y.copy()

    result = pairwise_concordance(y, pred, seed=3, max_pairs=100)

    assert result["status"] == "completed"
    assert result["concordance"] == 1.0


def test_ordinal_audit_is_derived_only() -> None:
    y = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.float32)
    pred = y.copy()

    result = ordinal_audit(y_true=y, y_pred=pred, quantiles=[1 / 3, 2 / 3])

    assert result["derived_ordinal_audit_only"]
    assert result["macro_f1"] == 1.0
    assert ndcg_score(y, pred) == 1.0
