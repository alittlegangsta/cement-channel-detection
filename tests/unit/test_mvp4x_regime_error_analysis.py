from __future__ import annotations

import numpy as np

from cement_channel.evaluation.mvp4x_regime_error_analysis import (
    interval_rows,
    residual_by_group,
    safe_corr,
)


def test_safe_corr_returns_positive_spearman() -> None:
    x = np.arange(10, dtype=np.float32)

    assert safe_corr(x, x, method="spearman") > 0.999


def test_residual_by_group_summarizes_mae() -> None:
    labels = np.asarray(["B", "B", "C"])
    residual = np.asarray([1.0, -1.0, 2.0], dtype=np.float32)

    result = residual_by_group(labels, residual)

    assert result["B"]["residual_mae"] == 1.0
    assert result["C"]["count"] == 1


def test_interval_rows_orders_over_prediction() -> None:
    depth = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    target = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    pred = np.asarray([1.0, 3.0, 2.0], dtype=np.float32)
    residual = pred - target

    rows = interval_rows(
        selected=np.asarray([0, 1, 2]),
        depth=depth,
        target=target,
        prediction=pred,
        residual=residual,
        order="over",
        limit=2,
    )

    assert rows[0]["sample_index"] == 1
