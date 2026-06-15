from __future__ import annotations

import numpy as np

from cement_channel.qc.cast_horizontal_stripe_qc import (
    HorizontalStripeQcConfig,
    build_horizontal_stripe_inventory,
    compute_horizontal_stripe_row_stats,
)


def test_horizontal_stripe_qc_classifies_near360_partial_and_background() -> None:
    zc = np.full((15, 6), 3.0, dtype=np.float32)
    zc[5, :] = 8.0
    zc[9, :2] = 8.0
    cfg = HorizontalStripeQcConfig(local_window_rows=5, exclude_center_rows=1)

    stats = compute_horizontal_stripe_row_stats(zc, config=cfg)
    inventory = build_horizontal_stripe_inventory(
        np.arange(15, dtype=np.float32),
        zc,
        stats,
        config=cfg,
    )

    assert stats["near360_row"][5]
    assert stats["partial_row"][9]
    assert stats["ordinary_row"][0]
    classes = {row["event_class"] for row in inventory}
    assert "one-row stripe" in classes
    assert "partial-azimuth high-Zc event" in classes


def test_horizontal_stripe_qc_groups_multi_row_band() -> None:
    zc = np.full((20, 8), 3.0, dtype=np.float32)
    zc[7:10, :] = 7.0
    cfg = HorizontalStripeQcConfig(local_window_rows=6, exclude_center_rows=1)

    stats = compute_horizontal_stripe_row_stats(zc, config=cfg)
    inventory = build_horizontal_stripe_inventory(
        np.arange(20, dtype=np.float32),
        zc,
        stats,
        config=cfg,
    )

    near = [row for row in inventory if row["event_family"] == "near_360_high_zc"]
    assert len(near) == 1
    assert near[0]["event_class"] == "multi-row band"
    assert near[0]["row_count"] == 3
