from __future__ import annotations

from pathlib import Path

import numpy as np

from cement_channel.visualization.mvp4x_cast_heatmap_supplement import (
    connected_mask_traceability,
    low_zc_candidate_mask,
)


def test_low_zc_candidate_mask_uses_strict_review_threshold() -> None:
    zc = np.asarray([[2.49, 2.50, 2.51]], dtype=np.float32)

    mask = low_zc_candidate_mask(zc)

    np.testing.assert_array_equal(mask, [[True, False, False]])


def test_connected_mask_traceability_reports_missing_2d_mask_without_fabrication() -> None:
    traceable, warning = connected_mask_traceability(None, expected_shape=(2, 3))

    assert traceable is False
    assert warning is not None
    assert "no saved 2D connected-component mask" in warning


def test_connected_mask_traceability_accepts_explicit_saved_2d_mask(tmp_path: Path) -> None:
    path = tmp_path / "connected_mask.npz"
    np.savez_compressed(
        path,
        connected_component_mask=np.asarray([[True, False], [False, True]], dtype=bool),
    )

    traceable, source = connected_mask_traceability(path, expected_shape=(2, 2))

    assert traceable is True
    assert source is not None
    assert "connected_component_mask" in source
