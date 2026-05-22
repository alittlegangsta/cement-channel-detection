from __future__ import annotations

import numpy as np

from cement_channel.labels.cast_baseline import build_cast_zc_baseline


def _write_input(path, zc: np.ndarray) -> None:
    np.savez_compressed(
        path,
        cast_depth=np.arange(zc.shape[0], dtype=np.float32),
        cast_azimuth_deg=np.arange(zc.shape[1], dtype=np.float32) * 2.0,
        cast_zc=zc.astype(np.float32),
    )


def test_build_cast_baseline_uses_depth_only_window(tmp_path) -> None:
    zc = np.full((9, 4), 10.0, dtype=np.float32)
    zc[:, 1] = np.array([10, 10, 10, 4, 4, 4, 10, 10, 10], dtype=np.float32)
    input_npz = tmp_path / "cast_label_input_v001.npz"
    _write_input(input_npz, zc)

    report, arrays = build_cast_zc_baseline(
        cast_label_input_npz=input_npz,
        label_config={
            "baseline": {
                "method": "rolling_quantile",
                "window_m": 5.0,
                "quantile": 0.90,
                "min_finite_fraction": 0.5,
            }
        },
    )

    assert report.errors == []
    assert arrays["zc_base"].shape == zc.shape
    assert arrays["relative_drop"][4, 1] > 0.5
    assert arrays["relative_drop"][4, 0] == 0.0
    assert report.window_samples == 5
    assert report.baseline_valid_ratio == 1.0


def test_build_cast_baseline_handles_nan_with_validity_mask(tmp_path) -> None:
    zc = np.full((7, 2), 10.0, dtype=np.float32)
    zc[2:5, 0] = np.nan
    input_npz = tmp_path / "cast_label_input_v001.npz"
    _write_input(input_npz, zc)

    report, arrays = build_cast_zc_baseline(
        cast_label_input_npz=input_npz,
        label_config={
            "baseline": {
                "method": "rolling_median",
                "window_m": 3.0,
                "quantile": 0.90,
                "min_finite_fraction": 0.5,
            }
        },
    )

    assert report.errors == []
    assert not arrays["baseline_valid"][3, 0]
    assert np.isnan(arrays["relative_drop"][3, 0])
    assert arrays["baseline_valid"][3, 1]
