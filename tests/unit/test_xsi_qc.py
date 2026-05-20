from __future__ import annotations

import numpy as np

from cement_channel.qc.xsi_qc import run_xsi_waveform_qc


def test_xsi_waveform_qc_basic_statistics() -> None:
    waveform = np.ones((3, 13, 8, 32), dtype=np.float32)
    waveform[0, 0, 0, 0] = np.nan

    result = run_xsi_waveform_qc(waveform)

    assert result.shape == [3, 13, 8, 32]
    assert result.nan_ratio and result.nan_ratio > 0
    assert result.finite_ratio and result.finite_ratio < 1.0
    assert any("non-finite" in warning for warning in result.warnings)


def test_xsi_waveform_qc_shape_error() -> None:
    result = run_xsi_waveform_qc(np.ones((3, 8, 32), dtype=np.float32))

    assert result.errors
    assert "rank 4" in result.errors[0]
