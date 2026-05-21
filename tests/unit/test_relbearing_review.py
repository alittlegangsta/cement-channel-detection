from __future__ import annotations

from pathlib import Path

import numpy as np

from cement_channel.alignment.relbearing_calibration import calibrate_relbearing_convention
from cement_channel.visualization.relbearing_review import (
    aligned_cast_heatmap,
    save_heatmap_png,
    write_relbearing_review_figures,
)


def _report_and_arrays() -> tuple[object, dict[str, np.ndarray]]:
    depth_count = 6
    depth = np.arange(depth_count, dtype=np.float32)
    relbearing = np.linspace(0.0, 60.0, depth_count, dtype=np.float32)
    orientation = np.ones(depth_count, dtype=np.float32)
    cast_zc = np.ones((depth_count, 16), dtype=np.float32)
    waveform = np.ones((depth_count, 2, 8, 4), dtype=np.float32)
    for index in range(depth_count):
        cast_zc[index, (4 - index) % 16] = 0.5
        waveform[index, :, (2 - index) % 8, :] = 5.0
    return calibrate_relbearing_convention(
        depth=depth,
        relbearing_deg=relbearing,
        orientation_confidence=orientation,
        cast_zc=cast_zc,
        xsi_waveform=waveform,
        window_depth_samples=2,
        min_valid_windows=2,
    )


def test_save_heatmap_png_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "heatmap.png"

    save_heatmap_png(np.arange(12, dtype=np.float32).reshape(3, 4), output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_aligned_cast_heatmap_preserves_shape() -> None:
    cast_zc = np.arange(12, dtype=np.float32).reshape(3, 4)
    relbearing = np.array([0.0, 45.0, 90.0], dtype=np.float32)

    aligned = aligned_cast_heatmap(
        cast_zc,
        np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
        relbearing,
        sign="plus",
        direction="normal",
    )

    assert aligned.shape == cast_zc.shape


def test_write_relbearing_review_figures(tmp_path: Path) -> None:
    report, arrays = _report_and_arrays()
    output_dir = tmp_path / "review"

    figures = write_relbearing_review_figures(
        report,  # type: ignore[arg-type]
        arrays,
        output_dir=output_dir,
        overwrite=False,
        max_windows=1,
    )

    assert (output_dir / "hypothesis_score_summary.png").exists()
    assert (output_dir / "review_summary_template.md").exists()
    assert any(key.startswith("cast_zc_raw_window") for key in figures)
