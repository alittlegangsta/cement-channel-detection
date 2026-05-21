from __future__ import annotations

import struct
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cement_channel.alignment.relbearing_calibration import (
    RelBearingCalibrationReport,
    align_with_relbearing,
    cast_azimuth_deg,
    xsi_side_azimuth_deg,
)


def save_heatmap_png(
    values: np.ndarray,
    output_path: Path,
    *,
    uncertain_rows: np.ndarray | None = None,
    overwrite: bool = False,
    upscale: int = 10,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True.")
    rgb = heatmap_rgb(values, uncertain_rows=uncertain_rows)
    scale = max(1, int(upscale))
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png_rgb(output_path, rgb)


def heatmap_rgb(values: np.ndarray, *, uncertain_rows: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [row, column].")
    if array.size == 0:
        array = np.zeros((1, 1), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size:
        vmin = float(np.nanpercentile(finite, 2.0))
        vmax = float(np.nanpercentile(finite, 98.0))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    rgb = _blue_white_red(norm)
    if uncertain_rows is not None:
        mask = np.asarray(uncertain_rows, dtype=bool).reshape(-1)
        for row_index, uncertain in enumerate(mask[: rgb.shape[0]]):
            if uncertain:
                rgb[row_index, :, :] = (0.55 * rgb[row_index, :, :] + 0.45 * 180).astype(np.uint8)
                rgb[row_index, : max(1, rgb.shape[1] // 32), :] = np.array(
                    [220, 40, 40],
                    dtype=np.uint8,
                )
    return rgb


def write_relbearing_review_figures(
    report: RelBearingCalibrationReport,
    arrays: dict[str, np.ndarray],
    *,
    output_dir: Path,
    overwrite: bool,
    max_windows: int = 3,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    windows = report.valid_windows[: max(1, int(max_windows))]
    if not windows and arrays["depth"].size:
        stop = min(3, int(arrays["depth"].size))
        windows = [SimpleNamespace(window_id="window_fallback", start_index=0, stop_index=stop)]
    best = _best_hypothesis_dict(report)
    side_order = str(best.get("xsi_side_order", "clockwise"))
    side_offset = float(best.get("side_a_offset_deg", 0.0))
    for index, window in enumerate(windows):
        window_label = f"{index:02d}_{window.window_id}"
        window_slice = slice(int(window.start_index), int(window.stop_index))
        uncertain = arrays["orientation_confidence"][window_slice] < 0.5
        cast_raw = arrays["cast_zc"][window_slice]
        xsi_energy = arrays["xsi_side_energy"][window_slice]
        rel = arrays["relbearing_deg"][window_slice]
        cast_axis = arrays["cast_azimuth_axis_deg"]

        path = output_dir / f"cast_zc_raw_window_{window_label}.png"
        save_heatmap_png(cast_raw, path, uncertain_rows=uncertain, overwrite=overwrite)
        figures[f"cast_zc_raw_window_{window_label}"] = str(path)

        plus_cast = aligned_cast_heatmap(cast_raw, cast_axis, rel, sign="plus", direction="normal")
        minus_cast = aligned_cast_heatmap(
            cast_raw,
            cast_axis,
            rel,
            sign="minus",
            direction="normal",
        )
        path = output_dir / f"cast_zc_plus_minus_window_{window_label}.png"
        save_heatmap_png(_stack_blocks([plus_cast, minus_cast]), path, overwrite=overwrite)
        figures[f"cast_zc_plus_minus_window_{window_label}"] = str(path)

        normal_cast = aligned_cast_heatmap(
            cast_raw,
            cast_axis,
            rel * 0.0,
            sign="plus",
            direction="normal",
        )
        reversed_cast = aligned_cast_heatmap(
            cast_raw,
            cast_axis,
            rel * 0.0,
            sign="plus",
            direction="reversed",
        )
        path = output_dir / f"cast_zc_normal_reversed_compare_window_{window_label}.png"
        save_heatmap_png(_stack_blocks([normal_cast, reversed_cast]), path, overwrite=overwrite)
        figures[f"cast_zc_normal_reversed_compare_window_{window_label}"] = str(path)

        path = output_dir / f"xsi_side_energy_raw_window_{window_label}.png"
        save_heatmap_png(
            xsi_energy,
            path,
            uncertain_rows=uncertain,
            overwrite=overwrite,
            upscale=20,
        )
        figures[f"xsi_side_energy_raw_window_{window_label}"] = str(path)

        plus_xsi = aligned_xsi_heatmap(
            xsi_energy,
            rel,
            sign="plus",
            side_order=side_order,
            side_a_offset_deg=side_offset,
        )
        minus_xsi = aligned_xsi_heatmap(
            xsi_energy,
            rel,
            sign="minus",
            side_order=side_order,
            side_a_offset_deg=side_offset,
        )
        cw_xsi = aligned_xsi_heatmap(
            xsi_energy,
            rel,
            sign="plus",
            side_order="clockwise",
            side_a_offset_deg=side_offset,
        )
        ccw_xsi = aligned_xsi_heatmap(
            xsi_energy,
            rel,
            sign="plus",
            side_order="counterclockwise",
            side_a_offset_deg=side_offset,
        )
        path = output_dir / f"xsi_side_energy_plus_minus_window_{window_label}.png"
        save_heatmap_png(
            _stack_blocks([plus_xsi, minus_xsi, cw_xsi, ccw_xsi]),
            path,
            overwrite=overwrite,
            upscale=20,
        )
        figures[f"xsi_side_energy_plus_minus_window_{window_label}"] = str(path)

    summary_path = output_dir / "hypothesis_score_summary.png"
    save_hypothesis_score_summary_png(report, summary_path, overwrite=overwrite)
    figures["hypothesis_score_summary"] = str(summary_path)
    template_path = output_dir / "review_summary_template.md"
    write_review_summary_template(report, template_path, overwrite=overwrite)
    figures["review_summary_template"] = str(template_path)
    return figures


def aligned_cast_heatmap(
    cast_zc: np.ndarray,
    cast_axis_deg: np.ndarray,
    relbearing_deg: np.ndarray,
    *,
    sign: str,
    direction: str,
) -> np.ndarray:
    values = np.asarray(cast_zc, dtype=np.float32)
    axis = cast_azimuth_deg(
        values.shape[1],
        cast_azimuth_direction=direction,  # type: ignore[arg-type]
        raw_axis_deg=cast_axis_deg,
    )
    aligned_axis = align_with_relbearing(
        axis.reshape(1, -1),
        np.asarray(relbearing_deg, dtype=np.float32).reshape(-1, 1),
        relbearing_sign=sign,  # type: ignore[arg-type]
    )
    return _sort_each_row(values, aligned_axis)


def aligned_xsi_heatmap(
    xsi_side_energy: np.ndarray,
    relbearing_deg: np.ndarray,
    *,
    sign: str,
    side_order: str,
    side_a_offset_deg: float,
) -> np.ndarray:
    values = np.asarray(xsi_side_energy, dtype=np.float32)
    axis = xsi_side_azimuth_deg(
        values.shape[1],
        side_order=side_order,  # type: ignore[arg-type]
        side_a_offset_deg=side_a_offset_deg,
    )
    aligned_axis = align_with_relbearing(
        axis.reshape(1, -1),
        np.asarray(relbearing_deg, dtype=np.float32).reshape(-1, 1),
        relbearing_sign=sign,  # type: ignore[arg-type]
    )
    return _sort_each_row(values, aligned_axis)


def save_hypothesis_score_summary_png(
    report: RelBearingCalibrationReport,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True.")
    scores = sorted(
        report.hypothesis_scores.values(),
        key=lambda item: item.total_score,
        reverse=True,
    )[:12]
    width = 320
    height = max(80, 24 + 14 * len(scores))
    rgb = np.full((height, width, 3), 245, dtype=np.uint8)
    max_score = max((score.total_score for score in scores), default=1.0) or 1.0
    for row, score in enumerate(scores):
        y0 = 12 + row * 14
        bar_width = int((score.total_score / max_score) * (width - 40))
        color = np.array([45, 120, 210], dtype=np.uint8)
        if score.hypothesis.relbearing_sign == "minus":
            color = np.array([220, 120, 45], dtype=np.uint8)
        rgb[y0 : y0 + 8, 20 : 20 + max(1, bar_width), :] = color
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png_rgb(output_path, rgb)


def write_review_summary_template(
    report: RelBearingCalibrationReport,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True.")
    data = report.to_dict()
    lines = [
        "# RelBearing Manual Review Template",
        "",
        "## Checklist",
        "",
        "- [ ] CAST raw window has interpretable azimuthal structure.",
        "- [ ] CAST normal/reversed direction is visually assessed.",
        "- [ ] XSI Side A-H clockwise/counterclockwise order is visually assessed.",
        "- [ ] plus and minus high-side rotations are compared.",
        "- [ ] Low orientation-confidence depths are treated as uncertain.",
        "- [ ] Recommendation is not treated as production confirmation.",
        "",
        "## Current Automated Summary",
        "",
        f"- final_recommendation: {data['final_recommendation']}",
        f"- valid_window_count: {data['valid_window_count']}",
        f"- best_vs_second_score_gap: {data['best_vs_second_score_gap']}",
        f"- single_sign_alignment_approved: {data['single_sign_alignment_approved']}",
        "",
        "## Reviewer Notes",
        "",
        "- RelBearing sign assessment:",
        "- Side A-H order assessment:",
        "- CAST azimuth matrix direction assessment:",
        "- Side A offset assessment:",
        "- Remaining uncertainty:",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape [height, width, 3].")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _blue_white_red(norm: np.ndarray) -> np.ndarray:
    values = np.asarray(norm, dtype=np.float32)
    rgb = np.zeros(values.shape + (3,), dtype=np.float32)
    lower = values <= 0.5
    upper = ~lower
    rgb[lower, 0] = values[lower] * 2.0 * 255.0
    rgb[lower, 1] = values[lower] * 2.0 * 255.0
    rgb[lower, 2] = 255.0
    rgb[upper, 0] = 255.0
    rgb[upper, 1] = (1.0 - (values[upper] - 0.5) * 2.0) * 255.0
    rgb[upper, 2] = (1.0 - (values[upper] - 0.5) * 2.0) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _sort_each_row(values: np.ndarray, azimuth_by_row: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for row in range(values.shape[0]):
        order = np.argsort(azimuth_by_row[row])
        output[row] = values[row, order]
    return output


def _stack_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.zeros((1, 1), dtype=np.float32)
    width = max(block.shape[1] for block in blocks)
    padded: list[np.ndarray] = []
    for block in blocks:
        values = np.asarray(block, dtype=np.float32)
        if values.shape[1] < width:
            pad = np.full((values.shape[0], width - values.shape[1]), np.nan, dtype=np.float32)
            values = np.concatenate([values, pad], axis=1)
        padded.append(values)
        padded.append(np.full((1, width), np.nan, dtype=np.float32))
    return np.concatenate(padded[:-1], axis=0)


def _best_hypothesis_dict(report: RelBearingCalibrationReport) -> dict[str, Any]:
    if not report.best_hypothesis:
        return {}
    hypothesis = report.best_hypothesis.get("hypothesis")
    return hypothesis if isinstance(hypothesis, dict) else {}
