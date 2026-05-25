from __future__ import annotations

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
from cement_channel.visualization.matplotlib_utils import (
    add_uncertain_row_spans,
    finite_percentile_limits,
    image_extent,
    require_pyplot,
    save_figure,
)


def save_heatmap_png(
    values: np.ndarray,
    output_path: Path,
    *,
    uncertain_rows: np.ndarray | None = None,
    overwrite: bool = False,
    upscale: int = 10,
    depth_axis: np.ndarray | None = None,
    azimuth_axis: np.ndarray | None = None,
    title: str = "Review Heatmap",
    colorbar_label: str = "value",
    xlabel: str = "Azimuth / side",
    ylabel: str = "Depth",
    cmap: str = "coolwarm",
) -> None:
    del upscale
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True.")
    image = np.asarray(values, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("values must have shape [row, column].")
    depth = _axis_or_default(depth_axis, image.shape[0], default_scale=1.0)
    azimuth = _axis_or_default(azimuth_axis, image.shape[1], default_scale=1.0)
    _save_heatmap_figure(
        image,
        output_path,
        depth_axis=depth,
        x_axis=azimuth,
        uncertain_rows=uncertain_rows,
        title=title,
        colorbar_label=colorbar_label,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        overwrite=overwrite,
    )


def heatmap_rgb(values: np.ndarray, *, uncertain_rows: np.ndarray | None = None) -> np.ndarray:
    del uncertain_rows
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [row, column].")
    vmin, vmax = finite_percentile_limits(array, 2.0, 98.0)
    norm = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    return _blue_white_red(norm)


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
        depth = np.asarray(arrays["depth"][window_slice], dtype=np.float32)
        uncertain = arrays["orientation_confidence"][window_slice] < 0.5
        cast_raw = arrays["cast_zc"][window_slice]
        xsi_energy = arrays["xsi_side_energy"][window_slice]
        rel = arrays["relbearing_deg"][window_slice]
        cast_axis = arrays["cast_azimuth_axis_deg"]

        path = output_dir / f"cast_zc_raw_window_{window_label}.png"
        save_heatmap_png(
            cast_raw,
            path,
            uncertain_rows=uncertain,
            overwrite=overwrite,
            depth_axis=depth,
            azimuth_axis=cast_axis,
            title=f"CAST Zc Raw | {window.window_id}",
            colorbar_label="Zc (MRayl)",
            xlabel="CAST azimuth (deg)",
            cmap="viridis",
        )
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
        _save_panel_heatmaps(
            [
                ("+RelBearing", plus_cast, depth, cast_axis),
                ("-RelBearing", minus_cast, depth, cast_axis),
            ],
            path,
            title=f"CAST Zc High-Side Rotation Comparison | {window.window_id}",
            colorbar_label="Zc (MRayl)",
            uncertain_rows=uncertain,
            overwrite=overwrite,
            cmap="viridis",
        )
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
        _save_panel_heatmaps(
            [
                ("CAST normal", normal_cast, depth, cast_axis),
                ("CAST reversed", reversed_cast, depth, cast_axis),
            ],
            path,
            title=f"CAST Azimuth Matrix Direction Comparison | {window.window_id}",
            colorbar_label="Zc (MRayl)",
            uncertain_rows=uncertain,
            overwrite=overwrite,
            cmap="viridis",
        )
        figures[f"cast_zc_normal_reversed_compare_window_{window_label}"] = str(path)

        side_axis = np.arange(1, xsi_energy.shape[1] + 1, dtype=np.float32)
        path = output_dir / f"xsi_side_energy_raw_window_{window_label}.png"
        save_heatmap_png(
            xsi_energy,
            path,
            uncertain_rows=uncertain,
            overwrite=overwrite,
            depth_axis=depth,
            azimuth_axis=side_axis,
            title=f"XSI Side Energy Raw | {window.window_id}",
            colorbar_label="side energy",
            xlabel="XSI side index",
            cmap="magma",
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
        aligned_side_axis = xsi_side_azimuth_deg(
            xsi_energy.shape[1],
            side_order=side_order,  # type: ignore[arg-type]
            side_a_offset_deg=side_offset,
        )
        path = output_dir / f"xsi_side_energy_plus_minus_window_{window_label}.png"
        _save_panel_heatmaps(
            [
                ("+RelBearing", plus_xsi, depth, aligned_side_axis),
                ("-RelBearing", minus_xsi, depth, aligned_side_axis),
                ("clockwise +RelBearing", cw_xsi, depth, aligned_side_axis),
                ("counterclockwise +RelBearing", ccw_xsi, depth, aligned_side_axis),
            ],
            path,
            title=f"XSI Side Energy Rotation/Order Comparison | {window.window_id}",
            colorbar_label="side energy",
            uncertain_rows=uncertain,
            overwrite=overwrite,
            cmap="magma",
            xlabel="Aligned side azimuth (deg)",
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
    plt = require_pyplot()
    colors = [
        "tab:blue" if score.hypothesis.relbearing_sign == "plus" else "tab:orange"
        for score in scores
    ]
    labels = [
        (
            f"{score.hypothesis.relbearing_sign} | {score.hypothesis.cast_azimuth_direction} | "
            f"{score.hypothesis.xsi_side_order}"
        )
        for score in scores
    ]
    values = [score.total_score for score in scores]
    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.45 * max(len(scores), 1))),
        constrained_layout=True,
    )
    y = np.arange(len(scores))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Hypothesis total score")
    ax.set_title("RelBearing Hypothesis Score Summary")
    ax.grid(True, axis="x", alpha=0.25)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="tab:blue", label="+RelBearing"),
            Patch(facecolor="tab:orange", label="-RelBearing"),
        ],
        loc="lower right",
    )
    save_figure(fig, output_path, overwrite=overwrite)


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


def _save_heatmap_figure(
    image: np.ndarray,
    output_path: Path,
    *,
    depth_axis: np.ndarray,
    x_axis: np.ndarray,
    uncertain_rows: np.ndarray | None,
    title: str,
    colorbar_label: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    vmin, vmax = finite_percentile_limits(image, 2.0, 98.0)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        image,
        aspect="auto",
        origin="upper",
        extent=image_extent(x_axis=x_axis, y_axis=depth_axis),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    add_uncertain_row_spans(ax, y_axis=depth_axis, uncertain_rows=uncertain_rows)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    if uncertain_rows is not None and np.any(uncertain_rows):
        from matplotlib.patches import Patch

        ax.legend(
            handles=[Patch(facecolor="tab:red", alpha=0.25, label="orientation uncertain")],
            loc="upper right",
        )
    save_figure(fig, output_path, overwrite=overwrite)


def _save_panel_heatmaps(
    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    uncertain_rows: np.ndarray | None,
    overwrite: bool,
    cmap: str,
    xlabel: str = "High-side aligned azimuth (deg)",
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True.")
    plt = require_pyplot()
    values = np.concatenate(
        [np.asarray(panel[1], dtype=np.float32).reshape(-1) for panel in panels]
    )
    vmin, vmax = finite_percentile_limits(values, 2.0, 98.0)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10, max(4, 3.0 * len(panels))),
        sharex=False,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    last_image = None
    for ax, (panel_title, image, depth_axis, x_axis) in zip(axes_array, panels, strict=False):
        last_image = ax.imshow(
            image,
            aspect="auto",
            origin="upper",
            extent=image_extent(x_axis=x_axis, y_axis=depth_axis),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        add_uncertain_row_spans(ax, y_axis=depth_axis, uncertain_rows=uncertain_rows)
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth")
    fig.suptitle(title)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes_array.tolist(), label=colorbar_label)
    save_figure(fig, output_path, overwrite=overwrite)


def _axis_or_default(axis: np.ndarray | None, count: int, *, default_scale: float) -> np.ndarray:
    if axis is None:
        return np.arange(count, dtype=np.float32) * default_scale
    values = np.asarray(axis, dtype=np.float32).reshape(-1)
    if values.size != count:
        return np.arange(count, dtype=np.float32) * default_scale
    return values


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
