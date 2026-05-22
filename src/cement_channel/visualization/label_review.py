from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.visualization.matplotlib_utils import (
    finite_percentile_limits,
    image_extent,
    require_pyplot,
    sampled_depth_axis,
    sampled_image,
    save_figure,
)

LABEL_REVIEW_VERSION = "label_review_v001"
REVIEW_FILENAMES = {
    "cast_zc_raw": "01_cast_zc_raw.png",
    "cast_zc_baseline": "02_cast_zc_baseline.png",
    "relative_drop": "03_relative_drop.png",
    "plus_overlay": "04_plus_candidate_overlay.png",
    "minus_overlay": "05_minus_ablation_overlay.png",
    "disagreement": "06_plus_minus_disagreement.png",
    "confidence": "07_confidence_map.png",
    "severity": "08_severity_map.png",
    "depth_coverage": "09_depth_coverage_summary.png",
    "zc_strength_confidence": "10_zc_strength_confidence.png",
    "baseline_confidence": "11_baseline_confidence.png",
    "orientation_confidence": "12_orientation_confidence.png",
    "relbearing_valid_confidence": "13_relbearing_valid_confidence.png",
    "bad_data_confidence": "14_bad_data_confidence.png",
    "bad_data_overlay": "15_bad_data_mask_overlay.png",
    "relative_drop_outlier_overlay": "16_relative_drop_outlier_overlay.png",
}


@dataclass(frozen=True)
class LabelReviewReport:
    label_review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figures: dict[str, str]
    review_summary_template: str
    no_final_labels: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_label_review_figures(
    *,
    cast_label_input_npz: Path | str,
    cast_baseline_npz: Path | str,
    weak_label_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_depth_pixels: int = 1200,
) -> LabelReviewReport:
    input_arrays = _load_npz(cast_label_input_npz)
    baseline_arrays = _load_npz(cast_baseline_npz)
    label_arrays = _load_npz(weak_label_npz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []

    zc = np.asarray(input_arrays["cast_zc"], dtype=np.float32)
    zc_base = np.asarray(baseline_arrays["zc_base"], dtype=np.float32)
    relative_drop = np.asarray(baseline_arrays["relative_drop"], dtype=np.float32)
    plus = np.asarray(label_arrays["presence_plus"], dtype=np.int8) == 1
    minus = np.asarray(label_arrays["presence_minus_ablation"], dtype=np.int8) == 1
    confidence = np.asarray(label_arrays["label_confidence_plus"], dtype=np.float32)
    severity = np.asarray(label_arrays["severity_plus"], dtype=np.int8)
    zc_strength_confidence = _array_or_default(
        label_arrays,
        "zc_strength_confidence_plus",
        confidence,
    )
    baseline_confidence = _array_or_default(label_arrays, "baseline_confidence_plus", confidence)
    orientation_confidence = _array_or_default(
        label_arrays,
        "orientation_confidence_on_cast_depth_plus",
        confidence,
    )
    relbearing_valid_confidence = _array_or_default(
        label_arrays,
        "relbearing_valid_confidence_plus",
        confidence,
    )
    bad_data_confidence = _array_or_default(label_arrays, "bad_data_confidence_plus", confidence)
    bad_data_mask = _array_or_default(label_arrays, "bad_data_mask_plus", np.zeros_like(plus))
    relative_drop_outlier = _array_or_default(
        label_arrays,
        "relative_drop_outlier_plus",
        np.zeros_like(plus),
    )
    depth = _first_array(label_arrays, input_arrays, key="cast_depth")
    azimuth = _first_array(label_arrays, input_arrays, key="cast_azimuth_aligned_deg")
    if azimuth is None:
        azimuth = _first_array(input_arrays, label_arrays, key="cast_azimuth_deg")
    depth_axis = sampled_depth_axis(depth, zc.shape[0], max_depth_pixels)
    azimuth_axis = _azimuth_axis(azimuth, zc.shape[1])
    no_final_labels = bool(np.asarray(label_arrays.get("no_final_labels", False)).reshape(()))
    if not no_final_labels:
        errors.append("Weak-label candidate NPZ does not set no_final_labels=true.")

    figures = {
        "cast_zc_raw": output / REVIEW_FILENAMES["cast_zc_raw"],
        "cast_zc_baseline": output / REVIEW_FILENAMES["cast_zc_baseline"],
        "relative_drop": output / REVIEW_FILENAMES["relative_drop"],
        "plus_overlay": output / REVIEW_FILENAMES["plus_overlay"],
        "minus_overlay": output / REVIEW_FILENAMES["minus_overlay"],
        "disagreement": output / REVIEW_FILENAMES["disagreement"],
        "confidence": output / REVIEW_FILENAMES["confidence"],
        "severity": output / REVIEW_FILENAMES["severity"],
        "depth_coverage": output / REVIEW_FILENAMES["depth_coverage"],
        "zc_strength_confidence": output / REVIEW_FILENAMES["zc_strength_confidence"],
        "baseline_confidence": output / REVIEW_FILENAMES["baseline_confidence"],
        "orientation_confidence": output / REVIEW_FILENAMES["orientation_confidence"],
        "relbearing_valid_confidence": output / REVIEW_FILENAMES["relbearing_valid_confidence"],
        "bad_data_confidence": output / REVIEW_FILENAMES["bad_data_confidence"],
        "bad_data_overlay": output / REVIEW_FILENAMES["bad_data_overlay"],
        "relative_drop_outlier_overlay": output / REVIEW_FILENAMES["relative_drop_outlier_overlay"],
    }

    _save_heatmap(
        sampled_image(zc, max_rows=max_depth_pixels),
        figures["cast_zc_raw"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="CAST Zc Raw",
        colorbar_label="Zc (MRayl)",
        cmap="viridis",
        overwrite=overwrite,
    )
    _save_heatmap(
        sampled_image(zc_base, max_rows=max_depth_pixels),
        figures["cast_zc_baseline"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="CAST Zc Adaptive Baseline",
        colorbar_label="Zc base (MRayl)",
        cmap="viridis",
        overwrite=overwrite,
    )
    _save_heatmap(
        sampled_image(relative_drop, max_rows=max_depth_pixels),
        figures["relative_drop"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="CAST Relative Drop",
        colorbar_label="Relative drop",
        cmap="magma",
        overwrite=overwrite,
    )
    _save_overlay(
        zc,
        plus,
        figures["plus_overlay"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="Plus Primary Candidate Overlay",
        overlay_label="plus candidate",
        overlay_color="tab:red",
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_overlay(
        zc,
        minus,
        figures["minus_overlay"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="Minus Ablation Candidate Overlay",
        overlay_label="minus ablation candidate",
        overlay_color="tab:blue",
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_mask(
        plus != minus,
        figures["disagreement"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="Plus/Minus Disagreement Map",
        label="disagreement",
        color="tab:purple",
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_heatmap(
        sampled_image(confidence, max_rows=max_depth_pixels),
        figures["confidence"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="Plus Candidate Confidence",
        colorbar_label="label confidence",
        cmap="cividis",
        overwrite=overwrite,
        vmin=0.0,
        vmax=1.0,
    )
    _save_severity(
        severity,
        plus,
        figures["severity"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_depth_coverage(
        plus,
        minus,
        figures["depth_coverage"],
        depth=depth,
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_confidence_components(
        {
            "Zc strength": zc_strength_confidence,
            "Baseline": baseline_confidence,
            "Orientation": orientation_confidence,
            "RelBearing valid": relbearing_valid_confidence,
            "Bad data": bad_data_confidence,
            "Final": confidence,
        },
        {
            "zc_strength_confidence": figures["zc_strength_confidence"],
            "baseline_confidence": figures["baseline_confidence"],
            "orientation_confidence": figures["orientation_confidence"],
            "relbearing_valid_confidence": figures["relbearing_valid_confidence"],
            "bad_data_confidence": figures["bad_data_confidence"],
        },
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_overlay(
        zc,
        bad_data_mask.astype(bool),
        figures["bad_data_overlay"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="Bad-Data Mask Overlay",
        overlay_label="bad data",
        overlay_color="tab:orange",
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_overlay(
        relative_drop,
        relative_drop_outlier.astype(bool),
        figures["relative_drop_outlier_overlay"],
        depth_axis=depth_axis,
        azimuth_axis=azimuth_axis,
        title="Relative-Drop Outlier Overlay",
        overlay_label="relative_drop > 0.95",
        overlay_color="tab:red",
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
        base_colorbar_label="relative drop",
    )

    template = output / "review_summary_template.md"
    _ensure_can_write(template, overwrite=overwrite)
    template.write_text(_review_template(), encoding="utf-8")

    report = LabelReviewReport(
        label_review_version=LABEL_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "cast_label_input_npz": str(cast_label_input_npz),
            "cast_baseline_npz": str(cast_baseline_npz),
            "weak_label_npz": str(weak_label_npz),
        },
        output_dir=str(output),
        figures={key: str(path) for key, path in figures.items()},
        review_summary_template=str(template),
        no_final_labels=no_final_labels,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "final label approval",
            "feature extraction",
            "STFT",
            "STC",
            "APES",
            "model training",
            "MVP-4 correlation validation",
        ],
    )
    (output / "label_review_summary_v001.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _save_heatmap(
    values: np.ndarray,
    output_path: Path,
    *,
    depth_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    overwrite: bool,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt = require_pyplot()
    image = np.asarray(values, dtype=np.float32)
    if vmin is None or vmax is None:
        vmin, vmax = finite_percentile_limits(image)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        image,
        aspect="auto",
        origin="upper",
        extent=image_extent(x_axis=azimuth_axis, y_axis=depth_axis),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("High-side aligned azimuth (deg)")
    ax.set_ylabel("Depth")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    save_figure(fig, output_path, overwrite=overwrite)


def _save_overlay(
    base_values: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    *,
    depth_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    title: str,
    overlay_label: str,
    overlay_color: str,
    max_depth_pixels: int,
    overwrite: bool,
    base_colorbar_label: str = "Zc (MRayl)",
) -> None:
    plt = require_pyplot()
    from matplotlib.patches import Patch

    image = sampled_image(base_values, max_rows=max_depth_pixels)
    overlay = sampled_image(mask.astype(np.float32), max_rows=max_depth_pixels)
    vmin, vmax = finite_percentile_limits(image)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        image,
        aspect="auto",
        origin="upper",
        extent=image_extent(x_axis=azimuth_axis, y_axis=depth_axis),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    masked_overlay = np.ma.masked_where(overlay <= 0.5, overlay)
    ax.imshow(
        masked_overlay,
        aspect="auto",
        origin="upper",
        extent=image_extent(x_axis=azimuth_axis, y_axis=depth_axis),
        cmap=_single_color_cmap(overlay_color),
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("High-side aligned azimuth (deg)")
    ax.set_ylabel("Depth")
    ax.legend(handles=[Patch(facecolor=overlay_color, label=overlay_label)], loc="upper right")
    fig.colorbar(im, ax=ax, label=base_colorbar_label)
    save_figure(fig, output_path, overwrite=overwrite)


def _save_mask(
    mask: np.ndarray,
    output_path: Path,
    *,
    depth_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    title: str,
    label: str,
    color: str,
    max_depth_pixels: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    from matplotlib.patches import Patch

    image = sampled_image(mask.astype(np.float32), max_rows=max_depth_pixels)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        image,
        aspect="auto",
        origin="upper",
        extent=image_extent(x_axis=azimuth_axis, y_axis=depth_axis),
        cmap=_binary_cmap(color),
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("High-side aligned azimuth (deg)")
    ax.set_ylabel("Depth")
    ax.legend(handles=[Patch(facecolor=color, label=label)], loc="upper right")
    fig.colorbar(im, ax=ax, label=label)
    save_figure(fig, output_path, overwrite=overwrite)


def _save_severity(
    severity: np.ndarray,
    candidate_mask: np.ndarray,
    output_path: Path,
    *,
    depth_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    max_depth_pixels: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    candidate = sampled_image(candidate_mask.astype(np.float32), max_rows=max_depth_pixels) > 0.5
    image = sampled_image(severity.astype(np.float32), max_rows=max_depth_pixels)
    image = np.where(candidate, image, 0.0)
    cmap = ListedColormap(["#d0d0d0", "#ffd166", "#f77f00", "#d62828"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        image,
        aspect="auto",
        origin="upper",
        extent=image_extent(x_axis=azimuth_axis, y_axis=depth_axis),
        cmap=cmap,
        norm=norm,
    )
    ax.set_title("Plus Candidate Severity")
    ax.set_xlabel("High-side aligned azimuth (deg)")
    ax.set_ylabel("Depth")
    ax.legend(
        handles=[
            Patch(facecolor="#d0d0d0", label="non-candidate"),
            Patch(facecolor="#ffd166", label="mild"),
            Patch(facecolor="#f77f00", label="moderate"),
            Patch(facecolor="#d62828", label="severe"),
        ],
        loc="upper right",
    )
    fig.colorbar(im, ax=ax, label="severity")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_confidence_components(
    components: dict[str, np.ndarray],
    figures: dict[str, Path],
    *,
    depth_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    max_depth_pixels: int,
    overwrite: bool,
) -> None:
    key_map = {
        "Zc strength": "zc_strength_confidence",
        "Baseline": "baseline_confidence",
        "Orientation": "orientation_confidence",
        "RelBearing valid": "relbearing_valid_confidence",
        "Bad data": "bad_data_confidence",
    }
    for title, key in key_map.items():
        _save_heatmap(
            sampled_image(components[title], max_rows=max_depth_pixels),
            figures[key],
            depth_axis=depth_axis,
            azimuth_axis=azimuth_axis,
            title=f"Confidence Component: {title}",
            colorbar_label="confidence component",
            cmap="cividis",
            overwrite=overwrite,
            vmin=0.0,
            vmax=1.0,
        )


def _save_depth_coverage(
    plus: np.ndarray,
    minus: np.ndarray,
    output_path: Path,
    *,
    depth: np.ndarray | None,
    max_depth_pixels: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    plus_coverage = np.mean(plus, axis=1)
    minus_coverage = np.mean(minus, axis=1)
    depth_axis = sampled_depth_axis(depth, plus.shape[0], max_depth_pixels)
    indices = np.linspace(0, plus.shape[0] - 1, num=depth_axis.size).astype(int)
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    ax.plot(plus_coverage[indices], depth_axis, color="tab:red", label="plus")
    ax.plot(minus_coverage[indices], depth_axis, color="tab:blue", label="minus ablation")
    ax.set_title("Candidate Coverage By Depth")
    ax.set_xlabel("Candidate azimuth fraction")
    ax.set_ylabel("Depth")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _single_color_cmap(color: str) -> Any:
    require_pyplot()
    from matplotlib.colors import ListedColormap, to_rgba

    rgba = to_rgba(color)
    return ListedColormap([(1.0, 1.0, 1.0, 0.0), rgba])


def _binary_cmap(color: str) -> Any:
    from matplotlib.colors import ListedColormap, to_rgba

    return ListedColormap(["#f2f2f2", to_rgba(color)])


def _review_template() -> str:
    return "\n".join(
        [
            "# CAST Weak-Label Review Summary",
            "",
            "- Reviewer:",
            "- Review date:",
            "- Depth intervals inspected:",
            "- CAST.Zc raw image acceptable: TODO",
            "- Baseline image acceptable: TODO",
            "- Plus candidate overlay acceptable: TODO",
            "- Minus ablation overlay inspected: TODO",
            "- Disagreement areas requiring follow-up: TODO",
            "- Bad-data mask explains white lines / invalid sectors: TODO",
            "- Relative-drop outlier overlay inspected: TODO",
            "- Confidence decomposition inspected for low-confidence intervals: TODO",
            "- Severity map inspected only within candidate regions: TODO",
            "- Threshold confirmation required: alpha / zc_min_limit / severity bins",
            "- zc_min_limit remains human-confirmed before any downstream use: TODO",
            "- Final label approval: not allowed in MVP-3",
            "",
        ]
    )


def _array_or_default(
    arrays: dict[str, np.ndarray],
    key: str,
    default: np.ndarray,
) -> np.ndarray:
    if key not in arrays:
        return np.asarray(default)
    return np.asarray(arrays[key])


def _first_array(*dicts: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    for values in dicts:
        if key in values:
            return np.asarray(values[key])
    return None


def _azimuth_axis(values: np.ndarray | None, width: int) -> np.ndarray:
    if values is None:
        return np.linspace(0.0, 360.0, num=width, endpoint=False, dtype=np.float32)
    axis = np.asarray(values, dtype=np.float32).reshape(-1)
    if axis.size != width:
        return np.linspace(0.0, 360.0, num=width, endpoint=False, dtype=np.float32)
    return axis


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
