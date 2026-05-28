from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.visualization.matplotlib_utils import (
    finite_percentile_limits,
    require_pyplot,
    save_figure,
)

DEPTH_LEVEL_MANUAL_REVIEW_FIGURE_VERSION = "depth_level_manual_review_figures_v001"

DEPTH_LEVEL_MANUAL_REVIEW_FIGURE_FILENAMES = {
    "overview": "overview_depth_label_score_confidence.png",
    "selected_intervals": "selected_intervals_overview.png",
    "5700_sensitivity": "5700_band_sensitivity.png",
    "confidence_disagreement": "confidence_and_disagreement_panels.png",
}

XSI_REVIEW_FEATURES = (
    "receiver_mean_peak_abs",
    "side_mean_rms_energy",
    "side_max_rms_energy",
    "side_std_rms_energy",
    "side_std_mean_abs",
    "side_mean_early_energy",
    "side_max_early_energy",
    "side_mean_late_energy",
    "side_max_late_energy",
    "side_max_late_over_early_ratio",
    "side_std_late_over_early_ratio",
)

XSI_SUMMARY_FIELDS = [
    "interval_id",
    "interval_type",
    "start_depth",
    "end_depth",
    "feature_name",
    "interval_min",
    "interval_mean",
    "interval_max",
    "interval_std",
    "global_percentile_of_interval_mean",
    "robust_z_of_interval_mean",
    "finite_ratio",
    "nonfinite_count",
    "mean_coefficient",
    "contribution_direction",
    "review_note",
]

CAST_SUMMARY_FIELDS = [
    "interval_id",
    "interval_type",
    "start_depth",
    "end_depth",
    "presence_fraction",
    "severity_max",
    "candidate_conf",
    "relative_drop",
    "zc_min",
    "zc_p05",
    "zc_p10",
    "low_inc_fraction",
    "evidence_category",
    "evidence_note",
    "has_raw_zc",
    "has_candidate_mask",
    "has_relative_drop",
    "has_severity",
]


@dataclass(frozen=True)
class DepthLevelManualReviewFigureReport:
    review_figure_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figure_count: int
    figures: dict[str, str]
    interval_cast_panel_count: int
    interval_cast_heatmap_count: int
    interval_xsi_raw_panel_count: int
    interval_xsi_normalized_panel_count: int
    cast_summary_table_csv: str
    cast_summary_table_json: str
    xsi_summary_table_csv: str
    xsi_summary_table_json: str
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    no_production_model: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    @property
    def interval_xsi_panel_count(self) -> int:
        return self.interval_xsi_raw_panel_count + self.interval_xsi_normalized_panel_count

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["interval_xsi_panel_count"] = self.interval_xsi_panel_count
        return data


def generate_depth_level_manual_review_figures(
    *,
    review_intervals_json: Path | str,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_interval_panels: int = 50,
    max_points: int = 20000,
    cast_weak_label_candidates_npz: Path | str | None = None,
    cast_label_input_npz: Path | str | None = None,
    refinement_report_json: Path | str | None = None,
) -> DepthLevelManualReviewFigureReport:
    review_data = _read_json(Path(review_intervals_json))
    label_arrays = _load_npz(depth_level_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    report = _as_dict(review_data.get("report"))
    intervals = [row for row in _as_list(review_data.get("intervals")) if isinstance(row, dict)]
    cast_candidates = _load_optional_npz(cast_weak_label_candidates_npz, warnings)
    cast_label_input = _load_optional_npz(cast_label_input_npz, warnings)
    refinement_report = _load_optional_json(refinement_report_json, warnings)
    coefficient_map = _coefficient_map(refinement_report, report, warnings)
    _validate_guardrails(report, label_arrays, feature_arrays, errors, warnings)

    figures = {
        key: output / filename
        for key, filename in DEPTH_LEVEL_MANUAL_REVIEW_FIGURE_FILENAMES.items()
    }
    _save_overview(
        intervals,
        label_arrays,
        figures["overview"],
        overwrite=overwrite,
        max_points=max_points,
    )
    _save_selected_intervals_overview(
        intervals,
        figures["selected_intervals"],
        overwrite=overwrite,
    )

    xsi_rows = build_xsi_feature_summary_rows(
        intervals=intervals,
        feature_arrays=feature_arrays,
        coefficient_map=coefficient_map,
        warnings=warnings,
    )
    cast_rows = build_cast_evidence_summary_rows(
        intervals=intervals,
        cast_candidates=cast_candidates,
        cast_label_input=cast_label_input,
    )
    xsi_csv = output / "interval_xsi_feature_summary_table.csv"
    xsi_json = output / "interval_xsi_feature_summary_table.json"
    cast_csv = output / "interval_cast_evidence_summary_table.csv"
    cast_json = output / "interval_cast_evidence_summary_table.json"
    _write_summary_table(
        xsi_rows, XSI_SUMMARY_FIELDS, csv_path=xsi_csv, json_path=xsi_json, overwrite=overwrite
    )
    _write_summary_table(
        cast_rows, CAST_SUMMARY_FIELDS, csv_path=cast_csv, json_path=cast_json, overwrite=overwrite
    )

    cast_panel_count = _save_per_interval_cast_panels(
        intervals,
        cast_rows,
        output / "interval_cast_panels",
        overwrite=overwrite,
        max_interval_panels=max_interval_panels,
    )
    cast_heatmap_count = _save_per_interval_cast_heatmaps(
        intervals,
        cast_candidates,
        cast_label_input,
        output / "interval_cast_heatmaps",
        overwrite=overwrite,
        max_interval_panels=max_interval_panels,
        warnings=warnings,
    )
    xsi_raw_count, xsi_normalized_count = _save_per_interval_xsi_panels(
        intervals,
        xsi_rows,
        feature_arrays,
        coefficient_map,
        output / "interval_xsi_feature_panels",
        overwrite=overwrite,
        max_interval_panels=max_interval_panels,
    )
    _save_5700_sensitivity(
        intervals,
        figures["5700_sensitivity"],
        overwrite=overwrite,
    )
    _save_confidence_disagreement(
        intervals,
        figures["confidence_disagreement"],
        overwrite=overwrite,
    )
    _update_review_summary(output / "review_summary.md", overwrite=overwrite)

    summary_json = output / "depth_level_manual_review_figures_summary_v001.json"
    figure_paths = {key: str(path) for key, path in figures.items()}
    if cast_panel_count:
        figure_paths["interval_cast_panels_dir"] = str(output / "interval_cast_panels")
    if cast_heatmap_count:
        figure_paths["interval_cast_heatmaps_dir"] = str(output / "interval_cast_heatmaps")
    if xsi_raw_count or xsi_normalized_count:
        figure_paths["interval_xsi_feature_panels_dir"] = str(
            output / "interval_xsi_feature_panels"
        )
    figure_paths["cast_summary_table_csv"] = str(cast_csv)
    figure_paths["cast_summary_table_json"] = str(cast_json)
    figure_paths["xsi_summary_table_csv"] = str(xsi_csv)
    figure_paths["xsi_summary_table_json"] = str(xsi_json)
    figure_report = DepthLevelManualReviewFigureReport(
        review_figure_version=DEPTH_LEVEL_MANUAL_REVIEW_FIGURE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "review_intervals_json": str(review_intervals_json),
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "cast_weak_label_candidates_npz": (
                ""
                if cast_weak_label_candidates_npz is None
                else str(cast_weak_label_candidates_npz)
            ),
            "cast_label_input_npz": ""
            if cast_label_input_npz is None
            else str(cast_label_input_npz),
            "refinement_report_json": ""
            if refinement_report_json is None
            else str(refinement_report_json),
        },
        output_dir=str(output),
        figure_count=len(figures)
        + cast_panel_count
        + cast_heatmap_count
        + xsi_raw_count
        + xsi_normalized_count,
        figures=figure_paths,
        interval_cast_panel_count=cast_panel_count,
        interval_cast_heatmap_count=cast_heatmap_count,
        interval_xsi_raw_panel_count=xsi_raw_count,
        interval_xsi_normalized_panel_count=xsi_normalized_count,
        cast_summary_table_csv=str(cast_csv),
        cast_summary_table_json=str(cast_json),
        xsi_summary_table_csv=str(xsi_csv),
        xsi_summary_table_json=str(xsi_json),
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        no_production_model=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "new model training",
            "model refit",
            "formal performance claim",
            "production inference",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    _ensure_can_write(summary_json, overwrite=overwrite)
    summary_json.write_text(
        json.dumps(figure_report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return figure_report


def build_xsi_feature_summary_rows(
    *,
    intervals: list[dict[str, Any]],
    feature_arrays: dict[str, np.ndarray],
    coefficient_map: dict[str, float],
    warnings: list[str],
) -> list[dict[str, Any]]:
    depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    available = set(feature_names.tolist())
    missing_required = [name for name in XSI_REVIEW_FEATURES if name not in available]
    if missing_required:
        warnings.append("Missing requested XSI review feature(s): " + ", ".join(missing_required))
    rows: list[dict[str, Any]] = []
    for interval in intervals:
        interval_indices = _interval_indices(depth, interval)
        selected_features = _selected_xsi_features(feature_names, interval, coefficient_map)
        for feature_name in selected_features:
            row = _xsi_feature_summary_row(
                interval=interval,
                depth_indices=interval_indices,
                features=features,
                feature_names=feature_names,
                feature_name=feature_name,
                coefficient=coefficient_map.get(feature_name),
            )
            rows.append(row)
    return rows


def build_cast_evidence_summary_rows(
    *,
    intervals: list[dict[str, Any]],
    cast_candidates: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_candidate_mask = "presence_plus" in cast_candidates
    has_severity = "severity_plus" in cast_candidates
    has_relative_drop = "relative_drop_plus" in cast_candidates
    has_raw_zc = "cast_zc" in cast_label_input
    for interval in intervals:
        summary = _as_dict(interval.get("cast_label_summary"))
        candidate = _as_dict(summary.get("weak_label_candidate_summary"))
        zc = _as_dict(summary.get("cast_zc_summary"))
        values = {
            "presence_fraction": _as_float(candidate.get("presence_plus_fraction")),
            "severity_max": _as_float(candidate.get("severity_plus_max")),
            "candidate_conf": _as_float(candidate.get("label_confidence_plus_mean")),
            "relative_drop": _as_float(candidate.get("relative_drop_plus_max")),
            "zc_min": _as_float(zc.get("zc_min")),
            "zc_p05": _as_float(zc.get("zc_p05")),
            "zc_p10": _as_float(zc.get("zc_p10")),
            "low_inc_fraction": _as_float(zc.get("low_inc_fraction")),
        }
        category, note = _cast_evidence_category(values)
        rows.append(
            {
                "interval_id": str(interval.get("review_id", "")),
                "interval_type": str(interval.get("interval_type", "")),
                "start_depth": float(interval.get("start_depth", 0.0)),
                "end_depth": float(interval.get("end_depth", 0.0)),
                **values,
                "evidence_category": category,
                "evidence_note": note,
                "has_raw_zc": has_raw_zc,
                "has_candidate_mask": has_candidate_mask,
                "has_relative_drop": has_relative_drop,
                "has_severity": has_severity,
            }
        )
    return rows


def _save_overview(
    intervals: list[dict[str, Any]],
    label_arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    overwrite: bool,
    max_points: int,
) -> None:
    plt = require_pyplot()
    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    sample = _sample_indices(depth.size, max_points)
    label = _optional_series(label_arrays, "depth_has_channel_any", depth.size)[sample]
    confidence = _optional_series(label_arrays, "depth_label_confidence", depth.size)[sample]
    review_band = _optional_series(label_arrays, "depth_review_band_mask", depth.size)[sample]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(depth[sample], label, color="tab:red", linewidth=0.9)
    axes[0].set_ylabel("weak-label candidate")
    axes[0].set_title("Depth-level weak-label candidate, score audit, confidence - no final labels")
    mid_depth, score_mean, score_available = _interval_score_arrays(intervals)
    if mid_depth.size:
        axes[1].scatter(
            mid_depth[score_available],
            score_mean[score_available],
            s=18,
            color="tab:blue",
            label="selected interval score mean",
        )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("score")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].plot(depth[sample], confidence, color="tab:green", linewidth=0.9)
    axes[2].fill_between(
        depth[sample],
        0.0,
        review_band,
        color="tab:orange",
        alpha=0.25,
        label="5700 review-band mask",
    )
    axes[2].set_ylabel("confidence")
    axes[2].set_xlabel("Depth")
    axes[2].set_ylim(-0.02, 1.02)
    for axis in axes:
        _add_interval_spans(axis, intervals)
    save_figure(fig, output_path, overwrite=overwrite)


def _save_selected_intervals_overview(
    intervals: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    sorted_intervals = sorted(intervals, key=lambda row: float(row.get("start_depth", 0.0)))
    fig, ax = plt.subplots(
        figsize=(11, max(5, len(sorted_intervals) * 0.18)),
        constrained_layout=True,
    )
    colors = _interval_type_colors()
    for row_index, interval in enumerate(sorted_intervals):
        start = float(interval.get("start_depth", 0.0))
        end = float(interval.get("end_depth", start))
        interval_type = str(interval.get("interval_type", "unknown"))
        ax.plot(
            [start, end],
            [row_index, row_index],
            color=colors.get(interval_type, "tab:gray"),
            linewidth=4,
        )
        ax.text(
            end,
            row_index,
            f" {interval.get('review_id')} {interval_type}",
            va="center",
            fontsize=7,
        )
    ax.set_xlabel("Depth")
    ax.set_ylabel("Selected review interval")
    ax.set_title("Selected depth-level manual review intervals - weak-label candidates")
    ax.set_yticks([])
    save_figure(fig, output_path, overwrite=overwrite)


def _save_per_interval_cast_panels(
    intervals: list[dict[str, Any]],
    cast_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    overwrite: bool,
    max_interval_panels: int,
) -> int:
    rows_by_id = {str(row["interval_id"]): row for row in cast_rows}
    count = 0
    for interval in intervals[:max_interval_panels]:
        interval_id = str(interval["review_id"])
        output_path = output_dir / f"{interval_id}_cast_review_panels.png"
        _save_cast_review_panel(
            interval,
            rows_by_id.get(interval_id, {}),
            output_path,
            overwrite=overwrite,
        )
        count += 1
    return count


def _save_cast_review_panel(
    interval: dict[str, Any],
    cast_row: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes_flat = axes.reshape(-1)

    normalized = {
        "presence": _zero_one(cast_row.get("presence_fraction")),
        "candidate_conf": _zero_one(cast_row.get("candidate_conf")),
        "relative_drop": _zero_one(cast_row.get("relative_drop")),
        "low_inc": _zero_one(cast_row.get("low_inc_fraction")),
    }
    _bar_axis(
        axes_flat[0],
        normalized,
        title="0-1 CAST weak-label candidate evidence",
        ylabel="fraction / confidence",
        ylim=(0.0, 1.0),
    )

    severity = _as_float(cast_row.get("severity_max"))
    _bar_axis(
        axes_flat[1],
        {"severity_max": 0.0 if severity is None else severity},
        title="CAST severity review-only scale",
        ylabel="severity",
        ylim=(0.0, 4.0),
    )
    for level, label in [(0, "none"), (1, "mild"), (2, "moderate"), (3, "severe")]:
        axes_flat[1].axhline(level, color="tab:gray", linewidth=0.6, alpha=0.4)
        axes_flat[1].text(0.55, level + 0.03, label, fontsize=7)

    zc_values = {
        "zc_min": _as_float(cast_row.get("zc_min")),
        "zc_p05": _as_float(cast_row.get("zc_p05")),
        "zc_p10": _as_float(cast_row.get("zc_p10")),
    }
    zc_values = {key: value for key, value in zc_values.items() if value is not None}
    _bar_axis(
        axes_flat[2],
        zc_values or {"not_available": 0.0},
        title="CAST Zc review panel",
        ylabel="Zc MRayl",
        ylim=None,
    )
    axes_flat[2].axhline(2.5, color="tab:red", linestyle="--", linewidth=1.0)
    axes_flat[2].text(0.0, 2.55, "zc_min_limit = 2.5 MRayl", fontsize=7, color="tab:red")

    axes_flat[3].axis("off")
    lines = [
        f"interval: {interval.get('review_id')} {interval.get('interval_type')}",
        "scope: weak-label candidate review only",
        "not ground truth; no final labels",
        f"category: {cast_row.get('evidence_category', 'not_available')}",
        f"note: {cast_row.get('evidence_note', 'not_available')}",
        f"depth: {float(interval.get('start_depth', 0.0)):.2f}-"
        f"{float(interval.get('end_depth', 0.0)):.2f}",
    ]
    axes_flat[3].text(0.02, 0.95, "\n".join(lines), va="top", fontsize=9)
    fig.suptitle(
        f"{interval.get('review_id')} CAST weak-label candidate review - no final labels",
        fontsize=11,
    )
    save_figure(fig, output_path, overwrite=overwrite)


def _save_per_interval_cast_heatmaps(
    intervals: list[dict[str, Any]],
    cast_candidates: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
    output_dir: Path,
    *,
    overwrite: bool,
    max_interval_panels: int,
    warnings: list[str],
) -> int:
    available = (
        any(
            key in cast_candidates
            for key in ("presence_plus", "severity_plus", "relative_drop_plus")
        )
        or "cast_zc" in cast_label_input
    )
    if not available:
        warnings.append("No CAST 2D candidate/severity/relative-drop/Zc arrays for heatmaps.")
        return 0
    count = 0
    for interval in intervals[:max_interval_panels]:
        output_path = output_dir / f"{interval['review_id']}_cast_heatmaps.png"
        wrote = _save_cast_heatmap(
            interval, cast_candidates, cast_label_input, output_path, overwrite=overwrite
        )
        if wrote:
            count += 1
    return count


def _save_cast_heatmap(
    interval: dict[str, Any],
    cast_candidates: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
    output_path: Path,
    *,
    overwrite: bool,
) -> bool:
    panels: list[tuple[str, np.ndarray, str]] = []
    candidate_rows = _cast_interval_rows(interval, cast_candidates)
    input_rows = _cast_interval_rows(interval, cast_label_input)
    if candidate_rows.size and "presence_plus" in cast_candidates:
        panels.append(
            (
                "candidate mask",
                (np.asarray(cast_candidates["presence_plus"])[candidate_rows] > 0).astype(float),
                "candidate",
            )
        )
    if candidate_rows.size and "severity_plus" in cast_candidates:
        panels.append(
            (
                "severity map",
                np.asarray(cast_candidates["severity_plus"])[candidate_rows],
                "severity",
            )
        )
    if candidate_rows.size and "relative_drop_plus" in cast_candidates:
        panels.append(
            (
                "relative_drop map",
                np.asarray(cast_candidates["relative_drop_plus"], dtype=np.float32)[candidate_rows],
                "relative_drop",
            )
        )
    if input_rows.size and "cast_zc" in cast_label_input:
        panels.append(
            (
                "Zc map",
                np.asarray(cast_label_input["cast_zc"], dtype=np.float32)[input_rows],
                "Zc MRayl",
            )
        )
    if not panels:
        return False

    plt = require_pyplot()
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9, max(3, len(panels) * 2.4)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, (title, values, colorbar_label) in zip(axes_array, panels, strict=False):
        array = np.asarray(values, dtype=np.float32)
        vmin, vmax = finite_percentile_limits(array)
        if colorbar_label == "candidate":
            vmin, vmax = 0.0, 1.0
        if colorbar_label == "severity":
            vmin, vmax = 0.0, 3.0
        image = axis.imshow(array, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        axis.set_ylabel("depth row")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label=colorbar_label)
    axes_array[-1].set_xlabel("CAST azimuth bin")
    fig.suptitle(
        f"{interval.get('review_id')} CAST 2D weak-label candidate heatmaps - "
        "review only, not ground truth",
        fontsize=10,
    )
    save_figure(fig, output_path, overwrite=overwrite)
    return True


def _save_per_interval_xsi_panels(
    intervals: list[dict[str, Any]],
    xsi_rows: list[dict[str, Any]],
    feature_arrays: dict[str, np.ndarray],
    coefficient_map: dict[str, float],
    output_dir: Path,
    *,
    overwrite: bool,
    max_interval_panels: int,
) -> tuple[int, int]:
    depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    rows_by_interval: dict[str, list[dict[str, Any]]] = {}
    for row in xsi_rows:
        rows_by_interval.setdefault(str(row["interval_id"]), []).append(row)

    raw_count = 0
    normalized_count = 0
    for interval in intervals[:max_interval_panels]:
        interval_id = str(interval["review_id"])
        interval_rows = rows_by_interval.get(interval_id, [])
        selected = [
            str(row["feature_name"])
            for row in interval_rows
            if str(row.get("review_note")) != "missing_feature"
        ]
        selected = _dedupe_preserve_order(selected)[:12]
        raw_path = output_dir / f"{interval_id}_xsi_raw_feature_multiples.png"
        normalized_path = output_dir / f"{interval_id}_xsi_normalized_feature_panel.png"
        _save_xsi_raw_small_multiples(
            interval,
            selected,
            depth,
            features,
            feature_names,
            raw_path,
            overwrite=overwrite,
        )
        raw_count += 1
        _save_xsi_normalized_panel(
            interval,
            selected,
            interval_rows,
            depth,
            features,
            feature_names,
            coefficient_map,
            normalized_path,
            overwrite=overwrite,
        )
        normalized_count += 1
    return raw_count, normalized_count


def _save_xsi_raw_small_multiples(
    interval: dict[str, Any],
    selected_features: list[str],
    depth: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    if not selected_features:
        selected_features = ["not_available"]
    interval_indices = _interval_indices(depth, interval)
    nrows = len(selected_features)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10, max(3, nrows * 1.55)),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, feature_name in zip(axes_array, selected_features, strict=False):
        column = _feature_column(feature_names, feature_name)
        if column is None:
            axis.text(0.02, 0.5, f"missing feature: {feature_name}", transform=axis.transAxes)
            axis.set_ylabel("missing")
            continue
        values = features[interval_indices, column]
        x_values = depth[interval_indices]
        axis.plot(x_values, values, color="tab:blue", linewidth=1.1)
        finite = values[np.isfinite(values)]
        if finite.size:
            stats = (
                f"mean={float(np.mean(finite)):.3g} "
                f"min={float(np.min(finite)):.3g} max={float(np.max(finite)):.3g}"
            )
        else:
            stats = "nonfinite"
        axis.set_ylabel(_short_feature_label(feature_name))
        axis.set_title(
            f"{feature_name} raw trace ({_feature_scale_note(feature_name)}); {stats}", fontsize=8
        )
    axes_array[-1].set_xlabel("Depth")
    fig.suptitle(
        f"{interval.get('review_id')} XSI raw feature small multiples - "
        "separate y-axis per feature",
        fontsize=10,
    )
    save_figure(fig, output_path, overwrite=overwrite)


def _save_xsi_normalized_panel(
    interval: dict[str, Any],
    selected_features: list[str],
    interval_rows: list[dict[str, Any]],
    depth: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    coefficient_map: dict[str, float],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    interval_indices = _interval_indices(depth, interval)
    fig, (axis, text_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        height_ratios=[3.0, 1.35],
        constrained_layout=True,
    )
    for feature_name in selected_features[:12]:
        column = _feature_column(feature_names, feature_name)
        if column is None:
            continue
        values = features[:, column]
        normalized = _robust_z(values)
        axis.plot(
            depth[interval_indices],
            np.clip(normalized[interval_indices], -6.0, 6.0),
            linewidth=1.0,
            label=_short_feature_label(feature_name),
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(2.0, color="tab:red", linewidth=0.7, linestyle="--")
    axis.axhline(-2.0, color="tab:red", linewidth=0.7, linestyle="--")
    axis.set_ylabel("robust z-score")
    axis.set_xlabel("Depth")
    axis.set_title(
        f"{interval.get('review_id')} XSI normalized feature panel - cross-feature comparison"
    )
    axis.legend(loc="best", fontsize=6, ncol=2)

    text_axis.axis("off")
    text_axis.text(
        0.01,
        0.98,
        _xsi_interval_text_summary(interval_rows, coefficient_map),
        va="top",
        fontsize=8,
        family="monospace",
    )
    save_figure(fig, output_path, overwrite=overwrite)


def _save_5700_sensitivity(
    intervals: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    groups = {
        "non_5700": [row for row in intervals if not bool(row.get("5700_band_flag"))],
        "5700_band": [row for row in intervals if bool(row.get("5700_band_flag"))],
    }
    labels = list(groups)
    score_values = [
        _mean_interval_summary_value(rows, "prediction_score_summary", "score_mean")
        for rows in groups.values()
    ]
    confidence_values = [
        _mean_interval_summary_value(rows, "confidence_summary", "depth_label_confidence_mean")
        for rows in groups.values()
    ]
    x_values = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(x_values - 0.18, score_values, width=0.36, label="score mean")
    ax.bar(x_values + 0.18, confidence_values, width=0.36, label="confidence mean")
    ax.set_xticks(x_values, labels=labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("5700 ft review-band sensitivity - weak-label candidate review")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_confidence_disagreement(
    intervals: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    colors = _interval_type_colors()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for interval in intervals:
        interval_type = str(interval.get("interval_type"))
        confidence = _summary_float(
            interval,
            "confidence_summary",
            "depth_label_confidence_mean",
        )
        disagreement = _summary_float(
            interval,
            "plus_minus_disagreement_summary",
            "plus_minus_disagreement_max",
        )
        score = _summary_float(interval, "prediction_score_summary", "score_mean")
        size = 30.0 + 60.0 * (0.0 if score is None else score)
        ax.scatter(
            confidence or 0.0,
            disagreement or 0.0,
            s=size,
            color=colors.get(interval_type, "tab:gray"),
            alpha=0.75,
            label=interval_type,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=7)
    ax.set_xlabel("Label confidence mean")
    ax.set_ylabel("Plus/minus disagreement max")
    ax.set_title("Confidence and disagreement panels - no final labels")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    save_figure(fig, output_path, overwrite=overwrite)


def _xsi_feature_summary_row(
    *,
    interval: dict[str, Any],
    depth_indices: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    feature_name: str,
    coefficient: float | None,
) -> dict[str, Any]:
    base = {
        "interval_id": str(interval.get("review_id", "")),
        "interval_type": str(interval.get("interval_type", "")),
        "start_depth": float(interval.get("start_depth", 0.0)),
        "end_depth": float(interval.get("end_depth", 0.0)),
        "feature_name": feature_name,
        "interval_min": None,
        "interval_mean": None,
        "interval_max": None,
        "interval_std": None,
        "global_percentile_of_interval_mean": None,
        "robust_z_of_interval_mean": None,
        "finite_ratio": 0.0,
        "nonfinite_count": None,
        "mean_coefficient": coefficient,
        "contribution_direction": _contribution_direction(coefficient),
        "review_note": "missing_feature",
    }
    column = _feature_column(feature_names, feature_name)
    if column is None:
        return base
    global_values = np.asarray(features[:, column], dtype=np.float32)
    interval_values = np.asarray(global_values[depth_indices], dtype=np.float32)
    finite = interval_values[np.isfinite(interval_values)]
    nonfinite_count = int(interval_values.size - finite.size)
    base["nonfinite_count"] = nonfinite_count
    base["finite_ratio"] = (
        0.0 if interval_values.size == 0 else float(finite.size / interval_values.size)
    )
    if finite.size == 0:
        base["review_note"] = "nonfinite_interval"
        return base
    interval_mean = float(np.mean(finite))
    base.update(
        {
            "interval_min": float(np.min(finite)),
            "interval_mean": interval_mean,
            "interval_max": float(np.max(finite)),
            "interval_std": float(np.std(finite)),
            "global_percentile_of_interval_mean": _percentile_rank(global_values, interval_mean),
            "robust_z_of_interval_mean": _robust_z_scalar(global_values, interval_mean),
            "review_note": "finite_ratio=1.0" if nonfinite_count == 0 else "contains_nonfinite",
        }
    )
    return base


def _selected_xsi_features(
    feature_names: np.ndarray,
    interval: dict[str, Any],
    coefficient_map: dict[str, float],
) -> list[str]:
    selected = list(XSI_REVIEW_FEATURES)
    selected.extend(_interval_top_feature_names(interval))
    selected.extend(
        name
        for name, _ in sorted(
            coefficient_map.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:8]
    )
    available = set(feature_names.tolist())
    missing_required = [name for name in XSI_REVIEW_FEATURES if name not in available]
    present = [name for name in selected if name in available]
    return _dedupe_preserve_order(present + missing_required)


def _interval_top_feature_names(interval: dict[str, Any]) -> list[str]:
    rows = _as_list(_as_dict(interval.get("xsi_feature_summary")).get("top_feature_values"))
    result = []
    for row in rows:
        if isinstance(row, dict) and row.get("feature_name"):
            result.append(str(row["feature_name"]))
    return result


def _cast_evidence_category(values: dict[str, float | None]) -> tuple[str, str]:
    presence = _zero_one(values.get("presence_fraction"))
    severity = _as_float(values.get("severity_max"))
    confidence = _zero_one(values.get("candidate_conf"))
    relative_drop = _zero_one(values.get("relative_drop"))
    zc_p05 = _as_float(values.get("zc_p05"))
    low_zc = zc_p05 is not None and zc_p05 <= 2.5
    near_low_zc = zc_p05 is not None and zc_p05 <= 3.0
    local_high = (severity is not None and severity >= 3.0) or relative_drop >= 0.6 or low_zc
    if confidence < 0.2 and (local_high or presence > 0.0):
        return (
            "uncertain_mixed_evidence",
            "low candidate confidence; review local CAST evidence cautiously",
        )
    if (
        presence <= 0.01
        and (severity is None or severity <= 0.0)
        and relative_drop < 0.15
        and not near_low_zc
    ):
        return ("clear_negative_evidence", "low presence, low relative drop, and no low-Zc support")
    if (severity is not None and severity >= 3.0) and (relative_drop >= 0.5 or low_zc):
        return (
            "strong_local_positive_evidence",
            "severe local CAST candidate with relative-drop or low-Zc support",
        )
    if (severity is not None and severity >= 2.0) or relative_drop >= 0.4 or low_zc:
        return (
            "moderate_positive_evidence",
            "moderate CAST severity, relative-drop, or low-Zc support",
        )
    if presence > 0.0 or relative_drop >= 0.2 or near_low_zc:
        return (
            "weak_local_positive_evidence",
            "some local CAST evidence, but interval-level support is weak",
        )
    return ("clear_negative_evidence", "no strong CAST review evidence in summary metrics")


def _bar_axis(
    axis: Any,
    values: dict[str, float],
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
) -> None:
    names = list(values)
    heights = [float(values[name]) for name in names]
    axis.bar(np.arange(len(names)), heights, color="tab:blue", alpha=0.8)
    axis.set_xticks(np.arange(len(names)), labels=names, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)


def _write_summary_table(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    csv_path: Path,
    json_path: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(csv_path, overwrite=overwrite)
    _ensure_can_write(json_path, overwrite=overwrite)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _update_review_summary(path: Path, *, overwrite: bool) -> None:
    existing = (
        path.read_text(encoding="utf-8") if path.exists() else "# Depth-Level Manual Review Pack\n"
    )
    start_marker = "<!-- depth-level-manual-review-visualization-caveats:start -->"
    end_marker = "<!-- depth-level-manual-review-visualization-caveats:end -->"
    section = "\n".join(
        [
            start_marker,
            "",
            "## Visualization Caveats",
            "",
            "- XSI raw features have different scales; do not compare raw feature "
            "panels by shared y-axis height.",
            "- Use XSI normalized robust-z panels for cross-feature comparison within an interval.",
            "- CAST summary metrics have mixed units; 0-1 metrics, severity, and Zc "
            "are plotted separately.",
            "- CAST severity, Zc, and fraction/confidence metrics should not be "
            "compared by bar height.",
            "",
            "## How to Review an Interval",
            "",
            "1. Inspect the CAST review-only evidence category and CAST heatmaps when available.",
            "2. Inspect the XSI normalized feature panel for relative interval anomalies.",
            "3. Compare the sanity score with the weak-label candidate class.",
            "4. Classify the interval as accept, reject, uncertain, or special handling.",
            "",
            "## Suggested Interpretation Rules",
            "",
            "- High CAST severity with low presence_fraction indicates a local anomaly, "
            "not necessarily a strong interval label.",
            "- Low zc_p05 near or below 2.5 MRayl supports low-Zc evidence.",
            "- High relative_drop supports relative anomaly evidence.",
            "- Low candidate_conf means review the interval with caution.",
            "- XSI high score with CAST clear negative may be a false positive or weak-label miss.",
            "- CAST positive with XSI low score may indicate XSI insensitivity or an "
            "overly broad weak-label candidate.",
            "",
            end_marker,
            "",
        ]
    )
    if start_marker in existing and end_marker in existing:
        prefix = existing.split(start_marker)[0].rstrip()
        suffix = existing.split(end_marker, 1)[1].lstrip()
        updated = f"{prefix}\n\n{section}{suffix}"
    else:
        updated = existing.rstrip() + "\n\n" + section
    if path.exists() and not overwrite and updated != existing:
        # Preserve the default no-overwrite contract for the summary addendum.
        return
    path.write_text(updated, encoding="utf-8")


def _xsi_interval_text_summary(
    interval_rows: list[dict[str, Any]],
    coefficient_map: dict[str, float],
) -> str:
    available_rows = [
        row for row in interval_rows if row.get("global_percentile_of_interval_mean") is not None
    ]
    positive = _top_coefficients(coefficient_map, positive=True)
    negative = _top_coefficients(coefficient_map, positive=False)
    highest = sorted(
        available_rows,
        key=lambda row: float(row.get("global_percentile_of_interval_mean") or -1.0),
        reverse=True,
    )[:4]
    lowest = sorted(
        available_rows,
        key=lambda row: float(row.get("global_percentile_of_interval_mean") or 101.0),
    )[:4]
    return "\n".join(
        [
            "top positive-contributing features: " + _format_feature_list(positive),
            "top negative-contributing features: " + _format_feature_list(negative),
            "highest percentile features: " + _format_row_feature_list(highest),
            "lowest percentile features: " + _format_row_feature_list(lowest),
            "normalized panel uses robust z-score; raw scales are shown only in small multiples",
        ]
    )


def _top_coefficients(
    coefficient_map: dict[str, float], *, positive: bool
) -> list[tuple[str, float]]:
    rows = [
        (name, value)
        for name, value in coefficient_map.items()
        if (value > 0.0 if positive else value < 0.0)
    ]
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows[:4]


def _format_feature_list(rows: list[tuple[str, float]]) -> str:
    if not rows:
        return "not_available"
    return ", ".join(f"{name} ({value:.3g})" for name, value in rows)


def _format_row_feature_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "not_available"
    return ", ".join(
        f"{row['feature_name']} ({float(row['global_percentile_of_interval_mean']):.1f} pct)"
        for row in rows
    )


def _coefficient_map(
    refinement_report: dict[str, Any],
    review_report: dict[str, Any],
    warnings: list[str],
) -> dict[str, float]:
    top_features = _as_dict(refinement_report.get("top_features"))
    best = _as_dict(refinement_report.get("best_result"))
    scenario_id = str(best.get("scenario_id") or review_report.get("source_scenario_id") or "")
    rows = _as_list(top_features.get(scenario_id))
    if not rows:
        for value in top_features.values():
            rows = _as_list(value)
            if rows:
                break
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("feature_name"):
            continue
        coefficient = _as_float(row.get("mean_coefficient"))
        if coefficient is None:
            continue
        result[str(row["feature_name"])] = coefficient
    if not result:
        warnings.append(
            "No refinement coefficient summary available for positive/negative feature lists."
        )
    return result


def _load_optional_npz(path: Path | str | None, warnings: list[str]) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    optional_path = Path(path)
    if not optional_path.exists():
        warnings.append(f"Optional manual review NPZ not found: {optional_path}")
        return {}
    with np.load(optional_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_optional_json(path: Path | str | None, warnings: list[str]) -> dict[str, Any]:
    if path is None:
        return {}
    optional_path = Path(path)
    if not optional_path.exists():
        warnings.append(f"Optional manual review JSON not found: {optional_path}")
        return {}
    return _read_json(optional_path)


def _cast_interval_rows(interval: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    if not arrays or "cast_depth" not in arrays:
        return np.asarray([], dtype=np.int32)
    depth = np.asarray(arrays["cast_depth"], dtype=np.float32).reshape(-1)
    start = float(interval.get("start_depth", 0.0))
    end = float(interval.get("end_depth", start))
    rows = np.flatnonzero((depth >= start) & (depth <= end)).astype(np.int32)
    if rows.size:
        return rows
    if depth.size == 0:
        return rows
    mid = np.asarray([(start + end) / 2.0], dtype=np.float32)
    return _nearest_indices(depth, mid)


def _interval_indices(depth: np.ndarray, interval: dict[str, Any]) -> np.ndarray:
    start = float(interval.get("start_depth", 0.0))
    end = float(interval.get("end_depth", start))
    indices = np.flatnonzero((depth >= start) & (depth <= end)).astype(np.int32)
    if indices.size:
        return indices
    if depth.size == 0:
        return indices
    return _nearest_indices(depth, np.asarray([(start + end) / 2.0], dtype=np.float32))


def _nearest_indices(reference_depth: np.ndarray, query_depth: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_depth, dtype=np.float32).reshape(-1)
    query = np.asarray(query_depth, dtype=np.float32).reshape(-1)
    order = np.argsort(reference)
    sorted_depth = reference[order]
    positions = np.searchsorted(sorted_depth, query)
    positions = np.clip(positions, 0, sorted_depth.size - 1)
    previous = np.clip(positions - 1, 0, sorted_depth.size - 1)
    use_previous = np.abs(sorted_depth[previous] - query) < np.abs(sorted_depth[positions] - query)
    nearest = np.where(use_previous, previous, positions)
    return order[nearest].astype(np.int32)


def _feature_column(feature_names: np.ndarray, feature_name: str) -> int | None:
    matches = np.flatnonzero(feature_names == feature_name)
    if matches.size == 0:
        return None
    return int(matches[0])


def _percentile_rank(values: np.ndarray, value: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(value):
        return None
    return float(np.mean(finite <= value) * 100.0)


def _robust_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    result = np.full(array.shape, np.nan, dtype=np.float32)
    if finite.size == 0:
        return result
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    result[np.isfinite(array)] = (finite - median) / scale
    return result


def _robust_z_scalar(values: np.ndarray, value: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(value):
        return None
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        return 0.0
    return float((value - median) / scale)


def _contribution_direction(coefficient: float | None) -> str:
    if coefficient is None:
        return "coefficient_unavailable"
    if coefficient > 0.0:
        return "positive"
    if coefficient < 0.0:
        return "negative"
    return "zero"


def _feature_scale_note(feature_name: str) -> str:
    if "energy" in feature_name:
        return "energy scale"
    if "peak_abs" in feature_name or "mean_abs" in feature_name:
        return "amplitude scale"
    if "late_over_early" in feature_name or "ratio" in feature_name:
        return "ratio scale"
    return "raw feature scale"


def _short_feature_label(feature_name: str) -> str:
    return feature_name.replace("_", "\n")[-34:]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _zero_one(value: Any) -> float:
    parsed = _as_float(value)
    if parsed is None or not np.isfinite(parsed):
        return 0.0
    return float(np.clip(parsed, 0.0, 1.0))


def _interval_score_arrays(
    intervals: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid_depth = []
    score_mean = []
    score_available = []
    for interval in intervals:
        start = float(interval.get("start_depth", 0.0))
        end = float(interval.get("end_depth", start))
        summary = _as_dict(interval.get("prediction_score_summary"))
        mid_depth.append((start + end) / 2.0)
        value = _as_float(summary.get("score_mean"))
        score_mean.append(np.nan if value is None else value)
        score_available.append(value is not None)
    return (
        np.asarray(mid_depth, dtype=np.float32),
        np.asarray(score_mean, dtype=np.float32),
        np.asarray(score_available, dtype=bool),
    )


def _optional_series(
    arrays: dict[str, np.ndarray],
    key: str,
    length: int,
) -> np.ndarray:
    if key not in arrays:
        return np.zeros(length, dtype=np.float32)
    return np.asarray(arrays[key], dtype=np.float32).reshape(-1)


def _add_interval_spans(axis: Any, intervals: list[dict[str, Any]]) -> None:
    for interval in intervals:
        start = float(interval.get("start_depth", 0.0))
        end = float(interval.get("end_depth", start))
        if bool(interval.get("5700_band_flag")):
            color = "tab:orange"
            alpha = 0.12
        else:
            color = "tab:gray"
            alpha = 0.06
        axis.axvspan(start, end, color=color, alpha=alpha, linewidth=0)


def _mean_interval_summary_value(
    intervals: list[dict[str, Any]],
    summary_key: str,
    value_key: str,
) -> float:
    values = [
        value
        for value in (_summary_float(interval, summary_key, value_key) for interval in intervals)
        if value is not None
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def _summary_float(
    interval: dict[str, Any],
    summary_key: str,
    value_key: str,
) -> float | None:
    return _as_float(_as_dict(interval.get(summary_key)).get(value_key))


def _validate_guardrails(
    report: dict[str, Any],
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    errors: list[str],
    warnings: list[str],
) -> None:
    if report.get("review_pack_version") != "depth_level_manual_review_v001":
        errors.append("review pack version is not depth_level_manual_review_v001.")
    if report.get("no_final_labels") is not True:
        errors.append("review pack report does not set no_final_labels=true.")
    for guardrail in ("no_final_labels", "no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        _check_npz_guardrail(label_arrays, guardrail, "label", errors, warnings)
        _check_npz_guardrail(feature_arrays, guardrail, "feature", errors, warnings)


def _check_npz_guardrail(
    arrays: dict[str, np.ndarray],
    guardrail: str,
    source: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if guardrail not in arrays:
        warnings.append(f"depth-level {source} NPZ has no {guardrail} field.")
        return
    if not bool(np.asarray(arrays[guardrail]).reshape(())):
        errors.append(f"depth-level {source} NPZ does not set {guardrail}=true.")


def _interval_type_colors() -> dict[str, str]:
    return {
        "true_positive_like": "tab:green",
        "clear_negative_like": "tab:blue",
        "false_positive_like": "tab:red",
        "false_negative_like": "tab:purple",
        "high_uncertainty": "tab:orange",
        "5700_band_review": "tab:brown",
        "boundary_case": "tab:gray",
    }


def _sample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, num=max_points).astype(int)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
