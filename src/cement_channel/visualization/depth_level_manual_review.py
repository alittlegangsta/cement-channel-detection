from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

DEPTH_LEVEL_MANUAL_REVIEW_FIGURE_VERSION = "depth_level_manual_review_figures_v001"

DEPTH_LEVEL_MANUAL_REVIEW_FIGURE_FILENAMES = {
    "overview": "overview_depth_label_score_confidence.png",
    "selected_intervals": "selected_intervals_overview.png",
    "5700_sensitivity": "5700_band_sensitivity.png",
    "confidence_disagreement": "confidence_and_disagreement_panels.png",
}


@dataclass(frozen=True)
class DepthLevelManualReviewFigureReport:
    review_figure_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figure_count: int
    figures: dict[str, str]
    interval_cast_panel_count: int
    interval_xsi_panel_count: int
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    no_production_model: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_depth_level_manual_review_figures(
    *,
    review_intervals_json: Path | str,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_interval_panels: int = 50,
    max_points: int = 20000,
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
    cast_panel_count = _save_per_interval_cast_panels(
        intervals,
        output / "interval_cast_panels",
        overwrite=overwrite,
        max_interval_panels=max_interval_panels,
    )
    xsi_panel_count = _save_per_interval_xsi_panels(
        intervals,
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
    summary_json = output / "depth_level_manual_review_figures_summary_v001.json"
    figure_paths = {key: str(path) for key, path in figures.items()}
    if cast_panel_count:
        figure_paths["interval_cast_panels_dir"] = str(output / "interval_cast_panels")
    if xsi_panel_count:
        figure_paths["interval_xsi_feature_panels_dir"] = str(
            output / "interval_xsi_feature_panels"
        )
    figure_report = DepthLevelManualReviewFigureReport(
        review_figure_version=DEPTH_LEVEL_MANUAL_REVIEW_FIGURE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "review_intervals_json": str(review_intervals_json),
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
        },
        output_dir=str(output),
        figure_count=len(figures) + cast_panel_count + xsi_panel_count,
        figures=figure_paths,
        interval_cast_panel_count=cast_panel_count,
        interval_xsi_panel_count=xsi_panel_count,
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
    output_dir: Path,
    *,
    overwrite: bool,
    max_interval_panels: int,
) -> int:
    count = 0
    for interval in intervals[:max_interval_panels]:
        output_path = output_dir / f"{interval['review_id']}_cast_label_panels.png"
        values = _cast_panel_values(interval)
        _save_bar_panel(
            values,
            output_path,
            title=(
                f"{interval['review_id']} CAST weak-label candidate summary "
                "- no final labels"
            ),
            ylabel="summary value",
            overwrite=overwrite,
        )
        count += 1
    return count


def _save_per_interval_xsi_panels(
    intervals: list[dict[str, Any]],
    output_dir: Path,
    *,
    overwrite: bool,
    max_interval_panels: int,
) -> int:
    count = 0
    for interval in intervals[:max_interval_panels]:
        output_path = output_dir / f"{interval['review_id']}_xsi_feature_panels.png"
        values = _xsi_panel_values(interval)
        _save_bar_panel(
            values,
            output_path,
            title=f"{interval['review_id']} XSI feature summary - review only",
            ylabel="interval feature mean",
            overwrite=overwrite,
        )
        count += 1
    return count


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


def _save_bar_panel(
    values: dict[str, float],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    if not values:
        values = {"not_available": 0.0}
    names = list(values)
    heights = [float(values[name]) for name in names]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(np.arange(len(names)), heights, color="tab:blue", alpha=0.8)
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(fig, output_path, overwrite=overwrite)


def _cast_panel_values(interval: dict[str, Any]) -> dict[str, float]:
    summary = _as_dict(interval.get("cast_label_summary"))
    candidate = _as_dict(summary.get("weak_label_candidate_summary"))
    zc = _as_dict(summary.get("cast_zc_summary"))
    values = {
        "presence_fraction": _as_float(candidate.get("presence_plus_fraction")),
        "severity_max": _as_float(candidate.get("severity_plus_max")),
        "candidate_conf": _as_float(candidate.get("label_confidence_plus_mean")),
        "relative_drop": _as_float(candidate.get("relative_drop_plus_max")),
        "zc_p05": _as_float(zc.get("zc_p05")),
        "low_inc_fraction": _as_float(zc.get("low_inc_fraction")),
    }
    return {key: float(value) for key, value in values.items() if value is not None}


def _xsi_panel_values(interval: dict[str, Any]) -> dict[str, float]:
    summary = _as_dict(interval.get("xsi_feature_summary"))
    rows = _as_list(summary.get("top_feature_values"))
    values: dict[str, float] = {}
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        value = _as_float(row.get("mean"))
        if value is None:
            continue
        name = str(row.get("feature_name", "feature"))
        values[name[-34:]] = value
    return values


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
        for value in (
            _summary_float(interval, summary_key, value_key) for interval in intervals
        )
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
