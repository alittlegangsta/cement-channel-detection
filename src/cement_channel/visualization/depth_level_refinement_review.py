from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.depth_level_refinement_schema import (
    DEPTH_LEVEL_REFINEMENT_REPORT_VERSION,
    DEPTH_LEVEL_REFINEMENT_REVIEW_VERSION,
)
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

DEPTH_LEVEL_REFINEMENT_REVIEW_FILENAMES = {
    "label_score_by_depth": "01_depth_label_score_by_depth.png",
    "feature_distributions": "02_positive_negative_feature_distributions.png",
    "feature_group_coefficients": "03_feature_group_coefficient_summary.png",
    "robustness_margin_heatmap": "04_robustness_margin_heatmap.png",
    "permutation_margin_distribution": "05_permutation_margin_distribution.png",
    "confidence_threshold_comparison": "06_confidence_threshold_comparison.png",
    "split_comparison": "07_depth_block_split_comparison.png",
    "exclude_5700_sensitivity": "08_exclude_5700_sensitivity.png",
}


@dataclass(frozen=True)
class DepthLevelRefinementReviewReport:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    preferred_scenario_id: str | None
    preferred_feature_group: str | None
    figures: dict[str, str]
    review_summary_template: str
    manual_confirmation_items: list[str]
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


def generate_depth_level_refinement_review_figures(
    *,
    refinement_report_json: Path | str,
    refinement_csv: Path | str,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_points: int = 20000,
) -> DepthLevelRefinementReviewReport:
    report = _read_json(Path(refinement_report_json))
    rows = _read_prediction_rows(Path(refinement_csv))
    label_arrays = _load_npz(depth_level_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(report, label_arrays, feature_arrays, errors, warnings)
    best = _as_dict(report.get("best_result"))
    scenario_id = None if not best else str(best.get("scenario_id"))
    feature_group = None if not best else str(best.get("feature_group"))
    selected_rows = [row for row in rows if row.get("scenario_id") == scenario_id]
    if not selected_rows and rows:
        selected_rows = rows
        warnings.append("No rows matched best scenario; plotted all refinement prediction rows.")

    figures = {
        key: output / filename
        for key, filename in DEPTH_LEVEL_REFINEMENT_REVIEW_FILENAMES.items()
    }
    _save_label_score_by_depth(
        selected_rows,
        figures["label_score_by_depth"],
        overwrite=overwrite,
        max_points=max_points,
    )
    _save_feature_distributions(
        selected_rows,
        report,
        feature_arrays,
        figures["feature_distributions"],
        overwrite=overwrite,
    )
    _save_feature_group_coefficients(
        report,
        figures["feature_group_coefficients"],
        overwrite=overwrite,
    )
    _save_robustness_margin_heatmap(
        report,
        figures["robustness_margin_heatmap"],
        overwrite=overwrite,
    )
    _save_permutation_margin_distribution(
        report,
        figures["permutation_margin_distribution"],
        overwrite=overwrite,
    )
    _save_comparison_bar(
        report,
        summary_key="confidence_threshold_summary",
        output_path=figures["confidence_threshold_comparison"],
        title="Confidence threshold robustness",
        xlabel="confidence threshold",
        overwrite=overwrite,
    )
    _save_comparison_bar(
        report,
        summary_key="split_summary",
        output_path=figures["split_comparison"],
        title="3-fold vs 5-fold depth-block split",
        xlabel="n_splits",
        overwrite=overwrite,
    )
    _save_comparison_bar(
        report,
        summary_key="exclude_5700_summary",
        output_path=figures["exclude_5700_sensitivity"],
        title="Exclude-5700 review-band sensitivity",
        xlabel="exclude_5700_band",
        overwrite=overwrite,
    )
    summary_template = output / "review_summary_template.md"
    summary_json = output / "depth_level_refinement_review_summary_v001.json"
    manual_items = [
        str(item)
        for item in _as_list(report.get("manual_confirmation_items"))
        if str(item)
    ]
    _write_summary_template(summary_template, manual_items=manual_items, overwrite=overwrite)
    review = DepthLevelRefinementReviewReport(
        review_version=DEPTH_LEVEL_REFINEMENT_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "refinement_report_json": str(refinement_report_json),
            "refinement_csv": str(refinement_csv),
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
        },
        output_dir=str(output),
        preferred_scenario_id=scenario_id,
        preferred_feature_group=feature_group,
        figures={key: str(path) for key, path in figures.items()},
        review_summary_template=str(summary_template),
        manual_confirmation_items=manual_items,
        no_final_labels=bool(report.get("no_final_labels") is True),
        no_stc=bool(report.get("no_stc") is True),
        no_apes=bool(report.get("no_apes") is True),
        no_deep_learning=bool(report.get("no_deep_learning") is True),
        no_mvp4c=bool(report.get("no_mvp4c") is True),
        no_production_model=bool(report.get("production_training") is False),
        warnings=warnings,
        errors=errors,
        not_performed=[
            "formal model performance claim",
            "production model training",
            "model weight export",
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
        json.dumps(review.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return review


def _save_label_score_by_depth(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
    max_points: int,
) -> None:
    plt = require_pyplot()
    plot_rows = _sample_rows(rows, max_points=max_points)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if plot_rows:
        depth = _row_float_array(plot_rows, "depth")
        score = _row_float_array(plot_rows, "score")
        label = _row_int_array(plot_rows, "label")
        scatter = ax.scatter(depth, score, c=label, s=8, cmap="coolwarm", alpha=0.75)
        fig.colorbar(scatter, ax=ax, label="weak-label candidate")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Sanity score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Depth-level weak-label candidate and score by depth - no final labels")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_feature_distributions(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    feature_arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    top_names = _top_feature_names(report, limit=4)
    if not top_names:
        top_names = feature_names[: min(4, feature_names.size)].tolist()
    selected_indices = [
        int(np.where(feature_names == name)[0][0])
        for name in top_names
        if np.any(feature_names == name)
    ]
    if not selected_indices:
        selected_indices = list(range(min(4, features.shape[1])))
        top_names = feature_names[selected_indices].tolist()
    row_depth = _row_float_array(rows, "depth") if rows else np.asarray([], dtype=np.float32)
    row_label = _row_int_array(rows, "label") if rows else np.asarray([], dtype=np.int32)
    feature_index = (
        _nearest_indices(feature_depth, row_depth)
        if row_depth.size
        else np.asarray([], dtype=np.int32)
    )
    fig, axes = plt.subplots(
        len(selected_indices),
        1,
        figsize=(8, max(3, len(selected_indices) * 2.6)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, feature_index_column, feature_name in zip(
        axes_array, selected_indices, top_names, strict=False
    ):
        values = (
            features[feature_index, feature_index_column]
            if feature_index.size
            else np.asarray([])
        )
        negative = values[row_label == 0] if values.size else np.asarray([])
        positive = values[row_label == 1] if values.size else np.asarray([])
        axis.boxplot(
            [_finite_or_zero(negative), _finite_or_zero(positive)],
            tick_labels=["negative", "positive"],
            showfliers=False,
        )
        axis.set_ylabel(feature_name)
    axes_array[0].set_title("Feature distributions by weak-label candidate class")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_feature_group_coefficients(
    report: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    rows = _top_feature_rows(report, limit=14)
    if not rows:
        rows = [{"feature_name": "no_coefficients", "mean_coefficient": 0.0}]
    names = [str(row["feature_name"]) for row in rows]
    values = [float(row.get("mean_coefficient") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh(np.arange(len(values)), values, color="tab:purple")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(values)), labels=names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean coefficient")
    ax.set_title("Best-scenario feature coefficients - review only")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_robustness_margin_heatmap(
    report: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    scenarios = _as_list(report.get("scenario_summaries"))
    groups = sorted({str(row.get("feature_group")) for row in scenarios if isinstance(row, dict)})
    thresholds = sorted(
        {float(row.get("confidence_threshold")) for row in scenarios if isinstance(row, dict)}
    )
    matrix = np.full((len(groups), len(thresholds)), np.nan, dtype=np.float32)
    for group_index, group in enumerate(groups):
        for threshold_index, threshold in enumerate(thresholds):
            values = [
                float(row["margin_mean"])
                for row in scenarios
                if isinstance(row, dict)
                and row.get("feature_group") == group
                and float(row.get("confidence_threshold")) == threshold
                and row.get("margin_mean") is not None
            ]
            if values:
                matrix[group_index, threshold_index] = float(np.max(values))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = ax.imshow(np.nan_to_num(matrix, nan=0.0), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(thresholds)), labels=[str(value) for value in thresholds])
    ax.set_yticks(np.arange(len(groups)), labels=groups)
    ax.set_xlabel("confidence threshold")
    ax.set_title("Robustness margin heatmap")
    fig.colorbar(image, ax=ax, label="best margin mean")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_permutation_margin_distribution(
    report: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    margins = np.asarray(
        [
            float(row["margin_mean"])
            for row in _as_list(report.get("scenario_summaries"))
            if isinstance(row, dict) and row.get("margin_mean") is not None
        ],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if margins.size:
        ax.hist(margins, bins=30, alpha=0.8, color="tab:blue")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.03, color="tab:orange", linewidth=1.0, linestyle="--", label="0.03")
    ax.axvline(0.05, color="tab:green", linewidth=1.0, linestyle="--", label="0.05")
    ax.set_xlabel("Real minus permutation balanced-accuracy margin")
    ax.set_ylabel("Scenario count")
    ax.set_title("Permutation margin distribution")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_comparison_bar(
    report: dict[str, Any],
    *,
    summary_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    summary = _as_dict(report.get(summary_key))
    labels = sorted(summary)
    best_values = [
        float(_as_dict(summary.get(label)).get("best_margin_mean") or 0.0)
        for label in labels
    ]
    passing = [
        int(_as_dict(summary.get(label)).get("passing_scenario_count") or 0)
        for label in labels
    ]
    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x_values = np.arange(len(labels))
    ax1.bar(x_values, best_values, color="tab:blue", alpha=0.75)
    ax1.set_xticks(x_values, labels=labels)
    ax1.set_ylabel("Best margin mean")
    ax1.set_xlabel(xlabel)
    ax1.set_title(title)
    ax2 = ax1.twinx()
    ax2.plot(x_values, passing, color="tab:red", marker="o", linewidth=1.5)
    ax2.set_ylabel("Passing scenario count")
    save_figure(fig, output_path, overwrite=overwrite)


def _validate_guardrails(
    report: dict[str, Any],
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    errors: list[str],
    warnings: list[str],
) -> None:
    if report.get("report_version") != DEPTH_LEVEL_REFINEMENT_REPORT_VERSION:
        errors.append("depth-level refinement report_version is not depth_level_refinement_v001.")
    if report.get("production_training") is not False:
        errors.append("depth-level refinement report indicates production_training.")
    for field_name in ("no_final_labels", "no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        if report.get(field_name) is not True:
            errors.append(f"depth-level refinement report does not set {field_name}=true.")
        _check_npz_guardrail(label_arrays, field_name, "label", errors, warnings)
        _check_npz_guardrail(feature_arrays, field_name, "feature", errors, warnings)


def _check_npz_guardrail(
    arrays: dict[str, np.ndarray],
    field_name: str,
    source: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if field_name not in arrays:
        warnings.append(f"depth-level {source} NPZ has no {field_name} field.")
        return
    if not bool(np.asarray(arrays[field_name]).reshape(())):
        errors.append(f"depth-level {source} NPZ does not set {field_name}=true.")


def _top_feature_names(report: dict[str, Any], *, limit: int) -> list[str]:
    return [
        str(row["feature_name"])
        for row in _top_feature_rows(report, limit=limit)
        if row.get("feature_name")
    ]


def _top_feature_rows(report: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    best = _as_dict(report.get("best_result"))
    scenario_id = str(best.get("scenario_id", ""))
    rows = _as_list(_as_dict(report.get("top_features")).get(scenario_id))
    if not rows:
        for values in _as_dict(report.get("top_features")).values():
            rows = _as_list(values)
            if rows:
                break
    sorted_rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: abs(float(row.get("mean_coefficient") or 0.0)),
        reverse=True,
    )
    return sorted_rows[:limit]


def _nearest_indices(reference_depth: np.ndarray, query_depth: np.ndarray) -> np.ndarray:
    order = np.argsort(reference_depth)
    sorted_depth = reference_depth[order]
    positions = np.searchsorted(sorted_depth, query_depth)
    positions = np.clip(positions, 0, sorted_depth.size - 1)
    previous = np.clip(positions - 1, 0, sorted_depth.size - 1)
    use_previous = np.abs(sorted_depth[previous] - query_depth) < np.abs(
        sorted_depth[positions] - query_depth
    )
    nearest = np.where(use_previous, previous, positions)
    return order[nearest]


def _write_summary_template(
    path: Path,
    *,
    manual_items: list[str],
    overwrite: bool,
) -> None:
    _ensure_can_write(path, overwrite=overwrite)
    lines = [
        "# MVP-4B-R4c Depth-Level Refinement Review Summary",
        "",
        "Review scope: depth-level weak-label candidate robustness only; no final labels.",
        "",
        "- Reviewer:",
        "- Review date:",
        "- Stable over permutation:",
        "- Robust across feature groups:",
        "- Robust across confidence thresholds:",
        "- Robust across 3-fold and 5-fold depth-block splits:",
        "- Robust when excluding the ~5700 ft review band:",
        "- No leakage concern:",
        "- MVP-4C/STC/APES/deep learning remain blocked:",
        "",
        "## Required Human Confirmation Items",
        "",
    ]
    lines.extend(_message_lines(manual_items))
    lines.extend(["", "## Notes", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _sample_rows(rows: list[dict[str, Any]], *, max_points: int) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=max_points).astype(int)
    return [rows[index] for index in indices]


def _row_float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float32)


def _row_int_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([int(row[key]) for row in rows], dtype=np.int32)


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    return finite


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
