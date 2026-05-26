from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.depth_level_baseline_schema import (
    DEPTH_LEVEL_BASELINE_REPORT_VERSION,
    DEPTH_LEVEL_BASELINE_REVIEW_VERSION,
)
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

DEPTH_LEVEL_BASELINE_REVIEW_FILENAMES = {
    "label_vs_prediction_score": "01_depth_label_vs_prediction_score.png",
    "feature_distributions": "02_positive_negative_feature_distributions.png",
    "depth_block_split": "03_depth_block_split_visualization.png",
    "permutation_comparison": "04_permutation_comparison.png",
    "top_feature_coefficients": "05_top_feature_coefficient_plot.png",
    "interval_review": "06_strong_positive_clear_negative_interval_review.png",
}


@dataclass(frozen=True)
class DepthLevelBaselineReviewReport:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    preferred_target_variant: str | None
    preferred_model_type: str | None
    figures: dict[str, str]
    review_summary_template: str
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


def generate_depth_level_baseline_review_figures(
    *,
    baseline_report_json: Path | str,
    baseline_csv: Path | str,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_points: int = 20000,
) -> DepthLevelBaselineReviewReport:
    report = _read_json(Path(baseline_report_json))
    rows = _read_prediction_rows(Path(baseline_csv))
    label_arrays = _load_npz(depth_level_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    _validate_report_guardrails(report, label_arrays, feature_arrays, errors, warnings)
    selection = _preferred_selection(report, rows)
    selected_rows = [
        row
        for row in rows
        if row.get("target_variant") == selection["target_variant"]
        and row.get("model_type") == selection["model_type"]
    ]
    if not selected_rows:
        errors.append("No prediction rows match the preferred depth-level baseline result.")
        selected_rows = rows

    figures = {
        key: output / filename
        for key, filename in DEPTH_LEVEL_BASELINE_REVIEW_FILENAMES.items()
    }
    _save_label_vs_prediction_score(
        selected_rows,
        figures["label_vs_prediction_score"],
        overwrite=overwrite,
        max_points=max_points,
    )
    _save_feature_distributions(
        selected_rows,
        report,
        feature_arrays,
        figures["feature_distributions"],
        selection=selection,
        overwrite=overwrite,
    )
    _save_depth_block_split(
        selected_rows,
        figures["depth_block_split"],
        overwrite=overwrite,
        max_points=max_points,
    )
    _save_permutation_comparison(report, figures["permutation_comparison"], overwrite=overwrite)
    _save_top_feature_coefficients(
        report,
        figures["top_feature_coefficients"],
        selection=selection,
        overwrite=overwrite,
    )
    _save_interval_review(
        selected_rows,
        label_arrays,
        figures["interval_review"],
        overwrite=overwrite,
        max_points=max_points,
    )
    summary_template = output / "review_summary_template.md"
    summary_json = output / "depth_level_baseline_review_summary_v001.json"
    _write_summary_template(summary_template, overwrite=overwrite)
    review = DepthLevelBaselineReviewReport(
        review_version=DEPTH_LEVEL_BASELINE_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "baseline_report_json": str(baseline_report_json),
            "baseline_csv": str(baseline_csv),
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
        },
        output_dir=str(output),
        preferred_target_variant=selection["target_variant"],
        preferred_model_type=selection["model_type"],
        figures={key: str(path) for key, path in figures.items()},
        review_summary_template=str(summary_template),
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


def _save_label_vs_prediction_score(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
    max_points: int,
) -> None:
    plt = require_pyplot()
    plot_rows = _sample_rows(rows, max_points=max_points)
    depth = _row_float_array(plot_rows, "depth")
    score = _row_float_array(plot_rows, "score")
    label = _row_int_array(plot_rows, "label")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    scatter = ax.scatter(depth, score, c=label, s=8, cmap="coolwarm", alpha=0.75)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Sanity score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Depth-level weak-label candidate vs baseline score")
    fig.colorbar(scatter, ax=ax, label="candidate label")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_feature_distributions(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    feature_arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    selection: dict[str, str | None],
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    top_names = _top_feature_names(report, selection, limit=4)
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
    row_depth = _row_float_array(rows, "depth")
    row_label = _row_int_array(rows, "label")
    feature_index = _nearest_indices(feature_depth, row_depth)
    fig, axes = plt.subplots(
        len(selected_indices),
        1,
        figsize=(8, max(3, len(selected_indices) * 2.6)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, feature_index_column, feature_name in zip(
        axes_array,
        selected_indices,
        top_names,
        strict=False,
    ):
        values = features[feature_index, feature_index_column]
        negative = values[row_label == 0]
        positive = values[row_label == 1]
        groups = [_finite_or_zero(negative), _finite_or_zero(positive)]
        axis.boxplot(groups, tick_labels=["negative", "positive"], showfliers=False)
        axis.set_ylabel(feature_name)
    axes_array[0].set_title("Positive/negative top-feature distributions")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_depth_block_split(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
    max_points: int,
) -> None:
    plt = require_pyplot()
    plot_rows = _sample_rows(rows, max_points=max_points)
    depth = _row_float_array(plot_rows, "depth")
    fold = _row_int_array(plot_rows, "fold_index")
    label = _row_int_array(plot_rows, "label")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    scatter = ax.scatter(depth, fold, c=label, s=8, cmap="coolwarm", alpha=0.75)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Validation fold")
    ax.set_title("Depth-block split visualization")
    fig.colorbar(scatter, ax=ax, label="candidate label")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_permutation_comparison(
    report: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    rows: list[tuple[str, float, float]] = []
    for variant, model_checks in _as_dict(report.get("permutation_check")).items():
        for model_type, check in _as_dict(model_checks).items():
            label = f"{variant}\n{model_type}"
            rows.append(
                (
                    label,
                    float(_as_float(_as_dict(check).get("real_balanced_accuracy")) or 0.0),
                    float(
                        _as_float(
                            _as_dict(check).get("permutation_balanced_accuracy")
                        )
                        or 0.0
                    ),
                )
            )
    if not rows:
        rows = [("no_result", 0.0, 0.0)]
    x_values = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5), constrained_layout=True)
    ax.bar(x_values - 0.18, [row[1] for row in rows], width=0.36, label="real labels")
    ax.bar(x_values + 0.18, [row[2] for row in rows], width=0.36, label="permuted labels")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_values, labels=[row[0] for row in rows], rotation=35, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Permutation comparison")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_top_feature_coefficients(
    report: dict[str, Any],
    output_path: Path,
    *,
    selection: dict[str, str | None],
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    rows = _top_feature_rows(report, selection, limit=12)
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
    ax.set_title("Top baseline coefficients")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_interval_review(
    rows: list[dict[str, Any]],
    label_arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    overwrite: bool,
    max_points: int,
) -> None:
    plt = require_pyplot()
    plot_rows = _sample_rows(rows, max_points=max_points)
    score_depth = _row_float_array(plot_rows, "depth")
    score = _row_float_array(plot_rows, "score")
    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    strong = np.asarray(label_arrays["depth_strong_positive_mask"], dtype=bool).reshape(-1)
    clear = np.asarray(label_arrays["depth_clear_negative_mask"], dtype=bool).reshape(-1)
    depth_min = float(np.min(score_depth)) if score_depth.size else float(np.min(depth))
    depth_max = float(np.max(score_depth)) if score_depth.size else float(np.max(depth))
    interval = (depth >= depth_min) & (depth <= depth_max)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(score_depth, score, color="tab:blue", linewidth=0.9, alpha=0.8, label="score")
    ax.scatter(
        depth[interval & clear],
        np.full(np.count_nonzero(interval & clear), 0.05),
        color="tab:green",
        s=12,
        alpha=0.8,
        label="clear negative",
    )
    ax.scatter(
        depth[interval & strong],
        np.full(np.count_nonzero(interval & strong), 0.95),
        color="tab:red",
        s=20,
        alpha=0.9,
        label="strong positive",
    )
    ax.set_xlabel("Depth")
    ax.set_ylabel("Score / interval marker")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Strong-positive / clear-negative interval review")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _validate_report_guardrails(
    report: dict[str, Any],
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    errors: list[str],
    warnings: list[str],
) -> None:
    if report.get("report_version") != DEPTH_LEVEL_BASELINE_REPORT_VERSION:
        errors.append("depth-level baseline report_version is not depth_level_baseline_v001.")
    if report.get("production_training") is not False:
        errors.append("depth-level baseline report indicates production_training.")
    for field_name in ("no_final_labels", "no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        if report.get(field_name) is not True:
            errors.append(f"depth-level baseline report does not set {field_name}=true.")
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


def _preferred_selection(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, str | None]:
    best = _as_dict(report.get("best_result"))
    variant = best.get("target_variant")
    model_type = best.get("model_type")
    if variant and model_type:
        return {"target_variant": str(variant), "model_type": str(model_type)}
    for row in rows:
        return {
            "target_variant": str(row.get("target_variant")),
            "model_type": str(row.get("model_type")),
        }
    return {"target_variant": None, "model_type": None}


def _top_feature_names(
    report: dict[str, Any],
    selection: dict[str, str | None],
    *,
    limit: int,
) -> list[str]:
    return [
        str(row["feature_name"])
        for row in _top_feature_rows(report, selection, limit=limit)
        if row.get("feature_name")
    ]


def _top_feature_rows(
    report: dict[str, Any],
    selection: dict[str, str | None],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    key = f"{selection['target_variant']}:{selection['model_type']}"
    rows = _as_list(_as_dict(report.get("top_features")).get(key))
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


def _write_summary_template(path: Path, *, overwrite: bool) -> None:
    _ensure_can_write(path, overwrite=overwrite)
    lines = [
        "# MVP-4B-R4b Depth-Level Baseline Review Summary",
        "",
        "- Reviewer:",
        "- Review date:",
        "- Preferred target variant is appropriate:",
        "- Real-label metric beats permutation with stable folds:",
        "- Predictions are not single-class degenerate:",
        "- Top features are physically plausible enough for further review:",
        "- Strong-positive and clear-negative intervals reviewed:",
        "- Final labels were not claimed:",
        "- MVP-4C/STC/APES/deep learning were not authorized:",
        "",
        "## Notes",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
