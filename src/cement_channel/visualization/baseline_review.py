from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.baseline_schema import MVP4B_BASELINE_REVIEW_VERSION
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

BASELINE_REVIEW_FILENAMES = {
    "fold_metric_summary": "01_fold_metric_summary.png",
    "score_distribution": "02_score_distribution_candidate_vs_non_candidate.png",
    "calibration_curve": "03_calibration_curve.png",
    "feature_coefficients": "04_feature_coefficient_plot.png",
    "depth_block_split": "05_depth_block_split_visualization.png",
    "permutation_check": "06_permutation_check_comparison.png",
    "plus_minus_audit": "07_plus_primary_vs_minus_audit_comparison.png",
}


@dataclass(frozen=True)
class BaselineReviewReport:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figures: dict[str, str]
    review_summary_template: str
    no_final_labels: bool
    no_deep_learning: bool
    no_stc: bool
    no_apes: bool
    no_production_model: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_baseline_review_figures(
    *,
    simple_baseline_report_json: Path | str,
    simple_baseline_csv: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_points: int = 20000,
) -> BaselineReviewReport:
    report = _read_json(Path(simple_baseline_report_json))
    rows = _read_prediction_rows(Path(simple_baseline_csv))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    if report.get("report_version") != "simple_baseline_v001":
        errors.append("simple baseline report_version must be simple_baseline_v001.")
    if report.get("production_training") is not False:
        errors.append("simple baseline report indicates production_training.")
    if report.get("no_final_labels") is not True:
        errors.append("simple baseline report does not set no_final_labels=true.")

    figures = {
        key: output / filename for key, filename in BASELINE_REVIEW_FILENAMES.items()
    }
    _save_fold_metric_summary(report, figures["fold_metric_summary"], overwrite=overwrite)
    _save_score_distribution(rows, figures["score_distribution"], overwrite=overwrite)
    _save_calibration_curve(rows, figures["calibration_curve"], overwrite=overwrite)
    _save_feature_coefficients(report, figures["feature_coefficients"], overwrite=overwrite)
    _save_depth_block_split(
        rows,
        figures["depth_block_split"],
        overwrite=overwrite,
        max_points=max_points,
    )
    _save_permutation_check(report, figures["permutation_check"], overwrite=overwrite)
    _save_plus_minus_audit(report, figures["plus_minus_audit"], overwrite=overwrite)
    summary_template = output / "review_summary_template.md"
    summary_json = output / "simple_baseline_review_summary_v001.json"
    _write_summary_template(summary_template, overwrite=overwrite)
    review_report = BaselineReviewReport(
        review_version=MVP4B_BASELINE_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "simple_baseline_report_json": str(simple_baseline_report_json),
            "simple_baseline_csv": str(simple_baseline_csv),
        },
        output_dir=str(output),
        figures={key: str(path) for key, path in figures.items()},
        review_summary_template=str(summary_template),
        no_final_labels=bool(report.get("no_final_labels") is True),
        no_deep_learning=bool(report.get("no_deep_learning") is True),
        no_stc=bool(report.get("no_stc") is True),
        no_apes=bool(report.get("no_apes") is True),
        no_production_model=bool(report.get("no_production_model") is True),
        warnings=warnings,
        errors=errors,
        not_performed=[
            "production training",
            "production inference",
            "deep learning",
            "STC",
            "APES",
            "final label generation",
            "model weight export",
            "MVP-4C",
            "MVP-5",
        ],
    )
    _ensure_can_write(summary_json, overwrite=overwrite)
    summary_json.write_text(
        json.dumps(review_report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return review_report


def _save_fold_metric_summary(
    report: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    rows = [
        row
        for row in report.get("fold_metrics", [])
        if not bool(row.get("permutation"))
    ]
    labels = [f"{row['model_type']} f{row['fold_index']}" for row in rows]
    balanced = [float(row["metrics"].get("balanced_accuracy") or 0.0) for row in rows]
    f1_values = [float(row["metrics"].get("f1") or 0.0) for row in rows]
    x_values = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.9), 5), constrained_layout=True)
    ax.bar(x_values - 0.18, balanced, width=0.36, label="balanced_accuracy")
    ax.bar(x_values + 0.18, f1_values, width=0.36, label="f1")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_values, labels=labels, rotation=35, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Fold metric summary")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_score_distribution(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    model = _preferred_model(rows)
    model_rows = [row for row in rows if row["model_type"] == model]
    candidate = np.array(
        [float(row["score"]) for row in model_rows if int(row["label_presence_plus"]) == 1],
        dtype=np.float32,
    )
    non_candidate = np.array(
        [float(row["score"]) for row in model_rows if int(row["label_presence_plus"]) == 0],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(non_candidate, bins=50, alpha=0.6, label="non-candidate", color="tab:blue")
    ax.hist(candidate, bins=50, alpha=0.6, label="candidate", color="tab:orange")
    ax.set_xlabel("Predicted sanity score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score distribution: {model}")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_calibration_curve(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    model = _preferred_model(rows)
    model_rows = [row for row in rows if row["model_type"] == model]
    score = np.array([float(row["score"]) for row in model_rows], dtype=np.float32)
    labels = np.array([int(row["label_presence_plus"]) for row in model_rows], dtype=np.int8)
    weights = np.array([float(row["sample_weight"]) for row in model_rows], dtype=np.float32)
    bins = np.linspace(0.0, 1.0, 11)
    mean_score: list[float] = []
    observed: list[float] = []
    for index in range(10):
        if index == 9:
            mask = (score >= bins[index]) & (score <= bins[index + 1])
        else:
            mask = (score >= bins[index]) & (score < bins[index + 1])
        weight_sum = float(np.sum(weights[mask]))
        if weight_sum <= 0.0:
            continue
        mean_score.append(float(np.average(score[mask], weights=weights[mask])))
        observed.append(float(np.average(labels[mask], weights=weights[mask])))
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0)
    ax.plot(mean_score, observed, marker="o", color="tab:green")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed candidate rate")
    ax.set_title(f"Calibration summary: {model}")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_feature_coefficients(
    report: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    coefficients = report.get("coefficient_summary", {})
    rows = [
        (key.split(":", 1)[1], float(value.get("mean_coefficient") or 0.0))
        for key, value in coefficients.items()
        if key.startswith("logistic_regression:")
    ]
    if not rows:
        rows = [
            (key.split(":", 1)[1], float(value.get("mean_coefficient") or 0.0))
            for key, value in coefficients.items()
        ]
    rows = sorted(rows, key=lambda item: abs(item[1]), reverse=True)[:12]
    if not rows:
        rows = [("no_coefficients", 0.0)]
    names = [row[0] for row in rows]
    values = [row[1] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(np.arange(len(values)), values, color="tab:purple")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(values)), labels=names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean coefficient")
    ax.set_title("Feature coefficient summary")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_depth_block_split(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
    max_points: int,
) -> None:
    plt = require_pyplot()
    model = _preferred_model(rows)
    model_rows = [row for row in rows if row["model_type"] == model]
    if len(model_rows) > max_points:
        indices = np.linspace(0, len(model_rows) - 1, num=max_points).astype(int)
        model_rows = [model_rows[index] for index in indices]
    depth = np.array([float(row["depth"]) for row in model_rows], dtype=np.float32)
    fold = np.array([int(row["fold_index"]) for row in model_rows], dtype=np.int16)
    label = np.array([int(row["label_presence_plus"]) for row in model_rows], dtype=np.int8)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    scatter = ax.scatter(depth, fold, c=label, s=5, cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Validation fold")
    ax.set_title(f"Depth-block split visualization: {model}")
    fig.colorbar(scatter, ax=ax, label="candidate label")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_permutation_check(report: dict[str, Any], output_path: Path, *, overwrite: bool) -> None:
    plt = require_pyplot()
    aggregate = report.get("aggregate_metrics", {})
    permutation = report.get("permutation_aggregate_metrics", {})
    model_types = sorted(aggregate)
    real = [float(aggregate[model].get("balanced_accuracy") or 0.0) for model in model_types]
    permuted = [
        float(_as_dict(permutation.get(model)).get("balanced_accuracy") or 0.0)
        for model in model_types
    ]
    x_values = np.arange(len(model_types))
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.bar(x_values - 0.18, real, width=0.36, label="real weak labels")
    ax.bar(x_values + 0.18, permuted, width=0.36, label="permuted labels")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_values, labels=model_types, rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Permutation sanity check")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_plus_minus_audit(report: dict[str, Any], output_path: Path, *, overwrite: bool) -> None:
    plt = require_pyplot()
    aggregate = report.get("aggregate_metrics", {})
    minus = report.get("minus_audit_comparison", {})
    model_types = sorted(aggregate)
    plus_values = [
        float(_as_dict(aggregate.get(model)).get("balanced_accuracy") or 0.0)
        for model in model_types
    ]
    minus_values = [
        float(_as_dict(minus.get(model)).get("balanced_accuracy") or 0.0)
        for model in model_types
    ]
    x_values = np.arange(len(model_types))
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.bar(x_values - 0.18, plus_values, width=0.36, label="plus primary")
    ax.bar(x_values + 0.18, minus_values, width=0.36, label="minus audit")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_values, labels=model_types, rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Plus primary vs minus audit comparison")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _write_summary_template(path: Path, *, overwrite: bool) -> None:
    _ensure_can_write(path, overwrite=overwrite)
    lines = [
        "# MVP-4B Simple Baseline Review Summary",
        "",
        "- Reviewer:",
        "- Review date:",
        "- Weak-label candidate agreement is interpretable:",
        "- Permutation check is lower than real weak-label sanity result:",
        "- No leakage concern from depth-block split:",
        "- Plus primary vs minus audit comparison reviewed:",
        "- Final labels were not claimed:",
        "- Production model was not claimed:",
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


def _preferred_model(rows: list[dict[str, Any]]) -> str:
    model_types = sorted({str(row["model_type"]) for row in rows})
    if "logistic_regression" in model_types:
        return "logistic_regression"
    if not model_types:
        raise ValueError("Prediction CSV has no model_type values.")
    return model_types[0]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
