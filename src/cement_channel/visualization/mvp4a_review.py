from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.evaluation.correlation_schema import MVP4A_REVIEW_VERSION
from cement_channel.visualization.matplotlib_utils import (
    finite_percentile_limits,
    image_extent,
    require_pyplot,
    sampled_depth_axis,
    sampled_image,
    save_figure,
)

REVIEW_FILENAMES = {
    "label_coverage_depth": "01_label_coverage_vs_depth.png",
    "xsi_side_energy_depth": "02_xsi_side_energy_vs_depth.png",
    "candidate_distribution": "03_candidate_vs_noncandidate_distribution.png",
    "severity_boxplot": "04_severity_vs_feature_boxplot.png",
    "high_confidence_comparison": "05_high_confidence_comparison.png",
    "disagreement_locations": "06_plus_minus_disagreement_locations.png",
    "correlation_heatmap": "07_correlation_summary_heatmap.png",
}


@dataclass(frozen=True)
class Mvp4aReviewReport:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figures: dict[str, str]
    review_summary_template: str
    no_model_training: bool
    no_final_labels: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_mvp4a_review_figures(
    *,
    label_samples_npz: Path | str,
    basic_features_npz: Path | str,
    correlation_csv: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_depth_pixels: int = 1200,
    max_distribution_samples: int = 20000,
) -> Mvp4aReviewReport:
    labels = _load_npz(label_samples_npz)
    features = _load_npz(basic_features_npz)
    correlation_rows = _read_correlation_rows(correlation_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    presence = np.asarray(labels["label_presence_plus"], dtype=np.int8)
    severity = np.asarray(labels["label_severity_plus"], dtype=np.int8)
    confidence = np.asarray(labels["label_confidence_plus"], dtype=np.float32)
    valid_high = np.asarray(labels["valid_for_azimuthal_validation"], dtype=bool)
    disagreement = np.asarray(labels["plus_minus_disagreement"], dtype=bool)
    depth = np.asarray(labels["xsi_depth"], dtype=np.float32)
    side_azimuth = np.asarray(labels["xsi_side_azimuth_deg"], dtype=np.float32)
    side_features = np.asarray(features["xsi_basic_features_by_side"], dtype=np.float32)
    feature_names = np.asarray(features["feature_names"]).astype(str).tolist()
    no_model_training = bool(np.asarray(features.get("no_model_training", False)).reshape(()))
    no_final_labels = bool(np.asarray(labels.get("no_final_labels", False)).reshape(()))
    if not no_model_training:
        errors.append("Basic feature artifact does not set no_model_training=true.")
    if not no_final_labels:
        errors.append("Label sample artifact does not set no_final_labels=true.")
    depth_count = min(presence.shape[0], side_features.shape[0], depth.size)
    presence = presence[:depth_count]
    severity = severity[:depth_count]
    confidence = confidence[:depth_count]
    valid_high = valid_high[:depth_count]
    disagreement = disagreement[:depth_count]
    depth = depth[:depth_count]
    side_features = side_features[:depth_count]
    primary_feature_index = _feature_index(feature_names, "rms_energy")
    primary_feature_name = feature_names[primary_feature_index]
    primary_feature = side_features[..., primary_feature_index]
    depth_axis = sampled_depth_axis(depth, depth_count, max_depth_pixels)
    side_axis = _side_axis(side_azimuth, presence.shape[1])
    figures = {key: output / filename for key, filename in REVIEW_FILENAMES.items()}

    _save_label_coverage(
        presence,
        confidence,
        valid_high,
        figures["label_coverage_depth"],
        depth=depth,
        max_depth_pixels=max_depth_pixels,
        overwrite=overwrite,
    )
    _save_heatmap(
        sampled_image(primary_feature, max_rows=max_depth_pixels),
        figures["xsi_side_energy_depth"],
        depth_axis=depth_axis,
        side_axis=side_axis,
        title=f"XSI Side Feature vs Depth: {primary_feature_name}",
        colorbar_label=primary_feature_name,
        cmap="viridis",
        overwrite=overwrite,
    )
    _save_candidate_distribution(
        primary_feature,
        presence,
        figures["candidate_distribution"],
        feature_name=primary_feature_name,
        max_samples=max_distribution_samples,
        overwrite=overwrite,
    )
    _save_severity_boxplot(
        primary_feature,
        severity,
        presence,
        figures["severity_boxplot"],
        feature_name=primary_feature_name,
        max_samples=max_distribution_samples,
        overwrite=overwrite,
    )
    _save_high_confidence_comparison(
        primary_feature,
        presence,
        valid_high,
        figures["high_confidence_comparison"],
        feature_name=primary_feature_name,
        max_samples=max_distribution_samples,
        overwrite=overwrite,
    )
    _save_mask_heatmap(
        disagreement,
        figures["disagreement_locations"],
        depth_axis=depth_axis,
        side_axis=side_axis,
        title="Plus/Minus Disagreement Locations",
        label="disagreement",
        overwrite=overwrite,
        max_depth_pixels=max_depth_pixels,
    )
    _save_correlation_heatmap(
        correlation_rows,
        figures["correlation_heatmap"],
        overwrite=overwrite,
    )

    template = output / "review_summary_template.md"
    _ensure_can_write(template, overwrite=overwrite)
    template.write_text(_review_template(), encoding="utf-8")
    report = Mvp4aReviewReport(
        review_version=MVP4A_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "label_samples_npz": str(label_samples_npz),
            "basic_features_npz": str(basic_features_npz),
            "correlation_csv": str(correlation_csv),
        },
        output_dir=str(output),
        figures={key: str(path) for key, path in figures.items()},
        review_summary_template=str(template),
        no_model_training=no_model_training,
        no_final_labels=no_final_labels,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "model training",
            "STC",
            "APES",
            "final label approval",
            "MVP-4B feature engineering",
        ],
    )
    (output / "mvp4a_review_summary_v001.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _save_label_coverage(
    presence: np.ndarray,
    confidence: np.ndarray,
    valid_high: np.ndarray,
    output_path: Path,
    *,
    depth: np.ndarray,
    max_depth_pixels: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    candidate_fraction = np.mean(presence == 1, axis=1)
    high_fraction = np.mean(valid_high, axis=1)
    mean_confidence = np.mean(confidence, axis=1)
    depth_axis = sampled_depth_axis(depth, presence.shape[0], max_depth_pixels)
    indices = np.linspace(0, presence.shape[0] - 1, num=depth_axis.size).astype(int)
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    ax.plot(candidate_fraction[indices], depth_axis, label="candidate fraction", color="tab:red")
    ax.plot(high_fraction[indices], depth_axis, label="high-confidence valid", color="tab:blue")
    ax.plot(mean_confidence[indices], depth_axis, label="mean label confidence", color="tab:green")
    ax.set_title("MVP-4A Label Coverage vs Depth")
    ax.set_xlabel("Fraction / confidence")
    ax.set_ylabel("Depth")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_heatmap(
    values: np.ndarray,
    output_path: Path,
    *,
    depth_axis: np.ndarray,
    side_axis: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    image = np.asarray(values, dtype=np.float32)
    vmin, vmax = finite_percentile_limits(image)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        image,
        aspect="auto",
        origin="upper",
        extent=image_extent(x_axis=side_axis, y_axis=depth_axis),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("XSI side azimuth (deg)")
    ax.set_ylabel("Depth")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    save_figure(fig, output_path, overwrite=overwrite)


def _save_candidate_distribution(
    values: np.ndarray,
    presence: np.ndarray,
    output_path: Path,
    *,
    feature_name: str,
    max_samples: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    candidate = _sample(values[presence == 1], max_samples)
    non_candidate = _sample(values[presence == 0], max_samples)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(non_candidate, bins=40, alpha=0.55, label="non-candidate", color="tab:blue")
    ax.hist(candidate, bins=40, alpha=0.55, label="candidate", color="tab:red")
    ax.set_title("Candidate vs Non-Candidate Feature Distribution")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_severity_boxplot(
    values: np.ndarray,
    severity: np.ndarray,
    presence: np.ndarray,
    output_path: Path,
    *,
    feature_name: str,
    max_samples: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    groups = [
        _sample(values[_severity_mask(severity, presence, level)], max_samples)
        for level in range(4)
    ]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.boxplot(groups, labels=["none", "mild", "moderate", "severe"], showfliers=False)
    ax.set_title("Severity vs Feature")
    ax.set_ylabel(feature_name)
    save_figure(fig, output_path, overwrite=overwrite)


def _severity_mask(severity: np.ndarray, presence: np.ndarray, level: int) -> np.ndarray:
    if level == 0:
        return (severity == 0) & (presence >= 0)
    return (severity == level) & (presence == 1)


def _save_high_confidence_comparison(
    values: np.ndarray,
    presence: np.ndarray,
    valid_high: np.ndarray,
    output_path: Path,
    *,
    feature_name: str,
    max_samples: int,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    candidate = _sample(values[valid_high & (presence == 1)], max_samples)
    non_candidate = _sample(values[valid_high & (presence == 0)], max_samples)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.boxplot([non_candidate, candidate], labels=["non-candidate", "candidate"], showfliers=False)
    ax.set_title("High-Confidence-Only Comparison")
    ax.set_ylabel(feature_name)
    save_figure(fig, output_path, overwrite=overwrite)


def _save_mask_heatmap(
    mask: np.ndarray,
    output_path: Path,
    *,
    depth_axis: np.ndarray,
    side_axis: np.ndarray,
    title: str,
    label: str,
    overwrite: bool,
    max_depth_pixels: int,
) -> None:
    _save_heatmap(
        sampled_image(mask.astype(np.float32), max_rows=max_depth_pixels),
        output_path,
        depth_axis=depth_axis,
        side_axis=side_axis,
        title=title,
        colorbar_label=label,
        cmap="magma",
        overwrite=overwrite,
    )


def _save_correlation_heatmap(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    subsets = ["all_known", "high_confidence", "low_confidence", "plus_minus_disagreement"]
    features = sorted(
        {
            row["feature"]
            for row in rows
            if row.get("label_convention") == "plus_primary" and row.get("feature")
        }
    )
    if not features:
        features = ["no_data"]
    matrix = np.zeros((len(subsets), len(features)), dtype=np.float32)
    for row in rows:
        if row.get("label_convention") != "plus_primary":
            continue
        subset = row.get("subset")
        feature = row.get("feature")
        if subset not in subsets or feature not in features:
            continue
        value = _float_or_none(row.get("point_biserial_effect_size"))
        if value is None:
            value = _float_or_none(row.get("weighted_difference_fraction"))
        matrix[subsets.index(subset), features.index(feature)] = 0.0 if value is None else value
    limit = max(float(np.nanmax(np.abs(matrix))), 1.0e-6)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title("Correlation Summary Heatmap")
    ax.set_xticks(np.arange(len(features)), labels=features, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(subsets)), labels=subsets)
    fig.colorbar(im, ax=ax, label="effect or weighted difference fraction")
    save_figure(fig, output_path, overwrite=overwrite)


def _review_template() -> str:
    return "\n".join(
        [
            "# MVP-4A Review Summary",
            "",
            "- Reviewer:",
            "- Review date:",
            "- Depth intervals inspected:",
            "- Label coverage vs depth acceptable: TODO",
            "- XSI side energy depth pattern physically plausible: TODO",
            "- Candidate/non-candidate distributions inspected: TODO",
            "- Severity trend inspected without treating candidates as ground truth: TODO",
            "- High-confidence-only comparison inspected: TODO",
            "- Plus/minus disagreement locations inspected: TODO",
            "- Correlation heatmap supports go / conditional_go / no_go: TODO",
            "- No model training was performed: TODO",
            "- No final labels were claimed: TODO",
            "",
        ]
    )


def _feature_index(feature_names: list[str], preferred: str) -> int:
    if preferred in feature_names:
        return feature_names.index(preferred)
    return 0


def _side_axis(values: np.ndarray, width: int) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float32).reshape(-1)
    if axis.size != width:
        return np.linspace(0.0, 360.0, num=width, endpoint=False, dtype=np.float32)
    return axis


def _sample(values: np.ndarray, max_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.array([np.nan], dtype=np.float32)
    if array.size <= max_samples:
        return array
    indices = np.linspace(0, array.size - 1, num=max_samples).astype(int)
    return array[indices]


def _read_correlation_rows(path: Path | str) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _float_or_none(value: str | None) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
