from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.labels.label_quality_schema import (
    MVP4B_LABEL_QUALITY_AUDIT_VERSION,
    LabelQualityConfig,
    load_label_quality_config,
)


@dataclass(frozen=True)
class SubsetFeatureAuditReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    output_csv: str
    output_figure_dir: str
    sample_count: int
    feature_count: int
    subset_pair_summaries: dict[str, dict[str, int | float | None]]
    feature_group_summaries: list[dict[str, Any]]
    top_feature_rows: list[dict[str, Any]]
    signal_enhancement: dict[str, float | bool | str | None]
    review_exclusion_sensitivity: dict[str, float | bool | None]
    label_noise_likely: bool
    controlled_time_frequency_sanity_recommended: bool
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_subset_feature_separation_from_config(
    *,
    sample_table_npz: Path | str,
    label_quality_subsets_npz: Path | str,
    label_quality_config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    output_figure_dir: Path | str,
    overwrite: bool = False,
) -> SubsetFeatureAuditReport:
    config = load_label_quality_config(label_quality_config_path)
    sample_arrays = _load_npz(sample_table_npz)
    subset_arrays = _load_npz(label_quality_subsets_npz)
    report, rows = audit_subset_feature_separation(
        sample_arrays=sample_arrays,
        subset_arrays=subset_arrays,
        config=config,
        inputs={
            "sample_table_npz": str(sample_table_npz),
            "label_quality_subsets_npz": str(label_quality_subsets_npz),
            "label_quality_config_path": str(label_quality_config_path),
        },
        output_csv=Path(output_csv),
        output_figure_dir=Path(output_figure_dir),
    )
    write_subset_feature_audit_outputs(
        report,
        rows,
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        output_csv=Path(output_csv),
        output_figure_dir=Path(output_figure_dir),
        overwrite=overwrite,
    )
    return report


def audit_subset_feature_separation(
    *,
    sample_arrays: dict[str, np.ndarray],
    subset_arrays: dict[str, np.ndarray],
    config: LabelQualityConfig,
    inputs: dict[str, str] | None = None,
    output_csv: Path | None = None,
    output_figure_dir: Path | None = None,
) -> tuple[SubsetFeatureAuditReport, list[dict[str, Any]]]:
    features = np.asarray(sample_arrays["transformed_features"], dtype=np.float32)
    feature_names = np.asarray(sample_arrays["transformed_feature_names"]).astype(str)
    label = np.asarray(sample_arrays["label_presence_plus"], dtype=np.int8).reshape(-1)
    _validate_inputs(features, feature_names, label, subset_arrays)
    subset_pairs = subset_pair_masks(label, subset_arrays)
    groups = feature_group_indices(feature_names, sample_arrays)
    rows = effect_size_rows(features, feature_names, subset_pairs, groups)
    summaries = feature_group_summaries(rows)
    subset_summaries = {
        name: {
            "candidate_count": int(np.count_nonzero(pair["candidate"])),
            "clear_negative_count": int(np.count_nonzero(pair["negative"])),
            "sample_count": int(
                np.count_nonzero(pair["candidate"]) + np.count_nonzero(pair["negative"])
            ),
            "candidate_fraction": _candidate_fraction(pair["candidate"], pair["negative"]),
        }
        for name, pair in subset_pairs.items()
    }
    top_rows = sorted(
        rows,
        key=lambda row: abs(float(row["standardized_difference"] or 0.0)),
        reverse=True,
    )[:30]
    enhancement = signal_enhancement_summary(
        summaries,
        config=config,
    )
    review_sensitivity = review_exclusion_sensitivity(
        summaries,
        max_flip_fraction=config.gate.max_result_flip_fraction_from_review_exclusion,
    )
    errors: list[str] = []
    warnings: list[str] = []
    if bool(review_sensitivity["result_flip_exceeds_threshold"]):
        errors.append("5700 ft review exclusion causes excessive sign-flip sensitivity.")
    if not bool(enhancement["label_noise_likely"]):
        warnings.append(
            "label-quality subsets did not produce clear feature-separation enhancement."
        )
    report = SubsetFeatureAuditReport(
        report_version=MVP4B_LABEL_QUALITY_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_csv=str(output_csv) if output_csv else "",
        output_figure_dir=str(output_figure_dir) if output_figure_dir else "",
        sample_count=int(label.size),
        feature_count=int(features.shape[1]),
        subset_pair_summaries=subset_summaries,
        feature_group_summaries=summaries,
        top_feature_rows=top_rows,
        signal_enhancement=enhancement,
        review_exclusion_sensitivity=review_sensitivity,
        label_noise_likely=bool(enhancement["label_noise_likely"]),
        controlled_time_frequency_sanity_recommended=not bool(enhancement["label_noise_likely"]),
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "model training",
            "baseline fitting",
            "permutation model evaluation",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, rows


def subset_pair_masks(
    label: np.ndarray,
    subset_arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    disagreement_free = _mask(subset_arrays, "disagreement_free_mask", label.size)
    high_orientation = _mask(subset_arrays, "high_confidence_orientation_mask", label.size)
    connected = _mask(subset_arrays, "connected_object_only_mask", label.size)
    review = _mask(subset_arrays, "review_exclusion_mask", label.size)
    strong = _mask(subset_arrays, "strong_positive_mask", label.size)
    clear = _mask(subset_arrays, "clear_negative_mask", label.size)
    quality_strong = _mask(subset_arrays, "quality_strong_positive_mask", label.size)
    quality_clear = _mask(subset_arrays, "quality_clear_negative_mask", label.size)
    return {
        "all_candidates_vs_non_candidates": {
            "candidate": label == 1,
            "negative": label == 0,
        },
        "disagreement_free": {
            "candidate": (label == 1) & disagreement_free,
            "negative": (label == 0) & disagreement_free,
        },
        "high_confidence_orientation": {
            "candidate": (label == 1) & disagreement_free & high_orientation,
            "negative": (label == 0) & disagreement_free & high_orientation,
        },
        "connected_object_vs_clear_negative": {
            "candidate": strong & high_orientation & connected,
            "negative": clear & high_orientation,
        },
        "quality_strong_vs_clear": {
            "candidate": quality_strong,
            "negative": quality_clear,
        },
        "quality_strong_vs_clear_with_review_band": {
            "candidate": strong & high_orientation & connected,
            "negative": clear & high_orientation,
        },
        "review_band_excluded_only": {
            "candidate": (label == 1) & review,
            "negative": (label == 0) & review,
        },
    }


def feature_group_indices(
    feature_names: np.ndarray,
    sample_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    names = feature_names.astype(str)
    receiver_added = {
        str(name)
        for name in np.asarray(
            sample_arrays.get("receiver_transformed_feature_names_added", []),
        ).astype(str)
    }
    receiver = np.asarray(
        [(name in receiver_added) or ("receiver_" in name) for name in names],
        dtype=bool,
    )
    late_ratio = np.asarray(["late_over_early" in name for name in names], dtype=bool)
    far_near = np.asarray(
        [
            ("receiver_" in name)
            and (
                "far_" in name
                or "_far" in name
                or "near_" in name
                or "_near" in name
            )
            for name in names
        ],
        dtype=bool,
    )
    side_level = ~receiver
    groups = {
        "side_level_enhanced": np.flatnonzero(side_level),
        "receiver_derived": np.flatnonzero(receiver),
        "late_over_early": np.flatnonzero(late_ratio),
        "far_near_receiver": np.flatnonzero(far_near),
        "side_plus_receiver": np.arange(names.size),
    }
    return {name: indices.astype(np.int32) for name, indices in groups.items() if indices.size}


def effect_size_rows(
    features: np.ndarray,
    feature_names: np.ndarray,
    subset_pairs: dict[str, dict[str, np.ndarray]],
    feature_groups: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset_name, pair in subset_pairs.items():
        candidate = pair["candidate"]
        negative = pair["negative"]
        for group_name, indices in feature_groups.items():
            for feature_index in indices:
                row = _effect_size_row(
                    features[:, feature_index],
                    feature_name=str(feature_names[feature_index]),
                    feature_group=group_name,
                    subset_name=subset_name,
                    candidate=candidate,
                    negative=negative,
                )
                rows.append(row)
    return rows


def feature_group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["subset_name"]), str(row["feature_group"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (subset_name, group_name), group_rows in sorted(grouped.items()):
        valid_rows = [
            row
            for row in group_rows
            if row["standardized_difference"] is not None
            and np.isfinite(float(row["standardized_difference"]))
        ]
        if not valid_rows:
            summaries.append(
                {
                    "subset_name": subset_name,
                    "feature_group": group_name,
                    "feature_count": len(group_rows),
                    "top_abs_standardized_difference": None,
                    "top_feature_name": None,
                    "top_standardized_difference": None,
                    "mean_abs_top5_standardized_difference": None,
                    "candidate_count": None,
                    "clear_negative_count": None,
                }
            )
            continue
        sorted_rows = sorted(
            valid_rows,
            key=lambda row: abs(float(row["standardized_difference"])),
            reverse=True,
        )
        top = sorted_rows[0]
        top5 = sorted_rows[:5]
        summaries.append(
            {
                "subset_name": subset_name,
                "feature_group": group_name,
                "feature_count": len(group_rows),
                "top_abs_standardized_difference": abs(
                    float(top["standardized_difference"])
                ),
                "top_feature_name": top["feature_name"],
                "top_standardized_difference": float(top["standardized_difference"]),
                "mean_abs_top5_standardized_difference": float(
                    np.mean([abs(float(row["standardized_difference"])) for row in top5])
                ),
                "candidate_count": int(top["candidate_count"]),
                "clear_negative_count": int(top["clear_negative_count"]),
            }
        )
    return summaries


def signal_enhancement_summary(
    summaries: list[dict[str, Any]],
    *,
    config: LabelQualityConfig,
) -> dict[str, float | bool | str | None]:
    all_best = _best_summary_value(summaries, "all_candidates_vs_non_candidates")
    quality_best = _best_summary_value(summaries, "quality_strong_vs_clear")
    delta = None if all_best is None or quality_best is None else quality_best - all_best
    label_noise_likely = (
        delta is not None
        and delta >= config.gate.signal_enhancement_effect_size_delta
        and quality_best is not None
        and quality_best >= config.gate.strong_signal_effect_size_threshold
    )
    return {
        "all_candidate_best_abs_effect_size": all_best,
        "quality_subset_best_abs_effect_size": quality_best,
        "quality_minus_all_delta": delta,
        "required_delta": config.gate.signal_enhancement_effect_size_delta,
        "strong_signal_threshold": config.gate.strong_signal_effect_size_threshold,
        "label_noise_likely": bool(label_noise_likely),
        "interpretation": (
            "label-quality subset strengthens separation"
            if label_noise_likely
            else "label-quality subset does not clearly strengthen separation"
        ),
    }


def review_exclusion_sensitivity(
    summaries: list[dict[str, Any]],
    *,
    max_flip_fraction: float | None = None,
) -> dict[str, float | bool | None]:
    without_review = _summary_by_subset(summaries, "quality_strong_vs_clear")
    with_review = _summary_by_subset(summaries, "quality_strong_vs_clear_with_review_band")
    if not without_review or not with_review:
        return {
            "compared_group_count": 0,
            "sign_flip_fraction": None,
            "max_abs_effect_size_delta": None,
            "result_flip_exceeds_threshold": False,
        }
    flips = 0
    deltas: list[float] = []
    compared = 0
    for group_name, row_without in without_review.items():
        row_with = with_review.get(group_name)
        if row_with is None:
            continue
        effect_without = _as_float(row_without.get("top_standardized_difference"))
        effect_with = _as_float(row_with.get("top_standardized_difference"))
        if effect_without is None or effect_with is None:
            continue
        compared += 1
        if np.sign(effect_without) != np.sign(effect_with):
            flips += 1
        deltas.append(abs(abs(effect_without) - abs(effect_with)))
    flip_fraction = None if compared == 0 else flips / compared
    threshold = 0.50 if max_flip_fraction is None else max_flip_fraction
    return {
        "compared_group_count": compared,
        "sign_flip_fraction": flip_fraction,
        "max_abs_effect_size_delta": None if not deltas else float(max(deltas)),
        "result_flip_exceeds_threshold": (
            False if flip_fraction is None else flip_fraction > threshold
        ),
    }


def write_subset_feature_audit_outputs(
    report: SubsetFeatureAuditReport,
    rows: list[dict[str, Any]],
    *,
    output_md: Path,
    output_json: Path,
    output_csv: Path,
    output_figure_dir: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_csv, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_subset_feature_audit_markdown(report), encoding="utf-8")
    write_effect_size_csv(rows, output_csv)
    write_review_figures(report, rows, output_figure_dir)


def write_effect_size_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "subset_name",
        "feature_group",
        "feature_name",
        "candidate_count",
        "clear_negative_count",
        "candidate_mean",
        "clear_negative_mean",
        "candidate_median",
        "clear_negative_median",
        "standardized_difference",
        "median_difference",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_review_figures(
    report: SubsetFeatureAuditReport,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    summaries = report.feature_group_summaries
    subset_names = sorted({str(row["subset_name"]) for row in summaries})
    group_names = sorted({str(row["feature_group"]) for row in summaries})
    matrix = np.full((len(subset_names), len(group_names)), np.nan, dtype=np.float32)
    lookup = {
        (str(row["subset_name"]), str(row["feature_group"])): row for row in summaries
    }
    for i, subset_name in enumerate(subset_names):
        for j, group_name in enumerate(group_names):
            value = lookup.get((subset_name, group_name), {}).get(
                "top_abs_standardized_difference"
            )
            if value is not None:
                matrix[i, j] = float(value)
    fig, ax = plt.subplots(figsize=(max(7, len(group_names) * 1.3), 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(group_names)), labels=group_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(subset_names)), labels=subset_names)
    ax.set_title("Top absolute effect size by subset and feature group")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "subset_feature_effect_size_heatmap.png", dpi=150)
    plt.close(fig)

    top_rows = rows[:]
    top_rows.sort(
        key=lambda row: abs(float(row["standardized_difference"] or 0.0)),
        reverse=True,
    )
    if top_rows:
        names = [str(row["feature_name"])[:40] for row in top_rows[:12]]
        values = [float(row["standardized_difference"] or 0.0) for row in top_rows[:12]]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(np.arange(len(values)), values)
        ax.set_yticks(np.arange(len(values)), labels=names)
        ax.invert_yaxis()
        ax.set_xlabel("standardized difference")
        ax.set_title("Top subset feature separations")
        fig.tight_layout()
        fig.savefig(output_dir / "subset_feature_top_effects.png", dpi=150)
        plt.close(fig)


def format_subset_feature_audit_markdown(report: SubsetFeatureAuditReport) -> str:
    lines = [
        "# MVP-4B-R3 Subset Feature Separation Audit",
        "",
        "This is a weak-label feature-separation audit. It does not train a model "
        "and does not claim final-label performance.",
        "",
        f"- report_version: `{report.report_version}`",
        f"- sample_count: {report.sample_count}",
        f"- feature_count: {report.feature_count}",
        f"- label_noise_likely: `{report.label_noise_likely}`",
        "- controlled_time_frequency_sanity_recommended: "
        f"`{report.controlled_time_frequency_sanity_recommended}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Signal Enhancement",
        "",
    ]
    for key, value in report.signal_enhancement.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Best Group Summaries", ""])
    for row in sorted(
        report.feature_group_summaries,
        key=lambda item: abs(float(item["top_abs_standardized_difference"] or 0.0)),
        reverse=True,
    )[:15]:
        lines.append(
            "- "
            f"{row['subset_name']} / {row['feature_group']}: "
            f"top_abs_effect={row['top_abs_standardized_difference']}, "
            f"top_feature={row['top_feature_name']}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _effect_size_row(
    values: np.ndarray,
    *,
    feature_name: str,
    feature_group: str,
    subset_name: str,
    candidate: np.ndarray,
    negative: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    pos = candidate & finite
    neg = negative & finite
    if not np.any(pos) or not np.any(neg):
        return {
            "subset_name": subset_name,
            "feature_group": feature_group,
            "feature_name": feature_name,
            "candidate_count": int(np.count_nonzero(pos)),
            "clear_negative_count": int(np.count_nonzero(neg)),
            "candidate_mean": None,
            "clear_negative_mean": None,
            "candidate_median": None,
            "clear_negative_median": None,
            "standardized_difference": None,
            "median_difference": None,
        }
    pos_values = values[pos].astype(np.float64)
    neg_values = values[neg].astype(np.float64)
    pos_mean = float(np.mean(pos_values))
    neg_mean = float(np.mean(neg_values))
    pooled = float(np.sqrt(0.5 * (np.var(pos_values) + np.var(neg_values))))
    effect = None if pooled <= 0.0 else (pos_mean - neg_mean) / pooled
    pos_median = float(np.median(pos_values))
    neg_median = float(np.median(neg_values))
    return {
        "subset_name": subset_name,
        "feature_group": feature_group,
        "feature_name": feature_name,
        "candidate_count": int(np.count_nonzero(pos)),
        "clear_negative_count": int(np.count_nonzero(neg)),
        "candidate_mean": pos_mean,
        "clear_negative_mean": neg_mean,
        "candidate_median": pos_median,
        "clear_negative_median": neg_median,
        "standardized_difference": None if effect is None else float(effect),
        "median_difference": pos_median - neg_median,
    }


def _best_summary_value(summaries: list[dict[str, Any]], subset_name: str) -> float | None:
    values = [
        float(row["top_abs_standardized_difference"])
        for row in summaries
        if row["subset_name"] == subset_name and row["top_abs_standardized_difference"] is not None
    ]
    return None if not values else float(max(values))


def _summary_by_subset(
    summaries: list[dict[str, Any]],
    subset_name: str,
) -> dict[str, dict[str, Any]]:
    return {
        str(row["feature_group"]): row
        for row in summaries
        if row["subset_name"] == subset_name
    }


def _candidate_fraction(candidate: np.ndarray, negative: np.ndarray) -> float | None:
    candidate_count = int(np.count_nonzero(candidate))
    negative_count = int(np.count_nonzero(negative))
    total = candidate_count + negative_count
    return None if total == 0 else candidate_count / total


def _mask(arrays: dict[str, np.ndarray], key: str, size: int) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"label-quality subset NPZ missing required mask: {key}")
    mask = np.asarray(arrays[key], dtype=bool).reshape(-1)
    if mask.size != size:
        raise ValueError(f"{key} length {mask.size} does not match sample size {size}.")
    return mask


def _validate_inputs(
    features: np.ndarray,
    feature_names: np.ndarray,
    label: np.ndarray,
    subset_arrays: dict[str, np.ndarray],
) -> None:
    if features.ndim != 2:
        raise ValueError("transformed_features must have shape [sample, feature].")
    if features.shape[0] != label.size:
        raise ValueError("transformed_features sample count must match label length.")
    if feature_names.size != features.shape[1]:
        raise ValueError("transformed_feature_names length must match feature count.")
    for guardrail in ("no_final_labels", "no_stc", "no_apes"):
        if guardrail in subset_arrays and not bool(np.asarray(subset_arrays[guardrail])):
            raise ValueError(f"label-quality subset NPZ must keep {guardrail}=true.")


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
