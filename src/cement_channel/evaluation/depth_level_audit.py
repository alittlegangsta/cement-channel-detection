from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.labels.depth_level_schema import (
    DEPTH_LEVEL_SEPARATION_AUDIT_VERSION,
    DepthLevelLabelConfig,
    load_depth_level_label_config,
)


@dataclass(frozen=True)
class DepthLevelSeparationAuditReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    output_csv: str
    output_figure_dir: str
    depth_count: int
    feature_count: int
    comparison_summaries: dict[str, dict[str, int | float | str | None]]
    feature_group_summaries: list[dict[str, Any]]
    top_feature_rows: list[dict[str, Any]]
    depth_vs_side_comparison: dict[str, float | bool | None]
    review_band_sensitivity: dict[str, float | bool | None]
    disagreement_sensitivity: dict[str, int | float | None]
    depth_level_separation_enhanced: bool
    depth_level_baseline_sanity_candidate: bool
    side_level_target_likely_too_fine: bool
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


def audit_depth_level_separation_from_paths(
    *,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    depth_level_config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    output_figure_dir: Path | str,
    side_level_audit_report_json: Path | str | None = None,
    overwrite: bool = False,
) -> DepthLevelSeparationAuditReport:
    label_arrays = _load_npz(depth_level_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    config = load_depth_level_label_config(depth_level_config_path)
    side_report = _read_optional_json(side_level_audit_report_json)
    report, rows = audit_depth_level_separation(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        config=config,
        side_level_audit_report=side_report,
        inputs={
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "depth_level_config_path": str(depth_level_config_path),
            "side_level_audit_report_json": str(side_level_audit_report_json or ""),
        },
        output_csv=Path(output_csv),
        output_figure_dir=Path(output_figure_dir),
    )
    write_depth_level_audit_outputs(
        report,
        rows,
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        output_csv=Path(output_csv),
        output_figure_dir=Path(output_figure_dir),
        overwrite=overwrite,
    )
    return report


def audit_depth_level_separation(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    config: DepthLevelLabelConfig,
    side_level_audit_report: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    output_csv: Path | None = None,
    output_figure_dir: Path | None = None,
) -> tuple[DepthLevelSeparationAuditReport, list[dict[str, Any]]]:
    labels = _required_label_arrays(label_arrays)
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    _validate_inputs(labels, features, feature_names, label_arrays, feature_arrays)
    comparisons = comparison_masks(labels, config)
    feature_groups = feature_group_indices(feature_names)
    rows = effect_size_rows(features, feature_names, comparisons, feature_groups)
    group_summaries = feature_group_summaries(rows)
    comparison_summaries = {
        name: comparison_summary(name, pair, rows)
        for name, pair in comparisons.items()
    }
    top_rows = sorted(
        rows,
        key=lambda row: abs(float(row["standardized_difference"] or 0.0)),
        reverse=True,
    )[:40]
    side_best = side_level_best_effect(side_level_audit_report)
    depth_best = best_effect_for_comparisons(
        rows,
        {
            "depth_has_channel_vs_no_channel",
            "strong_positive_vs_clear_negative",
            "high_confidence_depth_only",
            "exclude_5700_band",
        },
    )
    improvement_delta = (
        None if side_best is None or depth_best is None else depth_best - side_best
    )
    enhanced = bool(
        depth_best is not None
        and depth_best >= config.gate.sanity_effect_size_threshold
        and improvement_delta is not None
        and improvement_delta >= config.gate.depth_level_improvement_effect_size_delta
    )
    review_sensitivity = review_band_sensitivity(group_summaries)
    disagreement = disagreement_sensitivity_summary(labels)
    errors = _stop_condition_errors(labels, config)
    warnings: list[str] = []
    if not enhanced:
        warnings.append(
            "depth-level separation does not clearly exceed the prior side-level audit."
        )
    report = DepthLevelSeparationAuditReport(
        report_version=DEPTH_LEVEL_SEPARATION_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_csv=str(output_csv) if output_csv else "",
        output_figure_dir=str(output_figure_dir) if output_figure_dir else "",
        depth_count=int(features.shape[0]),
        feature_count=int(features.shape[1]),
        comparison_summaries=comparison_summaries,
        feature_group_summaries=group_summaries,
        top_feature_rows=top_rows,
        depth_vs_side_comparison={
            "side_level_best_abs_effect_size": side_best,
            "depth_level_best_abs_effect_size": depth_best,
            "depth_minus_side_delta": improvement_delta,
            "required_delta": config.gate.depth_level_improvement_effect_size_delta,
            "sanity_effect_size_threshold": config.gate.sanity_effect_size_threshold,
            "depth_level_separation_enhanced": enhanced,
        },
        review_band_sensitivity=review_sensitivity,
        disagreement_sensitivity=disagreement,
        depth_level_separation_enhanced=enhanced,
        depth_level_baseline_sanity_candidate=enhanced,
        side_level_target_likely_too_fine=enhanced,
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
            "production inference",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, rows


def comparison_masks(
    labels: dict[str, np.ndarray],
    config: DepthLevelLabelConfig,
) -> dict[str, dict[str, np.ndarray]]:
    has_channel = labels["depth_has_channel_any"]
    strong = labels["depth_strong_positive_mask"]
    clear = labels["depth_clear_negative_mask"]
    review = labels["depth_review_band_mask"]
    confidence = labels["depth_label_confidence"]
    orientation = labels["depth_orientation_confidence"]
    disagreement = labels["depth_plus_minus_disagreement_fraction"]
    high_conf = (
        (confidence >= config.quality_policy.strong_positive.min_label_confidence)
        & (orientation >= config.quality_policy.strong_positive.min_orientation_confidence)
        & (
            disagreement
            <= config.quality_policy.strong_positive.max_plus_minus_disagreement_fraction
        )
    )
    low_disagreement = (
        disagreement
        <= config.quality_policy.strong_positive.max_plus_minus_disagreement_fraction
    )
    high_disagreement = (
        disagreement
        > config.quality_policy.strong_positive.max_plus_minus_disagreement_fraction
    )
    return {
        "depth_has_channel_vs_no_channel": {
            "candidate": has_channel,
            "negative": ~has_channel,
        },
        "strong_positive_vs_clear_negative": {
            "candidate": strong,
            "negative": clear,
        },
        "high_confidence_depth_only": {
            "candidate": has_channel & high_conf,
            "negative": (~has_channel) & high_conf,
        },
        "exclude_5700_band": {
            "candidate": has_channel & ~review,
            "negative": (~has_channel) & ~review,
        },
        "include_5700_band_for_sensitivity": {
            "candidate": has_channel,
            "negative": ~has_channel,
        },
        "low_vs_high_plus_minus_disagreement": {
            "candidate": low_disagreement,
            "negative": high_disagreement,
        },
    }


def feature_group_indices(feature_names: np.ndarray) -> dict[str, np.ndarray]:
    names = feature_names.astype(str)
    high_side = np.asarray(["high_side" in name for name in names], dtype=bool)
    receiver = np.asarray(
        [
            ("receiver_" in name)
            or ("near_far" in name)
            or ("far_" in name)
            or ("near_" in name)
            for name in names
        ],
        dtype=bool,
    )
    late = np.asarray(["late_over_early" in name for name in names], dtype=bool)
    side = np.asarray(
        [
            (name.startswith("side_"))
            or name.startswith("max_side")
            or name.startswith("late_over_early")
            for name in names
        ],
        dtype=bool,
    )
    groups = {
        "side_depth_summary": np.flatnonzero(side & ~high_side),
        "receiver_depth_summary": np.flatnonzero(receiver & ~high_side),
        "late_over_early": np.flatnonzero(late),
        "high_side_audit": np.flatnonzero(high_side),
        "all_depth_features": np.arange(names.size),
    }
    return {name: indices.astype(np.int32) for name, indices in groups.items() if indices.size}


def effect_size_rows(
    features: np.ndarray,
    feature_names: np.ndarray,
    comparisons: dict[str, dict[str, np.ndarray]],
    feature_groups: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison_name, pair in comparisons.items():
        candidate = pair["candidate"]
        negative = pair["negative"]
        for group_name, indices in feature_groups.items():
            for feature_index in indices:
                rows.append(
                    _effect_size_row(
                        features[:, feature_index],
                        feature_name=str(feature_names[feature_index]),
                        feature_group=group_name,
                        comparison_name=comparison_name,
                        candidate=candidate,
                        negative=negative,
                    )
                )
    return rows


def feature_group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["comparison_name"]), str(row["feature_group"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (comparison_name, group_name), group_rows in sorted(grouped.items()):
        valid = [
            row
            for row in group_rows
            if row["standardized_difference"] is not None
            and np.isfinite(float(row["standardized_difference"]))
        ]
        if not valid:
            summaries.append(
                {
                    "comparison_name": comparison_name,
                    "feature_group": group_name,
                    "feature_count": len(group_rows),
                    "top_abs_standardized_difference": None,
                    "top_feature_name": None,
                    "top_standardized_difference": None,
                    "top_threshold_balanced_accuracy": None,
                    "top_threshold_margin_over_shift": None,
                }
            )
            continue
        sorted_rows = sorted(
            valid,
            key=lambda row: abs(float(row["standardized_difference"])),
            reverse=True,
        )
        top = sorted_rows[0]
        summaries.append(
            {
                "comparison_name": comparison_name,
                "feature_group": group_name,
                "feature_count": len(group_rows),
                "top_abs_standardized_difference": abs(
                    float(top["standardized_difference"])
                ),
                "top_feature_name": top["feature_name"],
                "top_standardized_difference": float(top["standardized_difference"]),
                "top_threshold_balanced_accuracy": top["threshold_balanced_accuracy"],
                "top_threshold_margin_over_shift": top["threshold_margin_over_depth_shift"],
            }
        )
    return summaries


def comparison_summary(
    comparison_name: str,
    pair: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> dict[str, int | float | str | None]:
    valid_rows = [
        row
        for row in rows
        if row["comparison_name"] == comparison_name
        and row["standardized_difference"] is not None
    ]
    if not valid_rows:
        return {
            "candidate_count": int(np.count_nonzero(pair["candidate"])),
            "negative_count": int(np.count_nonzero(pair["negative"])),
            "top_abs_standardized_difference": None,
            "top_feature_name": None,
            "top_threshold_balanced_accuracy": None,
            "top_threshold_margin_over_shift": None,
        }
    top = max(valid_rows, key=lambda row: abs(float(row["standardized_difference"])))
    return {
        "candidate_count": int(np.count_nonzero(pair["candidate"])),
        "negative_count": int(np.count_nonzero(pair["negative"])),
        "top_abs_standardized_difference": abs(float(top["standardized_difference"])),
        "top_feature_name": str(top["feature_name"]),
        "top_threshold_balanced_accuracy": top["threshold_balanced_accuracy"],
        "top_threshold_margin_over_shift": top["threshold_margin_over_depth_shift"],
    }


def review_band_sensitivity(
    summaries: list[dict[str, Any]],
) -> dict[str, float | bool | None]:
    without = _best_summary(summaries, "exclude_5700_band")
    with_band = _best_summary(summaries, "include_5700_band_for_sensitivity")
    if without is None or with_band is None:
        return {
            "with_review_band_best_abs_effect_size": with_band,
            "without_review_band_best_abs_effect_size": without,
            "abs_effect_size_delta": None,
            "review_band_flips_conclusion": False,
        }
    delta = without - with_band
    return {
        "with_review_band_best_abs_effect_size": with_band,
        "without_review_band_best_abs_effect_size": without,
        "abs_effect_size_delta": delta,
        "review_band_flips_conclusion": abs(delta) > 0.05,
    }


def disagreement_sensitivity_summary(
    labels: dict[str, np.ndarray],
) -> dict[str, int | float | None]:
    disagreement = labels["depth_plus_minus_disagreement_fraction"]
    low = disagreement <= 0.25
    high = disagreement > 0.25
    return {
        "low_disagreement_depth_count": int(np.count_nonzero(low)),
        "high_disagreement_depth_count": int(np.count_nonzero(high)),
        "low_disagreement_positive_fraction": _positive_fraction(
            labels["depth_has_channel_any"],
            low,
        ),
        "high_disagreement_positive_fraction": _positive_fraction(
            labels["depth_has_channel_any"],
            high,
        ),
    }


def side_level_best_effect(report: dict[str, Any] | None) -> float | None:
    if not report:
        return None
    enhancement = _as_dict(report.get("signal_enhancement"))
    for key in (
        "all_candidate_best_abs_effect_size",
        "quality_subset_best_abs_effect_size",
    ):
        value = _as_float(enhancement.get(key))
        if value is not None:
            return value
    summaries = report.get("feature_group_summaries")
    if isinstance(summaries, list):
        values = [
            _as_float(_as_dict(row).get("top_abs_standardized_difference"))
            for row in summaries
        ]
        values = [value for value in values if value is not None]
        if values:
            return float(max(values))
    return None


def best_effect_for_comparisons(
    rows: list[dict[str, Any]],
    comparison_names: set[str],
) -> float | None:
    values = [
        abs(float(row["standardized_difference"]))
        for row in rows
        if row["comparison_name"] in comparison_names
        and row["standardized_difference"] is not None
        and np.isfinite(float(row["standardized_difference"]))
    ]
    return None if not values else float(max(values))


def write_depth_level_audit_outputs(
    report: DepthLevelSeparationAuditReport,
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
    output_md.write_text(format_depth_level_audit_markdown(report), encoding="utf-8")
    write_depth_level_audit_csv(rows, output_csv)
    write_depth_level_audit_figures(report, rows, output_figure_dir)


def write_depth_level_audit_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "comparison_name",
        "feature_group",
        "feature_name",
        "candidate_count",
        "negative_count",
        "candidate_mean",
        "negative_mean",
        "candidate_median",
        "negative_median",
        "standardized_difference",
        "median_difference",
        "threshold_balanced_accuracy",
        "depth_shift_balanced_accuracy",
        "threshold_margin_over_depth_shift",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_depth_level_audit_figures(
    report: DepthLevelSeparationAuditReport,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    summaries = report.feature_group_summaries
    comparison_names = sorted({str(row["comparison_name"]) for row in summaries})
    group_names = sorted({str(row["feature_group"]) for row in summaries})
    matrix = np.full((len(comparison_names), len(group_names)), np.nan, dtype=np.float32)
    lookup = {
        (str(row["comparison_name"]), str(row["feature_group"])): row for row in summaries
    }
    for i, comparison_name in enumerate(comparison_names):
        for j, group_name in enumerate(group_names):
            value = lookup.get((comparison_name, group_name), {}).get(
                "top_abs_standardized_difference"
            )
            if value is not None:
                matrix[i, j] = float(value)
    fig, ax = plt.subplots(figsize=(max(8, len(group_names) * 1.4), 5.5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(group_names)), labels=group_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(comparison_names)), labels=comparison_names)
    ax.set_title("Depth-level top absolute effect size")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "depth_level_effect_size_heatmap.png", dpi=150)
    plt.close(fig)

    top_rows = rows[:]
    top_rows.sort(
        key=lambda row: abs(float(row["standardized_difference"] or 0.0)),
        reverse=True,
    )
    if top_rows:
        names = [str(row["feature_name"])[:42] for row in top_rows[:12]]
        values = [float(row["standardized_difference"] or 0.0) for row in top_rows[:12]]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(np.arange(len(values)), values)
        ax.set_yticks(np.arange(len(values)), labels=names)
        ax.invert_yaxis()
        ax.set_xlabel("standardized difference")
        ax.set_title("Top depth-level feature separations")
        fig.tight_layout()
        fig.savefig(output_dir / "depth_level_top_effects.png", dpi=150)
        plt.close(fig)


def format_depth_level_audit_markdown(report: DepthLevelSeparationAuditReport) -> str:
    comparison = report.depth_vs_side_comparison
    lines = [
        "# MVP-4B-R4 Depth-Level Separation Audit",
        "",
        "This is a weak-label target review and feature-separation sanity audit. It "
        "does not train a model, generate final labels, or authorize MVP-4C.",
        "",
        f"- depth_level_separation_enhanced: `{report.depth_level_separation_enhanced}`",
        "- depth_level_baseline_sanity_candidate: "
        f"`{report.depth_level_baseline_sanity_candidate}`",
        f"- side_level_target_likely_too_fine: `{report.side_level_target_likely_too_fine}`",
        f"- depth_level_best_abs_effect_size: {comparison['depth_level_best_abs_effect_size']}",
        f"- side_level_best_abs_effect_size: {comparison['side_level_best_abs_effect_size']}",
        f"- depth_minus_side_delta: {comparison['depth_minus_side_delta']}",
        "",
        "## Comparison Summaries",
        "",
    ]
    for name, summary in report.comparison_summaries.items():
        lines.append(
            "- "
            f"{name}: candidate={summary['candidate_count']}, "
            f"negative={summary['negative_count']}, "
            f"top_abs_effect={summary['top_abs_standardized_difference']}, "
            f"top_feature={summary['top_feature_name']}, "
            f"threshold_bal_acc={summary['top_threshold_balanced_accuracy']}"
        )
    lines.extend(["", "## Review Band Sensitivity", ""])
    lines.extend(_dict_lines(report.review_band_sensitivity))
    lines.extend(["", "## Disagreement Sensitivity", ""])
    lines.extend(_dict_lines(report.disagreement_sensitivity))
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
    comparison_name: str,
    candidate: np.ndarray,
    negative: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    pos = np.asarray(candidate, dtype=bool) & finite
    neg = np.asarray(negative, dtype=bool) & finite
    if not np.any(pos) or not np.any(neg):
        return _empty_row(
            comparison_name,
            feature_group,
            feature_name,
            candidate_count=int(np.count_nonzero(pos)),
            negative_count=int(np.count_nonzero(neg)),
        )
    pos_values = values[pos].astype(np.float64)
    neg_values = values[neg].astype(np.float64)
    pos_mean = float(np.mean(pos_values))
    neg_mean = float(np.mean(neg_values))
    pooled = float(np.sqrt(0.5 * (np.var(pos_values) + np.var(neg_values))))
    effect = None if pooled <= 0.0 else (pos_mean - neg_mean) / pooled
    pos_median = float(np.median(pos_values))
    neg_median = float(np.median(neg_values))
    threshold_ba, shifted_ba = _threshold_sanity(values, pos, neg)
    return {
        "comparison_name": comparison_name,
        "feature_group": feature_group,
        "feature_name": feature_name,
        "candidate_count": int(np.count_nonzero(pos)),
        "negative_count": int(np.count_nonzero(neg)),
        "candidate_mean": pos_mean,
        "negative_mean": neg_mean,
        "candidate_median": pos_median,
        "negative_median": neg_median,
        "standardized_difference": None if effect is None else float(effect),
        "median_difference": pos_median - neg_median,
        "threshold_balanced_accuracy": threshold_ba,
        "depth_shift_balanced_accuracy": shifted_ba,
        "threshold_margin_over_depth_shift": (
            None if threshold_ba is None or shifted_ba is None else threshold_ba - shifted_ba
        ),
    }


def _threshold_sanity(
    values: np.ndarray,
    candidate: np.ndarray,
    negative: np.ndarray,
) -> tuple[float | None, float | None]:
    pos_values = values[candidate]
    neg_values = values[negative]
    if pos_values.size == 0 or neg_values.size == 0:
        return None, None
    pos_median = float(np.median(pos_values))
    neg_median = float(np.median(neg_values))
    if pos_median == neg_median:
        return 0.5, 0.5
    threshold = 0.5 * (pos_median + neg_median)
    positive_high = pos_median > neg_median
    selected = candidate | negative
    y_true = candidate[selected]
    scores = values[selected]
    pred = scores >= threshold if positive_high else scores <= threshold
    shifted_scores = np.roll(scores, max(1, scores.size // 2))
    shifted_pred = (
        shifted_scores >= threshold if positive_high else shifted_scores <= threshold
    )
    return _balanced_accuracy(y_true, pred), _balanced_accuracy(y_true, shifted_pred)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    pos = y_true
    neg = ~y_true
    if not np.any(pos) or not np.any(neg):
        return None
    tpr = np.count_nonzero(y_pred & pos) / np.count_nonzero(pos)
    tnr = np.count_nonzero((~y_pred) & neg) / np.count_nonzero(neg)
    return float(0.5 * (tpr + tnr))


def _empty_row(
    comparison_name: str,
    feature_group: str,
    feature_name: str,
    *,
    candidate_count: int,
    negative_count: int,
) -> dict[str, Any]:
    return {
        "comparison_name": comparison_name,
        "feature_group": feature_group,
        "feature_name": feature_name,
        "candidate_count": candidate_count,
        "negative_count": negative_count,
        "candidate_mean": None,
        "negative_mean": None,
        "candidate_median": None,
        "negative_median": None,
        "standardized_difference": None,
        "median_difference": None,
        "threshold_balanced_accuracy": None,
        "depth_shift_balanced_accuracy": None,
        "threshold_margin_over_depth_shift": None,
    }


def _required_label_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = (
        "depth",
        "depth_has_channel_any",
        "depth_strong_positive_mask",
        "depth_clear_negative_mask",
        "depth_review_band_mask",
        "depth_label_confidence",
        "depth_orientation_confidence",
        "depth_plus_minus_disagreement_fraction",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError("depth-level label NPZ missing required field(s): " + ", ".join(missing))
    return {
        "depth": np.asarray(arrays["depth"], dtype=np.float32).reshape(-1),
        "depth_has_channel_any": np.asarray(arrays["depth_has_channel_any"], dtype=bool).reshape(
            -1
        ),
        "depth_strong_positive_mask": np.asarray(
            arrays["depth_strong_positive_mask"],
            dtype=bool,
        ).reshape(-1),
        "depth_clear_negative_mask": np.asarray(
            arrays["depth_clear_negative_mask"],
            dtype=bool,
        ).reshape(-1),
        "depth_review_band_mask": np.asarray(arrays["depth_review_band_mask"], dtype=bool).reshape(
            -1
        ),
        "depth_label_confidence": np.asarray(
            arrays["depth_label_confidence"],
            dtype=np.float32,
        ).reshape(-1),
        "depth_orientation_confidence": np.asarray(
            arrays["depth_orientation_confidence"],
            dtype=np.float32,
        ).reshape(-1),
        "depth_plus_minus_disagreement_fraction": np.asarray(
            arrays["depth_plus_minus_disagreement_fraction"],
            dtype=np.float32,
        ).reshape(-1),
    }


def _validate_inputs(
    labels: dict[str, np.ndarray],
    features: np.ndarray,
    feature_names: np.ndarray,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
) -> None:
    if features.ndim != 2:
        raise ValueError("depth_level_xsi_features must have shape [depth, feature].")
    if feature_names.size != features.shape[1]:
        raise ValueError("depth_level_xsi_feature_names length must match feature count.")
    depth_count = labels["depth"].size
    if features.shape[0] != depth_count:
        raise ValueError("feature depth count must match label depth count.")
    for key, values in labels.items():
        if values.size != depth_count:
            raise ValueError(f"{key} length {values.size} does not match depth count.")
    for guardrail in ("no_final_labels", "no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        if guardrail in label_arrays and not bool(np.asarray(label_arrays[guardrail]).reshape(())):
            raise ValueError(f"depth-level label NPZ must keep {guardrail}=true.")
        if guardrail in feature_arrays and not bool(
            np.asarray(feature_arrays[guardrail]).reshape(())
        ):
            raise ValueError(f"depth-level feature NPZ must keep {guardrail}=true.")


def _stop_condition_errors(
    labels: dict[str, np.ndarray],
    config: DepthLevelLabelConfig,
) -> list[str]:
    errors: list[str] = []
    strong_count = int(np.count_nonzero(labels["depth_strong_positive_mask"]))
    clear_count = int(np.count_nonzero(labels["depth_clear_negative_mask"]))
    if strong_count < config.gate.min_depth_positive_count:
        errors.append("depth-level strong-positive subset is empty or below minimum.")
    if clear_count < config.gate.min_depth_negative_count:
        errors.append("depth-level clear-negative subset is empty or below minimum.")
    has_channel = labels["depth_has_channel_any"]
    review = labels["depth_review_band_mask"]
    positive_count = int(np.count_nonzero(has_channel))
    review_positive = int(np.count_nonzero(has_channel & review))
    review_fraction = 0.0 if positive_count == 0 else review_positive / positive_count
    if review_fraction > config.gate.max_5700_band_positive_fraction:
        errors.append("depth-level positive subset is dominated by the ~5700 ft review band.")
    return errors


def _positive_fraction(label: np.ndarray, mask: np.ndarray) -> float | None:
    total = int(np.count_nonzero(mask))
    if total == 0:
        return None
    return float(np.count_nonzero(label & mask) / total)


def _best_summary(summaries: list[dict[str, Any]], comparison_name: str) -> float | None:
    values = [
        _as_float(row.get("top_abs_standardized_difference"))
        for row in summaries
        if row.get("comparison_name") == comparison_name
    ]
    values = [value for value in values if value is not None]
    return None if not values else float(max(values))


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _read_optional_json(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


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


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
