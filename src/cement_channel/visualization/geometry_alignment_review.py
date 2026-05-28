from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.alignment.xsi_geometry import ReceiverGeometry
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

GEOMETRY_ALIGNMENT_REVIEW_VERSION = "geometry_alignment_review_v001"

MODE_COMPARISON_FIELDS = [
    "mode",
    "sign",
    "status",
    "positive_count",
    "negative_count",
    "positive_fraction",
    "best_feature_name",
    "best_abs_effect_size",
    "balanced_accuracy",
    "permutation_balanced_accuracy",
    "real_minus_permutation_margin",
    "predicted_positive_rate",
    "folds_above_permutation",
    "depends_on_5700_band",
    "sample_count_collapse",
]

INTERVAL_COMPARISON_FIELDS = [
    "review_id",
    "interval_type",
    "start_depth",
    "end_depth",
    "original_cast_evidence_category",
    "mode",
    "sign",
    "geometry_candidate_fraction_mean",
    "geometry_candidate_fraction_max",
    "geometry_has_channel_fraction",
    "geometry_max_severity",
    "geometry_max_confidence",
    "geometry_max_relative_drop",
    "geometry_depth_label_confidence_mean",
    "geometry_evidence_category",
    "evidence_category_changed",
    "review_decision_should_be_revisited",
    "recommended_human_question",
]


@dataclass(frozen=True)
class GeometryAlignmentReviewReport:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    interval_count: int
    mode_sign_count: int
    evidence_category_changed_interval_count: int
    revisit_interval_count: int
    figure_count: int
    figures: dict[str, str]
    geometry_mode_comparison_csv: str
    geometry_mode_comparison_json: str
    interval_geometry_comparison_csv: str
    interval_geometry_comparison_json: str
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


def generate_geometry_alignment_review(
    *,
    geometry_depth_labels_npz: Path | str,
    geometry_alignment_audit_json: Path | str,
    review_intervals_json: Path | str,
    depth_level_features_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool = False,
    geometry_config_path: Path | str = "configs/xsi_geometry.example.yaml",
) -> GeometryAlignmentReviewReport:
    geometry_arrays = _load_npz(geometry_depth_labels_npz)
    audit_report = _read_json(Path(geometry_alignment_audit_json))
    review_data = _read_json(Path(review_intervals_json))
    feature_arrays = _load_npz(depth_level_features_npz)
    intervals = [row for row in _as_list(review_data.get("intervals")) if isinstance(row, dict)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(geometry_arrays, feature_arrays, audit_report, errors)
    original_cast_rows = _read_original_cast_summary(Path(review_intervals_json), warnings)
    mode_rows = build_geometry_mode_comparison_rows(audit_report)
    interval_rows, interval_json = build_interval_geometry_comparison(
        intervals=intervals,
        geometry_arrays=geometry_arrays,
        original_cast_rows=original_cast_rows,
    )

    mode_csv = output / "geometry_mode_comparison.csv"
    mode_json = output / "geometry_mode_comparison.json"
    interval_csv = output / "interval_geometry_comparison.csv"
    interval_json_path = output / "interval_geometry_comparison.json"
    review_summary = output / "review_summary.md"
    _ensure_can_write(mode_csv, overwrite=overwrite)
    _ensure_can_write(mode_json, overwrite=overwrite)
    _ensure_can_write(interval_csv, overwrite=overwrite)
    _ensure_can_write(interval_json_path, overwrite=overwrite)
    _ensure_can_write(review_summary, overwrite=overwrite)
    _write_csv(mode_rows, MODE_COMPARISON_FIELDS, mode_csv)
    mode_json.write_text(
        json.dumps(mode_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(interval_rows, INTERVAL_COMPARISON_FIELDS, interval_csv)
    interval_json_path.write_text(
        json.dumps(interval_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    geometry = ReceiverGeometry.from_yaml(geometry_config_path)
    figures = write_geometry_review_figures(
        output_dir=output,
        geometry=geometry,
        mode_rows=mode_rows,
        interval_rows=interval_rows,
        geometry_arrays=geometry_arrays,
        overwrite=overwrite,
    )
    changed_interval_count = _changed_interval_count(interval_json)
    revisit_count = _revisit_interval_count(interval_json)
    report = GeometryAlignmentReviewReport(
        review_version=GEOMETRY_ALIGNMENT_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "geometry_depth_labels_npz": str(geometry_depth_labels_npz),
            "geometry_alignment_audit_json": str(geometry_alignment_audit_json),
            "review_intervals_json": str(review_intervals_json),
            "depth_level_features_npz": str(depth_level_features_npz),
            "geometry_config_path": str(geometry_config_path),
        },
        output_dir=str(output),
        interval_count=len(intervals),
        mode_sign_count=len(mode_rows),
        evidence_category_changed_interval_count=changed_interval_count,
        revisit_interval_count=revisit_count,
        figure_count=len(figures),
        figures=figures,
        geometry_mode_comparison_csv=str(mode_csv),
        geometry_mode_comparison_json=str(mode_json),
        interval_geometry_comparison_csv=str(interval_csv),
        interval_geometry_comparison_json=str(interval_json_path),
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "original manual review pack overwrite",
            "new model training",
            "model refit",
            "production inference",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    review_summary.write_text(
        format_geometry_review_summary(report, interval_json),
        encoding="utf-8",
    )
    summary_json = output / "geometry_alignment_review_summary_v001.json"
    _ensure_can_write(summary_json, overwrite=overwrite)
    summary_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_geometry_mode_comparison_rows(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _as_list(audit_report.get("combo_summaries")):
        if not isinstance(row, dict):
            continue
        rows.append({field: row.get(field) for field in MODE_COMPARISON_FIELDS})
    return rows


def build_interval_geometry_comparison(
    *,
    intervals: list[dict[str, Any]],
    geometry_arrays: dict[str, np.ndarray],
    original_cast_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    depth = np.asarray(geometry_arrays["depth"], dtype=np.float32).reshape(-1)
    modes = geometry_arrays["mode"].astype(str)
    signs = geometry_arrays["sign"].astype(int)
    flat_rows: list[dict[str, Any]] = []
    interval_json: list[dict[str, Any]] = []
    for interval in intervals:
        review_id = str(interval.get("review_id", ""))
        start = float(interval.get("start_depth", 0.0))
        end = float(interval.get("end_depth", start))
        depth_mask = (depth >= min(start, end)) & (depth <= max(start, end))
        original_category = _original_evidence_category(interval, original_cast_rows)
        evidence_by_mode: dict[str, dict[str, Any]] = {}
        changed = False
        revisit = False
        for combo_index, mode in enumerate(modes):
            sign = int(signs[combo_index])
            evidence = _geometry_interval_evidence(
                geometry_arrays=geometry_arrays,
                combo_index=combo_index,
                depth_mask=depth_mask,
            )
            category, note = _evidence_category(evidence)
            key = f"{mode}__sign_{sign:+d}"
            category_changed = bool(original_category and category != original_category)
            should_revisit = _should_revisit(
                interval_type=str(interval.get("interval_type", "")),
                original_category=original_category,
                geometry_category=category,
                category_changed=category_changed,
            )
            changed |= category_changed
            revisit |= should_revisit
            question = _recommended_question(
                interval=interval,
                mode=mode,
                sign=sign,
                original_category=original_category,
                geometry_category=category,
                should_revisit=should_revisit,
            )
            evidence_by_mode[key] = {
                **evidence,
                "geometry_evidence_category": category,
                "geometry_evidence_note": note,
                "evidence_category_changed": category_changed,
                "review_decision_should_be_revisited": should_revisit,
                "recommended_human_question": question,
            }
            flat_rows.append(
                {
                    "review_id": review_id,
                    "interval_type": str(interval.get("interval_type", "")),
                    "start_depth": start,
                    "end_depth": end,
                    "original_cast_evidence_category": original_category,
                    "mode": mode,
                    "sign": sign,
                    **evidence,
                    "geometry_evidence_category": category,
                    "evidence_category_changed": category_changed,
                    "review_decision_should_be_revisited": should_revisit,
                    "recommended_human_question": question,
                }
            )
        interval_json.append(
            {
                "review_id": review_id,
                "original_interval_type": str(interval.get("interval_type", "")),
                "original_depth_range": {"start_depth": start, "end_depth": end},
                "original_cast_evidence_category": original_category,
                "geometry_aware_cast_evidence_by_mode_sign": evidence_by_mode,
                "evidence_category_changed": changed,
                "review_decision_should_be_revisited": revisit,
                "recommended_human_question": _interval_level_question(
                    interval,
                    changed=changed,
                    revisit=revisit,
                ),
            }
        )
    return flat_rows, interval_json


def write_geometry_review_figures(
    *,
    output_dir: Path,
    geometry: ReceiverGeometry,
    mode_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    geometry_arrays: dict[str, np.ndarray],
    overwrite: bool,
) -> dict[str, str]:
    figures = {
        "alignment_mode_comparison": output_dir / "alignment_mode_comparison_overview.png",
        "sign_comparison": output_dir / "sign_plus_minus_comparison.png",
        "depth_schematic": output_dir / "source_receiver_depth_schematic.png",
        "selected_interval_comparison": (
            output_dir / "selected_dlr_interval_geometry_comparison.png"
        ),
        "interval_examples": output_dir / "source_receiver_interval_aggregation_examples.png",
        "band_5700_sensitivity": output_dir / "geometry_5700_band_sensitivity.png",
        "manual_review_delta_table": output_dir / "manual_review_delta_table.png",
    }
    _save_mode_overview(mode_rows, figures["alignment_mode_comparison"], overwrite=overwrite)
    _save_sign_comparison(mode_rows, figures["sign_comparison"], overwrite=overwrite)
    _save_depth_schematic(geometry, figures["depth_schematic"], overwrite=overwrite)
    _save_selected_interval_comparison(
        interval_rows,
        figures["selected_interval_comparison"],
        overwrite=overwrite,
    )
    _save_interval_examples(interval_rows, figures["interval_examples"], overwrite=overwrite)
    _save_5700_sensitivity(
        geometry_arrays,
        figures["band_5700_sensitivity"],
        overwrite=overwrite,
    )
    _save_delta_table(interval_rows, figures["manual_review_delta_table"], overwrite=overwrite)
    return {key: str(path) for key, path in figures.items()}


def format_geometry_review_summary(
    report: GeometryAlignmentReviewReport,
    interval_json: list[dict[str, Any]],
) -> str:
    revisit = [
        row["review_id"]
        for row in interval_json
        if bool(row.get("review_decision_should_be_revisited"))
    ]
    lines = [
        "# Geometry-Aware Manual Review Supplement",
        "",
        "This MVP-4B-G supplement is review-only. It compares CAST weak-label "
        "candidate evidence under explicit XSI source-to-receiver geometry. It "
        "does not replace human review and does not create final labels.",
        "",
        f"- interval_count: {report.interval_count}",
        "- evidence_category_changed_interval_count: "
        f"{report.evidence_category_changed_interval_count}",
        f"- revisit_interval_count: {report.revisit_interval_count}",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Intervals Recommended For Geometry Re-Review",
        "",
    ]
    lines.extend(_message_lines(revisit))
    lines.extend(["", "## Outputs", ""])
    lines.extend(
        [
            f"- geometry_mode_comparison_csv: `{report.geometry_mode_comparison_csv}`",
            f"- interval_geometry_comparison_csv: `{report.interval_geometry_comparison_csv}`",
            f"- figure_count: {report.figure_count}",
        ]
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _geometry_interval_evidence(
    *,
    geometry_arrays: dict[str, np.ndarray],
    combo_index: int,
    depth_mask: np.ndarray,
) -> dict[str, Any]:
    if not np.any(depth_mask):
        return {
            "geometry_candidate_fraction_mean": None,
            "geometry_candidate_fraction_max": None,
            "geometry_has_channel_fraction": None,
            "geometry_max_severity": None,
            "geometry_max_confidence": None,
            "geometry_max_relative_drop": None,
            "geometry_depth_label_confidence_mean": None,
        }
    candidate_fraction = geometry_arrays["candidate_fraction"][combo_index][depth_mask]
    has_channel = geometry_arrays["has_channel_any"][combo_index][depth_mask]
    severity = geometry_arrays["max_severity"][combo_index][depth_mask]
    confidence = geometry_arrays["max_confidence"][combo_index][depth_mask]
    relative_drop = geometry_arrays["max_relative_drop"][combo_index][depth_mask]
    label_confidence = geometry_arrays["depth_label_confidence"][combo_index][depth_mask]
    return {
        "geometry_candidate_fraction_mean": _nanmean(candidate_fraction),
        "geometry_candidate_fraction_max": _nanmax(candidate_fraction),
        "geometry_has_channel_fraction": _fraction(has_channel),
        "geometry_max_severity": _nanmax(severity),
        "geometry_max_confidence": _nanmax(confidence),
        "geometry_max_relative_drop": _nanmax(relative_drop),
        "geometry_depth_label_confidence_mean": _nanmean(label_confidence),
    }


def _original_evidence_category(
    interval: dict[str, Any],
    original_cast_rows: dict[str, dict[str, Any]],
) -> str:
    review_id = str(interval.get("review_id", ""))
    if review_id in original_cast_rows:
        category = original_cast_rows[review_id].get("evidence_category")
        if category:
            return str(category)
    summary = _as_dict(interval.get("cast_label_summary"))
    candidate = _as_dict(summary.get("weak_label_candidate_summary"))
    zc = _as_dict(summary.get("cast_zc_summary"))
    values = {
        "presence_fraction": _as_float(candidate.get("presence_plus_fraction")),
        "severity_max": _as_float(candidate.get("severity_plus_max")),
        "candidate_conf": _as_float(candidate.get("label_confidence_plus_mean")),
        "relative_drop": _as_float(candidate.get("relative_drop_plus_max")),
        "zc_p05": _as_float(zc.get("zc_p05")),
    }
    category, _note = _evidence_category_from_original(values)
    return category


def _evidence_category(evidence: dict[str, Any]) -> tuple[str, str]:
    values = {
        "presence_fraction": _as_float(evidence.get("geometry_has_channel_fraction")),
        "severity_max": _as_float(evidence.get("geometry_max_severity")),
        "candidate_conf": _as_float(evidence.get("geometry_depth_label_confidence_mean")),
        "relative_drop": _as_float(evidence.get("geometry_max_relative_drop")),
        "zc_p05": None,
    }
    return _evidence_category_from_original(values)


def _evidence_category_from_original(values: dict[str, float | None]) -> tuple[str, str]:
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
            "low confidence with local CAST evidence; review cautiously",
        )
    if presence <= 0.01 and (severity is None or severity <= 0.0) and relative_drop < 0.15:
        if not near_low_zc:
            return ("clear_negative_evidence", "low presence, low drop, and no low-Zc support")
    if (severity is not None and severity >= 3.0) and (relative_drop >= 0.5 or low_zc):
        return (
            "strong_local_positive_evidence",
            "severe local candidate with relative-drop or low-Zc support",
        )
    if (severity is not None and severity >= 2.0) or relative_drop >= 0.4 or low_zc:
        return (
            "moderate_positive_evidence",
            "moderate CAST severity, relative-drop, or low-Zc support",
        )
    if presence > 0.0 or relative_drop >= 0.2 or near_low_zc:
        return (
            "weak_local_positive_evidence",
            "some local evidence, but interval-level support is weak",
        )
    return ("clear_negative_evidence", "no strong CAST review evidence")


def _should_revisit(
    *,
    interval_type: str,
    original_category: str,
    geometry_category: str,
    category_changed: bool,
) -> bool:
    if not category_changed:
        return False
    positive_geometry = "positive" in geometry_category
    if interval_type in {"clear_negative_like", "false_negative_like"} and positive_geometry:
        return True
    if original_category == "clear_negative_evidence" and geometry_category != original_category:
        return True
    if interval_type == "false_positive_like":
        return True
    return geometry_category == "uncertain_mixed_evidence"


def _recommended_question(
    *,
    interval: dict[str, Any],
    mode: str,
    sign: int,
    original_category: str,
    geometry_category: str,
    should_revisit: bool,
) -> str:
    if should_revisit:
        return (
            f"Does {mode} sign={sign:+d} geometry explain why "
            f"{interval.get('review_id')} changes from {original_category} "
            f"to {geometry_category}?"
        )
    return (
        f"Does {mode} sign={sign:+d} geometry preserve the original CAST evidence "
        f"interpretation for {interval.get('review_id')}?"
    )


def _interval_level_question(
    interval: dict[str, Any],
    *,
    changed: bool,
    revisit: bool,
) -> str:
    if revisit:
        return (
            f"Re-review {interval.get('review_id')} with geometry-aware CAST evidence "
            "before using it as a depth-level review reference."
        )
    if changed:
        return (
            f"Check whether {interval.get('review_id')} category changes are physical "
            "scale effects or label noise."
        )
    return (
        f"Confirm whether {interval.get('review_id')} remains physically plausible "
        "under geometry-aware aggregation."
    )


def _save_mode_overview(rows: list[dict[str, Any]], output_path: Path, *, overwrite: bool) -> None:
    plt = require_pyplot()
    labels = [f"{row['mode']}\n{int(row['sign']):+d}" for row in rows]
    margins = [_as_float(row.get("real_minus_permutation_margin")) for row in rows]
    values = [0.0 if value is None else value for value in margins]
    colors = ["tab:gray" if value is None else "tab:blue" for value in margins]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0.03, color="tab:red", linestyle="--", linewidth=1.0, label="review margin")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_ylabel("real - permutation balanced accuracy")
    ax.set_title("Geometry alignment mode comparison - weak-label candidate review")
    ax.legend(loc="best")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_sign_comparison(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    modes = sorted({str(row["mode"]) for row in rows})
    plus = []
    minus = []
    for mode in modes:
        plus.append(_row_value(rows, mode, 1, "real_minus_permutation_margin"))
        minus.append(_row_value(rows, mode, -1, "real_minus_permutation_margin"))
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(x - 0.18, plus, width=0.36, label="sign +1")
    ax.bar(x + 0.18, minus, width=0.36, label="sign -1")
    ax.set_xticks(x, labels=modes, rotation=25, ha="right")
    ax.set_ylabel("margin")
    ax.set_title("Depth-axis sign comparison - audit only")
    ax.legend()
    save_figure(fig, output_path, overwrite=overwrite)


def _save_depth_schematic(
    geometry: ReceiverGeometry,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    offsets = geometry.receiver_offsets_ft
    source = geometry.source_offset_ft
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.scatter(offsets, np.zeros_like(offsets), color="tab:blue", label="receivers")
    ax.scatter([source], [0.0], color="tab:red", label="source")
    ax.scatter([(source + offsets[6]) * 0.5], [0.0], color="tab:green", label="R7 midpoint")
    for label, offset in zip(geometry.receiver_labels, offsets, strict=False):
        if label in {"R1", "R7", "R13"}:
            ax.text(float(offset), 0.05, label, ha="center", fontsize=9)
    ax.text(source, -0.08, "source", ha="center", fontsize=9, color="tab:red")
    ax.set_xlabel("Offset relative to R7 (ft; sign audited separately)")
    ax.set_yticks([])
    ax.set_title("XSI source / R7 / receiver / midpoint depth schematic")
    ax.legend(loc="upper right")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_selected_interval_comparison(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    selected_ids = _top_interval_ids(rows, limit=12)
    mode_keys = _mode_keys(rows)
    matrix = np.full((len(selected_ids), len(mode_keys)), np.nan, dtype=np.float32)
    lookup = {
        (str(row["review_id"]), f"{row['mode']} {int(row['sign']):+d}"): _as_float(
            row.get("geometry_candidate_fraction_max")
        )
        for row in rows
    }
    for i, review_id in enumerate(selected_ids):
        for j, key in enumerate(mode_keys):
            value = lookup.get((review_id, key))
            if value is not None:
                matrix[i, j] = value
    fig, ax = plt.subplots(figsize=(12, max(4, len(selected_ids) * 0.35)), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(np.arange(len(mode_keys)), labels=mode_keys, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(selected_ids)), labels=selected_ids)
    ax.set_title("Selected DLR interval geometry-aware CAST candidate fraction")
    fig.colorbar(image, ax=ax, label="candidate fraction max")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_interval_examples(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    interval_rows = [
        row for row in rows if str(row.get("mode")) == "source_receiver_interval"
    ]
    selected = interval_rows[:30]
    labels = [str(row["review_id"]) for row in selected]
    values = [
        _as_float(row.get("geometry_candidate_fraction_max")) or 0.0 for row in selected
    ]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color="tab:purple")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=90)
    ax.set_ylabel("candidate fraction max")
    ax.set_title("Source-receiver interval aggregation examples - review only")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_5700_sensitivity(
    geometry_arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    depth = np.asarray(geometry_arrays["depth"], dtype=np.float32)
    review = (depth >= 5680.0) & (depth <= 5720.0)
    labels = [
        f"{mode}\n{int(sign):+d}"
        for mode, sign in zip(
            geometry_arrays["mode"].astype(str),
            geometry_arrays["sign"].astype(int),
            strict=False,
        )
    ]
    values = [
        int(np.count_nonzero(geometry_arrays["high_confidence_positive_mask"][i] & review))
        for i in range(len(labels))
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color="tab:orange")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_ylabel("high-confidence positives in 5680-5720 ft")
    ax.set_title("5700 ft band sensitivity under geometry-aware labels")
    save_figure(fig, output_path, overwrite=overwrite)


def _save_delta_table(rows: list[dict[str, Any]], output_path: Path, *, overwrite: bool) -> None:
    plt = require_pyplot()
    changed = [
        row
        for row in rows
        if bool(row.get("review_decision_should_be_revisited"))
    ][:14]
    fig, ax = plt.subplots(figsize=(12, max(3, len(changed) * 0.35)), constrained_layout=True)
    ax.axis("off")
    if not changed:
        ax.text(0.02, 0.8, "No intervals flagged for geometry re-review.", fontsize=11)
    else:
        cell_text = [
            [
                row["review_id"],
                row["interval_type"],
                row["original_cast_evidence_category"],
                row["geometry_evidence_category"],
                f"{row['mode']} {int(row['sign']):+d}",
            ]
            for row in changed
        ]
        table = ax.table(
            cellText=cell_text,
            colLabels=["DLR", "type", "original", "geometry", "mode/sign"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
    ax.set_title("Manual review delta table - review-only category changes")
    save_figure(fig, output_path, overwrite=overwrite)


def _read_original_cast_summary(
    review_intervals_json: Path,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    summary_path = review_intervals_json.parent / "interval_cast_evidence_summary_table.json"
    if not summary_path.exists():
        warnings.append("Original interval_cast_evidence_summary_table.json not found.")
        return {}
    rows = _read_json(summary_path)
    if not isinstance(rows, list):
        warnings.append("Original CAST evidence summary table is not a list.")
        return {}
    return {str(row.get("interval_id")): row for row in rows if isinstance(row, dict)}


def _validate_guardrails(
    geometry_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    audit_report: dict[str, Any],
    errors: list[str],
) -> None:
    if not bool(np.asarray(geometry_arrays.get("no_final_labels", False))):
        errors.append("Geometry-aware labels must preserve no_final_labels=true.")
    if not bool(np.asarray(feature_arrays.get("no_final_labels", False))):
        errors.append("Depth-level features must preserve no_final_labels=true.")
    if audit_report.get("no_final_labels") is not True:
        errors.append("Geometry alignment audit must preserve no_final_labels=true.")


def _write_csv(rows: list[dict[str, Any]], fieldnames: list[str], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _top_interval_ids(rows: list[dict[str, Any]], *, limit: int) -> list[str]:
    changed = [
        str(row["review_id"])
        for row in rows
        if bool(row.get("review_decision_should_be_revisited"))
    ]
    ordered = []
    for review_id in changed + [str(row["review_id"]) for row in rows]:
        if review_id not in ordered:
            ordered.append(review_id)
        if len(ordered) >= limit:
            break
    return ordered


def _mode_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = []
    for row in rows:
        key = f"{row['mode']} {int(row['sign']):+d}"
        if key not in keys:
            keys.append(key)
    return keys


def _row_value(rows: list[dict[str, Any]], mode: str, sign: int, key: str) -> float:
    for row in rows:
        if row.get("mode") == mode and int(row.get("sign", 0)) == sign:
            return _as_float(row.get(key)) or 0.0
    return 0.0


def _changed_interval_count(interval_json: list[dict[str, Any]]) -> int:
    return int(sum(bool(row.get("evidence_category_changed")) for row in interval_json))


def _revisit_interval_count(interval_json: list[dict[str, Any]]) -> int:
    return int(sum(bool(row.get("review_decision_should_be_revisited")) for row in interval_json))


def _zero_one(value: Any) -> float:
    number = _as_float(value)
    if number is None:
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _nanmean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.mean(finite))


def _nanmax(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.max(finite))


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
