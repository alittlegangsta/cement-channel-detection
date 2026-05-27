from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.evaluation.depth_level_review_schema import (
    DEPTH_LEVEL_MANUAL_REVIEW_PACK_VERSION,
    DepthLevelManualReviewConfig,
    DepthLevelReviewIntervalConfig,
    DepthLevelReviewSelectionConfig,
    load_depth_level_manual_review_config,
)

REVIEW_INTERVAL_CSV_FIELDS = [
    "review_id",
    "start_depth",
    "end_depth",
    "interval_type",
    "selection_rule",
    "depth_count",
    "weak_label_summary",
    "prediction_score_summary",
    "confidence_summary",
    "top_feature_summary",
    "plus_minus_disagreement_summary",
    "5700_band_flag",
    "cast_label_summary",
    "xsi_feature_summary",
    "recommended_review_question",
]

OPTIONAL_CAST_WEAK_LABEL_FIELDS = {
    "cast_depth",
    "presence_plus",
    "severity_plus",
    "label_confidence_plus",
    "relative_drop_plus",
    "presence_minus_ablation",
}
OPTIONAL_CAST_INPUT_FIELDS = {
    "cast_depth",
    "cast_zc",
    "orientation_confidence",
    "low_inc_mask",
}


@dataclass(frozen=True)
class DepthLevelReviewPackReport:
    review_pack_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str | None
    target_variant: str
    label_status: str
    source_scenario_id: str | None
    source_gate_decision: str | None
    interval_count: int
    interval_type_counts: dict[str, int]
    selected_interval_ids: list[str]
    optional_inputs: dict[str, bool]
    best_feature_group: str | None
    best_confidence_threshold: float | None
    depends_on_5700_band: bool | None
    stable_over_permutation: bool | None
    no_model_training_claim: bool
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


def build_depth_level_review_pack_from_config(
    *,
    depth_level_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    refinement_report_json: Path | str,
    refinement_gate_report_json: Path | str,
    refinement_csv: Path | str,
    review_config_path: Path | str,
    cast_weak_label_candidates_npz: Path | str | None = None,
    cast_label_input_npz: Path | str | None = None,
    output_dir: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[DepthLevelReviewPackReport, list[dict[str, Any]]]:
    config = load_depth_level_manual_review_config(review_config_path)
    label_arrays = _load_npz(depth_level_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    refinement_report = _read_json(Path(refinement_report_json))
    gate_report = _read_json(Path(refinement_gate_report_json))
    prediction_rows = _read_prediction_rows(Path(refinement_csv))
    warnings: list[str] = []
    cast_candidates = _load_optional_npz(
        cast_weak_label_candidates_npz,
        OPTIONAL_CAST_WEAK_LABEL_FIELDS,
        warnings,
    )
    cast_label_input = _load_optional_npz(
        cast_label_input_npz,
        OPTIONAL_CAST_INPUT_FIELDS,
        warnings,
    )
    report, intervals = build_depth_level_review_pack(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        refinement_report=refinement_report,
        refinement_gate_report=gate_report,
        prediction_rows=prediction_rows,
        config=config,
        inputs={
            "depth_level_labels_npz": str(depth_level_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "refinement_report_json": str(refinement_report_json),
            "refinement_gate_report_json": str(refinement_gate_report_json),
            "refinement_csv": str(refinement_csv),
            "review_config_path": str(review_config_path),
            "cast_weak_label_candidates_npz": (
                ""
                if cast_weak_label_candidates_npz is None
                else str(cast_weak_label_candidates_npz)
            ),
            "cast_label_input_npz": (
                "" if cast_label_input_npz is None else str(cast_label_input_npz)
            ),
        },
        output_dir=None if output_dir is None else str(output_dir),
        cast_candidates=cast_candidates,
        cast_label_input=cast_label_input,
        preexisting_warnings=warnings,
    )
    if output_dir is not None:
        write_depth_level_review_pack_outputs(
            report,
            intervals,
            output_dir=Path(output_dir),
            overwrite=overwrite,
        )
    return report, intervals


def build_depth_level_review_pack(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    refinement_report: dict[str, Any],
    refinement_gate_report: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    config: DepthLevelManualReviewConfig,
    inputs: dict[str, str] | None = None,
    output_dir: str | None = None,
    cast_candidates: dict[str, np.ndarray] | None = None,
    cast_label_input: dict[str, np.ndarray] | None = None,
    preexisting_warnings: list[str] | None = None,
) -> tuple[DepthLevelReviewPackReport, list[dict[str, Any]]]:
    warnings = list(preexisting_warnings or [])
    errors: list[str] = []
    prepared = prepare_review_pack_inputs(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
    )
    _validate_review_guardrails(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        refinement_report=refinement_report,
        refinement_gate_report=refinement_gate_report,
        errors=errors,
        warnings=warnings,
    )
    best = _as_dict(refinement_report.get("best_result"))
    scenario_id = None if not best else str(best.get("scenario_id"))
    selected_prediction_rows = _rows_for_scenario(prediction_rows, scenario_id)
    if not selected_prediction_rows:
        warnings.append("No refinement CSV rows matched the best scenario; using all CSV rows.")
        selected_prediction_rows = prediction_rows
    prediction_records = _prediction_records(selected_prediction_rows, prepared["depth"])
    top_feature_rows = _top_feature_rows(refinement_report, scenario_id=scenario_id, limit=8)
    intervals = select_review_intervals(
        prepared=prepared,
        prediction_records=prediction_records,
        refinement_report=refinement_report,
        config=config,
        top_feature_rows=top_feature_rows,
        cast_candidates=cast_candidates or {},
        cast_label_input=cast_label_input or {},
        warnings=warnings,
    )
    interval_type_counts: dict[str, int] = {}
    for interval in intervals:
        interval_type = str(interval["interval_type"])
        interval_type_counts[interval_type] = interval_type_counts.get(interval_type, 0) + 1
    robustness = _as_dict(refinement_report.get("robustness_summary"))
    report = DepthLevelReviewPackReport(
        review_pack_version=DEPTH_LEVEL_MANUAL_REVIEW_PACK_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_dir=output_dir,
        target_variant=config.target_variant,
        label_status=config.label_status,
        source_scenario_id=scenario_id,
        source_gate_decision=_gate_decision(refinement_gate_report),
        interval_count=len(intervals),
        interval_type_counts=interval_type_counts,
        selected_interval_ids=[str(interval["review_id"]) for interval in intervals],
        optional_inputs={
            "cast_weak_label_candidates": bool(cast_candidates),
            "cast_label_input": bool(cast_label_input),
        },
        best_feature_group=None if not best else str(best.get("feature_group")),
        best_confidence_threshold=_as_float(best.get("confidence_threshold")),
        depends_on_5700_band=_as_optional_bool(robustness.get("depends_on_5700_band")),
        stable_over_permutation=_as_optional_bool(robustness.get("stable_over_permutation")),
        no_model_training_claim=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
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
    return report, intervals


def prepare_review_pack_inputs(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    required_labels = (
        "depth",
        "depth_has_channel_any",
        "depth_candidate_fraction",
        "depth_max_severity",
        "depth_label_confidence",
        "depth_orientation_confidence",
        "depth_plus_minus_disagreement_fraction",
        "depth_clear_negative_mask",
        "depth_review_band_mask",
    )
    required_features = ("depth", "depth_level_xsi_features", "depth_level_xsi_feature_names")
    missing_labels = [key for key in required_labels if key not in label_arrays]
    missing_features = [key for key in required_features if key not in feature_arrays]
    if missing_labels:
        raise KeyError("depth-level label NPZ missing field(s): " + ", ".join(missing_labels))
    if missing_features:
        raise KeyError(
            "depth-level feature NPZ missing field(s): " + ", ".join(missing_features)
        )
    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    if features.ndim != 2:
        raise ValueError("depth_level_xsi_features must have shape [depth, feature].")
    if depth.size != features.shape[0]:
        raise ValueError("depth-level label and feature arrays must have the same depth count.")
    if feature_names.size != features.shape[1]:
        raise ValueError("depth_level_xsi_feature_names length must match feature count.")
    return {
        "depth": depth,
        "features": np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        "feature_names": feature_names.astype(str).tolist(),
        "label_arrays": label_arrays,
        "feature_arrays": feature_arrays,
        "has_channel": np.asarray(label_arrays["depth_has_channel_any"], dtype=bool).reshape(-1),
        "clear_negative": np.asarray(
            label_arrays["depth_clear_negative_mask"], dtype=bool
        ).reshape(-1),
        "review_band": np.asarray(label_arrays["depth_review_band_mask"], dtype=bool).reshape(-1),
        "label_confidence": np.asarray(
            label_arrays["depth_label_confidence"], dtype=np.float32
        ).reshape(-1),
        "orientation_confidence": np.asarray(
            label_arrays["depth_orientation_confidence"], dtype=np.float32
        ).reshape(-1),
        "disagreement": np.asarray(
            label_arrays["depth_plus_minus_disagreement_fraction"], dtype=np.float32
        ).reshape(-1),
    }


def select_review_intervals(
    *,
    prepared: dict[str, Any],
    prediction_records: list[dict[str, Any]],
    refinement_report: dict[str, Any],
    config: DepthLevelManualReviewConfig,
    top_feature_rows: list[dict[str, Any]],
    cast_candidates: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
    warnings: list[str],
) -> list[dict[str, Any]]:
    records_by_index = _records_by_label_index(prediction_records)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, float, float]] = set()
    next_id = 1
    for selection_name, selection in config.selections.items():
        if not selection.enabled:
            continue
        indices = _candidate_indices_for_selection(
            selection_name,
            prepared=prepared,
            records_by_index=records_by_index,
            config=config,
        )
        clusters = _select_clusters(
            indices=indices,
            prepared=prepared,
            records_by_index=records_by_index,
            selection=selection,
            config=config,
        )
        for cluster in clusters[: selection.count]:
            interval = _build_interval_row(
                review_number=next_id,
                selection_rule=selection_name,
                interval_type=selection.interval_type,
                label_indices=cluster,
                prepared=prepared,
                records_by_index=records_by_index,
                refinement_report=refinement_report,
                top_feature_rows=top_feature_rows,
                cast_candidates=cast_candidates,
                cast_label_input=cast_label_input,
            )
            key = (
                str(interval["interval_type"]),
                round(float(interval["start_depth"]), 3),
                round(float(interval["end_depth"]), 3),
            )
            if key in selected_keys:
                continue
            selected.append(interval)
            selected_keys.add(key)
            next_id += 1
    for review_interval in config.review_intervals:
        interval = _fixed_review_interval_row(
            review_number=next_id,
            review_interval=review_interval,
            prepared=prepared,
            records_by_index=records_by_index,
            refinement_report=refinement_report,
            top_feature_rows=top_feature_rows,
            cast_candidates=cast_candidates,
            cast_label_input=cast_label_input,
        )
        if interval is None:
            warnings.append(f"No depth samples found for review interval: {review_interval.name}.")
            continue
        selected.append(interval)
        next_id += 1
    return selected


def write_depth_level_review_pack_outputs(
    report: DepthLevelReviewPackReport,
    intervals: list[dict[str, Any]],
    *,
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "review_intervals.csv"
    output_json = output_dir / "review_intervals.json"
    output_summary = output_dir / "review_summary.md"
    for path in (output_csv, output_json, output_summary):
        _ensure_can_write(path, overwrite=overwrite)
    write_review_interval_csv(intervals, output_csv)
    output_json.write_text(
        json.dumps(
            {"report": report.to_dict(), "intervals": intervals},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_summary.write_text(
        format_depth_level_review_summary(report, intervals),
        encoding="utf-8",
    )


def write_review_interval_csv(intervals: list[dict[str, Any]], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_INTERVAL_CSV_FIELDS)
        writer.writeheader()
        for interval in intervals:
            writer.writerow(
                {
                    key: (
                        json.dumps(interval.get(key), ensure_ascii=False, sort_keys=True)
                        if isinstance(interval.get(key), dict | list)
                        else interval.get(key)
                    )
                    for key in REVIEW_INTERVAL_CSV_FIELDS
                }
            )


def format_depth_level_review_summary(
    report: DepthLevelReviewPackReport,
    intervals: list[dict[str, Any]],
) -> str:
    lines = [
        "# Depth-Level Manual Review Pack",
        "",
        "Scope: manual review of depth-level CAST weak-label candidate anomaly "
        "agreement only. This pack does not train a model and does not create "
        "final labels.",
        "",
        f"- review_pack_version: `{report.review_pack_version}`",
        f"- source_gate_decision: `{report.source_gate_decision}`",
        f"- source_scenario_id: `{report.source_scenario_id}`",
        f"- best_feature_group: `{report.best_feature_group}`",
        f"- best_confidence_threshold: {report.best_confidence_threshold}",
        f"- depends_on_5700_band: `{report.depends_on_5700_band}`",
        f"- stable_over_permutation: `{report.stable_over_permutation}`",
        f"- interval_count: {report.interval_count}",
        "",
        "## Interval Type Counts",
        "",
    ]
    for interval_type, count in sorted(report.interval_type_counts.items()):
        lines.append(f"- {interval_type}: {count}")
    lines.extend(["", "## Selected Intervals", ""])
    lines.append("| review_id | type | depth range | question |")
    lines.append("|---|---|---:|---|")
    for interval in intervals:
        lines.append(
            "| {review_id} | {interval_type} | {start:.2f}-{end:.2f} | {question} |".format(
                review_id=interval["review_id"],
                interval_type=interval["interval_type"],
                start=float(interval["start_depth"]),
                end=float(interval["end_depth"]),
                question=str(interval["recommended_review_question"]).replace("|", "/"),
            )
        )
    lines.extend(["", "## Guardrails", ""])
    lines.extend(
        [
            "- weak-label candidate only; do not call these intervals ground truth",
            "- no final labels",
            "- no MVP-4C",
            "- no STC/APES",
            "- no deep learning",
            "- no new model training",
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


def _candidate_indices_for_selection(
    selection_name: str,
    *,
    prepared: dict[str, Any],
    records_by_index: dict[int, dict[str, Any]],
    config: DepthLevelManualReviewConfig,
) -> np.ndarray:
    indices = np.asarray(sorted(records_by_index), dtype=np.int32)
    if indices.size == 0:
        return indices
    labels = _record_values(records_by_index, indices, "label")
    scores = _record_values(records_by_index, indices, "score")
    high = config.selection_defaults.score_high_threshold
    low = config.selection_defaults.score_low_threshold
    boundary = config.selection_defaults.boundary_score_band
    disagreement = np.asarray(prepared["disagreement"], dtype=np.float32)[indices]
    label_confidence = np.asarray(prepared["label_confidence"], dtype=np.float32)[indices]
    low_confidence = label_confidence <= config.selection_defaults.low_confidence_threshold
    high_disagreement = disagreement >= config.selection_defaults.high_disagreement_threshold
    if selection_name == "select_top_positive_intervals":
        mask = labels == 1
    elif selection_name == "select_clear_negative_intervals":
        mask = (labels == 0) & (scores <= low)
        if not np.any(mask):
            mask = labels == 0
    elif selection_name == "select_high_score_positive_intervals":
        mask = (labels == 1) & (scores >= high)
    elif selection_name == "select_high_score_negative_or_disagreement_intervals":
        mask = (labels == 0) & ((scores >= high) | high_disagreement)
    elif selection_name == "select_low_score_positive_intervals":
        mask = (labels == 1) & (scores <= low)
    elif selection_name == "select_high_uncertainty_intervals":
        mask = (np.abs(scores - 0.5) <= boundary) | low_confidence | high_disagreement
    elif selection_name == "select_boundary_case_intervals":
        mask = np.abs(scores - 0.5) <= boundary
    else:
        mask = np.zeros(indices.shape, dtype=bool)
    return indices[mask]


def _select_clusters(
    *,
    indices: np.ndarray,
    prepared: dict[str, Any],
    records_by_index: dict[int, dict[str, Any]],
    selection: DepthLevelReviewSelectionConfig,
    config: DepthLevelManualReviewConfig,
) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    clusters = _cluster_indices_by_depth(
        indices,
        depth=np.asarray(prepared["depth"], dtype=np.float32),
        max_gap_ft=config.selection_defaults.max_interval_gap_ft,
    )
    if config.selection_defaults.min_interval_depth_span_ft > 0.0:
        clusters = [
            cluster
            for cluster in clusters
            if _depth_span(prepared["depth"], cluster)
            >= config.selection_defaults.min_interval_depth_span_ft
        ]
    scored = [
        (_cluster_priority(cluster, selection.sort_by, prepared, records_by_index), cluster)
        for cluster in clusters
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [cluster for _, cluster in scored]


def _cluster_priority(
    cluster: np.ndarray,
    sort_by: str,
    prepared: dict[str, Any],
    records_by_index: dict[int, dict[str, Any]],
) -> float:
    scores = _record_values(records_by_index, cluster, "score")
    if scores.size == 0:
        return 0.0
    if sort_by == "score_desc":
        return float(np.nanmax(scores))
    if sort_by == "score_asc":
        return float(1.0 - np.nanmin(scores))
    if sort_by == "boundary_score_asc":
        return float(1.0 - np.nanmin(np.abs(scores - 0.5)))
    disagreement = np.asarray(prepared["disagreement"], dtype=np.float32)[cluster]
    label_confidence = np.asarray(prepared["label_confidence"], dtype=np.float32)[cluster]
    uncertainty = np.maximum.reduce(
        [
            1.0 - np.abs(scores - 0.5) * 2.0,
            disagreement,
            1.0 - np.clip(label_confidence, 0.0, 1.0),
        ]
    )
    return float(np.nanmax(uncertainty))


def _build_interval_row(
    *,
    review_number: int,
    selection_rule: str,
    interval_type: str,
    label_indices: np.ndarray,
    prepared: dict[str, Any],
    records_by_index: dict[int, dict[str, Any]],
    refinement_report: dict[str, Any],
    top_feature_rows: list[dict[str, Any]],
    cast_candidates: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
) -> dict[str, Any]:
    depth = np.asarray(prepared["depth"], dtype=np.float32)
    start = float(np.nanmin(depth[label_indices]))
    end = float(np.nanmax(depth[label_indices]))
    flag_5700 = _interval_overlaps_5700(start, end) or bool(
        np.any(np.asarray(prepared["review_band"], dtype=bool)[label_indices])
    )
    return {
        "review_id": f"DLR-{review_number:03d}",
        "start_depth": start,
        "end_depth": end,
        "interval_type": interval_type,
        "selection_rule": selection_rule,
        "depth_count": int(label_indices.size),
        "weak_label_summary": _weak_label_summary(prepared, label_indices),
        "prediction_score_summary": _prediction_summary(records_by_index, label_indices),
        "confidence_summary": _confidence_summary(prepared, label_indices),
        "top_feature_summary": _top_feature_summary(
            prepared,
            label_indices,
            top_feature_rows,
            refinement_report,
        ),
        "plus_minus_disagreement_summary": _disagreement_summary(prepared, label_indices),
        "5700_band_flag": flag_5700,
        "cast_label_summary": _cast_label_summary(
            label_depth=depth[label_indices],
            cast_candidates=cast_candidates,
            cast_label_input=cast_label_input,
        ),
        "xsi_feature_summary": _xsi_feature_summary(
            prepared,
            label_indices,
            top_feature_rows,
        ),
        "recommended_review_question": _recommended_question(interval_type),
    }


def _fixed_review_interval_row(
    *,
    review_number: int,
    review_interval: DepthLevelReviewIntervalConfig,
    prepared: dict[str, Any],
    records_by_index: dict[int, dict[str, Any]],
    refinement_report: dict[str, Any],
    top_feature_rows: list[dict[str, Any]],
    cast_candidates: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    depth = np.asarray(prepared["depth"], dtype=np.float32)
    indices = np.flatnonzero(
        (depth >= review_interval.depth_min_ft) & (depth <= review_interval.depth_max_ft)
    ).astype(np.int32)
    if indices.size == 0:
        return None
    interval = _build_interval_row(
        review_number=review_number,
        selection_rule=review_interval.name,
        interval_type=review_interval.interval_type,
        label_indices=indices,
        prepared=prepared,
        records_by_index=records_by_index,
        refinement_report=refinement_report,
        top_feature_rows=top_feature_rows,
        cast_candidates=cast_candidates,
        cast_label_input=cast_label_input,
    )
    interval["recommended_review_question"] = (
        "Should the 5700 ft review band be retained, excluded, or handled separately?"
    )
    return interval


def _weak_label_summary(prepared: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    arrays = _as_dict(prepared.get("label_arrays"))
    has_channel = np.asarray(prepared["has_channel"], dtype=bool)[indices]
    return {
        "depth_count": int(indices.size),
        "depth_has_channel_fraction": _safe_mean(has_channel.astype(np.float32)),
        "clear_negative_fraction": _safe_mean(
            np.asarray(prepared["clear_negative"], dtype=bool)[indices].astype(np.float32)
        ),
        "candidate_fraction_mean": _optional_array_stat(
            arrays,
            "depth_candidate_fraction",
            indices,
            "mean",
        ),
        "candidate_fraction_max": _optional_array_stat(
            arrays,
            "depth_candidate_fraction",
            indices,
            "max",
        ),
        "max_severity": _optional_array_stat(arrays, "depth_max_severity", indices, "max"),
        "max_confidence": _optional_array_stat(
            arrays,
            "depth_max_confidence",
            indices,
            "max",
        ),
        "min_zc": _optional_array_stat(arrays, "depth_min_zc", indices, "min"),
        "p05_zc_min": _optional_array_stat(arrays, "depth_p05_zc", indices, "min"),
        "max_relative_drop": _optional_array_stat(
            arrays,
            "depth_max_relative_drop",
            indices,
            "max",
        ),
        "largest_azimuth_object_width_max": _optional_array_stat(
            arrays,
            "depth_largest_azimuth_object_width",
            indices,
            "max",
        ),
    }


def _prediction_summary(
    records_by_index: dict[int, dict[str, Any]],
    indices: np.ndarray,
) -> dict[str, Any]:
    records = [records_by_index[int(index)] for index in indices if int(index) in records_by_index]
    if not records:
        return {
            "available": False,
            "score_count": 0,
            "reason": "No score available for this interval in the selected refinement scenario.",
        }
    scores = np.asarray([float(record["score"]) for record in records], dtype=np.float32)
    predictions = np.asarray([int(record["prediction"]) for record in records], dtype=np.int32)
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int32)
    return {
        "available": True,
        "score_count": int(scores.size),
        "score_min": _safe_min(scores),
        "score_mean": _safe_mean(scores),
        "score_max": _safe_max(scores),
        "predicted_positive_fraction": _safe_mean(predictions.astype(np.float32)),
        "weak_label_positive_fraction": _safe_mean(labels.astype(np.float32)),
        "fold_indices": sorted({int(record["fold_index"]) for record in records}),
        "scenario_id": str(records[0].get("scenario_id")),
        "feature_group": str(records[0].get("feature_group")),
    }


def _confidence_summary(prepared: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    arrays = _as_dict(prepared.get("label_arrays"))
    label_confidence = np.asarray(prepared["label_confidence"], dtype=np.float32)[indices]
    orientation_confidence = np.asarray(prepared["orientation_confidence"], dtype=np.float32)[
        indices
    ]
    return {
        "depth_label_confidence_min": _safe_min(label_confidence),
        "depth_label_confidence_mean": _safe_mean(label_confidence),
        "depth_orientation_confidence_min": _safe_min(orientation_confidence),
        "depth_orientation_confidence_mean": _safe_mean(orientation_confidence),
        "depth_valid_fraction_mean": _optional_array_stat(
            arrays,
            "depth_valid_fraction",
            indices,
            "mean",
        ),
    }


def _top_feature_summary(
    prepared: dict[str, Any],
    indices: np.ndarray,
    top_feature_rows: list[dict[str, Any]],
    refinement_report: dict[str, Any],
) -> dict[str, Any]:
    values = _feature_value_summaries(prepared, indices, top_feature_rows, limit=6)
    return {
        "source": "refinement top coefficient features",
        "best_feature_group": refinement_report.get("best_feature_group"),
        "features": values,
    }


def _disagreement_summary(prepared: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    disagreement = np.asarray(prepared["disagreement"], dtype=np.float32)[indices]
    return {
        "plus_minus_disagreement_min": _safe_min(disagreement),
        "plus_minus_disagreement_mean": _safe_mean(disagreement),
        "plus_minus_disagreement_max": _safe_max(disagreement),
        "plus_minus_disagreement_p95": _safe_percentile(disagreement, 95.0),
    }


def _cast_label_summary(
    *,
    label_depth: np.ndarray,
    cast_candidates: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
) -> dict[str, Any]:
    candidate_summary = _cast_candidate_summary(label_depth, cast_candidates)
    raw_summary = _cast_input_summary(label_depth, cast_label_input)
    return {
        "candidate_npz_available": bool(cast_candidates),
        "cast_label_input_available": bool(cast_label_input),
        "weak_label_candidate_summary": candidate_summary,
        "cast_zc_summary": raw_summary,
    }


def _xsi_feature_summary(
    prepared: dict[str, Any],
    indices: np.ndarray,
    top_feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": "depth_level_xsi_features_v001",
        "top_feature_values": _feature_value_summaries(
            prepared,
            indices,
            top_feature_rows,
            limit=8,
        ),
    }


def _feature_value_summaries(
    prepared: dict[str, Any],
    indices: np.ndarray,
    top_feature_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    features = np.asarray(prepared["features"], dtype=np.float32)
    feature_names = np.asarray(prepared["feature_names"]).astype(str)
    top_names = [
        str(row.get("feature_name"))
        for row in top_feature_rows
        if row.get("feature_name")
    ]
    if not top_names:
        top_names = feature_names[: min(limit, feature_names.size)].tolist()
    summaries: list[dict[str, Any]] = []
    for name in top_names[:limit]:
        matches = np.flatnonzero(feature_names == name)
        if matches.size == 0:
            continue
        values = features[indices, int(matches[0])]
        summaries.append(
            {
                "feature_name": name,
                "mean": _safe_mean(values),
                "std": _safe_std(values),
                "min": _safe_min(values),
                "max": _safe_max(values),
            }
        )
    return summaries


def _cast_candidate_summary(
    label_depth: np.ndarray,
    cast_candidates: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not cast_candidates or "cast_depth" not in cast_candidates:
        return {"available": False}
    rows = _cast_rows_for_interval(label_depth, cast_candidates["cast_depth"])
    if rows.size == 0:
        return {"available": False, "reason": "No CAST weak-label candidate rows in interval."}
    result: dict[str, Any] = {"available": True, "cast_depth_count": int(rows.size)}
    if "presence_plus" in cast_candidates:
        presence = np.asarray(cast_candidates["presence_plus"])[rows]
        result["presence_plus_fraction"] = _safe_mean((presence > 0).astype(np.float32))
    if "severity_plus" in cast_candidates:
        result["severity_plus_max"] = _safe_max(np.asarray(cast_candidates["severity_plus"])[rows])
    if "label_confidence_plus" in cast_candidates:
        result["label_confidence_plus_mean"] = _safe_mean(
            np.asarray(cast_candidates["label_confidence_plus"])[rows]
        )
        result["label_confidence_plus_max"] = _safe_max(
            np.asarray(cast_candidates["label_confidence_plus"])[rows]
        )
    if "relative_drop_plus" in cast_candidates:
        result["relative_drop_plus_max"] = _safe_max(
            np.asarray(cast_candidates["relative_drop_plus"])[rows]
        )
    if "presence_minus_ablation" in cast_candidates and "presence_plus" in cast_candidates:
        plus = np.asarray(cast_candidates["presence_plus"])[rows] > 0
        minus = np.asarray(cast_candidates["presence_minus_ablation"])[rows] > 0
        result["plus_minus_presence_disagreement_fraction"] = _safe_mean(
            (plus != minus).astype(np.float32)
        )
    return result


def _cast_input_summary(
    label_depth: np.ndarray,
    cast_label_input: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not cast_label_input or "cast_depth" not in cast_label_input:
        return {"available": False}
    rows = _cast_rows_for_interval(label_depth, cast_label_input["cast_depth"])
    if rows.size == 0:
        return {"available": False, "reason": "No CAST label input rows in interval."}
    result: dict[str, Any] = {"available": True, "cast_depth_count": int(rows.size)}
    if "cast_zc" in cast_label_input:
        zc = np.asarray(cast_label_input["cast_zc"], dtype=np.float32)[rows]
        result["zc_min"] = _safe_min(zc)
        result["zc_p05"] = _safe_percentile(zc, 5.0)
        result["zc_p10"] = _safe_percentile(zc, 10.0)
        result["zc_finite_fraction"] = _safe_mean(np.isfinite(zc).astype(np.float32))
    if "orientation_confidence" in cast_label_input:
        result["orientation_confidence_mean"] = _safe_mean(
            np.asarray(cast_label_input["orientation_confidence"], dtype=np.float32)[rows]
        )
    if "low_inc_mask" in cast_label_input:
        result["low_inc_fraction"] = _safe_mean(
            np.asarray(cast_label_input["low_inc_mask"], dtype=bool)[rows].astype(np.float32)
        )
    return result


def _prediction_records(
    rows: list[dict[str, Any]],
    label_depth: np.ndarray,
) -> list[dict[str, Any]]:
    row_depth = np.asarray([float(row["depth"]) for row in rows], dtype=np.float32)
    label_indices = _nearest_indices(label_depth, row_depth) if row_depth.size else np.asarray([])
    records = []
    for index, row in enumerate(rows):
        records.append(
            {
                "label_index": int(label_indices[index]),
                "scenario_id": str(row.get("scenario_id", "")),
                "feature_group": str(row.get("feature_group", "")),
                "fold_index": int(float(row.get("fold_index") or 0)),
                "depth": float(row.get("depth") or 0.0),
                "label": int(float(row.get("label") or 0)),
                "score": float(row.get("score") or 0.0),
                "prediction": int(float(row.get("prediction") or 0)),
            }
        )
    return records


def _records_by_label_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        label_index = int(record["label_index"])
        existing = result.get(label_index)
        if existing is None or float(record["score"]) > float(existing["score"]):
            result[label_index] = record
    return result


def _record_values(
    records_by_index: dict[int, dict[str, Any]],
    indices: np.ndarray,
    key: str,
) -> np.ndarray:
    return np.asarray(
        [float(records_by_index[int(index)][key]) for index in indices],
        dtype=np.float32,
    )


def _cluster_indices_by_depth(
    indices: np.ndarray,
    *,
    depth: np.ndarray,
    max_gap_ft: float,
) -> list[np.ndarray]:
    sorted_indices = indices[np.argsort(depth[indices])]
    clusters: list[list[int]] = []
    current: list[int] = []
    previous_depth: float | None = None
    for index in sorted_indices:
        value = float(depth[int(index)])
        if previous_depth is None or value - previous_depth <= max_gap_ft:
            current.append(int(index))
        else:
            clusters.append(current)
            current = [int(index)]
        previous_depth = value
    if current:
        clusters.append(current)
    return [np.asarray(cluster, dtype=np.int32) for cluster in clusters]


def _rows_for_scenario(
    rows: list[dict[str, Any]],
    scenario_id: str | None,
) -> list[dict[str, Any]]:
    if not scenario_id:
        return rows
    return [row for row in rows if row.get("scenario_id") == scenario_id]


def _top_feature_rows(
    report: dict[str, Any],
    *,
    scenario_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    top_features = _as_dict(report.get("top_features"))
    rows = _as_list(top_features.get(str(scenario_id))) if scenario_id else []
    if not rows:
        for value in top_features.values():
            rows = _as_list(value)
            if rows:
                break
    cleaned = [row for row in rows if isinstance(row, dict)]
    cleaned.sort(
        key=lambda row: abs(float(row.get("mean_coefficient") or 0.0)),
        reverse=True,
    )
    return cleaned[:limit]


def _validate_review_guardrails(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    refinement_report: dict[str, Any],
    refinement_gate_report: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if refinement_report.get("report_version") != "depth_level_refinement_v001":
        errors.append("refinement report_version must be depth_level_refinement_v001.")
    decision = _gate_decision(refinement_gate_report)
    if decision != "go":
        errors.append(f"refinement gate decision must be go before manual review: {decision}.")
    if refinement_report.get("production_training") is not False:
        errors.append("refinement report indicates production_training.")
    for guardrail in ("no_final_labels", "no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        if refinement_report.get(guardrail) is not True:
            errors.append(f"refinement report does not set {guardrail}=true.")
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


def _fixed_depth_mask(
    depth: np.ndarray,
    interval: DepthLevelReviewIntervalConfig,
) -> np.ndarray:
    return (depth >= interval.depth_min_ft) & (depth <= interval.depth_max_ft)


def _interval_overlaps_5700(start_depth: float, end_depth: float) -> bool:
    return start_depth <= 5700.0 <= end_depth


def _depth_span(depth: np.ndarray, indices: np.ndarray) -> float:
    values = np.asarray(depth, dtype=np.float32)[indices]
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values) - np.nanmin(values))


def _cast_rows_for_interval(label_depth: np.ndarray, cast_depth: np.ndarray) -> np.ndarray:
    values = np.asarray(label_depth, dtype=np.float32)
    cast_values = np.asarray(cast_depth, dtype=np.float32).reshape(-1)
    if values.size == 0 or cast_values.size == 0:
        return np.asarray([], dtype=np.int32)
    start = float(np.nanmin(values))
    end = float(np.nanmax(values))
    rows = np.flatnonzero((cast_values >= start) & (cast_values <= end)).astype(np.int32)
    if rows.size:
        return rows
    nearest = _nearest_indices(cast_values, np.asarray([start, end], dtype=np.float32))
    return np.unique(nearest).astype(np.int32)


def _nearest_indices(reference_depth: np.ndarray, query_depth: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_depth, dtype=np.float32).reshape(-1)
    query = np.asarray(query_depth, dtype=np.float32).reshape(-1)
    order = np.argsort(reference)
    sorted_depth = reference[order]
    positions = np.searchsorted(sorted_depth, query)
    positions = np.clip(positions, 0, sorted_depth.size - 1)
    previous = np.clip(positions - 1, 0, sorted_depth.size - 1)
    use_previous = np.abs(sorted_depth[previous] - query) < np.abs(
        sorted_depth[positions] - query
    )
    nearest = np.where(use_previous, previous, positions)
    return order[nearest].astype(np.int32)


def _optional_array_stat(
    arrays: dict[str, Any],
    key: str,
    indices: np.ndarray,
    stat: str,
) -> float | int | None:
    if key not in arrays:
        return None
    values = np.asarray(arrays[key])[indices]
    if stat == "mean":
        return _safe_mean(values)
    if stat == "max":
        return _safe_max(values)
    if stat == "min":
        return _safe_min(values)
    raise ValueError(f"Unsupported stat: {stat}")


def _safe_mean(values: np.ndarray) -> float | None:
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _safe_std(values: np.ndarray) -> float | None:
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(np.std(finite))


def _safe_min(values: np.ndarray) -> float | int | None:
    finite = _finite(values)
    if finite.size == 0:
        return None
    value = np.min(finite)
    return int(value) if np.issubdtype(finite.dtype, np.integer) else float(value)


def _safe_max(values: np.ndarray) -> float | int | None:
    finite = _finite(values)
    if finite.size == 0:
        return None
    value = np.max(finite)
    return int(value) if np.issubdtype(finite.dtype, np.integer) else float(value)


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == np.bool_:
        array = array.astype(np.float32)
    finite = array[np.isfinite(array)]
    return finite.reshape(-1)


def _recommended_question(interval_type: str) -> str:
    questions = {
        "true_positive_like": (
            "Does this high-score positive weak-label candidate interval show "
            "physically plausible CAST and XSI anomaly evidence?"
        ),
        "clear_negative_like": "Does this clear negative interval look physically normal?",
        "false_positive_like": (
            "Is this high-score negative/disagreement interval a missed weak-label "
            "candidate, XSI noise, or model artifact?"
        ),
        "false_negative_like": (
            "Is this low-score positive interval label noise, weak XSI sensitivity, "
            "or a physically subtle anomaly?"
        ),
        "high_uncertainty": (
            "Should low-confidence or plus/minus-disagreement depths be excluded "
            "or handled separately?"
        ),
        "5700_band_review": (
            "Should the 5700 ft review band be retained, excluded, or handled separately?"
        ),
        "boundary_case": (
            "Is this boundary-score interval acceptable for review or should it be uncertain?"
        ),
    }
    return questions.get(interval_type, "Review physical plausibility of this interval.")


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


def _load_optional_npz(
    path: Path | str | None,
    field_names: set[str],
    warnings: list[str],
) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    optional_path = Path(path)
    if not optional_path.exists():
        warnings.append(f"Optional review input not found: {optional_path}.")
        return {}
    with np.load(optional_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key in field_names}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _gate_decision(gate_report: dict[str, Any]) -> str | None:
    value = gate_report.get("decision", gate_report.get("gate_decision"))
    return None if value is None else str(value)


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


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]
