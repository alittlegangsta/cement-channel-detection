from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MORPHOLOGY_AUDIT_VERSION = "geometry_regression_morphology_audit_v001"
MORPHOLOGY_FIELDS = (
    "largest_connected_component_fraction",
    "max_azimuth_channel_fraction",
    "relative_anomaly_fraction",
    "combined_channel_fraction",
    "depth_label_confidence",
)
DEFAULT_FOLD_COUNT = 3


@dataclass(frozen=True)
class GeometryRegressionMorphologyAuditConfig:
    fold_count: int = DEFAULT_FOLD_COUNT
    interval_limit_per_type: int = 12
    local_sparse_simple_mean_max: float = 0.10
    local_sparse_simple_max_min: float = 0.25
    local_sparse_max_azimuth_min: float = 0.50
    broad_combined_mean_min: float = 0.20
    broad_lcc_mean_min: float = 0.10
    scattered_simple_mean_min: float = 0.03
    scattered_lcc_mean_max: float = 0.03
    scattered_max_azimuth_max: float = 0.20
    boundary_confidence_min: float = 0.50


@dataclass(frozen=True)
class GeometryRegressionMorphologyAuditReport:
    audit_version: str
    generated_at: str
    inputs: dict[str, str]
    morphology_fields: list[str]
    fold_summaries: list[dict[str, Any]]
    simple_low_zc_relationships: list[dict[str, Any]]
    xsi_feature_relationships: list[dict[str, Any]]
    selected_intervals: list[dict[str, Any]]
    interval_selection_policy: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    no_morphology_target_added: bool
    no_target_semantics_change: bool
    no_model_training: bool
    no_model_weights: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_geometry_regression_morphology(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    config: GeometryRegressionMorphologyAuditConfig | None = None,
    inputs: dict[str, str] | None = None,
) -> tuple[GeometryRegressionMorphologyAuditReport, list[dict[str, Any]]]:
    cfg = config or GeometryRegressionMorphologyAuditConfig()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_inputs(label_arrays, feature_arrays)

    depth = np.asarray(label_arrays["depth"], dtype=np.float32).reshape(-1)
    kernels = np.asarray(label_arrays["geometry_kernel"]).astype(str)
    simple = np.asarray(
        label_arrays["weighted_channel_fraction_zc_lt_2p5"],
        dtype=np.float32,
    )
    feature_matrix = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    finite_feature_rows = np.all(np.isfinite(feature_matrix), axis=1)
    feature_matrix = np.nan_to_num(
        feature_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    fold_ids = _fold_ids(depth, cfg.fold_count)

    fold_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for kernel_index, kernel in enumerate(kernels):
        simple_kernel = simple[kernel_index]
        simple_depth_mean = np.mean(simple_kernel, axis=1)
        simple_depth_max = np.max(simple_kernel, axis=1)
        for field in MORPHOLOGY_FIELDS:
            values = np.asarray(label_arrays[field], dtype=np.float32)[kernel_index]
            depth_mean = np.mean(values, axis=1)
            depth_max = np.max(values, axis=1)
            for fold in range(cfg.fold_count):
                mask = fold_ids == fold
                fold_rows.append(
                    {
                        "section": "fold_morphology_distribution",
                        "geometry_kernel": str(kernel),
                        "morphology_field": field,
                        "fold": fold,
                        "depth_min": _finite_min(depth[mask]),
                        "depth_max": _finite_max(depth[mask]),
                        "sample_count": int(np.count_nonzero(mask)),
                        "mean": _finite_mean(depth_mean[mask]),
                        "median": _finite_percentile(depth_mean[mask], 50.0),
                        "p90": _finite_percentile(depth_mean[mask], 90.0),
                        "p95": _finite_percentile(depth_mean[mask], 95.0),
                        "zero_fraction": _fraction(depth_mean[mask] <= 0.0),
                        "simple_low_zc_mean": _finite_mean(simple_depth_mean[mask]),
                        "simple_low_zc_max_mean": _finite_mean(simple_depth_max[mask]),
                    }
                )
            relation_rows.append(
                {
                    "section": "simple_low_zc_relationship",
                    "geometry_kernel": str(kernel),
                    "morphology_field": field,
                    "aggregation": "receiver_mean_by_depth",
                    "pearson_vs_simple_low_zc_mean": _pearson(
                        depth_mean,
                        simple_depth_mean,
                    ),
                    "spearman_vs_simple_low_zc_mean": _spearman(
                        depth_mean,
                        simple_depth_mean,
                    ),
                    "pearson_vs_simple_low_zc_max": _pearson(
                        depth_max,
                        simple_depth_max,
                    ),
                    "spearman_vs_simple_low_zc_max": _spearman(
                        depth_max,
                        simple_depth_max,
                    ),
                }
            )
            feature_rows.append(
                _top_feature_relationship(
                    kernel=str(kernel),
                    field=field,
                    target=depth_mean,
                    features=feature_matrix,
                    feature_names=feature_names,
                    finite_feature_rows=finite_feature_rows,
                )
            )

    selected = _selected_intervals(
        label_arrays=label_arrays,
        depth=depth,
        kernels=kernels,
        config=cfg,
    )
    rows = [*fold_rows, *relation_rows, *feature_rows]
    rows.extend({"section": "selected_interval", **row} for row in selected)
    report = GeometryRegressionMorphologyAuditReport(
        audit_version=MORPHOLOGY_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        morphology_fields=list(MORPHOLOGY_FIELDS),
        fold_summaries=[_without_section(row) for row in fold_rows],
        simple_low_zc_relationships=[_without_section(row) for row in relation_rows],
        xsi_feature_relationships=[_without_section(row) for row in feature_rows],
        selected_intervals=selected,
        interval_selection_policy=asdict(cfg),
        warnings=warnings,
        errors=errors,
        no_morphology_target_added=True,
        no_target_semantics_change=True,
        no_model_training=True,
        no_model_weights=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "morphology-aware target redesign",
            "new morphology target",
            "target semantics change",
            "new XSI feature",
            "formal model training",
            "model weight export",
            "final label generation",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, rows


def audit_geometry_regression_morphology_from_paths(
    *,
    labels_npz: Path | str,
    features_npz: Path | str,
    output_md: Path | str,
    output_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> GeometryRegressionMorphologyAuditReport:
    label_arrays = _load_npz(Path(labels_npz))
    feature_arrays = _load_npz(Path(features_npz))
    report, rows = audit_geometry_regression_morphology(
        label_arrays=label_arrays,
        feature_arrays=feature_arrays,
        inputs={"labels_npz": str(labels_npz), "features_npz": str(features_npz)},
    )
    write_geometry_regression_morphology_outputs(
        report,
        rows,
        output_md=Path(output_md),
        output_json=Path(output_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def write_geometry_regression_morphology_outputs(
    report: GeometryRegressionMorphologyAuditReport,
    rows: list[dict[str, Any]],
    *,
    output_md: Path,
    output_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_csv, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(
        format_geometry_regression_morphology_markdown(report),
        encoding="utf-8",
    )
    _write_csv(rows, output_csv)


def format_geometry_regression_morphology_markdown(
    report: GeometryRegressionMorphologyAuditReport,
) -> str:
    lines = [
        "# Geometry Regression Morphology Audit",
        "",
        "This audit uses existing morphology arrays only. It does not add a "
        "morphology target, change label semantics, train a model, or authorize "
        "MVP-4C/STC/APES/deep learning/final labels.",
        "",
        f"- audit_version: `{report.audit_version}`",
        f"- morphology_fields: `{report.morphology_fields}`",
        f"- no_morphology_target_added: `{report.no_morphology_target_added}`",
        "",
        "## Fold Morphology Distributions",
        "",
    ]
    lines.extend(_dict_message_lines(report.fold_summaries))
    lines.extend(["", "## Simple Low-Zc Relationships", ""])
    lines.extend(_dict_message_lines(report.simple_low_zc_relationships))
    lines.extend(["", "## XSI Feature Relationships", ""])
    lines.extend(_dict_message_lines(report.xsi_feature_relationships))
    lines.extend(["", "## Selected Morphology-Sensitive Intervals", ""])
    lines.extend(_dict_message_lines(report.selected_intervals))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _selected_intervals(
    *,
    label_arrays: dict[str, np.ndarray],
    depth: np.ndarray,
    kernels: np.ndarray,
    config: GeometryRegressionMorphologyAuditConfig,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    simple = np.asarray(label_arrays["weighted_channel_fraction_zc_lt_2p5"], dtype=np.float32)
    combined = np.asarray(label_arrays["combined_channel_fraction"], dtype=np.float32)
    lcc = np.asarray(
        label_arrays["largest_connected_component_fraction"],
        dtype=np.float32,
    )
    max_az = np.asarray(label_arrays["max_azimuth_channel_fraction"], dtype=np.float32)
    confidence = np.asarray(label_arrays["depth_label_confidence"], dtype=np.float32)
    total_count = np.asarray(label_arrays.get("total_cell_count", np.ones_like(simple)))
    for kernel_index, kernel in enumerate(kernels.astype(str)):
        metrics = {
            "simple_mean": np.mean(simple[kernel_index], axis=1),
            "simple_max": np.max(simple[kernel_index], axis=1),
            "combined_mean": np.mean(combined[kernel_index], axis=1),
            "lcc_mean": np.mean(lcc[kernel_index], axis=1),
            "max_azimuth_max": np.max(max_az[kernel_index], axis=1),
            "confidence_min": np.min(confidence[kernel_index], axis=1),
            "no_overlap": np.any(total_count[kernel_index] <= 0, axis=1),
        }
        masks = {
            "local_severe_but_sparse": (
                (metrics["simple_mean"] <= config.local_sparse_simple_mean_max)
                & (metrics["simple_max"] >= config.local_sparse_simple_max_min)
                & (metrics["max_azimuth_max"] >= config.local_sparse_max_azimuth_min)
            ),
            "broad_moderate_anomaly": (
                (metrics["combined_mean"] >= config.broad_combined_mean_min)
                & (metrics["lcc_mean"] >= config.broad_lcc_mean_min)
            ),
            "scattered_low_zc_noise_like": (
                (metrics["simple_mean"] >= config.scattered_simple_mean_min)
                & (metrics["lcc_mean"] <= config.scattered_lcc_mean_max)
                & (metrics["max_azimuth_max"] <= config.scattered_max_azimuth_max)
                & (metrics["confidence_min"] >= 0.9)
            ),
            "boundary_affected_interval": (
                (metrics["confidence_min"] < config.boundary_confidence_min)
                | metrics["no_overlap"]
            ),
            "morphology_sensitive": (
                np.abs(metrics["combined_mean"] - metrics["simple_mean"]) >= 0.10
            )
            | (metrics["lcc_mean"] >= 0.15),
        }
        for interval_type, mask in masks.items():
            segments = _segments_from_mask(depth, mask)
            ranked = _rank_segments(segments, metrics, interval_type)
            for segment in ranked[: config.interval_limit_per_type]:
                selected.append(
                    {
                        "interval_type": interval_type,
                        "geometry_kernel": str(kernel),
                        **segment,
                    }
                )
    return selected


def _segments_from_mask(depth: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    segments = []
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return segments
    parts = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    for part in parts:
        depth_values = depth[part]
        segments.append(
            {
                "start_index": int(part[0]),
                "end_index": int(part[-1]),
                "depth_min": float(np.min(depth_values)),
                "depth_max": float(np.max(depth_values)),
                "depth_count": int(part.size),
            }
        )
    return segments


def _rank_segments(
    segments: list[dict[str, Any]],
    metrics: dict[str, np.ndarray],
    interval_type: str,
) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        slc = slice(segment["start_index"], segment["end_index"] + 1)
        score = _segment_score(metrics, slc, interval_type)
        rows.append(
            {
                **segment,
                "selection_score": score,
                "simple_mean": _finite_mean(metrics["simple_mean"][slc]),
                "simple_max": _finite_max(metrics["simple_max"][slc]),
                "combined_mean": _finite_mean(metrics["combined_mean"][slc]),
                "lcc_mean": _finite_mean(metrics["lcc_mean"][slc]),
                "max_azimuth_max": _finite_max(metrics["max_azimuth_max"][slc]),
                "confidence_min": _finite_min(metrics["confidence_min"][slc]),
                "boundary_no_overlap": bool(np.any(metrics["no_overlap"][slc])),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["selection_score"]), row["depth_min"]))


def _segment_score(
    metrics: dict[str, np.ndarray],
    slc: slice,
    interval_type: str,
) -> float:
    if interval_type == "boundary_affected_interval":
        return float(1.0 - (_finite_min(metrics["confidence_min"][slc]) or 0.0))
    if interval_type == "local_severe_but_sparse":
        return float(_finite_max(metrics["simple_max"][slc]) or 0.0)
    if interval_type == "broad_moderate_anomaly":
        return float(_finite_mean(metrics["combined_mean"][slc]) or 0.0)
    if interval_type == "scattered_low_zc_noise_like":
        return float(_finite_mean(metrics["simple_mean"][slc]) or 0.0)
    return float(_finite_mean(metrics["lcc_mean"][slc]) or 0.0)


def _top_feature_relationship(
    *,
    kernel: str,
    field: str,
    target: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    finite_feature_rows: np.ndarray,
) -> dict[str, Any]:
    selected = finite_feature_rows & np.isfinite(target)
    rows = []
    for index, name in enumerate(feature_names):
        values = features[:, index]
        rows.append(
            {
                "feature_index": index,
                "feature_name": str(name),
                "pearson": _pearson(values[selected], target[selected]),
                "spearman": _spearman(values[selected], target[selected]),
            }
        )
    top_pearson = max(rows, key=lambda row: _abs_rank(row["pearson"]))
    top_spearman = max(rows, key=lambda row: _abs_rank(row["spearman"]))
    return {
        "section": "xsi_feature_relationship",
        "geometry_kernel": kernel,
        "morphology_field": field,
        "aggregation": "receiver_mean_by_depth",
        "top_abs_pearson_feature": top_pearson["feature_name"],
        "top_abs_pearson": _abs_or_none(top_pearson["pearson"]),
        "top_abs_spearman_feature": top_spearman["feature_name"],
        "top_abs_spearman": _abs_or_none(top_spearman["spearman"]),
        "sample_count": int(np.count_nonzero(selected)),
    }


def _validate_inputs(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
) -> None:
    required_labels = {
        "depth",
        "geometry_kernel",
        "weighted_channel_fraction_zc_lt_2p5",
        *MORPHOLOGY_FIELDS,
    }
    required_features = {
        "depth",
        "depth_level_xsi_features",
        "depth_level_xsi_feature_names",
    }
    missing_labels = sorted(required_labels - set(label_arrays))
    missing_features = sorted(required_features - set(feature_arrays))
    if missing_labels:
        raise KeyError("Morphology audit missing label field(s): " + ", ".join(missing_labels))
    if missing_features:
        raise KeyError(
            "Morphology audit missing feature field(s): " + ", ".join(missing_features)
        )
    label_depth = np.asarray(label_arrays["depth"]).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"]).reshape(-1)
    if label_depth.size != feature_depth.size:
        raise ValueError("Morphology audit label/feature depth counts differ.")


def _fold_ids(depth: np.ndarray, fold_count: int) -> np.ndarray:
    order = np.argsort(depth)
    folds = np.empty(depth.size, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(order, fold_count)):
        folds[indices] = fold
    return folds


def _pearson(values: np.ndarray, target: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return None
    x = x[finite]
    y = y[finite]
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(values: np.ndarray, target: np.ndarray) -> float | None:
    return _pearson(_rank(values), _rank(target))


def _rank(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    return ranks


def _abs_rank(value: Any) -> float:
    number = _as_float(value)
    return -1.0 if number is None else abs(number)


def _abs_or_none(value: Any) -> float | None:
    number = _as_float(value)
    return None if number is None else abs(number)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _without_section(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "section"}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _finite_mean(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.mean(finite))


def _finite_min(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.min(finite))


def _finite_max(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.max(finite))


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.percentile(finite, percentile))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _dict_message_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row}" for row in rows]
