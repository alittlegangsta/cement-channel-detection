from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import label as connected_label

from cement_channel.alignment.xsi_geometry import ReceiverGeometry
from cement_channel.labels.cast_zc_source import CastZcSource, load_controlled_cast_zc_source

GEOMETRY_AWARE_REGRESSION_LABEL_VERSION = "geometry_aware_regression_labels_v001"
GEOMETRY_AWARE_REGRESSION_REPORT_VERSION = "geometry_aware_regression_labels_report_v001"

REGRESSION_KERNELS = (
    "r7_reference_point",
    "midpoint_window",
    "uniform_source_receiver_interval",
    "triangular_midpoint_weighted",
)

DEPTH_LEVEL_TARGET_VIEWS = (
    "receiver_mean",
    "receiver_max",
    "receiver_p90",
    "receiver_std",
    "full_360_fraction",
)

TARGET_VIEW_DEFINITIONS = {
    "weighted_channel_fraction_zc_lt_2p5": (
        "Receiver-level primary target: weighted fraction of finite CAST cells in "
        "the receiver/kernel geometry region with raw Zc < 2.5 MRayl."
    ),
    "raw_channel_fraction_zc_lt_2p5": (
        "Receiver-level unweighted fraction of finite CAST cells in the "
        "receiver/kernel geometry region with raw Zc < 2.5 MRayl."
    ),
    "relative_anomaly_fraction": (
        "Receiver-level fraction of finite CAST cells satisfying the configured "
        "relative-drop anomaly threshold."
    ),
    "combined_channel_fraction": (
        "Receiver-level fraction of finite CAST cells satisfying raw-Zc or "
        "relative-drop candidate rules."
    ),
    "receiver_mean": (
        "Depth-level view: mean of the receiver-level primary target across the "
        "13 receivers for each XSI reference depth."
    ),
    "receiver_max": (
        "Depth-level view: maximum of the receiver-level primary target across "
        "the 13 receivers for each XSI reference depth."
    ),
    "receiver_p90": (
        "Depth-level view: 90th percentile of the receiver-level primary target "
        "across the 13 receivers for each XSI reference depth."
    ),
    "receiver_std": (
        "Depth-level view: standard deviation of the receiver-level primary "
        "target across the 13 receivers for each XSI reference depth."
    ),
    "full_360_fraction": (
        "Depth-level view: combined-channel cell fraction over the full "
        "source/receiver geometry depth span for all azimuths."
    ),
    "derived_positive_at_fraction_0p01": (
        "Derived binary sanity view only: receiver_max >= 0.01. Not a final label."
    ),
    "derived_positive_at_fraction_0p05": (
        "Derived binary sanity view only: receiver_max >= 0.05. Not a final label."
    ),
    "derived_positive_at_fraction_0p10": (
        "Derived binary sanity view only: receiver_max >= 0.10. Not a final label."
    ),
}


@dataclass(frozen=True)
class GeometryAwareRegressionConfig:
    primary_target: str = "weighted_channel_fraction_zc_lt_2p5"
    zc_threshold_mrayl: float = 2.5
    relative_drop_threshold: float = 0.35
    zc_threshold_status: str = "human_reviewed_candidate_v001"
    no_final_labels: bool = True
    geometry_kernels: tuple[str, ...] = REGRESSION_KERNELS
    midpoint_window_half_width_ft: float = 0.25
    interval_padding_ft: float = 0.0
    derived_positive_thresholds: tuple[float, ...] = (0.01, 0.05, 0.10)
    min_raw_zc_finite_ratio: float = 0.5


@dataclass(frozen=True)
class GeometryAwareRegressionLabelReport:
    report_version: str
    label_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    primary_target: str
    raw_zc_available: bool
    raw_zc_source: dict[str, Any]
    geometry_sign_status: dict[str, Any]
    primary_target_layer: str
    depth_level_target_views: list[str]
    target_view_definitions: dict[str, str]
    kernel_summaries: list[dict[str, Any]]
    target_view_summaries: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_geometry_aware_regression_config(
    path: Path | str,
) -> GeometryAwareRegressionConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Geometry-aware regression config must contain a mapping.")
    labels = _as_dict(data.get("labels"))
    kernels = data.get("geometry_kernels", REGRESSION_KERNELS)
    if isinstance(kernels, dict):
        kernels = kernels.get("enabled", REGRESSION_KERNELS)
    kernel_tuple = tuple(str(item) for item in _as_list(kernels))
    invalid = [kernel for kernel in kernel_tuple if kernel not in REGRESSION_KERNELS]
    if invalid:
        raise ValueError("Unsupported geometry regression kernel(s): " + ", ".join(invalid))
    derived = _as_dict(data.get("derived_binary_views"))
    thresholds = tuple(
        float(item)
        for item in _as_list(derived.get("thresholds", (0.01, 0.05, 0.10)))
    )
    return GeometryAwareRegressionConfig(
        primary_target=str(
            labels.get("primary_target", "weighted_channel_fraction_zc_lt_2p5")
        ),
        zc_threshold_mrayl=float(labels.get("zc_threshold_mrayl", 2.5)),
        relative_drop_threshold=float(labels.get("relative_drop_threshold", 0.35)),
        zc_threshold_status=str(
            labels.get("zc_threshold_status", "human_reviewed_candidate_v001")
        ),
        no_final_labels=bool(labels.get("no_final_labels", True)),
        geometry_kernels=kernel_tuple,
        midpoint_window_half_width_ft=float(
            data.get("midpoint_window_half_width_ft", 0.25)
        ),
        interval_padding_ft=float(data.get("interval_padding_ft", 0.0)),
        derived_positive_thresholds=thresholds,
        min_raw_zc_finite_ratio=float(data.get("min_raw_zc_finite_ratio", 0.5)),
    )


def build_geometry_aware_regression_labels_from_paths(
    *,
    raw_cast_zc_npz_candidates: Sequence[Path | str],
    depth_level_features_npz: Path | str,
    geometry_config_path: Path | str,
    config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    cast_baseline_npz: Path | str | None = None,
    overwrite: bool = False,
) -> GeometryAwareRegressionLabelReport:
    config = load_geometry_aware_regression_config(config_path)
    raw_source = load_controlled_cast_zc_source(
        raw_cast_zc_npz_candidates,
        min_finite_ratio=config.min_raw_zc_finite_ratio,
    )
    feature_arrays = _load_npz(depth_level_features_npz)
    baseline_arrays = _load_npz(cast_baseline_npz) if cast_baseline_npz else None
    arrays, report = build_geometry_aware_regression_labels(
        raw_source=raw_source,
        feature_arrays=feature_arrays,
        geometry=ReceiverGeometry.from_yaml(geometry_config_path),
        config=config,
        baseline_arrays=baseline_arrays,
        inputs={
            "raw_cast_zc_npz_candidates": ", ".join(
                str(path) for path in raw_cast_zc_npz_candidates
            ),
            "cast_baseline_npz": str(cast_baseline_npz or ""),
            "depth_level_features_npz": str(depth_level_features_npz),
            "geometry_config_path": str(geometry_config_path),
            "config_path": str(config_path),
        },
        output_npz=Path(output_npz),
    )
    write_geometry_aware_regression_label_outputs(
        arrays,
        report,
        output_npz=Path(output_npz),
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return report


def build_geometry_aware_regression_labels(
    *,
    raw_source: CastZcSource,
    feature_arrays: dict[str, np.ndarray],
    geometry: ReceiverGeometry,
    config: GeometryAwareRegressionConfig,
    baseline_arrays: dict[str, np.ndarray] | None = None,
    inputs: dict[str, str] | None = None,
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], GeometryAwareRegressionLabelReport]:
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(feature_arrays, geometry, config, errors)
    reference_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    relative_drop = _relative_drop_array(
        baseline_arrays,
        raw_source=raw_source,
        warnings=warnings,
    )
    sorted_context = _build_cast_context(
        raw_source=raw_source,
        relative_drop=relative_drop,
        zc_threshold=config.zc_threshold_mrayl,
        relative_drop_threshold=config.relative_drop_threshold,
    )

    per_kernel: list[dict[str, np.ndarray]] = []
    full_360_rows: list[np.ndarray] = []
    for kernel in config.geometry_kernels:
        regions = _kernel_regions(
            kernel,
            reference_depth=reference_depth,
            geometry=geometry,
            config=config,
        )
        metrics = _aggregate_kernel(
            regions,
            context=sorted_context,
            weighted=kernel == "triangular_midpoint_weighted",
        )
        per_kernel.append({**regions, **metrics})
        full_360_rows.append(_full_360_fraction(regions, context=sorted_context))

    arrays = _stack_kernel_arrays(
        per_kernel,
        kernels=config.geometry_kernels,
        reference_depth=reference_depth,
        geometry=geometry,
        config=config,
        full_360_rows=full_360_rows,
        raw_source=raw_source,
    )
    errors.extend(_collapse_errors(arrays))
    report = _build_report(
        arrays,
        raw_source=raw_source,
        geometry=geometry,
        config=config,
        inputs=inputs or {},
        output_npz=output_npz,
        warnings=warnings,
        errors=errors,
    )
    return arrays, report


def write_geometry_aware_regression_label_outputs(
    arrays: dict[str, np.ndarray],
    report: GeometryAwareRegressionLabelReport,
    *,
    output_npz: Path,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_geometry_aware_regression_markdown(report), encoding="utf-8")


def format_geometry_aware_regression_markdown(
    report: GeometryAwareRegressionLabelReport,
) -> str:
    lines = [
        "# Geometry-Aware Continuous CAST Regression Weak Labels",
        "",
        "This is an MVP-4B-GR weak-label candidate artifact. The primary target is "
        "a continuous CAST channel-like cell fraction, not a final binary label. "
        "Derived binary views are sanity checks only.",
        "",
        f"- label_version: `{report.label_version}`",
        f"- primary_target: `{report.primary_target}`",
        f"- primary_target_layer: `{report.primary_target_layer}`",
        f"- raw_zc_available: `{report.raw_zc_available}`",
        f"- geometry_sign_status: `{report.geometry_sign_status}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Receiver-Level Primary Target",
        "",
        "These rows summarize `weighted_channel_fraction_zc_lt_2p5` at the "
        "receiver level with shape `[kernel, depth, receiver]`. They are not "
        "the depth-level audit view used by Stage 10 unless explicitly aggregated.",
        "",
    ]
    for summary in report.kernel_summaries:
        lines.append(
            "- "
            f"{summary['geometry_kernel']}: zero={summary['zero_fraction']}, "
            f"nonzero={summary['nonzero_fraction']}, "
            f"mean={summary['distribution']['mean']}, "
            f"median={summary['distribution']['median']}, "
            f"p75={summary['distribution']['p75']}, "
            f"p90={summary['distribution']['p90']}, "
            f"p95={summary['distribution']['p95']}, "
            f"max={summary['distribution']['max']}"
        )
    by_view = _summaries_by_view(report.target_view_summaries)
    for view in report.depth_level_target_views:
        lines.extend(["", f"## Depth-Level View: {view}", ""])
        lines.append(report.target_view_definitions.get(view, "No definition recorded."))
        lines.append("")
        for summary in by_view.get(view, []):
            lines.append(
                "- "
                f"{summary['geometry_kernel']}: zero={summary['zero_fraction']}, "
                f"nonzero={summary['nonzero_fraction']}, "
                f"mean={summary['distribution']['mean']}, "
                f"median={summary['distribution']['median']}, "
                f"p75={summary['distribution']['p75']}, "
                f"p90={summary['distribution']['p90']}, "
                f"p95={summary['distribution']['p95']}, "
                f"max={summary['distribution']['max']}, "
                f"fraction_gt_0p01={summary['fraction_gt_threshold']['0p01']}, "
                f"fraction_gt_0p05={summary['fraction_gt_threshold']['0p05']}, "
                f"fraction_gt_0p10={summary['fraction_gt_threshold']['0p10']}"
            )
    lines.extend(["", "## Raw Zc Source", ""])
    lines.extend(_dict_lines(report.raw_zc_source))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class CastContext:
    depth: np.ndarray
    zc: np.ndarray
    relative_drop: np.ndarray | None
    finite: np.ndarray
    raw_candidate: np.ndarray
    relative_candidate: np.ndarray | None
    combined_candidate: np.ndarray
    finite_count_by_depth: np.ndarray
    raw_count_by_depth: np.ndarray
    relative_count_by_depth: np.ndarray | None
    combined_count_by_depth: np.ndarray
    finite_count_cumsum: np.ndarray
    raw_count_cumsum: np.ndarray
    relative_count_cumsum: np.ndarray | None
    combined_count_cumsum: np.ndarray
    finite_azimuth_cumsum: np.ndarray
    combined_azimuth_cumsum: np.ndarray


def _build_cast_context(
    *,
    raw_source: CastZcSource,
    relative_drop: np.ndarray | None,
    zc_threshold: float,
    relative_drop_threshold: float,
) -> CastContext:
    order = np.argsort(raw_source.cast_depth)
    zc = raw_source.cast_zc[order].astype(np.float32)
    rel = None if relative_drop is None else relative_drop[order].astype(np.float32)
    finite = np.isfinite(zc)
    raw_candidate = finite & (zc < zc_threshold)
    if rel is None:
        relative_candidate = None
        combined = raw_candidate
    else:
        relative_candidate = np.isfinite(rel) & (rel >= relative_drop_threshold)
        combined = raw_candidate | relative_candidate
    finite_count = np.count_nonzero(finite, axis=1).astype(np.int64)
    raw_count = np.count_nonzero(raw_candidate, axis=1).astype(np.int64)
    relative_count = (
        None
        if relative_candidate is None
        else np.count_nonzero(relative_candidate & finite, axis=1).astype(np.int64)
    )
    combined_count = np.count_nonzero(combined & finite, axis=1).astype(np.int64)
    return CastContext(
        depth=raw_source.cast_depth[order].astype(np.float32),
        zc=zc,
        relative_drop=rel,
        finite=finite,
        raw_candidate=raw_candidate,
        relative_candidate=relative_candidate,
        combined_candidate=combined & finite,
        finite_count_by_depth=finite_count,
        raw_count_by_depth=raw_count,
        relative_count_by_depth=relative_count,
        combined_count_by_depth=combined_count,
        finite_count_cumsum=_count_cumsum(finite_count),
        raw_count_cumsum=_count_cumsum(raw_count),
        relative_count_cumsum=None if relative_count is None else _count_cumsum(relative_count),
        combined_count_cumsum=_count_cumsum(combined_count),
        finite_azimuth_cumsum=_azimuth_cumsum(finite),
        combined_azimuth_cumsum=_azimuth_cumsum(combined & finite),
    )


def _kernel_regions(
    kernel: str,
    *,
    reference_depth: np.ndarray,
    geometry: ReceiverGeometry,
    config: GeometryAwareRegressionConfig,
) -> dict[str, np.ndarray]:
    if geometry.audit_signs != (-1,):
        raise ValueError("Geometry-aware regression labels require human-confirmed sign=-1.")
    sign = -1
    table = geometry.geometry_table(
        reference_depth,
        sign=sign,
        interval_padding_ft=config.interval_padding_ft,
    )
    source = table["source_depth"][:, None].repeat(geometry.receiver_count, axis=1)
    receiver = table["receiver_depths"]
    midpoint = table["midpoint_depths"]
    if kernel == "r7_reference_point":
        lower = reference_depth[:, None].repeat(geometry.receiver_count, axis=1)
        upper = lower.copy()
    elif kernel == "midpoint_window":
        lower = midpoint - config.midpoint_window_half_width_ft
        upper = midpoint + config.midpoint_window_half_width_ft
    elif kernel in {"uniform_source_receiver_interval", "triangular_midpoint_weighted"}:
        lower = table["interval_min_depths"]
        upper = table["interval_max_depths"]
    else:
        raise ValueError(f"Unsupported geometry kernel: {kernel}")
    return {
        "reference_depth": reference_depth.astype(np.float32),
        "source_depth": source.astype(np.float32),
        "receiver_depth": receiver.astype(np.float32),
        "midpoint_depth": midpoint.astype(np.float32),
        "interval_min_depth": np.minimum(lower, upper).astype(np.float32),
        "interval_max_depth": np.maximum(lower, upper).astype(np.float32),
    }


def _aggregate_kernel(
    regions: dict[str, np.ndarray],
    *,
    context: CastContext,
    weighted: bool,
) -> dict[str, np.ndarray]:
    shape = regions["receiver_depth"].shape
    output = _empty_metric_arrays(shape)
    cache: dict[tuple[int, int, bool, float], dict[str, float | int]] = {}
    for index in np.ndindex(shape):
        lower = float(regions["interval_min_depth"][index])
        upper = float(regions["interval_max_depth"][index])
        midpoint = float(regions["midpoint_depth"][index])
        start, stop = _depth_region_bounds(context.depth, lower, upper)
        cache_key = (
            start,
            stop,
            weighted,
            round(midpoint, 6) if weighted else 0.0,
        )
        if cache_key in cache:
            metrics = cache[cache_key]
        else:
            metrics = _aggregate_region_bounds(
                start,
                stop,
                lower,
                upper,
                midpoint=midpoint,
                context=context,
                weighted=weighted,
            )
            cache[cache_key] = metrics
        for key, value in metrics.items():
            output[key][index] = value
    return output


def _aggregate_region(
    lower: float,
    upper: float,
    *,
    midpoint: float,
    context: CastContext,
    weighted: bool,
) -> dict[str, float | int]:
    start, stop = _depth_region_bounds(context.depth, lower, upper)
    return _aggregate_region_bounds(
        start,
        stop,
        lower,
        upper,
        midpoint=midpoint,
        context=context,
        weighted=weighted,
    )


def _aggregate_region_bounds(
    start: int,
    stop: int,
    lower: float,
    upper: float,
    *,
    midpoint: float,
    context: CastContext,
    weighted: bool,
) -> dict[str, float | int]:
    row_count = stop - start
    possible_cell_count = int(row_count * context.zc.shape[1])
    if possible_cell_count == 0:
        return _empty_region_metrics()
    total = _range_sum(context.finite_count_cumsum, start, stop)
    if total == 0:
        return {**_empty_region_metrics(), "total_cell_count": 0}
    raw_count = _range_sum(context.raw_count_cumsum, start, stop)
    relative_count = (
        0
        if context.relative_count_cumsum is None
        else _range_sum(context.relative_count_cumsum, start, stop)
    )
    combined_count = _range_sum(context.combined_count_cumsum, start, stop)
    weighted_fraction = (
        _weighted_raw_fraction(
            start,
            stop,
            lower,
            upper,
            midpoint=midpoint,
            context=context,
        )
        if weighted
        else float(raw_count / total)
    )
    zc = context.zc[start:stop]
    finite = context.finite[start:stop]
    zc_finite = zc[finite]
    relative_drop = None if context.relative_drop is None else context.relative_drop[start:stop]
    max_relative_drop = np.nan
    if relative_drop is not None:
        rel_values = relative_drop[np.isfinite(relative_drop)]
        if rel_values.size:
            max_relative_drop = float(np.max(rel_values))
    combined = context.combined_candidate[start:stop]
    return {
        "raw_channel_fraction_zc_lt_2p5": float(raw_count / total),
        "weighted_channel_fraction_zc_lt_2p5": weighted_fraction,
        "relative_anomaly_fraction": float(relative_count / total),
        "combined_channel_fraction": float(combined_count / total),
        "min_zc": float(np.min(zc_finite)),
        "p05_zc": float(np.percentile(zc_finite, 5.0)),
        "p10_zc": float(np.percentile(zc_finite, 10.0)),
        "max_relative_drop": max_relative_drop,
        "candidate_cell_count": raw_count,
        "total_cell_count": total,
        "largest_connected_component_fraction": _largest_component_fraction(
            combined,
            finite,
            total=total,
        ),
        "max_azimuth_channel_fraction": _max_azimuth_fraction_from_cumsum(
            context,
            start,
            stop,
        ),
        "depth_label_confidence": float(total / possible_cell_count),
    }


def _weighted_raw_fraction(
    start: int,
    stop: int,
    lower: float,
    upper: float,
    *,
    midpoint: float,
    context: CastContext,
) -> float:
    depth_subset = context.depth[start:stop]
    weights = _depth_weights(
        depth_subset,
        lower=lower,
        upper=upper,
        midpoint=midpoint,
    )
    denominator = float(np.sum(weights * context.finite_count_by_depth[start:stop]))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(weights * context.raw_count_by_depth[start:stop]))
    return numerator / denominator


def _depth_weights(
    depth: np.ndarray,
    *,
    lower: float,
    upper: float,
    midpoint: float,
) -> np.ndarray:
    half_width = max(abs(upper - midpoint), abs(midpoint - lower), 1e-6)
    depth_weights = 1.0 - np.abs(depth.astype(np.float32) - midpoint) / half_width
    return np.clip(depth_weights, 0.0, 1.0).astype(np.float32)


def _max_azimuth_fraction_from_cumsum(
    context: CastContext,
    start: int,
    stop: int,
) -> float:
    denominator = context.finite_azimuth_cumsum[stop] - context.finite_azimuth_cumsum[start]
    numerator = (
        context.combined_azimuth_cumsum[stop] - context.combined_azimuth_cumsum[start]
    )
    fractions = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )
    return float(np.max(fractions)) if fractions.size else 0.0


def _range_sum(cumsum: np.ndarray, start: int, stop: int) -> int:
    return int(cumsum[stop] - cumsum[start])


def _count_cumsum(counts: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    )


def _azimuth_cumsum(mask: np.ndarray) -> np.ndarray:
    leading = np.zeros((1, mask.shape[1]), dtype=np.int64)
    return np.vstack([leading, np.cumsum(mask.astype(np.int64), axis=0)])


def _depth_region_bounds(depth: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    region_slice = _depth_region_slice(depth, lower, upper)
    return int(region_slice.start or 0), int(region_slice.stop or 0)


def _depth_region_slice(depth: np.ndarray, lower: float, upper: float) -> slice:
    if depth.size == 0:
        return slice(0, 0)
    if np.isclose(lower, upper):
        insert = int(np.searchsorted(depth, lower, side="left"))
        if insert <= 0:
            nearest = 0
        elif insert >= depth.size:
            nearest = depth.size - 1
        else:
            before = insert - 1
            after = insert
            nearest = (
                before
                if abs(float(depth[before]) - lower) <= abs(float(depth[after]) - lower)
                else after
            )
        return slice(nearest, nearest + 1)
    return slice(
        int(np.searchsorted(depth, min(lower, upper), side="left")),
        int(np.searchsorted(depth, max(lower, upper), side="right")),
    )


def _full_360_fraction(regions: dict[str, np.ndarray], *, context: CastContext) -> np.ndarray:
    lower = np.min(regions["interval_min_depth"], axis=1)
    upper = np.max(regions["interval_max_depth"], axis=1)
    output = np.zeros(lower.size, dtype=np.float32)
    for index in range(lower.size):
        metrics = _aggregate_region(
            float(lower[index]),
            float(upper[index]),
            midpoint=float(0.5 * (lower[index] + upper[index])),
            context=context,
            weighted=False,
        )
        output[index] = float(metrics["combined_channel_fraction"])
    return output


def _stack_kernel_arrays(
    per_kernel: list[dict[str, np.ndarray]],
    *,
    kernels: tuple[str, ...],
    reference_depth: np.ndarray,
    geometry: ReceiverGeometry,
    config: GeometryAwareRegressionConfig,
    full_360_rows: list[np.ndarray],
    raw_source: CastZcSource,
) -> dict[str, np.ndarray]:
    stack_keys = tuple(per_kernel[0].keys())
    arrays = {
        key: np.stack([kernel_arrays[key] for kernel_arrays in per_kernel], axis=0)
        for key in stack_keys
    }
    primary = arrays["weighted_channel_fraction_zc_lt_2p5"]
    arrays.update(
        {
            "depth": reference_depth.astype(np.float32),
            "geometry_kernel": np.asarray(kernels),
            "receiver_index": np.arange(1, geometry.receiver_count + 1, dtype=np.int16),
            "receiver_mean": np.mean(primary, axis=2).astype(np.float32),
            "receiver_max": np.max(primary, axis=2).astype(np.float32),
            "receiver_p90": np.percentile(primary, 90.0, axis=2).astype(np.float32),
            "receiver_std": np.std(primary, axis=2).astype(np.float32),
            "full_360_fraction": np.stack(full_360_rows, axis=0).astype(np.float32),
            "raw_zc_source_file": np.asarray(raw_source.report.source_file),
            "raw_zc_source_field": np.asarray(raw_source.report.source_field),
            "raw_zc_finite_ratio": np.asarray(raw_source.report.finite_ratio, dtype=np.float32),
            "depth_axis_sign": np.asarray(-1, dtype=np.int8),
            "sign_convention_status": np.asarray(geometry.sign_convention_status),
            "label_version": np.asarray(GEOMETRY_AWARE_REGRESSION_LABEL_VERSION),
            "primary_target": np.asarray(config.primary_target),
            "zc_threshold_mrayl": np.asarray(config.zc_threshold_mrayl, dtype=np.float32),
            "relative_drop_threshold": np.asarray(
                config.relative_drop_threshold,
                dtype=np.float32,
            ),
            "zc_threshold_status": np.asarray(config.zc_threshold_status),
            "no_model_training": np.asarray(True),
            "no_final_labels": np.asarray(True),
            "no_stc": np.asarray(True),
            "no_apes": np.asarray(True),
            "no_deep_learning": np.asarray(True),
            "no_mvp4c": np.asarray(True),
        }
    )
    for threshold in config.derived_positive_thresholds:
        suffix = f"{threshold:.2f}".replace("0.", "0p")
        arrays[f"derived_positive_at_fraction_{suffix}"] = (
            arrays["receiver_max"] >= threshold
        )
    return arrays


def _build_report(
    arrays: dict[str, np.ndarray],
    *,
    raw_source: CastZcSource,
    geometry: ReceiverGeometry,
    config: GeometryAwareRegressionConfig,
    inputs: dict[str, str],
    output_npz: Path | None,
    warnings: list[str],
    errors: list[str],
) -> GeometryAwareRegressionLabelReport:
    summaries = _receiver_level_primary_summaries(arrays)
    target_view_summaries = _target_view_summaries(
        arrays,
        thresholds=config.derived_positive_thresholds,
    )
    raw_source_dict = raw_source.report.to_dict()
    return _report_from_summaries(
        primary_target=config.primary_target,
        raw_zc_source=raw_source_dict,
        geometry_sign_status={
            "depth_axis_sign": -1,
            "sign_convention_status": geometry.sign_convention_status,
            "depth_increases_toward": geometry.depth_increases_toward,
            "sample_index_direction": geometry.sample_index_direction,
            "receiver_index_direction": geometry.receiver_index_direction,
            "source_position": geometry.source_position,
            "relbearing_plus_minus_independent": True,
        },
        kernel_summaries=summaries,
        target_view_summaries=target_view_summaries,
        inputs=inputs,
        output_npz=output_npz,
        warnings=[*raw_source.report.warnings, *warnings],
        errors=errors,
    )


def build_geometry_aware_regression_label_report_from_arrays(
    *,
    arrays: dict[str, np.ndarray],
    previous_report: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    output_npz: Path | None = None,
    thresholds: Sequence[float] = (0.01, 0.05, 0.10),
) -> GeometryAwareRegressionLabelReport:
    previous = previous_report or {}
    raw_source = _as_dict(previous.get("raw_zc_source"))
    if not raw_source:
        raw_source = {
            "source_file": _scalar_string(arrays.get("raw_zc_source_file")),
            "source_field": _scalar_string(arrays.get("raw_zc_source_field")),
            "finite_ratio": _scalar_float(arrays.get("raw_zc_finite_ratio")),
        }
    geometry_sign = _as_dict(previous.get("geometry_sign_status"))
    if not geometry_sign:
        geometry_sign = {
            "depth_axis_sign": int(np.asarray(arrays.get("depth_axis_sign", -1))),
            "sign_convention_status": _scalar_string(
                arrays.get("sign_convention_status"),
                default="human_confirmed",
            ),
            "relbearing_plus_minus_independent": True,
        }
    return _report_from_summaries(
        primary_target=_scalar_string(
            arrays.get("primary_target"),
            default="weighted_channel_fraction_zc_lt_2p5",
        ),
        raw_zc_source=raw_source,
        geometry_sign_status=geometry_sign,
        kernel_summaries=_receiver_level_primary_summaries(arrays),
        target_view_summaries=_target_view_summaries(arrays, thresholds=thresholds),
        inputs=inputs or _as_dict(previous.get("inputs")),
        output_npz=output_npz or Path(str(previous.get("output_npz", ""))),
        warnings=list(previous.get("warnings", [])),
        errors=list(previous.get("errors", [])),
    )


def _receiver_level_primary_summaries(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    summaries = []
    for index, kernel in enumerate(arrays["geometry_kernel"].astype(str)):
        target = arrays["weighted_channel_fraction_zc_lt_2p5"][index]
        summaries.append(
            {
                "geometry_kernel": kernel,
                "target_view": "weighted_channel_fraction_zc_lt_2p5",
                "semantic_layer": "receiver-level",
                "shape": list(target.shape),
                "sample_count": int(target.size),
                "zero_fraction": _fraction_values(target <= 0.0),
                "nonzero_fraction": _fraction_values(target > 0.0),
                "distribution": _numeric_distribution(target),
                "candidate_cell_count": int(np.sum(arrays["candidate_cell_count"][index])),
                "total_cell_count": int(np.sum(arrays["total_cell_count"][index])),
            }
        )
    return summaries


def _target_view_summaries(
    arrays: dict[str, np.ndarray],
    *,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for view in DEPTH_LEVEL_TARGET_VIEWS:
        if view not in arrays:
            continue
        values_by_kernel = np.asarray(arrays[view], dtype=np.float32)
        for index, kernel in enumerate(arrays["geometry_kernel"].astype(str)):
            target = values_by_kernel[index]
            summaries.append(
                {
                    "geometry_kernel": kernel,
                    "target_view": view,
                    "semantic_layer": "depth-level",
                    "shape": list(target.shape),
                    "sample_count": int(target.size),
                    "zero_fraction": _fraction_values(target <= 0.0),
                    "nonzero_fraction": _fraction_values(target > 0.0),
                    "distribution": _numeric_distribution(target),
                    "fraction_gt_threshold": {
                        _threshold_key(threshold): _fraction_values(target > threshold)
                        for threshold in thresholds
                    },
                }
            )
    return summaries


def _report_from_summaries(
    *,
    primary_target: str,
    raw_zc_source: dict[str, Any],
    geometry_sign_status: dict[str, Any],
    kernel_summaries: list[dict[str, Any]],
    target_view_summaries: list[dict[str, Any]],
    inputs: dict[str, str],
    output_npz: Path | None,
    warnings: list[str],
    errors: list[str],
) -> GeometryAwareRegressionLabelReport:
    return GeometryAwareRegressionLabelReport(
        report_version=GEOMETRY_AWARE_REGRESSION_REPORT_VERSION,
        label_version=GEOMETRY_AWARE_REGRESSION_LABEL_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        output_npz=str(output_npz) if output_npz else "",
        primary_target=primary_target,
        raw_zc_available=True,
        raw_zc_source=raw_zc_source,
        geometry_sign_status=geometry_sign_status,
        primary_target_layer="receiver-level",
        depth_level_target_views=list(DEPTH_LEVEL_TARGET_VIEWS),
        target_view_definitions=dict(TARGET_VIEW_DEFINITIONS),
        kernel_summaries=kernel_summaries,
        target_view_summaries=target_view_summaries,
        warnings=warnings,
        errors=errors,
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        not_performed=[
            "binary final label generation",
            "ground truth claim",
            "raw MAT reading",
            "raw waveform reading",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
            "model training",
        ],
    )


def _relative_drop_array(
    baseline_arrays: dict[str, np.ndarray] | None,
    *,
    raw_source: CastZcSource,
    warnings: list[str],
) -> np.ndarray | None:
    if baseline_arrays is None:
        warnings.append("CAST relative_drop baseline is unavailable; using raw-Zc target only.")
        return None
    if "relative_drop" not in baseline_arrays:
        warnings.append("CAST baseline NPZ has no relative_drop; using raw-Zc target only.")
        return None
    relative_drop = np.asarray(baseline_arrays["relative_drop"], dtype=np.float32)
    if relative_drop.shape != raw_source.cast_zc.shape:
        warnings.append("CAST relative_drop shape does not match raw Zc; ignoring it.")
        return None
    if "cast_depth" in baseline_arrays and not np.allclose(
        np.asarray(baseline_arrays["cast_depth"], dtype=np.float32).reshape(-1),
        raw_source.cast_depth,
        atol=1e-3,
    ):
        warnings.append("CAST baseline depth grid differs from raw Zc source; ignoring it.")
        return None
    return relative_drop


def _empty_metric_arrays(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    float_keys = (
        "raw_channel_fraction_zc_lt_2p5",
        "weighted_channel_fraction_zc_lt_2p5",
        "relative_anomaly_fraction",
        "combined_channel_fraction",
        "min_zc",
        "p05_zc",
        "p10_zc",
        "max_relative_drop",
        "largest_connected_component_fraction",
        "max_azimuth_channel_fraction",
        "depth_label_confidence",
    )
    output = {key: np.full(shape, np.nan, dtype=np.float32) for key in float_keys}
    output["candidate_cell_count"] = np.zeros(shape, dtype=np.int32)
    output["total_cell_count"] = np.zeros(shape, dtype=np.int32)
    return output


def _empty_region_metrics() -> dict[str, float | int]:
    return {
        "raw_channel_fraction_zc_lt_2p5": 0.0,
        "weighted_channel_fraction_zc_lt_2p5": 0.0,
        "relative_anomaly_fraction": 0.0,
        "combined_channel_fraction": 0.0,
        "min_zc": np.nan,
        "p05_zc": np.nan,
        "p10_zc": np.nan,
        "max_relative_drop": np.nan,
        "candidate_cell_count": 0,
        "total_cell_count": 0,
        "largest_connected_component_fraction": 0.0,
        "max_azimuth_channel_fraction": 0.0,
        "depth_label_confidence": 0.0,
    }


def _largest_component_fraction(
    candidate: np.ndarray,
    finite: np.ndarray,
    *,
    total: int,
) -> float:
    valid_candidate = candidate & finite
    if total == 0 or not np.any(valid_candidate):
        return 0.0
    azimuth_count = valid_candidate.shape[1]
    tiled = np.concatenate([valid_candidate, valid_candidate, valid_candidate], axis=1)
    structure = np.asarray(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    labels, _count = connected_label(tiled, structure=structure)
    center = labels[:, azimuth_count : 2 * azimuth_count].astype(np.int32)
    component_ids = np.unique(center[center > 0])
    if component_ids.size == 0:
        return 0.0
    parent = {int(component_id): int(component_id) for component_id in component_ids}
    for row in range(center.shape[0]):
        left_id = int(center[row, 0])
        right_id = int(center[row, -1])
        if left_id > 0 and right_id > 0:
            _union_parent(parent, left_id, right_id)
    label_values = center[center > 0]
    lookup = np.arange(int(np.max(label_values)) + 1, dtype=np.int32)
    for component_id in component_ids:
        lookup[int(component_id)] = _find_parent(parent, int(component_id))
    counts = np.bincount(lookup[label_values])
    return float(np.max(counts) / total)


def _find_parent(parent: dict[int, int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union_parent(parent: dict[int, int], left: int, right: int) -> None:
    left_root = _find_parent(parent, left)
    right_root = _find_parent(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _collapse_errors(arrays: dict[str, np.ndarray]) -> list[str]:
    errors: list[str] = []
    for index, kernel in enumerate(arrays["geometry_kernel"].astype(str)):
        values = arrays["weighted_channel_fraction_zc_lt_2p5"][index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            errors.append(f"{kernel} has no finite continuous regression targets.")
        elif np.nanstd(finite) <= 1e-8:
            errors.append(f"{kernel} continuous regression targets collapsed to a constant.")
    return errors


def _validate_guardrails(
    feature_arrays: dict[str, np.ndarray],
    geometry: ReceiverGeometry,
    config: GeometryAwareRegressionConfig,
    errors: list[str],
) -> None:
    if "depth" not in feature_arrays:
        raise KeyError("Depth-level XSI feature NPZ missing depth.")
    if "depth_level_xsi_features" not in feature_arrays:
        raise KeyError("Depth-level XSI feature NPZ missing depth_level_xsi_features.")
    if not bool(np.asarray(feature_arrays.get("no_final_labels", True))):
        errors.append("Depth-level XSI features must preserve no_final_labels=true.")
    if not config.no_final_labels:
        errors.append("Regression label config must preserve labels.no_final_labels=true.")
    if geometry.audit_signs != (-1,) or geometry.sign_convention_status != "human_confirmed":
        errors.append("Geometry config must be human_confirmed with depth_axis_sign=-1.")


def _numeric_distribution(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _fraction_values(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".replace("0.", "0p")


def _summaries_by_view(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("target_view")), []).append(row)
    return grouped


def _scalar_string(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array.reshape(-1)[0]) if array.size else default


def _scalar_float(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    result = float(array.reshape(-1)[0])
    return result if np.isfinite(result) else None


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _load_npz(path: Path | str | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
