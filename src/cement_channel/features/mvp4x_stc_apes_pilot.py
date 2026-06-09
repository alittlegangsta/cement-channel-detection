from __future__ import annotations

import csv
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.data.manifest import load_paths_config
from cement_channel.data.small_slice_reader import (
    MatReadRequest,
    load_mapping_config,
    read_mat_file_slices,
)

SA_PILOT_VERSION = "mvp4x_sa_pilot_auto_v001"
SA_PILOT_REPORT_VERSION = "mvp4x_sa_pilot_report_v001"
MIN_INTERVALS = 80
MAX_INTERVALS = 160
DEFAULT_INTERVALS = 120
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
    "no_raw_mat_modified": True,
    "no_full_well_stc": True,
    "no_full_well_apes": True,
    "no_deep_learning": True,
}


@dataclass(frozen=True)
class SaPilotOutputs:
    output_dir: str
    report_json: str
    report_md: str
    interval_csv: str
    feature_npz: str


@dataclass(frozen=True)
class SaPilotReport:
    report_version: str
    pilot_version: str
    generated_at: str
    inputs: dict[str, str]
    outputs: SaPilotOutputs
    interval_count: int
    requested_interval_count: int
    interval_bounds: dict[str, int]
    interval_selection_policy: dict[str, Any]
    waveform_read_policy: dict[str, Any]
    stc_policy: dict[str, Any]
    apes_policy: dict[str, Any]
    resource_micro_benchmark: dict[str, Any]
    runtime_seconds: float
    peak_memory_bytes: int
    estimated_waveform_bytes_read: int
    skipped_interval_count: int
    warnings: list[str]
    errors: list[str]
    summary_stats: dict[str, Any]
    research_only: bool
    exploratory_only: bool
    weak_label_target: bool
    no_final_labels: bool
    no_ground_truth_claim: bool
    no_production_claim: bool
    not_validated_for_deployment: bool
    no_raw_mat_modified: bool
    no_full_well_stc: bool
    no_full_well_apes: bool
    no_deep_learning: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outputs"] = asdict(self.outputs)
        return data


def run_sa_pilot_from_paths(
    *,
    paths_config: Path | str,
    mapping_path: Path | str,
    snapshot_npz: Path | str,
    output_dir: Path | str | None = None,
    interval_count: int = DEFAULT_INTERVALS,
    interval_half_window_ft: float = 2.5,
    chunk_depth_samples: int = 8,
    max_time_samples: int = 1024,
    micro_benchmark_intervals: int = 8,
    receiver_limit: int = 13,
    side_limit: int = 8,
    overwrite: bool = False,
) -> SaPilotReport:
    _validate_interval_count(interval_count)
    paths = load_paths_config(paths_config)
    mapping = load_mapping_config(mapping_path)
    snapshot = _load_npz(snapshot_npz)
    out_dir = Path(output_dir) if output_dir is not None else default_output_dir(paths)
    report = run_sa_pilot(
        paths=paths,
        mapping=mapping,
        snapshot=snapshot,
        output_dir=out_dir,
        inputs={
            "paths_config": str(paths_config),
            "mapping_path": str(mapping_path),
            "snapshot_npz": str(snapshot_npz),
        },
        interval_count=interval_count,
        interval_half_window_ft=interval_half_window_ft,
        chunk_depth_samples=chunk_depth_samples,
        max_time_samples=max_time_samples,
        micro_benchmark_intervals=micro_benchmark_intervals,
        receiver_limit=receiver_limit,
        side_limit=side_limit,
        overwrite=overwrite,
    )
    return report


def default_output_dir(paths: dict[str, Any]) -> Path:
    remote_run_dir = os.environ.get("CEMENT_REMOTE_RUN_DIR")
    if remote_run_dir:
        return Path(remote_run_dir) / "reports" / SA_PILOT_VERSION
    data = _as_dict(paths.get("data"))
    reports_root = data.get("reports")
    if not reports_root:
        raise ValueError("data.reports is not configured.")
    return Path(str(reports_root)) / SA_PILOT_VERSION


def run_sa_pilot(
    *,
    paths: dict[str, Any],
    mapping: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    output_dir: Path,
    inputs: dict[str, str],
    interval_count: int,
    interval_half_window_ft: float,
    chunk_depth_samples: int,
    max_time_samples: int,
    micro_benchmark_intervals: int,
    receiver_limit: int,
    side_limit: int,
    overwrite: bool,
) -> SaPilotReport:
    _validate_interval_count(interval_count)
    if chunk_depth_samples < 1 or chunk_depth_samples > 32:
        raise ValueError("chunk_depth_samples must be in [1, 32] for bounded pilot reads.")
    if max_time_samples < 32 or max_time_samples > 1024:
        raise ValueError("max_time_samples must be in [32, 1024].")
    if micro_benchmark_intervals < 1 or micro_benchmark_intervals > 16:
        raise ValueError("micro_benchmark_intervals must be in [1, 16].")
    if receiver_limit < 1 or receiver_limit > 13:
        raise ValueError("receiver_limit must be in [1, 13].")
    if side_limit < 1 or side_limit > 8:
        raise ValueError("side_limit must be in [1, 8].")

    started = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_snapshot_scope(snapshot)
    _ensure_output_dir(output_dir, overwrite=overwrite)
    raw_dir = Path(str(_as_dict(paths.get("data")).get("raw", "")))
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    xsi = _as_dict(mapping.get("xsi"))
    _validate_xsi_mapping(xsi)
    expected_receivers = int(xsi.get("expected_receiver_files", 13))
    receiver_count = min(expected_receivers, receiver_limit)
    side_labels = _as_str_list(xsi.get("side_labels"), list("ABCDEFGH"))[:side_limit]
    intervals = select_representative_intervals(
        snapshot,
        interval_count=interval_count,
        half_window_ft=interval_half_window_ft,
    )
    depth_vectors = load_xsi_depth_vectors(
        raw_dir=raw_dir,
        xsi_mapping=xsi,
        receiver_count=receiver_count,
    )

    rows: list[dict[str, Any]] = []
    stc_peaks: list[float] = []
    apes_peaks: list[float] = []
    center_depths: list[float] = []
    micro_rows: list[dict[str, Any]] = []
    estimated_bytes = 0
    peak_memory = _peak_rss_bytes()
    skipped = 0
    for ordinal, interval in enumerate(intervals, start=1):
        interval_started = time.perf_counter()
        try:
            chunk, read_info = read_interval_waveform_chunk(
                raw_dir=raw_dir,
                xsi_mapping=xsi,
                depth_vectors=depth_vectors,
                center_depth_ft=float(interval["depth_center"]),
                chunk_depth_samples=chunk_depth_samples,
                max_time_samples=max_time_samples,
                receiver_count=receiver_count,
                side_labels=side_labels,
            )
            estimated_bytes += int(chunk.nbytes)
            trace = np.nanmean(chunk, axis=0)
            stc = compute_relative_stc_summary(trace, side_labels=side_labels)
            apes = compute_apes_proxy_summary(trace, side_labels=side_labels)
            row = {
                **interval,
                "status": "succeeded",
                "receiver_count_read": int(chunk.shape[1]),
                "side_count_read": int(chunk.shape[2]),
                "chunk_depth_samples": int(chunk.shape[0]),
                "time_samples": int(chunk.shape[3]),
                "stc_peak_coherence": float(stc["peak_coherence"]),
                "stc_peak_side": str(stc["peak_side"]),
                "stc_peak_delay_samples_per_receiver": float(
                    stc["peak_delay_samples_per_receiver"]
                ),
                "apes_peak_power": float(apes["peak_power"]),
                "apes_peak_side": str(apes["peak_side"]),
                "apes_peak_frequency_cycles_per_sample": float(apes["peak_frequency"]),
                "read_start_indices_json": json.dumps(read_info["source_start_indices"]),
                "runtime_seconds": float(time.perf_counter() - interval_started),
            }
            stc_peaks.append(float(stc["peak_coherence"]))
            apes_peaks.append(float(apes["peak_power"]))
            center_depths.append(float(interval["depth_center"]))
        except Exception as exc:
            skipped += 1
            warning = f"{interval['interval_id']}: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            row = {
                **interval,
                "status": "failed_read_or_compute",
                "receiver_count_read": 0,
                "side_count_read": 0,
                "chunk_depth_samples": chunk_depth_samples,
                "time_samples": max_time_samples,
                "stc_peak_coherence": np.nan,
                "stc_peak_side": "",
                "stc_peak_delay_samples_per_receiver": np.nan,
                "apes_peak_power": np.nan,
                "apes_peak_side": "",
                "apes_peak_frequency_cycles_per_sample": np.nan,
                "read_start_indices_json": "{}",
                "runtime_seconds": float(time.perf_counter() - interval_started),
            }
        rows.append(row)
        if ordinal <= micro_benchmark_intervals:
            micro_rows.append(
                {
                    "interval_id": row["interval_id"],
                    "status": row["status"],
                    "runtime_seconds": row["runtime_seconds"],
                    "estimated_waveform_bytes": int(
                        chunk_depth_samples
                        * receiver_count
                        * len(side_labels)
                        * max_time_samples
                        * np.dtype(np.float32).itemsize
                    ),
                }
            )
        peak_memory = max(peak_memory, _peak_rss_bytes())

    if skipped == len(intervals):
        errors.append("All selected intervals failed to read or compute.")

    outputs = _write_outputs(
        output_dir=output_dir,
        rows=rows,
        center_depths=np.asarray(center_depths, dtype=np.float32),
        stc_peaks=np.asarray(stc_peaks, dtype=np.float32),
        apes_peaks=np.asarray(apes_peaks, dtype=np.float32),
        overwrite=overwrite,
    )
    runtime = time.perf_counter() - started
    report = SaPilotReport(
        report_version=SA_PILOT_REPORT_VERSION,
        pilot_version=SA_PILOT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        outputs=outputs,
        interval_count=len(intervals),
        requested_interval_count=interval_count,
        interval_bounds={"min": MIN_INTERVALS, "max": MAX_INTERVALS},
        interval_selection_policy={
            "source": "mvp4x_research_snapshot_v001",
            "representative_interval_count": interval_count,
            "depth_dedupe_rounding_ft": 0.1,
            "half_window_ft": interval_half_window_ft,
            "no_full_well_scan": True,
        },
        waveform_read_policy={
            "selected_interval_chunked_reads_only": True,
            "chunk_depth_samples": chunk_depth_samples,
            "max_time_samples": max_time_samples,
            "receiver_count": receiver_count,
            "side_count": len(side_labels),
            "raw_mat_read_only": True,
        },
        stc_policy={
            "method": "bounded_relative_delay_semblance_proxy",
            "delay_unit": "samples_per_receiver",
            "not_full_well_stc_map": True,
        },
        apes_policy={
            "method": "bounded_regularized_adaptive_spectrum_proxy",
            "frequency_unit": "cycles_per_sample",
            "not_full_well_apes_map": True,
        },
        resource_micro_benchmark=_micro_benchmark_summary(micro_rows),
        runtime_seconds=float(runtime),
        peak_memory_bytes=peak_memory,
        estimated_waveform_bytes_read=estimated_bytes,
        skipped_interval_count=skipped,
        warnings=warnings,
        errors=errors,
        summary_stats={
            "stc_peak_coherence": _summary_stats(np.asarray(stc_peaks, dtype=np.float64)),
            "apes_peak_power": _summary_stats(np.asarray(apes_peaks, dtype=np.float64)),
        },
        research_only=True,
        exploratory_only=True,
        weak_label_target=True,
        no_final_labels=True,
        no_ground_truth_claim=True,
        no_production_claim=True,
        not_validated_for_deployment=True,
        no_raw_mat_modified=True,
        no_full_well_stc=True,
        no_full_well_apes=True,
        no_deep_learning=True,
    )
    _rewrite_report_outputs(report, rows, overwrite=True)
    return report


def select_representative_intervals(
    snapshot: dict[str, np.ndarray],
    *,
    interval_count: int,
    half_window_ft: float,
) -> list[dict[str, Any]]:
    _validate_interval_count(interval_count)
    depth = np.asarray(snapshot["depth"], dtype=np.float64).reshape(-1)
    valid = np.isfinite(depth)
    if "no_overlap_flag" in snapshot:
        valid &= ~np.asarray(snapshot["no_overlap_flag"], dtype=bool).reshape(-1)
    if np.count_nonzero(valid) < interval_count:
        raise ValueError("Not enough valid snapshot rows for bounded SA pilot interval selection.")

    receiver_p90 = _metric(snapshot, "receiver_p90", depth.size)
    receiver_max = _metric(snapshot, "receiver_max", depth.size)
    full_360 = _metric(snapshot, "full_360_fraction", depth.size)
    orient = _metric(snapshot, "orientation_confidence", depth.size)
    morph_drop = _matrix_metric(snapshot, "morphology_max_relative_drop", depth.size)
    morph_component = _matrix_metric(
        snapshot,
        "morphology_largest_connected_component_fraction",
        depth.size,
    )
    categories = [
        ("high_receiver_p90", receiver_p90, True, max(interval_count // 10, 8), valid),
        ("low_receiver_p90", receiver_p90, False, max(interval_count // 12, 6), valid),
        ("high_receiver_max", receiver_max, True, max(interval_count // 12, 6), valid),
        ("high_full_360_fraction", full_360, True, max(interval_count // 10, 8), valid),
        ("low_full_360_fraction", full_360, False, max(interval_count // 16, 5), valid),
        ("high_morphology_drop", morph_drop, True, max(interval_count // 10, 8), valid),
        (
            "high_morphology_component",
            morph_component,
            True,
            max(interval_count // 12, 6),
            valid,
        ),
        ("low_orientation_confidence", orient, False, max(interval_count // 12, 6), valid),
        ("high_orientation_confidence", orient, True, max(interval_count // 12, 6), valid),
    ]
    broad_regime = np.asarray(snapshot.get("broad_regime_id", np.full(depth.size, "")), dtype=str)
    for regime in sorted({str(item) for item in broad_regime[valid] if str(item)}):
        regime_mask = valid & (broad_regime == regime)
        categories.append(
            (
                f"regime_{regime}_depth_stratified",
                depth,
                True,
                max(interval_count // 18, 4),
                regime_mask,
            )
        )
    if "any_special_flag" in snapshot:
        special = valid & np.asarray(snapshot["any_special_flag"], dtype=bool).reshape(-1)
        if np.any(special):
            categories.append(("special_band", depth, True, max(interval_count // 16, 5), special))

    selected: list[tuple[int, str]] = []
    used_indices: set[int] = set()
    used_depth_bins: set[int] = set()
    for category, metric, descending, count, mask in categories:
        for index in _pick_metric_quantiles(metric, mask=mask, count=count, descending=descending):
            if (
                _add_selected(
                    selected,
                    used_indices,
                    used_depth_bins,
                    int(index),
                    category,
                    depth,
                    enforce_depth_bin=True,
                )
                and len(selected) >= interval_count
            ):
                break
        if len(selected) >= interval_count:
            break

    fill_indices = _pick_metric_quantiles(
        depth, mask=valid, count=interval_count * 3, descending=True
    )
    for index in fill_indices:
        if (
            _add_selected(
                selected,
                used_indices,
                used_depth_bins,
                int(index),
                "depth_stratified_fill",
                depth,
                enforce_depth_bin=True,
            )
            and len(selected) >= interval_count
        ):
            break
    if len(selected) < interval_count:
        for index in np.where(valid)[0]:
            if (
                _add_selected(
                    selected,
                    used_indices,
                    used_depth_bins,
                    int(index),
                    "valid_fill",
                    depth,
                    enforce_depth_bin=False,
                )
                and len(selected) >= interval_count
            ):
                break
    if len(selected) != interval_count:
        raise ValueError(f"Selected {len(selected)} intervals, expected {interval_count}.")

    intervals: list[dict[str, Any]] = []
    for ordinal, (index, category) in enumerate(selected, start=1):
        center = float(depth[index])
        intervals.append(
            {
                "interval_id": f"SA{ordinal:03d}",
                "selection_category": category,
                "snapshot_index": int(index),
                "depth_center": center,
                "depth_min": center - half_window_ft,
                "depth_max": center + half_window_ft,
                "receiver_p90": float(receiver_p90[index]),
                "receiver_max": float(receiver_max[index]),
                "full_360_fraction": float(full_360[index]),
                "orientation_confidence": float(orient[index]),
                "morphology_max_relative_drop": float(morph_drop[index]),
                "morphology_component_fraction": float(morph_component[index]),
                "broad_regime_id": str(broad_regime[index]),
                "research_only": True,
                "no_final_labels": True,
            }
        )
    return intervals


def load_xsi_depth_vectors(
    *,
    raw_dir: Path,
    xsi_mapping: dict[str, Any],
    receiver_count: int,
) -> list[np.ndarray]:
    receiver_dir = raw_dir / str(xsi_mapping.get("receiver_dir", "XSILMR"))
    depth_pattern = str(xsi_mapping["depth_variable_pattern"])
    vectors: list[np.ndarray] = []
    for receiver_index in range(1, receiver_count + 1):
        receiver_file = receiver_dir / f"XSILMR{receiver_index:02d}.mat"
        request = MatReadRequest(
            variable_path=_format_pattern(depth_pattern, receiver_index),
            role="depth",
            source_orientation=_as_str_list(xsi_mapping.get("depth_source_shape_order"), ["depth"]),
            canonical_orientation=["depth"],
            max_depth_samples=10_000_000,
            max_time_samples=1,
            max_cast_azimuth=0,
            source_start_index=0,
        )
        data = read_mat_file_slices(receiver_file, [request])
        vectors.append(np.asarray(data[request.variable_path], dtype=np.float32).reshape(-1))
    return vectors


def read_interval_waveform_chunk(
    *,
    raw_dir: Path,
    xsi_mapping: dict[str, Any],
    depth_vectors: list[np.ndarray],
    center_depth_ft: float,
    chunk_depth_samples: int,
    max_time_samples: int,
    receiver_count: int,
    side_labels: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    receiver_dir = raw_dir / str(xsi_mapping.get("receiver_dir", "XSILMR"))
    depth_pattern = str(xsi_mapping["depth_variable_pattern"])
    time_pattern = str(xsi_mapping["time_variable_pattern"])
    field_pattern = str(xsi_mapping["waveform_variable_pattern"])
    receiver_chunks: list[np.ndarray] = []
    source_starts: dict[str, int] = {}
    observed_depths: dict[str, list[float]] = {}
    for receiver_index in range(1, receiver_count + 1):
        depth_vector = np.asarray(depth_vectors[receiver_index - 1], dtype=np.float32).reshape(-1)
        if depth_vector.size < chunk_depth_samples:
            raise ValueError(f"receiver {receiver_index} has too few depth samples.")
        nearest = int(np.nanargmin(np.abs(depth_vector - center_depth_ft)))
        start = max(
            0, min(nearest - chunk_depth_samples // 2, depth_vector.size - chunk_depth_samples)
        )
        receiver_file = receiver_dir / f"XSILMR{receiver_index:02d}.mat"
        requests = [
            MatReadRequest(
                variable_path=_format_pattern(depth_pattern, receiver_index),
                role="depth",
                source_orientation=_as_str_list(
                    xsi_mapping.get("depth_source_shape_order"),
                    ["depth"],
                ),
                canonical_orientation=["depth"],
                max_depth_samples=chunk_depth_samples,
                max_time_samples=max_time_samples,
                max_cast_azimuth=0,
                source_start_index=start,
            ),
            MatReadRequest(
                variable_path=_format_pattern(time_pattern, receiver_index),
                role="xsi_time",
                source_orientation=["scalar"],
                canonical_orientation=["receiver"],
                max_depth_samples=chunk_depth_samples,
                max_time_samples=max_time_samples,
                max_cast_azimuth=0,
                source_start_index=0,
            ),
        ]
        requests.extend(
            MatReadRequest(
                variable_path=_format_pattern(field_pattern, receiver_index, side=side),
                role="xsi_waveform",
                source_orientation=_as_str_list(xsi_mapping.get("waveform_source_shape_order")),
                canonical_orientation=_as_str_list(
                    xsi_mapping.get("waveform_canonical_shape_order")
                ),
                max_depth_samples=chunk_depth_samples,
                max_time_samples=max_time_samples,
                max_cast_azimuth=0,
                source_start_index=start,
            )
            for side in side_labels
        )
        data = read_mat_file_slices(receiver_file, requests)
        depth_key = requests[0].variable_path
        side_arrays = [
            np.asarray(data[request.variable_path], dtype=np.float32) for request in requests[2:]
        ]
        receiver_chunk = np.stack(side_arrays, axis=1)
        if receiver_chunk.shape != (chunk_depth_samples, len(side_labels), max_time_samples):
            raise ValueError(
                f"receiver {receiver_index} chunk shape {receiver_chunk.shape} is not "
                f"{(chunk_depth_samples, len(side_labels), max_time_samples)}."
            )
        receiver_chunks.append(receiver_chunk)
        source_starts[f"receiver_{receiver_index:02d}"] = start
        observed_depths[f"receiver_{receiver_index:02d}"] = [
            float(item) for item in np.asarray(data[depth_key], dtype=np.float32).reshape(-1)
        ]
    chunk = np.stack(receiver_chunks, axis=1).astype(np.float32)
    return chunk, {"source_start_indices": source_starts, "observed_depths": observed_depths}


def compute_relative_stc_summary(
    trace_by_receiver_side: np.ndarray,
    *,
    side_labels: list[str],
    max_delay_samples_per_receiver: float = 4.0,
    delay_count: int = 25,
    epsilon: float = 1.0e-8,
) -> dict[str, Any]:
    traces = _normalize_traces(np.asarray(trace_by_receiver_side, dtype=np.float32))
    if traces.ndim != 3:
        raise ValueError("trace_by_receiver_side must have shape [receiver, side, time].")
    receiver_count, side_count, time_count = traces.shape
    offsets = np.arange(receiver_count, dtype=np.float64) - (receiver_count - 1.0) / 2.0
    delays = np.linspace(
        -max_delay_samples_per_receiver, max_delay_samples_per_receiver, delay_count
    )
    side_rows: list[dict[str, Any]] = []
    for side_index in range(side_count):
        side_trace = traces[:, side_index, :]
        best_score = -np.inf
        best_delay = 0.0
        for delay in delays:
            shifts = np.rint(delay * offsets).astype(int)
            start = max(0, int(np.max(shifts)))
            stop = min(time_count, int(time_count + np.min(shifts)))
            if stop - start < 16:
                continue
            aligned = np.vstack(
                [
                    side_trace[row, start - shifts[row] : stop - shifts[row]]
                    for row in range(receiver_count)
                ]
            )
            numerator = np.sum(aligned, axis=0) ** 2
            denominator = receiver_count * np.sum(aligned * aligned, axis=0) + epsilon
            score = float(np.nanmean(numerator / denominator))
            if score > best_score:
                best_score = score
                best_delay = float(delay)
        side_rows.append(
            {
                "side": side_labels[side_index]
                if side_index < len(side_labels)
                else str(side_index),
                "peak_coherence": float(best_score),
                "peak_delay_samples_per_receiver": best_delay,
            }
        )
    peak = max(side_rows, key=lambda row: float(row["peak_coherence"]))
    return {
        "peak_coherence": float(peak["peak_coherence"]),
        "peak_side": str(peak["side"]),
        "peak_delay_samples_per_receiver": float(peak["peak_delay_samples_per_receiver"]),
        "side_summaries": side_rows,
    }


def compute_apes_proxy_summary(
    trace_by_receiver_side: np.ndarray,
    *,
    side_labels: list[str],
    frequency_count: int = 24,
    covariance_window: int = 32,
    regularization: float = 1.0e-3,
    epsilon: float = 1.0e-10,
) -> dict[str, Any]:
    traces = _normalize_traces(np.asarray(trace_by_receiver_side, dtype=np.float32))
    if traces.ndim != 3:
        raise ValueError("trace_by_receiver_side must have shape [receiver, side, time].")
    side_trace = np.nanmean(traces, axis=0)
    frequencies = np.linspace(0.02, 0.45, frequency_count, dtype=np.float64)
    side_rows: list[dict[str, Any]] = []
    for side_index in range(side_trace.shape[0]):
        x = np.asarray(side_trace[side_index], dtype=np.float64)
        window = min(max(covariance_window, 8), max(x.size // 3, 8))
        if window >= x.size:
            window = max(x.size // 2, 8)
        windows = np.lib.stride_tricks.sliding_window_view(x, window)
        cov = (windows.T @ windows) / max(windows.shape[0], 1)
        scale = float(np.trace(cov) / max(cov.shape[0], 1))
        cov = cov + np.eye(cov.shape[0]) * max(scale * regularization, epsilon)
        inv_cov = np.linalg.pinv(cov, hermitian=True)
        axis = np.arange(window, dtype=np.float64)
        powers: list[float] = []
        for frequency in frequencies:
            steering = np.exp(-2j * np.pi * frequency * axis)
            denom = np.real(np.conjugate(steering) @ inv_cov @ steering)
            powers.append(float(1.0 / max(denom, epsilon)))
        power_array = np.asarray(powers, dtype=np.float64)
        peak_index = int(np.nanargmax(power_array))
        side_rows.append(
            {
                "side": side_labels[side_index]
                if side_index < len(side_labels)
                else str(side_index),
                "peak_power": float(power_array[peak_index]),
                "peak_frequency": float(frequencies[peak_index]),
                "median_power": float(np.nanmedian(power_array)),
            }
        )
    peak = max(side_rows, key=lambda row: float(row["peak_power"]))
    return {
        "peak_power": float(peak["peak_power"]),
        "peak_side": str(peak["side"]),
        "peak_frequency": float(peak["peak_frequency"]),
        "side_summaries": side_rows,
    }


def format_sa_pilot_markdown(report: SaPilotReport) -> str:
    lines = [
        "# MVP-4X-SA Bounded STC/APES Pilot",
        "",
        "Scope: research-only bounded pilot. No full-well STC/APES maps, deep learning, "
        "final labels, ground-truth claims, or production claims.",
        "",
        f"- pilot_version: `{report.pilot_version}`",
        f"- interval_count: {report.interval_count}",
        f"- skipped_interval_count: {report.skipped_interval_count}",
        f"- chunk_depth_samples: {report.waveform_read_policy['chunk_depth_samples']}",
        f"- max_time_samples: {report.waveform_read_policy['max_time_samples']}",
        f"- receiver_count: {report.waveform_read_policy['receiver_count']}",
        f"- side_count: {report.waveform_read_policy['side_count']}",
        f"- runtime_seconds: {report.runtime_seconds:.3f}",
        f"- peak_memory_bytes: {report.peak_memory_bytes}",
        f"- estimated_waveform_bytes_read: {report.estimated_waveform_bytes_read}",
        f"- stc_peak_coherence_p50: {report.summary_stats['stc_peak_coherence'].get('p50')}",
        f"- apes_peak_power_p50: {report.summary_stats['apes_peak_power'].get('p50')}",
        f"- report_json: `{report.outputs.report_json}`",
        f"- interval_csv: `{report.outputs.interval_csv}`",
        "",
        "The STC result is a relative-delay semblance proxy and the APES result is a "
        "regularized adaptive spectral proxy. They are feasibility signals only.",
    ]
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report.warnings[:20])
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in report.errors)
    return "\n".join(lines) + "\n"


def _write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    center_depths: np.ndarray,
    stc_peaks: np.ndarray,
    apes_peaks: np.ndarray,
    overwrite: bool,
) -> SaPilotOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_csv = output_dir / "mvp4x_sa_pilot_intervals.csv"
    feature_npz = output_dir / "mvp4x_sa_pilot_features.npz"
    report_json = output_dir / "mvp4x_sa_pilot_report.json"
    report_md = output_dir / "mvp4x_sa_pilot_report.md"
    for path in (interval_csv, feature_npz, report_json, report_md):
        _ensure_can_write(path, overwrite=overwrite)
    if rows:
        fieldnames = list(rows[0].keys())
        with interval_csv.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        interval_csv.write_text("", encoding="utf-8")
    np.savez_compressed(
        feature_npz,
        pilot_version=np.asarray(SA_PILOT_VERSION),
        depth_center=center_depths,
        stc_peak_coherence=stc_peaks,
        apes_peak_power=apes_peaks,
        **{key: np.asarray(value) for key, value in RESEARCH_FLAGS.items()},
    )
    report_json.write_text("{}\n", encoding="utf-8")
    report_md.write_text("", encoding="utf-8")
    return SaPilotOutputs(
        output_dir=str(output_dir),
        report_json=str(report_json),
        report_md=str(report_md),
        interval_csv=str(interval_csv),
        feature_npz=str(feature_npz),
    )


def _rewrite_report_outputs(
    report: SaPilotReport,
    rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    report_json = Path(report.outputs.report_json)
    report_md = Path(report.outputs.report_md)
    _ensure_can_write(report_json, overwrite=overwrite)
    _ensure_can_write(report_md, overwrite=overwrite)
    payload = report.to_dict()
    payload["interval_preview"] = rows[:10]
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(format_sa_pilot_markdown(report), encoding="utf-8")


def _validate_interval_count(interval_count: int) -> None:
    if interval_count < MIN_INTERVALS or interval_count > MAX_INTERVALS:
        raise ValueError(f"interval_count must be in [{MIN_INTERVALS}, {MAX_INTERVALS}].")


def _validate_snapshot_scope(snapshot: dict[str, np.ndarray]) -> None:
    for key in ("research_only", "exploratory_only", "weak_label_target", "no_final_labels"):
        value = snapshot.get(key)
        if value is None or bool(np.asarray(value).reshape(())) is not True:
            raise ValueError(f"snapshot flag {key} must be true.")
    if "depth" not in snapshot:
        raise ValueError("snapshot must contain depth.")


def _validate_xsi_mapping(xsi: dict[str, Any]) -> None:
    required = (
        "receiver_dir",
        "depth_variable_pattern",
        "time_variable_pattern",
        "waveform_variable_pattern",
        "waveform_source_shape_order",
        "waveform_canonical_shape_order",
    )
    missing = [key for key in required if key not in xsi]
    if missing:
        raise ValueError("XSI mapping is missing keys: " + ", ".join(missing))
    if _as_str_list(xsi["waveform_source_shape_order"]) != ["time", "depth"]:
        raise ValueError("XSI waveform source order must be [time, depth].")
    if _as_str_list(xsi["waveform_canonical_shape_order"]) != ["depth", "time"]:
        raise ValueError("XSI waveform canonical order must be [depth, time].")


def _pick_metric_quantiles(
    metric: np.ndarray,
    *,
    mask: np.ndarray,
    count: int,
    descending: bool,
) -> np.ndarray:
    values = np.asarray(metric, dtype=np.float64).reshape(-1)
    candidates = np.where(mask & np.isfinite(values))[0]
    if candidates.size == 0 or count <= 0:
        return np.asarray([], dtype=int)
    order = candidates[np.argsort(values[candidates])]
    if descending:
        order = order[::-1]
    pool = order[: min(order.size, max(count * 4, count))]
    positions = np.linspace(0, pool.size - 1, min(count, pool.size), dtype=int)
    return pool[positions]


def _add_selected(
    selected: list[tuple[int, str]],
    used_indices: set[int],
    used_depth_bins: set[int],
    index: int,
    category: str,
    depth: np.ndarray,
    *,
    enforce_depth_bin: bool,
) -> bool:
    if index in used_indices:
        return False
    depth_bin = int(round(float(depth[index]) * 10.0))
    if enforce_depth_bin and depth_bin in used_depth_bins:
        return False
    selected.append((index, category))
    used_indices.add(index)
    used_depth_bins.add(depth_bin)
    return True


def _metric(snapshot: dict[str, np.ndarray], key: str, length: int) -> np.ndarray:
    value = snapshot.get(key)
    if value is None:
        return np.zeros(length, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(length, float(array), dtype=np.float64)
    if array.shape[0] != length:
        return np.zeros(length, dtype=np.float64)
    if array.ndim > 1:
        return np.nanmean(array.reshape(length, -1), axis=1)
    return array.reshape(-1)


def _matrix_metric(snapshot: dict[str, np.ndarray], key: str, length: int) -> np.ndarray:
    value = snapshot.get(key)
    if value is None:
        return np.zeros(length, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape[0] != length:
        return np.zeros(length, dtype=np.float64)
    return np.nanmax(array.reshape(length, -1), axis=1)


def _normalize_traces(values: np.ndarray) -> np.ndarray:
    traces = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    traces = traces - np.mean(traces, axis=-1, keepdims=True)
    scale = np.std(traces, axis=-1, keepdims=True)
    return traces / np.maximum(scale, 1.0e-6)


def _micro_benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = np.asarray([float(row["runtime_seconds"]) for row in rows], dtype=np.float64)
    if runtimes.size == 0:
        return {"status": "skipped", "intervals": []}
    return {
        "status": "succeeded",
        "interval_count": int(runtimes.size),
        "runtime_seconds": {
            "min": float(np.min(runtimes)),
            "p50": float(np.quantile(runtimes, 0.50)),
            "max": float(np.max(runtimes)),
        },
        "intervals": rows,
    }


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "finite_count": 0, "finite_ratio": 0.0}
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "finite_ratio": float(finite.size / max(values.size, 1)),
        "min": float(np.min(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "max": float(np.max(finite)),
    }


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _format_pattern(pattern: str, receiver: int, *, side: str | None = None) -> str:
    return pattern.format(receiver=receiver, side=side or "")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
