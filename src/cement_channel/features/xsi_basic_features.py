from __future__ import annotations

import json
import resource
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.data.small_slice_reader import (
    MatReadRequest,
    load_mapping_config,
    read_mat_file_slices,
)
from cement_channel.evaluation.correlation_schema import (
    MVP4A_BASIC_FEATURE_VERSION,
    CorrelationConfig,
    expected_feature_names,
    load_correlation_config,
)

XSI_BASIC_FEATURE_NAMES = (
    "rms_energy",
    "peak_abs",
    "mean_abs",
    "early_energy",
    "late_energy",
    "late_over_early_ratio",
)


@dataclass(frozen=True)
class XsiBasicFeatureReport:
    feature_version: str
    generated_at: str
    inputs: dict[str, str]
    shape: dict[str, Any]
    feature_names: list[str]
    chunk_depth_samples: int
    max_time_samples: int
    memory_usage: dict[str, int | float | None]
    summaries: dict[str, dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    no_model_training: bool
    no_stc: bool
    no_apes: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_xsi_basic_features_from_config(
    *,
    paths_config: dict[str, Any],
    mapping_path: Path | str,
    label_samples_npz: Path | str,
    correlation_config_path: Path | str,
    input_waveform_npz: Path | str | None = None,
    limit_depth: int | None = None,
    chunk_depth_samples: int | None = None,
    max_time_samples: int | None = None,
) -> tuple[XsiBasicFeatureReport, dict[str, np.ndarray]]:
    config = load_correlation_config(correlation_config_path)
    sample_arrays = _load_npz(label_samples_npz)
    if input_waveform_npz is not None:
        waveform_arrays = _load_npz(input_waveform_npz)
        return extract_xsi_basic_features_from_waveform_arrays(
            waveform_arrays=waveform_arrays,
            sample_arrays=sample_arrays,
            correlation_config=config,
            inputs={
                "label_samples_npz": str(label_samples_npz),
                "correlation_config_path": str(correlation_config_path),
                "input_waveform_npz": str(input_waveform_npz),
            },
            limit_depth=limit_depth,
        )
    mapping = load_mapping_config(mapping_path)
    return extract_xsi_basic_features_from_raw(
        paths_config=paths_config,
        mapping=mapping,
        sample_arrays=sample_arrays,
        correlation_config=config,
        inputs={
            "mapping_path": str(mapping_path),
            "label_samples_npz": str(label_samples_npz),
            "correlation_config_path": str(correlation_config_path),
        },
        limit_depth=limit_depth,
        chunk_depth_samples=chunk_depth_samples,
        max_time_samples=max_time_samples,
    )


def extract_xsi_basic_features_from_waveform_arrays(
    *,
    waveform_arrays: dict[str, np.ndarray],
    sample_arrays: dict[str, np.ndarray],
    correlation_config: CorrelationConfig,
    inputs: dict[str, str] | None = None,
    limit_depth: int | None = None,
) -> tuple[XsiBasicFeatureReport, dict[str, np.ndarray]]:
    waveform = np.asarray(waveform_arrays["xsi_waveform"], dtype=np.float32)
    if waveform.ndim != 4:
        raise ValueError("xsi_waveform must have shape [depth, receiver, side, time].")
    depth_count = _limited_depth_count(waveform.shape[0], limit_depth)
    waveform = waveform[:depth_count]
    features = extract_basic_features(waveform)
    side_mean, side_median = aggregate_receiver_features(features)
    arrays = _feature_output_arrays(
        features,
        side_mean,
        side_median,
        sample_arrays=sample_arrays,
        correlation_config=correlation_config,
    )
    warnings: list[str] = []
    errors: list[str] = []
    report = _feature_report(
        inputs=inputs or {},
        arrays=arrays,
        feature_names=list(arrays["feature_names"].astype(str)),
        chunk_depth_samples=depth_count,
        max_time_samples=int(waveform.shape[-1]),
        memory_usage={
            "mode": 1,
            "input_waveform_bytes": int(waveform.nbytes),
            "max_observed_chunk_waveform_bytes": int(waveform.nbytes),
            "feature_array_bytes": int(features.nbytes),
            "ru_maxrss_kb": _max_rss_kb(),
        },
        warnings=warnings,
        errors=errors,
    )
    return report, arrays


def extract_xsi_basic_features_from_raw(
    *,
    paths_config: dict[str, Any],
    mapping: dict[str, Any],
    sample_arrays: dict[str, np.ndarray],
    correlation_config: CorrelationConfig,
    inputs: dict[str, str] | None = None,
    limit_depth: int | None = None,
    chunk_depth_samples: int | None = None,
    max_time_samples: int | None = None,
) -> tuple[XsiBasicFeatureReport, dict[str, np.ndarray]]:
    data_config = _as_dict(paths_config.get("data"))
    raw_dir = Path(str(data_config.get("raw", "")))
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    xsi_depth_index = np.asarray(sample_arrays["xsi_depth_index"], dtype=np.int32).reshape(-1)
    depth_count = _limited_depth_count(xsi_depth_index.size, limit_depth)
    xsi_depth_index = xsi_depth_index[:depth_count]
    if not _is_contiguous_index(xsi_depth_index):
        raise ValueError("xsi_depth_index must be contiguous for controlled chunked MAT reading.")
    active_chunk = int(chunk_depth_samples or correlation_config.chunk_depth_samples)
    active_time = int(max_time_samples or correlation_config.max_time_samples)
    if active_chunk <= 0:
        raise ValueError("chunk_depth_samples must be positive.")
    if active_time <= 0:
        raise ValueError("max_time_samples must be positive.")
    xsi_config = _as_dict(mapping.get("xsi"))
    receiver_count = int(xsi_config.get("expected_receiver_files", 13))
    side_labels = _as_str_list(xsi_config.get("side_labels"), list("ABCDEFGH"))
    feature_count = len(XSI_BASIC_FEATURE_NAMES)
    features = np.empty(
        (depth_count, receiver_count, len(side_labels), feature_count),
        dtype=np.float32,
    )
    max_chunk_bytes = 0
    warnings: list[str] = []
    errors: list[str] = []
    for output_start, output_stop in _chunk_ranges(depth_count, active_chunk):
        source_start_index = int(xsi_depth_index[output_start])
        sample_count = int(output_stop - output_start)
        for receiver_index in range(1, receiver_count + 1):
            waveform = read_xsi_waveform_chunk(
                raw_dir=raw_dir,
                mapping=mapping,
                receiver_index=receiver_index,
                side_labels=side_labels,
                source_start_index=source_start_index,
                sample_count=sample_count,
                max_time_samples=active_time,
            )
            max_chunk_bytes = max(max_chunk_bytes, int(waveform.nbytes))
            features[output_start:output_stop, receiver_index - 1] = extract_basic_features(
                waveform[:, None, :, :],
            )[:, 0]
    side_mean, side_median = aggregate_receiver_features(features)
    arrays = _feature_output_arrays(
        features,
        side_mean,
        side_median,
        sample_arrays=sample_arrays,
        correlation_config=correlation_config,
    )
    report = _feature_report(
        inputs=inputs or {},
        arrays=arrays,
        feature_names=list(arrays["feature_names"].astype(str)),
        chunk_depth_samples=active_chunk,
        max_time_samples=active_time,
        memory_usage={
            "mode": 0,
            "input_waveform_bytes": None,
            "max_observed_chunk_waveform_bytes": max_chunk_bytes,
            "feature_array_bytes": int(features.nbytes),
            "ru_maxrss_kb": _max_rss_kb(),
        },
        warnings=warnings,
        errors=errors,
    )
    return report, arrays


def read_xsi_waveform_chunk(
    *,
    raw_dir: Path,
    mapping: dict[str, Any],
    receiver_index: int,
    side_labels: list[str],
    source_start_index: int,
    sample_count: int,
    max_time_samples: int,
) -> np.ndarray:
    xsi = _as_dict(mapping.get("xsi"))
    receiver_dir = raw_dir / str(xsi.get("receiver_dir", "XSILMR"))
    receiver_file = receiver_dir / f"XSILMR{receiver_index:02d}.mat"
    pattern = str(xsi.get("waveform_variable_pattern", ""))
    requests = [
        MatReadRequest(
            variable_path=pattern.format(receiver=receiver_index, side=side),
            role="xsi_waveform",
            source_orientation=_as_str_list(xsi.get("waveform_source_shape_order")),
            canonical_orientation=_as_str_list(xsi.get("waveform_canonical_shape_order")),
            max_depth_samples=sample_count,
            max_time_samples=max_time_samples,
            max_cast_azimuth=180,
            source_start_index=source_start_index,
        )
        for side in side_labels
    ]
    data = read_mat_file_slices(receiver_file, requests)
    side_arrays = [
        np.asarray(data[request.variable_path], dtype=np.float32) for request in requests
    ]
    return np.stack(side_arrays, axis=1).astype(np.float32)


def extract_basic_features(
    waveform: np.ndarray,
    *,
    early_fraction: tuple[float, float] = (0.0, 0.35),
    late_fraction: tuple[float, float] = (0.35, 1.0),
) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError("waveform must have shape [depth, receiver, side, time].")
    if array.shape[-1] <= 0:
        raise ValueError("waveform time dimension is empty.")
    early = _time_window(array, early_fraction)
    late = _time_window(array, late_fraction)
    abs_values = np.abs(array, dtype=np.float32)
    square = np.square(array, dtype=np.float32)
    early_square = np.square(early, dtype=np.float32)
    late_square = np.square(late, dtype=np.float32)
    rms = np.sqrt(np.nanmean(square, axis=-1)).astype(np.float32)
    peak_abs = np.nanmax(abs_values, axis=-1).astype(np.float32)
    mean_abs = np.nanmean(abs_values, axis=-1).astype(np.float32)
    early_energy = np.nanmean(early_square, axis=-1).astype(np.float32)
    late_energy = np.nanmean(late_square, axis=-1).astype(np.float32)
    ratio = np.divide(
        late_energy,
        np.maximum(early_energy, np.float32(1.0e-12)),
    ).astype(np.float32)
    return np.stack(
        [rms, peak_abs, mean_abs, early_energy, late_energy, ratio],
        axis=-1,
    ).astype(np.float32)


def aggregate_receiver_features(
    receiver_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(receiver_features, dtype=np.float32)
    if features.ndim != 4:
        raise ValueError("receiver_features must have shape [depth, receiver, side, feature].")
    mean = np.nanmean(features, axis=1).astype(np.float32)
    median = np.nanmedian(features, axis=1).astype(np.float32)
    return mean, median


def write_xsi_basic_feature_outputs(
    report: XsiBasicFeatureReport,
    arrays: dict[str, np.ndarray],
    *,
    output_npz: Path,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_xsi_basic_feature_markdown(report), encoding="utf-8")


def format_xsi_basic_feature_markdown(report: XsiBasicFeatureReport) -> str:
    data = report.to_dict()
    lines = [
        "# XSI Basic Feature Report",
        "",
        f"- Version: {data['feature_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Feature names: {', '.join(data['feature_names'])}",
        f"- Chunk depth samples: {data['chunk_depth_samples']}",
        f"- Max time samples: {data['max_time_samples']}",
        f"- No model training: {data['no_model_training']}",
        f"- No STC: {data['no_stc']}",
        f"- No APES: {data['no_apes']}",
        "",
        "## Shapes",
        "",
    ]
    for key, value in data["shape"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Memory Usage", ""])
    for key, value in data["memory_usage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Summaries", ""])
    for key, summary in data["summaries"].items():
        lines.append(f"### {key}")
        for summary_key, value in summary.items():
            lines.append(f"- {summary_key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _feature_output_arrays(
    features: np.ndarray,
    side_mean: np.ndarray,
    side_median: np.ndarray,
    *,
    sample_arrays: dict[str, np.ndarray],
    correlation_config: CorrelationConfig,
) -> dict[str, np.ndarray]:
    depth_count = features.shape[0]
    feature_names = np.asarray(expected_feature_names(correlation_config))
    if feature_names.size != features.shape[-1]:
        feature_names = np.asarray(XSI_BASIC_FEATURE_NAMES)
    return {
        "xsi_depth": np.asarray(sample_arrays["xsi_depth"], dtype=np.float32)[:depth_count],
        "xsi_depth_index": np.asarray(sample_arrays["xsi_depth_index"], dtype=np.int32)[
            :depth_count
        ],
        "receiver_index": np.arange(1, features.shape[1] + 1, dtype=np.int16),
        "side_labels": np.asarray(correlation_config.side_labels),
        "xsi_side_azimuth_deg": np.asarray(sample_arrays["xsi_side_azimuth_deg"], dtype=np.float32),
        "feature_names": feature_names,
        "xsi_basic_features": features.astype(np.float32),
        "xsi_basic_features_by_side": side_mean.astype(np.float32),
        "xsi_basic_features_by_side_mean": side_mean.astype(np.float32),
        "xsi_basic_features_by_side_median": side_median.astype(np.float32),
        "no_model_training": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "feature_version": MVP4A_BASIC_FEATURE_VERSION,
                    "feature_names": feature_names.astype(str).tolist(),
                    "no_model_training": True,
                    "no_stc": True,
                    "no_apes": True,
                    "no_final_labels": True,
                },
                ensure_ascii=False,
            )
        ),
    }


def _feature_report(
    *,
    inputs: dict[str, str],
    arrays: dict[str, np.ndarray],
    feature_names: list[str],
    chunk_depth_samples: int,
    max_time_samples: int,
    memory_usage: dict[str, int | float | None],
    warnings: list[str],
    errors: list[str],
) -> XsiBasicFeatureReport:
    return XsiBasicFeatureReport(
        feature_version=MVP4A_BASIC_FEATURE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        shape={
            "xsi_basic_features": [int(value) for value in arrays["xsi_basic_features"].shape],
            "xsi_basic_features_by_side": [
                int(value) for value in arrays["xsi_basic_features_by_side"].shape
            ],
        },
        feature_names=feature_names,
        chunk_depth_samples=int(chunk_depth_samples),
        max_time_samples=int(max_time_samples),
        memory_usage=memory_usage,
        summaries={
            "xsi_basic_features": _numeric_summary(arrays["xsi_basic_features"]),
            "xsi_basic_features_by_side": _numeric_summary(
                arrays["xsi_basic_features_by_side"]
            ),
        },
        warnings=warnings,
        errors=errors,
        no_model_training=True,
        no_stc=True,
        no_apes=True,
        not_performed=[
            "STC",
            "APES",
            "model training",
            "train/test split",
            "final label generation",
        ],
    )


def _time_window(waveform: np.ndarray, fraction: tuple[float, float]) -> np.ndarray:
    start_fraction, stop_fraction = fraction
    if not 0.0 <= start_fraction < stop_fraction <= 1.0:
        raise ValueError("time window fractions must satisfy 0 <= start < stop <= 1.")
    time_count = waveform.shape[-1]
    start = int(np.floor(start_fraction * time_count))
    stop = int(np.ceil(stop_fraction * time_count))
    stop = min(max(stop, start + 1), time_count)
    return waveform[..., start:stop]


def _limited_depth_count(depth_count: int, limit_depth: int | None) -> int:
    if limit_depth is None:
        return int(depth_count)
    if limit_depth <= 0:
        raise ValueError("limit_depth must be positive.")
    return min(int(depth_count), int(limit_depth))


def _is_contiguous_index(indices: np.ndarray) -> bool:
    if indices.size == 0:
        return True
    expected = np.arange(int(indices[0]), int(indices[0]) + indices.size, dtype=np.int32)
    return bool(np.array_equal(indices, expected))


def _chunk_ranges(depth_count: int, chunk_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + chunk_size, depth_count))
        for start in range(0, depth_count, chunk_size)
    ]


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "finite_ratio": None,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    finite = np.isfinite(array)
    finite_ratio = float(np.mean(finite))
    if not np.any(finite):
        return {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "finite_ratio": finite_ratio,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    finite_values = array[finite]
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "finite_ratio": finite_ratio,
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(default or [])
