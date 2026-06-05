from __future__ import annotations

import json
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
from cement_channel.features.mvp4x_waveform_features import (
    feature_correlation_summary,
)

TIME_FREQUENCY_FEATURE_VERSION = "mvp4x_time_frequency_features_v002"
TIME_FREQUENCY_REPORT_VERSION = "mvp4x_time_frequency_features_v002"
TF_BASE_FEATURE_NAMES = (
    "early_band_energy_low",
    "early_band_energy_mid",
    "early_band_energy_high",
    "middle_band_energy_low",
    "middle_band_energy_mid",
    "middle_band_energy_high",
    "late_band_energy_low",
    "late_band_energy_mid",
    "late_band_energy_high",
    "log_low_mid_energy_ratio",
    "log_mid_high_energy_ratio",
    "log_early_late_low_ratio",
    "log_early_late_mid_ratio",
    "log_early_late_high_ratio",
    "spectral_entropy",
    "spectral_flatness",
    "dominant_frequency",
    "dominant_frequency_stability",
    "low_mid_band_contrast",
    "mid_high_band_contrast",
    "low_high_band_contrast",
    "early_spectral_centroid",
    "late_spectral_centroid",
    "early_late_spectral_shift",
)
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}


@dataclass(frozen=True)
class TimeFrequencyFeatureReport:
    report_version: str
    feature_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    sample_count: int
    receiver_count: int
    side_count: int
    base_feature_count: int
    depth_feature_count: int
    chunk_count: int
    chunk_depth_count: int
    chunk_memory_cap_bytes: int
    estimated_max_waveform_chunk_bytes: int
    peak_memory_bytes: int
    runtime_seconds: float
    waveform_source_paths: list[str]
    waveform_source_field_pattern: str
    finite_ratio: dict[str, float | None]
    feature_schema: dict[str, Any]
    summary_stats: dict[str, dict[str, float | int | None]]
    correlation_summary: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    research_only: bool
    exploratory_only: bool
    weak_label_target: bool
    no_final_labels: bool
    no_ground_truth_claim: bool
    no_production_claim: bool
    not_validated_for_deployment: bool
    no_raw_mat_modified: bool
    no_full_waveform_loaded: bool
    no_dynamic_band_search: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_time_frequency_features_from_paths(
    *,
    paths_config: Path | str,
    mapping_path: Path | str,
    snapshot_npz: Path | str,
    rapid_config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
) -> TimeFrequencyFeatureReport:
    paths = load_paths_config(paths_config)
    mapping = load_mapping_config(mapping_path)
    config = _load_yaml(rapid_config_path)
    snapshot = _load_npz(snapshot_npz)
    arrays, report = extract_time_frequency_features(
        paths_config=paths,
        mapping=mapping,
        snapshot=snapshot,
        config=config,
        inputs={
            "paths_config": str(paths_config),
            "mapping_path": str(mapping_path),
            "snapshot_npz": str(snapshot_npz),
            "rapid_config_path": str(rapid_config_path),
        },
        output_npz=Path(output_npz),
    )
    write_time_frequency_feature_outputs(
        arrays,
        report,
        output_npz=Path(output_npz),
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return report


def extract_time_frequency_features(
    *,
    paths_config: dict[str, Any],
    mapping: dict[str, Any],
    snapshot: dict[str, np.ndarray],
    config: dict[str, Any],
    inputs: dict[str, str],
    output_npz: Path | None = None,
) -> tuple[dict[str, np.ndarray], TimeFrequencyFeatureReport]:
    start_time = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_trigger_scope(snapshot, errors)
    raw_dir = Path(str(_as_dict(paths_config.get("data")).get("raw", "")))
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    waveform_config = _as_dict(config.get("waveform_features"))
    xsi_mapping = _as_dict(mapping.get("xsi"))
    _validate_waveform_mapping(xsi_mapping)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    receiver_count = int(
        waveform_config.get("receiver_count", xsi_mapping.get("expected_receiver_files", 13))
    )
    side_labels = [
        str(item) for item in _as_list(waveform_config.get("side_labels"), list("ABCDEFGH"))
    ]
    side_labels = side_labels[: int(xsi_mapping.get("expected_side_count", len(side_labels)))]
    max_time_samples = int(
        waveform_config.get("max_time_samples", xsi_mapping.get("expected_time_sample_count", 1024))
    )
    cap_bytes = int(float(waveform_config.get("chunk_memory_cap_mb", 128)) * 1024 * 1024)
    if cap_bytes > 256 * 1024 * 1024:
        raise ValueError("time-frequency chunk_memory_cap_mb must not exceed 256 MB.")
    chunk_depth = _chunk_depth_count(
        cap_bytes=cap_bytes,
        side_count=len(side_labels),
        time_count=max_time_samples,
    )
    receiver_side = np.empty(
        (depth.size, receiver_count, len(side_labels), len(TF_BASE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    receiver_side.fill(np.nan)
    source_paths: list[str] = []
    chunk_count = 0
    estimated_max_chunk_bytes = 0
    peak_memory_bytes = _peak_rss_bytes()
    receiver_dir = raw_dir / str(xsi_mapping.get("receiver_dir", "XSILMR"))
    field_pattern = str(xsi_mapping["waveform_variable_pattern"])
    depth_pattern = str(xsi_mapping["depth_variable_pattern"])
    time_pattern = str(xsi_mapping["time_variable_pattern"])
    for receiver_index in range(1, receiver_count + 1):
        receiver_file = receiver_dir / f"XSILMR{receiver_index:02d}.mat"
        if not receiver_file.exists():
            raise FileNotFoundError(f"XSI receiver MAT file does not exist: {receiver_file}")
        source_paths.append(str(receiver_file))
        for start in range(0, depth.size, chunk_depth):
            count = min(chunk_depth, depth.size - start)
            requests = [
                MatReadRequest(
                    variable_path=_format_pattern(depth_pattern, receiver_index),
                    role="depth",
                    source_orientation=_as_str_list(
                        xsi_mapping.get("depth_source_shape_order"),
                        ["depth"],
                    ),
                    canonical_orientation=["depth"],
                    max_depth_samples=count,
                    max_time_samples=max_time_samples,
                    max_cast_azimuth=0,
                    source_start_index=start,
                ),
                MatReadRequest(
                    variable_path=_format_pattern(time_pattern, receiver_index),
                    role="xsi_time",
                    source_orientation=["scalar"],
                    canonical_orientation=["receiver"],
                    max_depth_samples=count,
                    max_time_samples=max_time_samples,
                    max_cast_azimuth=0,
                    source_start_index=0,
                ),
            ]
            requests.extend(
                MatReadRequest(
                    variable_path=_format_pattern(field_pattern, receiver_index, side=side),
                    role="xsi_waveform",
                    source_orientation=_as_str_list(
                        xsi_mapping.get("waveform_source_shape_order"),
                    ),
                    canonical_orientation=_as_str_list(
                        xsi_mapping.get("waveform_canonical_shape_order"),
                    ),
                    max_depth_samples=count,
                    max_time_samples=max_time_samples,
                    max_cast_azimuth=0,
                    source_start_index=start,
                )
                for side in side_labels
            )
            data = read_mat_file_slices(receiver_file, requests)
            raw_depth = np.asarray(data[requests[0].variable_path], dtype=np.float32).reshape(-1)
            if raw_depth.shape[0] != count:
                raise ValueError(f"Receiver {receiver_index} chunk {start}: depth count mismatch.")
            if not np.allclose(raw_depth, depth[start : start + count], atol=1.0e-3):
                max_diff = float(np.max(np.abs(raw_depth - depth[start : start + count])))
                raise ValueError(
                    f"Receiver {receiver_index} chunk {start}: depth mismatch {max_diff:.6f}."
                )
            for side_index, request in enumerate(requests[2:]):
                waveform = np.asarray(data[request.variable_path], dtype=np.float32)
                if waveform.shape != (count, max_time_samples):
                    raise ValueError(
                        f"{request.variable_path} shape {waveform.shape} does not match "
                        f"expected {(count, max_time_samples)}."
                    )
                estimated_max_chunk_bytes = max(estimated_max_chunk_bytes, int(waveform.nbytes))
                receiver_side[start : start + count, receiver_index - 1, side_index, :] = (
                    compute_time_frequency_base_features(waveform)
                )
            chunk_count += 1
            peak_memory_bytes = max(peak_memory_bytes, _peak_rss_bytes())
    depth_features, depth_feature_names, depth_feature_groups = build_depth_time_frequency_features(
        receiver_side,
        base_feature_names=list(TF_BASE_FEATURE_NAMES),
    )
    arrays: dict[str, np.ndarray] = {
        "feature_version": np.asarray(TIME_FREQUENCY_FEATURE_VERSION),
        "depth": depth.astype(np.float32),
        "tf_receiver_side_features": receiver_side.astype(np.float32),
        "tf_receiver_side_feature_names": np.asarray(TF_BASE_FEATURE_NAMES),
        "tf_depth_features": depth_features.astype(np.float32),
        "tf_depth_feature_names": np.asarray(depth_feature_names),
        "tf_depth_feature_group": np.asarray(depth_feature_groups),
        "receiver_index": np.arange(1, receiver_count + 1, dtype=np.int16),
        "side_label": np.asarray(side_labels),
        "chunk_count": np.asarray(chunk_count, dtype=np.int32),
        "chunk_depth_count": np.asarray(chunk_depth, dtype=np.int32),
        "chunk_memory_cap_bytes": np.asarray(cap_bytes, dtype=np.int64),
        "peak_memory_bytes": np.asarray(peak_memory_bytes, dtype=np.int64),
        "waveform_source_paths_json": np.asarray(json.dumps(source_paths)),
        **{key: np.asarray(value) for key, value in RESEARCH_FLAGS.items()},
        "no_raw_mat_modified": np.asarray(True),
        "no_full_waveform_loaded": np.asarray(True),
        "no_dynamic_band_search": np.asarray(True),
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
    }
    runtime_seconds = time.perf_counter() - start_time
    report = TimeFrequencyFeatureReport(
        report_version=TIME_FREQUENCY_REPORT_VERSION,
        feature_version=TIME_FREQUENCY_FEATURE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        output_npz=str(output_npz or ""),
        sample_count=int(depth.size),
        receiver_count=receiver_count,
        side_count=len(side_labels),
        base_feature_count=len(TF_BASE_FEATURE_NAMES),
        depth_feature_count=int(depth_features.shape[1]),
        chunk_count=chunk_count,
        chunk_depth_count=chunk_depth,
        chunk_memory_cap_bytes=cap_bytes,
        estimated_max_waveform_chunk_bytes=estimated_max_chunk_bytes * len(side_labels),
        peak_memory_bytes=peak_memory_bytes,
        runtime_seconds=float(runtime_seconds),
        waveform_source_paths=source_paths,
        waveform_source_field_pattern=field_pattern,
        finite_ratio=_finite_ratios(arrays),
        feature_schema={
            "fixed_windows": ["early", "middle", "late"],
            "fixed_bands": ["low", "mid", "high"],
            "frequency_unit": "cycles_per_sample",
            "no_dynamic_band_search": True,
            "receiver_side_shape": ["depth", "receiver", "side", "tf_base_feature"],
            "depth_feature_shape": ["depth", "tf_depth_feature"],
        },
        summary_stats={
            "tf_depth_features": _summary_stats(depth_features),
            "tf_receiver_side_features": _summary_stats(receiver_side),
        },
        correlation_summary=feature_correlation_summary(depth_features, depth_feature_names),
        warnings=warnings,
        errors=errors,
        research_only=True,
        exploratory_only=True,
        weak_label_target=True,
        no_final_labels=True,
        no_ground_truth_claim=True,
        no_production_claim=True,
        not_validated_for_deployment=True,
        no_raw_mat_modified=True,
        no_full_waveform_loaded=True,
        no_dynamic_band_search=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
    )
    return arrays, report


def compute_time_frequency_base_features(
    waveform: np.ndarray, *, epsilon: float = 1.0e-12
) -> np.ndarray:
    values = np.nan_to_num(
        np.asarray(waveform, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if values.ndim != 2:
        raise ValueError("waveform must have shape [depth, time].")
    time_count = values.shape[1]
    q1 = max(time_count // 4, 1)
    q3 = max((3 * time_count) // 4, q1 + 1)
    windows = {
        "early": values[:, :q1],
        "middle": values[:, q1:q3],
        "late": values[:, -q1:],
    }
    window_band_energy: dict[str, np.ndarray] = {}
    window_centroid: dict[str, np.ndarray] = {}
    window_dominant: dict[str, np.ndarray] = {}
    for name, window in windows.items():
        band_energy, centroid, dominant = _window_spectrum_features(window, epsilon=epsilon)
        window_band_energy[name] = band_energy
        window_centroid[name] = centroid
        window_dominant[name] = dominant
    total_bands, total_centroid, total_dominant = _window_spectrum_features(
        values,
        epsilon=epsilon,
    )
    entropy, flatness = _entropy_flatness(values, epsilon=epsilon)
    dom_stack = np.column_stack(
        [window_dominant["early"], window_dominant["middle"], window_dominant["late"]]
    )
    dominant_stability = np.std(dom_stack, axis=1)
    low = total_bands[:, 0]
    mid = total_bands[:, 1]
    high = total_bands[:, 2]
    early_bands = window_band_energy["early"]
    late_bands = window_band_energy["late"]
    return np.column_stack(
        [
            window_band_energy["early"],
            window_band_energy["middle"],
            window_band_energy["late"],
            np.log((low + epsilon) / (mid + epsilon)),
            np.log((mid + epsilon) / (high + epsilon)),
            np.log((early_bands[:, 0] + epsilon) / (late_bands[:, 0] + epsilon)),
            np.log((early_bands[:, 1] + epsilon) / (late_bands[:, 1] + epsilon)),
            np.log((early_bands[:, 2] + epsilon) / (late_bands[:, 2] + epsilon)),
            entropy,
            flatness,
            total_dominant,
            dominant_stability,
            low - mid,
            mid - high,
            low - high,
            window_centroid["early"],
            window_centroid["late"],
            window_centroid["late"] - window_centroid["early"],
        ]
    ).astype(np.float32)


def build_depth_time_frequency_features(
    receiver_side_features: np.ndarray,
    *,
    base_feature_names: list[str],
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, list[str], list[str]]:
    values = np.asarray(receiver_side_features, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError("receiver_side_features must have shape [depth, receiver, side, feature].")
    flat = values.reshape(values.shape[0], values.shape[1] * values.shape[2], values.shape[3])
    receiver = np.nanmean(values, axis=2)
    side = np.nanmean(values, axis=1)
    blocks: list[tuple[str, np.ndarray, list[str]]] = []
    blocks.extend(_aggregate_block(flat, base_feature_names, "tf_global_receiver_side"))
    blocks.extend(_receiver_structure_blocks(receiver, base_feature_names, epsilon=epsilon))
    blocks.extend(_side_structure_blocks(side, base_feature_names, epsilon=epsilon))
    matrix = np.column_stack([block for _, block, _ in blocks]).astype(np.float32)
    names = [name for _, _, block_names in blocks for name in block_names]
    groups = [group for group, block, _ in blocks for _ in range(block.shape[1])]
    return matrix, names, groups


def write_time_frequency_feature_outputs(
    arrays: dict[str, np.ndarray],
    report: TimeFrequencyFeatureReport,
    *,
    output_npz: Path,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    for path in (output_npz, output_report_md, output_report_json):
        _ensure_can_write(path, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_time_frequency_feature_markdown(report), encoding="utf-8")


def format_time_frequency_feature_markdown(report: TimeFrequencyFeatureReport) -> str:
    lines = [
        "# MVP-4X Fixed Time-Frequency Features v2",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
        f"- feature_version: `{report.feature_version}`",
        f"- sample_count: {report.sample_count}",
        f"- receiver_count: {report.receiver_count}",
        f"- side_count: {report.side_count}",
        f"- base_feature_count: {report.base_feature_count}",
        f"- depth_feature_count: {report.depth_feature_count}",
        f"- chunk_count: {report.chunk_count}",
        f"- chunk_depth_count: {report.chunk_depth_count}",
        f"- chunk_memory_cap_bytes: {report.chunk_memory_cap_bytes}",
        f"- peak_memory_bytes: {report.peak_memory_bytes}",
        f"- runtime_seconds: {report.runtime_seconds:.3f}",
        f"- finite_ratio_tf_depth_features: {report.finite_ratio.get('tf_depth_features')}",
        f"- warnings: {len(report.warnings)}",
        f"- errors: {len(report.errors)}",
        "",
        "No dynamic band search, STC, APES, deep learning, final labels, or production claims.",
    ]
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report.warnings)
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in report.errors)
    return "\n".join(lines) + "\n"


def _window_spectrum_features(
    values: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(values.shape[1], d=1.0).astype(np.float64)
    non_dc = np.arange(1, power.shape[1])
    bands = np.array_split(non_dc, 3)
    band_energy = np.column_stack(
        [
            np.mean(power[:, band], axis=1) if band.size else np.zeros(values.shape[0])
            for band in bands
        ]
    )
    total_power = np.sum(power, axis=1) + epsilon
    centroid = np.sum(power * freqs[None, :], axis=1) / total_power
    if power.shape[1] > 1:
        dominant_index = np.argmax(power[:, 1:], axis=1) + 1
    else:
        dominant_index = np.zeros(values.shape[0], dtype=int)
    return band_energy, centroid, freqs[dominant_index]


def _entropy_flatness(values: np.ndarray, *, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(centered, axis=1)) ** 2).astype(np.float64)
    total = np.sum(power, axis=1, keepdims=True) + epsilon
    probability = power / total
    entropy = -np.sum(probability * np.log(probability + epsilon), axis=1) / np.log(
        max(power.shape[1], 2)
    )
    nonzero_power = power[:, 1:] + epsilon if power.shape[1] > 1 else power + epsilon
    flatness = np.exp(np.mean(np.log(nonzero_power), axis=1)) / np.mean(nonzero_power, axis=1)
    return entropy, flatness


def _aggregate_block(
    values: np.ndarray,
    base_names: list[str],
    prefix: str,
) -> list[tuple[str, np.ndarray, list[str]]]:
    q75 = np.nanpercentile(values, 75.0, axis=1)
    q25 = np.nanpercentile(values, 25.0, axis=1)
    return [
        (f"{prefix}_median", np.nanmedian(values, axis=1), _names(f"{prefix}_median", base_names)),
        (
            f"{prefix}_p90",
            np.nanpercentile(values, 90.0, axis=1),
            _names(f"{prefix}_p90", base_names),
        ),
        (f"{prefix}_iqr", q75 - q25, _names(f"{prefix}_iqr", base_names)),
        (f"{prefix}_mean", np.nanmean(values, axis=1), _names(f"{prefix}_mean", base_names)),
    ]


def _receiver_structure_blocks(
    receiver: np.ndarray,
    base_names: list[str],
    *,
    epsilon: float,
) -> list[tuple[str, np.ndarray, list[str]]]:
    near = receiver[:, 0, :]
    far = receiver[:, -1, :]
    q75 = np.nanpercentile(receiver, 75.0, axis=1)
    q25 = np.nanpercentile(receiver, 25.0, axis=1)
    return [
        (
            "tf_receiver_near_far_contrast",
            far - near,
            _names("tf_receiver_near_far_contrast", base_names),
        ),
        (
            "tf_receiver_near_far_ratio",
            far / (near + epsilon),
            _names("tf_receiver_near_far_ratio", base_names),
        ),
        (
            "tf_receiver_spectral_gradient",
            np.nanmean(np.diff(receiver, axis=1), axis=1),
            _names("tf_receiver_spectral_gradient", base_names),
        ),
        (
            "tf_receiver_spectral_decay_trend",
            _axis_slope(receiver),
            _names("tf_receiver_spectral_decay_trend", base_names),
        ),
        (
            "tf_receiver_median",
            np.nanmedian(receiver, axis=1),
            _names("tf_receiver_median", base_names),
        ),
        (
            "tf_receiver_p90",
            np.nanpercentile(receiver, 90.0, axis=1),
            _names("tf_receiver_p90", base_names),
        ),
        ("tf_receiver_iqr", q75 - q25, _names("tf_receiver_iqr", base_names)),
    ]


def _side_structure_blocks(
    side: np.ndarray,
    base_names: list[str],
    *,
    epsilon: float,
) -> list[tuple[str, np.ndarray, list[str]]]:
    side_abs = np.abs(side)
    side_sum = np.sum(side_abs, axis=1, keepdims=True)
    probabilities = side_abs / np.maximum(side_sum, epsilon)
    entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1) / np.log(
        side.shape[1]
    )
    q75 = np.nanpercentile(side, 75.0, axis=1)
    q25 = np.nanpercentile(side, 25.0, axis=1)
    median = np.nanmedian(side, axis=1)
    return [
        (
            "tf_side_contrast",
            np.nanmax(side, axis=1) - np.nanmin(side, axis=1),
            _names("tf_side_contrast", base_names),
        ),
        ("tf_side_entropy", entropy, _names("tf_side_entropy", base_names)),
        ("tf_side_median", median, _names("tf_side_median", base_names)),
        ("tf_side_p90", np.nanpercentile(side, 90.0, axis=1), _names("tf_side_p90", base_names)),
        ("tf_side_iqr", q75 - q25, _names("tf_side_iqr", base_names)),
        (
            "tf_side_max_minus_median",
            np.nanmax(side, axis=1) - median,
            _names("tf_side_max_minus_median", base_names),
        ),
    ]


def _axis_slope(values: np.ndarray) -> np.ndarray:
    axis = np.arange(values.shape[1], dtype=np.float32)
    centered = axis - float(np.mean(axis))
    denom = float(np.sum(centered * centered))
    if denom <= 0.0:
        return np.zeros((values.shape[0], values.shape[2]), dtype=np.float32)
    centered_values = values - np.nanmean(values, axis=1, keepdims=True)
    return np.nansum(centered_values * centered[None, :, None], axis=1) / denom


def _names(prefix: str, base_names: list[str]) -> list[str]:
    return [f"{prefix}_{name}" for name in base_names]


def _chunk_depth_count(*, cap_bytes: int, side_count: int, time_count: int) -> int:
    bytes_per_depth = max(side_count * time_count * 16, 1)
    return max(int(cap_bytes // bytes_per_depth), 1)


def _validate_trigger_scope(snapshot: dict[str, np.ndarray], errors: list[str]) -> None:
    for key, expected in RESEARCH_FLAGS.items():
        if key == "not_validated_for_deployment":
            continue
        value = snapshot.get(key)
        if value is None or bool(np.asarray(value).reshape(())) != expected:
            errors.append(f"snapshot flag {key} must be {expected}.")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_waveform_mapping(xsi: dict[str, Any]) -> None:
    required = (
        "waveform_variable_pattern",
        "depth_variable_pattern",
        "time_variable_pattern",
        "waveform_source_shape_order",
        "waveform_canonical_shape_order",
    )
    missing = [key for key in required if key not in xsi]
    if missing:
        raise ValueError("Waveform schema is unclear; missing mapping keys: " + ", ".join(missing))
    if _as_str_list(xsi["waveform_source_shape_order"]) != ["time", "depth"]:
        raise ValueError("Waveform schema is unclear; expected source order [time, depth].")
    if _as_str_list(xsi["waveform_canonical_shape_order"]) != ["depth", "time"]:
        raise ValueError("Waveform schema is unclear; expected canonical order [depth, time].")


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0, "finite_ratio": 0.0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "finite_ratio": float(finite.size / max(array.size, 1)),
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p50": float(np.quantile(finite, 0.50)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _finite_ratios(arrays: dict[str, np.ndarray]) -> dict[str, float | None]:
    ratios: dict[str, float | None] = {}
    for key, values in arrays.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.number):
            ratios[key] = None if array.size == 0 else float(np.isfinite(array).mean())
    return ratios


def _format_pattern(pattern: str, receiver: int, *, side: str | None = None) -> str:
    return pattern.format(receiver=receiver, side=side or "")


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Required NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_yaml(path: Path | str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any, default: list[str]) -> list[Any]:
    if isinstance(value, list):
        return value
    return default


def _as_str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(default or [])
