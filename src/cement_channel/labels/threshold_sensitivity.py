from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.labels.cast_label_input import load_label_config
from cement_channel.labels.cast_weak_label import generate_cast_weak_labels
from cement_channel.labels.label_audit import _component_summary
from cement_channel.labels.schema import EvidenceFlag, PresenceLabel, SeverityLabel

THRESHOLD_SENSITIVITY_VERSION = "label_threshold_sensitivity_v001"
DEFAULT_ALPHA_GRID = [0.30, 0.35, 0.40]
DEFAULT_ZC_MIN_LIMIT_GRID = [2.0, 2.5, 3.0]
DEFAULT_SEVERITY_THRESHOLD_SETS = {
    "default": [0.30, 0.45, 0.60],
    "conservative": [0.35, 0.50, 0.65],
    "aggressive": [0.25, 0.40, 0.55],
}


@dataclass(frozen=True)
class ThresholdSensitivityReport:
    threshold_sensitivity_version: str
    generated_at: str
    inputs: dict[str, str]
    grid: dict[str, Any]
    results: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    no_final_labels: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_threshold_sensitivity_from_config(
    *,
    cast_label_input_npz: Path | str,
    cast_baseline_npz: Path | str,
    label_config_path: Path | str,
) -> ThresholdSensitivityReport:
    return run_threshold_sensitivity(
        cast_label_input_npz=cast_label_input_npz,
        cast_baseline_npz=cast_baseline_npz,
        label_config=load_label_config(label_config_path),
        label_config_path=label_config_path,
    )


def run_threshold_sensitivity(
    *,
    cast_label_input_npz: Path | str,
    cast_baseline_npz: Path | str,
    label_config: dict[str, Any],
    label_config_path: Path | str | None = None,
    alpha_grid: list[float] | None = None,
    zc_min_limit_grid: list[float] | None = None,
    severity_threshold_sets: dict[str, list[float]] | None = None,
) -> ThresholdSensitivityReport:
    alphas = alpha_grid if alpha_grid is not None else DEFAULT_ALPHA_GRID
    zc_limits = zc_min_limit_grid if zc_min_limit_grid is not None else DEFAULT_ZC_MIN_LIMIT_GRID
    severity_sets = (
        severity_threshold_sets
        if severity_threshold_sets is not None
        else DEFAULT_SEVERITY_THRESHOLD_SETS
    )
    input_arrays = _load_npz(cast_label_input_npz)
    depth = np.asarray(input_arrays["cast_depth"], dtype=np.float32)
    azimuth = _cast_azimuth(input_arrays)
    isolated_max_pixels = int(
        _as_dict(label_config.get("audit")).get("isolated_object_max_pixels", 3)
    )

    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for alpha in alphas:
        for zc_min_limit in zc_limits:
            for severity_set_name, thresholds in severity_sets.items():
                run_config = _grid_label_config(
                    label_config,
                    alpha=alpha,
                    zc_min_limit=zc_min_limit,
                    severity_thresholds=thresholds,
                )
                report, arrays = generate_cast_weak_labels(
                    cast_label_input_npz=cast_label_input_npz,
                    cast_baseline_npz=cast_baseline_npz,
                    label_config=run_config,
                    label_config_path=label_config_path,
                )
                result = _summarize_run(
                    arrays,
                    alpha=alpha,
                    zc_min_limit=zc_min_limit,
                    severity_set_name=severity_set_name,
                    severity_thresholds=thresholds,
                    depth=depth,
                    azimuth=azimuth,
                    isolated_max_pixels=isolated_max_pixels,
                )
                result["warnings"] = report.warnings
                result["errors"] = report.errors
                warnings.extend(
                    f"alpha={alpha}, zc_min_limit={zc_min_limit}, "
                    f"severity_set={severity_set_name}: {message}"
                    for message in report.warnings
                )
                errors.extend(
                    f"alpha={alpha}, zc_min_limit={zc_min_limit}, "
                    f"severity_set={severity_set_name}: {message}"
                    for message in report.errors
                )
                results.append(result)

    return ThresholdSensitivityReport(
        threshold_sensitivity_version=THRESHOLD_SENSITIVITY_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "cast_label_input_npz": str(cast_label_input_npz),
            "cast_baseline_npz": str(cast_baseline_npz),
            "label_config_path": str(label_config_path) if label_config_path is not None else "",
        },
        grid={
            "alpha": [float(value) for value in alphas],
            "zc_min_limit": [float(value) for value in zc_limits],
            "severity_threshold_sets": {
                key: [float(item) for item in value] for key, value in severity_sets.items()
            },
        },
        results=results,
        warnings=warnings,
        errors=errors,
        no_final_labels=True,
        not_performed=[
            "final label generation",
            "manual threshold approval",
            "feature extraction",
            "STFT",
            "STC",
            "APES",
            "model training",
            "MVP-4 correlation validation",
        ],
    )


def write_threshold_sensitivity_outputs(
    report: ThresholdSensitivityReport,
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_report_csv: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_report_md, overwrite=overwrite)
    _ensure_can_write(output_report_json, overwrite=overwrite)
    _ensure_can_write(output_report_csv, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_report_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_threshold_sensitivity_markdown(report), encoding="utf-8")
    _write_csv(report.results, output_report_csv)


def format_threshold_sensitivity_markdown(report: ThresholdSensitivityReport) -> str:
    data = report.to_dict()
    lines = [
        "# CAST Weak-Label Threshold Sensitivity",
        "",
        f"- Version: {data['threshold_sensitivity_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- No final labels: {data['no_final_labels']}",
        f"- Runs: {len(data['results'])}",
        "",
        "## Grid",
        "",
        f"- alpha: {data['grid']['alpha']}",
        f"- zc_min_limit: {data['grid']['zc_min_limit']}",
        f"- severity threshold sets: {list(data['grid']['severity_threshold_sets'])}",
        "",
        "## Results",
        "",
        "| alpha | zc_min_limit | severity_set | plus_cov | minus_cov | disagree | "
        "mean_conf | low_conf | components | speckles | bad_zc | outlier_drop |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in data["results"]:
        lines.append(
            f"| {item['alpha']:.2f} | {item['zc_min_limit']:.2f} | "
            f"{item['severity_set']} | {_fmt(item['plus_coverage'])} | "
            f"{_fmt(item['minus_coverage'])} | {_fmt(item['plus_minus_disagreement'])} | "
            f"{_fmt(item['mean_label_confidence'])} | "
            f"{_fmt(item['low_confidence_fraction'])} | "
            f"{item['connected_component_count']} | "
            f"{_fmt(item['isolated_speckle_ratio'])} | "
            f"{_fmt(item['invalid_bad_zc_fraction'])} | "
            f"{_fmt(item['relative_drop_outlier_fraction'])} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _grid_label_config(
    label_config: dict[str, Any],
    *,
    alpha: float,
    zc_min_limit: float,
    severity_thresholds: list[float],
) -> dict[str, Any]:
    config = copy.deepcopy(label_config)
    threshold = _as_dict(config.setdefault("threshold", {}))
    threshold["relative_drop_alpha"] = float(alpha)
    threshold["zc_min_limit"] = float(zc_min_limit)
    threshold["require_confirmed_zc_min_limit"] = False
    severity = _as_dict(config.setdefault("severity", {}))
    severity["mild_min_drop"] = float(severity_thresholds[0])
    severity["moderate_min_drop"] = float(severity_thresholds[1])
    severity["severe_min_drop"] = float(severity_thresholds[2])
    return config


def _summarize_run(
    arrays: dict[str, np.ndarray],
    *,
    alpha: float,
    zc_min_limit: float,
    severity_set_name: str,
    severity_thresholds: list[float],
    depth: np.ndarray,
    azimuth: np.ndarray,
    isolated_max_pixels: int,
) -> dict[str, Any]:
    presence_plus = np.asarray(arrays["presence_plus"], dtype=np.int8)
    presence_minus = np.asarray(arrays["presence_minus_ablation"], dtype=np.int8)
    flags_plus = np.asarray(arrays["evidence_flags_plus"], dtype=np.int16)
    confidence_plus = np.asarray(arrays["label_confidence_plus"], dtype=np.float32)
    severity_plus = np.asarray(arrays["severity_plus"], dtype=np.int8)
    candidate_plus = presence_plus == PresenceLabel.CHANNEL_CANDIDATE
    valid_plus = presence_plus != PresenceLabel.UNKNOWN
    relative_flag = (flags_plus & int(EvidenceFlag.RELATIVE_DROP)) > 0
    absolute_flag = (flags_plus & int(EvidenceFlag.ABS_THRESHOLD)) > 0
    components = _component_summary(
        candidate_plus,
        depth,
        azimuth,
        isolated_max_pixels=isolated_max_pixels,
    )
    low_conf_threshold = 0.25
    return {
        "alpha": float(alpha),
        "zc_min_limit": float(zc_min_limit),
        "severity_set": severity_set_name,
        "severity_thresholds": [float(item) for item in severity_thresholds],
        "plus_coverage": _candidate_coverage(presence_plus),
        "minus_coverage": _candidate_coverage(presence_minus),
        "plus_minus_disagreement": _disagreement_rate(presence_plus, presence_minus),
        "relative_drop_only_coverage": _valid_fraction(
            relative_flag & ~absolute_flag,
            valid_plus,
        ),
        "zc_min_limit_only_coverage": _valid_fraction(
            absolute_flag & ~relative_flag,
            valid_plus,
        ),
        "both_triggered_coverage": _valid_fraction(relative_flag & absolute_flag, valid_plus),
        "mean_label_confidence": _candidate_mean(confidence_plus, candidate_plus),
        "low_confidence_fraction": _candidate_fraction(
            confidence_plus < low_conf_threshold,
            candidate_plus,
        ),
        "severity_distribution": _severity_distribution(severity_plus),
        "connected_component_count": components.component_count,
        "isolated_speckle_ratio": components.isolated_speckle_ratio,
        "relative_drop_outlier_fraction": _mask_fraction(
            arrays.get("relative_drop_outlier_plus"),
        ),
        "invalid_bad_zc_fraction": _mask_fraction(arrays.get("bad_data_mask_plus")),
        "no_final_labels": bool(np.asarray(arrays.get("no_final_labels", False)).reshape(())),
    }


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "alpha",
        "zc_min_limit",
        "severity_set",
        "severity_thresholds",
        "plus_coverage",
        "minus_coverage",
        "plus_minus_disagreement",
        "relative_drop_only_coverage",
        "zc_min_limit_only_coverage",
        "both_triggered_coverage",
        "mean_label_confidence",
        "low_confidence_fraction",
        "severity_distribution",
        "connected_component_count",
        "isolated_speckle_ratio",
        "relative_drop_outlier_fraction",
        "invalid_bad_zc_fraction",
        "no_final_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fieldnames}
            row["severity_thresholds"] = json.dumps(row["severity_thresholds"])
            row["severity_distribution"] = json.dumps(row["severity_distribution"])
            writer.writerow(row)


def _candidate_coverage(presence: np.ndarray) -> float | None:
    valid = presence != PresenceLabel.UNKNOWN
    if not np.any(valid):
        return None
    return float(np.mean(presence[valid] == PresenceLabel.CHANNEL_CANDIDATE))


def _disagreement_rate(presence_a: np.ndarray, presence_b: np.ndarray) -> float | None:
    valid = (presence_a != PresenceLabel.UNKNOWN) & (presence_b != PresenceLabel.UNKNOWN)
    if not np.any(valid):
        return None
    candidate_a = presence_a == PresenceLabel.CHANNEL_CANDIDATE
    candidate_b = presence_b == PresenceLabel.CHANNEL_CANDIDATE
    return float(np.mean(candidate_a[valid] != candidate_b[valid]))


def _valid_fraction(mask: np.ndarray, valid: np.ndarray) -> float | None:
    if not np.any(valid):
        return None
    return float(np.mean(np.asarray(mask, dtype=bool)[valid]))


def _candidate_mean(values: np.ndarray, candidate: np.ndarray) -> float | None:
    if not np.any(candidate):
        return None
    finite = np.asarray(values, dtype=np.float32)[candidate]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _candidate_fraction(mask: np.ndarray, candidate: np.ndarray) -> float | None:
    if not np.any(candidate):
        return None
    return float(np.mean(np.asarray(mask, dtype=bool)[candidate]))


def _severity_distribution(severity: np.ndarray) -> dict[str, int]:
    return {
        "unknown": int(np.count_nonzero(severity == SeverityLabel.UNKNOWN)),
        "none": int(np.count_nonzero(severity == SeverityLabel.NONE)),
        "mild": int(np.count_nonzero(severity == SeverityLabel.MILD)),
        "moderate": int(np.count_nonzero(severity == SeverityLabel.MODERATE)),
        "severe": int(np.count_nonzero(severity == SeverityLabel.SEVERE)),
    }


def _mask_fraction(mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    array = np.asarray(mask, dtype=bool)
    if array.size == 0:
        return None
    return float(np.mean(array))


def _cast_azimuth(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "cast_azimuth_aligned_deg" in arrays:
        return np.asarray(arrays["cast_azimuth_aligned_deg"], dtype=np.float32)
    if "cast_azimuth_deg" in arrays:
        return np.asarray(arrays["cast_azimuth_deg"], dtype=np.float32)
    width = np.asarray(arrays["cast_zc"]).shape[1]
    return np.linspace(0.0, 360.0, num=width, endpoint=False, dtype=np.float32)


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


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6g}"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
