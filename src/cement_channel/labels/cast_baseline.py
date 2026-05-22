from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter

from cement_channel.labels.cast_label_input import load_label_config, summarize_array

CAST_BASELINE_VERSION = "cast_zc_baseline_v001"


@dataclass(frozen=True)
class CastBaselineReport:
    cast_baseline_version: str
    generated_at: str
    inputs: dict[str, str]
    method: str
    window_m: float
    window_samples: int
    quantile: float
    min_finite_fraction: float
    depth_step_median: float | None
    arrays: dict[str, dict[str, Any]]
    baseline_valid_ratio: float | None
    relative_drop_positive_ratio: float | None
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cast_zc_baseline_from_config(
    *,
    cast_label_input_npz: Path | str,
    label_config_path: Path | str,
) -> tuple[CastBaselineReport, dict[str, np.ndarray]]:
    return build_cast_zc_baseline(
        cast_label_input_npz=cast_label_input_npz,
        label_config=load_label_config(label_config_path),
        label_config_path=label_config_path,
    )


def build_cast_zc_baseline(
    *,
    cast_label_input_npz: Path | str,
    label_config: dict[str, Any],
    label_config_path: Path | str | None = None,
) -> tuple[CastBaselineReport, dict[str, np.ndarray]]:
    with np.load(cast_label_input_npz) as data:
        cast_depth = np.asarray(data["cast_depth"], dtype=np.float32)
        cast_zc = np.asarray(data["cast_zc"], dtype=np.float32)
        cast_azimuth = np.asarray(data["cast_azimuth_deg"], dtype=np.float32)

    baseline_config = _as_dict(label_config.get("baseline"))
    method = str(baseline_config.get("method", "rolling_quantile"))
    window_m = float(baseline_config.get("window_m", 75.0))
    quantile = float(baseline_config.get("quantile", 0.90))
    min_finite_fraction = float(baseline_config.get("min_finite_fraction", 0.70))
    if method not in {"rolling_quantile", "rolling_median"}:
        raise ValueError(f"Unsupported CAST baseline method: {method}")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("baseline.quantile must be in (0, 1].")
    if window_m <= 0.0:
        raise ValueError("baseline.window_m must be positive.")

    depth_step = _median_depth_step(cast_depth)
    window_samples = _window_samples(window_m, depth_step)
    percentile = 50.0 if method == "rolling_median" else quantile * 100.0
    filled_zc = _fill_nan_by_depth(cast_depth, cast_zc)
    zc_base = percentile_filter(
        filled_zc,
        percentile=percentile,
        size=(window_samples, 1),
        mode="nearest",
    ).astype(np.float32)
    finite = np.isfinite(cast_zc)
    finite_fraction = uniform_filter(
        finite.astype(np.float32),
        size=(window_samples, 1),
        mode="nearest",
    ).astype(np.float32)
    baseline_valid = (
        finite
        & np.isfinite(zc_base)
        & (zc_base > 0.0)
        & (finite_fraction >= min_finite_fraction)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        zc_ratio = (cast_zc / zc_base).astype(np.float32)
        relative_drop = ((zc_base - cast_zc) / zc_base).astype(np.float32)
    zc_ratio = np.where(baseline_valid, zc_ratio, np.nan).astype(np.float32)
    relative_drop = np.where(baseline_valid, relative_drop, np.nan).astype(np.float32)
    zc_base = np.where(baseline_valid, zc_base, np.nan).astype(np.float32)

    arrays = {
        "cast_depth": cast_depth,
        "cast_azimuth_deg": cast_azimuth,
        "zc_base": zc_base,
        "relative_drop": relative_drop,
        "zc_ratio": zc_ratio,
        "baseline_valid": baseline_valid.astype(bool),
        "finite_fraction": finite_fraction.astype(np.float32),
    }
    warnings: list[str] = []
    errors: list[str] = []
    valid_ratio = _ratio(baseline_valid)
    if valid_ratio is not None and valid_ratio < 0.70:
        warnings.append("Baseline valid ratio is below 70%.")
    if np.nanmax(relative_drop) > 0.95:
        warnings.append("relative_drop contains values above 0.95; inspect outliers.")
    if depth_step is None:
        errors.append("Cannot estimate median CAST depth step.")

    report = CastBaselineReport(
        cast_baseline_version=CAST_BASELINE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "cast_label_input_npz": str(cast_label_input_npz),
            "label_config_path": str(label_config_path) if label_config_path is not None else "",
        },
        method=method,
        window_m=window_m,
        window_samples=window_samples,
        quantile=quantile,
        min_finite_fraction=min_finite_fraction,
        depth_step_median=depth_step,
        arrays={key: summarize_array(key, value).to_dict() for key, value in arrays.items()},
        baseline_valid_ratio=valid_ratio,
        relative_drop_positive_ratio=_ratio(relative_drop > 0.0),
        warnings=warnings,
        errors=errors,
        not_performed=[
            "azimuthal smoothing",
            "weak label generation",
            "final label generation",
            "feature extraction",
            "model training",
        ],
    )
    return report, arrays


def write_cast_baseline_outputs(
    report: CastBaselineReport,
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
    output_report_md.write_text(format_cast_baseline_markdown(report), encoding="utf-8")


def format_cast_baseline_markdown(report: CastBaselineReport) -> str:
    data = report.to_dict()
    lines = [
        "# CAST Zc Baseline Report",
        "",
        f"- Version: {data['cast_baseline_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Method: {data['method']}",
        f"- Window m: {data['window_m']}",
        f"- Window samples: {data['window_samples']}",
        f"- Quantile: {data['quantile']}",
        f"- Baseline valid ratio: {data['baseline_valid_ratio']}",
        "",
        "## Arrays",
        "",
    ]
    for name, summary in data["arrays"].items():
        lines.append(
            f"- {name}: shape={summary['shape']}, finite_ratio={summary['finite_ratio']}, "
            f"range=[{summary['min']}, {summary['max']}]"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _fill_nan_by_depth(depth: np.ndarray, values: np.ndarray) -> np.ndarray:
    filled = np.asarray(values, dtype=np.float32).copy()
    for column in range(filled.shape[1]):
        column_values = filled[:, column]
        finite = np.isfinite(column_values)
        if np.all(finite):
            continue
        if not np.any(finite):
            filled[:, column] = 0.0
            continue
        filled[:, column] = np.interp(
            depth.astype(np.float64),
            depth[finite].astype(np.float64),
            column_values[finite].astype(np.float64),
        ).astype(np.float32)
    return filled


def _median_depth_step(depth: np.ndarray) -> float | None:
    finite = np.asarray(depth, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return None
    diffs = np.diff(finite)
    diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0.0)])
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _window_samples(window_m: float, depth_step: float | None) -> int:
    if depth_step is None or depth_step <= 0.0:
        return 3
    samples = max(int(round(window_m / depth_step)), 3)
    if samples % 2 == 0:
        samples += 1
    return samples


def _ratio(mask: np.ndarray) -> float | None:
    values = np.asarray(mask)
    if values.size == 0:
        return None
    return float(np.mean(values))


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
