from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LABEL_REVIEW_VERSION = "label_review_v001"
REVIEW_FILENAMES = {
    "cast_zc_raw": "01_cast_zc_raw.png",
    "cast_zc_baseline": "02_cast_zc_baseline.png",
    "relative_drop": "03_relative_drop.png",
    "plus_overlay": "04_plus_candidate_overlay.png",
    "minus_overlay": "05_minus_ablation_overlay.png",
    "disagreement": "06_plus_minus_disagreement.png",
    "confidence": "07_confidence_map.png",
    "severity": "08_severity_map.png",
    "depth_coverage": "09_depth_coverage_summary.png",
}


@dataclass(frozen=True)
class LabelReviewReport:
    label_review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figures: dict[str, str]
    review_summary_template: str
    no_final_labels: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_label_review_figures(
    *,
    cast_label_input_npz: Path | str,
    cast_baseline_npz: Path | str,
    weak_label_npz: Path | str,
    output_dir: Path | str,
    overwrite: bool,
    max_depth_pixels: int = 1200,
) -> LabelReviewReport:
    input_arrays = _load_npz(cast_label_input_npz)
    baseline_arrays = _load_npz(cast_baseline_npz)
    label_arrays = _load_npz(weak_label_npz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []

    zc = np.asarray(input_arrays["cast_zc"], dtype=np.float32)
    zc_base = np.asarray(baseline_arrays["zc_base"], dtype=np.float32)
    relative_drop = np.asarray(baseline_arrays["relative_drop"], dtype=np.float32)
    plus = np.asarray(label_arrays["presence_plus"], dtype=np.int8) == 1
    minus = np.asarray(label_arrays["presence_minus_ablation"], dtype=np.int8) == 1
    confidence = np.asarray(label_arrays["label_confidence_plus"], dtype=np.float32)
    severity = np.asarray(label_arrays["severity_plus"], dtype=np.int8)
    no_final_labels = bool(np.asarray(label_arrays.get("no_final_labels", False)).reshape(()))
    if not no_final_labels:
        errors.append("Weak-label candidate NPZ does not set no_final_labels=true.")

    figures = {
        "cast_zc_raw": output / REVIEW_FILENAMES["cast_zc_raw"],
        "cast_zc_baseline": output / REVIEW_FILENAMES["cast_zc_baseline"],
        "relative_drop": output / REVIEW_FILENAMES["relative_drop"],
        "plus_overlay": output / REVIEW_FILENAMES["plus_overlay"],
        "minus_overlay": output / REVIEW_FILENAMES["minus_overlay"],
        "disagreement": output / REVIEW_FILENAMES["disagreement"],
        "confidence": output / REVIEW_FILENAMES["confidence"],
        "severity": output / REVIEW_FILENAMES["severity"],
        "depth_coverage": output / REVIEW_FILENAMES["depth_coverage"],
    }
    for path in figures.values():
        _ensure_can_write(path, overwrite=overwrite)

    _write_png(figures["cast_zc_raw"], _heatmap(zc, max_depth_pixels=max_depth_pixels))
    _write_png(
        figures["cast_zc_baseline"],
        _heatmap(zc_base, max_depth_pixels=max_depth_pixels),
    )
    _write_png(
        figures["relative_drop"],
        _heatmap(relative_drop, max_depth_pixels=max_depth_pixels, palette="drop"),
    )
    _write_png(
        figures["plus_overlay"],
        _overlay(_heatmap(zc, max_depth_pixels=max_depth_pixels), plus, (220, 30, 30)),
    )
    _write_png(
        figures["minus_overlay"],
        _overlay(_heatmap(zc, max_depth_pixels=max_depth_pixels), minus, (30, 90, 220)),
    )
    _write_png(
        figures["disagreement"],
        _mask_image(plus != minus, max_depth_pixels=max_depth_pixels, color=(180, 30, 180)),
    )
    _write_png(
        figures["confidence"],
        _heatmap(confidence, max_depth_pixels=max_depth_pixels, palette="confidence"),
    )
    _write_png(
        figures["severity"],
        _severity_image(severity, max_depth_pixels=max_depth_pixels),
    )
    _write_png(
        figures["depth_coverage"],
        _coverage_summary_image(plus, minus, max_depth_pixels=max_depth_pixels),
    )

    template = output / "review_summary_template.md"
    _ensure_can_write(template, overwrite=overwrite)
    template.write_text(_review_template(), encoding="utf-8")

    report = LabelReviewReport(
        label_review_version=LABEL_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "cast_label_input_npz": str(cast_label_input_npz),
            "cast_baseline_npz": str(cast_baseline_npz),
            "weak_label_npz": str(weak_label_npz),
        },
        output_dir=str(output),
        figures={key: str(path) for key, path in figures.items()},
        review_summary_template=str(template),
        no_final_labels=no_final_labels,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "final label approval",
            "feature extraction",
            "STFT",
            "STC",
            "APES",
            "model training",
            "MVP-4 correlation validation",
        ],
    )
    (output / "label_review_summary_v001.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _heatmap(
    values: np.ndarray,
    *,
    max_depth_pixels: int,
    palette: str = "gray",
) -> np.ndarray:
    array = _downsample_depth(np.asarray(values), max_depth_pixels=max_depth_pixels)
    scaled = _scale_to_unit(array)
    if palette == "drop":
        red = (255 * scaled).astype(np.uint8)
        green = (255 * (1.0 - 0.65 * scaled)).astype(np.uint8)
        blue = (255 * (1.0 - scaled)).astype(np.uint8)
    elif palette == "confidence":
        red = (40 * (1.0 - scaled)).astype(np.uint8)
        green = (180 * scaled + 30).astype(np.uint8)
        blue = (220 * (1.0 - 0.5 * scaled)).astype(np.uint8)
    else:
        gray = (255 * scaled).astype(np.uint8)
        red = gray
        green = gray
        blue = gray
    return np.stack([red, green, blue], axis=2)


def _overlay(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    resized_mask = _downsample_depth(mask.astype(np.float32), max_depth_pixels=base_rgb.shape[0])
    active = resized_mask > 0.5
    output = base_rgb.copy().astype(np.float32)
    color_array = np.asarray(color, dtype=np.float32)
    output[active] = (0.55 * output[active]) + (0.45 * color_array)
    return np.clip(output, 0, 255).astype(np.uint8)


def _mask_image(
    mask: np.ndarray,
    *,
    max_depth_pixels: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    resized = _downsample_depth(mask.astype(np.float32), max_depth_pixels=max_depth_pixels)
    image = np.full((resized.shape[0], resized.shape[1], 3), 245, dtype=np.uint8)
    image[resized > 0.5] = np.asarray(color, dtype=np.uint8)
    return image


def _severity_image(severity: np.ndarray, *, max_depth_pixels: int) -> np.ndarray:
    resized = _downsample_depth(severity.astype(np.float32), max_depth_pixels=max_depth_pixels)
    image = np.full((resized.shape[0], resized.shape[1], 3), 230, dtype=np.uint8)
    image[resized < 0] = np.asarray([80, 80, 80], dtype=np.uint8)
    image[resized == 1] = np.asarray([255, 210, 80], dtype=np.uint8)
    image[resized == 2] = np.asarray([245, 130, 50], dtype=np.uint8)
    image[resized >= 3] = np.asarray([200, 40, 40], dtype=np.uint8)
    return image


def _coverage_summary_image(
    plus: np.ndarray,
    minus: np.ndarray,
    *,
    max_depth_pixels: int,
) -> np.ndarray:
    plus_cov = np.mean(plus, axis=1)
    minus_cov = np.mean(minus, axis=1)
    indices = _sample_indices(plus_cov.size, min(max_depth_pixels, plus_cov.size))
    plus_cov = plus_cov[indices]
    minus_cov = minus_cov[indices]
    height = 240
    width = max(400, plus_cov.size)
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    x_positions = np.linspace(0, width - 1, num=plus_cov.size).astype(int)
    for x, value in zip(x_positions, plus_cov, strict=False):
        y = height - 1 - int(np.clip(value, 0.0, 1.0) * (height - 1))
        image[y:, x, :] = np.asarray([220, 40, 40], dtype=np.uint8)
    for x, value in zip(x_positions, minus_cov, strict=False):
        y = height - 1 - int(np.clip(value, 0.0, 1.0) * (height - 1))
        image[y:, x, 2] = 220
        image[y:, x, 0] = np.minimum(image[y:, x, 0], 80)
    return image


def _downsample_depth(values: np.ndarray, *, max_depth_pixels: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D image array, observed shape {array.shape}.")
    indices = _sample_indices(array.shape[0], min(max_depth_pixels, array.shape[0]))
    return array[indices]


def _sample_indices(length: int, target: int) -> np.ndarray:
    if length <= target:
        return np.arange(length)
    return np.linspace(0, length - 1, num=target).astype(int)


def _scale_to_unit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.float32)
    low, high = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    scaled = (array - low) / (high - low)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"PNG image must be RGB, observed shape {image.shape}.")
    height, width, _channels = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_rows, level=6)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _review_template() -> str:
    return "\n".join(
        [
            "# CAST Weak-Label Review Summary",
            "",
            "- Reviewer:",
            "- Review date:",
            "- Depth intervals inspected:",
            "- CAST.Zc raw image acceptable: TODO",
            "- Baseline image acceptable: TODO",
            "- Plus candidate overlay acceptable: TODO",
            "- Minus ablation overlay inspected: TODO",
            "- Disagreement areas requiring follow-up: TODO",
            "- Threshold confirmation required: alpha / zc_min_limit / severity bins",
            "- Final label approval: not allowed in MVP-3",
            "",
        ]
    )


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
