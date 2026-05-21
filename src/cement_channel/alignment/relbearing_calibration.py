from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from cement_channel.utils.angles import circular_distance_deg, signed_circular_delta_deg, wrap_deg

RELBearingSign = Literal["plus", "minus"]
XSISideOrder = Literal["clockwise", "counterclockwise"]
CASTAzimuthDirection = Literal["normal", "reversed"]

RELBearing_CALIBRATION_VERSION = "relbearing_calibration_v001"
SIDe_A_OFFSETS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


@dataclass(frozen=True)
class RelBearingHypothesis:
    relbearing_sign: RELBearingSign
    xsi_side_order: XSISideOrder
    cast_azimuth_direction: CASTAzimuthDirection
    side_a_offset_deg: float

    @property
    def hypothesis_id(self) -> str:
        return (
            f"sign={self.relbearing_sign}|side_order={self.xsi_side_order}|"
            f"cast_direction={self.cast_azimuth_direction}|side_a_offset={self.side_a_offset_deg:g}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"hypothesis_id": self.hypothesis_id}


@dataclass(frozen=True)
class CalibrationWindow:
    window_id: str
    start_index: int
    stop_index: int
    depth_min: float | None
    depth_max: float | None
    orientation_confidence_mean: float | None
    relbearing_jump_max_deg: float | None
    cast_azimuthal_contrast: float | None
    xsi_side_energy_contrast: float | None
    quality_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExcludedWindow:
    window_id: str
    start_index: int
    stop_index: int
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowHypothesisScore:
    window_id: str
    hypothesis_id: str
    score: float
    cast_theta_min_continuity: float
    xsi_strong_side_continuity: float
    cast_xsi_circular_consistency: float
    cast_azimuthal_contrast: float | None
    xsi_side_energy_contrast: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisScoreSummary:
    hypothesis: RelBearingHypothesis
    total_score: float
    mean_score: float
    window_vote_count: int
    support_ratio: float
    mean_cast_theta_min_continuity: float | None
    mean_xsi_strong_side_continuity: float | None
    mean_cast_xsi_circular_consistency: float | None
    mean_cast_azimuthal_contrast: float | None
    mean_xsi_side_energy_contrast: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "total_score": self.total_score,
            "mean_score": self.mean_score,
            "window_vote_count": self.window_vote_count,
            "support_ratio": self.support_ratio,
            "mean_cast_theta_min_continuity": self.mean_cast_theta_min_continuity,
            "mean_xsi_strong_side_continuity": self.mean_xsi_strong_side_continuity,
            "mean_cast_xsi_circular_consistency": self.mean_cast_xsi_circular_consistency,
            "mean_cast_azimuthal_contrast": self.mean_cast_azimuthal_contrast,
            "mean_xsi_side_energy_contrast": self.mean_xsi_side_energy_contrast,
        }


@dataclass
class RelBearingCalibrationReport:
    calibration_version: str
    generated_at: str
    inputs: dict[str, str]
    parameters: dict[str, Any]
    hypothesis_count: int
    valid_window_count: int
    valid_windows: list[CalibrationWindow]
    excluded_windows: list[ExcludedWindow]
    hypothesis_scores: dict[str, HypothesisScoreSummary]
    best_hypothesis: dict[str, Any] | None
    second_hypothesis: dict[str, Any] | None
    best_vs_second_score_gap: float | None
    multi_window_vote: dict[str, int]
    best_support_ratio: float | None
    attribute_votes: dict[str, dict[str, int]]
    attribute_support_ratio: dict[str, float | None]
    enough_to_suggest: dict[str, bool]
    final_recommendation: str
    recommendation: dict[str, Any] | None
    manual_confirmation_required: bool
    single_sign_alignment_approved: bool
    production_alignment_config_written: bool
    warnings: list[str]
    errors: list[str]
    figures: dict[str, str] = field(default_factory=dict)
    not_performed: list[str] = field(
        default_factory=lambda: [
            "production alignment config writing",
            "raw full XSI waveform reading",
            "raw full CAST Zc reading",
            "weak label generation",
            "feature extraction",
            "STC/APES",
            "model training",
            "MVP-3 execution",
        ],
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_version": self.calibration_version,
            "generated_at": self.generated_at,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "hypothesis_count": self.hypothesis_count,
            "valid_window_count": self.valid_window_count,
            "valid_windows": [window.to_dict() for window in self.valid_windows],
            "excluded_windows": [window.to_dict() for window in self.excluded_windows],
            "hypothesis_scores": {
                key: value.to_dict() for key, value in self.hypothesis_scores.items()
            },
            "best_hypothesis": self.best_hypothesis,
            "second_hypothesis": self.second_hypothesis,
            "best_vs_second_score_gap": self.best_vs_second_score_gap,
            "multi_window_vote": self.multi_window_vote,
            "best_support_ratio": self.best_support_ratio,
            "attribute_votes": self.attribute_votes,
            "attribute_support_ratio": self.attribute_support_ratio,
            "enough_to_suggest": self.enough_to_suggest,
            "final_recommendation": self.final_recommendation,
            "recommendation": self.recommendation,
            "manual_confirmation_required": self.manual_confirmation_required,
            "single_sign_alignment_approved": self.single_sign_alignment_approved,
            "production_alignment_config_written": self.production_alignment_config_written,
            "warnings": self.warnings,
            "errors": self.errors,
            "figures": self.figures,
            "not_performed": self.not_performed,
        }


def generate_hypotheses(
    *,
    side_a_offsets_deg: list[float] | None = None,
) -> list[RelBearingHypothesis]:
    offsets = side_a_offsets_deg or SIDe_A_OFFSETS_DEG
    return [
        RelBearingHypothesis(
            relbearing_sign=sign,  # type: ignore[arg-type]
            xsi_side_order=side_order,  # type: ignore[arg-type]
            cast_azimuth_direction=cast_direction,  # type: ignore[arg-type]
            side_a_offset_deg=float(offset),
        )
        for sign in ("plus", "minus")
        for side_order in ("clockwise", "counterclockwise")
        for cast_direction in ("normal", "reversed")
        for offset in offsets
    ]


def xsi_side_azimuth_deg(
    side_count: int,
    *,
    side_order: XSISideOrder,
    side_a_offset_deg: float,
) -> np.ndarray:
    if side_count <= 0:
        raise ValueError("side_count must be positive.")
    step = 360.0 / float(side_count)
    multiplier = 1.0 if side_order == "clockwise" else -1.0
    return np.asarray(
        wrap_deg(float(side_a_offset_deg) + multiplier * np.arange(side_count) * step),
        dtype=np.float32,
    )


def cast_azimuth_deg(
    azimuth_count: int,
    *,
    cast_azimuth_direction: CASTAzimuthDirection,
    raw_axis_deg: np.ndarray | None = None,
) -> np.ndarray:
    if raw_axis_deg is None:
        raw_axis = np.linspace(0.0, 360.0, num=azimuth_count, endpoint=False, dtype=np.float32)
    else:
        raw_axis = np.asarray(raw_axis_deg, dtype=np.float32).reshape(-1)
        if raw_axis.size != azimuth_count:
            raise ValueError("raw_axis_deg length must match azimuth_count.")
    if cast_azimuth_direction == "normal":
        return np.asarray(wrap_deg(raw_axis), dtype=np.float32)
    return np.asarray(wrap_deg(-raw_axis), dtype=np.float32)


def align_with_relbearing(
    theta_raw_deg: np.ndarray,
    relbearing_deg: np.ndarray,
    *,
    relbearing_sign: RELBearingSign,
) -> np.ndarray:
    theta = np.asarray(theta_raw_deg, dtype=np.float32)
    rel = np.asarray(relbearing_deg, dtype=np.float32)
    if relbearing_sign == "plus":
        return np.asarray(wrap_deg(theta + rel), dtype=np.float32)
    return np.asarray(wrap_deg(theta - rel), dtype=np.float32)


def xsi_side_rms_energy(xsi_waveform: np.ndarray) -> np.ndarray:
    values = np.asarray(xsi_waveform, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError("xsi_waveform must have shape [depth, receiver, side, time].")
    return np.sqrt(np.nanmean(values * values, axis=(1, 3))).astype(np.float32)


def select_calibration_windows(
    *,
    depth: np.ndarray,
    relbearing_deg: np.ndarray,
    orientation_confidence: np.ndarray,
    cast_zc: np.ndarray,
    xsi_side_energy: np.ndarray,
    window_depth_samples: int = 3,
    window_stride: int = 1,
    min_orientation_confidence: float = 0.5,
    max_relbearing_jump_deg: float = 45.0,
    min_cast_contrast: float = 0.0,
    min_xsi_contrast: float = 0.0,
    max_windows: int = 8,
) -> tuple[list[CalibrationWindow], list[ExcludedWindow]]:
    depth_values = np.asarray(depth, dtype=np.float32).reshape(-1)
    rel = np.asarray(relbearing_deg, dtype=np.float32).reshape(-1)
    orient = np.asarray(orientation_confidence, dtype=np.float32).reshape(-1)
    cast_values = np.asarray(cast_zc, dtype=np.float32)
    xsi_values = np.asarray(xsi_side_energy, dtype=np.float32)
    count = int(
        min(
            depth_values.size,
            rel.size,
            orient.size,
            cast_values.shape[0],
            xsi_values.shape[0],
        )
    )
    if count <= 0:
        return [], [ExcludedWindow("window_000", 0, 0, ["no common samples available"])]
    size = max(1, min(int(window_depth_samples), count))
    stride = max(1, int(window_stride))
    starts = list(range(0, max(count - size + 1, 1), stride))
    if starts[-1] != count - size:
        starts.append(count - size)

    valid: list[CalibrationWindow] = []
    excluded: list[ExcludedWindow] = []
    for window_number, start in enumerate(sorted(set(starts))):
        stop = min(start + size, count)
        window_id = f"window_{window_number:03d}"
        reasons: list[str] = []
        orient_mean = _nanmean_or_none(orient[start:stop])
        rel_jump = _max_relbearing_jump(rel[start:stop])
        cast_contrast = cast_azimuthal_contrast(cast_values[start:stop])
        xsi_contrast = xsi_side_energy_contrast(xsi_values[start:stop])
        if orient_mean is None or orient_mean < float(min_orientation_confidence):
            reasons.append("low_orientation_confidence")
        if rel_jump is not None and rel_jump > float(max_relbearing_jump_deg):
            reasons.append("relbearing_jump_too_large")
        if cast_contrast is None or cast_contrast <= float(min_cast_contrast):
            reasons.append("low_cast_azimuthal_contrast")
        if xsi_contrast is None or xsi_contrast <= float(min_xsi_contrast):
            reasons.append("low_xsi_side_energy_contrast")
        if reasons:
            excluded.append(ExcludedWindow(window_id, start, stop, reasons))
            continue
        valid.append(
            CalibrationWindow(
                window_id=window_id,
                start_index=start,
                stop_index=stop,
                depth_min=float(np.nanmin(depth_values[start:stop])),
                depth_max=float(np.nanmax(depth_values[start:stop])),
                orientation_confidence_mean=orient_mean,
                relbearing_jump_max_deg=rel_jump,
                cast_azimuthal_contrast=cast_contrast,
                xsi_side_energy_contrast=xsi_contrast,
                quality_score=float((cast_contrast or 0.0) + (xsi_contrast or 0.0)),
            )
        )
    valid = sorted(valid, key=lambda item: item.quality_score, reverse=True)
    return valid[: max(1, int(max_windows))], excluded


def calibrate_relbearing_convention(
    *,
    depth: np.ndarray,
    relbearing_deg: np.ndarray,
    orientation_confidence: np.ndarray,
    cast_zc: np.ndarray,
    xsi_waveform: np.ndarray,
    cast_azimuth_axis_deg: np.ndarray | None = None,
    inputs: dict[str, str] | None = None,
    window_depth_samples: int = 3,
    window_stride: int = 1,
    max_windows: int = 8,
    min_valid_windows: int = 5,
    min_orientation_confidence: float = 0.5,
    max_relbearing_jump_deg: float = 45.0,
    min_support_ratio: float = 0.70,
    min_score_gap: float = 0.05,
) -> tuple[RelBearingCalibrationReport, dict[str, np.ndarray]]:
    cast_values = np.asarray(cast_zc, dtype=np.float32)
    xsi_energy = xsi_side_rms_energy(xsi_waveform)
    common_count = min(
        np.asarray(depth).size,
        np.asarray(relbearing_deg).size,
        np.asarray(orientation_confidence).size,
        cast_values.shape[0],
        xsi_energy.shape[0],
    )
    arrays = {
        "depth": np.asarray(depth, dtype=np.float32).reshape(-1)[:common_count],
        "relbearing_deg": np.asarray(relbearing_deg, dtype=np.float32).reshape(-1)[:common_count],
        "orientation_confidence": np.asarray(orientation_confidence, dtype=np.float32).reshape(-1)[
            :common_count
        ],
        "cast_zc": cast_values[:common_count],
        "xsi_side_energy": xsi_energy[:common_count],
        "cast_azimuth_axis_deg": _cast_axis(cast_values, cast_azimuth_axis_deg),
    }
    warnings: list[str] = []
    errors: list[str] = []
    if common_count == 0:
        errors.append("No common depth samples available for RelBearing calibration.")

    valid_windows, excluded_windows = select_calibration_windows(
        depth=arrays["depth"],
        relbearing_deg=arrays["relbearing_deg"],
        orientation_confidence=arrays["orientation_confidence"],
        cast_zc=arrays["cast_zc"],
        xsi_side_energy=arrays["xsi_side_energy"],
        window_depth_samples=window_depth_samples,
        window_stride=window_stride,
        min_orientation_confidence=min_orientation_confidence,
        max_relbearing_jump_deg=max_relbearing_jump_deg,
        max_windows=max_windows,
    )
    if len(valid_windows) < min_valid_windows:
        warnings.append(
            f"Only {len(valid_windows)} valid calibration windows; at least {min_valid_windows} "
            "are required for a data-supported recommendation."
        )

    hypotheses = generate_hypotheses()
    hypothesis_scores, vote_counts = _score_hypotheses(
        hypotheses=hypotheses,
        windows=valid_windows,
        arrays=arrays,
    )
    sorted_scores = sorted(
        hypothesis_scores.values(),
        key=lambda item: item.total_score,
        reverse=True,
    )
    best = sorted_scores[0] if sorted_scores else None
    second = sorted_scores[1] if len(sorted_scores) > 1 else None
    score_gap = (
        float(best.total_score - second.total_score)
        if best is not None and second is not None
        else None
    )
    best_support_ratio = best.support_ratio if best is not None else None
    attribute_votes = _attribute_votes(hypothesis_scores)
    attribute_support = _attribute_support_ratio(attribute_votes, max(len(valid_windows), 1))
    enough = {
        "relbearing_sign": _enough(
            best_support_ratio,
            score_gap,
            len(valid_windows),
            min_valid_windows,
            min_support_ratio,
            min_score_gap,
        ),
        "xsi_side_order": _attribute_enough(
            attribute_support.get("xsi_side_order"),
            len(valid_windows),
            min_valid_windows,
            min_support_ratio,
        ),
        "cast_azimuth_direction": _attribute_enough(
            attribute_support.get("cast_azimuth_direction"),
            len(valid_windows),
            min_valid_windows,
            min_support_ratio,
        ),
        "side_a_offset_deg": _attribute_enough(
            attribute_support.get("side_a_offset_deg"),
            len(valid_windows),
            min_valid_windows,
            min_support_ratio,
        ),
    }
    final_recommendation, recommendation = _recommendation(best, enough)
    report = RelBearingCalibrationReport(
        calibration_version=RELBearing_CALIBRATION_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        parameters={
            "window_depth_samples": int(window_depth_samples),
            "window_stride": int(window_stride),
            "max_windows": int(max_windows),
            "min_valid_windows": int(min_valid_windows),
            "min_orientation_confidence": float(min_orientation_confidence),
            "max_relbearing_jump_deg": float(max_relbearing_jump_deg),
            "min_support_ratio": float(min_support_ratio),
            "min_score_gap": float(min_score_gap),
        },
        hypothesis_count=len(hypotheses),
        valid_window_count=len(valid_windows),
        valid_windows=valid_windows,
        excluded_windows=excluded_windows,
        hypothesis_scores=hypothesis_scores,
        best_hypothesis=best.to_dict() if best is not None else None,
        second_hypothesis=second.to_dict() if second is not None else None,
        best_vs_second_score_gap=score_gap,
        multi_window_vote=vote_counts,
        best_support_ratio=best_support_ratio,
        attribute_votes=attribute_votes,
        attribute_support_ratio=attribute_support,
        enough_to_suggest=enough,
        final_recommendation=final_recommendation,
        recommendation=recommendation,
        manual_confirmation_required=True,
        single_sign_alignment_approved=False,
        production_alignment_config_written=False,
        warnings=warnings,
        errors=errors,
    )
    return report, arrays


def build_calibration_report_from_files(
    *,
    depth_resample_overlap_preview_npz: Path | str,
    orientation_confidence_npz: Path | str,
    small_slice_overlap_npz: Path | str | None = None,
    relbearing_validation_report_json: Path | str | None = None,
    **kwargs: Any,
) -> tuple[RelBearingCalibrationReport, dict[str, np.ndarray]]:
    with np.load(depth_resample_overlap_preview_npz) as preview:
        depth = np.asarray(preview["small_slice_preview_depth"], dtype=np.float32)
        cast_zc = np.asarray(preview["small_slice_cast_zc_on_preview"], dtype=np.float32)
        xsi_waveform = np.asarray(
            preview["small_slice_xsi_waveform_on_preview"],
            dtype=np.float32,
        )
        canonical_depth = np.asarray(preview["canonical_depth"], dtype=np.float32)
        relbearing_on_grid = np.asarray(preview["relbearing_deg_on_grid"], dtype=np.float32)
    if depth.size == 0:
        fallback_count = min(canonical_depth.size, cast_zc.shape[0], xsi_waveform.shape[0])
        depth = canonical_depth[:fallback_count]
    relbearing = _interp_on_depth(canonical_depth, relbearing_on_grid, depth)

    with np.load(orientation_confidence_npz) as orientation:
        orient_depth = np.asarray(orientation["pose_depth"], dtype=np.float32)
        orient_conf = np.asarray(orientation["orientation_confidence"], dtype=np.float32)
    orientation_confidence = _interp_on_depth(orient_depth, orient_conf, depth)

    cast_axis: np.ndarray | None = None
    if small_slice_overlap_npz is not None and Path(small_slice_overlap_npz).exists():
        with np.load(small_slice_overlap_npz) as small:
            if "cast_azimuth_deg" in small.files:
                cast_axis = np.asarray(small["cast_azimuth_deg"], dtype=np.float32)
    validation_decision = None
    if (
        relbearing_validation_report_json is not None
        and Path(relbearing_validation_report_json).exists()
    ):
        data = json.loads(Path(relbearing_validation_report_json).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            validation_decision = data.get("decision")

    report, arrays = calibrate_relbearing_convention(
        depth=depth,
        relbearing_deg=relbearing,
        orientation_confidence=orientation_confidence,
        cast_zc=cast_zc,
        xsi_waveform=xsi_waveform,
        cast_azimuth_axis_deg=cast_axis,
        inputs={
            "depth_resample_overlap_preview_npz": str(depth_resample_overlap_preview_npz),
            "orientation_confidence_npz": str(orientation_confidence_npz),
            "small_slice_overlap_npz": str(small_slice_overlap_npz or ""),
            "relbearing_validation_report_json": str(relbearing_validation_report_json or ""),
            "relbearing_validation_decision": str(validation_decision),
        },
        **kwargs,
    )
    return report, arrays


def format_calibration_markdown(report: RelBearingCalibrationReport) -> str:
    data = report.to_dict()
    lines = [
        "# RelBearing Convention Calibration Report",
        "",
        f"- Version: {data['calibration_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Final recommendation: {data['final_recommendation']}",
        f"- Valid window count: {data['valid_window_count']}",
        f"- Best-vs-second score gap: {data['best_vs_second_score_gap']}",
        f"- Best support ratio: {data['best_support_ratio']}",
        f"- Manual confirmation required: {data['manual_confirmation_required']}",
        f"- Single-sign alignment approved: {data['single_sign_alignment_approved']}",
        "",
        "## Best Hypothesis",
        "",
        json.dumps(data["best_hypothesis"], ensure_ascii=False, indent=2),
        "",
        "## Second Hypothesis",
        "",
        json.dumps(data["second_hypothesis"], ensure_ascii=False, indent=2),
        "",
        "## Enough To Suggest",
        "",
    ]
    for key, value in data["enough_to_suggest"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Attribute Votes", ""])
    lines.append(json.dumps(data["attribute_votes"], ensure_ascii=False, indent=2))
    lines.extend(["", "## Hypothesis Scores", ""])
    for key, value in sorted(
        data["hypothesis_scores"].items(),
        key=lambda item: item[1]["total_score"],
        reverse=True,
    )[:10]:
        lines.append(
            f"- {key}: total={value['total_score']:.6f}, "
            f"mean={value['mean_score']:.6f}, votes={value['window_vote_count']}"
        )
    lines.extend(["", "## Valid Windows", ""])
    for window in data["valid_windows"]:
        lines.append(
            f"- {window['window_id']}: depth={window['depth_min']}..{window['depth_max']}, "
            f"cast_contrast={window['cast_azimuthal_contrast']}, "
            f"xsi_contrast={window['xsi_side_energy_contrast']}"
        )
    lines.extend(["", "## Excluded Windows", ""])
    if data["excluded_windows"]:
        for window in data["excluded_windows"]:
            lines.append(f"- {window['window_id']}: reasons={window['reasons']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Figures", ""])
    if data["figures"]:
        for key, value in data["figures"].items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def calibration_config_dict(report: RelBearingCalibrationReport) -> dict[str, Any]:
    data = report.to_dict()
    return {
        "schema_version": "schema_v001",
        "calibration_config_version": RELBearing_CALIBRATION_VERSION,
        "status": "manual_review_required",
        "production_alignment_config": "not_written",
        "single_sign_alignment_approved": False,
        "final_recommendation": data["final_recommendation"],
        "recommendation": data["recommendation"],
        "decision_rules": {
            "min_valid_windows": data["parameters"]["min_valid_windows"],
            "min_support_ratio": data["parameters"]["min_support_ratio"],
            "min_score_gap": data["parameters"]["min_score_gap"],
        },
        "hypothesis_space": {
            "relbearing_sign": ["plus", "minus"],
            "xsi_side_order": ["clockwise", "counterclockwise"],
            "cast_azimuth_direction": ["normal", "reversed"],
            "side_a_offset_deg": SIDe_A_OFFSETS_DEG,
        },
        "notes": [
            "This is a manual review/calibration example, not a production alignment config.",
            "Do not mark plus or minus as confirmed from this file alone.",
            "If unresolved, continue plus-primary / minus-ablation downstream mode only.",
        ],
    }


def write_calibration_outputs(
    report: RelBearingCalibrationReport,
    *,
    output_json: Path,
    output_md: Path,
    output_config: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_config, overwrite=overwrite)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_calibration_markdown(report), encoding="utf-8")
    output_config.write_text(
        yaml.safe_dump(calibration_config_dict(report), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def cast_azimuthal_contrast(cast_zc: np.ndarray) -> float | None:
    values = np.asarray(cast_zc, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    per_depth = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    finite = per_depth[np.isfinite(per_depth)]
    return float(np.median(finite)) if finite.size else None


def xsi_side_energy_contrast(xsi_side_energy: np.ndarray) -> float | None:
    values = np.asarray(xsi_side_energy, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    per_depth = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    finite = per_depth[np.isfinite(per_depth)]
    return float(np.median(finite)) if finite.size else None


def _score_hypotheses(
    *,
    hypotheses: list[RelBearingHypothesis],
    windows: list[CalibrationWindow],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, HypothesisScoreSummary], dict[str, int]]:
    per_hypothesis: dict[str, list[WindowHypothesisScore]] = {
        hypothesis.hypothesis_id: [] for hypothesis in hypotheses
    }
    votes = {hypothesis.hypothesis_id: 0 for hypothesis in hypotheses}
    for window in windows:
        window_scores = [
            _score_window_hypothesis(hypothesis, window, arrays) for hypothesis in hypotheses
        ]
        if window_scores:
            best = max(window_scores, key=lambda item: item.score)
            votes[best.hypothesis_id] += 1
        for score in window_scores:
            per_hypothesis[score.hypothesis_id].append(score)

    summary: dict[str, HypothesisScoreSummary] = {}
    denominator = max(len(windows), 1)
    hypothesis_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    for hypothesis_id, scores in per_hypothesis.items():
        total = float(sum(score.score for score in scores))
        summary[hypothesis_id] = HypothesisScoreSummary(
            hypothesis=hypothesis_by_id[hypothesis_id],
            total_score=total,
            mean_score=total / denominator,
            window_vote_count=votes[hypothesis_id],
            support_ratio=float(votes[hypothesis_id] / denominator),
            mean_cast_theta_min_continuity=_mean_attr(scores, "cast_theta_min_continuity"),
            mean_xsi_strong_side_continuity=_mean_attr(scores, "xsi_strong_side_continuity"),
            mean_cast_xsi_circular_consistency=_mean_attr(scores, "cast_xsi_circular_consistency"),
            mean_cast_azimuthal_contrast=_mean_optional_attr(scores, "cast_azimuthal_contrast"),
            mean_xsi_side_energy_contrast=_mean_optional_attr(
                scores,
                "xsi_side_energy_contrast",
            ),
        )
    return summary, votes


def _score_window_hypothesis(
    hypothesis: RelBearingHypothesis,
    window: CalibrationWindow,
    arrays: dict[str, np.ndarray],
) -> WindowHypothesisScore:
    window_slice = slice(window.start_index, window.stop_index)
    rel = arrays["relbearing_deg"][window_slice]
    cast_values = arrays["cast_zc"][window_slice]
    xsi_energy = arrays["xsi_side_energy"][window_slice]
    cast_axis = cast_azimuth_deg(
        cast_values.shape[1],
        cast_azimuth_direction=hypothesis.cast_azimuth_direction,
        raw_axis_deg=arrays["cast_azimuth_axis_deg"],
    )
    side_axis = xsi_side_azimuth_deg(
        xsi_energy.shape[1],
        side_order=hypothesis.xsi_side_order,
        side_a_offset_deg=hypothesis.side_a_offset_deg,
    )
    cast_theta = cast_theta_min_zc(cast_values, cast_axis)
    strong_theta = xsi_strong_side_theta(xsi_energy, side_axis)
    cast_aligned = align_with_relbearing(
        cast_theta,
        rel,
        relbearing_sign=hypothesis.relbearing_sign,
    )
    xsi_aligned = align_with_relbearing(
        strong_theta,
        rel,
        relbearing_sign=hypothesis.relbearing_sign,
    )
    cast_cont = circular_track_continuity(cast_aligned)
    xsi_cont = circular_track_continuity(xsi_aligned)
    consistency = circular_track_consistency(cast_aligned, xsi_aligned)
    score = float(
        0.25 * cast_cont
        + 0.25 * xsi_cont
        + 0.50 * consistency
        + _review_tie_breaker_prior(hypothesis)
    )
    return WindowHypothesisScore(
        window_id=window.window_id,
        hypothesis_id=hypothesis.hypothesis_id,
        score=score,
        cast_theta_min_continuity=cast_cont,
        xsi_strong_side_continuity=xsi_cont,
        cast_xsi_circular_consistency=consistency,
        cast_azimuthal_contrast=window.cast_azimuthal_contrast,
        xsi_side_energy_contrast=window.xsi_side_energy_contrast,
    )


def cast_theta_min_zc(cast_zc: np.ndarray, cast_axis_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(cast_zc, dtype=np.float32)
    axis = np.asarray(cast_axis_deg, dtype=np.float32).reshape(-1)
    if values.ndim != 2:
        raise ValueError("cast_zc must have shape [depth, cast_azimuth].")
    indices = np.nanargmin(values, axis=1)
    return axis[indices].astype(np.float32)


def xsi_strong_side_theta(xsi_side_energy: np.ndarray, side_axis_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(xsi_side_energy, dtype=np.float32)
    axis = np.asarray(side_axis_deg, dtype=np.float32).reshape(-1)
    if values.ndim != 2:
        raise ValueError("xsi_side_energy must have shape [depth, side].")
    indices = np.nanargmax(values, axis=1)
    return axis[indices].astype(np.float32)


def circular_track_continuity(theta_deg: np.ndarray) -> float:
    theta = np.asarray(theta_deg, dtype=np.float32).reshape(-1)
    finite = theta[np.isfinite(theta)]
    if finite.size < 2:
        return 1.0 if finite.size == 1 else 0.0
    distances = circular_distance_deg(finite[1:], finite[:-1])
    return float(np.clip(1.0 - float(np.nanmean(distances)) / 180.0, 0.0, 1.0))


def circular_track_consistency(theta_a_deg: np.ndarray, theta_b_deg: np.ndarray) -> float:
    a = np.asarray(theta_a_deg, dtype=np.float32).reshape(-1)
    b = np.asarray(theta_b_deg, dtype=np.float32).reshape(-1)
    count = min(a.size, b.size)
    if count == 0:
        return 0.0
    distances = circular_distance_deg(a[:count], b[:count])
    return float(np.clip(1.0 - float(np.nanmedian(distances)) / 180.0, 0.0, 1.0))


def _attribute_votes(
    hypothesis_scores: dict[str, HypothesisScoreSummary],
) -> dict[str, dict[str, int]]:
    votes = {
        "relbearing_sign": {"plus": 0, "minus": 0},
        "xsi_side_order": {"clockwise": 0, "counterclockwise": 0},
        "cast_azimuth_direction": {"normal": 0, "reversed": 0},
        "side_a_offset_deg": {f"{offset:g}": 0 for offset in SIDe_A_OFFSETS_DEG},
    }
    for score in hypothesis_scores.values():
        count = score.window_vote_count
        hyp = score.hypothesis
        votes["relbearing_sign"][hyp.relbearing_sign] += count
        votes["xsi_side_order"][hyp.xsi_side_order] += count
        votes["cast_azimuth_direction"][hyp.cast_azimuth_direction] += count
        votes["side_a_offset_deg"][f"{hyp.side_a_offset_deg:g}"] += count
    return votes


def _review_tie_breaker_prior(hypothesis: RelBearingHypothesis) -> float:
    prior = 0.0
    if hypothesis.xsi_side_order == "clockwise":
        prior += 0.01
    if hypothesis.cast_azimuth_direction == "normal":
        prior += 0.01
    if float(hypothesis.side_a_offset_deg) == 0.0:
        prior += 0.01
    return prior


def _attribute_support_ratio(
    votes: dict[str, dict[str, int]],
    denominator: int,
) -> dict[str, float | None]:
    if denominator <= 0:
        return {key: None for key in votes}
    return {
        key: float(max(value.values()) / denominator) if value else None
        for key, value in votes.items()
    }


def _recommendation(
    best: HypothesisScoreSummary | None,
    enough_to_suggest: dict[str, bool],
) -> tuple[str, dict[str, Any] | None]:
    if best is None or not enough_to_suggest.get("relbearing_sign", False):
        return "unresolved_keep_plus_primary_minus_ablation", None
    sign = best.hypothesis.relbearing_sign
    recommendation = {
        "hypothesis": best.hypothesis.to_dict(),
        "confidence": "data_supported_recommendation_not_confirmation",
        "single_sign_alignment_approved": False,
    }
    if sign == "plus":
        return "data_supported_plus_recommendation", recommendation
    return "data_supported_minus_recommendation", recommendation


def _enough(
    support_ratio: float | None,
    score_gap: float | None,
    valid_windows: int,
    min_valid_windows: int,
    min_support_ratio: float,
    min_score_gap: float,
) -> bool:
    return bool(
        valid_windows >= min_valid_windows
        and support_ratio is not None
        and support_ratio >= min_support_ratio
        and score_gap is not None
        and score_gap >= min_score_gap
    )


def _attribute_enough(
    support_ratio: float | None,
    valid_windows: int,
    min_valid_windows: int,
    min_support_ratio: float,
) -> bool:
    return bool(
        valid_windows >= min_valid_windows
        and support_ratio is not None
        and support_ratio >= min_support_ratio
    )


def _interp_on_depth(
    source_depth: np.ndarray,
    values: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_depth, dtype=np.float32).reshape(-1)
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    target = np.asarray(target_depth, dtype=np.float32).reshape(-1)
    count = min(source.size, vals.size)
    source = source[:count]
    vals = vals[:count]
    finite = np.isfinite(source) & np.isfinite(vals)
    if np.sum(finite) == 0:
        return np.zeros_like(target, dtype=np.float32)
    order = np.argsort(source[finite])
    return np.interp(target, source[finite][order], vals[finite][order]).astype(np.float32)


def _cast_axis(cast_zc: np.ndarray, raw_axis_deg: np.ndarray | None) -> np.ndarray:
    count = int(np.asarray(cast_zc).shape[1])
    if raw_axis_deg is None:
        return np.linspace(0.0, 360.0, num=count, endpoint=False, dtype=np.float32)
    axis = np.asarray(raw_axis_deg, dtype=np.float32).reshape(-1)
    if axis.size != count:
        return np.linspace(0.0, 360.0, num=count, endpoint=False, dtype=np.float32)
    return axis


def _max_relbearing_jump(relbearing_deg: np.ndarray) -> float | None:
    rel = np.asarray(relbearing_deg, dtype=np.float32).reshape(-1)
    finite = rel[np.isfinite(rel)]
    if finite.size < 2:
        return 0.0 if finite.size == 1 else None
    jumps = np.abs(signed_circular_delta_deg(finite[1:], finite[:-1]))
    return float(np.max(jumps)) if np.asarray(jumps).size else 0.0


def _nanmean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def _mean_attr(scores: list[WindowHypothesisScore], attr: str) -> float | None:
    if not scores:
        return None
    return float(np.mean([float(getattr(score, attr)) for score in scores]))


def _mean_optional_attr(scores: list[WindowHypothesisScore], attr: str) -> float | None:
    values = [getattr(score, attr) for score in scores if getattr(score, attr) is not None]
    if not values:
        return None
    return float(np.mean([float(value) for value in values]))


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")
