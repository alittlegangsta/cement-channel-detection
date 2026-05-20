from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cement_channel.alignment.azimuth_normalization import (
    align_azimuth_to_high_side,
    default_cast_azimuth_deg,
    default_xsi_side_azimuth_deg,
)

RELBEARING_VALIDATION_VERSION = "relbearing_sign_validation_v001"
CANDIDATES = ["plus", "minus", "no_rotation", "random_rotation"]
DOCUMENTATION_PREFERRED_CONCLUSION: dict[str, Any] = {
    "relbearing_sign_status": "documentation_preferred_plus_data_unresolved",
    "documentation_preferred_sign": "plus",
    "documentation_formula": "theta_aligned = (theta_raw + RelBearing) mod 360",
    "data_driven_validation": "insufficient_evidence",
    "single_sign_alignment_approved": False,
    "approved_downstream_mode": "plus_primary_minus_ablation",
    "documentation_basis": (
        "Halliburton Relative Bearing documentation suggests a plus convention when raw side "
        "azimuth is clockwise from tool key and measured looking downhole."
    ),
    "unconfirmed_assumptions": [
        "Side A-H ordering relative to tool key has not been independently confirmed.",
        "Exported matrix or image orientation may still include looking-uphole / looking-downhole "
        "flips.",
        "Data-driven plus/minus/no/random metrics are not sign-discriminative in the current "
        "small-slice evidence.",
    ],
}


@dataclass(frozen=True)
class RelBearingCandidateMetrics:
    candidate: str
    wrap_valid: bool
    azimuth_min: float | None
    azimuth_max: float | None
    circular_contrast: float | None
    score: float | None
    evidence_available: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelBearingValidationReport:
    validation_version: str
    generated_at: str
    inputs: dict[str, str]
    decision: str
    selected_convention: str | None
    confidence: float
    convention_conclusion: dict[str, Any]
    candidate_metrics: dict[str, RelBearingCandidateMetrics]
    orientation_confidence_summary: dict[str, float | int | None]
    aligned_azimuth_preview: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    manual_confirmation_required: bool
    mvp3_allowed_without_confirmation: bool
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_version": self.validation_version,
            "generated_at": self.generated_at,
            "inputs": self.inputs,
            "decision": self.decision,
            "selected_convention": self.selected_convention,
            "confidence": self.confidence,
            "convention_conclusion": self.convention_conclusion,
            "candidate_metrics": {
                key: value.to_dict() for key, value in self.candidate_metrics.items()
            },
            "orientation_confidence_summary": self.orientation_confidence_summary,
            "aligned_azimuth_preview": self.aligned_azimuth_preview,
            "warnings": self.warnings,
            "errors": self.errors,
            "manual_confirmation_required": self.manual_confirmation_required,
            "mvp3_allowed_without_confirmation": self.mvp3_allowed_without_confirmation,
            "not_performed": self.not_performed,
        }


def validate_relbearing_sign(
    *,
    depth_resample_preview_npz: Path | str,
    small_slice_summary_json: Path | str,
    depth_resample_report_json: Path | str,
    random_seed: int = 20260520,
) -> RelBearingValidationReport:
    warnings: list[str] = []
    errors: list[str] = []
    with np.load(depth_resample_preview_npz) as data:
        preview = {key: data[key] for key in data.files}
    resample_report = _read_json(depth_resample_report_json)
    small_slice_summary = _read_json(small_slice_summary_json)
    relbearing = _require_array(preview, "relbearing_deg_on_grid")
    inc = _require_array(preview, "inc_deg_on_grid")
    canonical_depth = _require_array(preview, "canonical_depth")
    cast_zc_preview = np.asarray(preview.get("small_slice_cast_zc_on_preview", np.empty((0, 0))))
    xsi_waveform_preview = np.asarray(
        preview.get("small_slice_xsi_waveform_on_preview", np.empty((0, 0, 0, 0)))
    )
    small_slice_status = _as_dict(resample_report.get("small_slice")).get("status")
    if small_slice_status != "completed":
        warnings.append(
            "Small-slice CAST/XSI azimuth preview is not available; "
            f"small_slice_status={small_slice_status}."
        )

    candidate_metrics = _candidate_metrics(
        relbearing,
        cast_zc_preview,
        xsi_waveform_preview,
        random_seed=random_seed,
    )
    orientation_summary = _orientation_confidence_summary(inc)
    aligned_preview = _aligned_azimuth_preview(canonical_depth, relbearing, random_seed=random_seed)
    decision, selected, confidence, decision_warnings = _decision(candidate_metrics)
    warnings.extend(decision_warnings)
    warnings.extend(str(item) for item in small_slice_summary.get("warnings", []) if item)
    warnings.extend(str(item) for item in resample_report.get("warnings", []) if item)
    errors.extend(str(item) for item in resample_report.get("errors", []) if item)
    return RelBearingValidationReport(
        validation_version=RELBEARING_VALIDATION_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "depth_resample_preview_npz": str(depth_resample_preview_npz),
            "small_slice_summary_json": str(small_slice_summary_json),
            "depth_resample_report_json": str(depth_resample_report_json),
        },
        decision=decision,
        selected_convention=selected,
        confidence=confidence,
        convention_conclusion=dict(DOCUMENTATION_PREFERRED_CONCLUSION),
        candidate_metrics=candidate_metrics,
        orientation_confidence_summary=orientation_summary,
        aligned_azimuth_preview=aligned_preview,
        warnings=warnings,
        errors=errors,
        manual_confirmation_required=True,
        mvp3_allowed_without_confirmation=False,
        not_performed=[
            "RelBearing final sign selection",
            "full waveform reading",
            "full CAST Zc reading",
            "weak label generation",
            "feature extraction",
            "STC/APES",
            "model training",
        ],
    )


def format_relbearing_validation_markdown(report: RelBearingValidationReport) -> str:
    data = report.to_dict()
    lines = [
        "# RelBearing Sign Validation Report",
        "",
        f"- Validation version: {data['validation_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Decision: {data['decision']}",
        f"- Selected convention: {data['selected_convention']}",
        f"- Confidence: {data['confidence']}",
        f"- Manual confirmation required: {data['manual_confirmation_required']}",
        f"- MVP-3 allowed without confirmation: {data['mvp3_allowed_without_confirmation']}",
        "",
        "## Convention Conclusion",
        "",
    ]
    conclusion = data["convention_conclusion"]
    lines.extend(
        [
            f"- RelBearing sign status: {conclusion['relbearing_sign_status']}",
            f"- Documentation preferred sign: {conclusion['documentation_preferred_sign']}",
            f"- Formula: {conclusion['documentation_formula']}",
            f"- Data-driven validation: {conclusion['data_driven_validation']}",
            (
                "- Single-sign alignment approved: "
                f"{conclusion['single_sign_alignment_approved']}"
            ),
            f"- Approved downstream mode: {conclusion['approved_downstream_mode']}",
            f"- Documentation basis: {conclusion['documentation_basis']}",
            "- Unconfirmed assumptions:",
        ]
    )
    lines.extend(f"  - {item}" for item in conclusion["unconfirmed_assumptions"])
    lines.extend(
        [
            "",
            "## Candidate Metrics",
            "",
        ]
    )
    for key, metric in data["candidate_metrics"].items():
        lines.append(
            f"- {key}: wrap_valid={metric['wrap_valid']}, "
            f"score={metric['score']}, evidence_available={metric['evidence_available']}, "
            f"notes={metric['notes']}"
        )
    lines.extend(["", "## Orientation Confidence", ""])
    for key, value in data["orientation_confidence_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Aligned Azimuth Preview", ""])
    lines.append(json.dumps(data["aligned_azimuth_preview"], ensure_ascii=False, indent=2))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def relbearing_config_dict(report: RelBearingValidationReport) -> dict[str, Any]:
    conclusion = report.convention_conclusion
    return {
        "schema_version": "schema_v001",
        "alignment_config_version": "alignment_relbearing_v001",
        "status": "requires_human_confirmation",
        "selected_convention": report.selected_convention or "unconfirmed",
        "relbearing_sign_status": conclusion["relbearing_sign_status"],
        "documentation_preferred_sign": conclusion["documentation_preferred_sign"],
        "documentation_formula": conclusion["documentation_formula"],
        "data_driven_validation": conclusion["data_driven_validation"],
        "single_sign_alignment_approved": conclusion["single_sign_alignment_approved"],
        "approved_downstream_mode": conclusion["approved_downstream_mode"],
        "documentation_basis": conclusion["documentation_basis"],
        "unconfirmed_assumptions": conclusion["unconfirmed_assumptions"],
        "allowed_candidate_conventions": CANDIDATES,
        "manual_confirmation_required": report.manual_confirmation_required,
        "mvp3_allowed_without_confirmation": report.mvp3_allowed_without_confirmation,
        "validation_decision": report.decision,
        "confidence": report.confidence,
        "notes": [
            "Do not mark plus as data-confirmed or production-approved.",
            "MVP-3 may proceed only with plus as documentation-preferred primary and minus as "
            "ablation/control.",
            "Do not use this config for single-sign production alignment.",
        ],
    }


def write_relbearing_validation_outputs(
    report: RelBearingValidationReport,
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
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_relbearing_validation_markdown(report), encoding="utf-8")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(relbearing_config_dict(report), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _candidate_metrics(
    relbearing_deg: np.ndarray,
    cast_zc_preview: np.ndarray,
    xsi_waveform_preview: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, RelBearingCandidateMetrics]:
    raw_cast = default_cast_azimuth_deg()
    metrics: dict[str, RelBearingCandidateMetrics] = {}
    for candidate in CANDIDATES:
        if candidate == "random_rotation":
            rng = np.random.default_rng(random_seed)
            offsets = rng.uniform(0.0, 360.0, size=max(relbearing_deg.size, 1)).astype(np.float32)
            azimuth = align_azimuth_to_high_side(
                raw_cast[None, :], offsets[:, None], convention="plus"
            )
        elif candidate == "no_rotation":
            azimuth = align_azimuth_to_high_side(raw_cast, 0.0, convention="no_rotation")
        else:
            azimuth = align_azimuth_to_high_side(
                raw_cast[None, :],
                relbearing_deg.reshape(-1, 1),
                convention=candidate,  # type: ignore[arg-type]
            )
        contrast = _circular_contrast(cast_zc_preview)
        xsi_contrast = _xsi_side_contrast(xsi_waveform_preview)
        evidence_available = (
            contrast is not None and cast_zc_preview.shape[0] > 0
        ) or xsi_contrast is not None
        notes = []
        score = None
        if evidence_available:
            notes.append("Overlap-targeted small-slice evidence is available.")
            if contrast is not None:
                notes.append(f"CAST circular contrast={contrast:.6g}.")
            if xsi_contrast is not None:
                notes.append(f"XSI side waveform contrast={xsi_contrast:.6g}.")
            notes.append(
                "Preview evidence is not sign-discriminative enough for automatic selection."
            )
        else:
            notes.append("No overlapping CAST/XSI azimuth preview evidence available.")
        metrics[candidate] = RelBearingCandidateMetrics(
            candidate=candidate,
            wrap_valid=_wrap_valid(azimuth),
            azimuth_min=float(np.nanmin(azimuth)) if np.asarray(azimuth).size else None,
            azimuth_max=float(np.nanmax(azimuth)) if np.asarray(azimuth).size else None,
            circular_contrast=contrast,
            score=score,
            evidence_available=evidence_available,
            notes=notes,
        )
    return metrics


def _aligned_azimuth_preview(
    canonical_depth: np.ndarray,
    relbearing_deg: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, Any]:
    raw_xsi = default_xsi_side_azimuth_deg()
    if canonical_depth.size == 0:
        indices: list[int] = []
    else:
        indices = sorted({0, int(canonical_depth.size // 2), int(canonical_depth.size - 1)})
    preview: dict[str, Any] = {
        "depth_indices": indices,
        "raw_xsi_side_azimuth_deg": raw_xsi.tolist(),
    }
    rng = np.random.default_rng(random_seed)
    for candidate in CANDIDATES:
        rows = []
        for index in indices:
            rel = float(relbearing_deg[index])
            if candidate == "random_rotation":
                rel = float(rng.uniform(0.0, 360.0))
                aligned = align_azimuth_to_high_side(raw_xsi, rel, convention="plus")
            elif candidate == "no_rotation":
                aligned = align_azimuth_to_high_side(raw_xsi, rel, convention="no_rotation")
            else:
                aligned = align_azimuth_to_high_side(
                    raw_xsi,
                    rel,
                    convention=candidate,  # type: ignore[arg-type]
                )
            rows.append(
                {
                    "depth": float(canonical_depth[index]),
                    "relbearing_deg": rel,
                    "xsi_side_azimuth_deg": np.asarray(aligned).round(6).tolist(),
                }
            )
        preview[candidate] = rows
    return preview


def _orientation_confidence_summary(inc_deg_on_grid: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(inc_deg_on_grid, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "inc_min": None,
            "inc_median": None,
            "inc_max": None,
            "low_inc_le_1deg_ratio": None,
            "stable_inc_ge_5deg_ratio": None,
        }
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "inc_min": float(np.min(finite)),
        "inc_median": float(np.median(finite)),
        "inc_max": float(np.max(finite)),
        "low_inc_le_1deg_ratio": float(np.mean(finite <= 1.0)),
        "stable_inc_ge_5deg_ratio": float(np.mean(finite >= 5.0)),
    }


def _decision(
    metrics: dict[str, RelBearingCandidateMetrics],
) -> tuple[str, str | None, float, list[str]]:
    if not any(
        metric.evidence_available and metric.score is not None for metric in metrics.values()
    ):
        return (
            "insufficient_evidence",
            None,
            0.0,
            [
                "Plus/minus RelBearing cannot be distinguished from available preview data; "
                "stop for human review or a dual-sign protocol."
            ],
        )
    return (
        "requires_human_confirmation",
        None,
        0.0,
        ["Candidate scoring exists but automatic selection is intentionally disabled."],
    )


def _circular_contrast(cast_zc_preview: np.ndarray) -> float | None:
    values = np.asarray(cast_zc_preview, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return None
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    per_depth = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    finite_contrast = per_depth[np.isfinite(per_depth)]
    if finite_contrast.size == 0:
        return None
    return float(np.median(finite_contrast))


def _xsi_side_contrast(xsi_waveform_preview: np.ndarray) -> float | None:
    values = np.asarray(xsi_waveform_preview, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] == 0 or values.shape[2] == 0:
        return None
    side_profile = np.nanmedian(np.abs(values), axis=(0, 1, 3))
    finite = side_profile[np.isfinite(side_profile)]
    if finite.size == 0:
        return None
    return float(np.max(finite) - np.min(finite))


def _wrap_valid(values: np.ndarray | float) -> bool:
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return False
    return bool(np.all((finite >= 0.0) & (finite < 360.0)))


def _require_array(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"Depth resample preview NPZ is missing array: {key}")
    return np.asarray(arrays[key])


def _read_json(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")
