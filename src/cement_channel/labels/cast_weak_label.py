from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.labels.cast_label_input import load_label_config, summarize_array
from cement_channel.labels.schema import (
    CAST_WEAK_LABEL_VERSION,
    CONVENTION_STATUS,
    EvidenceFlag,
    LabelCandidateMetadata,
    LabelSource,
    PresenceLabel,
    SeverityLabel,
    validate_candidate_arrays,
    validate_candidate_metadata,
)

CAST_WEAK_LABEL_CANDIDATE_VERSION = "cast_weak_label_candidates_v001"


@dataclass(frozen=True)
class CastWeakLabelReport:
    cast_weak_label_candidate_version: str
    generated_at: str
    inputs: dict[str, str]
    label_version: str
    convention_status: str
    no_final_labels: bool
    threshold: dict[str, Any]
    severity_thresholds: dict[str, float]
    coverage: dict[str, float | None]
    confidence: dict[str, dict[str, Any]]
    arrays: dict[str, dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_cast_weak_labels_from_config(
    *,
    cast_label_input_npz: Path | str,
    cast_baseline_npz: Path | str,
    label_config_path: Path | str,
) -> tuple[CastWeakLabelReport, dict[str, np.ndarray]]:
    return generate_cast_weak_labels(
        cast_label_input_npz=cast_label_input_npz,
        cast_baseline_npz=cast_baseline_npz,
        label_config=load_label_config(label_config_path),
        label_config_path=label_config_path,
    )


def generate_cast_weak_labels(
    *,
    cast_label_input_npz: Path | str,
    cast_baseline_npz: Path | str,
    label_config: dict[str, Any],
    label_config_path: Path | str | None = None,
) -> tuple[CastWeakLabelReport, dict[str, np.ndarray]]:
    input_arrays = _load_npz(cast_label_input_npz)
    baseline_arrays = _load_npz(cast_baseline_npz)
    cast_depth = np.asarray(input_arrays["cast_depth"], dtype=np.float32)
    cast_zc = np.asarray(input_arrays["cast_zc"], dtype=np.float32)
    relbearing = np.asarray(input_arrays["relbearing_deg"], dtype=np.float32)
    orientation_confidence = np.asarray(
        input_arrays["orientation_confidence"],
        dtype=np.float32,
    )
    orientation_uncertain = np.asarray(input_arrays["orientation_uncertain"], dtype=bool)
    cast_azimuth = np.asarray(input_arrays["cast_azimuth_deg"], dtype=np.float32)
    zc_base = np.asarray(baseline_arrays["zc_base"], dtype=np.float32)
    relative_drop = np.asarray(baseline_arrays["relative_drop"], dtype=np.float32)
    zc_ratio = np.asarray(baseline_arrays["zc_ratio"], dtype=np.float32)
    baseline_valid = np.asarray(baseline_arrays["baseline_valid"], dtype=bool)

    warnings: list[str] = []
    errors: list[str] = []
    threshold = _threshold_config(label_config, warnings, errors)
    alpha = threshold["relative_drop_alpha"]
    zc_min_limit = threshold["zc_min_limit_effective"]
    severity_thresholds = _severity_thresholds(label_config)
    confidence_config = _as_dict(label_config.get("confidence"))

    finite_zc = np.isfinite(cast_zc)
    valid = finite_zc & baseline_valid & np.isfinite(relative_drop) & np.isfinite(zc_base)
    relative_rule = valid & (cast_zc < (zc_base * (1.0 - alpha)))
    absolute_rule = finite_zc & (cast_zc < zc_min_limit)
    candidate = valid & (relative_rule | absolute_rule)

    presence = np.full(cast_zc.shape, PresenceLabel.UNKNOWN, dtype=np.int8)
    presence[valid] = PresenceLabel.NO_CHANNEL_CANDIDATE
    presence[candidate] = PresenceLabel.CHANNEL_CANDIDATE
    severity = _severity_from_relative_drop(relative_drop, valid, candidate, severity_thresholds)
    evidence_flags = _evidence_flags(relative_rule, absolute_rule, valid)
    confidence = _label_confidence(
        cast_zc=cast_zc,
        relative_drop=relative_drop,
        candidate=candidate,
        valid=valid,
        orientation_confidence=orientation_confidence,
        orientation_uncertain=orientation_uncertain,
        config=confidence_config,
        alpha=alpha,
    )

    plus = _rotate_candidate_set(
        presence=presence,
        severity=severity,
        confidence=confidence,
        evidence_flags=evidence_flags,
        relative_drop=relative_drop,
        zc_ratio=zc_ratio,
        relbearing_deg=relbearing,
        cast_azimuth_deg=cast_azimuth,
        convention="plus",
    )
    minus = _rotate_candidate_set(
        presence=presence,
        severity=severity,
        confidence=confidence,
        evidence_flags=evidence_flags,
        relative_drop=relative_drop,
        zc_ratio=zc_ratio,
        relbearing_deg=relbearing,
        cast_azimuth_deg=cast_azimuth,
        convention="minus",
    )

    metadata_plus = LabelCandidateMetadata(
        label_version=CAST_WEAK_LABEL_VERSION,
        label_source=LabelSource.CAST_WEAK_PLUS.value,
        convention="plus",
        convention_status=CONVENTION_STATUS,
        no_final_labels=True,
    )
    metadata_minus = LabelCandidateMetadata(
        label_version=CAST_WEAK_LABEL_VERSION,
        label_source=LabelSource.CAST_WEAK_MINUS_ABLATION.value,
        convention="minus",
        convention_status=CONVENTION_STATUS,
        no_final_labels=True,
    )
    errors.extend(validate_candidate_metadata(metadata_plus).errors)
    errors.extend(validate_candidate_metadata(metadata_minus).errors)
    errors.extend(
        validate_candidate_arrays(
            presence=plus["presence"],
            severity=plus["severity"],
            label_confidence=plus["confidence"],
        ).errors
    )
    errors.extend(
        validate_candidate_arrays(
            presence=minus["presence"],
            severity=minus["severity"],
            label_confidence=minus["confidence"],
        ).errors
    )

    plus_coverage = _candidate_coverage(plus["presence"])
    minus_coverage = _candidate_coverage(minus["presence"])
    disagreement = _disagreement_rate(plus["presence"], minus["presence"])
    _coverage_warnings(
        plus_coverage,
        prefix="plus",
        threshold_config=_as_dict(label_config.get("threshold")),
        warnings=warnings,
        errors=errors,
    )
    _coverage_warnings(
        minus_coverage,
        prefix="minus_ablation",
        threshold_config=_as_dict(label_config.get("threshold")),
        warnings=warnings,
        errors=errors,
    )

    metadata = {
        "label_version": CAST_WEAK_LABEL_VERSION,
        "candidate_version": CAST_WEAK_LABEL_CANDIDATE_VERSION,
        "convention_status": CONVENTION_STATUS,
        "plus": metadata_plus.to_dict(),
        "minus_ablation": metadata_minus.to_dict(),
        "threshold": threshold,
        "no_final_labels": True,
    }
    arrays = {
        "cast_depth": cast_depth,
        "cast_azimuth_aligned_deg": cast_azimuth.astype(np.float32),
        "presence_plus": plus["presence"],
        "severity_plus": plus["severity"],
        "label_confidence_plus": plus["confidence"],
        "evidence_flags_plus": plus["evidence_flags"],
        "relative_drop_plus": plus["relative_drop"],
        "zc_ratio_plus": plus["zc_ratio"],
        "presence_minus_ablation": minus["presence"],
        "severity_minus_ablation": minus["severity"],
        "label_confidence_minus_ablation": minus["confidence"],
        "evidence_flags_minus_ablation": minus["evidence_flags"],
        "relative_drop_minus_ablation": minus["relative_drop"],
        "zc_ratio_minus_ablation": minus["zc_ratio"],
        "no_final_labels": np.asarray(True),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    report = CastWeakLabelReport(
        cast_weak_label_candidate_version=CAST_WEAK_LABEL_CANDIDATE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "cast_label_input_npz": str(cast_label_input_npz),
            "cast_baseline_npz": str(cast_baseline_npz),
            "label_config_path": str(label_config_path) if label_config_path is not None else "",
        },
        label_version=CAST_WEAK_LABEL_VERSION,
        convention_status=CONVENTION_STATUS,
        no_final_labels=True,
        threshold=threshold,
        severity_thresholds=severity_thresholds,
        coverage={
            "plus": plus_coverage,
            "minus_ablation": minus_coverage,
            "plus_minus_disagreement": disagreement,
        },
        confidence={
            "plus": summarize_array("label_confidence_plus", plus["confidence"]).to_dict(),
            "minus_ablation": summarize_array(
                "label_confidence_minus_ablation",
                minus["confidence"],
            ).to_dict(),
        },
        arrays={key: summarize_array(key, value).to_dict() for key, value in arrays.items()},
        warnings=warnings,
        errors=errors,
        not_performed=[
            "final label generation",
            "object continuity approval",
            "feature extraction",
            "STFT",
            "STC",
            "APES",
            "model training",
            "MVP-4 correlation validation",
        ],
    )
    return report, arrays


def write_cast_weak_label_outputs(
    report: CastWeakLabelReport,
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
    output_report_md.write_text(format_cast_weak_label_markdown(report), encoding="utf-8")


def format_cast_weak_label_markdown(report: CastWeakLabelReport) -> str:
    data = report.to_dict()
    lines = [
        "# CAST Weak-Label Candidate Report",
        "",
        f"- Version: {data['cast_weak_label_candidate_version']}",
        f"- Generated at: {data['generated_at']}",
        f"- Label version: {data['label_version']}",
        f"- Convention status: {data['convention_status']}",
        f"- No final labels: {data['no_final_labels']}",
        f"- Zc min limit effective: {data['threshold']['zc_min_limit_effective']}",
        f"- Zc min limit status: {data['threshold']['zc_min_limit_status']}",
        "",
        "## Coverage",
        "",
    ]
    for key, value in data["coverage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(data["warnings"]))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(data["errors"]))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(data["not_performed"]))
    lines.append("")
    return "\n".join(lines)


def _threshold_config(
    label_config: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    config = _as_dict(label_config.get("threshold"))
    alpha = float(config.get("relative_drop_alpha", 0.35))
    if not 0.0 < alpha < 1.0:
        errors.append("threshold.relative_drop_alpha must be in (0, 1).")
    raw_limit = config.get("zc_min_limit", "TODO_CONFIRM")
    require_confirmed = bool(config.get("require_confirmed_zc_min_limit", False))
    fallback = float(config.get("conservative_fallback_mrayl", 2.5))
    status = "confirmed"
    try:
        zc_min_limit = float(raw_limit)
    except (TypeError, ValueError):
        if require_confirmed:
            errors.append("threshold.zc_min_limit is not confirmed and fallback is disabled.")
            zc_min_limit = np.nan
            status = "missing_required"
        else:
            zc_min_limit = fallback
            status = "requires_human_threshold_confirmation"
            warnings.append(
                "threshold.zc_min_limit is TODO/unconfirmed; using conservative fallback "
                f"{fallback} MRayl for candidate generation."
            )
    return {
        "relative_drop_alpha": alpha,
        "zc_min_limit_raw": raw_limit,
        "zc_min_limit_effective": float(zc_min_limit),
        "zc_min_limit_status": status,
        "conservative_fallback_mrayl": fallback,
    }


def _severity_thresholds(label_config: dict[str, Any]) -> dict[str, float]:
    config = _as_dict(label_config.get("severity"))
    return {
        "mild_min_drop": float(config.get("mild_min_drop", 0.30)),
        "moderate_min_drop": float(config.get("moderate_min_drop", 0.45)),
        "severe_min_drop": float(config.get("severe_min_drop", 0.60)),
    }


def _severity_from_relative_drop(
    relative_drop: np.ndarray,
    valid: np.ndarray,
    candidate: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    severity = np.full(relative_drop.shape, SeverityLabel.UNKNOWN, dtype=np.int8)
    severity[valid] = SeverityLabel.NONE
    severity[candidate] = SeverityLabel.MILD
    severity[valid & (relative_drop >= thresholds["moderate_min_drop"])] = (
        SeverityLabel.MODERATE
    )
    severity[valid & (relative_drop >= thresholds["severe_min_drop"])] = SeverityLabel.SEVERE
    severity[valid & candidate & (relative_drop < thresholds["mild_min_drop"])] = (
        SeverityLabel.MILD
    )
    return severity


def _evidence_flags(
    relative_rule: np.ndarray,
    absolute_rule: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    flags = np.full(relative_rule.shape, int(EvidenceFlag.UNCERTAIN), dtype=np.int16)
    flags[valid] = int(EvidenceFlag.NONE)
    flags[relative_rule] |= int(EvidenceFlag.RELATIVE_DROP)
    flags[absolute_rule] |= int(EvidenceFlag.ABS_THRESHOLD)
    flags[relative_rule & absolute_rule] |= int(EvidenceFlag.FUSED_RULE)
    return flags


def _label_confidence(
    *,
    cast_zc: np.ndarray,
    relative_drop: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    orientation_confidence: np.ndarray,
    orientation_uncertain: np.ndarray,
    config: dict[str, Any],
    alpha: float,
) -> np.ndarray:
    full_drop = float(config.get("relative_drop_full_confidence", 0.70))
    non_candidate_scale = float(config.get("non_candidate_confidence_scale", 0.25))
    orientation_floor = float(config.get("orientation_floor", 0.05))
    strength = np.clip(relative_drop / max(full_drop, alpha), 0.0, 1.0)
    strength = np.where(candidate, np.maximum(strength, 0.25), strength * non_candidate_scale)
    orientation_weight = np.clip(orientation_confidence.reshape(-1, 1), 0.0, 1.0)
    if bool(config.get("low_inc_downweight", True)):
        orientation_weight = np.where(
            orientation_uncertain.reshape(-1, 1),
            np.minimum(orientation_weight, orientation_floor),
            orientation_weight,
        )
    outlier_weight = _outlier_weight(cast_zc, config)
    confidence = np.where(valid, strength * orientation_weight * outlier_weight, 0.0)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _outlier_weight(cast_zc: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if not bool(config.get("environment_outlier_downweight", True)):
        return np.ones(cast_zc.shape, dtype=np.float32)
    zc_min = float(config.get("outlier_zc_min", 0.0))
    zc_max = float(config.get("outlier_zc_max", 20.0))
    downweight = float(config.get("outlier_downweight", 0.50))
    outlier = (~np.isfinite(cast_zc)) | (cast_zc < zc_min) | (cast_zc > zc_max)
    return np.where(outlier, downweight, 1.0).astype(np.float32)


def _rotate_candidate_set(
    *,
    presence: np.ndarray,
    severity: np.ndarray,
    confidence: np.ndarray,
    evidence_flags: np.ndarray,
    relative_drop: np.ndarray,
    zc_ratio: np.ndarray,
    relbearing_deg: np.ndarray,
    cast_azimuth_deg: np.ndarray,
    convention: str,
) -> dict[str, np.ndarray]:
    return {
        "presence": _rotate_rows(presence, relbearing_deg, cast_azimuth_deg, convention).astype(
            np.int8
        ),
        "severity": _rotate_rows(severity, relbearing_deg, cast_azimuth_deg, convention).astype(
            np.int8
        ),
        "confidence": _rotate_rows(
            confidence,
            relbearing_deg,
            cast_azimuth_deg,
            convention,
        ).astype(np.float32),
        "evidence_flags": _rotate_rows(
            evidence_flags,
            relbearing_deg,
            cast_azimuth_deg,
            convention,
        ).astype(np.int16),
        "relative_drop": _rotate_rows(
            relative_drop,
            relbearing_deg,
            cast_azimuth_deg,
            convention,
        ).astype(np.float32),
        "zc_ratio": _rotate_rows(zc_ratio, relbearing_deg, cast_azimuth_deg, convention).astype(
            np.float32
        ),
    }


def _rotate_rows(
    values: np.ndarray,
    relbearing_deg: np.ndarray,
    cast_azimuth_deg: np.ndarray,
    convention: str,
) -> np.ndarray:
    array = np.asarray(values)
    output = np.empty_like(array)
    step = float(np.median(np.diff(cast_azimuth_deg))) if cast_azimuth_deg.size > 1 else 360.0
    direction = 1 if convention == "plus" else -1
    for row_index in range(array.shape[0]):
        relbearing = relbearing_deg[row_index] if row_index < relbearing_deg.size else np.nan
        offset_bins = (
            0 if not np.isfinite(relbearing) else int(round(direction * relbearing / step))
        )
        output[row_index] = np.roll(array[row_index], offset_bins)
    return output


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


def _coverage_warnings(
    coverage: float | None,
    *,
    prefix: str,
    threshold_config: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> None:
    if coverage is None:
        errors.append(f"{prefix} candidate coverage is undefined because no valid cells exist.")
        return
    warning_min = float(threshold_config.get("candidate_coverage_warning_min", 0.001))
    warning_max = float(threshold_config.get("candidate_coverage_warning_max", 0.40))
    blocking_min = float(threshold_config.get("candidate_coverage_blocking_min", 0.000001))
    blocking_max = float(threshold_config.get("candidate_coverage_blocking_max", 0.80))
    if coverage < blocking_min or coverage > blocking_max:
        errors.append(f"{prefix} candidate coverage is extreme: {coverage}.")
    elif coverage < warning_min or coverage > warning_max:
        warnings.append(f"{prefix} candidate coverage is outside warning range: {coverage}.")


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
