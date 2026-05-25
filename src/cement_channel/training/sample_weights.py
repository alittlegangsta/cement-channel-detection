from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.training.depth_splits import make_depth_block_splits

SAMPLE_WEIGHT_POLICY_VERSION = "sample_weight_policy_v001"

WEIGHT_POLICY_NAMES = (
    "confidence_only",
    "class_balanced_confidence",
    "capped_class_balanced_confidence",
    "unweighted",
)


@dataclass(frozen=True)
class SampleWeightPolicyConfig:
    min_label_confidence: float = 0.5
    max_depth_match_error_ft: float = 0.5
    exclude_large_depth_match_error: bool = True
    disagreement_policy: str = "downweight"
    disagreement_weight_multiplier: float = 0.5
    target_candidate_weight_fraction: float = 0.5
    max_candidate_weight_fraction: float = 0.6
    capped_weight_quantile: float = 0.99
    default_policy: str = "capped_class_balanced_confidence"
    n_splits: int = 3
    depth_block_size_ft: float | None = 250.0
    min_gap_ft: float = 5.0
    min_samples_per_class_per_fold: int = 10

    def validate(self) -> None:
        if not 0.0 <= self.min_label_confidence <= 1.0:
            raise ValueError("min_label_confidence must be in [0, 1].")
        if self.max_depth_match_error_ft < 0.0:
            raise ValueError("max_depth_match_error_ft must be non-negative.")
        if self.disagreement_policy not in {"include", "downweight", "exclude"}:
            raise ValueError("disagreement_policy must be include, downweight, or exclude.")
        if not 0.0 <= self.disagreement_weight_multiplier <= 1.0:
            raise ValueError("disagreement_weight_multiplier must be in [0, 1].")
        if not 0.0 < self.target_candidate_weight_fraction < 1.0:
            raise ValueError("target_candidate_weight_fraction must be in (0, 1).")
        if not 0.0 < self.max_candidate_weight_fraction < 1.0:
            raise ValueError("max_candidate_weight_fraction must be in (0, 1).")
        if self.target_candidate_weight_fraction > self.max_candidate_weight_fraction:
            raise ValueError(
                "target_candidate_weight_fraction must not exceed "
                "max_candidate_weight_fraction."
            )
        if not 0.0 < self.capped_weight_quantile <= 1.0:
            raise ValueError("capped_weight_quantile must be in (0, 1].")
        if self.default_policy not in WEIGHT_POLICY_NAMES:
            raise ValueError(f"default_policy must be one of {WEIGHT_POLICY_NAMES}.")
        if self.n_splits <= 1:
            raise ValueError("n_splits must be greater than 1.")
        if self.depth_block_size_ft is not None and self.depth_block_size_ft <= 0.0:
            raise ValueError("depth_block_size_ft must be positive when provided.")
        if self.min_gap_ft < 0.0:
            raise ValueError("min_gap_ft must be non-negative.")


@dataclass(frozen=True)
class SampleWeightPolicyReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    config: dict[str, Any]
    selected_default_policy: str
    sample_counts: dict[str, int]
    policy_summary: dict[str, dict[str, float | int | None]]
    per_fold_effective_weight_balance: list[dict[str, Any]]
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_sample_weight_policy_config(config_path: Path | str) -> SampleWeightPolicyConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample table config does not exist: {path}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load sample weight policy config.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Sample table config must be a YAML mapping: {path}")
    raw = _as_dict(data.get("sample_weight_policy"))
    split = _as_dict(data.get("split"))
    sample_policy = _as_dict(data.get("sample_policy"))
    config = SampleWeightPolicyConfig(
        min_label_confidence=float(
            raw.get("min_label_confidence", sample_policy.get("min_label_confidence", 0.5))
        ),
        max_depth_match_error_ft=float(
            raw.get(
                "max_depth_match_error_ft",
                sample_policy.get("max_depth_match_error_ft", 0.5),
            )
        ),
        exclude_large_depth_match_error=bool(
            raw.get(
                "exclude_large_depth_match_error",
                sample_policy.get("exclude_large_depth_match_error", True),
            )
        ),
        disagreement_policy=str(raw.get("disagreement_policy", "downweight")),
        disagreement_weight_multiplier=float(raw.get("disagreement_weight_multiplier", 0.5)),
        target_candidate_weight_fraction=float(
            raw.get("target_candidate_weight_fraction", 0.5)
        ),
        max_candidate_weight_fraction=float(raw.get("max_candidate_weight_fraction", 0.6)),
        capped_weight_quantile=float(raw.get("capped_weight_quantile", 0.99)),
        default_policy=str(raw.get("default_policy", "capped_class_balanced_confidence")),
        n_splits=int(split.get("n_splits", raw.get("n_splits", 3))),
        depth_block_size_ft=_optional_float(
            split.get("depth_block_size_ft", raw.get("depth_block_size_ft", 250.0))
        ),
        min_gap_ft=float(split.get("min_gap_ft", raw.get("min_gap_ft", 5.0))),
        min_samples_per_class_per_fold=int(
            split.get(
                "min_samples_per_class_per_fold",
                raw.get("min_samples_per_class_per_fold", 10),
            )
        ),
    )
    config.validate()
    return config


def rebuild_sample_weights_from_npz(
    *,
    input_npz: Path | str,
    output_npz: Path | str,
    report_md: Path | str,
    report_json: Path | str,
    config: SampleWeightPolicyConfig,
    config_path: Path | str | None = None,
    overwrite: bool = False,
) -> SampleWeightPolicyReport:
    arrays = _load_npz(input_npz)
    updated, report = rebuild_sample_weights(
        arrays,
        config=config,
        input_npz=Path(input_npz),
        output_npz=Path(output_npz),
        config_path=Path(config_path) if config_path else None,
    )
    write_reweighted_sample_table(updated, Path(output_npz), overwrite=overwrite)
    write_sample_weight_policy_report(
        report,
        output_md=Path(report_md),
        output_json=Path(report_json),
        overwrite=overwrite,
    )
    return report


def rebuild_sample_weights(
    arrays: dict[str, np.ndarray],
    *,
    config: SampleWeightPolicyConfig,
    input_npz: Path | None = None,
    output_npz: Path | None = None,
    config_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], SampleWeightPolicyReport]:
    config.validate()
    depth = _required_array(arrays, "depth").astype(np.float32).reshape(-1)
    labels = _required_array(arrays, "label_presence_plus").astype(np.int8).reshape(-1)
    confidence = _required_array(arrays, "label_confidence_plus").astype(np.float32).reshape(-1)
    disagreement = _optional_bool_array(arrays, "plus_minus_disagreement", depth.size)
    valid_for_azimuthal = _optional_bool_array(
        arrays,
        "valid_for_azimuthal_validation",
        depth.size,
        default=True,
    )
    depth_match_error = _optional_float_array(arrays, "depth_match_error", depth.size)
    large_depth_error = _optional_bool_array(
        arrays,
        "exclude_large_depth_match_error",
        depth.size,
        default=False,
    )
    shape_errors = _shape_errors(
        depth=depth,
        labels=labels,
        confidence=confidence,
        disagreement=disagreement,
        valid_for_azimuthal=valid_for_azimuthal,
        depth_match_error=depth_match_error,
        large_depth_error=large_depth_error,
    )
    if shape_errors:
        raise ValueError("; ".join(shape_errors))

    base_mask = build_azimuthal_valid_mask(
        labels=labels,
        confidence=confidence,
        valid_for_azimuthal=valid_for_azimuthal,
        depth_match_error=depth_match_error,
        large_depth_error=large_depth_error,
        config=config,
    )
    reliability = build_reliability(
        confidence=confidence,
        labels=labels,
        valid_mask=base_mask,
        disagreement=disagreement,
        config=config,
    )
    policies = build_policy_weights(
        labels=labels,
        reliability=reliability,
        disagreement=disagreement,
        config=config,
    )
    updated = dict(arrays)
    if "sample_weight" in arrays:
        updated["sample_weight_original"] = np.asarray(arrays["sample_weight"], dtype=np.float32)
    updated["sample_weight_policy_names"] = np.asarray(WEIGHT_POLICY_NAMES)
    for name, weights in policies.items():
        updated[f"sample_weight_{name}"] = weights.astype(np.float32)
    updated["sample_weight"] = policies[config.default_policy].astype(np.float32)
    updated["sample_weight_policy_version"] = np.asarray(SAMPLE_WEIGHT_POLICY_VERSION)
    updated["sample_weight_default_policy"] = np.asarray(config.default_policy)
    updated["no_final_labels"] = np.asarray(True)
    updated["no_stc"] = np.asarray(True)
    updated["no_apes"] = np.asarray(True)

    warnings, errors = _policy_warnings_and_errors(
        labels=labels,
        valid_mask=base_mask,
        policies=policies,
        config=config,
    )
    policy_summary = {
        name: summarize_policy_weights(labels, disagreement, base_mask, weights)
        for name, weights in policies.items()
    }
    per_fold = per_fold_effective_weight_balance(
        depth=depth,
        labels=labels,
        policies=policies,
        config=config,
    )
    sample_counts = {
        "total_samples": int(depth.size),
        "known_label_samples": int(np.count_nonzero(np.isin(labels, [0, 1]))),
        "azimuthal_valid_samples": int(np.count_nonzero(base_mask)),
        "candidate_samples": int(np.count_nonzero(base_mask & (labels == 1))),
        "non_candidate_samples": int(np.count_nonzero(base_mask & (labels == 0))),
        "low_confidence_candidate_zero_weight_samples": int(
            np.count_nonzero((labels == 1) & (confidence < config.min_label_confidence))
        ),
        "zero_confidence_non_candidate_zero_weight_samples": int(
            np.count_nonzero((labels == 0) & (confidence <= 0.0))
        ),
        "large_depth_error_zero_weight_samples": int(
            np.count_nonzero(
                np.isin(labels, [0, 1])
                & (
                    large_depth_error
                    | (np.abs(depth_match_error) > config.max_depth_match_error_ft)
                )
            )
        ),
        "disagreement_samples": int(np.count_nonzero(base_mask & disagreement)),
    }
    report = SampleWeightPolicyReport(
        report_version=SAMPLE_WEIGHT_POLICY_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "input_npz": str(input_npz) if input_npz else "",
            "config_path": str(config_path) if config_path else "",
        },
        output_npz=str(output_npz) if output_npz else "",
        config=asdict(config),
        selected_default_policy=config.default_policy,
        sample_counts=sample_counts,
        policy_summary=policy_summary,
        per_fold_effective_weight_balance=per_fold,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
            "production model training",
        ],
    )
    return updated, report


def build_azimuthal_valid_mask(
    *,
    labels: np.ndarray,
    confidence: np.ndarray,
    valid_for_azimuthal: np.ndarray,
    depth_match_error: np.ndarray,
    large_depth_error: np.ndarray,
    config: SampleWeightPolicyConfig,
) -> np.ndarray:
    known = np.isin(labels, [0, 1])
    finite = np.isfinite(confidence) & np.isfinite(depth_match_error)
    depth_ok = np.abs(depth_match_error) <= config.max_depth_match_error_ft
    if config.exclude_large_depth_match_error:
        depth_ok &= ~large_depth_error
    return (
        known
        & finite
        & valid_for_azimuthal.astype(bool)
        & (
            ((labels == 1) & (confidence >= config.min_label_confidence))
            | ((labels == 0) & (confidence > 0.0))
        )
        & depth_ok
    )


def build_reliability(
    *,
    confidence: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    disagreement: np.ndarray,
    config: SampleWeightPolicyConfig,
) -> np.ndarray:
    reliability = np.clip(confidence.astype(np.float32), 0.0, 1.0)
    reliability[~valid_mask] = 0.0
    reliability[~np.isin(labels, [0, 1])] = 0.0
    if config.disagreement_policy == "exclude":
        reliability[disagreement] = 0.0
    elif config.disagreement_policy == "downweight":
        reliability[disagreement] *= float(config.disagreement_weight_multiplier)
    return reliability.astype(np.float32)


def build_policy_weights(
    *,
    labels: np.ndarray,
    reliability: np.ndarray,
    disagreement: np.ndarray,
    config: SampleWeightPolicyConfig,
) -> dict[str, np.ndarray]:
    confidence_only = reliability.astype(np.float32)
    class_balanced = _class_balance(
        labels=labels,
        base_weights=reliability,
        target_candidate_fraction=config.target_candidate_weight_fraction,
    )
    capped = _cap_by_quantile(class_balanced, config.capped_weight_quantile)
    capped = _cap_candidate_fraction(
        labels=labels,
        weights=capped,
        max_candidate_fraction=config.max_candidate_weight_fraction,
    )
    unweighted = np.where(reliability > 0.0, 1.0, 0.0).astype(np.float32)
    if config.disagreement_policy == "downweight":
        unweighted[disagreement & (unweighted > 0.0)] *= float(
            config.disagreement_weight_multiplier
        )
    return {
        "confidence_only": _normalize_to_nonzero_count(confidence_only),
        "class_balanced_confidence": _normalize_to_nonzero_count(class_balanced),
        "capped_class_balanced_confidence": _normalize_to_nonzero_count(capped),
        "unweighted": _normalize_to_nonzero_count(unweighted),
    }


def summarize_policy_weights(
    labels: np.ndarray,
    disagreement: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int | None]:
    labels = labels.reshape(-1)
    weights = weights.reshape(-1).astype(np.float64)
    nonzero = weights > 0.0
    candidate = (labels == 1) & nonzero
    non_candidate = (labels == 0) & nonzero
    total_weight = float(np.sum(weights[nonzero]))
    candidate_weight = float(np.sum(weights[candidate]))
    non_candidate_weight = float(np.sum(weights[non_candidate]))
    count_total = int(np.count_nonzero(valid_mask))
    candidate_count = int(np.count_nonzero(valid_mask & (labels == 1)))
    disagreement_weight = float(np.sum(weights[nonzero & disagreement]))
    return {
        "valid_count": count_total,
        "candidate_count": candidate_count,
        "non_candidate_count": int(np.count_nonzero(valid_mask & (labels == 0))),
        "candidate_count_fraction": _safe_fraction(candidate_count, count_total),
        "nonzero_weight_count": int(np.count_nonzero(nonzero)),
        "total_effective_weight": total_weight,
        "candidate_effective_weight": candidate_weight,
        "non_candidate_effective_weight": non_candidate_weight,
        "candidate_effective_weight_fraction": _safe_fraction(
            candidate_weight,
            total_weight,
        ),
        "non_candidate_effective_weight_fraction": _safe_fraction(
            non_candidate_weight,
            total_weight,
        ),
        "weight_min": _masked_stat(weights, nonzero, np.min),
        "weight_median": _masked_stat(weights, nonzero, np.median),
        "weight_max": _masked_stat(weights, nonzero, np.max),
        "disagreement_effective_weight": disagreement_weight,
        "disagreement_effective_weight_fraction": _safe_fraction(
            disagreement_weight,
            total_weight,
        ),
    }


def per_fold_effective_weight_balance(
    *,
    depth: np.ndarray,
    labels: np.ndarray,
    policies: dict[str, np.ndarray],
    config: SampleWeightPolicyConfig,
) -> list[dict[str, Any]]:
    active = policies[config.default_policy] > 0.0
    plan = make_depth_block_splits(
        depth=depth[active],
        labels=labels[active],
        n_splits=config.n_splits,
        min_gap_ft=config.min_gap_ft,
        block_size_ft=config.depth_block_size_ft,
        min_samples_per_class=config.min_samples_per_class_per_fold,
    )
    rows: list[dict[str, Any]] = []
    for policy_name, full_weights in policies.items():
        weights = full_weights[active]
        active_labels = labels[active]
        for fold in plan.folds:
            for split_name, mask in (
                ("train", fold.train_mask),
                ("validation", fold.validation_mask),
            ):
                row = _weight_balance_row(
                    policy_name=policy_name,
                    fold_index=fold.fold_index,
                    split_name=split_name,
                    labels=active_labels,
                    weights=weights,
                    mask=mask,
                )
                rows.append(row)
    return rows


def write_reweighted_sample_table(
    arrays: dict[str, np.ndarray],
    output_npz: Path,
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_npz, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def write_sample_weight_policy_report(
    report: SampleWeightPolicyReport,
    *,
    output_md: Path,
    output_json: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_sample_weight_policy_markdown(report), encoding="utf-8")


def format_sample_weight_policy_markdown(report: SampleWeightPolicyReport) -> str:
    lines = [
        "# MVP-4B-R Sample Weight Policy Report",
        "",
        "This report rebuilds weak-label sanity sample weights. CAST weak-label "
        "candidates are not final labels or ground truth.",
        "",
        "## Scope",
        "",
        f"- report_version: `{report.report_version}`",
        f"- default_policy: `{report.selected_default_policy}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        f"- no_stc: `{report.no_stc}`",
        f"- no_apes: `{report.no_apes}`",
        f"- no_deep_learning: `{report.no_deep_learning}`",
        f"- no_mvp4c: `{report.no_mvp4c}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in report.sample_counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Policy Summary", ""])
    for name, summary in report.policy_summary.items():
        lines.append(f"### {name}")
        lines.append("")
        for key, value in summary.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## Per-Fold Effective Weight Balance", ""])
    for row in report.per_fold_effective_weight_balance:
        lines.append(
            "- "
            f"{row['policy_name']} fold={row['fold_index']} split={row['split']}: "
            f"candidate_weight_fraction={row['candidate_effective_weight_fraction']}, "
            f"candidate_count={row['candidate_count']}, "
            f"non_candidate_count={row['non_candidate_count']}"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.warnings)
    if not report.warnings:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.errors)
    if not report.errors:
        lines.append("- none")
    lines.extend(["", "## Not Performed", ""])
    lines.extend(f"- {item}" for item in report.not_performed)
    lines.append("")
    return "\n".join(lines)


def _class_balance(
    *,
    labels: np.ndarray,
    base_weights: np.ndarray,
    target_candidate_fraction: float,
) -> np.ndarray:
    weights = base_weights.astype(np.float64).copy()
    candidate = (labels == 1) & (weights > 0.0)
    non_candidate = (labels == 0) & (weights > 0.0)
    candidate_sum = float(np.sum(weights[candidate]))
    non_candidate_sum = float(np.sum(weights[non_candidate]))
    if candidate_sum <= 0.0 or non_candidate_sum <= 0.0:
        return np.zeros_like(weights, dtype=np.float32)
    weights[candidate] *= target_candidate_fraction / candidate_sum
    weights[non_candidate] *= (1.0 - target_candidate_fraction) / non_candidate_sum
    return weights.astype(np.float32)


def _cap_candidate_fraction(
    *,
    labels: np.ndarray,
    weights: np.ndarray,
    max_candidate_fraction: float,
) -> np.ndarray:
    adjusted = weights.astype(np.float64).copy()
    candidate = labels == 1
    candidate_sum = float(np.sum(adjusted[candidate]))
    non_candidate_sum = float(np.sum(adjusted[~candidate]))
    total = candidate_sum + non_candidate_sum
    if total <= 0.0 or non_candidate_sum <= 0.0:
        return adjusted.astype(np.float32)
    current_fraction = candidate_sum / total
    if current_fraction <= max_candidate_fraction:
        return adjusted.astype(np.float32)
    scale = max_candidate_fraction * non_candidate_sum / (
        (1.0 - max_candidate_fraction) * candidate_sum
    )
    adjusted[candidate] *= scale
    return adjusted.astype(np.float32)


def _cap_by_quantile(weights: np.ndarray, quantile: float) -> np.ndarray:
    adjusted = weights.astype(np.float64).copy()
    nonzero = adjusted > 0.0
    if not np.any(nonzero):
        return adjusted.astype(np.float32)
    cap = float(np.quantile(adjusted[nonzero], quantile))
    adjusted[nonzero] = np.minimum(adjusted[nonzero], cap)
    return adjusted.astype(np.float32)


def _normalize_to_nonzero_count(weights: np.ndarray) -> np.ndarray:
    adjusted = weights.astype(np.float64).copy()
    nonzero_count = int(np.count_nonzero(adjusted > 0.0))
    total = float(np.sum(adjusted))
    if nonzero_count == 0 or total <= 0.0:
        return np.zeros_like(weights, dtype=np.float32)
    adjusted *= nonzero_count / total
    return adjusted.astype(np.float32)


def _weight_balance_row(
    *,
    policy_name: str,
    fold_index: int,
    split_name: str,
    labels: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = mask & (weights > 0.0)
    candidate = selected & (labels == 1)
    non_candidate = selected & (labels == 0)
    candidate_weight = float(np.sum(weights[candidate]))
    non_candidate_weight = float(np.sum(weights[non_candidate]))
    total_weight = candidate_weight + non_candidate_weight
    return {
        "policy_name": policy_name,
        "fold_index": int(fold_index),
        "split": split_name,
        "sample_count": int(np.count_nonzero(selected)),
        "candidate_count": int(np.count_nonzero(candidate)),
        "non_candidate_count": int(np.count_nonzero(non_candidate)),
        "candidate_effective_weight": candidate_weight,
        "non_candidate_effective_weight": non_candidate_weight,
        "candidate_effective_weight_fraction": _safe_fraction(
            candidate_weight,
            total_weight,
        ),
    }


def _policy_warnings_and_errors(
    *,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    policies: dict[str, np.ndarray],
    config: SampleWeightPolicyConfig,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    if np.count_nonzero(valid_mask & (labels == 1)) == 0:
        errors.append("No high-confidence candidate samples remain after weighting filters.")
    if np.count_nonzero(valid_mask & (labels == 0)) == 0:
        errors.append("No high-confidence non-candidate samples remain after weighting filters.")
    for name, weights in policies.items():
        summary = summarize_policy_weights(
            labels,
            np.zeros(labels.shape, dtype=bool),
            valid_mask,
            weights,
        )
        fraction = summary["candidate_effective_weight_fraction"]
        if (
            name != "confidence_only"
            and isinstance(fraction, float)
            and fraction > config.max_candidate_weight_fraction + 1.0e-6
        ):
            errors.append(
                f"{name} candidate effective weight fraction exceeds configured cap: "
                f"{fraction:.4f} > {config.max_candidate_weight_fraction:.4f}."
            )
        if isinstance(fraction, float) and fraction > 0.65:
            warnings.append(
                f"{name} candidate effective weight fraction remains high: {fraction:.4f}."
            )
    if not np.any(policies[config.default_policy] > 0.0):
        errors.append(f"Default policy {config.default_policy} produced all-zero weights.")
    return warnings, errors


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _required_array(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise KeyError(f"Sample table is missing required field: {name}")
    return np.asarray(arrays[name])


def _optional_bool_array(
    arrays: dict[str, np.ndarray],
    name: str,
    size: int,
    *,
    default: bool = False,
) -> np.ndarray:
    if name not in arrays:
        return np.full(size, default, dtype=bool)
    return np.asarray(arrays[name], dtype=bool).reshape(-1)


def _optional_float_array(arrays: dict[str, np.ndarray], name: str, size: int) -> np.ndarray:
    if name not in arrays:
        return np.zeros(size, dtype=np.float32)
    return np.asarray(arrays[name], dtype=np.float32).reshape(-1)


def _shape_errors(**arrays: np.ndarray) -> list[str]:
    sizes = {name: value.shape[0] for name, value in arrays.items()}
    expected = next(iter(sizes.values()))
    return [
        f"{name} length {size} does not match expected {expected}"
        for name, size in sizes.items()
        if size != expected
    ]


def _masked_stat(
    values: np.ndarray,
    mask: np.ndarray,
    reducer: Any,
) -> float | None:
    selected = values[mask]
    if selected.size == 0:
        return None
    return float(reducer(selected))


def _safe_fraction(numerator: float | int, denominator: float | int) -> float | None:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return None
    return float(numerator) / denominator_float


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
