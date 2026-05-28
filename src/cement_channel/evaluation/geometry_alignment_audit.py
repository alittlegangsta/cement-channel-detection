from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

GEOMETRY_ALIGNMENT_AUDIT_VERSION = "geometry_alignment_audit_v001"


@dataclass(frozen=True)
class GeometryAlignmentAuditConfig:
    min_positive_count: int = 5
    min_negative_count: int = 5
    permutation_repeats: int = 11
    fold_count: int = 3
    min_margin_over_permutation: float = 0.03
    match_margin_tolerance: float = 0.01
    suspicious_high_balanced_accuracy: float = 0.9
    review_band_min_ft: float = 5680.0
    review_band_max_ft: float = 5720.0


@dataclass(frozen=True)
class GeometryAlignmentAuditReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    output_csv: str
    geometry_combo_count: int
    feature_count: int
    combo_summaries: list[dict[str, Any]]
    sign_sensitivity: list[dict[str, Any]]
    best_combo: dict[str, Any] | None
    r7_reference_summary: dict[str, Any] | None
    source_receiver_interval_summary: dict[str, Any] | None
    recommendation: str
    recommendation_reason: str
    manual_confirmation_required: bool
    manual_confirmation_items: list[str]
    no_model_training: bool
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


def audit_geometry_alignment_from_paths(
    *,
    geometry_depth_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    refinement_report_json: Path | str,
    geometry_config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
    config: GeometryAlignmentAuditConfig | None = None,
) -> GeometryAlignmentAuditReport:
    geometry_arrays = _load_npz(geometry_depth_labels_npz)
    feature_arrays = _load_npz(depth_level_features_npz)
    refinement_report = _read_json(Path(refinement_report_json))
    report, rows = audit_geometry_alignment(
        geometry_arrays=geometry_arrays,
        feature_arrays=feature_arrays,
        refinement_report=refinement_report,
        config=config or GeometryAlignmentAuditConfig(),
        inputs={
            "geometry_depth_labels_npz": str(geometry_depth_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "refinement_report_json": str(refinement_report_json),
            "geometry_config_path": str(geometry_config_path),
        },
        output_csv=Path(output_csv),
    )
    write_geometry_alignment_audit_outputs(
        report,
        rows,
        output_md=Path(output_report_md),
        output_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def audit_geometry_alignment(
    *,
    geometry_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    refinement_report: dict[str, Any],
    config: GeometryAlignmentAuditConfig | None = None,
    inputs: dict[str, str] | None = None,
    output_csv: Path | None = None,
) -> tuple[GeometryAlignmentAuditReport, list[dict[str, Any]]]:
    cfg = config or GeometryAlignmentAuditConfig()
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(geometry_arrays, feature_arrays, refinement_report, errors)
    depth = np.asarray(geometry_arrays["depth"], dtype=np.float32).reshape(-1)
    feature_depth = np.asarray(feature_arrays["depth"], dtype=np.float32).reshape(-1)
    if depth.size != feature_depth.size:
        raise ValueError("Geometry label depth count must match feature depth count.")
    if not np.allclose(depth, feature_depth, atol=1e-3):
        warnings.append("Geometry label and feature depth arrays differ; using label order.")
    features = np.asarray(feature_arrays["depth_level_xsi_features"], dtype=np.float32)
    feature_names = np.asarray(feature_arrays["depth_level_xsi_feature_names"]).astype(str)
    finite_rows = np.all(np.isfinite(features), axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    leakage_warnings = _feature_leakage_warnings(feature_names)
    warnings.extend(leakage_warnings)

    rows: list[dict[str, Any]] = []
    combo_summaries: list[dict[str, Any]] = []
    modes = geometry_arrays["mode"].astype(str)
    signs = geometry_arrays["sign"].astype(int)
    for combo_index, mode in enumerate(modes):
        positive = np.asarray(
            geometry_arrays["high_confidence_positive_mask"][combo_index],
            dtype=bool,
        )
        negative = np.asarray(geometry_arrays["clear_negative_mask"][combo_index], dtype=bool)
        selected = (positive | negative) & finite_rows
        summary, feature_rows = _audit_combo(
            combo_index=combo_index,
            mode=mode,
            sign=int(signs[combo_index]),
            depth=depth,
            features=features,
            feature_names=feature_names,
            positive=positive & finite_rows,
            negative=negative & finite_rows,
            selected=selected,
            config=cfg,
            leakage_warnings=leakage_warnings,
        )
        combo_summaries.append(summary)
        rows.extend(feature_rows)
    sign_sensitivity = _sign_sensitivity(combo_summaries)
    best_combo = _best_runnable_combo(combo_summaries)
    r7_summary = _best_for_mode(combo_summaries, "r7_reference_depth")
    interval_summary = _best_for_mode(combo_summaries, "source_receiver_interval")
    recommendation, reason, manual_items = _recommendation(
        combo_summaries,
        sign_sensitivity,
        best_combo,
        r7_summary,
        interval_summary,
        config=cfg,
    )
    report = GeometryAlignmentAuditReport(
        report_version=GEOMETRY_ALIGNMENT_AUDIT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs or {},
        output_csv=str(output_csv) if output_csv else "",
        geometry_combo_count=len(combo_summaries),
        feature_count=int(features.shape[1]),
        combo_summaries=combo_summaries,
        sign_sensitivity=sign_sensitivity,
        best_combo=best_combo,
        r7_reference_summary=r7_summary,
        source_receiver_interval_summary=interval_summary,
        recommendation=recommendation,
        recommendation_reason=reason,
        manual_confirmation_required=bool(manual_items),
        manual_confirmation_items=manual_items,
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "new model training",
            "model weight export",
            "production inference",
            "final label generation",
            "ground truth claim",
            "STC",
            "APES",
            "deep learning",
            "MVP-4C",
        ],
    )
    return report, rows


def write_geometry_alignment_audit_outputs(
    report: GeometryAlignmentAuditReport,
    rows: list[dict[str, Any]],
    *,
    output_md: Path,
    output_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    _ensure_can_write(output_md, overwrite=overwrite)
    _ensure_can_write(output_json, overwrite=overwrite)
    _ensure_can_write(output_csv, overwrite=overwrite)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_geometry_alignment_audit_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_geometry_alignment_audit_markdown(report: GeometryAlignmentAuditReport) -> str:
    lines = [
        "# Geometry-Aware XSI-CAST Alignment Audit",
        "",
        "This is an MVP-4B-G review-only audit against CAST weak-label candidates. "
        "It does not train a new model, generate final labels, run STC/APES, use "
        "deep learning, or authorize MVP-4C.",
        "",
        f"- recommendation: `{report.recommendation}`",
        f"- recommendation_reason: {report.recommendation_reason}",
        f"- manual_confirmation_required: `{report.manual_confirmation_required}`",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Mode/Sign Summaries",
        "",
    ]
    for row in report.combo_summaries:
        lines.append(
            "- "
            f"{row['mode']} sign={row['sign']}: status={row['status']}, "
            f"positive={row['positive_count']}, negative={row['negative_count']}, "
            f"best_abs_effect={row['best_abs_effect_size']}, "
            f"balanced_accuracy={row['balanced_accuracy']}, "
            f"permutation_balanced_accuracy={row['permutation_balanced_accuracy']}, "
            f"margin={row['real_minus_permutation_margin']}, "
            f"predicted_positive_rate={row['predicted_positive_rate']}"
        )
    lines.extend(["", "## Sign Sensitivity", ""])
    for row in report.sign_sensitivity:
        lines.append(f"- {row}")
    lines.extend(["", "## Manual Confirmation Items", ""])
    lines.extend(_message_lines(report.manual_confirmation_items))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _audit_combo(
    *,
    combo_index: int,
    mode: str,
    sign: int,
    depth: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    selected: np.ndarray,
    config: GeometryAlignmentAuditConfig,
    leakage_warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))
    base = {
        "combo_index": combo_index,
        "mode": mode,
        "sign": sign,
        "sample_count": int(depth.size),
        "selected_count": int(np.count_nonzero(selected)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction": _fraction(positive),
        "high_confidence_subset_count": positive_count,
        "leakage_warnings": leakage_warnings,
        "depends_on_5700_band": _depends_on_review_band(depth, positive, config),
    }
    if positive_count < config.min_positive_count or negative_count < config.min_negative_count:
        return (
            {
                **base,
                "status": "skipped_sample_count",
                "best_feature_name": None,
                "best_abs_effect_size": None,
                "feature_separation_summary": {},
                "balanced_accuracy": None,
                "permutation_balanced_accuracy": None,
                "real_minus_permutation_margin": None,
                "predicted_positive_rate": None,
                "folds_above_permutation": 0,
                "folds_above_permutation_fraction": 0.0,
                "sample_count_collapse": True,
                "suspicious_leakage": False,
            },
            [],
        )
    feature_rows = _feature_rows(
        mode=mode,
        sign=sign,
        features=features,
        feature_names=feature_names,
        positive=positive,
        negative=negative,
    )
    best = max(
        feature_rows,
        key=lambda row: abs(float(row["standardized_difference"] or 0.0)),
    )
    sanity = _threshold_sanity_with_permutation(
        values=features[:, int(best["feature_index"])],
        positive=positive,
        negative=negative,
        depth=depth,
        config=config,
        seed=_stable_seed(mode, sign),
    )
    return (
        {
            **base,
            "status": "runnable",
            "best_feature_name": best["feature_name"],
            "best_feature_index": int(best["feature_index"]),
            "best_abs_effect_size": (
                None
                if best["standardized_difference"] is None
                else abs(float(best["standardized_difference"]))
            ),
            "feature_separation_summary": {
                "candidate_mean": best["candidate_mean"],
                "negative_mean": best["negative_mean"],
                "standardized_difference": best["standardized_difference"],
            },
            "balanced_accuracy": sanity["balanced_accuracy"],
            "permutation_balanced_accuracy": sanity["permutation_balanced_accuracy"],
            "real_minus_permutation_margin": sanity["real_minus_permutation_margin"],
            "predicted_positive_rate": sanity["predicted_positive_rate"],
            "folds_above_permutation": sanity["folds_above_permutation"],
            "folds_above_permutation_fraction": sanity["folds_above_permutation_fraction"],
            "sample_count_collapse": False,
            "suspicious_leakage": bool(
                sanity["balanced_accuracy"] is not None
                and sanity["balanced_accuracy"] >= config.suspicious_high_balanced_accuracy
            ),
        },
        feature_rows,
    )


def _feature_rows(
    *,
    mode: str,
    sign: int,
    features: np.ndarray,
    feature_names: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_names.astype(str)):
        values = features[:, index]
        pos_values = values[positive & np.isfinite(values)]
        neg_values = values[negative & np.isfinite(values)]
        if pos_values.size == 0 or neg_values.size == 0:
            rows.append(
                {
                    "mode": mode,
                    "sign": sign,
                    "feature_index": index,
                    "feature_name": feature_name,
                    "candidate_count": int(pos_values.size),
                    "negative_count": int(neg_values.size),
                    "candidate_mean": None,
                    "negative_mean": None,
                    "standardized_difference": None,
                }
            )
            continue
        pos_mean = float(np.mean(pos_values))
        neg_mean = float(np.mean(neg_values))
        pooled = float(np.sqrt(0.5 * (np.var(pos_values) + np.var(neg_values))))
        effect = None if pooled <= 0.0 else (pos_mean - neg_mean) / pooled
        rows.append(
            {
                "mode": mode,
                "sign": sign,
                "feature_index": index,
                "feature_name": feature_name,
                "candidate_count": int(pos_values.size),
                "negative_count": int(neg_values.size),
                "candidate_mean": pos_mean,
                "negative_mean": neg_mean,
                "standardized_difference": None if effect is None else float(effect),
            }
        )
    return rows


def _threshold_sanity_with_permutation(
    *,
    values: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    depth: np.ndarray,
    config: GeometryAlignmentAuditConfig,
    seed: int,
) -> dict[str, Any]:
    selected = positive | negative
    y_true = positive[selected]
    scores = values[selected].astype(np.float64)
    threshold, positive_high = _median_threshold(scores, y_true)
    pred = scores >= threshold if positive_high else scores <= threshold
    real_ba = _balanced_accuracy(y_true, pred)
    predicted_positive_rate = float(np.mean(pred)) if pred.size else None
    rng = np.random.default_rng(seed)
    permutation_scores: list[float] = []
    for _ in range(config.permutation_repeats):
        permuted = y_true.copy()
        rng.shuffle(permuted)
        perm_threshold, perm_positive_high = _median_threshold(scores, permuted)
        perm_pred = scores >= perm_threshold if perm_positive_high else scores <= perm_threshold
        perm_ba = _balanced_accuracy(permuted, perm_pred)
        if perm_ba is not None:
            permutation_scores.append(perm_ba)
    permutation_ba = None if not permutation_scores else float(np.mean(permutation_scores))
    margin = None if real_ba is None or permutation_ba is None else real_ba - permutation_ba
    folds_above = _folds_above_permutation(
        depth=depth[selected],
        scores=scores,
        y_true=y_true,
        threshold=threshold,
        positive_high=positive_high,
        config=config,
        rng=rng,
    )
    return {
        "balanced_accuracy": real_ba,
        "permutation_balanced_accuracy": permutation_ba,
        "real_minus_permutation_margin": margin,
        "predicted_positive_rate": predicted_positive_rate,
        **folds_above,
    }


def _folds_above_permutation(
    *,
    depth: np.ndarray,
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    positive_high: bool,
    config: GeometryAlignmentAuditConfig,
    rng: np.random.Generator,
) -> dict[str, int | float]:
    order = np.argsort(depth)
    splits = np.array_split(order, config.fold_count)
    folds_above = 0
    valid_folds = 0
    for indices in splits:
        if indices.size == 0:
            continue
        fold_y = y_true[indices]
        if not np.any(fold_y) or not np.any(~fold_y):
            continue
        fold_scores = scores[indices]
        fold_pred = fold_scores >= threshold if positive_high else fold_scores <= threshold
        real_ba = _balanced_accuracy(fold_y, fold_pred)
        permuted = fold_y.copy()
        rng.shuffle(permuted)
        perm_ba = _balanced_accuracy(permuted, fold_pred)
        if real_ba is None or perm_ba is None:
            continue
        valid_folds += 1
        if real_ba > perm_ba:
            folds_above += 1
    return {
        "folds_above_permutation": folds_above,
        "valid_fold_count": valid_folds,
        "folds_above_permutation_fraction": (
            0.0 if valid_folds == 0 else folds_above / valid_folds
        ),
    }


def _median_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, bool]:
    pos = scores[y_true]
    neg = scores[~y_true]
    if pos.size == 0 or neg.size == 0:
        return float(np.nanmedian(scores)), True
    pos_median = float(np.median(pos))
    neg_median = float(np.median(neg))
    return 0.5 * (pos_median + neg_median), pos_median >= neg_median


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    pos = y_true
    neg = ~y_true
    if not np.any(pos) or not np.any(neg):
        return None
    tpr = np.count_nonzero(y_pred & pos) / np.count_nonzero(pos)
    tnr = np.count_nonzero((~y_pred) & neg) / np.count_nonzero(neg)
    return float(0.5 * (tpr + tnr))


def _sign_sensitivity(combo_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in sorted({str(row["mode"]) for row in combo_summaries}):
        plus = _summary_for(combo_summaries, mode, 1)
        minus = _summary_for(combo_summaries, mode, -1)
        if not plus or not minus:
            continue
        plus_margin = _as_float(plus.get("real_minus_permutation_margin"))
        minus_margin = _as_float(minus.get("real_minus_permutation_margin"))
        margin_delta = (
            None if plus_margin is None or minus_margin is None else plus_margin - minus_margin
        )
        rows.append(
            {
                "mode": mode,
                "positive_fraction_delta_plus_minus": _nullable_delta(
                    plus.get("positive_fraction"),
                    minus.get("positive_fraction"),
                ),
                "best_abs_effect_delta_plus_minus": _nullable_delta(
                    plus.get("best_abs_effect_size"),
                    minus.get("best_abs_effect_size"),
                ),
                "margin_delta_plus_minus": margin_delta,
                "sign_sensitive": bool(
                    margin_delta is not None and abs(margin_delta) > 0.03
                ),
            }
        )
    return rows


def _recommendation(
    combo_summaries: list[dict[str, Any]],
    sign_sensitivity: list[dict[str, Any]],
    best_combo: dict[str, Any] | None,
    r7_summary: dict[str, Any] | None,
    interval_summary: dict[str, Any] | None,
    *,
    config: GeometryAlignmentAuditConfig,
) -> tuple[str, str, list[str]]:
    manual_items: list[str] = []
    runnable = [row for row in combo_summaries if row["status"] == "runnable"]
    if not runnable:
        return "no_go", "No geometry-aware mode had enough positive/negative support.", manual_items
    if any(bool(row.get("suspicious_leakage")) for row in runnable):
        return (
            "no_go",
            "Suspiciously high threshold sanity metric; leakage review required.",
            manual_items,
        )
    if any(bool(row.get("sign_sensitive")) for row in sign_sensitivity):
        manual_items.append("Confirm depth-axis sign before choosing a geometry-aware target.")
    if any(bool(row.get("sample_count_collapse")) for row in combo_summaries):
        manual_items.append("Review mode/sign targets with collapsed positive or negative support.")
    if best_combo is None:
        return "no_go", "No runnable geometry-aware result was available.", manual_items
    margin = _as_float(best_combo.get("real_minus_permutation_margin"))
    if margin is None or margin < config.min_margin_over_permutation:
        return (
            "no_go",
            "Best geometry-aware sanity margin does not exceed permutation.",
            manual_items,
        )
    interval_margin = None if interval_summary is None else _as_float(
        interval_summary.get("real_minus_permutation_margin")
    )
    r7_margin = None if r7_summary is None else _as_float(
        r7_summary.get("real_minus_permutation_margin")
    )
    if (
        interval_summary is not None
        and interval_margin is not None
        and r7_margin is not None
        and interval_margin > r7_margin + config.match_margin_tolerance
    ):
        manual_items.append("Source-receiver interval outperforms R7; consider interval target.")
        return (
            "conditional_go",
            "Source-receiver interval improved over R7 but requires human review.",
            manual_items,
        )
    if r7_summary is not None and best_combo.get("mode") == "r7_reference_depth":
        return (
            "go",
            "R7 reference-depth remains the strongest or matched review target.",
            manual_items,
        )
    manual_items.append("Non-R7 geometry mode is competitive; human target choice required.")
    return "conditional_go", "A non-R7 geometry mode matched or improved R7.", manual_items


def _best_runnable_combo(combo_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    runnable = [
        row
        for row in combo_summaries
        if row["status"] == "runnable"
        and _as_float(row.get("real_minus_permutation_margin")) is not None
    ]
    if not runnable:
        return None
    return max(runnable, key=lambda row: float(row["real_minus_permutation_margin"]))


def _best_for_mode(combo_summaries: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in combo_summaries
        if row["mode"] == mode and _as_float(row.get("real_minus_permutation_margin")) is not None
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["real_minus_permutation_margin"]))


def _summary_for(
    combo_summaries: list[dict[str, Any]],
    mode: str,
    sign: int,
) -> dict[str, Any] | None:
    for row in combo_summaries:
        if row["mode"] == mode and int(row["sign"]) == sign:
            return row
    return None


def _feature_leakage_warnings(feature_names: np.ndarray) -> list[str]:
    suspicious = [
        name
        for name in feature_names.astype(str)
        if any(token in name.lower() for token in ("cast", "label", "zc"))
    ]
    if not suspicious:
        return []
    return ["Potential leakage feature names found: " + ", ".join(suspicious[:10])]


def _depends_on_review_band(
    depth: np.ndarray,
    positive: np.ndarray,
    config: GeometryAlignmentAuditConfig,
) -> bool:
    positive_count = np.count_nonzero(positive)
    if positive_count == 0:
        return False
    review = (depth >= config.review_band_min_ft) & (depth <= config.review_band_max_ft)
    return bool(np.count_nonzero(positive & review) / positive_count > 0.5)


def _validate_guardrails(
    geometry_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    refinement_report: dict[str, Any],
    errors: list[str],
) -> None:
    required_geometry = (
        "depth",
        "mode",
        "sign",
        "high_confidence_positive_mask",
        "clear_negative_mask",
        "no_final_labels",
    )
    required_features = ("depth", "depth_level_xsi_features", "depth_level_xsi_feature_names")
    missing_geometry = [key for key in required_geometry if key not in geometry_arrays]
    missing_features = [key for key in required_features if key not in feature_arrays]
    if missing_geometry:
        raise KeyError("Geometry-aware label NPZ missing field(s): " + ", ".join(missing_geometry))
    if missing_features:
        raise KeyError("Depth-level feature NPZ missing field(s): " + ", ".join(missing_features))
    if not bool(np.asarray(geometry_arrays.get("no_final_labels", False))):
        errors.append("Geometry-aware labels must preserve no_final_labels=true.")
    for forbidden in ("no_stc", "no_apes", "no_deep_learning", "no_mvp4c"):
        if forbidden in geometry_arrays and not bool(np.asarray(geometry_arrays[forbidden])):
            errors.append(f"Geometry-aware labels must preserve {forbidden}=true.")
        if forbidden in feature_arrays and not bool(np.asarray(feature_arrays[forbidden])):
            errors.append(f"Depth-level features must preserve {forbidden}=true.")
    if refinement_report.get("no_final_labels") is not True:
        errors.append("Depth-level refinement report must preserve no_final_labels=true.")


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "mode",
        "sign",
        "feature_index",
        "feature_name",
        "candidate_count",
        "negative_count",
        "candidate_mean",
        "negative_mean",
        "standardized_difference",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _nullable_delta(value_a: Any, value_b: Any) -> float | None:
    a = _as_float(value_a)
    b = _as_float(value_b)
    return None if a is None or b is None else a - b


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))


def _stable_seed(mode: str, sign: int) -> int:
    total = 17 + sign * 101
    for char in mode:
        total = (total * 131 + ord(char)) % (2**32 - 1)
    return int(total)


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
