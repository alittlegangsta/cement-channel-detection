from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.evaluation.depth_regime_review_policy import (
    BoundaryReviewZone,
    DepthRegimeReviewPolicy,
    load_depth_regime_review_policy,
)
from cement_channel.evaluation.geometry_regression_depth_regime_review import (
    MORPHOLOGY_FIELDS,
    NEAR_FAR_FEATURES,
    TARGET_VIEWS,
    _build_context,
    _ensure_can_write,
    _finite_mean,
    _finite_percentile,
    _load_npz,
    _near_far_extreme_mask,
    _primary_label_confidence,
)
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

FORMAL_REVIEW_VERSION = "geometry_regression_formal_review_v001"
DECISION_PACK_VERSION = "geometry_regression_formal_review_decision_pack_v001"
REVIEW_BAND_MIN_FT = 5680.0
REVIEW_BAND_MAX_FT = 5720.0
PRIMARY_KERNEL = "triangular_midpoint_weighted"

FIGURE_NAMES = (
    "broad_regime_overview.png",
    "target_views_by_regime.png",
    "target_views_vs_depth.png",
    "receiver_mean_p90_max_comparison.png",
    "receiver_std_vs_depth.png",
    "morphology_by_regime.png",
    "morphology_vs_depth.png",
    "boundary_zone_overview.png",
    "orientation_inclination_vs_depth.png",
    "near_far_ratio_vs_depth.png",
    "near_far_outliers_vs_boundaries.png",
    "5700_band_review.png",
    "secondary_boundary_zones_summary.png",
)

REVIEWER_CHECKLIST_QUESTIONS = (
    "Do Regime A / B / C show different target prevalence in the figures?",
    "Is Regime A morphology more continuous and stronger?",
    "Is Regime B relatively weak?",
    "Does Regime C show a rebound?",
    "Is 2582.79 ft a credible physical-change boundary?",
    "Does 4219.52 ft align with morphology / orientation / inclination changes?",
    "Should 5680 ft receive special handling?",
    "Are 2826.85 ft and 3984.03 ft only weak change points?",
    "Is receiver_max often driven by a single receiver extreme?",
    "Is receiver_p90 more robust than receiver_max?",
    "Is receiver_mean suitable as a conservative reference?",
    "Are morphology arrays worth discussing in the next label-design round?",
    "Are near/far extremes mainly in Regime A?",
    "Approve next step: formal CV stratification study?",
    "Approve next step: target-view decision study?",
    "Approve next step: morphology-aware label redesign discussion?",
    "Approve next step: preprocessing review?",
    "Approve next step: controlled feature review?",
    "Decision: stop or continue manual review?",
)


@dataclass(frozen=True)
class GeometryRegressionFormalReviewPack:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    broad_regime_count: int
    boundary_zone_count: int
    selected_manual_review_interval_count: int
    figure_count: int
    review_policy: dict[str, Any]
    stage_10_stop_still_valid: bool
    stage_11_12_still_blocked: bool
    mvp4c_stc_apes_deep_learning_final_labels_still_forbidden: bool
    no_formal_cv_split_change: bool
    no_regime_specific_modeling: bool
    no_primary_target_change: bool
    no_new_label_target: bool
    no_new_feature: bool
    no_model_training: bool
    no_final_labels: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_geometry_regression_formal_review(
    *,
    labels_npz: Path | str,
    features_npz: Path | str,
    cast_npz: Path | str,
    policy_config: Path | str,
    output_dir: Path | str,
    decision_pack_md: Path | str,
    decision_pack_json: Path | str,
    overwrite: bool = False,
) -> GeometryRegressionFormalReviewPack:
    policy = load_depth_regime_review_policy(policy_config)
    labels = _load_npz(Path(labels_npz))
    features = _load_npz(Path(features_npz))
    cast = _load_npz(Path(cast_npz))
    context = _build_context(labels, features, cast)

    output_path = Path(output_dir)
    decision_md_path = Path(decision_pack_md)
    decision_json_path = Path(decision_pack_json)
    paths = _output_paths(output_path, policy)
    for path in [*paths.values(), decision_md_path, decision_json_path]:
        _ensure_can_write(path, overwrite=overwrite)
    output_path.mkdir(parents=True, exist_ok=True)
    decision_md_path.parent.mkdir(parents=True, exist_ok=True)
    decision_json_path.parent.mkdir(parents=True, exist_ok=True)

    broad_rows = _broad_regime_summary(context, policy)
    boundary_rows = _boundary_zone_summary(context, policy)
    intervals = _selected_manual_review_intervals(context, policy)
    figure_paths = {name: output_path / name for name in FIGURE_NAMES}
    for zone in policy.priority_boundary_review_zones:
        if zone.priority == "primary":
            figure_paths[f"{zone.zone_id}.png"] = output_path / f"{zone.zone_id}.png"

    _write_json(paths["broad_regime_summary_json"], broad_rows)
    _write_csv(broad_rows, paths["broad_regime_summary_csv"])
    _write_json(paths["boundary_review_zones_json"], boundary_rows)
    _write_csv(boundary_rows, paths["boundary_review_zones_csv"])
    _write_json(paths["selected_manual_review_intervals_json"], intervals)
    _write_csv(intervals, paths["selected_manual_review_intervals_csv"])
    paths["reviewer_checklist_md"].write_text(_format_reviewer_checklist(), encoding="utf-8")
    paths["reviewer_decision_template_md"].write_text(
        _format_reviewer_decision_template(policy, intervals),
        encoding="utf-8",
    )
    _write_figures(context, policy, broad_rows, boundary_rows, figure_paths, overwrite=overwrite)
    pack = GeometryRegressionFormalReviewPack(
        review_version=FORMAL_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "labels_npz": str(labels_npz),
            "features_npz": str(features_npz),
            "cast_npz": str(cast_npz),
            "policy_config": str(policy_config),
        },
        output_dir=str(output_path),
        broad_regime_count=len(policy.broad_review_regimes),
        boundary_zone_count=len(policy.priority_boundary_review_zones),
        selected_manual_review_interval_count=len(intervals),
        figure_count=len(figure_paths),
        review_policy=policy.to_dict(),
        stage_10_stop_still_valid=True,
        stage_11_12_still_blocked=True,
        mvp4c_stc_apes_deep_learning_final_labels_still_forbidden=True,
        no_formal_cv_split_change=True,
        no_regime_specific_modeling=True,
        no_primary_target_change=True,
        no_new_label_target=True,
        no_new_feature=True,
        no_model_training=True,
        no_final_labels=True,
        warnings=list(context["warnings"]),
    )
    paths["review_summary_md"].write_text(
        _format_review_summary(pack, broad_rows, boundary_rows, intervals, figure_paths),
        encoding="utf-8",
    )
    decision_pack = _decision_pack(pack, broad_rows, boundary_rows, intervals)
    decision_json_path.write_text(
        json.dumps(decision_pack, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    decision_md_path.write_text(_format_decision_pack_markdown(decision_pack), encoding="utf-8")
    return pack


def _output_paths(output_dir: Path, policy: DepthRegimeReviewPolicy) -> dict[str, Path]:
    paths = {
        "review_summary_md": output_dir / "review_summary.md",
        "reviewer_checklist_md": output_dir / "reviewer_checklist.md",
        "broad_regime_summary_csv": output_dir / "broad_regime_summary.csv",
        "broad_regime_summary_json": output_dir / "broad_regime_summary.json",
        "boundary_review_zones_csv": output_dir / "boundary_review_zones.csv",
        "boundary_review_zones_json": output_dir / "boundary_review_zones.json",
        "selected_manual_review_intervals_csv": (
            output_dir / "selected_manual_review_intervals.csv"
        ),
        "selected_manual_review_intervals_json": (
            output_dir / "selected_manual_review_intervals.json"
        ),
        "reviewer_decision_template_md": output_dir / "reviewer_decision_template.md",
    }
    for name in FIGURE_NAMES:
        paths[f"figure_{name}"] = output_dir / name
    for zone in policy.priority_boundary_review_zones:
        if zone.priority == "primary":
            paths[f"figure_{zone.zone_id}"] = output_dir / f"{zone.zone_id}.png"
    return paths


def _broad_regime_summary(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
) -> list[dict[str, Any]]:
    rows = []
    depth = context["depth"]
    for regime in policy.broad_review_regimes:
        mask = (depth >= regime.depth_min_ft) & (depth <= regime.depth_max_ft)
        row: dict[str, Any] = {
            "regime_id": regime.id,
            "depth_min_ft": regime.depth_min_ft,
            "depth_max_ft": regime.depth_max_ft,
            "purpose": regime.purpose,
            "sample_count": int(np.count_nonzero(mask)),
            "review_only": True,
            "no_formal_cv_split_change": True,
            "no_final_labels": True,
        }
        row.update(_target_view_summary(context, mask))
        row.update(_morphology_summary(context, mask))
        row.update(_physical_summary(context, mask))
        row.update(_near_far_summary(context, mask))
        rows.append(row)
    return rows


def _boundary_zone_summary(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
) -> list[dict[str, Any]]:
    depth = context["depth"]
    rows = []
    for zone in policy.priority_boundary_review_zones:
        mask = (depth >= zone.depth_min_ft) & (depth <= zone.depth_max_ft)
        regime = policy.regime_for_depth(zone.depth_ft)
        row: dict[str, Any] = {
            "zone_id": zone.zone_id,
            "depth_ft": zone.depth_ft,
            "depth_min_ft": zone.depth_min_ft,
            "depth_max_ft": zone.depth_max_ft,
            "priority": zone.priority,
            "support_count": zone.support_count,
            "regime_id": regime.id if regime else "",
            "sample_count": int(np.count_nonzero(mask)),
            "support_summary": _boundary_support_summary(zone),
            "review_only": True,
            "formal_boundary_adopted": False,
            "formal_cv_split_changed": False,
            "no_final_labels": True,
        }
        row.update(_target_view_summary(context, mask))
        row.update(_morphology_summary(context, mask))
        row.update(_physical_summary(context, mask))
        row.update(_near_far_summary(context, mask))
        rows.append(row)
    return rows


def _selected_manual_review_intervals(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    intervals.extend(_broad_regime_intervals(context, policy))
    intervals.extend(_primary_boundary_intervals(context, policy))
    intervals.extend(_special_intervals(context, policy))
    return intervals


def _broad_regime_intervals(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
) -> list[dict[str, Any]]:
    depth = context["depth"]
    target = _target_values(context, "receiver_max")
    receiver_mean = _target_values(context, "receiver_mean")
    receiver_p90 = _target_values(context, "receiver_p90")
    combined = _morphology_values(context, "combined_channel_fraction")
    rows = []
    for regime in policy.broad_review_regimes:
        mask = (depth >= regime.depth_min_ft) & (depth <= regime.depth_max_ft)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        selectors = {
            "representative_high_target": indices[int(np.nanargmax(target[indices]))],
            "representative_low_target": indices[int(np.nanargmin(target[indices]))],
            "morphology_sensitive": indices[int(np.nanargmax(combined[indices]))],
            "receiver_max_sensitive": indices[
                int(np.nanargmax((target - receiver_p90)[indices]))
            ],
            "receiver_mean_p90_disagreement": indices[
                int(np.nanargmax(np.abs(receiver_p90 - receiver_mean)[indices]))
            ],
        }
        for review_type, index in selectors.items():
            depth_min, depth_max = _window_around_depth(
                depth,
                float(depth[index]),
                max(5.0, policy.boundary_review_half_width_ft / 2.0),
            )
            rows.append(
                _interval_row(
                    context,
                    policy,
                    interval_id=f"{regime.id}_{review_type}",
                    review_type=review_type,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    recommended_human_question=_interval_question(review_type, regime.id),
                )
            )
    return rows


def _primary_boundary_intervals(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
) -> list[dict[str, Any]]:
    rows = []
    half = policy.boundary_review_half_width_ft
    for zone in policy.priority_boundary_review_zones:
        if zone.priority != "primary":
            continue
        segments = {
            "before_boundary": (zone.depth_ft - 2.0 * half, zone.depth_ft - half),
            "boundary_window": (zone.depth_ft - half, zone.depth_ft + half),
            "after_boundary": (zone.depth_ft + half, zone.depth_ft + 2.0 * half),
        }
        for review_type, (depth_min, depth_max) in segments.items():
            rows.append(
                _interval_row(
                    context,
                    policy,
                    interval_id=f"{zone.zone_id}_{review_type}",
                    review_type=review_type,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    boundary_zone=zone.zone_id,
                    recommended_human_question=_boundary_question(zone, review_type),
                )
            )
    return rows


def _special_intervals(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
) -> list[dict[str, Any]]:
    orient = context["covariates"]["orientation_confidence"]
    combined = _morphology_values(context, "combined_channel_fraction")
    near_far_mask = _near_far_extreme_mask(context)
    rows = [
        _interval_row(
            context,
            policy,
            interval_id="5700_band_review",
            review_type="5700_band_review",
            depth_min=REVIEW_BAND_MIN_FT,
            depth_max=REVIEW_BAND_MAX_FT,
            boundary_zone="boundary_5680p00_ft",
            recommended_human_question="Should the 5700 ft band receive separate handling?",
        )
    ]
    rows.append(
        _interval_from_index(
            context,
            policy,
            int(np.nanargmin(orient)),
            "low_orientation_confidence",
            "Does low orientation confidence explain target or morphology behavior here?",
        )
    )
    near_far_indices = np.flatnonzero(near_far_mask)
    if near_far_indices.size:
        index = int(near_far_indices[0])
    else:
        index = int(np.nanargmax(np.abs(context["feature_matrix"][:, 0])))
    rows.append(
        _interval_from_index(
            context,
            policy,
            index,
            "near_far_ratio_outlier",
            "Is this near/far extreme a physical signal or preprocessing artifact?",
        )
    )
    rows.append(
        _interval_from_index(
            context,
            policy,
            int(np.nanargmax(combined)),
            "morphology_sensitive_outlier",
            "Does this interval show continuous morphology or scattered low-Zc noise?",
        )
    )
    return rows


def _interval_from_index(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    index: int,
    review_type: str,
    question: str,
) -> dict[str, Any]:
    depth = context["depth"]
    depth_min, depth_max = _window_around_depth(
        depth,
        float(depth[index]),
        max(5.0, policy.boundary_review_half_width_ft / 2.0),
    )
    return _interval_row(
        context,
        policy,
        interval_id=review_type,
        review_type=review_type,
        depth_min=depth_min,
        depth_max=depth_max,
        recommended_human_question=question,
    )


def _interval_row(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    *,
    interval_id: str,
    review_type: str,
    depth_min: float,
    depth_max: float,
    recommended_human_question: str,
    boundary_zone: str | None = None,
) -> dict[str, Any]:
    depth = context["depth"]
    mask = (depth >= depth_min) & (depth <= depth_max)
    if not np.any(mask):
        nearest = int(np.nanargmin(np.abs(depth - ((depth_min + depth_max) / 2.0))))
        mask = np.zeros(depth.size, dtype=bool)
        mask[nearest] = True
        depth_min = float(depth[nearest])
        depth_max = float(depth[nearest])
    mid = (depth_min + depth_max) / 2.0
    regime = policy.regime_for_depth(mid)
    zone = boundary_zone or _optional_boundary_zone(policy, mid)
    row: dict[str, Any] = {
        "interval_id": interval_id,
        "review_type": review_type,
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
        "sample_count": int(np.count_nonzero(mask)),
        "regime_id": regime.id if regime else "",
        "boundary_zone": zone or "",
        "recommended_human_question": recommended_human_question,
        "review_only": True,
        "no_final_labels": True,
    }
    row.update(_target_view_scalar_summary(context, mask))
    row["morphology_summary"] = _morphology_nested_summary(context, mask)
    row["orientation_confidence"] = _finite_mean(
        context["covariates"]["orientation_confidence"][mask]
    )
    row["inclination"] = _finite_mean(context["covariates"]["inc_deg"][mask])
    row["near_far_ratio_summary"] = _near_far_nested_summary(context, mask)
    return row


def _target_view_summary(context: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for view in TARGET_VIEWS:
        values = _target_values(context, view)[mask]
        row[f"{view}_mean"] = _finite_mean(values)
        row[f"{view}_median"] = _finite_percentile(values, 50)
        row[f"{view}_zero_fraction"] = _fraction(values <= 0.0)
        row[f"{view}_p90"] = _finite_percentile(values, 90)
        row[f"{view}_p95"] = _finite_percentile(values, 95)
    return row


def _target_view_scalar_summary(context: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {view: _finite_mean(_target_values(context, view)[mask]) for view in TARGET_VIEWS}


def _morphology_summary(context: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in MORPHOLOGY_FIELDS:
        values = _morphology_values(context, field)[mask]
        row[f"{field}_mean"] = _finite_mean(values)
        row[f"{field}_p90"] = _finite_percentile(values, 90)
    return row


def _morphology_nested_summary(context: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        field: {
            "mean": _finite_mean(_morphology_values(context, field)[mask]),
            "p90": _finite_percentile(_morphology_values(context, field)[mask], 90),
        }
        for field in MORPHOLOGY_FIELDS
    }


def _physical_summary(context: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    invalid = context["invalid_by_depth"]
    label_conf = _primary_label_confidence(context)
    return {
        "orientation_confidence_mean": _finite_mean(
            context["covariates"]["orientation_confidence"][mask]
        ),
        "orientation_confidence_p10": _finite_percentile(
            context["covariates"]["orientation_confidence"][mask],
            10,
        ),
        "inclination_mean": _finite_mean(context["covariates"]["inc_deg"][mask]),
        "relbearing_finite_fraction": _fraction(
            np.isfinite(context["covariates"]["relbearing_deg"][mask])
        ),
        "label_confidence_mean": _finite_mean(label_conf[mask]),
        "invalid_zc_count": int(np.sum(invalid["negative_count"][mask])),
        "nonfinite_zc_count": int(np.sum(invalid["nonfinite_count"][mask])),
        "upper_zc_review_count": int(np.sum(invalid["upper_count"][mask])),
        "overlaps_5700_band": bool(
            np.any(
                (context["depth"][mask] >= REVIEW_BAND_MIN_FT)
                & (context["depth"][mask] <= REVIEW_BAND_MAX_FT)
            )
        ),
    }


def _near_far_summary(context: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    nested = _near_far_nested_summary(context, mask)
    return {
        "near_far_ratio_extreme_count": nested["extreme_count"],
        "near_far_ratio_mean_early_energy_mean": nested[
            "near_far_ratio_mean_early_energy"
        ]["mean"],
        "near_far_ratio_mean_rms_energy_mean": nested[
            "near_far_ratio_mean_rms_energy"
        ]["mean"],
        "near_far_ratio_mean_peak_abs_mean": nested[
            "near_far_ratio_mean_peak_abs"
        ]["mean"],
    }


def _near_far_nested_summary(context: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {
        "extreme_count": int(np.count_nonzero(_near_far_extreme_mask(context)[mask]))
    }
    for name in NEAR_FAR_FEATURES:
        index = context["near_far_indices"][name]
        values = context["feature_matrix"][:, index][mask]
        output[name] = {
            "mean": _finite_mean(values),
            "p90": _finite_percentile(values, 90),
            "max_abs": _finite_percentile(np.abs(values), 100),
        }
    return output


def _target_values(context: dict[str, Any], view: str) -> np.ndarray:
    return np.asarray(context["labels"][view], dtype=np.float32)[context["primary_index"]]


def _morphology_values(context: dict[str, Any], field: str) -> np.ndarray:
    return np.mean(
        np.asarray(context["labels"][field], dtype=np.float32)[context["primary_index"]],
        axis=1,
    )


def _window_around_depth(
    depth: np.ndarray,
    center_depth: float,
    half_width_ft: float,
) -> tuple[float, float]:
    low = center_depth - half_width_ft
    high = center_depth + half_width_ft
    mask = (depth >= low) & (depth <= high)
    if np.any(mask):
        return float(np.min(depth[mask])), float(np.max(depth[mask]))
    nearest = float(depth[int(np.nanargmin(np.abs(depth - center_depth)))])
    return nearest, nearest


def _optional_boundary_zone(policy: DepthRegimeReviewPolicy, depth_ft: float) -> str | None:
    zone = policy.boundary_zone_for_depth(depth_ft)
    return zone.zone_id if zone else None


def _boundary_support_summary(zone: BoundaryReviewZone) -> str:
    lookup = {
        "boundary_2582p79_ft": "target / feature / morphology / near-far support",
        "boundary_4219p52_ft": "morphology / orientation / inclination support",
        "boundary_5680p00_ft": "feature / 5700 band support",
        "boundary_2826p85_ft": "weak target change-point support",
        "boundary_3534p31_ft": "existing broad regime boundary",
        "boundary_3984p03_ft": "feature / morphology support",
        "boundary_4678p49_ft": "existing broad regime boundary",
    }
    return lookup.get(zone.zone_id, "review-only support")


def _interval_question(review_type: str, regime_id: str) -> str:
    questions = {
        "representative_high_target": f"Does {regime_id} high target prevalence look physical?",
        "representative_low_target": f"Does {regime_id} low target prevalence look stable?",
        "morphology_sensitive": f"Does {regime_id} morphology show continuity or scattered noise?",
        "receiver_max_sensitive": "Is receiver_max driven by a localized receiver extreme?",
        "receiver_mean_p90_disagreement": (
            "Does receiver_p90 look more robust than receiver_mean here?"
        ),
    }
    return questions[review_type]


def _boundary_question(zone: BoundaryReviewZone, review_type: str) -> str:
    return (
        f"At {zone.depth_ft:.2f} ft, does the {review_type} interval support a "
        "manual-review boundary only?"
    )


def _write_figures(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    broad_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    figure_paths: dict[str, Path],
    *,
    overwrite: bool,
) -> None:
    _plot_broad_regime_overview(
        context,
        policy,
        figure_paths["broad_regime_overview.png"],
        overwrite,
    )
    _plot_target_views_by_regime(
        broad_rows,
        figure_paths["target_views_by_regime.png"],
        overwrite,
    )
    _plot_target_views_vs_depth(
        context,
        policy,
        figure_paths["target_views_vs_depth.png"],
        overwrite,
    )
    _plot_receiver_mean_p90_max(
        context,
        policy,
        figure_paths["receiver_mean_p90_max_comparison.png"],
        overwrite,
    )
    _plot_single_series(
        context,
        policy,
        _target_values(context, "receiver_std"),
        "receiver_std",
        figure_paths["receiver_std_vs_depth.png"],
        overwrite,
    )
    _plot_morphology_by_regime(broad_rows, figure_paths["morphology_by_regime.png"], overwrite)
    _plot_morphology_vs_depth(context, policy, figure_paths["morphology_vs_depth.png"], overwrite)
    _plot_boundary_zone_overview(
        context,
        policy,
        figure_paths["boundary_zone_overview.png"],
        overwrite,
    )
    _plot_orientation_inclination(
        context,
        policy,
        figure_paths["orientation_inclination_vs_depth.png"],
        overwrite,
    )
    _plot_near_far_ratio_vs_depth(
        context,
        policy,
        figure_paths["near_far_ratio_vs_depth.png"],
        overwrite,
    )
    _plot_near_far_outliers_vs_boundaries(
        context,
        policy,
        figure_paths["near_far_outliers_vs_boundaries.png"],
        overwrite,
    )
    _plot_5700_band_review(context, policy, figure_paths["5700_band_review.png"], overwrite)
    _plot_secondary_boundary_zones(
        boundary_rows,
        figure_paths["secondary_boundary_zones_summary.png"],
        overwrite,
    )
    for zone in policy.priority_boundary_review_zones:
        if zone.priority == "primary":
            _plot_boundary_panel(
                context,
                policy,
                zone,
                figure_paths[f"{zone.zone_id}.png"],
                overwrite,
            )


def _plot_broad_regime_overview(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(context["depth"], _target_values(context, "receiver_max"), lw=1, label="receiver_max")
    ax.plot(context["depth"], _target_values(context, "receiver_p90"), lw=1, label="receiver_p90")
    _shade_regimes(ax, policy)
    _draw_boundary_zones(ax, policy)
    ax.set_title("Broad review regimes and priority boundary zones")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel("Target fraction")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, path, overwrite=overwrite)


def _plot_target_views_by_regime(
    rows: list[dict[str, Any]],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    labels = [row["regime_id"] for row in rows]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for index, view in enumerate(TARGET_VIEWS):
        values = [row.get(f"{view}_mean") or 0.0 for row in rows]
        ax.bar(x + (index - 2) * width, values, width=width, label=view)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Target-view means by formal review regime")
    ax.set_ylabel("Mean")
    ax.legend(fontsize=7, ncol=3)
    save_figure(fig, path, overwrite=overwrite)


def _plot_target_views_vs_depth(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for view in TARGET_VIEWS:
        ax.plot(context["depth"], _target_values(context, view), lw=1, label=view)
    _shade_regimes(ax, policy)
    ax.set_title("Target views vs depth")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=7, ncol=3)
    save_figure(fig, path, overwrite=overwrite)


def _plot_receiver_mean_p90_max(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for view in ("receiver_mean", "receiver_p90", "receiver_max"):
        ax.plot(context["depth"], _target_values(context, view), lw=1, label=view)
    _draw_boundary_zones(ax, policy)
    ax.set_title("receiver_mean / receiver_p90 / receiver_max comparison")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _plot_single_series(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    values: np.ndarray,
    title: str,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(context["depth"], values, lw=1, color="tab:purple")
    _draw_boundary_zones(ax, policy)
    ax.set_title(title)
    ax.set_xlabel("Depth ft")
    ax.set_ylabel(title)
    save_figure(fig, path, overwrite=overwrite)


def _plot_morphology_by_regime(
    rows: list[dict[str, Any]],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    labels = [row["regime_id"] for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for field in MORPHOLOGY_FIELDS[:-1]:
        ax.plot(x, [row.get(f"{field}_mean") or 0.0 for row in rows], marker="o", label=field)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Morphology arrays by formal review regime")
    ax.set_ylabel("Mean")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, path, overwrite=overwrite)


def _plot_morphology_vs_depth(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for field in MORPHOLOGY_FIELDS[:-1]:
        ax.plot(context["depth"], _morphology_values(context, field), lw=1, label=field)
    _draw_boundary_zones(ax, policy)
    ax.set_title("Morphology arrays vs depth")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, path, overwrite=overwrite)


def _plot_boundary_zone_overview(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(context["depth"], _target_values(context, "receiver_max"), lw=1, label="receiver_max")
    for zone in policy.priority_boundary_review_zones:
        color = "tab:red" if zone.priority == "primary" else "tab:gray"
        ax.axvspan(zone.depth_min_ft, zone.depth_max_ft, color=color, alpha=0.12)
        ax.axvline(zone.depth_ft, color=color, lw=0.8, alpha=0.6)
    ax.set_title("Boundary review zones")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel("receiver_max")
    ax.legend(fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _plot_orientation_inclination(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(
        context["depth"],
        context["covariates"]["orientation_confidence"],
        color="tab:blue",
        lw=1,
        label="orientation_confidence",
    )
    ax1.set_ylabel("Orientation confidence")
    ax2 = ax1.twinx()
    ax2.plot(
        context["depth"],
        context["covariates"]["inc_deg"],
        color="tab:orange",
        lw=1,
        label="inclination",
    )
    ax2.set_ylabel("Inclination deg")
    _draw_boundary_zones(ax1, policy)
    ax1.set_title("Orientation confidence and inclination vs depth")
    ax1.set_xlabel("Depth ft")
    save_figure(fig, path, overwrite=overwrite)


def _plot_near_far_ratio_vs_depth(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name in NEAR_FAR_FEATURES:
        index = context["near_far_indices"][name]
        ax.plot(context["depth"], context["feature_matrix"][:, index], lw=1, label=name)
    _draw_boundary_zones(ax, policy)
    ax.set_title("Near/far ratios vs depth")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _plot_near_far_outliers_vs_boundaries(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    name = "near_far_ratio_mean_early_energy"
    index = context["near_far_indices"][name]
    values = context["feature_matrix"][:, index]
    mask = np.abs(context["near_far_z"][name]) >= 5.0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(context["depth"], values, lw=1, color="tab:blue")
    ax.scatter(context["depth"][mask], values[mask], s=10, color="tab:red", label="robust outlier")
    _draw_boundary_zones(ax, policy)
    ax.set_title("Near/far outliers vs review boundaries")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel(name)
    ax.legend(fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _plot_5700_band_review(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    mask = (context["depth"] >= 5620.0) & (context["depth"] <= 5760.0)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(
        context["depth"][mask],
        _target_values(context, "receiver_max")[mask],
        lw=1,
        label="receiver_max",
    )
    ax.plot(
        context["depth"][mask],
        _morphology_values(context, "combined_channel_fraction")[mask],
        lw=1,
        label="combined_channel_fraction",
    )
    ax.axvspan(REVIEW_BAND_MIN_FT, REVIEW_BAND_MAX_FT, color="tab:green", alpha=0.16)
    _draw_boundary_zones(ax, policy)
    ax.set_title("5700 ft special review band")
    ax.set_xlabel("Depth ft")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _plot_secondary_boundary_zones(
    rows: list[dict[str, Any]],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    secondary = [row for row in rows if row["priority"] == "secondary"]
    labels = [row["zone_id"].replace("boundary_", "").replace("_ft", "") for row in secondary]
    support = [int(row["support_count"]) for row in secondary]
    target = [row.get("receiver_max_mean") or 0.0 for row in secondary]
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(labels))
    ax1.bar(x, support, color="tab:gray", alpha=0.7, label="support_count")
    ax1.set_ylabel("Support count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(x, target, color="tab:blue", marker="o", label="receiver_max_mean")
    ax2.set_ylabel("receiver_max mean")
    ax1.set_title("Secondary boundary-zone summary")
    save_figure(fig, path, overwrite=overwrite)


def _plot_boundary_panel(
    context: dict[str, Any],
    policy: DepthRegimeReviewPolicy,
    zone: BoundaryReviewZone,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    width = max(60.0, 4.0 * policy.boundary_review_half_width_ft)
    mask = (context["depth"] >= zone.depth_ft - width) & (context["depth"] <= zone.depth_ft + width)
    if not np.any(mask):
        mask = np.ones(context["depth"].size, dtype=bool)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(
        context["depth"][mask],
        _target_values(context, "receiver_mean")[mask],
        label="receiver_mean",
    )
    axes[0].plot(
        context["depth"][mask],
        _target_values(context, "receiver_p90")[mask],
        label="receiver_p90",
    )
    axes[0].plot(
        context["depth"][mask],
        _target_values(context, "receiver_max")[mask],
        label="receiver_max",
    )
    axes[0].set_ylabel("Target")
    axes[0].legend(fontsize=7, ncol=3)
    axes[1].plot(
        context["depth"][mask],
        _morphology_values(context, "combined_channel_fraction")[mask],
        color="tab:green",
        label="combined morphology",
    )
    axes[1].set_ylabel("Morphology")
    axes[1].legend(fontsize=7)
    axes[2].plot(
        context["depth"][mask],
        context["covariates"]["orientation_confidence"][mask],
        color="tab:blue",
        label="orientation",
    )
    axes[2].plot(
        context["depth"][mask],
        context["covariates"]["inc_deg"][mask],
        color="tab:orange",
        label="inclination",
    )
    axes[2].set_ylabel("Covariates")
    axes[2].set_xlabel("Depth ft")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.axvspan(zone.depth_min_ft, zone.depth_max_ft, color="tab:red", alpha=0.12)
        ax.axvline(zone.depth_ft, color="tab:red", lw=0.8)
    fig.suptitle(f"Primary boundary review panel: {zone.depth_ft:.2f} ft")
    save_figure(fig, path, overwrite=overwrite)


def _shade_regimes(ax: Any, policy: DepthRegimeReviewPolicy) -> None:
    colors = ("tab:blue", "tab:green", "tab:orange")
    for regime, color in zip(policy.broad_review_regimes, colors, strict=True):
        ax.axvspan(regime.depth_min_ft, regime.depth_max_ft, color=color, alpha=0.05)


def _draw_boundary_zones(ax: Any, policy: DepthRegimeReviewPolicy) -> None:
    for zone in policy.priority_boundary_review_zones:
        color = "tab:red" if zone.priority == "primary" else "black"
        alpha = 0.18 if zone.priority == "primary" else 0.08
        ax.axvspan(zone.depth_min_ft, zone.depth_max_ft, color=color, alpha=alpha, linewidth=0)
        ax.axvline(zone.depth_ft, color=color, lw=0.7, alpha=0.45)


def _format_review_summary(
    pack: GeometryRegressionFormalReviewPack,
    broad_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    figure_paths: dict[str, Path],
) -> str:
    lines = [
        "# Geometry Regression Formal Manual Review Pack",
        "",
        "This pack applies the formal review-only depth-regime policy. It does not "
        "modify CV splits, train models, change primary target view, create labels, "
        "add features, or enter MVP-4C.",
        "",
        f"- review_version: `{pack.review_version}`",
        f"- broad_regime_count: `{len(broad_rows)}`",
        f"- boundary_zone_count: `{len(boundary_rows)}`",
        f"- selected_manual_review_interval_count: `{len(intervals)}`",
        f"- stage_10_stop_still_valid: `{pack.stage_10_stop_still_valid}`",
        f"- stage_11_12_still_blocked: `{pack.stage_11_12_still_blocked}`",
        "",
        "## Broad Regimes",
        "",
    ]
    for row in broad_rows:
        lines.append(
            "- "
            f"{row['regime_id']}: {row['depth_min_ft']:.2f}-{row['depth_max_ft']:.2f} ft, "
            f"receiver_max_mean={row.get('receiver_max_mean')}"
        )
    lines.extend(["", "## Boundary Zones", ""])
    for row in boundary_rows:
        lines.append(
            "- "
            f"{row['zone_id']}: depth={row['depth_ft']:.2f} ft, "
            f"priority={row['priority']}, support={row['support_count']}, "
            f"{row['support_summary']}"
        )
    lines.extend(["", "## Figures", ""])
    for name, path in figure_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in pack.warnings] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def _format_reviewer_checklist() -> str:
    lines = [
        "# Geometry Regression Formal Review Checklist",
        "",
        "All answers must be interpreted as manual-review guidance only. Do not "
        "treat any checked item as a final label, CV split change, target-view "
        "approval, or model-training approval.",
        "",
    ]
    for index, question in enumerate(REVIEWER_CHECKLIST_QUESTIONS, start=1):
        lines.append(f"{index}. [ ] {question}")
    lines.append("")
    return "\n".join(lines)


def _format_reviewer_decision_template(
    policy: DepthRegimeReviewPolicy,
    intervals: list[dict[str, Any]],
) -> str:
    lines = [
        "# Geometry Regression Reviewer Decision Template",
        "",
        "- reviewer:",
        "- review_date:",
        "- reviewed_interval_count:",
        f"- available_interval_count: `{len(intervals)}`",
        "- decision_scope: `manual_review_only`",
        "- formal_cv_split_change_approved: `false unless separately approved`",
        "- final_labels_approved: `false`",
        "",
        "## Policy",
        "",
        f"- boundary_review_half_width_ft: `{policy.boundary_review_half_width_ft}`",
        "- broad regimes are descriptive review groups only.",
        "- receiver_p90 is a candidate for discussion only, not the primary target.",
        "",
        "## Reviewer Decision",
        "",
        "- [ ] Continue manual review only",
        "- [ ] Request formal CV stratification study",
        "- [ ] Request target-view decision study",
        "- [ ] Request morphology-aware label redesign discussion",
        "- [ ] Request preprocessing review",
        "- [ ] Request controlled feature review",
        "- [ ] Stop current line",
        "",
        "## Notes",
        "",
    ]
    return "\n".join(lines)


def _decision_pack(
    pack: GeometryRegressionFormalReviewPack,
    broad_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    user_decisions = [
        "Whether to approve a formal CV stratification study.",
        "Whether to approve a target-view decision study.",
        "Whether to discuss morphology-aware label redesign.",
        "Whether any preprocessing review is warranted.",
        "Whether controlled feature review should remain blocked or be scoped separately.",
    ]
    return {
        "decision_pack_version": DECISION_PACK_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "approved_review_only_policy": pack.review_policy["review_policy"],
        "broad_regimes": broad_rows,
        "boundary_zones": boundary_rows,
        "target_view_policy": pack.review_policy["target_view_policy"],
        "morphology_review_policy": pack.review_policy["morphology_policy"],
        "manual_review_interval_count": len(intervals),
        "questions_requiring_user_decision": user_decisions,
        "stage_10_stop_still_valid": True,
        "stage_11_12_still_blocked": True,
        "mvp4c_stc_apes_deep_learning_final_labels_still_forbidden": True,
        "not_authorized": [
            "formal CV split change",
            "regime-specific model training",
            "primary target-view change",
            "new morphology target",
            "label definition change",
            "new feature",
            "near/far preprocessing change",
            "waveform read",
            "MVP-4C",
            "STC",
            "APES",
            "deep learning",
            "final labels",
            "ground-truth claim",
        ],
    }


def _format_decision_pack_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Geometry Regression Formal Review Decision Pack",
        "",
        f"- decision_pack_version: `{decision['decision_pack_version']}`",
        f"- manual_review_interval_count: `{decision['manual_review_interval_count']}`",
        f"- stage_10_stop_still_valid: `{decision['stage_10_stop_still_valid']}`",
        f"- stage_11_12_still_blocked: `{decision['stage_11_12_still_blocked']}`",
        (
            "- mvp4c_stc_apes_deep_learning_final_labels_still_forbidden: "
            f"`{decision['mvp4c_stc_apes_deep_learning_final_labels_still_forbidden']}`"
        ),
        "",
        "## Approved Review-Only Policy",
        "",
    ]
    for key, value in decision["approved_review_only_policy"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Broad Regimes", ""])
    for row in decision["broad_regimes"]:
        lines.append(
            f"- {row['regime_id']}: {row['depth_min_ft']:.2f}-{row['depth_max_ft']:.2f} ft"
        )
    lines.extend(["", "## Boundary Zones", ""])
    for row in decision["boundary_zones"]:
        lines.append(
            "- "
            f"{row['zone_id']}: {row['depth_ft']:.2f} ft, "
            f"priority={row['priority']}, support={row['support_count']}"
        )
    lines.extend(["", "## Target-View Policy", ""])
    for key, value in decision["target_view_policy"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Morphology Review Policy", ""])
    for key, value in decision["morphology_review_policy"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## User Decisions Required", ""])
    for question in decision["questions_requiring_user_decision"]:
        lines.append(f"- {question}")
    lines.extend(["", "## Not Authorized", ""])
    for item in decision["not_authorized"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return None if values.size == 0 else float(np.mean(values))
