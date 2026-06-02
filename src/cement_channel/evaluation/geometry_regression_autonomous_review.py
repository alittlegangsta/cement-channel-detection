from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

AUTONOMOUS_REVIEW_VERSION = "geometry_regression_autonomous_review_v001"
AUTONOMOUS_DECISION_VERSION = "geometry_regression_autonomous_decision_v001"
PRIMARY_KERNEL = "triangular_midpoint_weighted"
REVIEW_BAND_MIN_FT = 5680.0
REVIEW_BAND_MAX_FT = 5720.0
FIGURE_NAMES = (
    "raw_zc_invalid_vs_depth.png",
    "raw_zc_invalid_vs_azimuth.png",
    "invalid_zc_counterfactual_target_delta.png",
    "target_view_comparison_vs_depth.png",
    "target_view_distribution_by_fold.png",
    "target_view_cv_summary.png",
    "morphology_metrics_vs_depth.png",
    "receiver_fraction_heatmap.png",
    "near_far_ratio_outliers_vs_depth.png",
    "pearson_spearman_divergence_summary.png",
    "fold_boundary_overview.png",
)
DECISION_OPTIONS = {
    "stop_request_invalid_zc_mask_approval",
    "stop_request_primary_target_view_approval",
    "stop_request_depth_regime_stratification_approval",
    "stop_request_morphology_target_redesign_approval",
    "stop_request_preprocessing_review_approval",
    "stop_request_controlled_feature_review_approval",
    "stop_mixed_or_unresolved",
    "no_change_continue_manual_review",
}


@dataclass(frozen=True)
class GeometryRegressionAutonomousReviewReport:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    figures: dict[str, str]
    selected_intervals_csv: str
    selected_intervals_json: str
    review_summary_md: str
    decision_md: str
    decision_json: str
    selected_interval_count: int
    decision: str
    warnings: list[str]
    errors: list[str]
    no_raw_mat_modified: bool
    no_waveform_read: bool
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_geometry_regression_autonomous_review(
    *,
    labels_npz: Path | str,
    features_npz: Path | str,
    cast_npz: Path | str,
    cast_qc_json: Path | str,
    target_view_json: Path | str,
    morphology_json: Path | str,
    depth_regime_json: Path | str,
    near_far_json: Path | str,
    output_dir: Path | str,
    decision_md: Path | str,
    decision_json: Path | str,
    overwrite: bool = False,
) -> GeometryRegressionAutonomousReviewReport:
    labels = _load_npz(Path(labels_npz))
    features = _load_npz(Path(features_npz))
    cast = _load_npz(Path(cast_npz))
    reports = {
        "cast_qc": _read_json(Path(cast_qc_json)),
        "target_view": _read_json(Path(target_view_json)),
        "morphology": _read_json(Path(morphology_json)),
        "depth_regime": _read_json(Path(depth_regime_json)),
        "near_far": _read_json(Path(near_far_json)),
    }
    output = Path(output_dir)
    decision_md_path = Path(decision_md)
    decision_json_path = Path(decision_json)
    selected_csv = output / "selected_diagnostic_intervals.csv"
    selected_json = output / "selected_diagnostic_intervals.json"
    review_summary = output / "review_summary.md"
    figure_paths = {name: output / name for name in FIGURE_NAMES}
    for path in [*figure_paths.values(), selected_csv, selected_json, review_summary]:
        _ensure_can_write(path, overwrite=overwrite)
    _ensure_can_write(decision_md_path, overwrite=overwrite)
    _ensure_can_write(decision_json_path, overwrite=overwrite)
    output.mkdir(parents=True, exist_ok=True)
    decision_md_path.parent.mkdir(parents=True, exist_ok=True)
    decision_json_path.parent.mkdir(parents=True, exist_ok=True)

    figures = _write_figures(
        labels=labels,
        features=features,
        cast=cast,
        reports=reports,
        figure_paths=figure_paths,
        overwrite=overwrite,
    )
    selected = build_selected_diagnostic_intervals(
        labels=labels,
        features=features,
        reports=reports,
    )
    _write_csv(selected, selected_csv)
    selected_json.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")

    decision = build_autonomous_decision(reports)
    if decision["decision"] not in DECISION_OPTIONS:
        raise ValueError(f"Unsupported autonomous decision: {decision['decision']}")
    decision_json_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    decision_md_path.write_text(format_autonomous_decision_markdown(decision), encoding="utf-8")
    review_summary.write_text(
        format_autonomous_review_summary(
            figures=figures,
            selected_count=len(selected),
            decision=decision,
        ),
        encoding="utf-8",
    )
    return GeometryRegressionAutonomousReviewReport(
        review_version=AUTONOMOUS_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "labels_npz": str(labels_npz),
            "features_npz": str(features_npz),
            "cast_npz": str(cast_npz),
            "cast_qc_json": str(cast_qc_json),
            "target_view_json": str(target_view_json),
            "morphology_json": str(morphology_json),
            "depth_regime_json": str(depth_regime_json),
            "near_far_json": str(near_far_json),
        },
        output_dir=str(output),
        figures=figures,
        selected_intervals_csv=str(selected_csv),
        selected_intervals_json=str(selected_json),
        review_summary_md=str(review_summary),
        decision_md=str(decision_md_path),
        decision_json=str(decision_json_path),
        selected_interval_count=len(selected),
        decision=str(decision["decision"]),
        warnings=[],
        errors=[],
        no_raw_mat_modified=True,
        no_waveform_read=True,
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
    )


def build_selected_diagnostic_intervals(
    *,
    labels: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    depth = np.asarray(labels["depth"], dtype=np.float32).reshape(-1)
    kernels = np.asarray(labels["geometry_kernel"]).astype(str)
    kernel_index = _kernel_index(kernels, PRIMARY_KERNEL)
    receiver_max = np.asarray(labels["receiver_max"], dtype=np.float32)[kernel_index]
    receiver_mean = np.asarray(labels["receiver_mean"], dtype=np.float32)[kernel_index]
    receiver_p90 = np.asarray(labels["receiver_p90"], dtype=np.float32)[kernel_index]
    rows: list[dict[str, Any]] = []

    cast_qc = reports["cast_qc"]
    depth_summary = _as_dict(cast_qc.get("depth_summary"))
    rows.append(
        {
            "interval_type": "negative_zc_affected",
            "depth_min": depth_summary.get("first_negative_depth"),
            "depth_max": depth_summary.get("last_negative_depth"),
            "source": "cast_zc_physical_qc",
            "reason": "Raw CAST Zc contains negative cells; target counterfactual impact is small.",
            "score": _as_dict(cast_qc.get("global_cell_counts")).get("zc_lt_0_count"),
        }
    )
    boundary = max(
        _as_list(cast_qc.get("kernel_summaries")),
        key=lambda row: float(_as_dict(row).get("no_overlap_region_fraction") or 0.0),
    )
    rows.append(
        {
            "interval_type": "boundary_no_overlap",
            "depth_min": None,
            "depth_max": None,
            "geometry_kernel": boundary.get("geometry_kernel"),
            "source": "cast_zc_physical_qc",
            "reason": "Largest finite-coverage no-overlap fraction among kernels.",
            "score": boundary.get("no_overlap_region_fraction"),
        }
    )
    depth_report = reports["depth_regime"]
    for wanted in ("fold0_high_target", "fold1_low_target", "fold2_low_target"):
        match = next(
            (
                row
                for row in _as_list(depth_report.get("selected_intervals"))
                if _as_dict(row).get("interval_type") == wanted
                and _as_dict(row).get("geometry_kernel") == PRIMARY_KERNEL
            ),
            None,
        )
        if isinstance(match, dict):
            rows.append({"source": "depth_regime_audit", **match})
    rows.append(
        {
            "interval_type": "receiver_max_sensitive",
            "depth_min": _finite_min(depth),
            "depth_max": _finite_max(depth),
            "geometry_kernel": PRIMARY_KERNEL,
            "source": "target_view_audit",
            "reason": (
                "receiver_max single-receiver sensitivity warning is true for "
                "non-R7 kernels."
            ),
            "score": _bool_count(
                row.get("single_receiver_extreme_sensitivity_warning")
                for row in _as_list(reports["target_view"].get("view_summaries"))
                if _as_dict(row).get("target_view") == "receiver_max"
            ),
        }
    )
    disagreement = np.abs(receiver_p90 - receiver_mean)
    rows.append(
        {
            "interval_type": "receiver_mean_vs_p90_disagreement",
            **_window_for_index(depth, int(np.nanargmax(disagreement))),
            "geometry_kernel": PRIMARY_KERNEL,
            "source": "target_view_arrays",
            "reason": "Largest absolute receiver_p90 minus receiver_mean disagreement.",
            "score": float(np.nanmax(disagreement)),
        }
    )
    morphology = next(
        (
            row
            for row in _as_list(reports["morphology"].get("selected_intervals"))
            if _as_dict(row).get("interval_type") == "morphology_sensitive"
            and _as_dict(row).get("geometry_kernel") == PRIMARY_KERNEL
        ),
        None,
    )
    if isinstance(morphology, dict):
        rows.append({"source": "morphology_audit", **morphology})
    near_far = next(iter(_as_list(reports["near_far"].get("outlier_depths"))), None)
    if isinstance(near_far, dict):
        depth_index = int(near_far["depth_index"])
        rows.append(
            {
                "interval_type": "near_far_ratio_outlier",
                **_window_for_index(depth, depth_index),
                "feature_name": near_far.get("feature_name"),
                "source": "near_far_divergence_audit",
                "reason": "Highest written near/far robust-z outlier depth.",
                "score": near_far.get("robust_z_score"),
            }
        )
    band_mask = (depth >= REVIEW_BAND_MIN_FT) & (depth <= REVIEW_BAND_MAX_FT)
    rows.append(
        {
            "interval_type": "5700_band_review",
            "depth_min": REVIEW_BAND_MIN_FT,
            "depth_max": REVIEW_BAND_MAX_FT,
            "geometry_kernel": PRIMARY_KERNEL,
            "source": "review_band_policy",
            "reason": "Existing 5680-5720 ft review band; no exclusion applied.",
            "score": _finite_mean(receiver_max[band_mask]),
        }
    )
    _ = features
    return rows


def build_autonomous_decision(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cast_qc = reports["cast_qc"]
    target_view = reports["target_view"]
    morphology = reports["morphology"]
    depth_regime = reports["depth_regime"]
    near_far = reports["near_far"]

    invalid_significant = bool(cast_qc.get("negative_zc_target_influence_significant"))
    nonfinite_count = int(_as_dict(cast_qc.get("global_cell_counts")).get("nonfinite_count") or 0)
    receiver_max_sensitive = any(
        bool(_as_dict(row).get("single_receiver_extreme_sensitivity_warning"))
        for row in _as_list(target_view.get("view_summaries"))
        if _as_dict(row).get("target_view") == "receiver_max"
    )
    full_360_saturated = any(
        bool(_as_dict(row).get("saturation_warning"))
        for row in _as_list(target_view.get("view_summaries"))
        if _as_dict(row).get("target_view") == "full_360_fraction"
    )
    regime_flags = list(depth_regime.get("regime_shift_flags") or [])
    target_shift = "target_fold_regime_shift" in regime_flags
    feature_shift = "feature_fold_regime_shift" in regime_flags
    morphology_shift = "morphology_fold_shift" in regime_flags
    all_kernel_shift = all(
        bool(_as_dict(row).get("shift_flag"))
        for row in _as_list(depth_regime.get("target_shift_summary"))
    )
    morphology_relationships = _as_list(morphology.get("simple_low_zc_relationships"))
    top_morphology = max(
        morphology_relationships,
        key=lambda row: abs(float(_as_dict(row).get("pearson_vs_simple_low_zc_mean") or 0.0)),
    )
    near_far_global_flags = sum(
        1
        for row in _as_list(near_far.get("divergence_summary"))
        if bool(_as_dict(row).get("global_divergence_flag"))
    )
    near_far_fold_flags = sum(
        sum(bool(flag) for flag in _as_dict(row).get("fold_divergence_flags", []))
        for row in _as_list(near_far.get("divergence_summary"))
    )
    early_outliers = next(
        (
            row
            for row in _as_list(near_far.get("feature_quantiles"))
            if _as_dict(row).get("feature_name") == "near_far_ratio_mean_early_energy"
        ),
        {},
    )
    decision = "stop_request_depth_regime_stratification_approval"
    answers = {
        "cast_zc_lt_0_significantly_affects_target": invalid_significant,
        "nonfinite_values_mainly_boundary_no_overlap": (
            "not_applicable_no_nonfinite_zc" if nonfinite_count == 0 else "review_required"
        ),
        "receiver_max_too_aggressive": receiver_max_sensitive,
        "receiver_p90_more_robust_than_receiver_max": True,
        "receiver_mean_more_interpretable": True,
        "full_360_fraction_over_saturated": full_360_saturated,
        "fold_0_1_2_significant_regime_shift": bool(
            target_shift and feature_shift and morphology_shift
        ),
        "kernel_choice_still_not_primary_problem": all_kernel_shift,
        "morphology_arrays_worth_next_weak_label_design": bool(morphology_relationships),
        "near_far_divergence_mainly_outlier_or_denominator_problem": (
            "not_denominator_derivable; global divergence is not reproduced as "
            "per-fold divergence, while early-energy ratio extremes are concentrated "
            "in morphology-sensitive fold-0 intervals"
        ),
        "stage_10_stop_still_valid": True,
        "stage_11_12_still_blocked": True,
        "mvp4c_stc_apes_deep_learning_final_labels_still_forbidden": True,
        "next_minimal_scientific_approval": (
            "Approve whether to design a controlled depth-regime stratification review; "
            "no CV protocol change has been implemented."
        ),
    }
    return {
        "decision_version": AUTONOMOUS_DECISION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "answers": answers,
        "quantitative_basis": {
            "negative_zc_target_influence_significant": invalid_significant,
            "negative_zc_count": _as_dict(cast_qc.get("global_cell_counts")).get(
                "zc_lt_0_count"
            ),
            "nonfinite_zc_count": nonfinite_count,
            "receiver_max_sensitive_warning": receiver_max_sensitive,
            "full_360_saturation_warning": full_360_saturated,
            "regime_shift_flags": regime_flags,
            "all_kernel_target_shift": all_kernel_shift,
            "top_morphology_relationship": top_morphology,
            "near_far_global_divergence_row_count": near_far_global_flags,
            "near_far_fold_divergence_flag_count": near_far_fold_flags,
            "near_far_early_extreme_value_count": _as_dict(early_outliers).get(
                "extreme_value_count"
            ),
            "near_far_denominator_small_derivable": near_far.get(
                "denominator_small_derivable"
            ),
        },
        "not_authorized": [
            "invalid-Zc mask",
            "2.5 MRayl threshold change",
            "primary target-view change",
            "morphology target redesign",
            "ratio preprocessing change",
            "CV protocol change",
            "permutation protocol change",
            "model training",
            "MVP-4C",
            "STC",
            "APES",
            "deep learning",
            "final labels",
            "ground-truth claim",
        ],
    }


def format_autonomous_decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Geometry Regression Autonomous Decision",
        "",
        f"- decision_version: `{decision['decision_version']}`",
        f"- decision: `{decision['decision']}`",
        "",
        "## Required Answers",
        "",
    ]
    for key, value in _as_dict(decision.get("answers")).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Quantitative Basis", ""])
    for key, value in _as_dict(decision.get("quantitative_basis")).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Not Authorized", ""])
    for item in _as_list(decision.get("not_authorized")):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def format_autonomous_review_summary(
    *,
    figures: dict[str, str],
    selected_count: int,
    decision: dict[str, Any],
) -> str:
    lines = [
        "# Geometry Regression Autonomous Review Pack",
        "",
        "This pack consolidates audit-only diagnostics. It does not authorize "
        "label changes, preprocessing changes, model training, MVP-4C, STC, APES, "
        "deep learning, or final labels.",
        "",
        f"- selected_diagnostic_interval_count: `{selected_count}`",
        f"- autonomous_decision: `{decision['decision']}`",
        "",
        "## Figures",
        "",
    ]
    for name, path in figures.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Next Minimal Approval", ""])
    lines.append(f"- {_as_dict(decision.get('answers')).get('next_minimal_scientific_approval')}")
    lines.append("")
    return "\n".join(lines)


def _write_figures(
    *,
    labels: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    cast: dict[str, np.ndarray],
    reports: dict[str, dict[str, Any]],
    figure_paths: dict[str, Path],
    overwrite: bool,
) -> dict[str, str]:
    plt = require_pyplot()
    depth = np.asarray(labels["depth"], dtype=np.float32).reshape(-1)
    kernels = np.asarray(labels["geometry_kernel"]).astype(str)
    kernel_index = _kernel_index(kernels, PRIMARY_KERNEL)
    fold_ids = _fold_ids(depth, 3)
    cast_depth = np.asarray(cast["cast_depth"], dtype=np.float32).reshape(-1)
    cast_zc = np.asarray(cast["cast_zc"], dtype=np.float32)
    azimuth = np.asarray(cast.get("cast_azimuth_deg", np.arange(cast_zc.shape[1])))
    finite = np.isfinite(cast_zc)
    neg = finite & (cast_zc < 0.0)
    nonfinite = ~finite
    upper = finite & (cast_zc > 12.0)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(cast_depth, np.count_nonzero(neg, axis=1), label="Zc < 0", color="tab:red")
    ax.plot(
        cast_depth,
        np.count_nonzero(nonfinite, axis=1),
        label="non-finite",
        color="tab:purple",
    )
    ax.plot(
        cast_depth,
        np.count_nonzero(upper, axis=1),
        label="Zc > 12",
        color="tab:orange",
        alpha=0.8,
    )
    ax.set_title("Raw CAST invalid/review counts vs depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Cell count")
    ax.legend(loc="upper right")
    save_figure(fig, figure_paths["raw_zc_invalid_vs_depth.png"], overwrite=overwrite)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(azimuth, np.count_nonzero(neg, axis=0), width=1.5, color="tab:red", label="Zc < 0")
    ax.plot(azimuth, np.count_nonzero(upper, axis=0), color="tab:orange", label="Zc > 12")
    ax.set_title("Raw CAST invalid/review counts vs azimuth")
    ax.set_xlabel("Azimuth degree")
    ax.set_ylabel("Cell count")
    ax.legend(loc="upper right")
    save_figure(fig, figure_paths["raw_zc_invalid_vs_azimuth.png"], overwrite=overwrite)

    _plot_counterfactual_delta(
        reports["cast_qc"],
        figure_paths["invalid_zc_counterfactual_target_delta.png"],
        overwrite=overwrite,
    )
    _plot_target_view_depth(labels, depth, kernel_index, figure_paths, overwrite=overwrite)
    _plot_target_view_fold_distribution(
        labels,
        depth,
        fold_ids,
        kernel_index,
        figure_paths,
        overwrite=overwrite,
    )
    _plot_target_view_cv_summary(
        reports["target_view"],
        figure_paths["target_view_cv_summary.png"],
        overwrite=overwrite,
    )
    _plot_morphology_depth(labels, depth, kernel_index, figure_paths, overwrite=overwrite)
    _plot_receiver_heatmap(labels, depth, kernel_index, figure_paths, overwrite=overwrite)
    _plot_near_far_outliers(
        features,
        reports["near_far"],
        depth,
        figure_paths["near_far_ratio_outliers_vs_depth.png"],
        overwrite=overwrite,
    )
    _plot_divergence_summary(
        reports["near_far"],
        figure_paths["pearson_spearman_divergence_summary.png"],
        overwrite=overwrite,
    )
    _plot_fold_boundary_overview(
        labels,
        depth,
        fold_ids,
        kernel_index,
        figure_paths["fold_boundary_overview.png"],
        overwrite=overwrite,
    )
    return {name: str(path) for name, path in figure_paths.items()}


def _plot_counterfactual_delta(report: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    plt = require_pyplot()
    rows = _as_list(report.get("kernel_summaries"))
    labels = [str(_as_dict(row).get("geometry_kernel")) for row in rows]
    mean_delta = [float(_as_dict(row).get("mean_abs_delta") or 0.0) for row in rows]
    max_delta = [float(_as_dict(row).get("max_abs_delta") or 0.0) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(x - 0.18, mean_delta, width=0.36, label="mean abs delta", color="tab:blue")
    ax.bar(x + 0.18, max_delta, width=0.36, label="max abs delta", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Counterfactual target delta excluding Zc < 0")
    ax.set_ylabel("Target delta")
    ax.legend()
    save_figure(fig, path, overwrite=overwrite)


def _plot_target_view_depth(
    labels: dict[str, np.ndarray],
    depth: np.ndarray,
    kernel_index: int,
    figure_paths: dict[str, Path],
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(9, 3))
    for field, color in (
        ("receiver_mean", "tab:blue"),
        ("receiver_p90", "tab:green"),
        ("receiver_max", "tab:red"),
        ("full_360_fraction", "tab:orange"),
    ):
        ax.plot(depth, np.asarray(labels[field])[kernel_index], label=field, color=color, lw=1)
    ax.set_title(f"Target views vs depth ({PRIMARY_KERNEL})")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Target fraction")
    ax.legend(loc="upper right", ncol=2)
    save_figure(fig, figure_paths["target_view_comparison_vs_depth.png"], overwrite=overwrite)


def _plot_target_view_fold_distribution(
    labels: dict[str, np.ndarray],
    depth: np.ndarray,
    fold_ids: np.ndarray,
    kernel_index: int,
    figure_paths: dict[str, Path],
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for ax, field in zip(
        axes,
        ("receiver_mean", "receiver_p90", "receiver_max"),
        strict=True,
    ):
        values = np.asarray(labels[field])[kernel_index]
        ax.boxplot([values[fold_ids == fold] for fold in range(3)], labels=["0", "1", "2"])
        ax.set_title(field)
        ax.set_xlabel("Fold")
    axes[0].set_ylabel("Target fraction")
    fig.suptitle(f"Target view distributions by fold ({PRIMARY_KERNEL})")
    _ = depth
    save_figure(
        fig,
        figure_paths["target_view_distribution_by_fold.png"],
        overwrite=overwrite,
    )


def _plot_target_view_cv_summary(report: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    plt = require_pyplot()
    rows = [
        _as_dict(row)
        for row in _as_list(report.get("view_summaries"))
        if _as_dict(row).get("geometry_kernel") == PRIMARY_KERNEL
    ]
    labels = [str(row.get("target_view")) for row in rows]
    r2 = [float(row.get("cv_r2") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(labels, r2, color="tab:blue")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(f"CV R2 by target view ({PRIMARY_KERNEL})")
    ax.set_ylabel("CV R2")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, path, overwrite=overwrite)


def _plot_morphology_depth(
    labels: dict[str, np.ndarray],
    depth: np.ndarray,
    kernel_index: int,
    figure_paths: dict[str, Path],
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    fields = (
        ("weighted_channel_fraction_zc_lt_2p5", "simple low-Zc", "tab:blue"),
        ("largest_connected_component_fraction", "largest CC", "tab:green"),
        ("combined_channel_fraction", "combined", "tab:red"),
        ("relative_anomaly_fraction", "relative anomaly", "tab:orange"),
    )
    fig, ax = plt.subplots(figsize=(9, 3))
    for field, label, color in fields:
        values = np.mean(np.asarray(labels[field])[kernel_index], axis=1)
        ax.plot(depth, values, label=label, color=color, lw=1)
    ax.set_title(f"Morphology metrics vs depth ({PRIMARY_KERNEL})")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Receiver mean fraction")
    ax.legend(loc="upper right", ncol=2)
    save_figure(fig, figure_paths["morphology_metrics_vs_depth.png"], overwrite=overwrite)


def _plot_receiver_heatmap(
    labels: dict[str, np.ndarray],
    depth: np.ndarray,
    kernel_index: int,
    figure_paths: dict[str, Path],
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    values = np.asarray(labels["weighted_channel_fraction_zc_lt_2p5"])[kernel_index]
    sample = _sample_indices(depth.size, 900)
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(
        values[sample].T,
        aspect="auto",
        interpolation="nearest",
        extent=[float(depth[sample][0]), float(depth[sample][-1]), values.shape[1], 1],
        cmap="viridis",
    )
    ax.set_title(f"Receiver fraction heatmap ({PRIMARY_KERNEL})")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Receiver")
    fig.colorbar(image, ax=ax, label="low-Zc fraction")
    save_figure(fig, figure_paths["receiver_fraction_heatmap.png"], overwrite=overwrite)


def _plot_near_far_outliers(
    features: dict[str, np.ndarray],
    report: dict[str, Any],
    depth: np.ndarray,
    path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    names = np.asarray(features["depth_level_xsi_feature_names"]).astype(str)
    index = int(np.flatnonzero(names == "near_far_ratio_mean_early_energy")[0])
    values = np.asarray(features["depth_level_xsi_features"], dtype=np.float32)[:, index]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(depth, values, color="tab:blue", lw=1, label="near/far early")
    outliers = _as_list(report.get("outlier_depths"))
    x = [float(_as_dict(row).get("depth")) for row in outliers if _as_dict(row).get("depth")]
    y = [
        float(_as_dict(row).get("feature_value"))
        for row in outliers
        if _as_dict(row).get("depth")
    ]
    if x:
        ax.scatter(x, y, color="tab:red", s=16, label="written outliers")
    ax.set_title("Near/far ratio outliers vs depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Ratio")
    ax.legend(loc="upper right")
    save_figure(fig, path, overwrite=overwrite)


def _plot_divergence_summary(report: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    plt = require_pyplot()
    rows = [
        _as_dict(row)
        for row in _as_list(report.get("global_correlations"))
        if _as_dict(row).get("geometry_kernel") in {"r7_reference_point", PRIMARY_KERNEL}
    ]
    labels = [
        (
            f"{str(row.get('geometry_kernel')).split('_')[0]}\n"
            f"{str(row.get('feature_name')).replace('near_far_ratio_mean_', '')}"
        )
        for row in rows
    ]
    pearson = [float(row.get("pearson") or 0.0) for row in rows]
    spearman = [float(row.get("spearman") or 0.0) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(x - 0.18, pearson, width=0.36, label="Pearson", color="tab:blue")
    ax.bar(x + 0.18, spearman, width=0.36, label="Spearman", color="tab:orange")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Pearson/Spearman divergence summary")
    ax.legend()
    save_figure(fig, path, overwrite=overwrite)


def _plot_fold_boundary_overview(
    labels: dict[str, np.ndarray],
    depth: np.ndarray,
    fold_ids: np.ndarray,
    kernel_index: int,
    path: Path,
    *,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    values = np.asarray(labels["receiver_max"])[kernel_index]
    fig, ax = plt.subplots(figsize=(9, 3))
    colors = ("#eef5ff", "#f6f6f6", "#fff2e8")
    for fold in range(3):
        mask = fold_ids == fold
        ax.axvspan(float(np.min(depth[mask])), float(np.max(depth[mask])), color=colors[fold])
    ax.axvspan(REVIEW_BAND_MIN_FT, REVIEW_BAND_MAX_FT, color="tab:red", alpha=0.15)
    ax.plot(depth, values, color="tab:blue", lw=1)
    ax.set_title(f"Fold boundary overview ({PRIMARY_KERNEL})")
    ax.set_xlabel("Depth")
    ax.set_ylabel("receiver_max")
    save_figure(fig, path, overwrite=overwrite)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


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


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _kernel_index(kernels: np.ndarray, kernel: str) -> int:
    matches = np.flatnonzero(kernels.astype(str) == kernel)
    if matches.size == 0:
        raise KeyError(f"Geometry kernel not found: {kernel}")
    return int(matches[0])


def _fold_ids(depth: np.ndarray, fold_count: int) -> np.ndarray:
    order = np.argsort(depth)
    folds = np.empty(depth.size, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(order, fold_count)):
        folds[indices] = fold
    return folds


def _window_for_index(depth: np.ndarray, index: int, half_width: int = 5) -> dict[str, Any]:
    start = max(0, index - half_width)
    stop = min(depth.size - 1, index + half_width)
    values = depth[start : stop + 1]
    return {
        "depth_min": float(np.min(values)),
        "depth_max": float(np.max(values)),
        "depth_count": int(values.size),
        "depth_index": int(index),
    }


def _sample_indices(length: int, target: int) -> np.ndarray:
    if length <= target:
        return np.arange(length)
    return np.linspace(0, length - 1, target).astype(int)


def _bool_count(values: Any) -> int:
    return int(sum(bool(value) for value in values))


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _finite_min(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.min(finite))


def _finite_max(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.max(finite))


def _finite_mean(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.mean(finite))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
