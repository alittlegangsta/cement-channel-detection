from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.visualization.matplotlib_utils import require_pyplot

CAST_HORIZONTAL_STRIPE_QC_VERSION = "cast_horizontal_stripe_qc_v001"
DEFAULT_ZC_THRESHOLD_MRAYL = 2.5
TARGET_NAMES = (
    "receiver_mean_v1_reference",
    "receiver_p90_robust_candidate",
    "local_worst_sector_fraction",
    "connected_channel_fraction",
    "interval_persistence_weighted_fraction",
)
RECOMPUTABLE_TARGETS = (
    "receiver_mean_v1_reference",
    "receiver_p90_robust_candidate",
    "local_worst_sector_fraction",
)
OPTIONAL_CAST_METADATA_FIELDS = (
    "thickness",
    "Thickness",
    "casing_thickness",
    "CasingThickness",
    "ID",
    "id",
    "OD",
    "od",
    "radius",
    "Radius",
    "quality",
    "Quality",
    "quality_flag",
    "QualityFlag",
    "collar",
    "Collar",
    "collar_mask",
    "depth_control",
    "DepthControl",
)
METHOD_FLAGS: dict[str, bool] = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
    "no_raw_mat_modified": True,
    "no_formal_mask_applied": True,
    "no_threshold_change": True,
    "no_label_formula_change": True,
    "no_geometry_sign_change": True,
    "no_relbearing_policy_change": True,
    "no_model_training": True,
    "no_stc": True,
    "no_apes": True,
    "no_deep_learning": True,
}
CHECKLIST_MARKER = "qc_flag_horizontal_stripe"


@dataclass(frozen=True)
class HorizontalStripeQcConfig:
    local_window_rows: int = 51
    exclude_center_rows: int = 3
    robust_z_threshold: float = 6.0
    near360_coverage_threshold: float = 0.75
    partial_coverage_min: float = 0.20
    minimum_high_offset_mrayl: float = 0.35
    target_kernel: str = "triangular_midpoint_weighted"
    practical_delta_threshold: float = 0.001
    zc_threshold_mrayl: float = DEFAULT_ZC_THRESHOLD_MRAYL


@dataclass(frozen=True)
class HorizontalStripeQcResult:
    qc_version: str
    generated_at: str
    inputs: dict[str, str]
    raw_zc_shape: list[int]
    zc_threshold_mrayl: float
    detection_config: dict[str, Any]
    raw_mat_check: dict[str, Any]
    optional_metadata_check: dict[str, Any]
    classification_summary: dict[str, Any]
    stripe_event_count: int
    near360_event_count: int
    partial_event_count: int
    one_row_stripe_count: int
    multi_row_band_count: int
    target_sensitivity_summary: dict[str, Any]
    recommendations: list[str]
    warnings: list[str]
    errors: list[str]
    output_files: dict[str, str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), **METHOD_FLAGS}


def audit_cast_horizontal_stripes_from_paths(
    *,
    cast_npz: Path | str,
    regression_labels_npz: Path | str,
    parallel_candidates_npz: Path | str,
    raw_cast_mat: Path | str | None,
    reports_dir: Path | str,
    manual_review_checklist: Path | str | None = None,
    overwrite: bool = False,
    config: HorizontalStripeQcConfig | None = None,
) -> HorizontalStripeQcResult:
    cast_arrays = _load_npz(Path(cast_npz))
    regression_labels = _load_npz(Path(regression_labels_npz))
    candidates = _load_npz(Path(parallel_candidates_npz))
    outputs = _output_paths(Path(reports_dir))
    for path in outputs.values():
        _ensure_can_write(path, overwrite=overwrite)
    result, inventory, sensitivity = audit_cast_horizontal_stripes(
        cast_arrays=cast_arrays,
        regression_labels=regression_labels,
        candidates=candidates,
        raw_cast_mat=Path(raw_cast_mat) if raw_cast_mat else None,
        outputs=outputs,
        config=config,
        inputs={
            "cast_npz": str(cast_npz),
            "regression_labels_npz": str(regression_labels_npz),
            "parallel_candidates_npz": str(parallel_candidates_npz),
            "raw_cast_mat": "" if raw_cast_mat is None else str(raw_cast_mat),
        },
    )
    write_cast_horizontal_stripe_outputs(
        result=result,
        inventory=inventory,
        sensitivity=sensitivity,
        outputs=outputs,
        manual_review_checklist=Path(manual_review_checklist)
        if manual_review_checklist is not None
        else None,
        overwrite=overwrite,
    )
    write_cast_horizontal_stripe_example_heatmaps(
        cast_arrays=cast_arrays,
        inventory=inventory,
        review_dir=outputs["review_dir"],
        overwrite=overwrite,
    )
    return result


def audit_cast_horizontal_stripes(
    *,
    cast_arrays: dict[str, np.ndarray],
    regression_labels: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    raw_cast_mat: Path | None,
    outputs: dict[str, Path] | None = None,
    config: HorizontalStripeQcConfig | None = None,
    inputs: dict[str, str] | None = None,
) -> tuple[HorizontalStripeQcResult, list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = config or HorizontalStripeQcConfig()
    warnings: list[str] = []
    errors: list[str] = []
    cast_depth = _required_1d(cast_arrays, "cast_depth")
    cast_zc = _required_2d(cast_arrays, "cast_zc")
    if cast_zc.shape[0] != cast_depth.size:
        raise ValueError("cast_zc first dimension must match cast_depth.")
    if cfg.zc_threshold_mrayl != DEFAULT_ZC_THRESHOLD_MRAYL:
        raise ValueError("Horizontal stripe QC must not change the 2.5 MRayl threshold.")

    row_stats = compute_horizontal_stripe_row_stats(cast_zc, config=cfg)
    inventory = build_horizontal_stripe_inventory(cast_depth, cast_zc, row_stats, config=cfg)
    raw_check = check_raw_cast_mat_for_stripes(
        raw_cast_mat=raw_cast_mat,
        cast_depth=cast_depth,
        cast_zc=cast_zc,
        stripe_rows=_near360_rows_from_inventory(inventory),
    )
    metadata_check = raw_check["optional_metadata_check"]
    warnings.extend(raw_check["warnings"])
    sensitivity = target_sensitivity_rows(
        stripe_rows=_near360_rows_from_inventory(inventory),
        cast_depth=cast_depth,
        cast_zc=cast_zc,
        regression_labels=regression_labels,
        candidates=candidates,
        config=cfg,
    )
    sensitivity_summary = _summarize_sensitivity(sensitivity)
    classification = _classification_summary(row_stats, inventory)
    near360_event_count = sum(row["event_family"] == "near_360_high_zc" for row in inventory)
    partial_event_count = sum(row["event_family"] == "partial_azimuth_high_zc" for row in inventory)
    one_row_count = sum(row["event_class"] == "one-row stripe" for row in inventory)
    multi_row_count = sum(row["event_class"] == "multi-row band" for row in inventory)
    recommendations = _recommendations(
        near360_event_count=near360_event_count,
        metadata_available=bool(metadata_check["available_fields"]),
        sensitivity_summary=sensitivity_summary,
    )
    if not metadata_check["available_fields"]:
        warnings.append(
            "No casing thickness, ID, OD, radius, quality flag, collar, or depth-control "
            "fields were found in raw CAST MAT; casing-collar synchronization was not inferred."
        )

    result = HorizontalStripeQcResult(
        qc_version=CAST_HORIZONTAL_STRIPE_QC_VERSION,
        generated_at=_utc_now(),
        inputs=inputs or {},
        raw_zc_shape=[int(cast_zc.shape[0]), int(cast_zc.shape[1])],
        zc_threshold_mrayl=cfg.zc_threshold_mrayl,
        detection_config=asdict(cfg),
        raw_mat_check={key: value for key, value in raw_check.items() if key != "warnings"},
        optional_metadata_check=metadata_check,
        classification_summary=classification,
        stripe_event_count=len(inventory),
        near360_event_count=int(near360_event_count),
        partial_event_count=int(partial_event_count),
        one_row_stripe_count=int(one_row_count),
        multi_row_band_count=int(multi_row_count),
        target_sensitivity_summary=sensitivity_summary,
        recommendations=recommendations,
        warnings=warnings,
        errors=errors,
        output_files={key: str(value) for key, value in (outputs or {}).items()},
        not_performed=[
            "raw MAT modification",
            "automatic CAST masking",
            "2.5 MRayl threshold change",
            "geometry sign change",
            "RelBearing policy change",
            "label formula change",
            "model training",
            "full-well STC/APES",
            "deep learning",
            "final label generation",
            "production claim",
        ],
    )
    return result, inventory, sensitivity


def compute_horizontal_stripe_row_stats(
    cast_zc: np.ndarray,
    *,
    config: HorizontalStripeQcConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or HorizontalStripeQcConfig()
    zc = np.asarray(cast_zc, dtype=np.float32)
    if zc.ndim != 2:
        raise ValueError("cast_zc must be a 2D [depth, azimuth] array.")
    row_median = np.nanmedian(zc, axis=1).astype(np.float32)
    row_p90 = np.nanpercentile(zc, 90.0, axis=1).astype(np.float32)
    row_p95 = np.nanpercentile(zc, 95.0, axis=1).astype(np.float32)
    row_max = np.nanmax(zc, axis=1).astype(np.float32)
    finite_fraction = np.mean(np.isfinite(zc), axis=1).astype(np.float32)
    local_median = np.zeros(row_median.size, dtype=np.float32)
    local_mad = np.zeros(row_median.size, dtype=np.float32)
    high_threshold = np.zeros(row_median.size, dtype=np.float32)
    high_coverage = np.zeros(row_median.size, dtype=np.float32)
    robust_z = np.zeros(row_median.size, dtype=np.float32)
    p95_robust_z = np.zeros(row_median.size, dtype=np.float32)
    for index in range(row_median.size):
        values = _local_values(row_median, index, cfg.local_window_rows, cfg.exclude_center_rows)
        p95_values = _local_values(row_p95, index, cfg.local_window_rows, cfg.exclude_center_rows)
        center = float(np.nanmedian(values))
        scale = float(1.4826 * np.nanmedian(np.abs(values - center)))
        p95_center = float(np.nanmedian(p95_values))
        p95_scale = float(1.4826 * np.nanmedian(np.abs(p95_values - p95_center)))
        threshold = center + max(3.0 * scale, cfg.minimum_high_offset_mrayl)
        local_median[index] = center
        local_mad[index] = scale
        high_threshold[index] = threshold
        robust_z[index] = (float(row_median[index]) - center) / (scale + 0.05)
        p95_robust_z[index] = (float(row_p95[index]) - p95_center) / (p95_scale + 0.05)
        high_coverage[index] = float(np.mean(zc[index] >= threshold))
    near360 = (robust_z >= cfg.robust_z_threshold) & (
        high_coverage >= cfg.near360_coverage_threshold
    )
    partial = (
        (p95_robust_z >= cfg.robust_z_threshold)
        & (high_coverage >= cfg.partial_coverage_min)
        & (high_coverage < cfg.near360_coverage_threshold)
    )
    ordinary = ~(near360 | partial)
    saturation_reference = float(np.nanpercentile(zc[np.isfinite(zc)], 99.9))
    saturation_fraction = np.mean(zc >= saturation_reference, axis=1).astype(np.float32)
    adjacent_contrast = _adjacent_row_contrast(row_median)
    return {
        "row_median_zc": row_median,
        "row_p90_zc": row_p90,
        "row_p95_zc": row_p95,
        "row_max_zc": row_max,
        "finite_fraction": finite_fraction,
        "local_background_median_zc": local_median,
        "local_background_mad_zc": local_mad,
        "high_cell_threshold_zc": high_threshold,
        "row_median_robust_z": robust_z,
        "row_p95_robust_z": p95_robust_z,
        "azimuth_high_coverage": high_coverage,
        "saturation_fraction": saturation_fraction,
        "saturation_reference_zc": np.full(row_median.size, saturation_reference, dtype=np.float32),
        "adjacent_row_contrast": adjacent_contrast,
        "near360_row": near360.astype(bool),
        "partial_row": partial.astype(bool),
        "ordinary_row": ordinary.astype(bool),
    }


def build_horizontal_stripe_inventory(
    cast_depth: np.ndarray,
    cast_zc: np.ndarray,
    row_stats: dict[str, np.ndarray],
    *,
    config: HorizontalStripeQcConfig | None = None,
) -> list[dict[str, Any]]:
    depth = np.asarray(cast_depth, dtype=np.float32).reshape(-1)
    zc = np.asarray(cast_zc, dtype=np.float32)
    row_step = _median_step(depth)
    events: list[dict[str, Any]] = []
    for family, mask in (
        ("near_360_high_zc", np.asarray(row_stats["near360_row"], dtype=bool)),
        ("partial_azimuth_high_zc", np.asarray(row_stats["partial_row"], dtype=bool)),
    ):
        for start, stop in _contiguous_groups(np.flatnonzero(mask)):
            rows = np.arange(start, stop + 1, dtype=np.int64)
            row_count = int(rows.size)
            if family == "near_360_high_zc":
                event_class = "one-row stripe" if row_count == 1 else "multi-row band"
            else:
                event_class = "partial-azimuth high-Zc event"
            center_depth = float(np.median(depth[rows]))
            event = {
                "event_id": "",
                "event_family": family,
                "event_class": event_class,
                "stripe_depth_ft": center_depth,
                "depth_min_ft": float(np.min(depth[rows])),
                "depth_max_ft": float(np.max(depth[rows])),
                "row_index_start": int(start),
                "row_index_end": int(stop),
                "row_count": row_count,
                "row_thickness_ft": float(
                    max(row_step, abs(np.max(depth[rows]) - np.min(depth[rows])) + row_step)
                ),
                "azimuth_coverage": _stat(row_stats["azimuth_high_coverage"], rows, np.mean),
                "max_azimuth_coverage": _stat(row_stats["azimuth_high_coverage"], rows, np.max),
                "row_median_zc": _stat(row_stats["row_median_zc"], rows, np.median),
                "row_p90_zc": _stat(row_stats["row_p90_zc"], rows, np.median),
                "row_p95_zc": _stat(row_stats["row_p95_zc"], rows, np.median),
                "row_max_zc": _stat(row_stats["row_max_zc"], rows, np.max),
                "local_background_median_zc": _stat(
                    row_stats["local_background_median_zc"], rows, np.median
                ),
                "row_median_robust_z": _stat(row_stats["row_median_robust_z"], rows, np.max),
                "row_p95_robust_z": _stat(row_stats["row_p95_robust_z"], rows, np.max),
                "adjacent_row_contrast": _group_adjacent_contrast(
                    row_stats["row_median_zc"], start, stop
                ),
                "saturation_fraction": _stat(row_stats["saturation_fraction"], rows, np.mean),
                "saturation_reference_zc": float(row_stats["saturation_reference_zc"][0]),
                "high_cell_threshold_source": (
                    "local row-median background + max(3*MAD, minimum_high_offset_mrayl); "
                    "QC-only, not a label threshold"
                ),
                "raw_zc_cell_min": float(np.nanmin(zc[rows])),
                "raw_zc_cell_max": float(np.nanmax(zc[rows])),
                **METHOD_FLAGS,
            }
            events.append(event)
    events = sorted(events, key=lambda row: (float(row["stripe_depth_ft"]), row["row_index_start"]))
    previous_depth: float | None = None
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"cast_hstripe_{index:04d}"
        if previous_depth is None:
            event["inter_stripe_spacing_ft"] = None
        else:
            event["inter_stripe_spacing_ft"] = abs(float(event["stripe_depth_ft"]) - previous_depth)
        previous_depth = float(event["stripe_depth_ft"])
    return events


def check_raw_cast_mat_for_stripes(
    *,
    raw_cast_mat: Path | None,
    cast_depth: np.ndarray,
    cast_zc: np.ndarray,
    stripe_rows: np.ndarray,
) -> dict[str, Any]:
    warnings: list[str] = []
    optional = {
        "available_fields": [],
        "missing_expected_fields": list(OPTIONAL_CAST_METADATA_FIELDS),
        "synchronization_checked": False,
        "synchronization_results": [],
        "warning": "",
    }
    if raw_cast_mat is None:
        warnings.append("No raw CAST MAT path was provided; raw-MAT stripe check skipped.")
        optional["warning"] = "No raw CAST MAT path was provided."
        return {
            "raw_cast_mat": None,
            "raw_mat_checked": False,
            "raw_mat_stripes_present": None,
            "controlled_npz_matches_raw_mat": None,
            "max_abs_depth_diff": None,
            "max_abs_zc_diff": None,
            "cast_struct_fields": [],
            "optional_metadata_check": optional,
            "warnings": warnings,
        }
    if not raw_cast_mat.exists():
        warnings.append(f"Raw CAST MAT does not exist: {raw_cast_mat}")
        optional["warning"] = "Raw CAST MAT does not exist."
        return {
            "raw_cast_mat": str(raw_cast_mat),
            "raw_mat_checked": False,
            "raw_mat_stripes_present": None,
            "controlled_npz_matches_raw_mat": None,
            "max_abs_depth_diff": None,
            "max_abs_zc_diff": None,
            "cast_struct_fields": [],
            "optional_metadata_check": optional,
            "warnings": warnings,
        }
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError:
        warnings.append("scipy is unavailable; raw CAST MAT check skipped.")
        optional["warning"] = "scipy unavailable."
        return {
            "raw_cast_mat": str(raw_cast_mat),
            "raw_mat_checked": False,
            "raw_mat_stripes_present": None,
            "controlled_npz_matches_raw_mat": None,
            "max_abs_depth_diff": None,
            "max_abs_zc_diff": None,
            "cast_struct_fields": [],
            "optional_metadata_check": optional,
            "warnings": warnings,
        }
    mat = loadmat(raw_cast_mat, variable_names=["CAST"], squeeze_me=True, struct_as_record=False)
    cast = mat.get("CAST")
    fields = list(getattr(cast, "_fieldnames", []) or [])
    depth = np.asarray(cast.Depth, dtype=np.float32).reshape(-1)
    raw_zc = np.asarray(cast.Zc, dtype=np.float32)
    if raw_zc.shape == cast_zc.T.shape:
        raw_zc = raw_zc.T
    depth_diff = float(np.max(np.abs(depth - np.asarray(cast_depth, dtype=np.float32))))
    zc_diff = float(np.nanmax(np.abs(raw_zc - np.asarray(cast_zc, dtype=np.float32))))
    available = [field for field in fields if field in OPTIONAL_CAST_METADATA_FIELDS]
    optional["available_fields"] = available
    optional["missing_expected_fields"] = [
        field for field in OPTIONAL_CAST_METADATA_FIELDS if field not in available
    ]
    if not available:
        optional["warning"] = (
            "Raw CAST MAT contains only fields needed for Zc/depth or lacks casing metadata; "
            "collar/quality synchronization cannot be checked."
        )
    return {
        "raw_cast_mat": str(raw_cast_mat),
        "raw_mat_checked": True,
        "raw_mat_stripes_present": bool(stripe_rows.size > 0 and zc_diff == 0.0),
        "controlled_npz_matches_raw_mat": bool(depth_diff == 0.0 and zc_diff == 0.0),
        "max_abs_depth_diff": depth_diff,
        "max_abs_zc_diff": zc_diff,
        "cast_struct_fields": fields,
        "optional_metadata_check": optional,
        "warnings": warnings,
    }


def target_sensitivity_rows(
    *,
    stripe_rows: np.ndarray,
    cast_depth: np.ndarray,
    cast_zc: np.ndarray,
    regression_labels: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    config: HorizontalStripeQcConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or HorizontalStripeQcConfig()
    kernel_names = np.asarray(regression_labels["geometry_kernel"]).astype(str)
    kernel_matches = np.flatnonzero(kernel_names == cfg.target_kernel)
    if not kernel_matches.size:
        raise ValueError(f"Target kernel not found in regression labels: {cfg.target_kernel}")
    kernel_index = int(kernel_matches[0])
    original_target = np.asarray(
        regression_labels["weighted_channel_fraction_zc_lt_2p5"], dtype=np.float32
    )[kernel_index]
    interval_min = np.asarray(regression_labels["interval_min_depth"], dtype=np.float32)[
        kernel_index
    ]
    interval_max = np.asarray(regression_labels["interval_max_depth"], dtype=np.float32)[
        kernel_index
    ]
    midpoint = np.asarray(regression_labels["midpoint_depth"], dtype=np.float32)[kernel_index]
    base_views = _aggregate_receiver_views(original_target)
    stripe_mask = np.zeros(cast_depth.size, dtype=bool)
    stripe_mask[np.asarray(stripe_rows, dtype=np.int64)] = True
    expanded_mask = stripe_mask.copy()
    for row in np.flatnonzero(stripe_mask):
        expanded_mask[max(0, row - 1) : min(expanded_mask.size, row + 2)] = True
    exclusion_masks = {
        "keep_all_rows": np.zeros(cast_depth.size, dtype=bool),
        "exclude_stripe_rows": stripe_mask,
        "exclude_stripe_rows_pm1": expanded_mask,
    }
    recomputed: dict[str, dict[str, np.ndarray]] = {}
    for name, mask in exclusion_masks.items():
        target = _counterfactual_fraction_target(
            cast_depth=cast_depth,
            cast_zc=cast_zc,
            exclude_rows=mask,
            interval_min=interval_min,
            interval_max=interval_max,
            midpoint=midpoint,
            zc_threshold=cfg.zc_threshold_mrayl,
        )
        recomputed[name] = _aggregate_receiver_views(target)

    overlap = _label_depth_overlap_mask(
        stripe_depths=np.asarray(cast_depth, dtype=np.float32)[stripe_mask],
        interval_min=interval_min,
        interval_max=interval_max,
    )
    rows: list[dict[str, Any]] = []
    for target_name in TARGET_NAMES:
        keep_values = np.asarray(candidates[target_name], dtype=np.float32)
        if target_name in RECOMPUTABLE_TARGETS:
            for scenario in ("keep_all_rows", "exclude_stripe_rows", "exclude_stripe_rows_pm1"):
                values = recomputed[scenario][target_name]
                base = base_views[target_name]
                delta = values - base
                rows.append(
                    {
                        "target": target_name,
                        "scenario": scenario,
                        "recompute_status": "recomputed_audit_only",
                        "sample_count": int(values.size),
                        "overlap_sample_count": int(np.count_nonzero(overlap)),
                        "mean_value": _finite_mean(values),
                        "mean_abs_delta_vs_keep_all": _finite_mean(np.abs(delta)),
                        "max_abs_delta_vs_keep_all": _finite_max(np.abs(delta)),
                        "practical_delta_sample_count": int(
                            np.count_nonzero(np.abs(delta) >= cfg.practical_delta_threshold)
                        ),
                        "practical_delta_fraction": _fraction(
                            np.abs(delta) >= cfg.practical_delta_threshold
                        ),
                        "mean_abs_delta_on_overlap": _finite_mean(np.abs(delta[overlap])),
                        "keep_all_candidate_mean": _finite_mean(keep_values),
                        **METHOD_FLAGS,
                    }
                )
        else:
            rows.append(
                {
                    "target": target_name,
                    "scenario": "keep_all_rows",
                    "recompute_status": "skipped_missing_recompute_contract",
                    "sample_count": int(keep_values.size),
                    "overlap_sample_count": int(np.count_nonzero(overlap)),
                    "mean_value": _finite_mean(keep_values),
                    "mean_abs_delta_vs_keep_all": None,
                    "max_abs_delta_vs_keep_all": None,
                    "practical_delta_sample_count": None,
                    "practical_delta_fraction": None,
                    "mean_abs_delta_on_overlap": None,
                    "warning": (
                        "Target depends on aggregate morphology/component or persistence arrays; "
                        "no saved 2D map/recompute contract is available."
                    ),
                    **METHOD_FLAGS,
                }
            )
    return rows


def write_cast_horizontal_stripe_outputs(
    *,
    result: HorizontalStripeQcResult,
    inventory: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    outputs: dict[str, Path],
    manual_review_checklist: Path | None,
    overwrite: bool,
) -> None:
    for path in outputs.values():
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(inventory, outputs["inventory_csv"])
    _write_json(outputs["inventory_json"], {"inventory": inventory, **METHOD_FLAGS})
    _write_csv(sensitivity, outputs["sensitivity_csv"])
    _write_json(outputs["report_json"], result.to_dict())
    outputs["report_md"].write_text(format_horizontal_stripe_qc_markdown(result), encoding="utf-8")
    _write_review_figures(
        inventory=inventory,
        sensitivity=sensitivity,
        report=result,
        review_dir=outputs["review_dir"],
        overwrite=overwrite,
    )
    if manual_review_checklist is not None:
        update_manual_review_checklist(manual_review_checklist)


def write_cast_horizontal_stripe_example_heatmaps(
    *,
    cast_arrays: dict[str, np.ndarray],
    inventory: list[dict[str, Any]],
    review_dir: Path,
    overwrite: bool,
) -> list[Path]:
    if not inventory:
        return []
    cast_depth = _required_1d(cast_arrays, "cast_depth")
    cast_zc = _required_2d(cast_arrays, "cast_zc")
    cast_azimuth = _azimuth_axis(cast_arrays, cast_zc.shape[1])
    events = _selected_example_events(inventory)
    plt = require_pyplot()
    written: list[Path] = []
    for event in events:
        path = review_dir / f"example_{event['event_id']}_heatmap.png"
        _ensure_can_write(path, overwrite=overwrite)
        _write_event_heatmap(
            plt=plt,
            event=event,
            cast_depth=cast_depth,
            cast_azimuth=cast_azimuth,
            cast_zc=cast_zc,
            path=path,
        )
        written.append(path)
    return written


def format_horizontal_stripe_qc_markdown(result: HorizontalStripeQcResult) -> str:
    lines = [
        "# CAST Horizontal Stripe QC Audit",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
        "This is a QC-only audit. It does not mask data, change the 2.5 MRayl threshold, "
        "change geometry/RelBearing policy, change label formulas, train models, or make "
        "production claims.",
        "",
        f"- qc_version: `{result.qc_version}`",
        f"- raw_zc_shape: `{result.raw_zc_shape}`",
        f"- zc_threshold_mrayl_preserved: `{result.zc_threshold_mrayl}`",
        f"- stripe_event_count: `{result.stripe_event_count}`",
        f"- near360_event_count: `{result.near360_event_count}`",
        f"- partial_event_count: `{result.partial_event_count}`",
        f"- one_row_stripe_count: `{result.one_row_stripe_count}`",
        f"- multi_row_band_count: `{result.multi_row_band_count}`",
        "",
        "## Classification Summary",
        "",
    ]
    lines.extend(_dict_lines(result.classification_summary))
    lines.extend(["", "## Raw MAT Check", ""])
    lines.extend(_dict_lines(result.raw_mat_check))
    lines.extend(["", "## Optional Metadata Check", ""])
    lines.extend(_dict_lines(result.optional_metadata_check))
    lines.extend(["", "## Target Sensitivity Summary", ""])
    lines.extend(_dict_lines(result.target_sensitivity_summary))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(_message_lines(result.recommendations))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(result.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(result.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(result.not_performed))
    lines.append("")
    return "\n".join(lines)


def update_manual_review_checklist(path: Path) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else "# Reviewer Checklist\n"
    section = "\n".join(
        [
            "## CAST Horizontal Stripe QC",
            "",
            f"- {CHECKLIST_MARKER}: check whether a selected interval overlaps a near-360-degree "
            "single-row or multi-row high-Zc horizontal stripe.",
            "- Treat this as QC-only evidence; do not mask CAST, change thresholds, or approve "
            "final labels from this flag alone.",
            "- If a stripe aligns with future casing collar/depth-control metadata, request "
            "processing review before using the affected target rows for claims.",
            "",
        ]
    )
    path.write_text(
        _replace_section(text, "## CAST Horizontal Stripe QC", section),
        encoding="utf-8",
    )


def _counterfactual_fraction_target(
    *,
    cast_depth: np.ndarray,
    cast_zc: np.ndarray,
    exclude_rows: np.ndarray,
    interval_min: np.ndarray,
    interval_max: np.ndarray,
    midpoint: np.ndarray,
    zc_threshold: float,
) -> np.ndarray:
    finite = np.isfinite(cast_zc) & ~exclude_rows[:, None]
    candidate = finite & (cast_zc < zc_threshold)
    finite_count = np.count_nonzero(finite, axis=1).astype(np.float64)
    candidate_count = np.count_nonzero(candidate, axis=1).astype(np.float64)
    output = np.zeros(interval_min.shape, dtype=np.float32)
    for index in np.ndindex(interval_min.shape):
        lower = float(interval_min[index])
        upper = float(interval_max[index])
        middle = float(midpoint[index])
        start, stop = _depth_region_bounds(cast_depth, lower, upper)
        if stop <= start:
            output[index] = 0.0
            continue
        weights = _depth_weights(cast_depth[start:stop], lower=lower, upper=upper, midpoint=middle)
        denominator = float(np.sum(weights * finite_count[start:stop]))
        if denominator <= 0.0:
            output[index] = 0.0
        else:
            output[index] = float(np.sum(weights * candidate_count[start:stop]) / denominator)
    return output


def _aggregate_receiver_views(target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "receiver_mean_v1_reference": np.nanmean(target, axis=1).astype(np.float32),
        "receiver_p90_robust_candidate": np.nanpercentile(target, 90.0, axis=1).astype(np.float32),
        "local_worst_sector_fraction": np.nanmax(target, axis=1).astype(np.float32),
    }


def _label_depth_overlap_mask(
    *,
    stripe_depths: np.ndarray,
    interval_min: np.ndarray,
    interval_max: np.ndarray,
) -> np.ndarray:
    if stripe_depths.size == 0:
        return np.zeros(interval_min.shape[0], dtype=bool)
    low = np.minimum(interval_min, interval_max)
    high = np.maximum(interval_min, interval_max)
    overlap = np.zeros(interval_min.shape, dtype=bool)
    for depth in stripe_depths.astype(np.float32):
        overlap |= (low <= depth) & (depth <= high)
    return np.any(overlap, axis=1)


def _output_paths(reports_dir: Path) -> dict[str, Path]:
    review_dir = reports_dir / "cast_horizontal_stripe_review_v001"
    return {
        "inventory_csv": reports_dir / "cast_horizontal_stripe_inventory_v001.csv",
        "inventory_json": reports_dir / "cast_horizontal_stripe_inventory_v001.json",
        "report_md": reports_dir / "cast_horizontal_stripe_qc_report_v001.md",
        "report_json": reports_dir / "cast_horizontal_stripe_qc_report_v001.json",
        "sensitivity_csv": reports_dir / "cast_horizontal_stripe_target_sensitivity_v001.csv",
        "review_dir": review_dir,
    }


def _write_review_figures(
    *,
    inventory: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    report: HorizontalStripeQcResult,
    review_dir: Path,
    overwrite: bool,
) -> None:
    _ensure_review_dir(review_dir, overwrite=overwrite)
    plt = require_pyplot()
    _write_event_depth_plot(plt, inventory, review_dir / "stripe_depth_inventory.png")
    _write_spacing_plot(plt, inventory, review_dir / "inter_stripe_spacing.png")
    _write_sensitivity_plot(plt, sensitivity, review_dir / "target_sensitivity_summary.png")
    (review_dir / "review_summary.md").write_text(
        "\n".join(
            [
                "# CAST Horizontal Stripe Review Pack",
                "",
                "Scope: QC-only review of near-360-degree high-Zc horizontal stripes.",
                "",
                f"- near360_event_count: `{report.near360_event_count}`",
                f"- partial_event_count: `{report.partial_event_count}`",
                f"- recommendations: `{', '.join(report.recommendations)}`",
                "- example heatmaps are generated for suspicious depths near 2412, 4032, "
                "5465 ft when matching events are present, plus high-score events.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _selected_example_events(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target_depth in (2412.0, 4032.0, 5465.0):
        nearest = min(
            inventory,
            key=lambda row: abs(float(row["stripe_depth_ft"]) - target_depth),
        )
        selected[str(nearest["event_id"])] = nearest
    top = sorted(
        inventory,
        key=lambda row: max(
            float(row.get("row_median_robust_z") or 0.0),
            float(row.get("row_p95_robust_z") or 0.0),
        ),
        reverse=True,
    )
    for row in top:
        selected[str(row["event_id"])] = row
        if len(selected) >= 8:
            break
    return list(selected.values())


def _write_event_heatmap(
    *,
    plt: Any,
    event: dict[str, Any],
    cast_depth: np.ndarray,
    cast_azimuth: np.ndarray,
    cast_zc: np.ndarray,
    path: Path,
) -> None:
    start = int(event["row_index_start"])
    stop = int(event["row_index_end"])
    pad = 20
    rows = np.arange(max(0, start - pad), min(cast_depth.size, stop + pad + 1), dtype=np.int64)
    values = cast_zc[rows]
    depth = cast_depth[rows]
    finite = values[np.isfinite(values)]
    vmin, vmax = (0.0, 1.0)
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            float(np.min(cast_azimuth)),
            float(np.max(cast_azimuth)),
            float(np.min(depth)),
            float(np.max(depth)),
        ),
        vmin=float(vmin),
        vmax=float(vmax),
        cmap="viridis",
    )
    ax.axhspan(
        float(event["depth_min_ft"]),
        float(event["depth_max_ft"]),
        color="cyan",
        alpha=0.22,
        label="detected stripe rows",
    )
    ax.set_title(
        f"{event['event_id']} {event['event_class']} at {float(event['stripe_depth_ft']):.2f} ft"
    )
    ax.set_xlabel("CAST azimuth degree")
    ax.set_ylabel("physical depth ft")
    ax.legend(fontsize=8)
    fig.colorbar(image, ax=ax, label="Zc MRayl")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_event_depth_plot(plt: Any, inventory: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    near = [row for row in inventory if row["event_family"] == "near_360_high_zc"]
    partial = [row for row in inventory if row["event_family"] == "partial_azimuth_high_zc"]
    if near:
        ax.scatter(
            [row["stripe_depth_ft"] for row in near],
            [row["row_median_robust_z"] for row in near],
            s=18,
            label="near-360 high-Zc",
        )
    if partial:
        ax.scatter(
            [row["stripe_depth_ft"] for row in partial],
            [row["row_median_robust_z"] for row in partial],
            s=18,
            label="partial azimuth",
        )
    ax.set_xlabel("physical depth ft")
    ax.set_ylabel("row median robust z")
    ax.set_title("CAST horizontal high-Zc stripe inventory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_spacing_plot(plt: Any, inventory: list[dict[str, Any]], path: Path) -> None:
    values = [
        float(row["inter_stripe_spacing_ft"])
        for row in inventory
        if row.get("inter_stripe_spacing_ft") is not None
        and row["event_family"] == "near_360_high_zc"
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    if values:
        ax.hist(values, bins=min(30, max(5, len(values) // 3)))
    ax.set_xlabel("inter-stripe spacing ft")
    ax.set_ylabel("event count")
    ax.set_title("Near-360 high-Zc stripe spacing")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_sensitivity_plot(plt: Any, sensitivity: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in sensitivity
        if row.get("scenario") == "exclude_stripe_rows_pm1"
        and row.get("recompute_status") == "recomputed_audit_only"
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    if rows:
        ax.bar(
            [str(row["target"]) for row in rows],
            [float(row["max_abs_delta_vs_keep_all"]) for row in rows],
        )
    ax.set_ylabel("max abs delta vs keep-all")
    ax.set_title("Audit-only stripe row +/-1 target sensitivity")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ensure_review_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Review output path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        for name in (
            "stripe_depth_inventory.png",
            "inter_stripe_spacing.png",
            "target_sensitivity_summary.png",
            "review_summary.md",
        ):
            _ensure_can_write(path / name, overwrite=False)


def _classification_summary(
    row_stats: dict[str, np.ndarray],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    near = np.asarray(row_stats["near360_row"], dtype=bool)
    partial = np.asarray(row_stats["partial_row"], dtype=bool)
    ordinary = np.asarray(row_stats["ordinary_row"], dtype=bool)
    return {
        "row_count": int(near.size),
        "near360_high_zc_row_count": int(np.count_nonzero(near)),
        "partial_azimuth_high_zc_row_count": int(np.count_nonzero(partial)),
        "ordinary_background_variation_row_count": int(np.count_nonzero(ordinary)),
        "near360_event_count": int(
            sum(row["event_family"] == "near_360_high_zc" for row in inventory)
        ),
        "partial_event_count": int(
            sum(row["event_family"] == "partial_azimuth_high_zc" for row in inventory)
        ),
        "ordinary_background_rule": (
            "rows not meeting robust-z plus azimuth-coverage QC criteria are ordinary "
            "background variation for this audit"
        ),
    }


def _summarize_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"targets": {}}
    for target in TARGET_NAMES:
        target_rows = [row for row in rows if row["target"] == target]
        summary["targets"][target] = {
            "rows": target_rows,
            "recompute_status": target_rows[0].get("recompute_status") if target_rows else None,
        }
    recomputed_pm1 = [
        row
        for row in rows
        if row.get("scenario") == "exclude_stripe_rows_pm1"
        and row.get("recompute_status") == "recomputed_audit_only"
    ]
    summary["max_abs_delta_any_recomputed_pm1"] = (
        max(float(row["max_abs_delta_vs_keep_all"]) for row in recomputed_pm1)
        if recomputed_pm1
        else None
    )
    summary["recompute_missing_targets"] = [
        row["target"]
        for row in rows
        if row.get("recompute_status") == "skipped_missing_recompute_contract"
    ]
    return summary


def _recommendations(
    *,
    near360_event_count: int,
    metadata_available: bool,
    sensitivity_summary: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if near360_event_count:
        recommendations.append("mark_as_qc_only")
        recommendations.append("request_processing_review")
    if not metadata_available:
        recommendations.append("request_casing_collar_metadata")
    max_delta = sensitivity_summary.get("max_abs_delta_any_recomputed_pm1")
    if max_delta is not None and float(max_delta) < 0.001 and near360_event_count:
        recommendations.append("retain_as_physical_pipe_event")
    return sorted(set(recommendations))


def _near360_rows_from_inventory(inventory: list[dict[str, Any]]) -> np.ndarray:
    rows: list[int] = []
    for event in inventory:
        if event["event_family"] != "near_360_high_zc":
            continue
        rows.extend(range(int(event["row_index_start"]), int(event["row_index_end"]) + 1))
    return np.asarray(sorted(set(rows)), dtype=np.int64)


def _local_values(values: np.ndarray, index: int, window: int, exclude: int) -> np.ndarray:
    start = max(0, index - window)
    stop = min(values.size, index + window + 1)
    mask = np.ones(stop - start, dtype=bool)
    mask[max(start, index - exclude) - start : min(stop, index + exclude + 1) - start] = False
    local = values[start:stop][mask]
    if local.size == 0:
        local = values[start:stop]
    return local[np.isfinite(local)]


def _adjacent_row_contrast(values: np.ndarray) -> np.ndarray:
    output = np.zeros(values.size, dtype=np.float32)
    for index in range(values.size):
        contrasts: list[float] = []
        if index > 0:
            contrasts.append(abs(float(values[index] - values[index - 1])))
        if index + 1 < values.size:
            contrasts.append(abs(float(values[index] - values[index + 1])))
        output[index] = max(contrasts) if contrasts else 0.0
    return output


def _group_adjacent_contrast(values: np.ndarray, start: int, stop: int) -> float:
    group_median = float(np.nanmedian(values[start : stop + 1]))
    neighbors: list[float] = []
    if start > 0:
        neighbors.append(float(values[start - 1]))
    if stop + 1 < values.size:
        neighbors.append(float(values[stop + 1]))
    if not neighbors:
        return 0.0
    return float(group_median - np.nanmedian(neighbors))


def _contiguous_groups(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        groups.append((start, previous))
        start = previous = current
    groups.append((start, previous))
    return groups


def _depth_region_bounds(depth: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    values = np.asarray(depth, dtype=np.float32).reshape(-1)
    lo = min(lower, upper)
    hi = max(lower, upper)
    mask = (values >= lo) & (values <= hi)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return 0, 0
    return int(rows.min()), int(rows.max()) + 1


def _depth_weights(
    depth: np.ndarray,
    *,
    lower: float,
    upper: float,
    midpoint: float,
) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    lo = min(lower, upper)
    hi = max(lower, upper)
    left = np.maximum(midpoint - lo, 1e-6)
    right = np.maximum(hi - midpoint, 1e-6)
    weights = np.where(
        values <= midpoint,
        (values - lo) / left,
        (hi - values) / right,
    )
    return np.clip(weights, 0.0, 1.0).astype(np.float64)


def _required_1d(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Required field missing: {key}")
    values = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Required field is empty: {key}")
    return values


def _required_2d(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Required field missing: {key}")
    values = np.asarray(data[key], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Required field must be 2-D: {key}")
    return values


def _azimuth_axis(data: dict[str, np.ndarray], width: int) -> np.ndarray:
    if "cast_azimuth_deg" in data:
        values = np.asarray(data["cast_azimuth_deg"], dtype=np.float32).reshape(-1)
        if values.size == width:
            return values
    return np.linspace(0.0, 360.0, num=width, endpoint=False, dtype=np.float32)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def _replace_section(existing: str, heading: str, section: str) -> str:
    lines = existing.rstrip().splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index
            break
    if start is None:
        return existing.rstrip() + "\n\n" + section
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join([*lines[:start], section.rstrip(), *lines[end:]]) + "\n"


def _dict_lines(data: dict[str, Any]) -> list[str]:
    return [f"- {key}: `{_short_value(value)}`" for key, value in data.items()]


def _message_lines(messages: list[str]) -> list[str]:
    return [f"- {message}" for message in messages] if messages else ["- none"]


def _short_value(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 500 else text[:497] + "..."


def _stat(values: np.ndarray, rows: np.ndarray, fn: Any) -> float:
    return float(fn(np.asarray(values)[rows]))


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.mean(finite))


def _finite_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.max(finite))


def _fraction(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool)
    return None if values.size == 0 else float(np.count_nonzero(values) / values.size)


def _median_step(depth: np.ndarray) -> float:
    diff = np.diff(np.sort(np.asarray(depth, dtype=np.float32).reshape(-1)))
    finite = np.abs(diff[np.isfinite(diff) & (diff != 0.0)])
    return float(np.median(finite)) if finite.size else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
