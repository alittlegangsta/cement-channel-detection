from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.labels.geometry_aware_regression_labels import (
    DEPTH_LEVEL_TARGET_VIEWS,
    TARGET_VIEW_DEFINITIONS,
)

CONTRACT_INVENTORY_VERSION = "geometry_regression_contract_inventory_v001"
CONTRACT_INVARIANTS_VERSION = "geometry_regression_contract_invariants_v001"

RECEIVER_LEVEL_TARGETS = (
    "raw_channel_fraction_zc_lt_2p5",
    "weighted_channel_fraction_zc_lt_2p5",
    "relative_anomaly_fraction",
    "combined_channel_fraction",
    "min_zc",
    "p05_zc",
    "p10_zc",
    "max_relative_drop",
    "candidate_cell_count",
    "total_cell_count",
    "largest_connected_component_fraction",
    "max_azimuth_channel_fraction",
    "depth_label_confidence",
    "reference_depth",
    "source_depth",
    "receiver_depth",
    "midpoint_depth",
    "interval_min_depth",
    "interval_max_depth",
)

DERIVED_BINARY_PREFIX = "derived_positive_at_fraction_"


@dataclass(frozen=True)
class GeometryRegressionContractInventory:
    inventory_version: str
    generated_at: str
    inputs: dict[str, str]
    array_inventory: list[dict[str, Any]]
    field_definitions: dict[str, str]
    stage9_summary_sources: dict[str, Any]
    stage10_audit_contract: dict[str, Any]
    issue_classification: str
    observations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryRegressionContractInvariants:
    invariants_version: str
    generated_at: str
    inputs: dict[str, str]
    passed: bool
    issue_classification: str
    checks: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_contract_inventory(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    stage9_report: dict[str, Any],
    stage10_report: dict[str, Any],
    inputs: dict[str, str],
) -> GeometryRegressionContractInventory:
    observations = _contract_observations(stage9_report, stage10_report)
    return GeometryRegressionContractInventory(
        inventory_version=CONTRACT_INVENTORY_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        array_inventory=[
            _array_inventory_row(key, value, label_arrays=label_arrays)
            for key, value in sorted(label_arrays.items())
        ],
        field_definitions={
            **TARGET_VIEW_DEFINITIONS,
            "geometry_kernel": "Metadata: ordered geometry kernel names.",
            "receiver_index": "Metadata: receiver indices R1 through R13.",
            "depth": "Depth-level XSI reference depth grid.",
            "depth_level_xsi_features": (
                "Existing depth-level XSI feature matrix used by the sanity audit."
            ),
        },
        stage9_summary_sources=_stage9_summary_sources(stage9_report),
        stage10_audit_contract=_stage10_audit_contract(stage10_report, feature_arrays),
        issue_classification=_classify_issue(observations),
        observations=observations,
    )


def build_contract_invariants(
    *,
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    stage9_report: dict[str, Any],
    stage10_report: dict[str, Any],
    inputs: dict[str, str],
) -> GeometryRegressionContractInvariants:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    depth_count = int(np.asarray(label_arrays["depth"]).size)
    kernels = np.asarray(label_arrays["geometry_kernel"]).astype(str)
    _target_value_checks(label_arrays, checks)
    _target_threshold_checks(label_arrays, checks)
    _depth_alignment_checks(label_arrays, feature_arrays, depth_count, checks)
    _stage9_report_checks(label_arrays, stage9_report, checks)
    _stage10_report_checks(label_arrays, stage10_report, kernels, checks)
    _feature_correlation_checks(stage10_report, checks)
    _protocol_transparency_checks(stage10_report, checks)
    errors = [str(check["name"]) for check in checks if not bool(check["passed"])]
    observations = _contract_observations(stage9_report, stage10_report)
    return GeometryRegressionContractInvariants(
        invariants_version=CONTRACT_INVARIANTS_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        passed=not errors,
        issue_classification=_classify_issue(observations, invariant_errors=errors),
        checks=checks,
        errors=errors,
        warnings=warnings,
    )


def write_contract_inventory_outputs(
    inventory: GeometryRegressionContractInventory,
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
        json.dumps(inventory.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_contract_inventory_markdown(inventory), encoding="utf-8")


def write_contract_invariant_outputs(
    invariants: GeometryRegressionContractInvariants,
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
        json.dumps(invariants.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(format_contract_invariants_markdown(invariants), encoding="utf-8")


def format_contract_inventory_markdown(
    inventory: GeometryRegressionContractInventory,
) -> str:
    lines = [
        "# Geometry Regression Contract Inventory",
        "",
        f"- inventory_version: `{inventory.inventory_version}`",
        f"- issue_classification: `{inventory.issue_classification}`",
        "",
        "## Stage 9 Summary Sources",
        "",
    ]
    lines.extend(_dict_lines(inventory.stage9_summary_sources))
    lines.extend(["", "## Stage 10 Audit Contract", ""])
    lines.extend(_dict_lines(inventory.stage10_audit_contract))
    lines.extend(["", "## Array Inventory", ""])
    for row in inventory.array_inventory:
        lines.append(
            "- "
            f"{row['key']}: shape={row['shape']}, dtype={row['dtype']}, "
            f"semantic_layer={row['semantic_layer']}, min={row['min']}, "
            f"max={row['max']}, finite_ratio={row['finite_ratio']}"
        )
    lines.extend(["", "## Observations", ""])
    lines.extend(_message_lines(inventory.observations))
    lines.append("")
    return "\n".join(lines)


def format_contract_invariants_markdown(
    invariants: GeometryRegressionContractInvariants,
) -> str:
    lines = [
        "# Geometry Regression Contract Invariants",
        "",
        f"- invariants_version: `{invariants.invariants_version}`",
        f"- passed: `{invariants.passed}`",
        f"- issue_classification: `{invariants.issue_classification}`",
        "",
        "## Checks",
        "",
    ]
    for check in invariants.checks:
        lines.append(
            "- "
            f"{check['name']}: passed={check['passed']}, "
            f"detail={check.get('detail', '')}"
        )
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(invariants.errors))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_message_lines(invariants.warnings))
    lines.append("")
    return "\n".join(lines)


def _array_inventory_row(
    key: str,
    value: np.ndarray,
    *,
    label_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    array = np.asarray(value)
    numeric = np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype(bool)
    finite_ratio = None
    minimum = None
    maximum = None
    if numeric:
        values = array.astype(np.float64, copy=False).reshape(-1)
        finite = np.isfinite(values)
        finite_ratio = None if values.size == 0 else float(np.mean(finite))
        if np.any(finite):
            minimum = float(np.min(values[finite]))
            maximum = float(np.max(values[finite]))
    return {
        "key": key,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": minimum,
        "max": maximum,
        "finite_ratio": finite_ratio,
        "semantic_layer": _semantic_layer(key, array, label_arrays),
        "definition": TARGET_VIEW_DEFINITIONS.get(key, ""),
    }


def _semantic_layer(
    key: str,
    array: np.ndarray,
    label_arrays: dict[str, np.ndarray],
) -> str:
    depth_count = int(np.asarray(label_arrays.get("depth", [])).size)
    kernel_count = int(np.asarray(label_arrays.get("geometry_kernel", [])).size)
    receiver_count = int(np.asarray(label_arrays.get("receiver_index", [])).size)
    if key.startswith(DERIVED_BINARY_PREFIX):
        return "derived binary sanity view"
    if key in DEPTH_LEVEL_TARGET_VIEWS:
        return "depth-level"
    if key in RECEIVER_LEVEL_TARGETS:
        return "receiver-level"
    if array.shape == (kernel_count, depth_count, receiver_count):
        return "receiver-level"
    if array.shape == (kernel_count, depth_count):
        return "depth-level"
    if key in {"geometry_kernel", "receiver_index"} or array.shape == ():
        return "metadata"
    return "metadata"


def _stage9_summary_sources(stage9_report: dict[str, Any]) -> dict[str, Any]:
    kernel_views = sorted(
        {str(row.get("target_view", "")) for row in stage9_report.get("kernel_summaries", [])}
    )
    depth_views = sorted(
        {
            str(row.get("target_view", ""))
            for row in stage9_report.get("target_view_summaries", [])
        }
    )
    return {
        "primary_target": stage9_report.get("primary_target"),
        "primary_target_layer": stage9_report.get("primary_target_layer"),
        "kernel_summary_target_views": kernel_views,
        "depth_level_target_views": depth_views,
        "derived_binary_support_source": (
            "depth-level threshold fractions are computed per target_view; persisted "
            "derived_positive_at_fraction_* arrays are receiver_max sanity views only"
        ),
    }


def _stage10_audit_contract(
    stage10_report: dict[str, Any],
    feature_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    features = np.asarray(feature_arrays.get("depth_level_xsi_features", []))
    return {
        "audited_target_view": stage10_report.get("audited_target_view")
        or stage10_report.get("target_field"),
        "audited_target_formula": stage10_report.get("audited_target_formula"),
        "audited_target_selection_reason": stage10_report.get(
            "audited_target_selection_reason"
        ),
        "permutation_margin_formula": stage10_report.get("permutation_margin_formula"),
        "permutation_seed_policy": stage10_report.get("permutation_seed_policy"),
        "permutation_unit": stage10_report.get("permutation_unit"),
        "permutation_count": stage10_report.get("permutation_count"),
        "cv_split_strategy": stage10_report.get("cv_split_strategy"),
        "cv_n_splits": stage10_report.get("cv_n_splits"),
        "feature_matrix_shape": list(features.shape),
    }


def _target_value_checks(label_arrays: dict[str, np.ndarray], checks: list[dict[str, Any]]) -> None:
    target_keys = [
        *DEPTH_LEVEL_TARGET_VIEWS,
        "raw_channel_fraction_zc_lt_2p5",
        "weighted_channel_fraction_zc_lt_2p5",
        "relative_anomaly_fraction",
        "combined_channel_fraction",
        "largest_connected_component_fraction",
        "max_azimuth_channel_fraction",
        "depth_label_confidence",
    ]
    for key in target_keys:
        if key not in label_arrays:
            continue
        values = np.asarray(label_arrays[key], dtype=np.float64)
        finite = np.isfinite(values)
        _add_check(
            checks,
            f"{key}: all values finite",
            bool(np.all(finite)),
            f"finite_ratio={float(np.mean(finite)) if values.size else None}",
        )
        if np.any(finite):
            _add_check(checks, f"{key}: min >= 0", bool(np.nanmin(values) >= -1e-8))
            _add_check(checks, f"{key}: max <= 1", bool(np.nanmax(values) <= 1.0 + 1e-8))


def _target_threshold_checks(
    label_arrays: dict[str, np.ndarray],
    checks: list[dict[str, Any]],
) -> None:
    for view in DEPTH_LEVEL_TARGET_VIEWS:
        if view not in label_arrays:
            continue
        values = np.asarray(label_arrays[view], dtype=np.float32)
        for kernel_index, kernel in enumerate(label_arrays["geometry_kernel"].astype(str)):
            target = values[kernel_index]
            gt0 = _fraction(target > 0.0)
            gt01 = _fraction(target > 0.01)
            gt05 = _fraction(target > 0.05)
            gt10 = _fraction(target > 0.10)
            _add_check(
                checks,
                f"{view}/{kernel}: threshold monotonicity",
                bool(gt10 <= gt05 <= gt01 <= gt0),
                f">0={gt0}, >0.01={gt01}, >0.05={gt05}, >0.10={gt10}",
            )


def _depth_alignment_checks(
    label_arrays: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray],
    depth_count: int,
    checks: list[dict[str, Any]],
) -> None:
    feature_depth = np.asarray(feature_arrays.get("depth", []), dtype=np.float64)
    label_depth = np.asarray(label_arrays.get("depth", []), dtype=np.float64)
    _add_check(
        checks,
        "label depth count aligns to XSI features",
        bool(depth_count == feature_depth.size),
        f"label={depth_count}, feature={feature_depth.size}",
    )
    _add_check(
        checks,
        "label depth grid aligns to XSI features",
        bool(label_depth.shape == feature_depth.shape and np.allclose(label_depth, feature_depth)),
    )


def _stage9_report_checks(
    label_arrays: dict[str, np.ndarray],
    stage9_report: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    _add_check(
        checks,
        "Stage 9 report has explicit target_view_summaries",
        bool(stage9_report.get("target_view_summaries")),
    )
    for row in stage9_report.get("target_view_summaries", []):
        view = str(row.get("target_view"))
        kernel = str(row.get("geometry_kernel"))
        if view not in label_arrays:
            continue
        kernel_index = _kernel_index(label_arrays, kernel)
        if kernel_index is None:
            continue
        target = np.asarray(label_arrays[view], dtype=np.float32)[kernel_index]
        nonzero = _fraction(target > 0.0)
        _add_check(
            checks,
            f"Stage 9 {view}/{kernel} nonzero summary matches NPZ",
            _close(nonzero, row.get("nonzero_fraction")),
            f"npz={nonzero}, report={row.get('nonzero_fraction')}",
        )
        for threshold, key in ((0.01, "0p01"), (0.05, "0p05"), (0.10, "0p10")):
            actual = _fraction(target > threshold)
            reported = _as_dict(row.get("fraction_gt_threshold")).get(key)
            _add_check(
                checks,
                f"Stage 9 {view}/{kernel} fraction_gt_{key} matches NPZ",
                _close(actual, reported),
                f"npz={actual}, report={reported}",
            )


def _stage10_report_checks(
    label_arrays: dict[str, np.ndarray],
    stage10_report: dict[str, Any],
    kernels: np.ndarray,
    checks: list[dict[str, Any]],
) -> None:
    target_view = stage10_report.get("audited_target_view") or stage10_report.get("target_field")
    _add_check(checks, "Stage 10 audited_target_view is explicit", bool(target_view))
    _add_check(
        checks,
        "Stage 10 audited target view exists in NPZ",
        bool(target_view in label_arrays),
        str(target_view),
    )
    summaries = stage10_report.get("kernel_summaries", [])
    _add_check(
        checks,
        "Stage 10 all kernels use one target view",
        bool({row.get("target_view") for row in summaries} in ({target_view}, {None})),
        str({row.get("target_view") for row in summaries}),
    )
    sensitivity = stage10_report.get("kernel_sensitivity", [])
    r7_row = _summary_for(summaries, "r7_reference_point")
    for row in sensitivity:
        kernel = str(row.get("geometry_kernel"))
        summary = _summary_for(summaries, kernel)
        if summary is None or r7_row is None:
            continue
        expected = _as_float(summary.get("nonzero_fraction")) - _as_float(
            r7_row.get("nonzero_fraction")
        )
        _add_check(
            checks,
            f"Stage 10 kernel sensitivity nonzero delta uses audited target view/{kernel}",
            _close(expected, row.get("nonzero_fraction_delta_vs_r7")),
            f"expected={expected}, report={row.get('nonzero_fraction_delta_vs_r7')}",
        )
    _add_check(
        checks,
        "Stage 10 includes permutation margin formula",
        stage10_report.get("permutation_margin_formula")
        == "abs(top_abs_spearman) - mean(abs(permutation_spearman))",
    )
    _add_check(
        checks,
        "Stage 10 summaries cover all kernels",
        bool(len(summaries) == len(kernels)),
    )


def _feature_correlation_checks(
    stage10_report: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    for row in stage10_report.get("kernel_summaries", []):
        kernel = row.get("geometry_kernel")
        _add_check(
            checks,
            f"{kernel}: top_abs_pearson_feature is explicit",
            bool(row.get("top_abs_pearson_feature")),
        )
        _add_check(
            checks,
            f"{kernel}: top_abs_spearman_feature is explicit",
            bool(row.get("top_abs_spearman_feature")),
        )


def _protocol_transparency_checks(
    stage10_report: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    for key in (
        "permutation_seed_policy",
        "permutation_unit",
        "permutation_count",
        "cv_split_strategy",
        "cv_n_splits",
        "cv_block_boundaries",
        "feature_matrix",
    ):
        _add_check(checks, f"Stage 10 {key} is explicit", bool(stage10_report.get(key)))
    _add_check(
        checks,
        "Stage 10 has no silent fallback",
        bool(stage10_report.get("no_silent_fallback")),
    )


def _contract_observations(
    stage9_report: dict[str, Any],
    stage10_report: dict[str, Any],
) -> list[str]:
    observations = []
    if stage9_report.get("primary_target") == "weighted_channel_fraction_zc_lt_2p5":
        observations.append(
            "Stage 9 primary_target is receiver-level weighted_channel_fraction_zc_lt_2p5."
        )
    audited = stage10_report.get("audited_target_view") or stage10_report.get("target_field")
    if audited:
        observations.append(f"Stage 10 audited_target_view is {audited}.")
    if stage9_report.get("target_view_summaries"):
        observations.append(
            "Stage 9 now separates receiver-level primary target summaries from depth-level views."
        )
    return observations


def _classify_issue(
    observations: list[str],
    *,
    invariant_errors: list[str] | None = None,
) -> str:
    if invariant_errors:
        return "multiple_issues"
    if any("separates receiver-level" in item for item in observations):
        return "report_ambiguity_only"
    return "unresolved_requires_manual_inspection"


def _summary_for(rows: list[dict[str, Any]], kernel: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("geometry_kernel") == kernel:
            return row
    return None


def _kernel_index(label_arrays: dict[str, np.ndarray], kernel: str) -> int | None:
    matches = np.where(label_arrays["geometry_kernel"].astype(str) == kernel)[0]
    return None if matches.size == 0 else int(matches[0])


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str = "",
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _fraction(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return 0.0 if values.size == 0 else float(np.mean(values))


def _close(left: Any, right: Any, *, atol: float = 1e-7) -> bool:
    left_value = _as_float(left)
    right_value = _as_float(right)
    if left_value is None or right_value is None:
        return False
    return bool(abs(left_value - right_value) <= atol)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
