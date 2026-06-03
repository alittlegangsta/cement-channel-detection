from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SNAPSHOT_VERSION = "mvp4x_research_snapshot_v001"
SNAPSHOT_REPORT_VERSION = "mvp4x_research_snapshot_report_v001"
MANIFEST_VERSION = "mvp4x_research_snapshot_manifest_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
TARGET_FIELDS = (
    "receiver_p90",
    "receiver_mean",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)
MORPHOLOGY_FIELDS = (
    "min_zc",
    "p05_zc",
    "p10_zc",
    "max_relative_drop",
    "largest_connected_component_fraction",
    "max_azimuth_channel_fraction",
    "candidate_cell_count",
    "total_cell_count",
)
FORBIDDEN_MODEL_FEATURE_PATTERNS = (
    "depth",
    "regime",
    "special",
    "orientation",
    "confidence",
    "label",
    "morphology",
    "cast",
    "zc",
    "target",
    "full_360_fraction",
)


@dataclass(frozen=True)
class ResearchSnapshotReport:
    report_version: str
    snapshot_version: str
    generated_at: str
    inputs: dict[str, str]
    output_npz: str
    manifest_json: str
    target_kernel: str
    target_kernel_index: int
    sample_count: int
    feature_count: int
    target_fields: list[str]
    feature_group_counts: dict[str, int]
    finite_ratios: dict[str, float | None]
    shapes: dict[str, list[int]]
    model_feature_policy: dict[str, Any]
    metadata_fields: list[str]
    morphology_fields: list[str]
    warnings: list[str]
    errors: list[str]
    research_only: bool
    exploratory_only: bool
    weak_label_target: bool
    no_final_labels: bool
    no_ground_truth_claim: bool
    no_production_claim: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_research_snapshot_from_paths(
    *,
    regression_labels_npz: Path | str,
    depth_level_features_npz: Path | str,
    cast_label_input_npz: Path | str,
    config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_manifest_json: Path | str,
    overwrite: bool = False,
) -> ResearchSnapshotReport:
    config = load_mvp4x_config(config_path)
    labels = _load_npz(regression_labels_npz)
    features = _load_npz(depth_level_features_npz)
    cast = _load_npz(cast_label_input_npz)
    arrays, report, manifest = build_research_snapshot(
        regression_labels=labels,
        depth_level_features=features,
        cast_label_input=cast,
        config=config,
        inputs={
            "regression_labels_npz": str(regression_labels_npz),
            "depth_level_features_npz": str(depth_level_features_npz),
            "cast_label_input_npz": str(cast_label_input_npz),
            "config_path": str(config_path),
        },
        output_npz=Path(output_npz),
        output_manifest_json=Path(output_manifest_json),
    )
    write_research_snapshot_outputs(
        arrays,
        report,
        manifest,
        output_npz=Path(output_npz),
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_manifest_json=Path(output_manifest_json),
        overwrite=overwrite,
    )
    return report


def load_mvp4x_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"MVP-4X config must contain a mapping: {config_path}")
    return data


def build_research_snapshot(
    *,
    regression_labels: dict[str, np.ndarray],
    depth_level_features: dict[str, np.ndarray],
    cast_label_input: dict[str, np.ndarray],
    config: dict[str, Any],
    inputs: dict[str, str],
    output_npz: Path | None = None,
    output_manifest_json: Path | None = None,
) -> tuple[dict[str, np.ndarray], ResearchSnapshotReport, dict[str, Any]]:
    warnings: list[str] = []
    errors: list[str] = []
    _validate_scope(config, errors)
    snapshot_config = _as_dict(config.get("snapshot"))
    target_kernel = str(snapshot_config.get("target_kernel", "triangular_midpoint_weighted"))
    kernels = np.asarray(regression_labels["geometry_kernel"]).astype(str)
    if target_kernel not in set(kernels.tolist()):
        raise ValueError(f"target_kernel {target_kernel!r} is absent from geometry_kernel.")
    kernel_index = int(np.flatnonzero(kernels == target_kernel)[0])

    depth = np.asarray(regression_labels["depth"], dtype=np.float32).reshape(-1)
    feature_depth = np.asarray(depth_level_features["depth"], dtype=np.float32).reshape(-1)
    if depth.shape != feature_depth.shape:
        raise ValueError("regression label depth and XSI feature depth shapes differ.")
    if not np.allclose(depth, feature_depth, atol=1.0e-3):
        raise ValueError("regression label depth and XSI feature depth values differ.")

    xsi_features = np.asarray(
        depth_level_features["depth_level_xsi_features"],
        dtype=np.float32,
    )
    xsi_feature_names = np.asarray(
        depth_level_features["depth_level_xsi_feature_names"],
    ).astype(str)
    if xsi_features.ndim != 2:
        raise ValueError("depth_level_xsi_features must have shape [depth, feature].")
    if xsi_features.shape[0] != depth.size:
        raise ValueError("XSI feature row count must match depth count.")
    if xsi_feature_names.size != xsi_features.shape[1]:
        raise ValueError("XSI feature name count must match feature count.")
    if not np.all(np.isfinite(xsi_features)):
        errors.append("depth_level_xsi_features contain non-finite values.")

    feature_group_counts = _feature_group_counts(depth_level_features)
    feature_groups = _feature_groups_by_column(depth_level_features, feature_group_counts)
    model_feature_mask = _allowed_model_feature_mask(xsi_feature_names)
    if int(np.count_nonzero(model_feature_mask)) != xsi_feature_names.size:
        blocked = xsi_feature_names[~model_feature_mask].tolist()
        errors.append("Forbidden model feature name(s): " + ", ".join(blocked[:20]))

    targets = {
        field: _select_kernel_depth_array(regression_labels, field, kernel_index)
        for field in TARGET_FIELDS
    }
    for field, values in targets.items():
        if values.shape != depth.shape:
            raise ValueError(f"{field} target shape must match depth.")
        if not np.all(np.isfinite(values)):
            errors.append(f"{field} contains non-finite values.")

    cast_depth = np.asarray(cast_label_input["cast_depth"], dtype=np.float32).reshape(-1)
    orientation_conf = _nearest_by_depth(
        source_depth=cast_depth,
        source_values=np.asarray(cast_label_input["orientation_confidence"], dtype=np.float32),
        query_depth=depth,
    )
    inc_deg = _nearest_by_depth(
        source_depth=cast_depth,
        source_values=np.asarray(cast_label_input["inc_deg"], dtype=np.float32),
        query_depth=depth,
    )
    cast_low_inc = _nearest_bool_by_depth(cast_depth, cast_label_input.get("low_inc_mask"), depth)
    cast_orientation_uncertain = _nearest_bool_by_depth(
        cast_depth,
        cast_label_input.get("orientation_uncertain"),
        depth,
    )
    low_orientation_threshold = float(
        snapshot_config.get("low_orientation_confidence_threshold", 0.5)
    )
    low_orientation_confidence_flag = (
        (orientation_conf < low_orientation_threshold)
        | cast_low_inc
        | cast_orientation_uncertain
        | ~np.isfinite(orientation_conf)
    )

    label_confidence = _depth_label_confidence(regression_labels, kernel_index, depth.size)
    broad_regime_id, broad_regime_code = _broad_regimes(depth, snapshot_config)
    special_flags = _special_flags(depth, snapshot_config)
    morphology = _morphology_arrays(regression_labels, kernel_index, depth.size, warnings)
    invalid_zc_impact_flag = _invalid_zc_impact_flag(morphology, depth.size, warnings)
    no_overlap_flag = _no_overlap_flag(morphology, depth.size, warnings)

    arrays: dict[str, np.ndarray] = {
        "snapshot_version": np.asarray(SNAPSHOT_VERSION),
        "depth": depth.astype(np.float32),
        "xsi_features": xsi_features.astype(np.float32),
        "xsi_feature_names": xsi_feature_names.astype(str),
        "xsi_feature_group": np.asarray(feature_groups).astype(str),
        "model_feature_mask": model_feature_mask.astype(bool),
        "finite_feature_mask": np.isfinite(xsi_features),
        "target_kernel": np.asarray(target_kernel),
        "target_kernel_index": np.asarray(kernel_index, dtype=np.int16),
        "broad_regime_id": broad_regime_id,
        "broad_regime_code": broad_regime_code.astype(np.int16),
        "label_confidence": label_confidence.astype(np.float32),
        "orientation_confidence": orientation_conf.astype(np.float32),
        "inclination_deg": inc_deg.astype(np.float32),
        "low_orientation_confidence_flag": low_orientation_confidence_flag.astype(bool),
        "invalid_zc_impact_flag": invalid_zc_impact_flag.astype(bool),
        "no_overlap_flag": no_overlap_flag.astype(bool),
        **{key: values.astype(np.float32) for key, values in targets.items()},
        **{key: values.astype(bool) for key, values in special_flags.items()},
        **morphology,
        **{key: np.asarray(value) for key, value in RESEARCH_FLAGS.items()},
        "no_stc": np.asarray(True),
        "no_apes": np.asarray(True),
        "no_deep_learning": np.asarray(True),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "target_policy": {
                        "receiver_p90": "exploratory_primary_target",
                        "receiver_mean": "conservative_reference_target",
                        "receiver_max": "sensitive_audit_target",
                        "full_360_fraction": "auxiliary_target",
                        "receiver_std": "heterogeneity_target",
                    },
                    "model_feature_exclusions": [
                        "depth",
                        "broad_regime_id",
                        "special_band_flags",
                        "orientation_confidence",
                        "label_confidence",
                        "morphology_arrays",
                        "CAST-derived_fields",
                    ],
                },
                sort_keys=True,
            )
        ),
    }
    finite_ratios = _finite_ratios(arrays)
    shapes = _shapes(arrays)
    report = ResearchSnapshotReport(
        report_version=SNAPSHOT_REPORT_VERSION,
        snapshot_version=SNAPSHOT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        output_npz=str(output_npz or ""),
        manifest_json=str(output_manifest_json or ""),
        target_kernel=target_kernel,
        target_kernel_index=kernel_index,
        sample_count=int(depth.size),
        feature_count=int(xsi_features.shape[1]),
        target_fields=list(TARGET_FIELDS),
        feature_group_counts=feature_group_counts,
        finite_ratios=finite_ratios,
        shapes=shapes,
        model_feature_policy={
            "broad_regime_id_allowed_as_model_feature": False,
            "special_flags_allowed_as_model_feature": False,
            "metadata_only_fields": _metadata_field_names(),
            "allowed_model_feature_count": int(np.count_nonzero(model_feature_mask)),
            "forbidden_model_feature_patterns": list(FORBIDDEN_MODEL_FEATURE_PATTERNS),
        },
        metadata_fields=_metadata_field_names(),
        morphology_fields=sorted(morphology),
        warnings=warnings,
        errors=errors,
        research_only=True,
        exploratory_only=True,
        weak_label_target=True,
        no_final_labels=True,
        no_ground_truth_claim=True,
        no_production_claim=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
    )
    manifest = build_snapshot_manifest(
        report=report,
        arrays=arrays,
        input_paths=inputs,
        config=config,
    )
    return arrays, report, manifest


def build_snapshot_manifest(
    *,
    report: ResearchSnapshotReport,
    arrays: dict[str, np.ndarray],
    input_paths: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": str(config.get("schema_version", "schema_v001")),
        "snapshot_version": report.snapshot_version,
        "created_at": report.generated_at,
        "scope": {**RESEARCH_FLAGS, "no_stc": True, "no_apes": True, "no_deep_learning": True},
        "inputs": {
            key: {
                "path": value,
                "sha256": _sha256_file(Path(value)) if value and Path(value).is_file() else None,
            }
            for key, value in input_paths.items()
        },
        "target_kernel": report.target_kernel,
        "shape": report.shapes,
        "finite_ratios": report.finite_ratios,
        "feature_groups": report.feature_group_counts,
        "model_feature_policy": report.model_feature_policy,
        "warnings": report.warnings,
        "errors": report.errors,
    }


def write_research_snapshot_outputs(
    arrays: dict[str, np.ndarray],
    report: ResearchSnapshotReport,
    manifest: dict[str, Any],
    *,
    output_npz: Path,
    output_report_md: Path,
    output_report_json: Path,
    output_manifest_json: Path,
    overwrite: bool,
) -> None:
    for path in (output_npz, output_report_md, output_report_json, output_manifest_json):
        _ensure_can_write(path, overwrite=overwrite)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_research_snapshot_markdown(report), encoding="utf-8")


def format_research_snapshot_markdown(report: ResearchSnapshotReport) -> str:
    lines = [
        "# MVP-4X Rapid Research Snapshot",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- snapshot_version: `{report.snapshot_version}`",
        f"- target_kernel: `{report.target_kernel}`",
        f"- sample_count: {report.sample_count}",
        f"- feature_count: {report.feature_count}",
        f"- target_fields: {', '.join(report.target_fields)}",
        f"- allowed_model_feature_count: "
        f"{report.model_feature_policy['allowed_model_feature_count']}",
        f"- warning_count: {len(report.warnings)}",
        f"- error_count: {len(report.errors)}",
        "",
        "## Feature Groups",
    ]
    for name, count in report.feature_group_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Finite Ratios"])
    for name, ratio in report.finite_ratios.items():
        if ratio is not None:
            lines.append(f"- {name}: {ratio:.6f}")
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report.warnings)
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in report.errors)
    lines.extend(
        [
            "",
            "Metadata fields are evaluation-only and are not model inputs: "
            + ", ".join(report.metadata_fields),
            "",
        ]
    )
    return "\n".join(lines)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Required NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _validate_scope(config: dict[str, Any], errors: list[str]) -> None:
    scope = _as_dict(config.get("scope"))
    for key in RESEARCH_FLAGS:
        if scope.get(key) is not True:
            errors.append(f"scope.{key} must be true.")


def _select_kernel_depth_array(
    arrays: dict[str, np.ndarray],
    field: str,
    kernel_index: int,
) -> np.ndarray:
    if field not in arrays:
        raise KeyError(f"Required target field is missing: {field}")
    values = np.asarray(arrays[field], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{field} must have shape [kernel, depth].")
    return values[kernel_index].reshape(-1)


def _depth_label_confidence(
    arrays: dict[str, np.ndarray],
    kernel_index: int,
    depth_count: int,
) -> np.ndarray:
    if "depth_label_confidence" not in arrays:
        raise KeyError("depth_label_confidence is required for the research snapshot.")
    values = np.asarray(arrays["depth_label_confidence"], dtype=np.float32)
    if values.ndim == 3:
        return np.nanmean(values[kernel_index], axis=1).reshape(depth_count)
    if values.ndim == 2:
        return values[kernel_index].reshape(depth_count)
    raise ValueError("depth_label_confidence must have shape [kernel, depth, receiver].")


def _feature_group_counts(feature_arrays: dict[str, np.ndarray]) -> dict[str, int]:
    raw = str(np.asarray(feature_arrays["feature_group_counts_json"]).reshape(()))
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("feature_group_counts_json must decode to a mapping.")
    return {str(key): int(value) for key, value in data.items()}


def _feature_groups_by_column(
    feature_arrays: dict[str, np.ndarray],
    counts: dict[str, int],
) -> list[str]:
    names = np.asarray(feature_arrays["feature_group_names"]).astype(str).tolist()
    groups: list[str] = []
    for name in names:
        groups.extend([name] * counts[name])
    feature_count = int(np.asarray(feature_arrays["depth_level_xsi_features"]).shape[1])
    if len(groups) != feature_count:
        raise ValueError(
            "feature group counts do not sum to feature count "
            f"({len(groups)} != {feature_count})."
        )
    return groups


def _allowed_model_feature_mask(feature_names: np.ndarray) -> np.ndarray:
    mask = np.ones(feature_names.size, dtype=bool)
    for index, name in enumerate(feature_names.astype(str)):
        lower = name.lower()
        if any(pattern in lower for pattern in FORBIDDEN_MODEL_FEATURE_PATTERNS):
            mask[index] = False
    return mask


def _nearest_by_depth(
    *,
    source_depth: np.ndarray,
    source_values: np.ndarray,
    query_depth: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(source_depth, dtype=np.float32).reshape(-1)
    values = np.asarray(source_values, dtype=np.float32).reshape(-1)
    if depth.size != values.size:
        raise ValueError("source_depth and source_values must have matching lengths.")
    order = np.argsort(depth)
    sorted_depth = depth[order]
    sorted_values = values[order]
    idx = np.searchsorted(sorted_depth, query_depth)
    idx = np.clip(idx, 1, sorted_depth.size - 1)
    left = idx - 1
    right = idx
    use_right = np.abs(sorted_depth[right] - query_depth) < np.abs(
        sorted_depth[left] - query_depth
    )
    nearest = np.where(use_right, right, left)
    return sorted_values[nearest].astype(np.float32)


def _nearest_bool_by_depth(
    source_depth: np.ndarray,
    source_values: np.ndarray | None,
    query_depth: np.ndarray,
) -> np.ndarray:
    if source_values is None:
        return np.zeros(query_depth.size, dtype=bool)
    values = _nearest_by_depth(
        source_depth=source_depth,
        source_values=np.asarray(source_values, dtype=np.float32),
        query_depth=query_depth,
    )
    return values >= 0.5


def _broad_regimes(
    depth: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    regimes = _as_list(config.get("broad_regimes"))
    ids = np.full(depth.size, "UNKNOWN", dtype="<U16")
    codes = np.full(depth.size, -1, dtype=np.int16)
    for code, regime in enumerate(regimes):
        row = _as_dict(regime)
        regime_id = str(row["id"])
        min_depth = float(row["min_depth_ft"])
        max_depth = float(row["max_depth_ft"])
        mask = (depth >= min_depth) & (depth < max_depth)
        if code == len(regimes) - 1:
            mask = (depth >= min_depth) & (depth <= max_depth)
        ids[mask] = regime_id
        codes[mask] = code
    return ids, codes


def _special_flags(depth: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    special = _as_dict(config.get("special_bands"))
    output: dict[str, np.ndarray] = {}
    for name, raw in special.items():
        row = _as_dict(raw)
        if "center_depth_ft" in row:
            center = float(row["center_depth_ft"])
            half_width = float(row.get("half_width_ft", 0.0))
            mask = np.abs(depth - center) <= half_width
        else:
            mask = (depth >= float(row["min_depth_ft"])) & (depth <= float(row["max_depth_ft"]))
        output[f"{name}_flag"] = mask.astype(bool)
    output["any_special_flag"] = np.logical_or.reduce(list(output.values())).astype(bool)
    return output


def _morphology_arrays(
    arrays: dict[str, np.ndarray],
    kernel_index: int,
    depth_count: int,
    warnings: list[str],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for field in MORPHOLOGY_FIELDS:
        if field not in arrays:
            warnings.append(f"morphology field unavailable: {field}")
            continue
        values = np.asarray(arrays[field])
        if values.ndim != 3:
            warnings.append(f"morphology field has unsupported shape: {field} {values.shape}")
            continue
        selected = values[kernel_index]
        if selected.shape[0] != depth_count:
            raise ValueError(f"{field} morphology depth dimension does not match depth.")
        output[f"morphology_{field}"] = selected
    return output


def _invalid_zc_impact_flag(
    morphology: dict[str, np.ndarray],
    depth_count: int,
    warnings: list[str],
) -> np.ndarray:
    min_zc = morphology.get("morphology_min_zc")
    if min_zc is None:
        warnings.append("invalid-Zc impact flag unavailable because morphology_min_zc is absent.")
        return np.zeros(depth_count, dtype=bool)
    return np.any(np.asarray(min_zc, dtype=np.float32) < 0.0, axis=1)


def _no_overlap_flag(
    morphology: dict[str, np.ndarray],
    depth_count: int,
    warnings: list[str],
) -> np.ndarray:
    total = morphology.get("morphology_total_cell_count")
    if total is None:
        warnings.append(
            "no-overlap flag unavailable because morphology_total_cell_count is absent."
        )
        return np.zeros(depth_count, dtype=bool)
    return np.any(np.asarray(total) <= 0, axis=1)


def _finite_ratios(arrays: dict[str, np.ndarray]) -> dict[str, float | None]:
    ratios: dict[str, float | None] = {}
    for key, values in arrays.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.number):
            ratios[key] = None if array.size == 0 else float(np.isfinite(array).mean())
    return ratios


def _shapes(arrays: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {key: [int(value) for value in np.asarray(array).shape] for key, array in arrays.items()}


def _metadata_field_names() -> list[str]:
    return [
        "depth",
        "broad_regime_id",
        "broad_regime_code",
        "special_band_flags",
        "label_confidence",
        "orientation_confidence",
        "inclination_deg",
        "low_orientation_confidence_flag",
        "invalid_zc_impact_flag",
        "no_overlap_flag",
        "morphology_*",
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
