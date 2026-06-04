from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
DESIGN_VERSION = "mvp4x_stratified_design_v001"
TARGET_VIEWS = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)


@dataclass(frozen=True)
class StratifiedDesignOutputs:
    design: dict[str, Any]
    cohort_names: list[str]
    cohort_masks: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "cohort_names": self.cohort_names,
            "cohort_masks_shape": list(self.cohort_masks.shape),
        }


class StratifiedDesignError(RuntimeError):
    """Raised when the MVP-4X stratified design cannot be frozen safely."""


def build_stratified_design_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    cf_decision_json: Path | str,
    cf_common_support_json: Path | str,
    cf_nuisance_json: Path | str,
    cf_spatial_json: Path | str,
    config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
) -> StratifiedDesignOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    cf_reports = {
        "cf_decision_json": _read_json(Path(cf_decision_json)),
        "cf_common_support_json": _read_json(Path(cf_common_support_json)),
        "cf_nuisance_json": _read_json(Path(cf_nuisance_json)),
        "cf_spatial_json": _read_json(Path(cf_spatial_json)),
    }
    config = _load_yaml(Path(config_path))
    outputs = build_stratified_design(
        snapshot=snapshot,
        waveform=waveform,
        cf_reports=cf_reports,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "cf_decision_json": str(cf_decision_json),
            "cf_common_support_json": str(cf_common_support_json),
            "cf_nuisance_json": str(cf_nuisance_json),
            "cf_spatial_json": str(cf_spatial_json),
            "config_path": str(config_path),
        },
    )
    write_stratified_design_outputs(
        outputs,
        output_npz=Path(output_npz),
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return outputs


def build_stratified_design(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    cf_reports: dict[str, dict[str, Any]],
    config: dict[str, Any],
    inputs: dict[str, str],
) -> StratifiedDesignOutputs:
    _validate_artifacts(snapshot=snapshot, waveform=waveform, cf_reports=cf_reports)
    masks = build_stratified_cohorts(snapshot)
    cohort_names = _configured_cohorts(config, masks)
    cohort_mask_matrix = np.vstack([masks[name] for name in cohort_names]).astype(bool)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    cohorts = {
        name: _cohort_summary(
            name=name,
            mask=masks[name],
            depth=depth,
            snapshot=snapshot,
            existing_features=existing,
            waveform_features=wave,
            masks=masks,
        )
        for name in cohort_names
    }
    support = {
        "regime_a_high_orientation_support": int(
            np.count_nonzero(
                (np.asarray(snapshot["broad_regime_id"]).astype(str) == "A")
                & ~np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool)
            )
        ),
        "regime_c_low_orientation_support": int(
            np.count_nonzero(
                (np.asarray(snapshot["broad_regime_id"]).astype(str) == "C")
                & np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool)
            )
        ),
        "regime_b_low_orientation_support": int(
            np.count_nonzero(masks["regime_b_low_orientation"])
        ),
        "regime_b_high_orientation_support": int(
            np.count_nonzero(masks["regime_b_high_orientation"])
        ),
    }
    policy = {
        "regime_a_no_high_orientation_support": support[
            "regime_a_high_orientation_support"
        ]
        == 0,
        "regime_c_no_low_orientation_support": support["regime_c_low_orientation_support"]
        == 0,
        "only_regime_b_orientation_common_support_audit_allowed": True,
        "regime_boundary_research_evaluation_only": True,
        "regime_id_not_model_input": True,
        "orientation_only_cohort_definition": True,
        "orientation_confidence_not_model_input": True,
        "no_formal_cohort_production_filter": True,
    }
    design = {
        "design_version": DESIGN_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "cohort_count": len(cohort_names),
        "cohort_names": cohort_names,
        "cohorts": cohorts,
        "support": support,
        "policy": policy,
        "target_views": list(TARGET_VIEWS),
        "feature_contract": {
            "existing_features": int(existing.shape[1]),
            "waveform_v1_features": int(wave.shape[1]),
            "combined_features": int(existing.shape[1] + wave.shape[1]),
            "finite_ratios": {
                "existing_features": _finite_ratio(existing),
                "waveform_v1_features": _finite_ratio(wave),
            },
            "target_kernel": str(snapshot["target_kernel"]),
        },
        "cf_evidence_summary": {
            "decision": cf_reports["cf_decision_json"].get("decision"),
            "orientation_effect_identifiable": cf_reports["cf_decision_json"]
            .get("answers", {})
            .get("1_orientation_effect_identifiable"),
            "stratified_study_basis": (
                "exploratory_regime_specific_signal_only; no formal regime split "
                "or orientation filter approved"
            ),
        },
        **_method_flags(),
    }
    return StratifiedDesignOutputs(
        design=design,
        cohort_names=cohort_names,
        cohort_masks=cohort_mask_matrix,
    )


def build_stratified_cohorts(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    all_samples = np.ones(depth.size, dtype=bool)
    low = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(-1)
    high = all_samples & ~low
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str).reshape(-1)
    regime_a = regimes == "A"
    regime_b = regimes == "B"
    regime_c = regimes == "C"
    regime_bc = regime_b | regime_c
    saturation = np.asarray(snapshot["saturation_platform_flag"], dtype=bool).reshape(-1)
    special_5680 = np.asarray(snapshot["special_5680_flag"], dtype=bool).reshape(-1)
    any_special = np.asarray(snapshot["any_special_flag"], dtype=bool).reshape(-1)
    return {
        "all_samples": all_samples,
        "high_orientation_cohort": high,
        "low_orientation_cohort": low,
        "regime_a_all": regime_a,
        "regime_b_all": regime_b,
        "regime_c_all": regime_c,
        "regime_b_low_orientation": regime_b & low,
        "regime_b_high_orientation": regime_b & high,
        "regime_c_high_orientation": regime_c & high,
        "regime_a_low_orientation": regime_a & low,
        "regime_bc_all": regime_bc,
        "regime_bc_high_orientation": regime_bc & high,
        "regime_bc_high_orientation_exclude_2400_2500": regime_bc & high & ~saturation,
        "regime_bc_high_orientation_exclude_5680": regime_bc & high & ~special_5680,
        "regime_bc_high_orientation_exclude_all_special": regime_bc & high & ~any_special,
    }


def write_stratified_design_outputs(
    outputs: StratifiedDesignOutputs,
    *,
    output_npz: Path,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    for path in (output_npz, output_report_md, output_report_json):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(_json(outputs.design), encoding="utf-8")
    output_report_md.write_text(format_design_markdown(outputs.design), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        design_version=np.asarray(DESIGN_VERSION),
        cohort_names=np.asarray(outputs.cohort_names),
        cohort_masks=outputs.cohort_masks,
        metadata_json=np.asarray(json.dumps(outputs.design, ensure_ascii=False)),
        **{key: np.asarray(value) for key, value in _method_flags().items()},
    )


def format_design_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Stratified Exploratory Design",
        "",
        _scope_line(),
        "",
        f"- design_version: `{report.get('design_version')}`",
        f"- cohort_count: {report.get('cohort_count')}",
        f"- cf_decision_basis: {report.get('cf_evidence_summary', {}).get('decision')}",
        "",
        "## Policy",
    ]
    for key, value in report.get("policy", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Cohorts"])
    for name, row in report.get("cohorts", {}).items():
        receiver_mean = row.get("target_distribution", {}).get("receiver_mean", {})
        lines.append(
            f"- {name}: n={row.get('sample_count')}, "
            f"depth={row.get('depth_range')}, "
            f"receiver_mean_p50={receiver_mean.get('p50')}"
        )
    return "\n".join(lines) + "\n"


def _cohort_summary(
    *,
    name: str,
    mask: np.ndarray,
    depth: np.ndarray,
    snapshot: dict[str, np.ndarray],
    existing_features: np.ndarray,
    waveform_features: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    return {
        "cohort": name,
        "sample_count": int(np.count_nonzero(valid)),
        "sample_fraction": float(np.count_nonzero(valid) / max(depth.size, 1)),
        "depth_range": _depth_range(depth, valid),
        "target_distribution": {
            target: _summary(np.asarray(snapshot[target], dtype=np.float32)[valid])
            for target in TARGET_VIEWS
        },
        "feature_distribution": {
            "existing_mean_abs": _summary(np.mean(np.abs(existing_features[valid]), axis=1)),
            "waveform_v1_mean_abs": _summary(np.mean(np.abs(waveform_features[valid]), axis=1)),
        },
        "finite_ratio": {
            "existing_features": _finite_ratio(existing_features[valid]),
            "waveform_v1_features": _finite_ratio(waveform_features[valid]),
        },
        "overlaps": {
            "regime_a_all": int(np.count_nonzero(valid & masks["regime_a_all"])),
            "regime_b_all": int(np.count_nonzero(valid & masks["regime_b_all"])),
            "regime_c_all": int(np.count_nonzero(valid & masks["regime_c_all"])),
            "high_orientation_cohort": int(
                np.count_nonzero(valid & masks["high_orientation_cohort"])
            ),
            "low_orientation_cohort": int(
                np.count_nonzero(valid & masks["low_orientation_cohort"])
            ),
        },
        "special_band_overlap": {
            "saturation_platform_2400_2500": int(
                np.count_nonzero(
                    valid & np.asarray(snapshot["saturation_platform_flag"], dtype=bool)
                )
            ),
            "special_band_5680": int(
                np.count_nonzero(valid & np.asarray(snapshot["special_5680_flag"], dtype=bool))
            ),
            "any_special_flag": int(
                np.count_nonzero(valid & np.asarray(snapshot["any_special_flag"], dtype=bool))
            ),
        },
    }


def _validate_artifacts(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    cf_reports: dict[str, dict[str, Any]],
) -> None:
    for target in TARGET_VIEWS:
        if target not in snapshot:
            raise StratifiedDesignError(f"Missing target view: {target}")
    if snapshot["depth"].shape[0] != 7108:
        raise StratifiedDesignError(f"Unexpected sample count: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise StratifiedDesignError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise StratifiedDesignError(
            "Unexpected waveform-v1 feature shape: "
            f"{waveform['waveform_depth_features'].shape}"
        )
    if snapshot["xsi_features"].shape[1] + waveform["waveform_depth_features"].shape[1] != 422:
        raise StratifiedDesignError("Combined feature count is not 422.")
    if _finite_ratio(snapshot["xsi_features"]) != 1.0:
        raise StratifiedDesignError("Existing feature finite ratio is not 1.0.")
    if _finite_ratio(waveform["waveform_depth_features"]) != 1.0:
        raise StratifiedDesignError("Waveform-v1 finite ratio is not 1.0.")
    if str(snapshot["target_kernel"]) != "triangular_midpoint_weighted":
        raise StratifiedDesignError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for name, container in (("snapshot", snapshot), ("waveform", waveform)):
        for flag in RESEARCH_FLAGS:
            if not bool(np.asarray(container.get(flag, False)).item()):
                raise StratifiedDesignError(f"{name} missing research flag {flag}.")
    for name, report in cf_reports.items():
        for flag in RESEARCH_FLAGS:
            if not bool(report.get(flag)):
                raise StratifiedDesignError(f"{name} missing research flag {flag}.")


def _configured_cohorts(config: dict[str, Any], masks: dict[str, np.ndarray]) -> list[str]:
    section = config.get("stratified_design", {})
    names = section.get("cohorts", list(masks))
    missing = [name for name in names if name not in masks]
    if missing:
        raise StratifiedDesignError(f"Configured cohort(s) are unavailable: {missing}")
    return list(names)


def _summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "p01": None, "p50": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _depth_range(depth: np.ndarray, mask: np.ndarray) -> list[float | None]:
    if not np.any(mask):
        return [None, None]
    return [float(np.min(depth[mask])), float(np.max(depth[mask]))]


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return 1.0
    return float(np.isfinite(arr).mean())


def _method_flags() -> dict[str, bool]:
    return {
        **RESEARCH_FLAGS,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_final_labels_generated": True,
        "no_raw_waveform_reread": True,
        "no_production_model_claim": True,
    }


def _scope_line() -> str:
    return (
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim."
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
