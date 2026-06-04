from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

POLICY_VERSION = "mvp4x_regime_policy_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}
TARGET_VIEWS = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)


class RegimePolicyError(RuntimeError):
    """Raised when the MVP-4X research-only regime policy cannot be frozen."""


@dataclass(frozen=True)
class RegimePolicyOutputs:
    policy: dict[str, Any]
    cohort_names: list[str]
    cohort_masks: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "cohort_names": self.cohort_names,
            "cohort_masks_shape": list(self.cohort_masks.shape),
        }


def build_regime_policy_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    cf_decision_json: Path | str,
    cf_spatial_json: Path | str,
    stratified_decision_json: Path | str,
    stratified_iteration_log: Path | str,
    config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
) -> RegimePolicyOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    cf_decision = _read_json(Path(cf_decision_json))
    cf_spatial = _read_json(Path(cf_spatial_json))
    stratified_decision = _read_json(Path(stratified_decision_json))
    stratified_log_path = Path(stratified_iteration_log)
    if not stratified_log_path.exists():
        raise RegimePolicyError(f"Missing stratified iteration log: {stratified_log_path}")
    config = _load_yaml(Path(config_path))
    outputs = build_regime_policy(
        snapshot=snapshot,
        waveform=waveform,
        cf_decision=cf_decision,
        cf_spatial=cf_spatial,
        stratified_decision=stratified_decision,
        stratified_iteration_log=stratified_log_path.read_text(encoding="utf-8"),
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "cf_decision_json": str(cf_decision_json),
            "cf_spatial_json": str(cf_spatial_json),
            "stratified_decision_json": str(stratified_decision_json),
            "stratified_iteration_log": str(stratified_iteration_log),
            "config_path": str(config_path),
        },
    )
    write_regime_policy_outputs(
        outputs,
        output_npz=Path(output_npz),
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return outputs


def build_regime_policy(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    cf_decision: dict[str, Any],
    cf_spatial: dict[str, Any],
    stratified_decision: dict[str, Any],
    stratified_iteration_log: str,
    config: dict[str, Any],
    inputs: dict[str, str],
) -> RegimePolicyOutputs:
    _validate_artifacts(snapshot, waveform, cf_decision, cf_spatial, stratified_decision)
    masks = build_policy_cohorts(snapshot)
    cohort_names = [
        "regime_a_all",
        "regime_b_all",
        "regime_b_high_orientation",
        "regime_c_all",
        "regime_c_high_orientation",
        "pooled_bc_all",
        "pooled_bc_high_orientation",
    ]
    cohort_masks = np.vstack([masks[name] for name in cohort_names]).astype(bool)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    support = {
        "regime_a_high_orientation_support": int(
            np.count_nonzero(masks["regime_a_high_orientation"])
        ),
        "regime_b_low_orientation_support": int(
            np.count_nonzero(masks["regime_b_low_orientation"])
        ),
        "regime_b_high_orientation_support": int(
            np.count_nonzero(masks["regime_b_high_orientation"])
        ),
        "regime_c_low_orientation_support": int(
            np.count_nonzero(masks["regime_c_low_orientation"])
        ),
        "regime_c_high_orientation_support": int(
            np.count_nonzero(masks["regime_c_high_orientation"])
        ),
    }
    answers = _as_dict(stratified_decision.get("answers"))
    policy_config = _as_dict(config.get("regime_policy"))
    policy = {
        "policy_version": POLICY_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "regime_policy": {
            "regime_a": {
                "depth_range_ft": [2350.0, 3534.31],
                "status": "audit_only",
                "reason": "no_high_orientation_support",
                "no_generalization_claim_allowed": True,
            },
            "regime_b": {
                "depth_range_ft": [3534.31, 4678.49],
                "status": "exploratory_modeling_domain",
                "low_high_orientation_common_support_audit_allowed": True,
            },
            "regime_c": {
                "depth_range_ft": [4678.49, 5760.0],
                "status": "exploratory_modeling_domain",
                "low_orientation_support_unavailable": True,
            },
            "pooled_bc": {
                "status": "exploratory_reference_only",
                "allowed_for_comparison": True,
                "not_automatically_preferred_over_regime_specific_models": True,
            },
            "orientation": {
                "high_orientation": "exploratory_cohort_only",
                "low_orientation": "audit_cohort",
                "orientation_filter_formally_approved": False,
            },
            "target_views": {
                "receiver_mean": "exploratory_primary",
                "receiver_p90": "robust_reference",
                "receiver_max": "sensitive_audit_only",
                "full_360_fraction": "auxiliary_only",
                "receiver_std": "heterogeneity_audit_only",
            },
            "model_use": {
                "research_only_anomaly_ranking_screening": True,
                "absolute_channel_fraction_prediction": False,
                "final_model": False,
                "production_model": False,
            },
        },
        "cohorts": {
            name: _cohort_summary(depth, mask, snapshot)
            for name, mask in masks.items()
            if name in cohort_names
        },
        "support": support,
        "evidence": {
            "cf_decision": cf_decision.get("decision"),
            "stratified_decision": stratified_decision.get("decision"),
            "blocked_gap_10_ft_spearman": answers.get("13_blocked_gap_cv_stable"),
            "permutation_passed": answers.get("14_within_bin_permutation_passed"),
            "domain_shift_exists": answers.get("12_domain_shift_exists"),
            "best_target_view": answers.get("8_most_stable_target_view"),
            "best_feature_set": answers.get("9_most_stable_feature_set"),
            "best_model": answers.get("10_most_stable_model"),
            "stable_feature_groups": answers.get("11_stable_feature_groups", [])[:10],
            "b_to_c_transfer_spearman": answers.get("4_b_to_c_transfer"),
            "c_to_b_transfer_spearman": answers.get("5_c_to_b_transfer"),
            "stratified_iteration_log_read": bool(stratified_iteration_log.strip()),
        },
        "feature_contract": {
            "samples": int(depth.size),
            "existing_features": int(snapshot["xsi_features"].shape[1]),
            "waveform_v1_features": int(waveform["waveform_depth_features"].shape[1]),
            "combined_features": int(
                snapshot["xsi_features"].shape[1]
                + waveform["waveform_depth_features"].shape[1]
            ),
            "finite_ratios": {
                "existing_features": _finite_ratio(snapshot["xsi_features"]),
                "waveform_v1_features": _finite_ratio(waveform["waveform_depth_features"]),
            },
            "target_kernel": str(snapshot["target_kernel"]),
            "target_views": list(TARGET_VIEWS),
        },
        "config_policy": policy_config,
        "forbidden_claims": [
            "production split",
            "orientation filter",
            "absolute channel-fraction prediction",
            "final labels",
            "ground-truth claim",
            "production model",
        ],
        **_method_flags(),
    }
    return RegimePolicyOutputs(
        policy=policy,
        cohort_names=cohort_names,
        cohort_masks=cohort_masks,
    )


def build_policy_cohorts(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    all_samples = np.ones(depth.size, dtype=bool)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str).reshape(-1)
    low = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(-1)
    high = all_samples & ~low
    regime_a = regimes == "A"
    regime_b = regimes == "B"
    regime_c = regimes == "C"
    pooled_bc = regime_b | regime_c
    return {
        "regime_a_all": regime_a,
        "regime_a_high_orientation": regime_a & high,
        "regime_b_all": regime_b,
        "regime_b_low_orientation": regime_b & low,
        "regime_b_high_orientation": regime_b & high,
        "regime_c_all": regime_c,
        "regime_c_low_orientation": regime_c & low,
        "regime_c_high_orientation": regime_c & high,
        "pooled_bc_all": pooled_bc,
        "pooled_bc_high_orientation": pooled_bc & high,
    }


def write_regime_policy_outputs(
    outputs: RegimePolicyOutputs,
    *,
    output_npz: Path,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    for path in (output_npz, output_report_md, output_report_json):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(_json(outputs.policy), encoding="utf-8")
    output_report_md.write_text(format_regime_policy_markdown(outputs.policy), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        policy_version=np.asarray(POLICY_VERSION),
        cohort_names=np.asarray(outputs.cohort_names),
        cohort_masks=outputs.cohort_masks.astype(bool),
        metadata_json=np.asarray(_json(outputs.policy)),
        **{key: np.asarray(value) for key, value in _method_flags().items()},
    )


def format_regime_policy_markdown(policy: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Research-Only Regime Policy",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "no_final_labels, no_ground_truth_claim, no_production_claim.",
        "",
        f"- policy_version: `{policy['policy_version']}`",
        f"- stratified_decision: `{policy['evidence'].get('stratified_decision')}`",
        f"- target_kernel: `{policy['feature_contract'].get('target_kernel')}`",
        "",
        "## Policy",
    ]
    for name, item in policy["regime_policy"].items():
        lines.append(f"- {name}: {item}")
    lines.extend(["", "## Cohorts"])
    for name, cohort in policy["cohorts"].items():
        lines.append(
            f"- {name}: n={cohort['sample_count']}, "
            f"depth={cohort['depth_range_ft']}, target_mean={cohort['receiver_mean_mean']}"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            f"- B_to_C_transfer_spearman: {policy['evidence'].get('b_to_c_transfer_spearman')}",
            f"- C_to_B_transfer_spearman: {policy['evidence'].get('c_to_b_transfer_spearman')}",
            f"- domain_shift_exists: {policy['evidence'].get('domain_shift_exists')}",
            "",
            "## Forbidden Claims",
        ]
    )
    for claim in policy["forbidden_claims"]:
        lines.append(f"- {claim}")
    lines.append("")
    return "\n".join(lines)


def _cohort_summary(
    depth: np.ndarray,
    mask: np.ndarray,
    snapshot: dict[str, np.ndarray],
) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    target = np.asarray(snapshot["receiver_mean"], dtype=np.float32)
    return {
        "sample_count": int(np.count_nonzero(valid)),
        "sample_fraction": float(np.count_nonzero(valid) / max(1, depth.size)),
        "depth_range_ft": _depth_range(depth, valid),
        "receiver_mean_mean": None if not np.any(valid) else float(np.mean(target[valid])),
        "receiver_mean_p50": None if not np.any(valid) else float(np.quantile(target[valid], 0.5)),
        "low_orientation_fraction": None
        if not np.any(valid)
        else float(
            np.mean(np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool)[valid])
        ),
        "special_fraction": None
        if not np.any(valid)
        else float(np.mean(np.asarray(snapshot["any_special_flag"], dtype=bool)[valid])),
    }


def _validate_artifacts(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    cf_decision: dict[str, Any],
    cf_spatial: dict[str, Any],
    stratified_decision: dict[str, Any],
) -> None:
    if snapshot["depth"].shape != (7108,):
        raise RegimePolicyError(f"Unexpected sample shape: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise RegimePolicyError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise RegimePolicyError(
            f"Unexpected waveform-v1 feature shape: {waveform['waveform_depth_features'].shape}"
        )
    if snapshot["xsi_features"].shape[1] + waveform["waveform_depth_features"].shape[1] != 422:
        raise RegimePolicyError("Combined feature count is not 422.")
    if _finite_ratio(snapshot["xsi_features"]) != 1.0:
        raise RegimePolicyError("Existing feature finite ratio is not 1.0.")
    if _finite_ratio(waveform["waveform_depth_features"]) != 1.0:
        raise RegimePolicyError("Waveform-v1 feature finite ratio is not 1.0.")
    if str(snapshot["target_kernel"]) != "triangular_midpoint_weighted":
        raise RegimePolicyError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for target in TARGET_VIEWS:
        if target not in snapshot:
            raise RegimePolicyError(f"Missing target view: {target}")
    for name, container in (("snapshot", snapshot), ("waveform", waveform)):
        for flag in RESEARCH_FLAGS:
            if not bool(np.asarray(container.get(flag, False)).item()):
                raise RegimePolicyError(f"{name} missing research flag {flag}.")
    for name, report in (
        ("cf_decision", cf_decision),
        ("cf_spatial", cf_spatial),
        ("stratified_decision", stratified_decision),
    ):
        for flag in RESEARCH_FLAGS:
            if report.get(flag) is not True:
                raise RegimePolicyError(f"{name} missing research flag {flag}.")
    if stratified_decision.get("decision") != "exploratory_regime_specific_baseline_supported":
        raise RegimePolicyError(
            "Stratified decision does not support regime policy consolidation."
        )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise RegimePolicyError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegimePolicyError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RegimePolicyError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegimePolicyError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _depth_range(depth: np.ndarray, mask: np.ndarray) -> list[float | None]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth)
    if not np.any(valid):
        return [None, None]
    return [float(np.min(depth[valid])), float(np.max(depth[valid]))]


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
