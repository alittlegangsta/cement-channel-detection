from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

POLICY_VERSION = "mvp4x_screening_policy_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}
TARGET_VIEWS = (
    "receiver_mean",
    "receiver_p90",
    "receiver_max",
    "full_360_fraction",
    "receiver_std",
)
DEFAULT_SUPPORTED = (
    "pooled_bc_all",
    "pooled_bc_high_orientation",
    "regime_b_high_orientation",
    "regime_c_all",
    "regime_c_high_orientation",
)


class ScreeningPolicyError(RuntimeError):
    """Raised when the MVP-4X screening policy cannot be frozen safely."""


@dataclass(frozen=True)
class ScreeningPolicyOutputs:
    policy: dict[str, Any]
    cohort_names: list[str]
    cohort_masks: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "cohort_names": self.cohort_names,
            "cohort_masks_shape": list(self.cohort_masks.shape),
        }


def build_screening_policy_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    regime_policy_json: Path | str,
    robustness_json: Path | str,
    ranking_json: Path | str,
    error_json: Path | str,
    screening_json: Path | str,
    regime_decision_json: Path | str,
    config_path: Path | str,
    output_npz: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
) -> ScreeningPolicyOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    regime_policy = _read_json(Path(regime_policy_json))
    robustness = _read_json(Path(robustness_json))
    ranking = _read_json(Path(ranking_json))
    error = _read_json(Path(error_json))
    screening = _read_json(Path(screening_json))
    regime_decision = _read_json(Path(regime_decision_json))
    config = _load_yaml(Path(config_path))
    outputs = build_screening_policy(
        snapshot=snapshot,
        waveform=waveform,
        regime_policy=regime_policy,
        robustness=robustness,
        ranking=ranking,
        error=error,
        screening=screening,
        regime_decision=regime_decision,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "regime_policy_json": str(regime_policy_json),
            "robustness_json": str(robustness_json),
            "ranking_json": str(ranking_json),
            "error_json": str(error_json),
            "screening_json": str(screening_json),
            "regime_decision_json": str(regime_decision_json),
            "config_path": str(config_path),
        },
    )
    write_screening_policy_outputs(
        outputs,
        output_npz=Path(output_npz),
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return outputs


def build_screening_policy(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    regime_policy: dict[str, Any],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
    screening: dict[str, Any],
    regime_decision: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
) -> ScreeningPolicyOutputs:
    _validate_artifacts(
        snapshot=snapshot,
        waveform=waveform,
        regime_policy=regime_policy,
        robustness=robustness,
        ranking=ranking,
        error=error,
        screening=screening,
        regime_decision=regime_decision,
    )
    masks = build_screening_cohorts(snapshot)
    screening_config = _as_dict(config.get("screening_baseline"))
    supported = list(screening.get("supported_cohorts") or DEFAULT_SUPPORTED)
    unsupported = sorted(
        set(screening.get("unsupported_cohorts") or [])
        | {"regime_a_all", "regime_b_all", "low_orientation_outside_supported_cohorts"}
    )
    cohort_names = [
        "pooled_bc_all",
        "pooled_bc_high_orientation",
        "regime_b_high_orientation",
        "regime_c_all",
        "regime_c_high_orientation",
        "regime_a_all",
        "regime_b_all",
        "low_orientation_outside_supported_cohorts",
    ]
    cohort_masks = np.vstack([masks[name] for name in cohort_names]).astype(bool)
    robustness_rows = {
        row["cohort"]: row for row in _as_list(robustness.get("candidate_rows"))
    }
    ranking_rows = {row["cohort"]: row for row in _as_list(ranking.get("rows"))}
    error_summary = _as_dict(error.get("summary"))
    policy = {
        "policy_version": POLICY_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "primary_research_score": {
            "target": "receiver_mean",
            "model": "Ridge",
            "feature_set": "existing_features_only",
            "random_seed": int(screening_config.get("random_seed", 20240603)),
            "interpretation": "research_only_anomaly_ranking_and_screening",
        },
        "ranking_use": {
            "anomaly_ranking": True,
            "candidate_screening": True,
            "manual_review_prioritization": True,
        },
        "unsupported_uses": {
            "absolute_channel_fraction_prediction": True,
            "production_inference": True,
            "final_labels": True,
            "ground_truth_claim": True,
            "cross_well_generalization_claim": True,
            "regime_a_generalization_claim": True,
        },
        "supported_screening_cohorts": supported,
        "unsupported_or_audit_only_cohorts": unsupported,
        "cohorts": {
            name: _cohort_summary(snapshot, masks[name], name=name, supported=supported)
            for name in cohort_names
        },
        "target_view_policy": {
            "receiver_mean": "exploratory_screening_primary",
            "receiver_p90": "robust_reference",
            "receiver_max": "sensitive_audit_only",
            "full_360_fraction": "auxiliary_only",
            "receiver_std": "heterogeneity_audit_only",
        },
        "calibration_policy": {
            "absolute_prediction_supported": False,
            "ranking_only_interpretation": True,
            "max_abs_calibration_bias": error_summary.get("max_abs_calibration_bias"),
        },
        "evidence": {
            "regime_policy_decision": regime_decision.get("decision"),
            "screening_baseline_decision": screening.get("screening_version"),
            "domain_shift": bool(screening.get("domain_shift")),
            "b_to_c_transfer_spearman": _as_dict(regime_decision.get("answers")).get(
                "b_to_c_transfer"
            ),
            "c_to_b_transfer_spearman": _as_dict(regime_decision.get("answers")).get(
                "c_to_b_transfer"
            ),
            "stable_features": _as_dict(regime_decision.get("answers")).get(
                "18_stable_features",
                [],
            )[:10],
            "policy_candidate_metrics": {
                name: _candidate_metric_summary(name, robustness_rows, ranking_rows)
                for name in supported
            },
        },
        "feature_contract": {
            "samples": int(np.asarray(snapshot["depth"]).size),
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
            "target_kernel": str(np.asarray(snapshot["target_kernel"]).item()),
            "target_views": list(TARGET_VIEWS),
        },
        "model_input_exclusions": [
            "depth",
            "regime_id",
            "orientation_confidence",
            "inclination",
            "special-band flags",
            "CAST-derived arrays",
            "morphology arrays",
        ],
        "forbidden_claims": [
            "production model",
            "deployment validation",
            "final labels",
            "ground-truth claim",
            "absolute channel-fraction prediction",
            "cross-well generalization",
            "formal production regime split",
            "formal orientation filter",
        ],
        **_method_flags(),
    }
    return ScreeningPolicyOutputs(
        policy=policy,
        cohort_names=cohort_names,
        cohort_masks=cohort_masks,
    )


def build_screening_cohorts(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    all_samples = np.ones(depth.size, dtype=bool)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str).reshape(-1)
    low = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(-1)
    high = all_samples & ~low
    regime_a = regimes == "A"
    regime_b = regimes == "B"
    regime_c = regimes == "C"
    pooled_bc = regime_b | regime_c
    supported_union = (pooled_bc & high) | regime_c | (regime_b & high) | pooled_bc
    return {
        "all_samples": all_samples,
        "pooled_bc_all": pooled_bc,
        "pooled_bc_high_orientation": pooled_bc & high,
        "regime_b_high_orientation": regime_b & high,
        "regime_c_all": regime_c,
        "regime_c_high_orientation": regime_c & high,
        "regime_a_all": regime_a,
        "regime_b_all": regime_b,
        "low_orientation_outside_supported_cohorts": low & ~supported_union,
    }


def write_screening_policy_outputs(
    outputs: ScreeningPolicyOutputs,
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
    output_report_md.write_text(format_screening_policy_markdown(outputs.policy), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        policy_version=np.asarray(POLICY_VERSION),
        cohort_names=np.asarray(outputs.cohort_names),
        cohort_masks=outputs.cohort_masks.astype(bool),
        metadata_json=np.asarray(_json(outputs.policy)),
        **{key: np.asarray(value) for key, value in _method_flags().items()},
    )


def format_screening_policy_markdown(policy: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Research Screening Baseline Policy",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
        f"- policy_version: `{policy['policy_version']}`",
        f"- target: `{policy['primary_research_score']['target']}`",
        f"- model: `{policy['primary_research_score']['model']}`",
        f"- feature_set: `{policy['primary_research_score']['feature_set']}`",
        f"- supported_screening_cohorts: {policy['supported_screening_cohorts']}",
        f"- unsupported_or_audit_only_cohorts: {policy['unsupported_or_audit_only_cohorts']}",
        f"- max_abs_calibration_bias: {policy['calibration_policy']['max_abs_calibration_bias']}",
        "",
        "## Cohorts",
    ]
    for name, cohort in policy["cohorts"].items():
        lines.append(
            f"- {name}: n={cohort['sample_count']}, status={cohort['score_status']}, "
            f"depth={cohort['depth_range_ft']}"
        )
    lines.extend(["", "## Forbidden Claims"])
    lines.extend(f"- {item}" for item in policy["forbidden_claims"])
    lines.append("")
    return "\n".join(lines)


def _cohort_summary(
    snapshot: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    name: str,
    supported: list[str],
) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    target = np.asarray(snapshot["receiver_mean"], dtype=np.float32).reshape(-1)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str).reshape(-1)
    low = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(-1)
    sample_count = int(np.count_nonzero(valid))
    regime_values = sorted(set(regimes[valid].tolist())) if sample_count else []
    return {
        "sample_count": sample_count,
        "sample_fraction": float(sample_count / max(1, depth.size)),
        "score_status": "supported_screening"
        if name in supported
        else "audit_only_or_unsupported",
        "depth_range_ft": _depth_range(depth, valid),
        "regimes": regime_values,
        "low_orientation_fraction": None if not sample_count else float(np.mean(low[valid])),
        "target_receiver_mean": _summary(target[valid]),
        "review_only": True,
        "supported_by_policy": name in supported,
        "supported_policy_names": supported,
    }


def _candidate_metric_summary(
    name: str,
    robustness_rows: dict[str, Any],
    ranking_rows: dict[str, Any],
) -> dict[str, Any]:
    robust = _as_dict(robustness_rows.get(name))
    ranking = _as_dict(ranking_rows.get(name))
    repeated = _as_dict(_as_dict(robust.get("repeated_cv")).get("metrics"))
    return {
        "sample_count": robust.get("sample_count"),
        "spearman_mean": _as_dict(repeated.get("spearman")).get("mean"),
        "r2_mean": _as_dict(repeated.get("r2")).get("mean"),
        "blocked_gap": robust.get("blocked_gap"),
        "permutation_margins": _as_dict(robust.get("permutation")).get("margins"),
        "top_10_lift": _as_dict(
            _as_dict(_as_dict(ranking.get("ranking")).get("top_k")).get("top_10pct")
        ).get("lift"),
        "ordinal_macro_f1": _as_dict(ranking.get("derived_ordinal_audit")).get("macro_f1"),
    }


def _validate_artifacts(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    regime_policy: dict[str, Any],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
    screening: dict[str, Any],
    regime_decision: dict[str, Any],
) -> None:
    if snapshot["depth"].shape != (7108,):
        raise ScreeningPolicyError(f"Unexpected sample shape: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise ScreeningPolicyError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise ScreeningPolicyError(
            f"Unexpected waveform feature shape: {waveform['waveform_depth_features'].shape}"
        )
    if snapshot["xsi_features"].shape[1] + waveform["waveform_depth_features"].shape[1] != 422:
        raise ScreeningPolicyError("Combined feature count is not 422.")
    if _finite_ratio(snapshot["xsi_features"]) != 1.0:
        raise ScreeningPolicyError("Existing feature finite ratio is not 1.0.")
    if _finite_ratio(waveform["waveform_depth_features"]) != 1.0:
        raise ScreeningPolicyError("Waveform feature finite ratio is not 1.0.")
    if str(np.asarray(snapshot["target_kernel"]).item()) != "triangular_midpoint_weighted":
        raise ScreeningPolicyError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for target in TARGET_VIEWS:
        if target not in snapshot:
            raise ScreeningPolicyError(f"Missing target view: {target}")
    for name, container in (("snapshot", snapshot), ("waveform", waveform)):
        for flag in RESEARCH_FLAGS:
            if flag == "not_validated_for_deployment":
                continue
            if not bool(np.asarray(container.get(flag, False)).item()):
                raise ScreeningPolicyError(f"{name} missing research flag {flag}.")
    for name, report in (
        ("regime_policy", regime_policy),
        ("robustness", robustness),
        ("ranking", ranking),
        ("error", error),
        ("screening", screening),
        ("regime_decision", regime_decision),
    ):
        for flag in RESEARCH_FLAGS:
            if flag == "not_validated_for_deployment":
                continue
            if report.get(flag) is not True:
                raise ScreeningPolicyError(f"{name} missing research flag {flag}.")
    if regime_decision.get("decision") != "research_screening_baseline_domain_shift_limited":
        raise ScreeningPolicyError("Regime policy decision is not the approved RP input.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise ScreeningPolicyError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ScreeningPolicyError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ScreeningPolicyError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ScreeningPolicyError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _summary(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "p50": None, "max": None, "mean": None}
    return {
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _depth_range(depth: np.ndarray, mask: np.ndarray) -> list[float | None]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth)
    if not np.any(valid):
        return [None, None]
    return [float(np.min(depth[valid])), float(np.max(depth[valid]))]


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values)
    return 0.0 if arr.size == 0 else float(np.isfinite(arr).mean())


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
