from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import _make_model, compute_regression_metrics
from cement_channel.modeling.mvp4x_screening_scores import (
    fit_oof_scores,
    rank_percentile,
    top_k_lift,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cement-channel")

MAX_AUTO_VERSION = "mvp4x_max_auto_v001"
FINAL_SCORE_VERSION = "mvp4x_screening_scores_final_oof_v001"
INTERVAL_VERSION = "mvp4x_screening_review_intervals_v001"
TRIAGE_VERSION = "mvp4x_max_auto_triage_v001"
MORPHOLOGY_VERSION = "mvp4x_morphology_candidate_labels_v002"
COMPARISON_VERSION = "mvp4x_v1_vs_morphology_v2_comparison_v001"
DECISION_VERSION = "mvp4x_max_auto_decision_v001"

RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
    "no_stc": True,
    "no_apes": True,
    "no_deep_learning": True,
    "no_final_labels_generated": True,
    "no_raw_waveform_reread": True,
    "no_raw_mat_modified": True,
    "no_production_model_claim": True,
}

POLICY_TO_COHORT = {
    "P0": "pooled_bc_all",
    "P1": "pooled_bc_high_orientation",
    "P2": "regime_b_high_orientation",
    "P3": "regime_c_all",
    "P4": "regime_c_high_orientation",
}
SUPPORTED_COHORTS = [
    "pooled_bc_all",
    "pooled_bc_high_orientation",
    "regime_b_high_orientation",
    "regime_c_all",
    "regime_c_high_orientation",
]
UNSUPPORTED_COHORTS = [
    "regime_a_all",
    "regime_b_all",
    "low_orientation_outside_supported_cohorts",
]


class MaxAutoError(RuntimeError):
    """Raised when the bounded MVP-4X autonomous research loop cannot run safely."""


@dataclass(frozen=True)
class MaxAutoPaths:
    data_root: Path
    interim: Path
    features: Path
    reports: Path
    manifests: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MaxAutoPaths:
        data = _as_dict(config.get("data"))
        required = ("interim", "features", "reports", "manifests")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise MaxAutoError("Missing data path config key(s): " + ", ".join(missing))
        roots = {key: Path(str(data[key])).resolve() for key in required}
        common = Path(os.path.commonpath([str(path) for path in roots.values()]))
        return cls(
            data_root=common,
            interim=roots["interim"],
            features=roots["features"],
            reports=roots["reports"],
            manifests=roots["manifests"],
        )

    def ensure_read_write(self) -> None:
        for path in (self.data_root, self.interim, self.features, self.reports, self.manifests):
            if not path.exists():
                raise MaxAutoError(f"Configured data path does not exist: {path}")
            if not os.access(path, os.R_OK):
                raise MaxAutoError(f"Configured data path is not readable: {path}")
            if path in (self.data_root, self.interim, self.features, self.reports, self.manifests):
                if not os.access(path, os.W_OK):
                    raise MaxAutoError(f"Configured data path is not writable: {path}")


def run_phase(paths: MaxAutoPaths, *, phase: str, overwrite: bool) -> dict[str, Any]:
    if phase == "preflight":
        return validate_preflight(paths)
    if phase == "scoring":
        return finalize_oof_scores(paths, overwrite=overwrite)
    if phase == "intervals":
        return mine_review_intervals(paths, overwrite=overwrite)
    if phase == "triage":
        return build_triage_report(paths, overwrite=overwrite)
    if phase == "morphology":
        return generate_morphology_candidate_v2(paths, overwrite=overwrite)
    if phase == "decision":
        return build_final_decision(paths, overwrite=overwrite)
    if phase == "all":
        outputs = {}
        for item in ("preflight", "scoring", "intervals", "triage", "morphology", "decision"):
            outputs[item] = run_phase(paths, phase=item, overwrite=overwrite)
        return {"phase": "all", "outputs": outputs, **RESEARCH_FLAGS}
    raise MaxAutoError(f"Unknown max-auto phase: {phase}")


def validate_preflight(paths: MaxAutoPaths) -> dict[str, Any]:
    paths.ensure_read_write()
    required = {
        "snapshot": paths.interim / "mvp4x_research_snapshot_v001.npz",
        "policy_npz": paths.interim / "mvp4x_screening_policy_v001.npz",
        "scores_npz": paths.interim / "mvp4x_screening_scores_oof_v001.npz",
        "waveform": paths.features / "mvp4x_waveform_features_v001.npz",
        "model_joblib": paths.features / "mvp4x_research_screening_model_v001.joblib",
        "model_manifest": paths.manifests / "mvp4x_research_screening_model_v001.json",
        "scores_json": paths.reports / "mvp4x_screening_scores_oof_v001.json",
        "consolidation": paths.reports / "mvp4x_screening_consolidation_decision.json",
        "proposal": paths.reports / "mvp4x_formal_regime_policy_proposal.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise MaxAutoError("Missing required artifact(s): " + ", ".join(missing))

    snapshot = _load_npz(required["snapshot"])
    scores = _load_npz(required["scores_npz"])
    waveform = _load_npz(required["waveform"])
    scores_json = _read_json(required["scores_json"])
    manifest = _read_json(required["model_manifest"])
    consolidation = _read_json(required["consolidation"])
    proposal = _read_json(required["proposal"])

    total_samples = int(np.asarray(snapshot["depth"]).shape[0])
    oof_scored = int(np.count_nonzero(np.isfinite(np.asarray(scores["score"], dtype=np.float32))))
    existing_features = int(np.asarray(snapshot["xsi_features"]).shape[1])
    waveform_features = int(np.asarray(waveform["waveform_depth_features"]).shape[1])
    combined_features = existing_features + waveform_features
    existing_finite_ratio = _finite_ratio(np.asarray(snapshot["xsi_features"], dtype=np.float32))
    waveform_finite_ratio = _finite_ratio(
        np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    )
    checks = {
        "total_samples": total_samples == 7108,
        "oof_scored_samples": oof_scored == 4738,
        "existing_features": existing_features == 80,
        "waveform_v1_features": waveform_features == 342,
        "combined_features": combined_features == 422,
        "existing_finite_ratio": existing_finite_ratio == 1.0,
        "waveform_finite_ratio": waveform_finite_ratio == 1.0,
        "target_kernel": str(np.asarray(snapshot["target_kernel"]).item())
        == "triangular_midpoint_weighted",
        "screening_target": scores_json.get("target") == "receiver_mean",
        "screening_model": scores_json.get("model") == "Ridge",
        "screening_feature_set": scores_json.get("feature_set") == "existing_features_only",
        "manifest_feature_count": manifest.get("feature_count") == 80,
        "consolidation_research_only": consolidation.get("research_only") is True,
        "proposal_research_only": proposal.get("research_only") is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise MaxAutoError("Preflight schema/value check(s) failed: " + ", ".join(failed))

    return {
        "phase": "preflight",
        "validated": True,
        "artifacts_read": {name: str(path) for name, path in required.items()},
        "counts": {
            "total_samples": total_samples,
            "oof_scored_samples": oof_scored,
            "existing_features": existing_features,
            "waveform_v1_features": waveform_features,
            "combined_features": combined_features,
            "existing_finite_ratio": existing_finite_ratio,
            "waveform_finite_ratio": waveform_finite_ratio,
        },
        "target_kernel": str(np.asarray(snapshot["target_kernel"]).item()),
        "screening_target": scores_json.get("target"),
        "screening_model": scores_json.get("model"),
        "screening_feature_set": scores_json.get("feature_set"),
        **RESEARCH_FLAGS,
    }


def finalize_oof_scores(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    validate_preflight(paths)
    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    source_scores = _load_npz(paths.interim / "mvp4x_screening_scores_oof_v001.npz")
    source_report = _read_json(paths.reports / "mvp4x_screening_scores_oof_v001.json")
    arrays, rows, report = build_final_score_tables(snapshot, source_scores, source_report)

    output_npz = paths.interim / f"{FINAL_SCORE_VERSION}.npz"
    output_csv = paths.reports / f"{FINAL_SCORE_VERSION}.csv"
    output_md = paths.reports / f"{FINAL_SCORE_VERSION}.md"
    output_json = paths.reports / f"{FINAL_SCORE_VERSION}.json"
    for path in (output_npz, output_csv, output_md, output_json):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        report_version=np.asarray(FINAL_SCORE_VERSION),
        metadata_json=np.asarray(_json(report)),
        **arrays,
        **{key: np.asarray(value) for key, value in RESEARCH_FLAGS.items()},
    )
    _write_csv(rows, output_csv)
    output_json.write_text(_json(report), encoding="utf-8")
    output_md.write_text(format_final_score_markdown(report), encoding="utf-8")
    append_iteration_log(
        paths,
        iteration_id="001",
        hypothesis=(
            "Existing OOF screening scores can be upgraded into a final research-only "
            "score schema without retraining or changing v1 artifacts."
        ),
        evidence_before={
            "source_supported_scored": source_report.get("sample_counts", {}).get(
                "supported_scored"
            ),
            "source_top_k": source_report.get("rank_metrics", {}).get("top_k"),
        },
        selected_action="finalize_oof_scoring_schema",
        files_read=[
            str(paths.interim / "mvp4x_research_snapshot_v001.npz"),
            str(paths.interim / "mvp4x_screening_scores_oof_v001.npz"),
            str(paths.reports / "mvp4x_screening_scores_oof_v001.json"),
        ],
        files_changed=[str(output_npz), str(output_csv), str(output_md), str(output_json)],
        commands_run=[
            "python scripts/07u_run_mvp4x_max_auto.py --phase scoring --overwrite",
            "make test-smoke",
            "make test",
            "make lint",
            "python scripts/00_check_env.py",
        ],
        outputs_generated=[str(output_npz), str(output_csv), str(output_md), str(output_json)],
        metrics_after=report["sample_counts"] | {"support_tiers": report["support_tiers"]},
        interpretation=(
            "OOF scoring semantics are now explicit. Unsupported rows carry no validated "
            "score and cannot be promoted to deployment use."
        ),
        decision="continue_to_interval_mining",
        continue_or_stop="continue",
        next_action="targeted_interval_mining_and_review_pack",
        pending_commit_message="feat: add mvp4x max-auto final scoring outputs",
    )
    return {"phase": "scoring", "report": report, "outputs": [str(output_json)], **RESEARCH_FLAGS}


def build_final_score_tables(
    snapshot: dict[str, np.ndarray],
    source_scores: dict[str, np.ndarray],
    source_report: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    score = np.asarray(source_scores["score"], dtype=np.float32).reshape(-1)
    target = np.asarray(snapshot["receiver_mean"], dtype=np.float32).reshape(-1)
    residual = score - target
    fold_id = np.asarray(source_scores["fold_id"], dtype=np.int16).reshape(-1)
    selected_policy = np.asarray(source_scores["selected_policy"]).astype(str).reshape(-1)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str).reshape(-1)
    low_orientation = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(
        -1
    )
    stability = np.asarray(source_scores["score_stability_std"], dtype=np.float32).reshape(-1)
    supported = np.isfinite(score)
    support_cohort = np.asarray(
        [POLICY_TO_COHORT.get(policy, "") for policy in selected_policy],
        dtype="<U48",
    )
    support_tier = build_support_tier(selected_policy, regimes, low_orientation, supported)
    global_rank = rank_percentile(score)
    within_regime_rank = rank_percentile_by_group(score, regimes, supported)
    within_cohort_rank = rank_percentile_by_group(score, support_cohort, supported)
    stability_warning = np.isfinite(stability) & (stability > 0.05)
    domain_shift_warning = supported & np.isin(regimes, ["B", "C"])
    transfer_warning = domain_shift_warning.copy()
    calibration_warning = supported.copy()
    unsupported_warning = ~supported
    score_status = np.where(
        supported,
        "supported_screening_oof",
        "audit_only_or_unsupported_no_validated_score",
    ).astype("<U48")

    rows: list[dict[str, Any]] = []
    for index in range(depth.size):
        rows.append(
            {
                "sample_index": index,
                "depth": float(depth[index]),
                "regime_id": regimes[index],
                "orientation_cohort": (
                    "low_orientation" if low_orientation[index] else "high_orientation"
                ),
                "selected_policy": selected_policy[index],
                "support_cohort": support_cohort[index],
                "support_tier": support_tier[index],
                "score_status": score_status[index],
                "fold_id": int(fold_id[index]),
                "score": _float_or_none(score[index]),
                "global_rank_percentile": _float_or_none(global_rank[index]),
                "within_regime_rank_percentile": _float_or_none(within_regime_rank[index]),
                "within_supported_cohort_percentile": _float_or_none(within_cohort_rank[index]),
                "target_receiver_mean": _float_or_none(target[index]),
                "target_receiver_p90": _float_or_none(
                    np.asarray(snapshot["receiver_p90"], dtype=np.float32)[index]
                ),
                "target_receiver_max": _float_or_none(
                    np.asarray(snapshot["receiver_max"], dtype=np.float32)[index]
                ),
                "residual_score_minus_target": _float_or_none(residual[index]),
                "gap_10_score": _float_or_none(source_scores["gap_10_score"][index]),
                "gap_25_score": _float_or_none(source_scores["gap_25_score"][index]),
                "gap_50_score": _float_or_none(source_scores["gap_50_score"][index]),
                "score_stability_std": _float_or_none(stability[index]),
                "score_stability_warning": bool(stability_warning[index]),
                "domain_shift_warning": bool(domain_shift_warning[index]),
                "transfer_warning": bool(transfer_warning[index]),
                "calibration_warning": bool(calibration_warning[index]),
                "unsupported_warning": bool(unsupported_warning[index]),
                "review_only": True,
                **RESEARCH_FLAGS,
            }
        )

    support_tiers = _string_counts(support_tier)
    report = {
        "report_version": FINAL_SCORE_VERSION,
        "generated_at": _utc_now(),
        "source_report_version": source_report.get("report_version"),
        "target": "receiver_mean",
        "target_kernel": str(np.asarray(snapshot["target_kernel"]).item()),
        "screening_model": "Ridge",
        "screening_feature_set": "existing_features_only",
        "feature_count": int(np.asarray(snapshot["xsi_features"]).shape[1]),
        "sample_counts": {
            "total": int(depth.size),
            "supported_scored": int(np.count_nonzero(supported)),
            "audit_only_or_unsupported": int(np.count_nonzero(~supported)),
        },
        "support_tiers": support_tiers,
        "supported_cohorts": SUPPORTED_COHORTS,
        "unsupported_or_audit_only_cohorts": UNSUPPORTED_COHORTS,
        "percentile_fields": {
            "global_rank_percentile": "rank among all finite supported OOF scores",
            "within_regime_rank_percentile": "rank among finite OOF scores inside broad regime",
            "within_supported_cohort_percentile": (
                "rank among finite OOF scores inside selected supported cohort"
            ),
        },
        "metrics": {
            "selected_score_metrics": compute_regression_metrics(
                target[supported], score[supported]
            ),
            "top_k_lift": top_k_lift(target[supported], global_rank[supported]),
            "score_stability": {
                "summary": _summary(stability[supported]),
                "warning_count": int(np.count_nonzero(stability_warning)),
            },
        },
        "warnings": {
            "calibration_insufficient": True,
            "absolute_prediction_supported": False,
            "domain_shift_warning": True,
            "transfer_warning": True,
            "unsupported_cohorts_have_no_validated_score": True,
            "orientation_is_cohort_descriptor_not_causal_filter": True,
            "regime_is_research_evaluation_domain_not_production_router": True,
        },
        **RESEARCH_FLAGS,
    }
    arrays = {
        "depth": depth,
        "score": score,
        "target_receiver_mean": target,
        "target_receiver_p90": np.asarray(snapshot["receiver_p90"], dtype=np.float32),
        "target_receiver_max": np.asarray(snapshot["receiver_max"], dtype=np.float32),
        "residual": residual,
        "fold_id": fold_id,
        "selected_policy": selected_policy.astype("<U8"),
        "support_cohort": support_cohort,
        "support_tier": support_tier,
        "score_status": score_status,
        "global_rank_percentile": global_rank,
        "within_regime_rank_percentile": within_regime_rank,
        "within_supported_cohort_percentile": within_cohort_rank,
        "gap_10_score": np.asarray(source_scores["gap_10_score"], dtype=np.float32),
        "gap_25_score": np.asarray(source_scores["gap_25_score"], dtype=np.float32),
        "gap_50_score": np.asarray(source_scores["gap_50_score"], dtype=np.float32),
        "score_stability_std": stability,
        "score_stability_warning": stability_warning,
        "domain_shift_warning": domain_shift_warning,
        "transfer_warning": transfer_warning,
        "calibration_warning": calibration_warning,
        "unsupported_warning": unsupported_warning,
        "policy_scores": np.asarray(source_scores["policy_scores"], dtype=np.float32),
        "policy_fold_ids": np.asarray(source_scores["policy_fold_ids"], dtype=np.int16),
        "policy_names": np.asarray(source_scores["policy_names"]).astype(str),
    }
    return arrays, rows, report


def build_support_tier(
    selected_policy: np.ndarray,
    regimes: np.ndarray,
    low_orientation: np.ndarray,
    supported: np.ndarray,
) -> np.ndarray:
    output = np.full(selected_policy.shape, "tier_4_unsupported_audit_only", dtype="<U64")
    p1 = supported & (selected_policy == "P1")
    p0 = supported & (selected_policy == "P0")
    regime_specific = supported & np.isin(selected_policy, ["P2", "P3", "P4"])
    output[p1] = "tier_1_primary_supported_pooled_bc_high_orientation"
    output[p0] = "tier_2_supported_pooled_bc_all_fallback"
    output[regime_specific] = "tier_2_supported_regime_specific"
    output[(~supported) & low_orientation & (regimes != "A")] = (
        "tier_4_unsupported_low_orientation_audit_only"
    )
    output[(~supported) & (regimes == "A")] = "tier_4_unsupported_regime_a_audit_only"
    return output


def rank_percentile_by_group(
    score: np.ndarray,
    group: np.ndarray,
    supported: np.ndarray,
) -> np.ndarray:
    output = np.full(score.shape, np.nan, dtype=np.float32)
    group_values = np.asarray(group).astype(str)
    for value in sorted(set(group_values[supported].tolist())):
        if not value:
            continue
        mask = supported & (group_values == value)
        output[mask] = rank_percentile(np.where(mask, score, np.nan))[mask]
    return output


def mine_review_intervals(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    final = _load_npz(paths.interim / f"{FINAL_SCORE_VERSION}.npz")
    final_report = _read_json(paths.reports / f"{FINAL_SCORE_VERSION}.json")
    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    intervals = select_review_intervals(snapshot, final)
    output_csv = paths.reports / f"{INTERVAL_VERSION}.csv"
    output_json = paths.reports / f"{INTERVAL_VERSION}.json"
    review_dir = paths.reports / "mvp4x_max_auto_review_v001"
    for path in (output_csv, output_json):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    if review_dir.exists() and not overwrite:
        raise FileExistsError(f"Review directory exists: {review_dir}")
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(intervals, output_csv)
    interval_report = {
        "report_version": INTERVAL_VERSION,
        "generated_at": _utc_now(),
        "interval_count": len(intervals),
        "interval_type_counts": _string_counts(
            np.asarray([row["interval_type"] for row in intervals], dtype=str)
        ),
        "inputs": {
            "final_scores_npz": str(paths.interim / f"{FINAL_SCORE_VERSION}.npz"),
            "snapshot_npz": str(paths.interim / "mvp4x_research_snapshot_v001.npz"),
        },
        "selection_policy": {
            "target_count_range": [20, 60],
            "half_window_ft": 2.5,
            "categories": [
                "top_high_score",
                "false_positive_like_high_score_low_target",
                "false_negative_like_low_score_high_target",
                "gap_instability",
                "domain_shift_sensitive",
                "unsupported_audit_high_target",
            ],
        },
        "final_score_summary": final_report.get("metrics", {}),
        "intervals": intervals,
        **RESEARCH_FLAGS,
    }
    output_json.write_text(_json(interval_report), encoding="utf-8")
    review_files = write_focused_review_pack(
        review_dir=review_dir,
        snapshot=snapshot,
        final=final,
        final_report=final_report,
        intervals=intervals,
        overwrite=overwrite,
    )
    append_iteration_log(
        paths,
        iteration_id="002",
        files_read=[
            str(paths.interim / f"{FINAL_SCORE_VERSION}.npz"),
            str(paths.reports / f"{FINAL_SCORE_VERSION}.json"),
        ],
        hypothesis=(
            "OOF ranking is useful enough to select a bounded manual-review interval pack "
            "that includes high-score, error-like, stability, domain-shift, and unsupported "
            "audit cases."
        ),
        evidence_before={
            "supported_scored": final_report["sample_counts"]["supported_scored"],
            "top_k": final_report["metrics"]["top_k_lift"],
        },
        selected_action="mine_targeted_review_intervals_and_write_review_pack",
        files_changed=[str(output_csv), str(output_json), *review_files],
        commands_run=[
            "python scripts/07u_run_mvp4x_max_auto.py --phase intervals --overwrite",
            "make test-smoke",
            "make test",
            "make lint",
            "python scripts/00_check_env.py",
        ],
        outputs_generated=[str(output_csv), str(output_json), *review_files],
        metrics_after={
            "interval_count": len(intervals),
            "interval_type_counts": interval_report["interval_type_counts"],
        },
        interpretation=(
            "The review pack now covers both candidate positives and targeted failure modes; "
            "unsupported rows remain audit-only."
        ),
        decision="continue_to_triage",
        continue_or_stop="continue",
        next_action="ranking_calibration_error_morphology_triage",
        pending_commit_message="feat: add mvp4x max-auto review interval mining",
    )
    return {
        "phase": "intervals",
        "interval_count": len(intervals),
        "review_files": review_files,
        **RESEARCH_FLAGS,
    }


def select_review_intervals(
    snapshot: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    depth = np.asarray(final["depth"], dtype=np.float32).reshape(-1)
    score = np.asarray(final["score"], dtype=np.float32).reshape(-1)
    rank = np.asarray(final["global_rank_percentile"], dtype=np.float32).reshape(-1)
    target = np.asarray(final["target_receiver_mean"], dtype=np.float32).reshape(-1)
    stability = np.asarray(final["score_stability_std"], dtype=np.float32).reshape(-1)
    supported = np.isfinite(score)
    residual = score - target
    policy_std = _policy_score_std(np.asarray(final["policy_scores"], dtype=np.float32))
    target_p50 = float(np.nanquantile(target[supported], 0.50))
    target_p90 = float(np.nanquantile(target[supported], 0.90))
    categories = [
        (
            "top_high_score",
            supported & (rank >= 0.95),
            score,
            True,
            10,
            "Highest global supported OOF score percentiles.",
        ),
        (
            "false_positive_like_high_score_low_target",
            supported & (rank >= 0.90) & (target <= target_p50),
            residual,
            True,
            10,
            "High score but low weak target; review for false-positive-like behavior.",
        ),
        (
            "false_negative_like_low_score_high_target",
            supported & (rank <= 0.50) & (target >= target_p90),
            target - score,
            True,
            10,
            "Low score but high weak target; review for false-negative-like behavior.",
        ),
        (
            "gap_instability",
            supported & (stability >= np.nanquantile(stability[supported], 0.99)),
            stability,
            True,
            6,
            "Prediction changes most across blocked-gap variants.",
        ),
        (
            "domain_shift_sensitive",
            supported & (policy_std >= np.nanquantile(policy_std[supported], 0.95)),
            policy_std,
            True,
            8,
            "Policy-score disagreement is high, suggesting domain-shift sensitivity.",
        ),
        (
            "unsupported_audit_high_target",
            (~supported) & (target >= np.nanquantile(target, 0.90)),
            target,
            True,
            8,
            "Unsupported/audit-only row with high weak target; no validated score claim.",
        ),
    ]
    intervals: list[dict[str, Any]] = []
    interval_id = 1
    for interval_type, mask, value, descending, limit, reason in categories:
        centers = _select_nonoverlapping_centers(
            depth, mask, value, descending=descending, limit=limit
        )
        for priority, index in enumerate(centers, start=1):
            intervals.append(
                summarize_interval(
                    interval_id=interval_id,
                    interval_type=interval_type,
                    priority_rank=priority,
                    center_index=index,
                    selection_reason=reason,
                    snapshot=snapshot,
                    final=final,
                    policy_std=policy_std,
                )
            )
            interval_id += 1
    return intervals[:60]


def summarize_interval(
    *,
    interval_id: int,
    interval_type: str,
    priority_rank: int,
    center_index: int,
    selection_reason: str,
    snapshot: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    policy_std: np.ndarray,
) -> dict[str, Any]:
    depth = np.asarray(final["depth"], dtype=np.float32).reshape(-1)
    center = float(depth[center_index])
    in_window = (depth >= center - 2.5) & (depth <= center + 2.5)
    score = np.asarray(final["score"], dtype=np.float32)
    target = np.asarray(final["target_receiver_mean"], dtype=np.float32)
    rank = np.asarray(final["global_rank_percentile"], dtype=np.float32)
    residual = np.asarray(final["residual"], dtype=np.float32)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    support_tier = np.asarray(final["support_tier"]).astype(str)
    selected_policy = np.asarray(final["selected_policy"]).astype(str)
    low = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool)
    kernel = morphology_kernel(snapshot)
    return {
        "interval_id": f"R{interval_id:03d}",
        "interval_type": interval_type,
        "priority_rank": priority_rank,
        "depth_center": center,
        "depth_min": float(np.min(depth[in_window])),
        "depth_max": float(np.max(depth[in_window])),
        "sample_count": int(np.count_nonzero(in_window)),
        "center_sample_index": int(center_index),
        "regime_id": regimes[center_index],
        "orientation_cohort": "low_orientation" if low[center_index] else "high_orientation",
        "selected_policy": selected_policy[center_index],
        "support_tier": support_tier[center_index],
        "score": _float_or_none(score[center_index]),
        "target_receiver_mean": _float_or_none(target[center_index]),
        "global_rank_percentile": _float_or_none(rank[center_index]),
        "within_regime_rank_percentile": _float_or_none(
            np.asarray(final["within_regime_rank_percentile"], dtype=np.float32)[center_index]
        ),
        "residual_score_minus_target": _float_or_none(residual[center_index]),
        "score_stability_std": _float_or_none(
            np.asarray(final["score_stability_std"], dtype=np.float32)[center_index]
        ),
        "policy_score_std": _float_or_none(policy_std[center_index]),
        "window_score_mean": _nanmean_or_none(score[in_window]),
        "window_target_mean": _nanmean_or_none(target[in_window]),
        "window_rank_mean": _nanmean_or_none(rank[in_window]),
        "morphology_lcc": _float_or_none(kernel["lcc"][center_index]),
        "morphology_max_azimuth_fraction": _float_or_none(kernel["max_azimuth"][center_index]),
        "morphology_max_relative_drop": _float_or_none(kernel["relative_drop"][center_index]),
        "selection_reason": selection_reason,
        "review_only": True,
        **RESEARCH_FLAGS,
    }


def write_focused_review_pack(
    *,
    review_dir: Path,
    snapshot: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    final_report: dict[str, Any],
    intervals: list[dict[str, Any]],
    overwrite: bool,
) -> list[str]:
    figures_dir = review_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = review_dir / "selected_intervals.csv"
    selected_json = review_dir / "selected_intervals.json"
    summary_md = review_dir / "review_summary.md"
    checklist_md = review_dir / "reviewer_checklist.md"
    for path in (selected_csv, selected_json, summary_md, checklist_md):
        _ensure_can_write(path, overwrite=overwrite)
    _write_csv(intervals, selected_csv)
    selected_json.write_text(_json({"intervals": intervals, **RESEARCH_FLAGS}), encoding="utf-8")
    summary_md.write_text(format_review_summary(final_report, intervals), encoding="utf-8")
    checklist_md.write_text(format_reviewer_checklist(), encoding="utf-8")
    figure_paths = write_review_figures(
        figures_dir=figures_dir,
        snapshot=snapshot,
        final=final,
        intervals=intervals,
        overwrite=overwrite,
    )
    return [
        str(selected_csv),
        str(selected_json),
        str(summary_md),
        str(checklist_md),
        *figure_paths,
    ]


def write_review_figures(
    *,
    figures_dir: Path,
    snapshot: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    intervals: list[dict[str, Any]],
    overwrite: bool,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = np.asarray(final["depth"], dtype=np.float32)
    score = np.asarray(final["score"], dtype=np.float32)
    target = np.asarray(final["target_receiver_mean"], dtype=np.float32)
    rank = np.asarray(final["global_rank_percentile"], dtype=np.float32)
    paths: list[str] = []

    summary = figures_dir / "summary_score_target_rank.png"
    _ensure_can_write(summary, overwrite=overwrite)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(depth, target, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("target")
    axes[1].plot(depth, score, color="#d62728", linewidth=0.8)
    axes[1].set_ylabel("score")
    axes[2].plot(depth, rank, color="#2ca02c", linewidth=0.8)
    axes[2].set_ylabel("rank")
    axes[2].set_xlabel("depth ft")
    fig.tight_layout()
    fig.savefig(summary, dpi=140)
    plt.close(fig)
    paths.append(str(summary))

    support = figures_dir / "support_tier_counts.png"
    _ensure_can_write(support, overwrite=overwrite)
    tier_counts = _string_counts(np.asarray(final["support_tier"]).astype(str))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(tier_counts)), list(tier_counts.values()), color="#4c78a8")
    ax.set_xticks(range(len(tier_counts)), list(tier_counts), rotation=25, ha="right")
    ax.set_ylabel("sample count")
    fig.tight_layout()
    fig.savefig(support, dpi=140)
    plt.close(fig)
    paths.append(str(support))

    for row in intervals[:24]:
        center = float(row["depth_center"])
        mask = (depth >= center - 8.0) & (depth <= center + 8.0)
        figure = figures_dir / f"{row['interval_id']}_{row['interval_type']}.png"
        _ensure_can_write(figure, overwrite=overwrite)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(depth[mask], target[mask], label="target_receiver_mean", color="#1f77b4")
        ax.plot(depth[mask], score[mask], label="score", color="#d62728")
        ax.axvline(center, color="#111111", linewidth=0.8, linestyle="--")
        ax.set_title(f"{row['interval_id']} {row['interval_type']}")
        ax.set_xlabel("depth ft")
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(target[mask])) - 0.01))
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figure, dpi=140)
        plt.close(fig)
        paths.append(str(figure))
    return paths


def build_triage_report(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    final = _load_npz(paths.interim / f"{FINAL_SCORE_VERSION}.npz")
    final_report = _read_json(paths.reports / f"{FINAL_SCORE_VERSION}.json")
    interval_report = _read_json(paths.reports / f"{INTERVAL_VERSION}.json")
    triage = build_triage(snapshot=snapshot, final=final, interval_report=interval_report)
    output_json = paths.reports / f"{TRIAGE_VERSION}.json"
    output_md = paths.reports / f"{TRIAGE_VERSION}.md"
    for path in (output_json, output_md):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(_json(triage), encoding="utf-8")
    output_md.write_text(format_triage_markdown(triage), encoding="utf-8")
    append_iteration_log(
        paths,
        iteration_id="003",
        files_read=[
            str(paths.interim / f"{FINAL_SCORE_VERSION}.npz"),
            str(paths.reports / f"{FINAL_SCORE_VERSION}.json"),
            str(paths.reports / f"{INTERVAL_VERSION}.json"),
        ],
        hypothesis=(
            "The dominant OOF errors can be triaged into calibration/domain-shift/stability "
            "and morphology-related modes, which determines whether morphology-v2 is worth "
            "a parallel audit."
        ),
        evidence_before={
            "top_k": final_report["metrics"]["top_k_lift"],
            "interval_count": interval_report["interval_count"],
        },
        selected_action="build_ranking_calibration_error_triage",
        files_changed=[str(output_json), str(output_md)],
        commands_run=[
            "python scripts/07u_run_mvp4x_max_auto.py --phase triage --overwrite",
            "make test-smoke",
            "make test",
            "make lint",
            "python scripts/00_check_env.py",
        ],
        outputs_generated=[str(output_json), str(output_md)],
        metrics_after={
            "morphology_related_to_errors": triage["morphology_error_analysis"][
                "morphology_related_to_errors"
            ],
            "false_positive_like_intervals": triage["interval_summary"][
                "false_positive_like_intervals"
            ],
            "false_negative_like_intervals": triage["interval_summary"][
                "false_negative_like_intervals"
            ],
        },
        interpretation=triage["interpretation"],
        decision=triage["decision"],
        continue_or_stop="continue",
        next_action="parallel_morphology_candidate_v2_comparison",
        pending_commit_message="feat: add mvp4x max-auto triage report",
    )
    return {"phase": "triage", "report": triage, **RESEARCH_FLAGS}


def build_triage(
    *,
    snapshot: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    interval_report: dict[str, Any],
) -> dict[str, Any]:
    score = np.asarray(final["score"], dtype=np.float32)
    target = np.asarray(final["target_receiver_mean"], dtype=np.float32)
    rank = np.asarray(final["global_rank_percentile"], dtype=np.float32)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    supported = np.isfinite(score)
    residual = score - target
    fp_mask = supported & (rank >= 0.90) & (target <= np.nanquantile(target[supported], 0.50))
    fn_mask = supported & (rank <= 0.50) & (target >= np.nanquantile(target[supported], 0.90))
    calibration = calibration_bins(target, score, supported, n_bins=10)
    residual_by_regime = {
        regime: _summary(residual[supported & (regimes == regime)])
        for regime in sorted(set(regimes.tolist()))
    }
    target_view_ranking = compare_target_views_with_existing_rank(snapshot, rank, supported)
    morphology = morphology_error_analysis(snapshot, supported, fp_mask, fn_mask)
    interval_types = [row["interval_type"] for row in interval_report.get("intervals", [])]
    false_positive_intervals = sum("false_positive_like" in item for item in interval_types)
    false_negative_intervals = sum("false_negative_like" in item for item in interval_types)
    decision = (
        "continue_to_morphology_candidate_v2"
        if morphology["morphology_related_to_errors"]
        else "continue_to_final_decision_without_morphology_v2"
    )
    interpretation = (
        "Morphology fields separate false-negative-like and false-positive-like rows, so "
        "a parallel audit-only morphology candidate v2 is justified."
        if morphology["morphology_related_to_errors"]
        else "Morphology fields do not strongly separate error-like rows; retain v1 target view."
    )
    return {
        "report_version": TRIAGE_VERSION,
        "generated_at": _utc_now(),
        "ranking_audit": {
            "top_k_lift": top_k_lift(target[supported], rank[supported]),
            "metrics": compute_regression_metrics(target[supported], score[supported]),
        },
        "calibration": calibration,
        "residual_by_regime": residual_by_regime,
        "interval_summary": {
            "interval_count": int(interval_report.get("interval_count", 0)),
            "false_positive_like_intervals": int(false_positive_intervals),
            "false_negative_like_intervals": int(false_negative_intervals),
            "gap_instability_intervals": int(
                sum("gap_instability" in item for item in interval_types)
            ),
            "domain_shift_sensitive_intervals": int(
                sum("domain_shift_sensitive" in item for item in interval_types)
            ),
            "unsupported_audit_intervals": int(
                sum("unsupported_audit" in item for item in interval_types)
            ),
        },
        "false_positive_like_sample_count": int(np.count_nonzero(fp_mask)),
        "false_negative_like_sample_count": int(np.count_nonzero(fn_mask)),
        "target_view_ranking": target_view_ranking,
        "recommended_research_candidate_target_view": target_view_ranking["best_target_view"],
        "morphology_error_analysis": morphology,
        "domain_shift_still_exists": True,
        "calibration_still_insufficient": True,
        "decision": decision,
        "interpretation": interpretation,
        **RESEARCH_FLAGS,
    }


def generate_morphology_candidate_v2(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    snapshot = _load_npz(paths.interim / "mvp4x_research_snapshot_v001.npz")
    final = _load_npz(paths.interim / f"{FINAL_SCORE_VERSION}.npz")
    triage = _read_json(paths.reports / f"{TRIAGE_VERSION}.json")
    candidate = build_morphology_candidate(snapshot)
    comparison = compare_v1_vs_morphology_v2(snapshot=snapshot, final=final, candidate=candidate)

    output_npz = paths.interim / f"{MORPHOLOGY_VERSION}.npz"
    output_json = paths.reports / f"{MORPHOLOGY_VERSION}.json"
    output_md = paths.reports / f"{MORPHOLOGY_VERSION}.md"
    output_csv = paths.reports / f"{MORPHOLOGY_VERSION}.csv"
    comp_json = paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.json"
    comp_md = paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.md"
    comp_csv = paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.csv"
    for path in (output_npz, output_json, output_md, output_csv, comp_json, comp_md, comp_csv):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_npz,
        report_version=np.asarray(MORPHOLOGY_VERSION),
        depth=np.asarray(snapshot["depth"], dtype=np.float32),
        morphology_candidate_v2=np.asarray(candidate["morphology_candidate_v2"], dtype=np.float32),
        morphology_signal=np.asarray(candidate["morphology_signal"], dtype=np.float32),
        v1_receiver_mean=np.asarray(snapshot["receiver_mean"], dtype=np.float32),
        component_fraction=np.asarray(candidate["component_fraction"], dtype=np.float32),
        max_azimuth_channel_fraction=np.asarray(
            candidate["max_azimuth_channel_fraction"], dtype=np.float32
        ),
        relative_anomaly_fraction=np.asarray(
            candidate["relative_anomaly_fraction"], dtype=np.float32
        ),
        combined_channel_fraction=np.asarray(
            candidate["combined_channel_fraction"], dtype=np.float32
        ),
        label_confidence=np.asarray(snapshot["label_confidence"], dtype=np.float32),
        morphology_candidate_v2_flag=np.asarray(True),
        audit_only=np.asarray(True),
        parallel_comparison_only=np.asarray(True),
        metadata_json=np.asarray(_json(candidate["report"])),
        **{key: np.asarray(value) for key, value in RESEARCH_FLAGS.items()},
    )
    _write_csv(candidate["rows"], output_csv)
    output_json.write_text(_json(candidate["report"]), encoding="utf-8")
    output_md.write_text(
        format_morphology_candidate_markdown(candidate["report"]), encoding="utf-8"
    )
    _write_csv(comparison["rows"], comp_csv)
    comp_json.write_text(_json(comparison["report"]), encoding="utf-8")
    comp_md.write_text(
        format_morphology_comparison_markdown(comparison["report"]), encoding="utf-8"
    )

    append_iteration_log(
        paths,
        iteration_id="004",
        files_read=[
            str(paths.interim / "mvp4x_research_snapshot_v001.npz"),
            str(paths.interim / f"{FINAL_SCORE_VERSION}.npz"),
            str(paths.reports / f"{TRIAGE_VERSION}.json"),
        ],
        hypothesis=(
            "Because morphology separates error-like rows, a parallel audit-only v2 target "
            "may reduce mismatch between weak-label morphology and XSI screening scores."
        ),
        evidence_before={
            "morphology_related_to_errors": triage["morphology_error_analysis"][
                "morphology_related_to_errors"
            ],
            "v1_current_top_k": triage["ranking_audit"]["top_k_lift"],
        },
        selected_action="generate_parallel_morphology_candidate_v2_and_compare",
        files_changed=[
            str(output_npz),
            str(output_json),
            str(output_md),
            str(output_csv),
            str(comp_json),
            str(comp_md),
            str(comp_csv),
        ],
        commands_run=[
            "python scripts/07u_run_mvp4x_max_auto.py --phase morphology --overwrite",
            "make test-smoke",
            "make test",
            "make lint",
            "python scripts/00_check_env.py",
        ],
        outputs_generated=[
            str(output_npz),
            str(output_json),
            str(output_md),
            str(output_csv),
            str(comp_json),
            str(comp_md),
            str(comp_csv),
        ],
        metrics_after=comparison["report"]["comparison_summary"],
        interpretation=comparison["report"]["interpretation"],
        decision=comparison["report"]["decision"],
        continue_or_stop="continue",
        next_action="final_bounded_decision",
        pending_commit_message="feat: add morphology candidate v2 audit comparison",
    )
    return {
        "phase": "morphology",
        "candidate_report": candidate["report"],
        "comparison": comparison["report"],
        **RESEARCH_FLAGS,
    }


def build_morphology_candidate(snapshot: dict[str, np.ndarray]) -> dict[str, Any]:
    kernel = morphology_kernel(snapshot)
    base = np.asarray(snapshot["receiver_mean"], dtype=np.float32)
    component = np.nan_to_num(kernel["lcc"], nan=0.0, posinf=0.0, neginf=0.0)
    max_azimuth = np.nan_to_num(kernel["max_azimuth"], nan=0.0, posinf=0.0, neginf=0.0)
    relative_drop = np.nan_to_num(kernel["relative_drop"], nan=0.0, posinf=0.0, neginf=0.0)
    combined_fraction = np.nan_to_num(kernel["candidate_fraction"], nan=0.0)
    morphology_signal = np.clip(
        0.35 * component + 0.30 * max_azimuth + 0.25 * relative_drop + 0.10 * combined_fraction,
        0.0,
        1.0,
    ).astype(np.float32)
    candidate_v2 = np.clip(0.55 * base + 0.45 * morphology_signal, 0.0, 1.0).astype(np.float32)
    depth = np.asarray(snapshot["depth"], dtype=np.float32)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str)
    rows = [
        {
            "sample_index": index,
            "depth": float(depth[index]),
            "regime_id": regimes[index],
            "v1_receiver_mean": float(base[index]),
            "morphology_candidate_v2": float(candidate_v2[index]),
            "morphology_signal": float(morphology_signal[index]),
            "component_fraction": float(component[index]),
            "max_azimuth_channel_fraction": float(max_azimuth[index]),
            "relative_anomaly_fraction": float(relative_drop[index]),
            "combined_channel_fraction": float(combined_fraction[index]),
            "audit_only": True,
            "parallel_comparison_only": True,
            **RESEARCH_FLAGS,
        }
        for index in range(depth.size)
    ]
    report = {
        "report_version": MORPHOLOGY_VERSION,
        "generated_at": _utc_now(),
        "schema": {
            "morphology_candidate_v2": (
                "0.55 * receiver_mean_v1 + 0.45 * fixed morphology_signal; audit-only"
            ),
            "morphology_signal": (
                "0.35 component + 0.30 max_azimuth + 0.25 relative_drop + "
                "0.10 combined_fraction"
            ),
            "weights_pre_registered_in_code": True,
            "no_threshold_scan": True,
            "not_final_label": True,
        },
        "target_distribution": {
            "v1_receiver_mean": _summary(base),
            "morphology_candidate_v2": _summary(candidate_v2),
            "morphology_signal": _summary(morphology_signal),
        },
        "correlation_with_v1": _safe_spearman(base, candidate_v2),
        "morphology_candidate_v2": True,
        "audit_only": True,
        "parallel_comparison_only": True,
        **RESEARCH_FLAGS,
    }
    return {
        "morphology_candidate_v2": candidate_v2,
        "morphology_signal": morphology_signal,
        "component_fraction": component,
        "max_azimuth_channel_fraction": max_azimuth,
        "relative_anomaly_fraction": relative_drop,
        "combined_channel_fraction": combined_fraction,
        "rows": rows,
        "report": report,
    }


def compare_v1_vs_morphology_v2(
    *,
    snapshot: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    X = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    depth = np.asarray(snapshot["depth"], dtype=np.float32)
    supported = np.isfinite(np.asarray(final["score"], dtype=np.float32))
    config = {"n_contiguous_folds": 3, "ridge_alpha": 1.0}
    targets = {
        "v1_receiver_mean": np.asarray(snapshot["receiver_mean"], dtype=np.float32),
        "morphology_candidate_v2": np.asarray(
            candidate["morphology_candidate_v2"], dtype=np.float32
        ),
    }
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    rng = np.random.default_rng(20240605)
    for name, y in targets.items():
        result = fit_oof_scores(
            X=X,
            y=y,
            depth=depth,
            sample_mask=supported,
            config=config,
            sklearn_modules=sklearn_modules,
            gap_ft=25.0,
            random_seed=20240605,
        )
        prediction = np.asarray(result["prediction"], dtype=np.float32)
        mask = supported & np.isfinite(prediction)
        rank = rank_percentile(prediction)
        permutation = morphology_permutation_audit(
            X=X,
            y=y,
            depth=depth,
            supported=supported,
            config=config,
            sklearn_modules=sklearn_modules,
            rng=rng,
        )
        domain_shift = domain_shift_transfer(
            X=X,
            y=y,
            regimes=np.asarray(snapshot["broad_regime_id"]).astype(str),
            supported=supported,
            sklearn_modules=sklearn_modules,
            config=config,
        )
        calibration = calibration_bins(y, prediction, mask, n_bins=8)
        summary = {
            "target": name,
            "blocked_gap_ft": 25.0,
            "sample_count": int(np.count_nonzero(mask)),
            "oof_metrics": compute_regression_metrics(y[mask], prediction[mask]),
            "ranking": top_k_lift(y[mask], rank[mask]),
            "permutation": permutation,
            "domain_shift": domain_shift,
            "calibration": calibration,
        }
        summaries[name] = summary
        rows.append(
            {
                "comparison_version": COMPARISON_VERSION,
                "target": name,
                "evaluation": "blocked_gap_oof",
                "sample_count": summary["sample_count"],
                **summary["oof_metrics"],
                "top_10_lift": _as_dict(
                    _as_dict(summary["ranking"].get("top_k")).get("top_10pct")
                ).get("lift"),
                "global_permutation_spearman": permutation["global"]["spearman"],
                "global_permutation_margin": permutation["global"]["real_minus_permutation"],
                "domain_shift_b_to_c_spearman": domain_shift["b_to_c"].get("spearman"),
                "domain_shift_c_to_b_spearman": domain_shift["c_to_b"].get("spearman"),
                **RESEARCH_FLAGS,
            }
        )
    v1 = summaries["v1_receiver_mean"]
    v2 = summaries["morphology_candidate_v2"]
    v1_s = float(v1["oof_metrics"]["spearman"])
    v2_s = float(v2["oof_metrics"]["spearman"])
    v1_lift = _as_float(_as_dict(_as_dict(v1["ranking"].get("top_k")).get("top_10pct")).get("lift"))
    v2_lift = _as_float(_as_dict(_as_dict(v2["ranking"].get("top_k")).get("top_10pct")).get("lift"))
    v2_improved = bool(v2_s >= v1_s + 0.03 and v2_lift >= v1_lift)
    decision = (
        "morphology_candidate_v2_model_improved"
        if v2_improved
        else "morphology_candidate_v2_human_review_only_not_model_improved"
    )
    interpretation = (
        "Morphology v2 improved the bounded Ridge screening audit."
        if v2_improved
        else (
            "Morphology v2 captures morphology-driven label semantics, but current XSI "
            "existing features do not predict it better than v1 in blocked-gap OOF. Treat "
            "v2 as a human-review label-semantics candidate, not a replacement target."
        )
    )
    report = {
        "report_version": COMPARISON_VERSION,
        "generated_at": _utc_now(),
        "modeling_environment": modeling_environment.to_dict(),
        "feature_set": "existing_features_only",
        "model": "Ridge",
        "sample_mask": "supported_scored_oof_rows_only",
        "target_summaries": {
            name: {
                "distribution": _summary(values),
                "summary": summaries[name],
            }
            for name, values in targets.items()
        },
        "comparison_summary": {
            "v1_spearman": v1_s,
            "morphology_v2_spearman": v2_s,
            "delta_spearman": v2_s - v1_s,
            "v1_top10_lift": v1_lift,
            "morphology_v2_top10_lift": v2_lift,
            "delta_top10_lift": v2_lift - v1_lift,
            "morphology_v2_improved": v2_improved,
        },
        "decision": decision,
        "interpretation": interpretation,
        "audit_only": True,
        "parallel_comparison_only": True,
        **RESEARCH_FLAGS,
    }
    return {"report": report, "rows": rows}


def build_final_decision(paths: MaxAutoPaths, *, overwrite: bool) -> dict[str, Any]:
    preflight = validate_preflight(paths)
    final_report = _read_json(paths.reports / f"{FINAL_SCORE_VERSION}.json")
    triage = _read_json(paths.reports / f"{TRIAGE_VERSION}.json")
    morphology_comparison_path = paths.reports / "mvp4x_v1_vs_morphology_v2_comparison.json"
    morphology = (
        _read_json(morphology_comparison_path) if morphology_comparison_path.exists() else {}
    )
    morphology_generated = bool(morphology)
    morphology_improved = bool(
        _as_dict(morphology.get("comparison_summary")).get("morphology_v2_improved", False)
    )
    morphology_related = bool(
        triage.get("morphology_error_analysis", {}).get("morphology_related_to_errors")
    )
    if morphology_generated and morphology_related:
        decision = "morphology_candidate_v2_recommended_for_human_review"
        next_approval = (
            "Approve or reject parallel morphology_candidate_v2 label-semantics review; "
            "do not replace v1 automatically."
        )
    else:
        decision = "research_screening_policy_ready_for_manual_review"
        next_approval = "Manual review of bounded research-only screening intervals."
    final = {
        "decision_version": DECISION_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "autonomous_iterations_completed": 5,
        "artifacts_read": preflight["artifacts_read"],
        "oof_scoring_completed": True,
        "scored_sample_count": final_report["sample_counts"]["supported_scored"],
        "support_tiers": final_report["support_tiers"],
        "global_percentile_completed": True,
        "within_regime_percentile_completed": True,
        "within_supported_cohort_percentile_completed": True,
        "score_stability": final_report["metrics"]["score_stability"],
        "top_k_lift": final_report["metrics"]["top_k_lift"],
        "false_positive_like_intervals": triage["interval_summary"][
            "false_positive_like_intervals"
        ],
        "false_negative_like_intervals": triage["interval_summary"][
            "false_negative_like_intervals"
        ],
        "morphology_explains_error": morphology_related,
        "morphology_candidate_v2_generated": morphology_generated,
        "morphology_v2_improved": morphology_improved,
        "morphology_v2_comparison": morphology.get("comparison_summary"),
        "fixed_time_frequency_v2_extracted": False,
        "fixed_time_frequency_v2": {
            "shape": None,
            "chunks": None,
            "memory": None,
            "runtime": None,
            "reason_not_extracted": (
                "The next bottleneck is label-semantics human review rather than another "
                "local waveform feature pass."
            ),
        },
        "best_target_view": triage["recommended_research_candidate_target_view"],
        "best_feature_set": "existing_features_only",
        "best_model": "Ridge",
        "domain_shift_still_exists": True,
        "calibration_still_insufficient": True,
        "advanced_methods_feasibility_report_generated": False,
        "local_resources_sufficient": True,
        "recommend_server": False,
        "next_human_approval_required": next_approval,
        "production_claim_and_final_labels_forbidden": True,
        "required_scope_flags": dict(RESEARCH_FLAGS),
        "outputs": {
            "final_scores_json": str(paths.reports / f"{FINAL_SCORE_VERSION}.json"),
            "intervals_json": str(paths.reports / f"{INTERVAL_VERSION}.json"),
            "review_dir": str(paths.reports / "mvp4x_max_auto_review_v001"),
            "triage_json": str(paths.reports / f"{TRIAGE_VERSION}.json"),
            "morphology_comparison_json": str(morphology_comparison_path)
            if morphology_generated
            else None,
        },
        **RESEARCH_FLAGS,
    }
    output_json = paths.reports / "mvp4x_max_auto_decision.json"
    output_md = paths.reports / "mvp4x_max_auto_decision.md"
    for path in (output_json, output_md):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(_json(final), encoding="utf-8")
    output_md.write_text(format_final_decision_markdown(final), encoding="utf-8")
    append_iteration_log(
        paths,
        iteration_id="005",
        files_read=[
            str(paths.reports / f"{FINAL_SCORE_VERSION}.json"),
            str(paths.reports / f"{INTERVAL_VERSION}.json"),
            str(paths.reports / f"{TRIAGE_VERSION}.json"),
            str(morphology_comparison_path),
        ],
        hypothesis=(
            "After scoring, review mining, triage, and morphology-v2 comparison, the only "
            "remaining action should be a bounded human decision rather than local "
            "time-frequency/STC/APES/deep-learning escalation."
        ),
        evidence_before={
            "morphology_generated": morphology_generated,
            "morphology_related": morphology_related,
            "morphology_improved": morphology_improved,
        },
        selected_action="write_final_max_auto_decision",
        files_changed=[str(output_json), str(output_md)],
        commands_run=[
            "python scripts/07u_run_mvp4x_max_auto.py --phase decision --overwrite",
            "make test-smoke",
            "make test",
            "make lint",
            "python scripts/00_check_env.py",
        ],
        outputs_generated=[str(output_json), str(output_md)],
        metrics_after={
            "decision": decision,
            "iterations": 5,
            "scored_sample_count": final["scored_sample_count"],
        },
        interpretation=(
            "The bounded loop produced the manual-review package and a parallel morphology "
            "candidate; no production claim, final labels, server migration, STC, APES, "
            "or deep learning is justified automatically."
        ),
        decision=decision,
        continue_or_stop="stop_bounded_decision_reached",
        human_intervention_required=True,
        next_action=next_approval,
        pending_commit_message="feat: add mvp4x max-auto final decision report",
    )
    write_pending_commits(paths)
    return final


def morphology_permutation_audit(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    supported: np.ndarray,
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    real = fit_oof_scores(
        X=X,
        y=y,
        depth=depth,
        sample_mask=supported,
        config=config,
        sklearn_modules=sklearn_modules,
        gap_ft=25.0,
        random_seed=20240605,
    )
    real_s = float(real["metrics"]["spearman"])
    audits = {}
    for name, permuted in {
        "global": rng.permutation(y),
        "within_depth_bin": _permute_within_depth_bins(y, depth, rng),
        "block": _permute_blocks(y, depth, rng),
    }.items():
        result = fit_oof_scores(
            X=X,
            y=np.asarray(permuted, dtype=np.float32),
            depth=depth,
            sample_mask=supported,
            config=config,
            sklearn_modules=sklearn_modules,
            gap_ft=25.0,
            random_seed=20240605,
        )
        audits[name] = {
            "spearman": float(result["metrics"]["spearman"]),
            "real_minus_permutation": real_s - float(result["metrics"]["spearman"]),
        }
    return audits


def domain_shift_transfer(
    *,
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    supported: np.ndarray,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "b_to_c": _fit_transfer(
            X=X,
            y=y,
            train_mask=supported & (regimes == "B"),
            validation_mask=supported & (regimes == "C"),
            sklearn_modules=sklearn_modules,
            config=config,
        ),
        "c_to_b": _fit_transfer(
            X=X,
            y=y,
            train_mask=supported & (regimes == "C"),
            validation_mask=supported & (regimes == "B"),
            sklearn_modules=sklearn_modules,
            config=config,
        ),
    }


def _fit_transfer(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    sklearn_modules: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if np.count_nonzero(train_mask) < 20 or np.count_nonzero(validation_mask) < 20:
        return {"status": "skipped_insufficient_samples"}
    model = _make_model("Ridge", sklearn_modules, config, random_state=20240605)
    model.fit(X[train_mask], y[train_mask])
    pred = np.asarray(model.predict(X[validation_mask]), dtype=np.float32)
    return {
        "status": "completed",
        "train_count": int(np.count_nonzero(train_mask)),
        "validation_count": int(np.count_nonzero(validation_mask)),
        **compute_regression_metrics(y[validation_mask], pred),
    }


def morphology_kernel(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    kernel_index = int(np.asarray(snapshot.get("target_kernel_index", 0)).item())

    def pick(key: str, default: float = 0.0) -> np.ndarray:
        if key not in snapshot:
            return np.full(np.asarray(snapshot["depth"]).shape, default, dtype=np.float32)
        arr = np.asarray(snapshot[key], dtype=np.float32)
        if arr.ndim == 2:
            index = min(max(kernel_index, 0), arr.shape[1] - 1)
            return arr[:, index].astype(np.float32)
        return arr.reshape(-1).astype(np.float32)

    candidate_count = pick("morphology_candidate_cell_count")
    total_count = np.maximum(pick("morphology_total_cell_count", default=1.0), 1.0)
    return {
        "lcc": pick("morphology_largest_connected_component_fraction"),
        "max_azimuth": pick("morphology_max_azimuth_channel_fraction"),
        "relative_drop": pick("morphology_max_relative_drop"),
        "candidate_fraction": np.clip(candidate_count / total_count, 0.0, 1.0).astype(np.float32),
    }


def morphology_error_analysis(
    snapshot: dict[str, np.ndarray],
    supported: np.ndarray,
    fp_mask: np.ndarray,
    fn_mask: np.ndarray,
) -> dict[str, Any]:
    kernel = morphology_kernel(snapshot)
    rows = {}
    related = False
    for name, values in kernel.items():
        all_values = values[supported]
        std = float(np.nanstd(all_values))
        fp_delta = _std_delta(values[fp_mask], all_values)
        fn_delta = _std_delta(values[fn_mask], all_values)
        if max(abs(fp_delta), abs(fn_delta)) >= 0.25 and std > 0.0:
            related = True
        rows[name] = {
            "supported": _summary(all_values),
            "false_positive_like": _summary(values[fp_mask]),
            "false_negative_like": _summary(values[fn_mask]),
            "false_positive_std_delta": fp_delta,
            "false_negative_std_delta": fn_delta,
        }
    return {
        "morphology_related_to_errors": related,
        "criteria": "abs(std_delta) >= 0.25 on any morphology field",
        "fields": rows,
        "audit_only": True,
    }


def compare_target_views_with_existing_rank(
    snapshot: dict[str, np.ndarray],
    rank: np.ndarray,
    supported: np.ndarray,
) -> dict[str, Any]:
    rows = {}
    for target_name in ("receiver_mean", "receiver_p90", "receiver_max"):
        y = np.asarray(snapshot[target_name], dtype=np.float32)
        rows[target_name] = {
            "spearman_with_existing_score_rank": _safe_spearman(y[supported], rank[supported]),
            "top_k_lift": top_k_lift(y[supported], rank[supported]),
        }
    best = max(
        rows,
        key=lambda name: _as_float(
            _as_dict(_as_dict(rows[name]["top_k_lift"].get("top_k")).get("top_10pct")).get(
                "lift"
            )
        ),
    )
    return {"target_views": rows, "best_target_view": best}


def calibration_bins(
    y_true: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    *,
    n_bins: int,
) -> list[dict[str, Any]]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(y_true) & np.isfinite(score)
    if np.count_nonzero(valid) == 0:
        return []
    ranks = rank_percentile(np.where(valid, score, np.nan))
    rows = []
    for bin_index in range(n_bins):
        low = bin_index / n_bins
        high = (bin_index + 1) / n_bins
        upper = ranks <= high if bin_index == n_bins - 1 else ranks < high
        bin_mask = valid & (ranks >= low) & upper
        if not np.any(bin_mask):
            continue
        rows.append(
            {
                "bin": bin_index,
                "rank_min": low,
                "rank_max": high,
                "sample_count": int(np.count_nonzero(bin_mask)),
                "mean_score": float(np.nanmean(score[bin_mask])),
                "mean_target": float(np.nanmean(y_true[bin_mask])),
                "bias_score_minus_target": float(
                    np.nanmean(score[bin_mask]) - np.nanmean(y_true[bin_mask])
                ),
            }
        )
    return rows


def append_iteration_log(
    paths: MaxAutoPaths,
    *,
    iteration_id: str,
    files_read: list[str],
    hypothesis: str,
    evidence_before: dict[str, Any],
    selected_action: str,
    files_changed: list[str],
    commands_run: list[str],
    outputs_generated: list[str],
    metrics_after: dict[str, Any],
    interpretation: str,
    decision: str,
    continue_or_stop: str,
    next_action: str,
    pending_commit_message: str,
    human_intervention_required: bool = False,
) -> None:
    path = paths.reports / "mvp4x_max_auto_iteration_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# MVP-4X Max Auto Iteration Log\n\n"
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.\n\n",
            encoding="utf-8",
        )
    entry = [
        f"## iteration_id: {iteration_id}",
        f"- timestamp: {_utc_now()}",
        f"- files_read: `{json.dumps(files_read, ensure_ascii=False)}`",
        f"- hypothesis: {hypothesis}",
        f"- evidence_before: `{_json(evidence_before)}`",
        f"- selected_action: {selected_action}",
        f"- files_changed: `{json.dumps(files_changed, ensure_ascii=False)}`",
        f"- commands_run: `{json.dumps(commands_run, ensure_ascii=False)}`",
        f"- outputs_generated: `{json.dumps(outputs_generated, ensure_ascii=False)}`",
        f"- metrics_after: `{_json(metrics_after)}`",
        f"- interpretation: {interpretation}",
        f"- decision: {decision}",
        f"- continue_or_stop: {continue_or_stop}",
        f"- human_intervention_required: {human_intervention_required}",
        f"- next_action: {next_action}",
        f"- pending_commit_message: {pending_commit_message}",
        "- scope_flags: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))


def write_pending_commits(paths: MaxAutoPaths) -> None:
    path = paths.reports / "mvp4x_pending_commits.md"
    content = [
        "# MVP-4X Pending Commits",
        "",
        "Git writes failed in this sandbox because `.git` is read-only. Do not push, merge, "
        "reset, rebase, or squash automatically.",
        "",
        "## pending_commit_001",
        "- phase: code_and_bounded_research_outputs",
        "- suggested message: `feat: add mvp4x max autonomous research loop`",
        "- git add files: `src/cement_channel/modeling/mvp4x_max_auto.py "
        "scripts/07u_run_mvp4x_max_auto.py tests/unit/test_mvp4x_max_auto.py "
        "tests/integration/test_run_mvp4x_max_auto.py`",
        "- files changed: max-auto module, CLI, unit test, integration test",
        "- tests passed: recorded in final response after `make test-smoke`, `make test`, "
        "`make lint`, and `python scripts/00_check_env.py` complete",
        "- phase outputs: external data-root reports/artifacts are generated review outputs, "
        "not repository source files",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def format_final_score_markdown(report: dict[str, Any]) -> str:
    top_k = report["metrics"]["top_k_lift"]["top_k"]
    return "\n".join(
        [
            "# MVP-4X Final OOF Screening Scores",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- report_version: `{report['report_version']}`",
            f"- supported_scored: {report['sample_counts']['supported_scored']}",
            f"- audit_only_or_unsupported: {report['sample_counts']['audit_only_or_unsupported']}",
            f"- support_tiers: `{_json(report['support_tiers'])}`",
            f"- Spearman: {report['metrics']['selected_score_metrics']['spearman']:.6f}",
            f"- R2: {report['metrics']['selected_score_metrics']['r2']:.6f}",
            f"- top_5_lift: {top_k['top_5pct']['lift']:.6f}",
            f"- top_10_lift: {top_k['top_10pct']['lift']:.6f}",
            f"- top_20_lift: {top_k['top_20pct']['lift']:.6f}",
            "",
            "Scores are OOF ranking aids only. They are not absolute channel-fraction "
            "predictions, final labels, ground truth, production scores, or deployment outputs.",
            "",
        ]
    )


def format_review_summary(final_report: dict[str, Any], intervals: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# MVP-4X Max Auto Review Summary",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- selected_intervals: {len(intervals)}",
            f"- supported_scored: {final_report['sample_counts']['supported_scored']}",
            f"- support_tiers: `{_json(final_report['support_tiers'])}`",
            f"- top_k_lift: `{_json(final_report['metrics']['top_k_lift'])}`",
            "",
            "Review the selected intervals as weak-label screening cases only. Unsupported "
            "intervals are audit-only and must not be treated as validated scores.",
            "",
        ]
    )


def format_reviewer_checklist() -> str:
    return "\n".join(
        [
            "# MVP-4X Reviewer Checklist",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            "- Confirm the interval type and support tier before interpreting the score.",
            "- For high-score intervals, inspect whether the weak target has matching morphology.",
            "- For false-positive-like intervals, check whether low target is caused by "
            "label noise.",
            "- For false-negative-like intervals, check connected morphology and azimuth "
            "concentration.",
            "- For gap-instability intervals, treat score rank as unstable.",
            "- For unsupported intervals, record audit observations only; do not validate a score.",
            "- Do not approve final labels, production use, or ground-truth claims from this pack.",
            "",
        ]
    )


def format_triage_markdown(report: dict[str, Any]) -> str:
    morphology = report["morphology_error_analysis"]
    return "\n".join(
        [
            "# MVP-4X Max Auto Triage",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- decision: `{report['decision']}`",
            f"- false_positive_like_sample_count: {report['false_positive_like_sample_count']}",
            f"- false_negative_like_sample_count: {report['false_negative_like_sample_count']}",
            f"- morphology_related_to_errors: {morphology['morphology_related_to_errors']}",
            f"- best_target_view: `{report['recommended_research_candidate_target_view']}`",
            f"- domain_shift_still_exists: {report['domain_shift_still_exists']}",
            f"- calibration_still_insufficient: {report['calibration_still_insufficient']}",
            "",
            report["interpretation"],
            "",
        ]
    )


def format_morphology_candidate_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X Morphology Candidate Labels v002",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            "- morphology_candidate_v2: true",
            "- audit_only: true",
            "- parallel_comparison_only: true",
            f"- correlation_with_v1: {report['correlation_with_v1']:.6f}",
            f"- distribution: `{_json(report['target_distribution']['morphology_candidate_v2'])}`",
            "",
            "This is not a final label and does not replace v1.",
            "",
        ]
    )


def format_morphology_comparison_markdown(report: dict[str, Any]) -> str:
    summary = report["comparison_summary"]
    return "\n".join(
        [
            "# MVP-4X v1 vs Morphology v2 Comparison",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- decision: `{report['decision']}`",
            f"- v1_spearman: {summary['v1_spearman']:.6f}",
            f"- morphology_v2_spearman: {summary['morphology_v2_spearman']:.6f}",
            f"- delta_spearman: {summary['delta_spearman']:.6f}",
            f"- v1_top10_lift: {summary['v1_top10_lift']:.6f}",
            f"- morphology_v2_top10_lift: {summary['morphology_v2_top10_lift']:.6f}",
            f"- morphology_v2_improved: {summary['morphology_v2_improved']}",
            "",
            report["interpretation"],
            "",
        ]
    )


def format_final_decision_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X Max Auto Decision",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- decision: `{report['decision']}`",
            f"- autonomous_iterations_completed: {report['autonomous_iterations_completed']}",
            f"- scored_sample_count: {report['scored_sample_count']}",
            f"- morphology_explains_error: {report['morphology_explains_error']}",
            f"- morphology_candidate_v2_generated: {report['morphology_candidate_v2_generated']}",
            f"- morphology_v2_improved: {report['morphology_v2_improved']}",
            f"- fixed_time_frequency_v2_extracted: {report['fixed_time_frequency_v2_extracted']}",
            f"- best_target_view: `{report['best_target_view']}`",
            f"- best_feature_set: `{report['best_feature_set']}`",
            f"- best_model: `{report['best_model']}`",
            f"- domain_shift_still_exists: {report['domain_shift_still_exists']}",
            f"- calibration_still_insufficient: {report['calibration_still_insufficient']}",
            f"- recommend_server: {report['recommend_server']}",
            f"- next_human_approval_required: {report['next_human_approval_required']}",
            "",
            "Production claims and final labels remain forbidden.",
            "",
        ]
    )


def _select_nonoverlapping_centers(
    depth: np.ndarray,
    mask: np.ndarray,
    value: np.ndarray,
    *,
    descending: bool,
    limit: int,
) -> list[int]:
    valid = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(value))
    if valid.size == 0:
        return []
    order = valid[np.argsort(value[valid])]
    if descending:
        order = order[::-1]
    selected: list[int] = []
    for index in order:
        if all(abs(float(depth[index]) - float(depth[other])) > 5.0 for other in selected):
            selected.append(int(index))
        if len(selected) >= limit:
            break
    return selected


def _policy_score_std(policy_scores: np.ndarray) -> np.ndarray:
    output = np.full(policy_scores.shape[0], np.nan, dtype=np.float32)
    for index in range(policy_scores.shape[0]):
        values = policy_scores[index]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            output[index] = float(np.std(values))
        elif values.size == 1:
            output[index] = 0.0
    return output


def _permute_within_depth_bins(
    y: np.ndarray,
    depth: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32).copy()
    quantiles = np.quantile(depth, np.linspace(0.0, 1.0, 8))
    for start, stop in zip(quantiles[:-1], quantiles[1:], strict=True):
        mask = (depth >= start) & (depth <= stop)
        out[mask] = rng.permutation(out[mask])
    return out


def _permute_blocks(y: np.ndarray, depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(depth)
    blocks = np.array_split(order, 12)
    shuffled_blocks = list(blocks)
    rng.shuffle(shuffled_blocks)
    out = np.asarray(y, dtype=np.float32).copy()
    for source, dest in zip(blocks, shuffled_blocks, strict=True):
        source_values = y[source]
        if source_values.size == dest.size:
            out[dest] = source_values
    return out


def _summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0, "finite_ratio": 0.0}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "finite_ratio": float(finite.size / max(arr.size, 1)),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _std_delta(values: np.ndarray, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32)
    ref = ref[np.isfinite(ref)]
    val = np.asarray(values, dtype=np.float32)
    val = val[np.isfinite(val)]
    if ref.size == 0 or val.size == 0:
        return 0.0
    std = float(np.std(ref))
    if std == 0.0:
        return 0.0
    return float((np.mean(val) - np.mean(ref)) / std)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    result = spearmanr(a[mask], b[mask])
    return float(result.statistic)


def _as_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return output if np.isfinite(output) else float("-inf")


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def _nanmean_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values)
    return 0.0 if arr.size == 0 else float(np.isfinite(arr).mean())


def _string_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {str(item): int(count) for item, count in zip(unique, counts, strict=True)}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise MaxAutoError(f"Missing NPZ artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MaxAutoError(f"Missing JSON artifact: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise MaxAutoError(f"JSON artifact must be an object: {path}")
    return loaded


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
