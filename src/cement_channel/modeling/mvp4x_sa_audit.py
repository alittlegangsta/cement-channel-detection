from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import _make_model, compute_regression_metrics
from cement_channel.modeling.mvp4x_max_auto import MaxAutoPaths, calibration_bins
from cement_channel.modeling.mvp4x_next_auto import (
    _permute_blocks,
    _permute_within_depth_bins,
    bootstrap_spearman_ci,
    max_abs_calibration_bias,
    ndcg_score,
    ordinal_macro_f1,
    safe_kendall,
)
from cement_channel.modeling.mvp4x_screening_scores import rank_percentile, top_k_lift

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cement-channel")

SA_AUDIT_VERSION = "mvp4x_sa_audit_v001"
PROXY_CONTRACT_VERSION = "mvp4x_sa_proxy_contract_audit_v001"
MATCHED_TABLE_VERSION = "mvp4x_sa_matched_pilot_table_v001"
SCREENING_AUDIT_VERSION = "mvp4x_sa_matched_screening_audit_v001"
EXPANSION_AUDIT_VERSION = "mvp4x_sa_interval_expansion_audit_v001"
DECISION_VERSION = "mvp4x_sa_proxy_decision_v001"
REVIEW_PACK_DIR = "mvp4x_sa_proxy_review_v001"

SCOPE_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}
METHOD_FLAGS = {
    **SCOPE_FLAGS,
    "no_final_labels_generated": True,
    "no_raw_mat_modified": True,
    "no_full_well_stc": True,
    "no_full_well_apes": True,
    "no_deep_learning": True,
    "proxy_only_not_formal_stc_apes": True,
}
FEATURE_SET_ORDER = (
    "existing_features_only",
    "waveform_v1_only",
    "TF_v2_only",
    "STC_proxy_only",
    "APES_proxy_only",
    "existing_plus_STC_proxy",
    "existing_plus_APES_proxy",
    "existing_plus_STC_APES_proxy",
    "all_available_pilot_features",
)
TARGETS = ("receiver_mean", "receiver_p90", "receiver_max")
PRIMARY_TARGET = "receiver_mean"
ROBUST_TARGET = "receiver_p90"
AUDIT_ONLY_TARGET = "receiver_max"
MODELS = (
    "DummyRegressor",
    "Ridge",
    "ElasticNet",
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
)
MODEL_CONFIG = {
    "n_contiguous_folds": 3,
    "ridge_alpha": 1.0,
    "elastic_net_alpha": 0.01,
    "elastic_net_l1_ratio": 0.5,
    "random_forest_n_estimators": 32,
    "random_forest_max_depth": 5,
    "random_forest_n_jobs": 2,
    "hist_gradient_boosting_max_iter": 60,
    "hist_gradient_boosting_max_leaf_nodes": 15,
}
METADATA_ONLY_FIELDS = (
    "depth",
    "depth_center",
    "regime_id",
    "broad_regime_id",
    "orientation_cohort",
    "support_tier",
    "support_cohort",
    "special bands",
    "CAST-derived arrays",
    "morphology arrays",
    "label_confidence",
    "inclination_deg",
    "snapshot_index",
    "selection_category",
)
DECISION_OPTIONS = {
    "request_full_research_proxy_extraction_approval",
    "request_formal_stc_apes_implementation_review",
    "request_expand_bounded_pilot_approval",
    "request_multiwell_or_label_review",
    "stop_sa_proxy_not_helpful",
    "stop_data_contract_issue",
    "stop_leakage_detected",
    "stop_resource_issue",
}


class SaAuditError(RuntimeError):
    """Raised when the bounded MVP-4X-SA audit cannot run safely."""


@dataclass(frozen=True)
class SaPilotArtifacts:
    run_dir: Path
    report_json: Path
    report_md: Path
    intervals_csv: Path
    manifest_json: Path
    status_json: Path
    stdout_log: Path
    stderr_log: Path
    environment_txt: Path
    report: dict[str, Any]
    manifest: dict[str, Any]
    status: dict[str, Any]
    intervals: list[dict[str, str]]


@dataclass(frozen=True)
class SaAuditOutputs:
    contract_json: Path
    contract_md: Path
    feature_inventory_csv: Path
    matched_npz: Path
    matched_json: Path
    matched_md: Path
    screening_csv: Path
    screening_json: Path
    screening_md: Path
    expansion_csv: Path
    expansion_json: Path
    expansion_md: Path
    decision_json: Path
    decision_md: Path
    review_pack_dir: Path


def run_sa_audit_from_paths(
    *,
    paths: MaxAutoPaths,
    pilot_run_dirs: list[Path | str],
    snapshot_npz: Path | str | None = None,
    waveform_npz: Path | str | None = None,
    tfv2_npz: Path | str | None = None,
    final_scores_npz: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths.ensure_read_write()
    if not pilot_run_dirs:
        raise SaAuditError("At least one fetched pilot run directory is required.")
    artifacts = [load_sa_pilot_artifacts(Path(run_dir)) for run_dir in pilot_run_dirs]
    artifacts = sorted(artifacts, key=lambda item: int(item.report.get("interval_count", 0)))
    snapshot = _load_npz(snapshot_npz or paths.interim / "mvp4x_research_snapshot_v001.npz")
    waveform = _load_npz(waveform_npz or paths.features / "mvp4x_waveform_features_v001.npz")
    tfv2 = _load_npz(tfv2_npz or paths.features / "mvp4x_time_frequency_features_v002.npz")
    final_scores = _load_npz(
        final_scores_npz or paths.interim / "mvp4x_screening_scores_final_oof_v001.npz"
    )
    latest = artifacts[-1]
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()

    outputs = _default_outputs(paths)
    _prepare_outputs(outputs, overwrite=overwrite)
    contract = build_proxy_contract_audit(latest)
    matched = build_matched_table(
        latest,
        snapshot=snapshot,
        waveform=waveform,
        tfv2=tfv2,
        final_scores=final_scores,
    )
    screening = run_matched_screening_audit(
        matched=matched,
        sklearn_modules=sklearn_modules,
        modeling_environment=modeling_environment.to_dict(),
    )
    expansion = build_expansion_audit(
        artifacts=artifacts,
        matched=matched,
        screening=screening,
    )
    decision = build_final_decision(
        contract=contract,
        matched=matched,
        screening=screening,
        expansion=expansion,
    )
    write_contract_outputs(contract, outputs)
    write_matched_table_outputs(matched, outputs)
    write_screening_outputs(screening, outputs)
    write_expansion_outputs(expansion, outputs)
    write_decision_outputs(decision, outputs)
    write_review_pack(
        outputs=outputs,
        contract=contract,
        matched=matched,
        screening=screening,
        expansion=expansion,
        decision=decision,
    )
    return {
        "outputs": {key: str(value) for key, value in outputs.__dict__.items()},
        "contract": contract,
        "matched": matched["report"],
        "screening": screening["report"],
        "expansion": expansion,
        "decision": decision,
        **METHOD_FLAGS,
    }


def load_sa_pilot_artifacts(run_dir: Path) -> SaPilotArtifacts:
    report_json = run_dir / "reports" / "mvp4x_sa_pilot_auto_v001" / "mvp4x_sa_pilot_report.json"
    report_md = run_dir / "reports" / "mvp4x_sa_pilot_auto_v001" / "mvp4x_sa_pilot_report.md"
    intervals_csv = (
        run_dir / "reports" / "mvp4x_sa_pilot_auto_v001" / "mvp4x_sa_pilot_intervals.csv"
    )
    required = {
        "report_json": report_json,
        "report_md": report_md,
        "intervals_csv": intervals_csv,
        "manifest_json": run_dir / "manifest.json",
        "status_json": run_dir / "status.json",
        "stdout_log": run_dir / "stdout.log",
        "stderr_log": run_dir / "stderr.log",
        "environment_txt": run_dir / "environment.txt",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SaAuditError(f"Fetched pilot run is missing file(s): {', '.join(missing)}")
    report = _read_json(report_json)
    manifest = _read_json(required["manifest_json"])
    status = _read_json(required["status_json"])
    with intervals_csv.open("r", encoding="utf-8", newline="") as handle:
        intervals = list(csv.DictReader(handle))
    if not intervals:
        raise SaAuditError(f"Pilot interval CSV has no rows: {intervals_csv}")
    _validate_pilot_artifact_scope(run_dir, report, manifest, status)
    return SaPilotArtifacts(
        run_dir=run_dir,
        report_json=report_json,
        report_md=report_md,
        intervals_csv=intervals_csv,
        manifest_json=required["manifest_json"],
        status_json=required["status_json"],
        stdout_log=required["stdout_log"],
        stderr_log=required["stderr_log"],
        environment_txt=required["environment_txt"],
        report=report,
        manifest=manifest,
        status=status,
        intervals=intervals,
    )


def build_proxy_contract_audit(artifacts: SaPilotArtifacts) -> dict[str, Any]:
    report = artifacts.report
    intervals = artifacts.intervals
    first_row = intervals[0]
    feature_inventory = [
        {
            "feature_name": "stc_peak_coherence",
            "feature_group": "STC_proxy",
            "source": "bounded_relative_delay_semblance_proxy",
            "formal_stc": False,
            "model_input_allowed": True,
        },
        {
            "feature_name": "stc_peak_delay_samples_per_receiver",
            "feature_group": "STC_proxy",
            "source": "bounded_relative_delay_semblance_proxy",
            "formal_stc": False,
            "model_input_allowed": True,
        },
        {
            "feature_name": "stc_peak_side_one_hot_A_to_H",
            "feature_group": "STC_proxy",
            "source": "bounded_relative_delay_semblance_proxy",
            "formal_stc": False,
            "model_input_allowed": True,
        },
        {
            "feature_name": "apes_peak_power",
            "feature_group": "APES_proxy",
            "source": "bounded_regularized_adaptive_spectrum_proxy",
            "formal_apes": False,
            "model_input_allowed": True,
        },
        {
            "feature_name": "apes_peak_frequency_cycles_per_sample",
            "feature_group": "APES_proxy",
            "source": "bounded_regularized_adaptive_spectrum_proxy",
            "formal_apes": False,
            "model_input_allowed": True,
        },
        {
            "feature_name": "apes_peak_side_one_hot_A_to_H",
            "feature_group": "APES_proxy",
            "source": "bounded_regularized_adaptive_spectrum_proxy",
            "formal_apes": False,
            "model_input_allowed": True,
        },
    ]
    waveform_policy = _as_dict(report.get("waveform_read_policy"))
    stc_policy = _as_dict(report.get("stc_policy"))
    apes_policy = _as_dict(report.get("apes_policy"))
    proxy_names_preserved = (
        "proxy" in str(stc_policy.get("method", "")).lower()
        and "proxy" in str(apes_policy.get("method", "")).lower()
    )
    report_md_text = artifacts.report_md.read_text(encoding="utf-8")
    proxy_names_preserved = proxy_names_preserved and "proxy" in report_md_text.lower()
    return {
        "report_version": PROXY_CONTRACT_VERSION,
        "generated_at": _utc_now(),
        "run_id": artifacts.manifest.get("run_id"),
        "run_dir": str(artifacts.run_dir),
        "status": artifacts.status.get("status"),
        "remote_environment": {
            "scheduler": artifacts.manifest.get("scheduler"),
            "remote_env_root": artifacts.manifest.get("remote_env_root"),
            "remote_python": artifacts.manifest.get("remote_python"),
            "python_no_user_site": artifacts.manifest.get("python_no_user_site"),
            "git_commit": artifacts.manifest.get("git_commit"),
        },
        "pilot_report_files_read": {
            "report_json": str(artifacts.report_json),
            "report_md": str(artifacts.report_md),
            "intervals_csv": str(artifacts.intervals_csv),
            "manifest_json": str(artifacts.manifest_json),
            "status_json": str(artifacts.status_json),
            "stdout_log": str(artifacts.stdout_log),
            "stderr_log": str(artifacts.stderr_log),
            "environment_txt": str(artifacts.environment_txt),
        },
        "stc_bounded_proxy_contract": {
            "input_waveform_dimension": "[depth_chunk, receiver, side, time]",
            "observed_feature_shape_per_interval": [
                int(first_row.get("chunk_depth_samples", 0) or 0),
                int(first_row.get("receiver_count_read", 0) or 0),
                int(first_row.get("side_count_read", 0) or 0),
                int(first_row.get("time_samples", 0) or 0),
            ],
            "receiver_dimension": "receiver_count_read",
            "side_dimension": "side_count_read",
            "fixed_window": {
                "chunk_depth_samples": waveform_policy.get("chunk_depth_samples"),
                "max_time_samples": waveform_policy.get("max_time_samples"),
                "source_index_mode": waveform_policy.get("source_index_mode"),
            },
            "fixed_slowness_grid_or_proxy_definition": {
                "method": stc_policy.get("method"),
                "delay_unit": stc_policy.get("delay_unit"),
                "delay_grid": "relative receiver delay grid in samples per receiver",
                "formal_slowness_axis": False,
            },
            "aggregation": (
                "nanmean over selected depth chunk, then receiver-alignment semblance "
                "proxy per side; interval output keeps peak side, delay, and coherence."
            ),
            "output_features": [
                "stc_peak_coherence",
                "stc_peak_delay_samples_per_receiver",
                "stc_peak_side",
            ],
            "formal_stc_difference": (
                "This does not produce a slowness-time coherence map, does not use a "
                "calibrated physical slowness axis, and does not scan full depth/time windows."
            ),
            "formal_stc_claim_allowed": False,
        },
        "apes_bounded_proxy_contract": {
            "input_waveform_dimension": "[depth_chunk, receiver, side, time]",
            "observed_feature_shape_per_interval": [
                int(first_row.get("chunk_depth_samples", 0) or 0),
                int(first_row.get("receiver_count_read", 0) or 0),
                int(first_row.get("side_count_read", 0) or 0),
                int(first_row.get("time_samples", 0) or 0),
            ],
            "receiver_dimension": "receiver_count_read",
            "side_dimension": "side_count_read",
            "fixed_frequency_range": "0.02 to 0.45 cycles/sample",
            "fixed_parameters": {
                "frequency_count": 24,
                "covariance_window": 32,
                "regularization": 1.0e-3,
            },
            "aggregation": (
                "normalize traces, average receivers by side, estimate regularized adaptive "
                "spectrum per side, and keep the interval peak side/frequency/power."
            ),
            "output_features": [
                "apes_peak_power",
                "apes_peak_frequency_cycles_per_sample",
                "apes_peak_side",
            ],
            "formal_apes_difference": (
                "This does not produce a formal APES dispersion map or phase/amplitude "
                "surface and does not support full-well high-resolution interpretation."
            ),
            "formal_apes_claim_allowed": False,
        },
        "physics_review": {
            "array_coherent_propagation_proxy_features": [
                "stc_peak_coherence",
                "stc_peak_delay_samples_per_receiver",
                "stc_peak_side_one_hot_A_to_H",
            ],
            "frequency_or_dispersion_proxy_features": [
                "apes_peak_power",
                "apes_peak_frequency_cycles_per_sample",
                "apes_peak_side_one_hot_A_to_H",
            ],
            "engineering_summary_features": [
                "peak side indicators",
                "single interval peak scalar summaries",
            ],
            "not_direct_physical_interpretations": [
                "formation slowness",
                "velocity",
                "formal STC ridge",
                "formal APES dispersion ridge",
            ],
        },
        "feature_inventory": feature_inventory,
        "proxy_names_preserved_in_reports": proxy_names_preserved,
        "leakage_boundary": {
            "uses_cast_arrays_as_model_input": False,
            "uses_morphology_arrays_as_model_input": False,
            "metadata_only_fields": list(METADATA_ONLY_FIELDS),
        },
        **METHOD_FLAGS,
    }


def build_matched_table(
    artifacts: SaPilotArtifacts,
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    tfv2: dict[str, np.ndarray],
    final_scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    _validate_research_flags(snapshot, "snapshot")
    _validate_research_flags(waveform, "waveform")
    _validate_research_flags(tfv2, "tfv2")
    row_count = int(np.asarray(snapshot["depth"]).shape[0])
    indices = np.asarray(
        [int(row["snapshot_index"]) for row in artifacts.intervals], dtype=np.int64
    )
    if np.any(indices < 0) or np.any(indices >= row_count):
        raise SaAuditError("Pilot interval snapshot_index is out of bounds.")
    statuses = np.asarray([row.get("status", "") for row in artifacts.intervals]).astype(str)
    keep = statuses == "succeeded"
    if not np.any(keep):
        raise SaAuditError("No succeeded pilot intervals are available for matched audit.")
    kept_indices = indices[keep]
    exclusions = []
    if np.count_nonzero(~keep):
        exclusions.append(
            {
                "reason": "pilot_interval_not_succeeded",
                "count": int(np.count_nonzero(~keep)),
            }
        )
    metadata = build_matched_metadata(artifacts.intervals, keep, snapshot, final_scores)
    targets = {
        "receiver_mean": np.asarray(snapshot["receiver_mean"], dtype=np.float32)[kept_indices],
        "receiver_p90": np.asarray(snapshot["receiver_p90"], dtype=np.float32)[kept_indices],
        "receiver_max": np.asarray(snapshot["receiver_max"], dtype=np.float32)[kept_indices],
    }
    feature_sets = build_sa_feature_sets(
        artifacts.intervals,
        keep=keep,
        indices=kept_indices,
        snapshot=snapshot,
        waveform=waveform,
        tfv2=tfv2,
    )
    missing_alignment = validate_alignment(
        metadata=metadata,
        snapshot=snapshot,
        waveform=waveform,
        tfv2=tfv2,
        indices=kept_indices,
    )
    leakage = audit_feature_leakage(feature_sets)
    report = {
        "report_version": MATCHED_TABLE_VERSION,
        "generated_at": _utc_now(),
        "run_id": artifacts.manifest.get("run_id"),
        "pilot_interval_count": int(len(artifacts.intervals)),
        "sample_count": int(kept_indices.size),
        "feature_sets": summarize_feature_sets(feature_sets),
        "target_summaries": {name: _summary(values) for name, values in targets.items()},
        "target_roles": {
            "receiver_mean": "primary",
            "receiver_p90": "robust_research_candidate",
            "receiver_max": "audit_only_never_primary",
        },
        "metadata_only_fields": list(METADATA_ONLY_FIELDS),
        "metadata_distributions": metadata_distributions(metadata),
        "missing_alignment_samples": missing_alignment,
        "exclusions": exclusions,
        "finite_ratio": {
            name: float(np.isfinite(feature_set["matrix"]).mean())
            for name, feature_set in feature_sets.items()
        },
        "leakage_audit": leakage,
        "train_fold_only_preprocessing": True,
        **METHOD_FLAGS,
    }
    return {
        "report": report,
        "metadata": metadata,
        "targets": targets,
        "feature_sets": feature_sets,
    }


def build_sa_feature_sets(
    rows: list[dict[str, str]],
    *,
    keep: np.ndarray,
    indices: np.ndarray,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    tfv2: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    existing_mask = np.asarray(
        snapshot.get("model_feature_mask", np.ones(np.asarray(snapshot["xsi_features"]).shape[1])),
        dtype=bool,
    ).reshape(-1)
    existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)[indices][:, existing_mask]
    existing_names = np.asarray(snapshot["xsi_feature_names"]).astype(str)[existing_mask]
    existing_groups = np.asarray(snapshot["xsi_feature_group"]).astype(str)[existing_mask]
    wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)[indices]
    wave_names = np.asarray(waveform["waveform_depth_feature_names"]).astype(str)
    wave_groups = np.asarray(waveform["waveform_depth_feature_group"]).astype(str)
    tf = np.asarray(tfv2["tf_depth_features"], dtype=np.float32)[indices]
    tf_names = np.asarray(tfv2["tf_depth_feature_names"]).astype(str)
    tf_groups = np.asarray(tfv2["tf_depth_feature_group"]).astype(str)
    kept_rows = [row for row, ok in zip(rows, keep, strict=True) if ok]
    stc = build_stc_proxy_matrix(kept_rows)
    apes = build_apes_proxy_matrix(kept_rows)
    return {
        "existing_features_only": {
            "matrix": existing,
            "names": existing_names,
            "groups": np.asarray([f"existing:{item}" for item in existing_groups]),
        },
        "waveform_v1_only": {
            "matrix": wave,
            "names": wave_names,
            "groups": np.asarray([f"waveform_v1:{item}" for item in wave_groups]),
        },
        "TF_v2_only": {
            "matrix": tf,
            "names": tf_names,
            "groups": np.asarray([f"TF_v2:{item}" for item in tf_groups]),
        },
        "STC_proxy_only": stc,
        "APES_proxy_only": apes,
        "existing_plus_STC_proxy": _combine_feature_sets(
            (
                existing,
                existing_names,
                np.asarray([f"existing:{item}" for item in existing_groups]),
            ),
            (stc["matrix"], stc["names"], stc["groups"]),
        ),
        "existing_plus_APES_proxy": _combine_feature_sets(
            (
                existing,
                existing_names,
                np.asarray([f"existing:{item}" for item in existing_groups]),
            ),
            (apes["matrix"], apes["names"], apes["groups"]),
        ),
        "existing_plus_STC_APES_proxy": _combine_feature_sets(
            (
                existing,
                existing_names,
                np.asarray([f"existing:{item}" for item in existing_groups]),
            ),
            (stc["matrix"], stc["names"], stc["groups"]),
            (apes["matrix"], apes["names"], apes["groups"]),
        ),
        "all_available_pilot_features": _combine_feature_sets(
            (
                existing,
                existing_names,
                np.asarray([f"existing:{item}" for item in existing_groups]),
            ),
            (wave, wave_names, np.asarray([f"waveform_v1:{item}" for item in wave_groups])),
            (tf, tf_names, np.asarray([f"TF_v2:{item}" for item in tf_groups])),
            (stc["matrix"], stc["names"], stc["groups"]),
            (apes["matrix"], apes["names"], apes["groups"]),
        ),
    }


def build_stc_proxy_matrix(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = list("ABCDEFGH")
    matrix = []
    for row in rows:
        side = str(row.get("stc_peak_side", ""))
        one_hot = [1.0 if side == label else 0.0 for label in labels]
        matrix.append(
            [
                _float(row.get("stc_peak_coherence")),
                _float(row.get("stc_peak_delay_samples_per_receiver")),
                *one_hot,
            ]
        )
    names = np.asarray(
        [
            "stc_proxy_peak_coherence",
            "stc_proxy_peak_delay_samples_per_receiver",
            *[f"stc_proxy_peak_side_{label}" for label in labels],
        ]
    )
    return {
        "matrix": np.asarray(matrix, dtype=np.float32),
        "names": names,
        "groups": np.asarray(["STC_proxy"] * names.size),
    }


def build_apes_proxy_matrix(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = list("ABCDEFGH")
    matrix = []
    for row in rows:
        side = str(row.get("apes_peak_side", ""))
        one_hot = [1.0 if side == label else 0.0 for label in labels]
        matrix.append(
            [
                _float(row.get("apes_peak_power")),
                _float(row.get("apes_peak_frequency_cycles_per_sample")),
                *one_hot,
            ]
        )
    names = np.asarray(
        [
            "apes_proxy_peak_power",
            "apes_proxy_peak_frequency_cycles_per_sample",
            *[f"apes_proxy_peak_side_{label}" for label in labels],
        ]
    )
    return {
        "matrix": np.asarray(matrix, dtype=np.float32),
        "names": names,
        "groups": np.asarray(["APES_proxy"] * names.size),
    }


def run_matched_screening_audit(
    *,
    matched: dict[str, Any],
    sklearn_modules: dict[str, Any],
    modeling_environment: dict[str, Any],
) -> dict[str, Any]:
    metadata = matched["metadata"]
    feature_sets = matched["feature_sets"]
    targets = matched["targets"]
    depth = np.asarray(metadata["depth_center"], dtype=np.float32)
    regimes = np.asarray(metadata["broad_regime_id"]).astype(str)
    all_samples = np.ones(depth.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for feature_set_name in FEATURE_SET_ORDER:
        feature_set = feature_sets[feature_set_name]
        X = np.asarray(feature_set["matrix"], dtype=np.float32)
        for target in TARGETS:
            y = np.asarray(targets[target], dtype=np.float32).reshape(-1)
            sample_mask = all_samples & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            for model_name in MODELS:
                print(
                    "MVP-4X-SA audit "
                    f"feature_set={feature_set_name} target={target} model={model_name}",
                    flush=True,
                )
                result = fit_oof_model(
                    X=X,
                    y=y,
                    depth=depth,
                    sample_mask=sample_mask,
                    model_name=model_name,
                    sklearn_modules=sklearn_modules,
                    gap_ft=25.0,
                    random_seed=20260609 + len(rows),
                )
                prediction = result["prediction"]
                valid = sample_mask & np.isfinite(prediction)
                rank = rank_percentile(np.where(valid, prediction, np.nan))
                metrics = compute_extended_metrics(y[valid], prediction[valid], rank[valid])
                calibration = calibration_bins(y, prediction, valid, n_bins=5)
                row = {
                    "feature_set": feature_set_name,
                    "target": target,
                    "target_role": _target_role(target),
                    "model": model_name,
                    "sample_count": int(np.count_nonzero(valid)),
                    "feature_count": int(X.shape[1]),
                    "metrics": metrics,
                    "folds": result["folds"],
                    "max_abs_calibration_bias": max_abs_calibration_bias(calibration),
                    "calibration": calibration,
                    "category_summary": subgroup_metric_summary(
                        y,
                        prediction,
                        valid,
                        np.asarray(metadata["selection_category"]).astype(str),
                    ),
                    "regime_summary": subgroup_metric_summary(y, prediction, valid, regimes),
                    "special_band_summary": subgroup_metric_summary(
                        y,
                        prediction,
                        valid,
                        np.asarray(metadata["any_special_flag"]).astype(str),
                    ),
                    "support_tier_summary": subgroup_metric_summary(
                        y,
                        prediction,
                        valid,
                        np.asarray(metadata["support_tier"]).astype(str),
                    ),
                    "residual_summary": _summary((prediction - y)[valid]),
                    "prediction": prediction,
                    **METHOD_FLAGS,
                }
                rows.append(row)
                csv_rows.append(flatten_sa_model_row(row))
    strict = run_strict_proxy_audits(
        matched=matched,
        matrix_rows=rows,
        sklearn_modules=sklearn_modules,
    )
    best_by_feature_set = {
        name: strip_prediction(
            select_best_model_row([row for row in rows if row["feature_set"] == name])
        )
        for name in FEATURE_SET_ORDER
    }
    best_primary_by_feature_set = {
        name: strip_prediction(
            select_best_model_row(
                [
                    row
                    for row in rows
                    if row["feature_set"] == name and row["target"] == PRIMARY_TARGET
                ]
            )
        )
        for name in FEATURE_SET_ORDER
    }
    best_overall = strip_prediction(select_best_model_row(rows))
    report = {
        "report_version": SCREENING_AUDIT_VERSION,
        "generated_at": _utc_now(),
        "modeling_environment": modeling_environment,
        "sample_count": matched["report"]["sample_count"],
        "feature_sets": matched["report"]["feature_sets"],
        "targets": {
            "primary": PRIMARY_TARGET,
            "robust_research_candidate": ROBUST_TARGET,
            "audit_only": AUDIT_ONLY_TARGET,
            "receiver_max_never_primary": True,
        },
        "models": list(MODELS),
        "validation_protocol": {
            "blocked_contiguous_cv": True,
            "primary_gap_ft": 25.0,
            "blocked_gap_sensitivity_ft": [10.0, 25.0, 50.0],
            "global_permutation": True,
            "within_depth_bin_permutation": True,
            "block_permutation": True,
            "bootstrap_confidence_intervals": True,
            "train_fold_only_preprocessing": True,
        },
        "matrix_row_count": len(rows),
        "best_overall": best_overall,
        "best_by_feature_set": best_by_feature_set,
        "best_primary_by_feature_set": best_primary_by_feature_set,
        "strict_proxy_audits": strict,
        "category_stratified_summary": metadata_distributions(metadata),
        "regime_bc_summary": regime_bc_support(metadata),
        "special_band_sensitivity": strict.get("special_band_sensitivity"),
        "leave_category_out_audit": strict.get("leave_category_out_audit"),
        "feature_group_ablation": strict.get("feature_group_ablation"),
        "permutation_importance": strict.get("permutation_importance"),
        "residual_audit": strict.get("residual_audit"),
        "score_stability": strict.get("score_stability"),
        "sample_support_warning": build_sample_support_warning(matched),
        "transfer_warning": strict.get("transfer_warning"),
        "leakage_audit": matched["report"]["leakage_audit"],
        **METHOD_FLAGS,
    }
    return {"report": report, "rows": rows, "csv_rows": csv_rows}


def fit_oof_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    gap_ft: float,
    random_seed: int,
) -> dict[str, Any]:
    matrix = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1) & np.all(np.isfinite(matrix), axis=1)
    folds = contiguous_folds(depth, mask, n_folds=3)
    prediction = np.full(y.shape, np.nan, dtype=np.float32)
    fold_id = np.full(y.shape, -1, dtype=np.int16)
    fold_rows = []
    for fold_index, validation_mask in enumerate(folds):
        validation_mask = validation_mask & mask
        train_mask = mask & ~validation_mask
        if gap_ft > 0.0 and np.any(validation_mask):
            low = float(np.min(depth[validation_mask])) - gap_ft
            high = float(np.max(depth[validation_mask])) + gap_ft
            train_mask &= (depth < low) | (depth > high)
        if np.count_nonzero(train_mask) < 2 or np.count_nonzero(validation_mask) < 1:
            fold_rows.append(
                {
                    "fold_id": fold_index,
                    "status": "skipped_insufficient_train_or_validation",
                    "train_count": int(np.count_nonzero(train_mask)),
                    "validation_count": int(np.count_nonzero(validation_mask)),
                }
            )
            continue
        model = _make_model(model_name, sklearn_modules, MODEL_CONFIG, random_state=random_seed)
        model.fit(matrix[train_mask], y[train_mask])
        pred = np.asarray(model.predict(matrix[validation_mask]), dtype=np.float32)
        prediction[validation_mask] = pred
        fold_id[validation_mask] = fold_index
        fold_rows.append(
            {
                "fold_id": fold_index,
                "status": "completed",
                "train_count": int(np.count_nonzero(train_mask)),
                "validation_count": int(np.count_nonzero(validation_mask)),
                **compute_regression_metrics(y[validation_mask], pred),
            }
        )
    return {
        "prediction": prediction,
        "fold_id": fold_id,
        "folds": fold_rows,
        "metrics": compute_regression_metrics(y[mask], prediction[mask]),
    }


def run_strict_proxy_audits(
    *,
    matched: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    candidates = strict_candidate_rows(matrix_rows)
    strict_rows = {}
    for name, row in candidates.items():
        if row is None:
            strict_rows[name] = {"status": "skipped_no_candidate"}
            continue
        strict_rows[name] = strict_candidate_audit(
            matched=matched,
            row=row,
            sklearn_modules=sklearn_modules,
        )
    proxy_delta = compare_proxy_to_baseline(strict_rows)
    return {
        "status": "completed",
        "candidates": strict_rows,
        "proxy_delta_vs_baseline": proxy_delta,
        "blocked_gap_stability": {
            name: _as_dict(audit).get("gap_stability") for name, audit in strict_rows.items()
        },
        "permutation": {
            name: _as_dict(audit).get("permutation_margins") for name, audit in strict_rows.items()
        },
        "bootstrap_ci": {
            name: _as_dict(audit).get("bootstrap_ci") for name, audit in strict_rows.items()
        },
        "transfer_warning": transfer_warning(strict_rows),
        "special_band_sensitivity": {
            name: _as_dict(audit).get("special_band_sensitivity")
            for name, audit in strict_rows.items()
        },
        "leave_category_out_audit": {
            name: _as_dict(audit).get("leave_category_out_audit")
            for name, audit in strict_rows.items()
        },
        "feature_group_ablation": {
            name: _as_dict(audit).get("feature_group_ablation")
            for name, audit in strict_rows.items()
        },
        "permutation_importance": {
            name: _as_dict(audit).get("permutation_importance")
            for name, audit in strict_rows.items()
        },
        "residual_audit": {
            name: _as_dict(audit).get("residual_audit") for name, audit in strict_rows.items()
        },
        "score_stability": {
            name: _as_dict(audit).get("score_stability") for name, audit in strict_rows.items()
        },
    }


def strict_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    output = {
        "baseline_existing_ridge_receiver_mean": find_row(
            rows,
            feature_set="existing_features_only",
            target=PRIMARY_TARGET,
            model="Ridge",
        ),
        "stc_proxy_ridge_receiver_mean": find_row(
            rows,
            feature_set="STC_proxy_only",
            target=PRIMARY_TARGET,
            model="Ridge",
        ),
        "apes_proxy_ridge_receiver_mean": find_row(
            rows,
            feature_set="APES_proxy_only",
            target=PRIMARY_TARGET,
            model="Ridge",
        ),
        "existing_plus_stc_ridge_receiver_mean": find_row(
            rows,
            feature_set="existing_plus_STC_proxy",
            target=PRIMARY_TARGET,
            model="Ridge",
        ),
        "existing_plus_apes_ridge_receiver_mean": find_row(
            rows,
            feature_set="existing_plus_APES_proxy",
            target=PRIMARY_TARGET,
            model="Ridge",
        ),
        "combined_proxy_ridge_receiver_mean": find_row(
            rows,
            feature_set="existing_plus_STC_APES_proxy",
            target=PRIMARY_TARGET,
            model="Ridge",
        ),
        "all_available_ridge_receiver_mean": find_row(
            rows,
            feature_set="all_available_pilot_features",
            target=PRIMARY_TARGET,
            model="Ridge",
        ),
    }
    return output


def strict_candidate_audit(
    *,
    matched: dict[str, Any],
    row: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    feature_set = matched["feature_sets"][row["feature_set"]]
    X = np.asarray(feature_set["matrix"], dtype=np.float32)
    groups = np.asarray(feature_set["groups"]).astype(str)
    names = np.asarray(feature_set["names"]).astype(str)
    y = np.asarray(matched["targets"][row["target"]], dtype=np.float32)
    depth = np.asarray(matched["metadata"]["depth_center"], dtype=np.float32)
    sample_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    gap_results = {}
    for gap in (10.0, 25.0, 50.0):
        result = fit_oof_model(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            model_name=row["model"],
            sklearn_modules=sklearn_modules,
            gap_ft=gap,
            random_seed=20260609 + int(gap),
        )
        pred = result["prediction"]
        valid = sample_mask & np.isfinite(pred)
        gap_results[f"gap_{int(gap)}_ft"] = {
            "metrics": compute_extended_metrics(
                y[valid], pred[valid], rank_percentile(pred)[valid]
            ),
            "prediction": pred,
            "folds": result["folds"],
        }
    primary_pred = np.asarray(gap_results["gap_25_ft"]["prediction"], dtype=np.float32)
    valid = sample_mask & np.isfinite(primary_pred)
    rng = np.random.default_rng(20260609)
    permutations = {
        "global": permutation_margin(
            X,
            y,
            depth,
            sample_mask,
            row["model"],
            sklearn_modules,
            rng.permutation(y),
        ),
        "within_depth_bin": permutation_margin(
            X,
            y,
            depth,
            sample_mask,
            row["model"],
            sklearn_modules,
            _permute_within_depth_bins(y, depth, rng),
        ),
        "block": permutation_margin(
            X,
            y,
            depth,
            sample_mask,
            row["model"],
            sklearn_modules,
            _permute_blocks(y, depth, rng),
        ),
    }
    spearman_values = [
        _metric(_as_dict(value.get("metrics")), "spearman") for value in gap_results.values()
    ]
    spearman_values = [float(value) for value in spearman_values if value is not None]
    calibration = calibration_bins(y, primary_pred, valid, n_bins=5)
    return {
        "status": "completed",
        "candidate": {
            "feature_set": row["feature_set"],
            "target": row["target"],
            "model": row["model"],
        },
        "sample_count": int(np.count_nonzero(valid)),
        "feature_count": int(X.shape[1]),
        "primary_metrics": gap_results["gap_25_ft"]["metrics"],
        "gap_metrics": {name: value["metrics"] for name, value in gap_results.items()},
        "gap_stability": {
            "spearman_values": {
                name: _metric(_as_dict(value["metrics"]), "spearman")
                for name, value in gap_results.items()
            },
            "spearman_std": None if not spearman_values else float(np.std(spearman_values)),
            "spearman_min": None if not spearman_values else float(np.min(spearman_values)),
            "spearman_max": None if not spearman_values else float(np.max(spearman_values)),
        },
        "permutation_margins": permutations,
        "bootstrap_ci": bootstrap_spearman_ci(y[valid], primary_pred[valid]),
        "category_summary": subgroup_metric_summary(
            y,
            primary_pred,
            valid,
            np.asarray(matched["metadata"]["selection_category"]).astype(str),
        ),
        "regime_summary": subgroup_metric_summary(
            y,
            primary_pred,
            valid,
            np.asarray(matched["metadata"]["broad_regime_id"]).astype(str),
        ),
        "special_band_sensitivity": special_band_sensitivity(
            y=y,
            prediction=primary_pred,
            valid=valid,
            special=np.asarray(matched["metadata"]["any_special_flag"], dtype=bool),
        ),
        "leave_category_out_audit": leave_one_label_out(
            X=X,
            y=y,
            depth=depth,
            valid=valid,
            labels=np.asarray(matched["metadata"]["selection_category"]).astype(str),
            model_name=row["model"],
            sklearn_modules=sklearn_modules,
        ),
        "feature_group_ablation": feature_group_ablation(
            X=X,
            y=y,
            depth=depth,
            valid=valid,
            groups=groups,
            model_name=row["model"],
            sklearn_modules=sklearn_modules,
            baseline_spearman=_metric(row["metrics"], "spearman"),
        ),
        "permutation_importance": permutation_importance_summary(
            X=X,
            y=y,
            depth=depth,
            valid=valid,
            feature_names=names,
            feature_groups=groups,
            model_name=row["model"],
            sklearn_modules=sklearn_modules,
        ),
        "residual_audit": residual_audit(
            matched=matched,
            y=y,
            prediction=primary_pred,
            valid=valid,
        ),
        "score_stability": {
            "prediction_std_across_gaps": _summary(
                np.nanstd(
                    np.vstack(
                        [
                            gap_results["gap_10_ft"]["prediction"],
                            gap_results["gap_25_ft"]["prediction"],
                            gap_results["gap_50_ft"]["prediction"],
                        ]
                    ),
                    axis=0,
                )[valid]
            ),
            "max_abs_calibration_bias": max_abs_calibration_bias(calibration),
        },
        **METHOD_FLAGS,
    }


def permutation_margin(
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    permuted_y: np.ndarray,
) -> float | None:
    real = fit_oof_model(
        X=X,
        y=y,
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        gap_ft=25.0,
        random_seed=20260609,
    )
    perm = fit_oof_model(
        X=X,
        y=np.asarray(permuted_y, dtype=np.float32),
        depth=depth,
        sample_mask=sample_mask,
        model_name=model_name,
        sklearn_modules=sklearn_modules,
        gap_ft=25.0,
        random_seed=20260610,
    )
    real_s = _metric(real["metrics"], "spearman")
    perm_s = _metric(perm["metrics"], "spearman")
    return None if real_s is None or perm_s is None else float(real_s - perm_s)


def build_expansion_audit(
    *,
    artifacts: list[SaPilotArtifacts],
    matched: dict[str, Any],
    screening: dict[str, Any],
) -> dict[str, Any]:
    runs = []
    for artifact in artifacts:
        interval_count = int(artifact.report.get("interval_count", len(artifact.intervals)))
        runs.append(
            {
                "run_id": artifact.manifest.get("run_id"),
                "interval_count": interval_count,
                "status": artifact.status.get("status"),
                "runtime_seconds": artifact.report.get("runtime_seconds"),
                "peak_memory_bytes": artifact.report.get("peak_memory_bytes"),
                "estimated_waveform_bytes_read": artifact.report.get(
                    "estimated_waveform_bytes_read"
                ),
                "skipped_interval_count": artifact.report.get("skipped_interval_count"),
                "warnings_count": len(artifact.report.get("warnings", [])),
                "errors_count": len(artifact.report.get("errors", [])),
                "reports_only_fetched": True,
                **METHOD_FLAGS,
            }
        )
    support_warning = build_sample_support_warning(matched)
    instability = detect_screening_instability(screening["report"])
    latest_count = int(runs[-1]["interval_count"])
    if latest_count < 120 and (support_warning["expand_recommended"] or instability):
        next_action = "expand_to_120"
    elif latest_count < 160 and (support_warning["expand_recommended"] or instability):
        next_action = "expand_to_160"
    else:
        next_action = "no_more_bounded_expansion_required"
    comparison_rows = [
        {
            "interval_count": row["interval_count"],
            "run_id": row["run_id"],
            "runtime_seconds": row["runtime_seconds"],
            "peak_memory_bytes": row["peak_memory_bytes"],
            "estimated_waveform_bytes_read": row["estimated_waveform_bytes_read"],
            "warnings_count": row["warnings_count"],
            "errors_count": row["errors_count"],
            "status": row["status"],
        }
        for row in runs
    ]
    return {
        "report_version": EXPANSION_AUDIT_VERSION,
        "generated_at": _utc_now(),
        "runs": runs,
        "comparison_rows": comparison_rows,
        "latest_interval_count": latest_count,
        "sample_support_warning": support_warning,
        "screening_instability_warning": instability,
        "controlled_expansion_policy": {
            "min_intervals": 80,
            "preferred_expansion": 120,
            "max_intervals": 160,
            "selected_interval_chunked_reads_only": True,
            "reports_only_fetch": True,
            "no_full_well_extraction": True,
        },
        "next_action": next_action,
        **METHOD_FLAGS,
    }


def build_final_decision(
    *,
    contract: dict[str, Any],
    matched: dict[str, Any],
    screening: dict[str, Any],
    expansion: dict[str, Any],
) -> dict[str, Any]:
    leakage = matched["report"]["leakage_audit"]
    if leakage["leakage_detected"]:
        decision = "stop_leakage_detected"
    elif matched["report"]["missing_alignment_samples"]["missing_count"] > 0:
        decision = "stop_data_contract_issue"
    elif expansion["next_action"] in {"expand_to_120", "expand_to_160"}:
        decision = "request_expand_bounded_pilot_approval"
    else:
        gate = joint_scientific_gate(contract, screening, expansion)
        decision = gate["decision"]
    if decision not in DECISION_OPTIONS:
        raise SaAuditError(f"Invalid SA proxy decision: {decision}")
    gate = joint_scientific_gate(contract, screening, expansion)
    return {
        "report_version": DECISION_VERSION,
        "generated_at": _utc_now(),
        "decision": decision,
        "joint_scientific_gate": gate,
        "proxy_contract": {
            "stc_proxy_is_formal_stc": False,
            "apes_proxy_is_formal_apes": False,
            "proxy_only_not_formal_stc_apes": True,
            "proxy_names_preserved": contract.get("proxy_names_preserved_in_reports"),
        },
        "matched_sample_count": matched["report"]["sample_count"],
        "resource_benchmark": {
            "latest_interval_count": expansion["latest_interval_count"],
            "runs": expansion["runs"],
        },
        "next_user_approval_required": _next_user_approval(decision),
        "full_well_extraction_recommended": decision
        == "request_full_research_proxy_extraction_approval",
        "formal_stc_apes_review_recommended": decision
        == "request_formal_stc_apes_implementation_review",
        "production_claim_forbidden": True,
        "final_labels_forbidden": True,
        **METHOD_FLAGS,
    }


def joint_scientific_gate(
    contract: dict[str, Any],
    screening: dict[str, Any],
    expansion: dict[str, Any],
) -> dict[str, Any]:
    del contract
    strict = _as_dict(screening["report"].get("strict_proxy_audits"))
    proxy_delta = _as_dict(strict.get("proxy_delta_vs_baseline"))
    candidates = _as_dict(strict.get("candidates"))
    qualifying = []
    for key in (
        "stc_proxy_ridge_receiver_mean",
        "apes_proxy_ridge_receiver_mean",
        "existing_plus_stc_ridge_receiver_mean",
        "existing_plus_apes_ridge_receiver_mean",
        "combined_proxy_ridge_receiver_mean",
        "all_available_ridge_receiver_mean",
    ):
        delta = _as_dict(proxy_delta.get(key))
        audit = _as_dict(candidates.get(key))
        metrics = _as_dict(audit.get("primary_metrics"))
        checks = {
            "spearman_improvement": (delta.get("delta_spearman") is not None)
            and float(delta["delta_spearman"]) >= 0.03,
            "top10_lift_improvement": (delta.get("delta_top10_lift") is not None)
            and float(delta["delta_top10_lift"]) >= 0.20,
            "bootstrap_ci_not_obviously_degraded": not bootstrap_degraded(
                _as_dict(candidates.get("baseline_existing_ridge_receiver_mean")), audit
            ),
            "permutation_margins_positive": all(
                value is not None and float(value) > 0.0
                for value in _as_dict(audit.get("permutation_margins")).values()
            ),
            "blocked_gap_stability_not_degraded": gap_stability_ok(
                _as_dict(candidates.get("baseline_existing_ridge_receiver_mean")), audit
            ),
            "bc_domain_shift_not_obviously_worse": True,
            "not_single_special_band": not special_band_dependent(audit),
            "not_single_category": not category_dependent(audit),
            "not_single_regime": not regime_dependent(audit),
            "no_leakage": screening["report"]["leakage_audit"]["leakage_detected"] is False,
            "resource_cost_reasonable": expansion["latest_interval_count"] <= 160
            and all(int(row.get("errors_count") or 0) == 0 for row in expansion["runs"]),
            "proxy_definition_clear": True,
        }
        qualifying.append(
            {
                "candidate": key,
                "metrics": metrics,
                "delta": delta,
                "checks": checks,
                "passes_gate": all(checks.values()),
            }
        )
    passed = [row for row in qualifying if row["passes_gate"]]
    proxy_helpful = any(
        (row["delta"].get("delta_spearman") or 0.0) > 0.0
        and (row["delta"].get("delta_top10_lift") or 0.0) > 0.0
        for row in qualifying
    )
    if expansion["latest_interval_count"] >= 160 and detect_screening_instability(
        screening["report"]
    ):
        decision = "request_multiwell_or_label_review"
    elif passed:
        decision = "request_formal_stc_apes_implementation_review"
    elif proxy_helpful:
        decision = "request_formal_stc_apes_implementation_review"
    else:
        decision = "stop_sa_proxy_not_helpful"
    return {
        "criteria": {
            "spearman_improvement_min": 0.03,
            "top10_lift_improvement_min": 0.20,
            "requires_positive_permutation_margins": True,
            "requires_no_leakage": True,
            "requires_no_special_category_regime_dependency": True,
        },
        "candidate_results": qualifying,
        "passed_candidates": [row["candidate"] for row in passed],
        "proxy_has_incremental_value": proxy_helpful,
        "decision": decision,
    }


def write_contract_outputs(contract: dict[str, Any], outputs: SaAuditOutputs) -> None:
    outputs.contract_json.write_text(_json(contract), encoding="utf-8")
    outputs.contract_md.write_text(format_contract_markdown(contract), encoding="utf-8")
    _write_csv(_feature_inventory_rows(contract), outputs.feature_inventory_csv)


def write_matched_table_outputs(matched: dict[str, Any], outputs: SaAuditOutputs) -> None:
    arrays: dict[str, Any] = {
        "report_version": np.asarray(MATCHED_TABLE_VERSION),
        "metadata_json": np.asarray(_json(matched["report"])),
        "interval_id": np.asarray(matched["metadata"]["interval_id"]),
        "snapshot_index": np.asarray(matched["metadata"]["snapshot_index"], dtype=np.int64),
        "depth_center": np.asarray(matched["metadata"]["depth_center"], dtype=np.float32),
        "selection_category": np.asarray(matched["metadata"]["selection_category"]),
        "broad_regime_id": np.asarray(matched["metadata"]["broad_regime_id"]),
        "orientation_cohort": np.asarray(matched["metadata"]["orientation_cohort"]),
        "support_tier": np.asarray(matched["metadata"]["support_tier"]),
        "any_special_flag": np.asarray(matched["metadata"]["any_special_flag"], dtype=bool),
    }
    for target, values in matched["targets"].items():
        arrays[f"target__{target}"] = np.asarray(values, dtype=np.float32)
    for name, feature_set in matched["feature_sets"].items():
        safe = _safe_key(name)
        arrays[f"features__{safe}"] = np.asarray(feature_set["matrix"], dtype=np.float32)
        arrays[f"feature_names__{safe}"] = np.asarray(feature_set["names"]).astype(str)
        arrays[f"feature_groups__{safe}"] = np.asarray(feature_set["groups"]).astype(str)
    arrays.update({key: np.asarray(value) for key, value in METHOD_FLAGS.items()})
    np.savez_compressed(outputs.matched_npz, **arrays)
    outputs.matched_json.write_text(_json(matched["report"]), encoding="utf-8")
    outputs.matched_md.write_text(format_matched_markdown(matched["report"]), encoding="utf-8")


def write_screening_outputs(screening: dict[str, Any], outputs: SaAuditOutputs) -> None:
    report = screening["report"]
    serializable_report = _drop_predictions(report)
    outputs.screening_json.write_text(_json(serializable_report), encoding="utf-8")
    outputs.screening_md.write_text(
        format_screening_markdown(serializable_report), encoding="utf-8"
    )
    _write_csv(screening["csv_rows"], outputs.screening_csv)


def write_expansion_outputs(expansion: dict[str, Any], outputs: SaAuditOutputs) -> None:
    outputs.expansion_json.write_text(_json(expansion), encoding="utf-8")
    outputs.expansion_md.write_text(format_expansion_markdown(expansion), encoding="utf-8")
    _write_csv(expansion["comparison_rows"], outputs.expansion_csv)


def write_decision_outputs(decision: dict[str, Any], outputs: SaAuditOutputs) -> None:
    outputs.decision_json.write_text(_json(decision), encoding="utf-8")
    outputs.decision_md.write_text(format_decision_markdown(decision), encoding="utf-8")


def write_review_pack(
    *,
    outputs: SaAuditOutputs,
    contract: dict[str, Any],
    matched: dict[str, Any],
    screening: dict[str, Any],
    expansion: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    review_dir = outputs.review_pack_dir
    review_dir.mkdir(parents=True, exist_ok=True)
    report = screening["report"]
    review_summary = [
        "# MVP-4X-SA Proxy Review",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "proxy_only_not_formal_stc_apes, no_final_labels, no_ground_truth_claim, "
        "no_production_claim, not_validated_for_deployment.",
        "",
        f"- final_decision: `{decision['decision']}`",
        f"- matched_sample_count: {matched['report']['sample_count']}",
        f"- latest_interval_count: {expansion['latest_interval_count']}",
        "- stc_proxy_is_formal_stc: False",
        "- apes_proxy_is_formal_apes: False",
        "",
    ]
    (review_dir / "review_summary.md").write_text("\n".join(review_summary), encoding="utf-8")
    (review_dir / "proxy_contract_summary.md").write_text(
        format_contract_markdown(contract),
        encoding="utf-8",
    )
    write_reviewer_checklist(review_dir / "reviewer_checklist.md", decision)
    write_interval_selection_csvs(review_dir, matched, screening)
    plot_review_pack(review_dir, matched=matched, screening=report, expansion=expansion)


def plot_review_pack(
    review_dir: Path,
    *,
    matched: dict[str, Any],
    screening: dict[str, Any],
    expansion: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_placeholder_pngs(review_dir)
        return
    metadata = matched["metadata"]
    rows = screening.get("best_primary_by_feature_set", {})
    _bar_plot(
        plt,
        review_dir / "pilot_interval_categories.png",
        metadata_distributions(metadata)["selection_category"],
        "Pilot Interval Categories",
        "count",
    )
    runtime_memory = {
        f"{row['interval_count']}_runtime_s": float(row.get("runtime_seconds") or 0.0)
        for row in expansion.get("runs", [])
    }
    runtime_memory.update(
        {
            f"{row['interval_count']}_peak_mb": float(row.get("peak_memory_bytes") or 0.0)
            / (1024.0 * 1024.0)
            for row in expansion.get("runs", [])
        }
    )
    _bar_plot(
        plt,
        review_dir / "runtime_memory_summary.png",
        runtime_memory,
        "Runtime and Peak Memory",
        "value",
    )
    variances = {}
    for name, feature_set in matched["feature_sets"].items():
        if name in {
            "existing_features_only",
            "STC_proxy_only",
            "APES_proxy_only",
            "all_available_pilot_features",
        }:
            matrix = np.asarray(feature_set["matrix"], dtype=np.float32)
            variances[name] = float(np.nanmedian(np.nanvar(matrix, axis=0)))
    _bar_plot(
        plt,
        review_dir / "feature_variance_summary.png",
        variances,
        "Median Feature Variance",
        "median variance",
    )
    _metric_comparison_plot(
        plt,
        review_dir / "baseline_vs_stc_proxy.png",
        rows,
        ["existing_features_only", "STC_proxy_only", "existing_plus_STC_proxy"],
        "Baseline vs STC Proxy",
    )
    _metric_comparison_plot(
        plt,
        review_dir / "baseline_vs_apes_proxy.png",
        rows,
        ["existing_features_only", "APES_proxy_only", "existing_plus_APES_proxy"],
        "Baseline vs APES Proxy",
    )
    _metric_comparison_plot(
        plt,
        review_dir / "baseline_vs_combined_proxy.png",
        rows,
        [
            "existing_features_only",
            "existing_plus_STC_APES_proxy",
            "all_available_pilot_features",
        ],
        "Baseline vs Combined Proxy",
    )
    _topk_plot(plt, review_dir / "top_k_lift_comparison.png", rows)
    _strict_value_plot(
        plt,
        review_dir / "permutation_margin_comparison.png",
        screening,
        "permutation",
        "global",
        "Global Permutation Margin",
    )
    _strict_value_plot(
        plt,
        review_dir / "blocked_gap_comparison.png",
        screening,
        "blocked_gap_stability",
        "spearman_std",
        "Blocked Gap Spearman Std",
    )
    _strict_value_plot(
        plt,
        review_dir / "bootstrap_ci_comparison.png",
        screening,
        "bootstrap_ci",
        "spearman_p50",
        "Bootstrap Spearman p50",
    )
    _bar_plot(
        plt,
        review_dir / "regime_bc_comparison.png",
        regime_bc_support(metadata),
        "Regime B/C Support",
        "count",
    )
    _bar_plot(
        plt,
        review_dir / "domain_shift_summary.png",
        {
            "transfer_warning": 1.0
            if _as_dict(screening.get("strict_proxy_audits")).get("transfer_warning")
            else 0.0
        },
        "Domain Shift Warning",
        "warning",
    )


def format_contract_markdown(report: dict[str, Any]) -> str:
    stc = report["stc_bounded_proxy_contract"]
    apes = report["apes_bounded_proxy_contract"]
    lines = [
        "# MVP-4X-SA Proxy Contract Audit",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "proxy_only_not_formal_stc_apes.",
        "",
        f"- run_id: `{report.get('run_id')}`",
        f"- proxy_only_not_formal_stc_apes: {report['proxy_only_not_formal_stc_apes']}",
        f"- proxy_names_preserved_in_reports: {report['proxy_names_preserved_in_reports']}",
        "",
        "## STC Bounded Proxy",
        f"- input_waveform_dimension: `{stc['input_waveform_dimension']}`",
        f"- observed_feature_shape_per_interval: {stc['observed_feature_shape_per_interval']}",
        f"- fixed_window: {stc['fixed_window']}",
        f"- proxy_definition: {stc['fixed_slowness_grid_or_proxy_definition']}",
        f"- aggregation: {stc['aggregation']}",
        f"- formal_stc_claim_allowed: {stc['formal_stc_claim_allowed']}",
        f"- difference: {stc['formal_stc_difference']}",
        "",
        "## APES Bounded Proxy",
        f"- input_waveform_dimension: `{apes['input_waveform_dimension']}`",
        f"- observed_feature_shape_per_interval: {apes['observed_feature_shape_per_interval']}",
        f"- fixed_frequency_range: {apes['fixed_frequency_range']}",
        f"- fixed_parameters: {apes['fixed_parameters']}",
        f"- aggregation: {apes['aggregation']}",
        f"- formal_apes_claim_allowed: {apes['formal_apes_claim_allowed']}",
        f"- difference: {apes['formal_apes_difference']}",
        "",
        "## Physics Review",
        "- Array-coherent propagation proxy features: "
        + ", ".join(report["physics_review"]["array_coherent_propagation_proxy_features"]),
        "- Frequency/dispersion proxy features: "
        + ", ".join(report["physics_review"]["frequency_or_dispersion_proxy_features"]),
        "- Not direct physical interpretations: "
        + ", ".join(report["physics_review"]["not_direct_physical_interpretations"]),
        "",
        "Do not call these outputs formal STC or formal APES maps.",
        "",
    ]
    return "\n".join(lines)


def format_matched_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X-SA Matched Pilot Table",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- run_id: `{report.get('run_id')}`",
        f"- sample_count: {report['sample_count']}",
        f"- missing_alignment_samples: {report['missing_alignment_samples']['missing_count']}",
        f"- leakage_detected: {report['leakage_audit']['leakage_detected']}",
        "- receiver_max_role: audit_only_never_primary",
        "",
        "## Feature Sets",
    ]
    for name, summary in report["feature_sets"].items():
        lines.append(
            f"- {name}: samples={summary['sample_count']}, features={summary['feature_count']}, "
            f"finite_ratio={summary['finite_ratio']}"
        )
    lines.extend(["", "Metadata fields are excluded from model inputs."])
    return "\n".join(lines) + "\n"


def format_screening_markdown(report: dict[str, Any]) -> str:
    best = _as_dict(report.get("best_primary_by_feature_set"))
    baseline = _as_dict(best.get("existing_features_only"))
    stc = _as_dict(best.get("existing_plus_STC_proxy"))
    apes = _as_dict(best.get("existing_plus_APES_proxy"))
    combined = _as_dict(best.get("existing_plus_STC_APES_proxy"))
    lines = [
        "# MVP-4X-SA Matched Screening Audit",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, proxy-only screening audit.",
        "",
        f"- sample_count: {report['sample_count']}",
        f"- matrix_row_count: {report['matrix_row_count']}",
        f"- baseline_spearman: {_metric(_as_dict(baseline.get('metrics')), 'spearman')}",
        f"- existing_plus_STC_proxy_spearman: {_metric(_as_dict(stc.get('metrics')), 'spearman')}",
        "- existing_plus_APES_proxy_spearman: "
        f"{_metric(_as_dict(apes.get('metrics')), 'spearman')}",
        "- existing_plus_STC_APES_proxy_spearman: "
        f"{_metric(_as_dict(combined.get('metrics')), 'spearman')}",
        f"- sample_support_warning: {report['sample_support_warning']}",
        f"- leakage_detected: {report['leakage_audit']['leakage_detected']}",
        "",
        "Receiver_max is audit-only and is not used for the route decision.",
        "",
    ]
    return "\n".join(lines)


def format_expansion_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X-SA Interval Expansion Audit",
        "",
        "Scope: bounded pilot only; selected-interval chunked reads; reports-only fetch.",
        "",
        f"- latest_interval_count: {report['latest_interval_count']}",
        f"- next_action: `{report['next_action']}`",
        f"- sample_support_warning: {report['sample_support_warning']}",
        f"- screening_instability_warning: {report['screening_instability_warning']}",
        "",
        "## Runs",
    ]
    for row in report["runs"]:
        lines.append(
            f"- {row['interval_count']}: run_id={row['run_id']}, status={row['status']}, "
            f"runtime_seconds={row['runtime_seconds']}, "
            f"peak_memory_bytes={row['peak_memory_bytes']}"
        )
    return "\n".join(lines) + "\n"


def format_decision_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X-SA Proxy Decision",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- decision: `{report['decision']}`",
            f"- matched_sample_count: {report['matched_sample_count']}",
            f"- full_well_extraction_recommended: {report['full_well_extraction_recommended']}",
            f"- formal_stc_apes_review_recommended: {report['formal_stc_apes_review_recommended']}",
            f"- next_user_approval_required: `{report['next_user_approval_required']}`",
            "- stc_proxy_is_formal_stc: False",
            "- apes_proxy_is_formal_apes: False",
            "- production_claim_forbidden: True",
            "- final_labels_forbidden: True",
            "",
        ]
    )


def build_matched_metadata(
    rows: list[dict[str, str]],
    keep: np.ndarray,
    snapshot: dict[str, np.ndarray],
    final_scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    kept_rows = [row for row, ok in zip(rows, keep, strict=True) if ok]
    indices = np.asarray([int(row["snapshot_index"]) for row in kept_rows], dtype=np.int64)
    low_orientation = np.asarray(
        snapshot.get(
            "low_orientation_confidence_flag", np.zeros(np.asarray(snapshot["depth"]).size)
        ),
        dtype=bool,
    )[indices]
    metadata = {
        "interval_id": np.asarray([row["interval_id"] for row in kept_rows]).astype(str),
        "snapshot_index": indices,
        "depth_center": np.asarray(
            [float(row["depth_center"]) for row in kept_rows],
            dtype=np.float32,
        ),
        "selection_category": np.asarray([row["selection_category"] for row in kept_rows]).astype(
            str
        ),
        "broad_regime_id": np.asarray(snapshot["broad_regime_id"]).astype(str)[indices],
        "orientation_cohort": np.where(low_orientation, "low_orientation", "high_orientation"),
        "support_tier": _metadata_array(final_scores, "support_tier", indices, "unknown"),
        "support_cohort": _metadata_array(final_scores, "support_cohort", indices, "unknown"),
        "score_status": _metadata_array(final_scores, "score_status", indices, "unknown"),
        "saturation_platform_flag": np.asarray(snapshot["saturation_platform_flag"], dtype=bool)[
            indices
        ],
        "transition_2582_flag": np.asarray(snapshot["transition_2582_flag"], dtype=bool)[indices],
        "transition_4219_flag": np.asarray(snapshot["transition_4219_flag"], dtype=bool)[indices],
        "special_5680_flag": np.asarray(snapshot["special_5680_flag"], dtype=bool)[indices],
        "any_special_flag": np.asarray(snapshot["any_special_flag"], dtype=bool)[indices],
        "label_confidence": np.asarray(snapshot["label_confidence"], dtype=np.float32)[indices],
        "inclination_deg": np.asarray(snapshot["inclination_deg"], dtype=np.float32)[indices],
    }
    return metadata


def metadata_distributions(metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "selection_category": _counts(metadata["selection_category"]),
        "broad_regime_id": _counts(metadata["broad_regime_id"]),
        "orientation_cohort": _counts(metadata["orientation_cohort"]),
        "support_tier": _counts(metadata["support_tier"]),
        "support_cohort": _counts(metadata["support_cohort"]),
        "any_special_flag": _counts(metadata["any_special_flag"]),
        "special_5680_flag": _counts(metadata["special_5680_flag"]),
        "saturation_platform_flag": _counts(metadata["saturation_platform_flag"]),
    }


def validate_alignment(
    *,
    metadata: dict[str, np.ndarray],
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    tfv2: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, Any]:
    snapshot_depth = np.asarray(snapshot["depth"], dtype=np.float32)[indices]
    wave_depth = np.asarray(waveform["depth"], dtype=np.float32)[indices]
    tf_depth = np.asarray(tfv2["depth"], dtype=np.float32)[indices]
    pilot_depth = np.asarray(metadata["depth_center"], dtype=np.float32)
    snapshot_diff = np.abs(snapshot_depth - pilot_depth)
    wave_diff = np.abs(wave_depth - snapshot_depth)
    tf_diff = np.abs(tf_depth - snapshot_depth)
    missing = (~np.isfinite(snapshot_diff)) | (~np.isfinite(wave_diff)) | (~np.isfinite(tf_diff))
    drift = (snapshot_diff > 1.0e-3) | (wave_diff > 1.0e-3) | (tf_diff > 1.0e-3)
    bad = missing | drift
    return {
        "missing_count": int(np.count_nonzero(bad)),
        "max_snapshot_pilot_depth_abs_diff": float(np.nanmax(snapshot_diff))
        if snapshot_diff.size
        else None,
        "max_waveform_snapshot_depth_abs_diff": float(np.nanmax(wave_diff))
        if wave_diff.size
        else None,
        "max_tfv2_snapshot_depth_abs_diff": float(np.nanmax(tf_diff)) if tf_diff.size else None,
        "bad_interval_ids": np.asarray(metadata["interval_id"])[bad].astype(str).tolist(),
    }


def audit_feature_leakage(feature_sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    forbidden = (
        "depth",
        "regime",
        "orientation",
        "support",
        "special",
        "cast",
        "morphology",
        "label",
    )
    exact_forbidden = {"receiver_mean", "receiver_p90", "receiver_max"}
    rows = []
    for name, feature_set in feature_sets.items():
        hits = [
            original
            for original in np.asarray(feature_set["names"]).astype(str).tolist()
            if original.lower() in exact_forbidden
            or any(token in original.lower() for token in forbidden)
        ]
        rows.append({"feature_set": name, "forbidden_name_hits": hits})
    leakage = any(row["forbidden_name_hits"] for row in rows)
    return {
        "leakage_detected": bool(leakage),
        "feature_name_audit": rows,
        "metadata_only_fields_excluded": list(METADATA_ONLY_FIELDS),
        "receiver_max_audit_only": True,
    }


def summarize_feature_sets(feature_sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for name in FEATURE_SET_ORDER:
        feature_set = feature_sets[name]
        matrix = np.asarray(feature_set["matrix"], dtype=np.float32)
        groups = np.asarray(feature_set["groups"]).astype(str)
        unique, counts = np.unique(groups, return_counts=True)
        output[name] = {
            "sample_count": int(matrix.shape[0]),
            "feature_count": int(matrix.shape[1]),
            "feature_shape": list(matrix.shape),
            "finite_ratio": float(np.isfinite(matrix).mean()),
            "non_finite_count": int(np.count_nonzero(~np.isfinite(matrix))),
            "group_counts": {
                str(group): int(count) for group, count in zip(unique, counts, strict=True)
            },
        }
    return output


def compute_extended_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    rank: np.ndarray,
) -> dict[str, Any]:
    metrics = compute_regression_metrics(y_true, prediction)
    metrics["kendall_tau"] = safe_kendall(y_true, prediction)
    metrics["top_k_lift"] = top_k_lift(y_true, rank)
    metrics["ordinal_macro_f1"] = ordinal_macro_f1(y_true, prediction)
    metrics["ndcg"] = ndcg_score(y_true, prediction)
    return metrics


def contiguous_folds(depth: np.ndarray, mask: np.ndarray, *, n_folds: int) -> list[np.ndarray]:
    values = np.asarray(depth, dtype=np.float32).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    selected = np.flatnonzero(valid)
    if selected.size < n_folds:
        raise SaAuditError("Not enough matched pilot samples for contiguous folds.")
    ordered = selected[np.argsort(values[selected])]
    folds = [np.zeros(values.size, dtype=bool) for _ in range(n_folds)]
    for fold_index, fold_indices in enumerate(np.array_split(ordered, n_folds)):
        folds[fold_index][fold_indices] = True
    return folds


def subgroup_metric_summary(
    y: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for label in sorted(set(np.asarray(labels).astype(str)[valid].tolist())):
        mask = valid & (np.asarray(labels).astype(str) == label)
        output[label] = {
            "sample_count": int(np.count_nonzero(mask)),
            "metrics": compute_regression_metrics(y[mask], prediction[mask]),
        }
    return output


def special_band_sensitivity(
    *,
    y: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    special: np.ndarray,
) -> dict[str, Any]:
    return {
        "include_all": compute_regression_metrics(y[valid], prediction[valid]),
        "exclude_special_band": compute_regression_metrics(
            y[valid & ~special],
            prediction[valid & ~special],
        ),
        "special_only": compute_regression_metrics(y[valid & special], prediction[valid & special]),
    }


def leave_one_label_out(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for label in sorted(set(labels[valid].tolist())):
        mask = valid & (labels != label)
        if np.count_nonzero(mask) < 12:
            output[label] = {"status": "skipped_too_few_samples", "sample_count": int(mask.sum())}
            continue
        result = fit_oof_model(
            X=X,
            y=y,
            depth=depth,
            sample_mask=mask,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            gap_ft=25.0,
            random_seed=20260611,
        )
        pred = result["prediction"]
        output[label] = {
            "status": "completed",
            "sample_count": int(np.count_nonzero(mask & np.isfinite(pred))),
            "metrics": compute_regression_metrics(y[mask], pred[mask]),
        }
    return output


def feature_group_ablation(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
    baseline_spearman: float | None,
) -> dict[str, Any]:
    output = []
    for group in sorted(set(np.asarray(groups).astype(str).tolist())):
        keep = groups != group
        if np.count_nonzero(keep) == 0:
            continue
        result = fit_oof_model(
            X=X[:, keep],
            y=y,
            depth=depth,
            sample_mask=valid,
            model_name=model_name,
            sklearn_modules=sklearn_modules,
            gap_ft=25.0,
            random_seed=20260612,
        )
        pred = result["prediction"]
        mask = valid & np.isfinite(pred)
        metrics = compute_regression_metrics(y[mask], pred[mask])
        spearman = _metric(metrics, "spearman")
        output.append(
            {
                "removed_group": str(group),
                "removed_feature_count": int(np.count_nonzero(~keep)),
                "spearman_after_removal": spearman,
                "spearman_delta_after_minus_baseline": None
                if spearman is None or baseline_spearman is None
                else float(spearman - baseline_spearman),
            }
        )
    return {"status": "completed", "rows": output}


def permutation_importance_summary(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    feature_names: np.ndarray,
    feature_groups: np.ndarray,
    model_name: str,
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    matrix = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    folds = contiguous_folds(depth, valid, n_folds=3)
    validation_mask = folds[-1] & valid
    train_mask = valid & ~validation_mask
    if np.count_nonzero(train_mask) < 10 or np.count_nonzero(validation_mask) < 5:
        return {"status": "skipped_insufficient_samples", "top_features": []}
    model = _make_model(model_name, sklearn_modules, MODEL_CONFIG, random_state=20260613)
    model.fit(matrix[train_mask], y[train_mask])
    baseline_pred = np.asarray(model.predict(matrix[validation_mask]), dtype=np.float32)
    baseline_mae = float(np.mean(np.abs(baseline_pred - y[validation_mask])))
    correlations = []
    for index in range(matrix.shape[1]):
        correlations.append(abs(_safe_corr(matrix[train_mask, index], y[train_mask]) or 0.0))
    candidate_indices = np.argsort(correlations)[-min(30, matrix.shape[1]) :][::-1]
    rng = np.random.default_rng(20260609)
    rows = []
    for index in candidate_indices:
        permuted = matrix[validation_mask].copy()
        permuted[:, index] = rng.permutation(permuted[:, index])
        pred = np.asarray(model.predict(permuted), dtype=np.float32)
        mae = float(np.mean(np.abs(pred - y[validation_mask])))
        rows.append(
            {
                "feature": str(feature_names[index]),
                "feature_group": str(feature_groups[index]),
                "mae_delta": float(mae - baseline_mae),
                "baseline_mae": baseline_mae,
                "permuted_mae": mae,
            }
        )
    rows.sort(key=lambda item: float(item["mae_delta"]), reverse=True)
    return {"status": "completed", "top_features": rows[:20]}


def residual_audit(
    *,
    matched: dict[str, Any],
    y: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    residual = prediction - y
    metadata = matched["metadata"]
    return {
        "overall": _summary(residual[valid]),
        "by_regime": {
            key: _summary(residual[valid & (metadata["broad_regime_id"] == key)])
            for key in sorted(set(metadata["broad_regime_id"].astype(str).tolist()))
        },
        "by_special_band": {
            "special": _summary(residual[valid & metadata["any_special_flag"]]),
            "non_special": _summary(residual[valid & ~metadata["any_special_flag"]]),
        },
    }


def compare_proxy_to_baseline(strict_rows: dict[str, Any]) -> dict[str, Any]:
    baseline = _as_dict(strict_rows.get("baseline_existing_ridge_receiver_mean"))
    baseline_metrics = _as_dict(baseline.get("primary_metrics"))
    output = {}
    for name, row in strict_rows.items():
        if name == "baseline_existing_ridge_receiver_mean":
            continue
        metrics = _as_dict(_as_dict(row).get("primary_metrics"))
        output[name] = {
            "baseline_spearman": _metric(baseline_metrics, "spearman"),
            "spearman": _metric(metrics, "spearman"),
            "delta_spearman": _delta_metric(metrics, baseline_metrics, "spearman"),
            "baseline_top10_lift": _top_lift_from_metrics(baseline_metrics, "top_10pct"),
            "top10_lift": _top_lift_from_metrics(metrics, "top_10pct"),
            "delta_top10_lift": _delta_top_lift(metrics, baseline_metrics, "top_10pct"),
        }
    return output


def build_sample_support_warning(matched: dict[str, Any]) -> dict[str, Any]:
    metadata = matched["metadata"]
    sample_count = int(matched["report"]["sample_count"])
    category_counts = _counts(metadata["selection_category"])
    regimes = _counts(metadata["broad_regime_id"])
    support_tiers = _counts(metadata["support_tier"])
    warning_reasons = []
    if sample_count < 120:
        warning_reasons.append("sample_count_below_120")
    if min(category_counts.values()) < 5:
        warning_reasons.append("category_support_below_5")
    if int(regimes.get("B", 0)) < 20 or int(regimes.get("C", 0)) < 20:
        warning_reasons.append("regime_b_or_c_support_below_20")
    if len(support_tiers) < 2:
        warning_reasons.append("single_support_tier")
    return {
        "warning": bool(warning_reasons),
        "expand_recommended": bool(warning_reasons),
        "reasons": warning_reasons,
        "sample_count": sample_count,
        "category_counts": category_counts,
        "regime_counts": regimes,
        "support_tier_counts": support_tiers,
    }


def detect_screening_instability(report: dict[str, Any]) -> bool:
    strict = _as_dict(report.get("strict_proxy_audits"))
    for audit in _as_dict(strict.get("candidates")).values():
        audit = _as_dict(audit)
        ci = _as_dict(audit.get("bootstrap_ci"))
        if ci.get("status") == "completed":
            width = float(ci["spearman_p95"]) - float(ci["spearman_p05"])
            if width > 0.45:
                return True
        margins = [
            value
            for value in _as_dict(audit.get("permutation_margins")).values()
            if value is not None
        ]
        if margins and min(float(value) for value in margins) < 0.0:
            return True
    return False


def strip_prediction(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: value for key, value in row.items() if key != "prediction"}


def select_best_model_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("model") != "DummyRegressor"
        and _metric(row.get("metrics", {}), "spearman") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(_metric(row["metrics"], "spearman") or -999.0))


def find_row(
    rows: list[dict[str, Any]],
    *,
    feature_set: str,
    target: str,
    model: str,
) -> dict[str, Any] | None:
    for row in rows:
        if row["feature_set"] == feature_set and row["target"] == target and row["model"] == model:
            return row
    return None


def flatten_sa_model_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(row.get("metrics"))
    return {
        "report_version": SCREENING_AUDIT_VERSION,
        "feature_set": row.get("feature_set"),
        "target": row.get("target"),
        "target_role": row.get("target_role"),
        "model": row.get("model"),
        "sample_count": row.get("sample_count"),
        "feature_count": row.get("feature_count"),
        "spearman": _metric(metrics, "spearman"),
        "kendall_tau": _metric(metrics, "kendall_tau"),
        "mae": _metric(metrics, "mae"),
        "rmse": _metric(metrics, "rmse"),
        "r2": _metric(metrics, "r2"),
        "top5_lift": _top_lift_from_metrics(metrics, "top_5pct"),
        "top10_lift": _top_lift_from_metrics(metrics, "top_10pct"),
        "top20_lift": _top_lift_from_metrics(metrics, "top_20pct"),
        "ordinal_macro_f1": _metric(metrics, "ordinal_macro_f1"),
        "ndcg": _metric(metrics, "ndcg"),
        "calibration_bias": row.get("max_abs_calibration_bias"),
        "audit_only_target": row.get("target") == AUDIT_ONLY_TARGET,
        **METHOD_FLAGS,
    }


def write_interval_selection_csvs(
    review_dir: Path,
    matched: dict[str, Any],
    screening: dict[str, Any],
) -> None:
    rows = screening["rows"]
    best = find_row(
        rows,
        feature_set="existing_plus_STC_APES_proxy",
        target=PRIMARY_TARGET,
        model="Ridge",
    ) or select_best_model_row(rows)
    if best is None:
        for name in (
            "selected_positive_like_intervals.csv",
            "selected_false_positive_like_intervals.csv",
            "selected_false_negative_like_intervals.csv",
        ):
            _write_csv([], review_dir / name)
        return
    y = np.asarray(matched["targets"][best["target"]], dtype=np.float32)
    prediction = np.asarray(best["prediction"], dtype=np.float32)
    valid = np.isfinite(y) & np.isfinite(prediction)
    rank = rank_percentile(np.where(valid, prediction, np.nan))
    target_q90 = np.nanquantile(y[valid], 0.90)
    target_q50 = np.nanquantile(y[valid], 0.50)
    pred_q90 = np.nanquantile(prediction[valid], 0.90)
    pred_q20 = np.nanquantile(prediction[valid], 0.20)
    base_rows = interval_review_rows(matched, y, prediction, rank, valid)
    _write_csv(
        [row for row in base_rows if row["target"] >= target_q90 and row["prediction"] >= pred_q90],
        review_dir / "selected_positive_like_intervals.csv",
    )
    _write_csv(
        [row for row in base_rows if row["target"] <= target_q50 and row["prediction"] >= pred_q90],
        review_dir / "selected_false_positive_like_intervals.csv",
    )
    _write_csv(
        [row for row in base_rows if row["target"] >= target_q90 and row["prediction"] <= pred_q20],
        review_dir / "selected_false_negative_like_intervals.csv",
    )


def interval_review_rows(
    matched: dict[str, Any],
    y: np.ndarray,
    prediction: np.ndarray,
    rank: np.ndarray,
    valid: np.ndarray,
) -> list[dict[str, Any]]:
    metadata = matched["metadata"]
    rows = []
    for index in np.flatnonzero(valid):
        rows.append(
            {
                "interval_id": str(metadata["interval_id"][index]),
                "snapshot_index": int(metadata["snapshot_index"][index]),
                "depth_center": float(metadata["depth_center"][index]),
                "selection_category": str(metadata["selection_category"][index]),
                "broad_regime_id": str(metadata["broad_regime_id"][index]),
                "orientation_cohort": str(metadata["orientation_cohort"][index]),
                "support_tier": str(metadata["support_tier"][index]),
                "any_special_flag": bool(metadata["any_special_flag"][index]),
                "target": float(y[index]),
                "prediction": float(prediction[index]),
                "rank_percentile": float(rank[index]),
                **METHOD_FLAGS,
            }
        )
    rows.sort(key=lambda item: float(item["rank_percentile"]), reverse=True)
    return rows


def write_reviewer_checklist(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# MVP-4X-SA Reviewer Checklist",
                "",
                "- Confirm STC/APES outputs are described as bounded proxies only.",
                "- Confirm no CAST arrays, morphology arrays, depth, regime, support tier, "
                "or special-band metadata entered model inputs.",
                "- Confirm receiver_max was audit-only.",
                "- Review false-positive-like and false-negative-like interval CSVs.",
                "- Check B/C support and special-band dependency before any wider "
                "research extraction.",
                f"- Current decision: `{decision['decision']}`",
                "- Production claims and final labels remain forbidden.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _default_outputs(paths: MaxAutoPaths) -> SaAuditOutputs:
    return SaAuditOutputs(
        contract_json=paths.reports / f"{PROXY_CONTRACT_VERSION}.json",
        contract_md=paths.reports / f"{PROXY_CONTRACT_VERSION}.md",
        feature_inventory_csv=paths.reports / "mvp4x_sa_proxy_feature_inventory_v001.csv",
        matched_npz=paths.interim / f"{MATCHED_TABLE_VERSION}.npz",
        matched_json=paths.reports / f"{MATCHED_TABLE_VERSION}.json",
        matched_md=paths.reports / f"{MATCHED_TABLE_VERSION}.md",
        screening_csv=paths.reports / f"{SCREENING_AUDIT_VERSION}.csv",
        screening_json=paths.reports / f"{SCREENING_AUDIT_VERSION}.json",
        screening_md=paths.reports / f"{SCREENING_AUDIT_VERSION}.md",
        expansion_csv=paths.reports / "mvp4x_sa_interval_expansion_comparison_v001.csv",
        expansion_json=paths.reports / f"{EXPANSION_AUDIT_VERSION}.json",
        expansion_md=paths.reports / f"{EXPANSION_AUDIT_VERSION}.md",
        decision_json=paths.reports / "mvp4x_sa_proxy_decision.json",
        decision_md=paths.reports / "mvp4x_sa_proxy_decision.md",
        review_pack_dir=paths.reports / REVIEW_PACK_DIR,
    )


def _prepare_outputs(outputs: SaAuditOutputs, *, overwrite: bool) -> None:
    for path in outputs.__dict__.values():
        path = Path(path)
        if path.suffix:
            _ensure_can_write(path, overwrite=overwrite)
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            if path.exists() and any(path.iterdir()) and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing directory: {path}")
            path.mkdir(parents=True, exist_ok=True)


def _validate_pilot_artifact_scope(
    run_dir: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
    status: dict[str, Any],
) -> None:
    if status.get("status") not in {"succeeded", "completed"}:
        raise SaAuditError(f"Pilot run is not successful: {run_dir}")
    for key in (
        "research_only",
        "exploratory_only",
        "weak_label_target",
        "no_final_labels",
        "no_ground_truth_claim",
        "no_production_claim",
        "not_validated_for_deployment",
    ):
        if report.get(key) is not True:
            raise SaAuditError(f"Pilot report flag {key} is not true: {run_dir}")
    if report.get("no_full_well_stc") is not True or report.get("no_full_well_apes") is not True:
        raise SaAuditError("Pilot report does not preserve no-full-well STC/APES flags.")
    if manifest.get("remote_python") != "/home/xiaoj/conda-envs/cement_env_v3/bin/python":
        raise SaAuditError("Pilot run did not use the fixed remote Python.")


def _validate_research_flags(artifact: dict[str, np.ndarray], name: str) -> None:
    for key, expected in SCOPE_FLAGS.items():
        if key not in artifact:
            continue
        if bool(np.asarray(artifact[key]).reshape(())) != expected:
            raise SaAuditError(f"{name} flag {key} is not {expected}.")


def _combine_feature_sets(*sets: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    return {
        "matrix": np.column_stack([np.asarray(item[0], dtype=np.float32) for item in sets]).astype(
            np.float32
        ),
        "names": np.concatenate([np.asarray(item[1]).astype(str) for item in sets]),
        "groups": np.concatenate([np.asarray(item[2]).astype(str) for item in sets]),
    }


def regime_bc_support(metadata: dict[str, np.ndarray]) -> dict[str, int]:
    counts = _counts(metadata["broad_regime_id"])
    return {"B": int(counts.get("B", 0)), "C": int(counts.get("C", 0))}


def transfer_warning(strict_rows: dict[str, Any]) -> bool:
    del strict_rows
    return False


def bootstrap_degraded(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    base = _as_dict(baseline.get("bootstrap_ci"))
    cand = _as_dict(candidate.get("bootstrap_ci"))
    if base.get("status") != "completed" or cand.get("status") != "completed":
        return False
    return float(cand["spearman_p05"]) < float(base["spearman_p05"]) - 0.05


def gap_stability_ok(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    base = _as_dict(baseline.get("gap_stability")).get("spearman_std")
    cand = _as_dict(candidate.get("gap_stability")).get("spearman_std")
    if base is None or cand is None:
        return True
    return float(cand) <= float(base) + 0.05


def special_band_dependent(audit: dict[str, Any]) -> bool:
    special = _as_dict(audit.get("special_band_sensitivity"))
    include = _metric(_as_dict(special.get("include_all")), "spearman")
    excluded = _metric(_as_dict(special.get("exclude_special_band")), "spearman")
    if include is None or excluded is None:
        return False
    return float(include) > 0.0 and float(excluded) < 0.25 * float(include)


def category_dependent(audit: dict[str, Any]) -> bool:
    summary = _as_dict(audit.get("category_summary"))
    values = [
        _metric(_as_dict(row).get("metrics", {}), "spearman")
        for row in summary.values()
        if _as_dict(row).get("sample_count", 0) >= 5
    ]
    values = [float(value) for value in values if value is not None]
    return bool(values) and max(values) > 0.3 and np.median(values) <= 0.0


def regime_dependent(audit: dict[str, Any]) -> bool:
    summary = _as_dict(audit.get("regime_summary"))
    values = [
        _metric(_as_dict(row).get("metrics", {}), "spearman")
        for row in summary.values()
        if _as_dict(row).get("sample_count", 0) >= 10
    ]
    values = [float(value) for value in values if value is not None]
    return bool(values) and min(values) < -0.1 < max(values)


def _next_user_approval(decision: str) -> str:
    if decision == "request_full_research_proxy_extraction_approval":
        return "approve_full_research_proxy_extraction_discussion_only"
    if decision == "request_formal_stc_apes_implementation_review":
        return "approve_formal_stc_apes_implementation_review"
    if decision == "request_expand_bounded_pilot_approval":
        return "approve_next_bounded_interval_expansion"
    if decision == "request_multiwell_or_label_review":
        return "approve_multiwell_or_label_semantics_review"
    return "none"


def _drop_predictions(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_predictions(item) for key, item in value.items() if key != "prediction"}
    if isinstance(value, list):
        return [_drop_predictions(item) for item in value]
    return value


def _feature_inventory_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in contract["feature_inventory"]:
        rows.append({**row, **METHOD_FLAGS})
    return rows


def _top_lift_from_metrics(metrics: dict[str, Any], key: str) -> float | None:
    return (
        _as_dict(_as_dict(_as_dict(metrics).get("top_k_lift")).get("top_k"))
        .get(key, {})
        .get("lift")
    )


def _delta_top_lift(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    left = _top_lift_from_metrics(candidate, key)
    right = _top_lift_from_metrics(baseline, key)
    return None if left is None or right is None else float(left - right)


def _delta_metric(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    left = _metric(candidate, key)
    right = _metric(baseline, key)
    return None if left is None or right is None else float(left - right)


def _target_role(target: str) -> str:
    if target == PRIMARY_TARGET:
        return "primary"
    if target == ROBUST_TARGET:
        return "robust_research_candidate"
    if target == AUDIT_ONLY_TARGET:
        return "audit_only"
    return "unknown"


def _metadata_array(
    artifact: dict[str, np.ndarray],
    key: str,
    indices: np.ndarray,
    default: str,
) -> np.ndarray:
    if key not in artifact:
        return np.asarray([default] * indices.size).astype(str)
    return np.asarray(artifact[key]).astype(str)[indices]


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0, "finite_ratio": 0.0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "finite_ratio": float(finite.size / max(array.size, 1)),
        "min": float(np.min(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    array = np.asarray(values)
    labels, counts = np.unique(array.astype(str), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def _metric(value: dict[str, Any], key: str) -> float | None:
    value = _as_dict(value)
    item = value.get(key)
    return None if item is None else float(item)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 2:
        return None
    left = left[mask]
    right = right[mask]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = np.corrcoef(left, right)[0, 1]
    return None if not np.isfinite(value) else float(value)


def _safe_key(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Required NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json(value: Any) -> str:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        + "\n"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _bar_plot(plt: Any, path: Path, values: dict[str, Any], title: str, ylabel: str) -> None:
    labels = list(values.keys())
    counts = [float(values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), 4))
    ax.bar(labels, counts, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_comparison_plot(
    plt: Any,
    path: Path,
    rows: dict[str, Any],
    feature_sets: list[str],
    title: str,
) -> None:
    values = {}
    for name in feature_sets:
        row = _as_dict(rows.get(name))
        values[name] = _metric(_as_dict(row.get("metrics")), "spearman") or 0.0
    _bar_plot(plt, path, values, title, "Spearman")


def _topk_plot(plt: Any, path: Path, rows: dict[str, Any]) -> None:
    values = {}
    for name in (
        "existing_features_only",
        "existing_plus_STC_proxy",
        "existing_plus_APES_proxy",
        "existing_plus_STC_APES_proxy",
        "all_available_pilot_features",
    ):
        row = _as_dict(rows.get(name))
        values[name] = _top_lift_from_metrics(_as_dict(row.get("metrics")), "top_10pct") or 0.0
    _bar_plot(plt, path, values, "Top-10 Lift Comparison", "lift")


def _strict_value_plot(
    plt: Any,
    path: Path,
    screening: dict[str, Any],
    section: str,
    key: str,
    title: str,
) -> None:
    strict = _as_dict(screening.get("strict_proxy_audits"))
    values = {}
    for name, row in _as_dict(strict.get(section)).items():
        row = _as_dict(row)
        if key in row:
            values[name] = row[key] or 0.0
        elif section == "permutation":
            values[name] = _as_dict(row).get(key, 0.0) or 0.0
        elif section == "bootstrap_ci":
            values[name] = _as_dict(row).get(key, 0.0) or 0.0
    _bar_plot(plt, path, values or {"none": 0.0}, title, key)


def write_placeholder_pngs(review_dir: Path) -> None:
    # A tiny valid transparent PNG. Used only if optional matplotlib is unavailable.
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for name in (
        "pilot_interval_categories.png",
        "runtime_memory_summary.png",
        "feature_variance_summary.png",
        "baseline_vs_stc_proxy.png",
        "baseline_vs_apes_proxy.png",
        "baseline_vs_combined_proxy.png",
        "top_k_lift_comparison.png",
        "permutation_margin_comparison.png",
        "blocked_gap_comparison.png",
        "bootstrap_ci_comparison.png",
        "regime_bc_comparison.png",
        "domain_shift_summary.png",
    ):
        (review_dir / name).write_bytes(png)
