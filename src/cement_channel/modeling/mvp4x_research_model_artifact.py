from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import _make_model
from cement_channel.modeling.mvp4x_screening_scores import load_policy_masks

ARTIFACT_VERSION = "mvp4x_research_screening_model_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}


class ResearchModelArtifactError(RuntimeError):
    """Raised when the MVP-4X research model artifact cannot be exported safely."""


def export_research_model_artifact_from_paths(
    *,
    snapshot_npz: Path | str,
    screening_policy_npz: Path | str,
    screening_policy_json: Path | str,
    scores_json: Path | str,
    robustness_json: Path | str,
    ranking_json: Path | str,
    error_json: Path | str,
    config_path: Path | str,
    output_joblib: Path | str,
    output_manifest_json: Path | str,
    output_report_md: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    snapshot = _load_npz(Path(snapshot_npz))
    policy_npz = _load_npz(Path(screening_policy_npz))
    policy_json = _read_json(Path(screening_policy_json))
    scores = _read_json(Path(scores_json))
    robustness = _read_json(Path(robustness_json))
    ranking = _read_json(Path(ranking_json))
    error = _read_json(Path(error_json))
    config = _load_yaml(Path(config_path))
    manifest = export_research_model_artifact(
        snapshot=snapshot,
        policy_npz=policy_npz,
        policy_json=policy_json,
        scores=scores,
        robustness=robustness,
        ranking=ranking,
        error=error,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "screening_policy_npz": str(screening_policy_npz),
            "screening_policy_json": str(screening_policy_json),
            "scores_json": str(scores_json),
            "robustness_json": str(robustness_json),
            "ranking_json": str(ranking_json),
            "error_json": str(error_json),
            "config_path": str(config_path),
        },
        input_hashes={
            "snapshot_npz": _sha256(Path(snapshot_npz)),
            "screening_policy_npz": _sha256(Path(screening_policy_npz)),
            "screening_policy_json": _sha256(Path(screening_policy_json)),
            "scores_json": _sha256(Path(scores_json)),
            "robustness_json": _sha256(Path(robustness_json)),
            "ranking_json": _sha256(Path(ranking_json)),
            "error_json": _sha256(Path(error_json)),
            "config_path": _sha256(Path(config_path)),
        },
        output_joblib=Path(output_joblib),
        overwrite=overwrite,
    )
    write_artifact_reports(
        manifest,
        output_manifest_json=Path(output_manifest_json),
        output_report_md=Path(output_report_md),
        overwrite=overwrite,
    )
    return manifest


def export_research_model_artifact(
    *,
    snapshot: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    policy_json: dict[str, Any],
    scores: dict[str, Any],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
    input_hashes: dict[str, str],
    output_joblib: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _validate_artifacts(snapshot, policy_json, scores, robustness, ranking, error)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    artifact_config = _as_dict(config.get("research_model_artifact"))
    training_cohort = str(artifact_config.get("training_cohort", "pooled_bc_high_orientation"))
    masks = load_policy_masks(policy_npz)
    if training_cohort not in masks:
        raise ResearchModelArtifactError(f"Missing training cohort mask: {training_cohort}")
    X, feature_names = _model_features(snapshot)
    y = np.asarray(snapshot["receiver_mean"], dtype=np.float32).reshape(-1)
    train_mask = masks[training_cohort] & np.isfinite(y)
    if np.count_nonzero(train_mask) < 30:
        raise ResearchModelArtifactError("Training cohort has insufficient support.")
    model = _make_model(
        "Ridge",
        sklearn_modules,
        _as_dict(config.get("screening_baseline")),
        random_state=int(artifact_config.get("random_seed", 20240603)),
    )
    model.fit(X[train_mask], y[train_mask])
    output_joblib.parent.mkdir(parents=True, exist_ok=True)
    _ensure_can_write(output_joblib, overwrite=overwrite)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "model": model,
        "feature_names": feature_names,
        "target": "receiver_mean",
        "training_cohort": training_cohort,
        "scope": dict(RESEARCH_FLAGS),
        "not_for_deployment": True,
    }
    joblib.dump(payload, output_joblib)
    error_summary = _as_dict(error.get("summary"))
    manifest = {
        "manifest_version": f"{ARTIFACT_VERSION}_manifest",
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(output_joblib),
        "inputs": inputs,
        "input_hashes": input_hashes,
        "model_type": "sklearn.pipeline.Pipeline(Ridge)",
        "target": "receiver_mean",
        "feature_schema": "mvp4x_existing_features_only_v001",
        "feature_order": feature_names,
        "feature_count": len(feature_names),
        "training_cohort": training_cohort,
        "training_sample_count": int(np.count_nonzero(train_mask)),
        "unsupported_cohorts": policy_json.get("unsupported_or_audit_only_cohorts", []),
        "cv_protocol": scores.get("leakage_checks", {}),
        "gap_sensitivity": scores.get("gap_stability", {}),
        "permutation_summary": _permutation_summary(robustness),
        "ranking_summary": scores.get("rank_metrics"),
        "calibration_limitations": {
            "absolute_prediction_supported": False,
            "max_abs_calibration_bias": error_summary.get("max_abs_calibration_bias"),
        },
        "domain_shift_warning": bool(error_summary.get("domain_shift_warning")),
        "random_seed": int(artifact_config.get("random_seed", 20240603)),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "modeling_environment": modeling_environment.to_dict(),
        "reproducibility_command": (
            "python scripts/07s_export_mvp4x_research_model_artifact.py --overwrite"
        ),
        "forbidden_use": [
            "deployment",
            "production inference",
            "absolute channel-fraction prediction",
            "final labels",
            "ground-truth claim",
            "cross-well generalization claim",
        ],
        **_method_flags(),
    }
    return manifest


def write_artifact_reports(
    manifest: dict[str, Any],
    *,
    output_manifest_json: Path,
    output_report_md: Path,
    overwrite: bool,
) -> None:
    for path in (output_manifest_json, output_report_md):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_json.write_text(_json(manifest), encoding="utf-8")
    output_report_md.write_text(format_artifact_markdown(manifest), encoding="utf-8")


def format_artifact_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MVP-4X Research Screening Model Artifact",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- artifact_version: `{manifest['artifact_version']}`",
            f"- model_type: {manifest['model_type']}",
            f"- target: {manifest['target']}",
            f"- feature_count: {manifest['feature_count']}",
            f"- training_cohort: {manifest['training_cohort']}",
            f"- training_sample_count: {manifest['training_sample_count']}",
            f"- artifact_path: {manifest['artifact_path']}",
            f"- domain_shift_warning: {manifest['domain_shift_warning']}",
            "",
            "This is an offline research artifact. It is not validated for deployment "
            "and must not be used as a production model.",
            "",
        ]
    )


def _model_features(snapshot: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    features = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    names = np.asarray(snapshot["xsi_feature_names"]).astype(str)
    mask = np.asarray(
        snapshot.get("model_feature_mask", np.ones(names.size, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    X = np.nan_to_num(features[:, mask], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, names[mask].tolist()


def _validate_artifacts(
    snapshot: dict[str, np.ndarray],
    policy_json: dict[str, Any],
    scores: dict[str, Any],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    error: dict[str, Any],
) -> None:
    if snapshot["depth"].shape != (7108,):
        raise ResearchModelArtifactError(f"Unexpected sample shape: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise ResearchModelArtifactError(
            f"Unexpected feature shape: {snapshot['xsi_features'].shape}"
        )
    if str(np.asarray(snapshot["target_kernel"]).item()) != "triangular_midpoint_weighted":
        raise ResearchModelArtifactError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for report_name, report in (
        ("policy_json", policy_json),
        ("scores", scores),
        ("robustness", robustness),
        ("ranking", ranking),
        ("error", error),
    ):
        for flag in RESEARCH_FLAGS:
            if flag == "not_validated_for_deployment" and report_name not in {
                "policy_json",
                "scores",
            }:
                continue
            if report.get(flag) is not True:
                raise ResearchModelArtifactError(f"{report_name} missing flag {flag}.")


def _permutation_summary(robustness: dict[str, Any]) -> dict[str, Any]:
    return {
        row["cohort"]: _as_dict(row.get("permutation")).get("margins")
        for row in robustness.get("candidate_rows", [])
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise ResearchModelArtifactError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ResearchModelArtifactError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ResearchModelArtifactError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise ResearchModelArtifactError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
