from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_baselines import (
    _make_model,
    _target_summary,
    compute_regression_metrics,
    contiguous_depth_folds,
)

SCORES_VERSION = "mvp4x_screening_scores_oof_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}
POLICY_CANDIDATES = {
    "P0": "pooled_bc_all",
    "P1": "pooled_bc_high_orientation",
    "P2": "regime_b_high_orientation",
    "P3": "regime_c_all",
    "P4": "regime_c_high_orientation",
}
POLICY_PRIORITY = ("P1", "P2", "P3", "P4", "P0")


class ScreeningScoresError(RuntimeError):
    """Raised when leakage-safe MVP-4X screening scores cannot be generated."""


@dataclass(frozen=True)
class ScreeningScoresOutputs:
    report: dict[str, Any]
    rows: list[dict[str, Any]]
    arrays: dict[str, np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "row_count": len(self.rows),
            "arrays": {key: list(value.shape) for key, value in self.arrays.items()},
        }


def generate_screening_scores_from_paths(
    *,
    snapshot_npz: Path | str,
    screening_policy_npz: Path | str,
    screening_policy_json: Path | str,
    config_path: Path | str,
    output_npz: Path | str,
    output_csv: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    overwrite: bool = False,
) -> ScreeningScoresOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    policy_npz = _load_npz(Path(screening_policy_npz))
    policy_json = _read_json(Path(screening_policy_json))
    config = _load_yaml(Path(config_path))
    outputs = generate_screening_scores(
        snapshot=snapshot,
        policy_npz=policy_npz,
        policy_json=policy_json,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "screening_policy_npz": str(screening_policy_npz),
            "screening_policy_json": str(screening_policy_json),
            "config_path": str(config_path),
        },
    )
    write_screening_score_outputs(
        outputs,
        output_npz=Path(output_npz),
        output_csv=Path(output_csv),
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        overwrite=overwrite,
    )
    return outputs


def generate_screening_scores(
    *,
    snapshot: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    policy_json: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
) -> ScreeningScoresOutputs:
    _validate_artifacts(snapshot, policy_npz, policy_json)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    score_config = _as_dict(config.get("screening_baseline"))
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    y = np.asarray(snapshot["receiver_mean"], dtype=np.float32).reshape(-1)
    X, feature_names = _model_features(snapshot)
    masks = load_policy_masks(policy_npz)
    policy_names = list(POLICY_CANDIDATES)
    policy_scores = np.full((depth.size, len(policy_names)), np.nan, dtype=np.float32)
    policy_fold_ids = np.full((depth.size, len(policy_names)), -1, dtype=np.int16)
    policy_metrics: dict[str, Any] = {}
    gap_scores = {
        10.0: np.full((depth.size, len(policy_names)), np.nan, dtype=np.float32),
        25.0: np.full((depth.size, len(policy_names)), np.nan, dtype=np.float32),
        50.0: np.full((depth.size, len(policy_names)), np.nan, dtype=np.float32),
    }
    for policy_index, policy_name in enumerate(policy_names):
        cohort = POLICY_CANDIDATES[policy_name]
        sample_mask = masks[cohort] & np.isfinite(y)
        result = fit_oof_scores(
            X=X,
            y=y,
            depth=depth,
            sample_mask=sample_mask,
            config=score_config,
            sklearn_modules=sklearn_modules,
            gap_ft=0.0,
            random_seed=int(score_config.get("random_seed", 20240603)) + policy_index,
        )
        policy_scores[:, policy_index] = result["prediction"]
        policy_fold_ids[:, policy_index] = result["fold_id"]
        policy_metrics[policy_name] = {
            "cohort": cohort,
            "sample_count": int(np.count_nonzero(sample_mask)),
            "oof_metrics": result["metrics"],
            "folds": result["folds"],
        }
        for gap in gap_scores:
            gap_result = fit_oof_scores(
                X=X,
                y=y,
                depth=depth,
                sample_mask=sample_mask,
                config=score_config,
                sklearn_modules=sklearn_modules,
                gap_ft=float(gap),
                random_seed=int(score_config.get("random_seed", 20240603))
                + policy_index
                + int(gap),
            )
            gap_scores[gap][:, policy_index] = gap_result["prediction"]
            policy_metrics[policy_name][f"gap_{int(gap)}_ft"] = {
                "metrics": gap_result["metrics"],
                "folds": gap_result["folds"],
            }
    selected_policy, score, fold_id = select_primary_scores(policy_scores, policy_fold_ids)
    rank = rank_percentile(score)
    gap_10 = select_by_policy(gap_scores[10.0], selected_policy, policy_names)
    gap_25 = select_by_policy(gap_scores[25.0], selected_policy, policy_names)
    gap_50 = select_by_policy(gap_scores[50.0], selected_policy, policy_names)
    stability = score_stability_std(score, gap_10, gap_25, gap_50)
    supported = np.isfinite(score)
    residual = score - y
    rows = build_csv_rows(
        snapshot=snapshot,
        score=score,
        rank=rank,
        residual=residual,
        selected_policy=selected_policy,
        fold_id=fold_id,
        gap_10=gap_10,
        gap_25=gap_25,
        gap_50=gap_50,
        stability=stability,
        supported=supported,
    )
    report = {
        "report_version": SCORES_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "modeling_environment": modeling_environment.to_dict(),
        "target": "receiver_mean",
        "model": "Ridge",
        "feature_set": "existing_features_only",
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "policy_candidates": {
            name: {"cohort": cohort, "model": "Ridge", "target": "receiver_mean"}
            for name, cohort in POLICY_CANDIDATES.items()
        },
        "policy_metrics": policy_metrics,
        "selected_score_metrics": compute_regression_metrics(y[supported], score[supported]),
        "rank_metrics": top_k_lift(y[supported], rank[supported]),
        "gap_stability": {
            "score_stability_std": _target_summary(stability[supported]),
            "warning_count": int(np.count_nonzero(stability[supported] > 0.05)),
        },
        "sample_counts": {
            "total": int(depth.size),
            "supported_scored": int(np.count_nonzero(supported)),
            "audit_only_or_unsupported": int(np.count_nonzero(~supported)),
        },
        "leakage_checks": {
            "scaler_fit_on_train_folds_only": True,
            "ridge_fit_on_train_folds_only": True,
            "validation_fold_predict_only": True,
            "metadata_not_model_input": True,
            "model_input_feature_set": "existing_features_only",
        },
        **_method_flags(),
    }
    arrays = {
        "depth": depth,
        "score": score,
        "score_rank_percentile": rank,
        "residual": residual,
        "fold_id": fold_id,
        "gap_10_score": gap_10,
        "gap_25_score": gap_25,
        "gap_50_score": gap_50,
        "score_stability_std": stability,
        "policy_scores": policy_scores,
        "policy_fold_ids": policy_fold_ids,
        "policy_names": np.asarray(policy_names),
        "selected_policy": selected_policy,
        "target_receiver_mean": y,
        "target_receiver_p90": np.asarray(snapshot["receiver_p90"], dtype=np.float32),
        "target_receiver_max": np.asarray(snapshot["receiver_max"], dtype=np.float32),
    }
    return ScreeningScoresOutputs(report=report, rows=rows, arrays=arrays)


def fit_oof_scores(
    *,
    X: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    sample_mask: np.ndarray,
    config: dict[str, Any],
    sklearn_modules: dict[str, Any],
    gap_ft: float,
    random_seed: int,
) -> dict[str, Any]:
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1) & np.all(np.isfinite(X), axis=1)
    folds = contiguous_depth_folds(depth, mask, n_folds=int(config.get("n_contiguous_folds", 3)))
    prediction = np.full(y.shape, np.nan, dtype=np.float32)
    fold_id = np.full(y.shape, -1, dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, validation_mask in enumerate(folds):
        validation_mask = validation_mask & mask
        train_mask = mask & ~validation_mask
        if gap_ft > 0.0 and np.any(validation_mask):
            low = float(np.min(depth[validation_mask])) - float(gap_ft)
            high = float(np.max(depth[validation_mask])) + float(gap_ft)
            train_mask = train_mask & ((depth < low) | (depth > high))
        if np.count_nonzero(train_mask) == 0 or np.count_nonzero(validation_mask) == 0:
            continue
        model = _make_model(
            "Ridge",
            sklearn_modules,
            config,
            random_state=random_seed + fold_index,
        )
        model.fit(X[train_mask], y[train_mask])
        pred = np.asarray(model.predict(X[validation_mask]), dtype=np.float32)
        prediction[validation_mask] = pred
        fold_id[validation_mask] = fold_index
        metrics = compute_regression_metrics(y[validation_mask], pred)
        fold_rows.append(
            {
                "fold_id": int(fold_index),
                "train_count": int(np.count_nonzero(train_mask)),
                "validation_count": int(np.count_nonzero(validation_mask)),
                "validation_depth_min": float(np.min(depth[validation_mask])),
                "validation_depth_max": float(np.max(depth[validation_mask])),
                **metrics,
            }
        )
    return {
        "prediction": prediction,
        "fold_id": fold_id,
        "folds": fold_rows,
        "metrics": compute_regression_metrics(y[mask], prediction[mask]),
    }


def load_policy_masks(policy_npz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = np.asarray(policy_npz["cohort_names"]).astype(str).tolist()
    masks = np.asarray(policy_npz["cohort_masks"], dtype=bool)
    if masks.ndim != 2 or masks.shape[0] != len(names):
        raise ScreeningScoresError("screening policy masks must have shape [cohort, sample].")
    return {name: masks[index] for index, name in enumerate(names)}


def select_primary_scores(
    policy_scores: np.ndarray,
    policy_fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_policy = np.full(policy_scores.shape[0], "", dtype="<U8")
    score = np.full(policy_scores.shape[0], np.nan, dtype=np.float32)
    fold_id = np.full(policy_scores.shape[0], -1, dtype=np.int16)
    policy_names = list(POLICY_CANDIDATES)
    for policy in POLICY_PRIORITY:
        index = policy_names.index(policy)
        take = ~np.isfinite(score) & np.isfinite(policy_scores[:, index])
        selected_policy[take] = policy
        score[take] = policy_scores[take, index]
        fold_id[take] = policy_fold_ids[take, index]
    return selected_policy, score, fold_id


def select_by_policy(
    matrix: np.ndarray,
    selected_policy: np.ndarray,
    policy_names: list[str],
) -> np.ndarray:
    output = np.full(matrix.shape[0], np.nan, dtype=np.float32)
    for index, name in enumerate(policy_names):
        take = selected_policy == name
        output[take] = matrix[take, index]
    return output


def rank_percentile(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    output = np.full(values.shape, np.nan, dtype=np.float32)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return output
    order = finite[np.argsort(values[finite])]
    if order.size == 1:
        output[order] = 1.0
        return output
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[np.arange(order.size)] = np.arange(order.size, dtype=np.float32) / float(
        order.size - 1
    )
    output[order] = ranks
    return output


def top_k_lift(y_true: np.ndarray, rank: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float32)
    r = np.asarray(rank, dtype=np.float32)
    mask = np.isfinite(y) & np.isfinite(r)
    y = y[mask]
    r = r[mask]
    if y.size == 0:
        return {"status": "skipped_no_samples"}
    rows = {}
    target_top = y >= np.quantile(y, 0.90)
    prevalence = float(np.mean(target_top))
    for fraction in (0.05, 0.10, 0.20):
        selected = r >= 1.0 - fraction
        precision = None if not np.any(selected) else float(np.mean(target_top[selected]))
        rows[f"top_{int(fraction * 100)}pct"] = {
            "selected_count": int(np.count_nonzero(selected)),
            "precision": precision,
            "lift": None
            if precision is None or prevalence == 0.0
            else float(precision / prevalence),
        }
    return {"status": "completed", "target_top_10pct_prevalence": prevalence, "top_k": rows}


def score_stability_std(*arrays: np.ndarray) -> np.ndarray:
    matrix = np.vstack([np.asarray(array, dtype=np.float32) for array in arrays])
    output = np.full(matrix.shape[1], np.nan, dtype=np.float32)
    for index in range(matrix.shape[1]):
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            output[index] = float(np.std(values))
    return output


def build_csv_rows(
    *,
    snapshot: dict[str, np.ndarray],
    score: np.ndarray,
    rank: np.ndarray,
    residual: np.ndarray,
    selected_policy: np.ndarray,
    fold_id: np.ndarray,
    gap_10: np.ndarray,
    gap_25: np.ndarray,
    gap_50: np.ndarray,
    stability: np.ndarray,
    supported: np.ndarray,
) -> list[dict[str, Any]]:
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    regimes = np.asarray(snapshot["broad_regime_id"]).astype(str).reshape(-1)
    low = np.asarray(snapshot["low_orientation_confidence_flag"], dtype=bool).reshape(-1)
    targets = {
        "target_receiver_mean": np.asarray(snapshot["receiver_mean"], dtype=np.float32),
        "target_receiver_p90": np.asarray(snapshot["receiver_p90"], dtype=np.float32),
        "target_receiver_max": np.asarray(snapshot["receiver_max"], dtype=np.float32),
    }
    rows: list[dict[str, Any]] = []
    for index in range(depth.size):
        flags = {
            "saturation_platform_2400_2500": bool(
                np.asarray(snapshot["saturation_platform_flag"], dtype=bool)[index]
            ),
            "transition_2582": bool(
                np.asarray(snapshot["transition_2582_flag"], dtype=bool)[index]
            ),
            "transition_4219": bool(
                np.asarray(snapshot["transition_4219_flag"], dtype=bool)[index]
            ),
            "special_band_5680": bool(np.asarray(snapshot["special_5680_flag"], dtype=bool)[index]),
            "any_special": bool(np.asarray(snapshot["any_special_flag"], dtype=bool)[index]),
        }
        orientation = "low_orientation" if low[index] else "high_orientation"
        rows.append(
            {
                "depth": float(depth[index]),
                "regime_id": regimes[index],
                "orientation_cohort": orientation,
                "supported_cohort": str(selected_policy[index]) if supported[index] else "",
                "score_status": "supported_screening_oof"
                if supported[index]
                else "audit_only_or_unsupported",
                "fold_id": int(fold_id[index]),
                "score": _float_or_none(score[index]),
                "score_rank_percentile": _float_or_none(rank[index]),
                "target_receiver_mean": _float_or_none(targets["target_receiver_mean"][index]),
                "target_receiver_p90": _float_or_none(targets["target_receiver_p90"][index]),
                "target_receiver_max": _float_or_none(targets["target_receiver_max"][index]),
                "residual": _float_or_none(residual[index]),
                "gap_10_score": _float_or_none(gap_10[index]),
                "gap_25_score": _float_or_none(gap_25[index]),
                "gap_50_score": _float_or_none(gap_50[index]),
                "score_stability_std": _float_or_none(stability[index]),
                "score_stability_warning": bool(
                    np.isfinite(stability[index]) and stability[index] > 0.05
                ),
                "special_band_flags": json.dumps(flags, sort_keys=True),
                "review_only": True,
                "no_final_labels": True,
                "research_only": True,
                "no_ground_truth_claim": True,
                "no_production_claim": True,
                "not_validated_for_deployment": True,
            }
        )
    return rows


def write_screening_score_outputs(
    outputs: ScreeningScoresOutputs,
    *,
    output_npz: Path,
    output_csv: Path,
    output_report_md: Path,
    output_report_json: Path,
    overwrite: bool,
) -> None:
    for path in (output_npz, output_csv, output_report_md, output_report_json):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        report_version=np.asarray(SCORES_VERSION),
        metadata_json=np.asarray(_json(outputs.report)),
        **outputs.arrays,
        **{key: np.asarray(value) for key, value in _method_flags().items()},
    )
    output_report_json.write_text(_json(outputs.report), encoding="utf-8")
    output_report_md.write_text(format_screening_scores_markdown(outputs.report), encoding="utf-8")
    _write_csv(outputs.rows, output_csv)


def format_screening_scores_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Leakage-Safe OOF Screening Scores",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
        "",
        f"- report_version: `{report['report_version']}`",
        f"- target: `{report['target']}`",
        f"- model: `{report['model']}`",
        f"- feature_set: `{report['feature_set']}`",
        f"- supported_scored: {report['sample_counts']['supported_scored']}",
        f"- audit_only_or_unsupported: {report['sample_counts']['audit_only_or_unsupported']}",
        f"- selected_score_metrics: {report['selected_score_metrics']}",
        f"- rank_metrics: {report['rank_metrics']}",
        "",
        "Scores are out-of-fold research screening scores, not final labels and not "
        "validated for deployment.",
        "",
    ]
    return "\n".join(lines)


def _model_features(snapshot: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    features = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    names = np.asarray(snapshot["xsi_feature_names"]).astype(str)
    mask = np.asarray(
        snapshot.get("model_feature_mask", np.ones(names.size, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if mask.size != names.size:
        raise ScreeningScoresError("model_feature_mask length does not match feature names.")
    X = np.nan_to_num(features[:, mask], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.all(np.isfinite(X)):
        raise ScreeningScoresError("Model feature matrix is not finite after fill.")
    return X, names[mask].tolist()


def _validate_artifacts(
    snapshot: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    policy_json: dict[str, Any],
) -> None:
    if snapshot["depth"].shape != (7108,):
        raise ScreeningScoresError(f"Unexpected sample shape: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise ScreeningScoresError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if _finite_ratio(snapshot["xsi_features"]) != 1.0:
        raise ScreeningScoresError("Existing feature finite ratio is not 1.0.")
    if str(np.asarray(snapshot["target_kernel"]).item()) != "triangular_midpoint_weighted":
        raise ScreeningScoresError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for key in (
        "receiver_mean",
        "receiver_p90",
        "receiver_max",
        "broad_regime_id",
        "low_orientation_confidence_flag",
    ):
        if key not in snapshot:
            raise ScreeningScoresError(f"Missing snapshot key: {key}")
    for flag in RESEARCH_FLAGS:
        if flag == "not_validated_for_deployment":
            continue
        if not bool(np.asarray(snapshot.get(flag, False)).item()):
            raise ScreeningScoresError(f"snapshot missing research flag {flag}.")
    for flag in RESEARCH_FLAGS:
        if policy_json.get(flag) is not True:
            raise ScreeningScoresError(f"policy_json missing research flag {flag}.")
    if policy_npz["cohort_masks"].shape[1] != snapshot["depth"].shape[0]:
        raise ScreeningScoresError("Policy cohort masks do not match snapshot sample count.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise ScreeningScoresError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ScreeningScoresError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ScreeningScoresError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise ScreeningScoresError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


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
