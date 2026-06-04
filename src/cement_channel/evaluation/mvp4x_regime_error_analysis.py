from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_autonomous import evaluate_model_cv
from cement_channel.modeling.mvp4x_model_analysis import build_feature_sets
from cement_channel.modeling.mvp4x_regime_robustness import load_policy
from cement_channel.modeling.mvp4x_stratified_baselines import _feature_matrix

REPORT_VERSION = "mvp4x_regime_error_analysis_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}


class RegimeErrorAnalysisError(RuntimeError):
    """Raised when MVP-4X regime error analysis cannot run safely."""


@dataclass(frozen=True)
class RegimeErrorAnalysisOutputs:
    report: dict[str, Any]
    csv_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_regime_error_analysis_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    policy_npz: Path | str,
    robustness_json: Path | str,
    ranking_json: Path | str,
    stratified_decision_json: Path | str,
    config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> RegimeErrorAnalysisOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    policy_npz_data = _load_npz(Path(policy_npz))
    robustness = _read_json(Path(robustness_json))
    ranking = _read_json(Path(ranking_json))
    stratified_decision = _read_json(Path(stratified_decision_json))
    config = _load_yaml(Path(config_path))
    outputs = run_regime_error_analysis(
        snapshot=snapshot,
        waveform=waveform,
        policy_npz=policy_npz_data,
        robustness=robustness,
        ranking=ranking,
        stratified_decision=stratified_decision,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "policy_npz": str(policy_npz),
            "robustness_json": str(robustness_json),
            "ranking_json": str(ranking_json),
            "stratified_decision_json": str(stratified_decision_json),
            "config_path": str(config_path),
        },
    )
    write_error_analysis_outputs(
        outputs,
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return outputs


def run_regime_error_analysis(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    stratified_decision: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
) -> RegimeErrorAnalysisOutputs:
    _validate_artifacts(snapshot, waveform, policy_npz, robustness, ranking, stratified_decision)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    model_config = _as_dict(config.get("robustness"))
    policy = load_policy(policy_npz)
    feature_sets = build_feature_sets(snapshot, waveform)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    rows = []
    csv_rows = []
    for candidate_row in robustness.get("candidate_rows", []):
        candidate = _as_dict(candidate_row.get("candidate"))
        cohort = str(candidate_row.get("cohort"))
        print(
            "MVP-4X error analysis "
            f"cohort={cohort} feature_set={candidate.get('feature_set')} "
            f"target={candidate.get('target')} model={candidate.get('model')}",
            flush=True,
        )
        row = analyze_candidate_errors(
            snapshot=snapshot,
            feature_sets=feature_sets,
            depth=depth,
            masks=policy["masks"],
            cohort=cohort,
            candidate=candidate,
            model_config=model_config,
            sklearn_modules=sklearn_modules,
        )
        rows.append(row)
        csv_rows.append(_csv_row(row))
    summary = summarize_error_analysis(rows, ranking, stratified_decision)
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "modeling_environment": modeling_environment.to_dict(),
        "rows": rows,
        "summary": summary,
        **_method_flags(),
    }
    return RegimeErrorAnalysisOutputs(report=report, csv_rows=csv_rows)


def analyze_candidate_errors(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    cohort: str,
    candidate: dict[str, Any],
    model_config: dict[str, Any],
    sklearn_modules: dict[str, Any],
) -> dict[str, Any]:
    X = _feature_matrix(feature_sets[candidate["feature_set"]])
    y = np.asarray(snapshot[candidate["target"]], dtype=np.float32).reshape(-1)
    mask = masks[cohort] & np.isfinite(y)
    cv = evaluate_model_cv(
        X=X,
        y=y,
        depth=depth,
        sample_mask=mask,
        model_name=candidate["model"],
        sklearn_modules=sklearn_modules,
        config=model_config,
        rng_seed=int(model_config.get("random_seed", 20240603)),
    )
    pred = np.asarray(cv["oof_prediction"], dtype=np.float32)
    valid = mask & np.isfinite(pred)
    residual = pred[valid] - y[valid]
    selected = np.flatnonzero(valid)
    morphology = morphology_score(snapshot)
    return {
        "status": "completed",
        "cohort": cohort,
        "candidate": candidate,
        "sample_count": int(np.count_nonzero(valid)),
        "predicted_vs_target": {
            "target_summary": _summary(y[valid]),
            "prediction_summary": _summary(pred[valid]),
            "pearson": safe_corr(y[valid], pred[valid], method="pearson"),
            "spearman": safe_corr(y[valid], pred[valid], method="spearman"),
        },
        "calibration_by_target_quantile": cv["summary"]["calibration_by_target_quantile"],
        "residual_summary": _summary(residual),
        "residual_vs_depth": safe_corr(depth[valid], residual, method="spearman"),
        "residual_vs_orientation_confidence": safe_corr(
            np.asarray(snapshot["orientation_confidence"], dtype=np.float32)[valid],
            residual,
            method="spearman",
        )
        if "orientation_confidence" in snapshot
        else None,
        "residual_vs_morphology": safe_corr(morphology[valid], residual, method="spearman")
        if morphology is not None
        else None,
        "residual_by_regime": residual_by_group(
            np.asarray(snapshot["broad_regime_id"]).astype(str)[valid],
            residual,
        ),
        "residual_by_special_band": residual_by_group(
            special_labels(snapshot, valid),
            residual,
        ),
        "over_predicted_intervals": interval_rows(
            selected=selected,
            depth=depth,
            target=y,
            prediction=pred,
            residual=pred - y,
            order="over",
            limit=10,
        ),
        "under_predicted_intervals": interval_rows(
            selected=selected,
            depth=depth,
            target=y,
            prediction=pred,
            residual=pred - y,
            order="under",
            limit=10,
        ),
        "high_ranking_intervals": ranking_interval_rows(
            selected=selected,
            depth=depth,
            target=y,
            prediction=pred,
            order="high",
            limit=10,
        ),
        "low_ranking_intervals": ranking_interval_rows(
            selected=selected,
            depth=depth,
            target=y,
            prediction=pred,
            order="low",
            limit=10,
        ),
        **_method_flags(),
    }


def morphology_score(snapshot: dict[str, np.ndarray]) -> np.ndarray | None:
    if "morphology_largest_connected_component_fraction" not in snapshot:
        return None
    values = np.asarray(
        snapshot["morphology_largest_connected_component_fraction"],
        dtype=np.float32,
    )
    return np.nanmax(values, axis=1)


def residual_by_group(labels: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    output = {}
    for label in sorted(set(labels.astype(str).tolist())):
        mask = labels.astype(str) == label
        if not np.any(mask):
            continue
        output[str(label)] = {
            "count": int(np.count_nonzero(mask)),
            "residual_mean": float(np.mean(residual[mask])),
            "residual_mae": float(np.mean(np.abs(residual[mask]))),
            "residual_p95_abs": float(np.quantile(np.abs(residual[mask]), 0.95)),
        }
    return output


def special_labels(snapshot: dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    labels = np.asarray(["none"] * valid.size, dtype=object)
    if "saturation_platform_flag" in snapshot:
        labels[np.asarray(snapshot["saturation_platform_flag"], dtype=bool)] = "2400_2500"
    if "special_5680_flag" in snapshot:
        labels[np.asarray(snapshot["special_5680_flag"], dtype=bool)] = "5680"
    if "any_special_flag" in snapshot:
        labels[
            np.asarray(snapshot["any_special_flag"], dtype=bool)
            & (labels.astype(str) == "none")
        ] = "other_special"
    return labels[valid].astype(str)


def interval_rows(
    *,
    selected: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    residual: np.ndarray,
    order: str,
    limit: int,
) -> list[dict[str, Any]]:
    values = residual[selected]
    local_order = np.argsort(values)
    if order == "over":
        local_order = local_order[::-1]
    rows = []
    for index in selected[local_order[:limit]]:
        rows.append(_interval_row(index, depth, target, prediction, residual))
    return rows


def ranking_interval_rows(
    *,
    selected: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    order: str,
    limit: int,
) -> list[dict[str, Any]]:
    values = prediction[selected]
    local_order = np.argsort(values)
    if order == "high":
        local_order = local_order[::-1]
    residual = prediction - target
    return [
        _interval_row(index, depth, target, prediction, residual)
        for index in selected[local_order[:limit]]
    ]


def safe_corr(a: np.ndarray, b: np.ndarray, *, method: str) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return None
    if float(np.std(a[mask])) == 0.0 or float(np.std(b[mask])) == 0.0:
        return None
    if method == "spearman":
        value = spearmanr(a[mask], b[mask]).correlation
    elif method == "pearson":
        value = pearsonr(a[mask], b[mask]).statistic
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return None if value is None or not np.isfinite(value) else float(value)


def summarize_error_analysis(
    rows: list[dict[str, Any]],
    ranking: dict[str, Any],
    stratified_decision: dict[str, Any],
) -> dict[str, Any]:
    calibration_limits = []
    for row in rows:
        biases = [
            abs(float(item["bias"]))
            for item in row["calibration_by_target_quantile"]
            if item.get("bias") is not None
        ]
        if biases:
            calibration_limits.append(max(biases))
    answers = _as_dict(stratified_decision.get("answers"))
    return {
        "absolute_calibration_insufficient": bool(
            calibration_limits and max(calibration_limits) > 0.05
        ),
        "max_abs_calibration_bias": None
        if not calibration_limits
        else float(max(calibration_limits)),
        "ranking_stable_cohorts": _as_dict(ranking.get("summary")).get(
            "ranking_stable_cohorts",
            [],
        ),
        "top_stable_features": answers.get("11_stable_feature_groups", [])[:10],
        "domain_shift_warning": bool(answers.get("12_domain_shift_exists")),
        "feature_depth_proxy_warning": True,
        "transfer_limits": {
            "b_to_c_spearman": answers.get("4_b_to_c_transfer"),
            "c_to_b_spearman": answers.get("5_c_to_b_transfer"),
        },
    }


def write_error_analysis_outputs(
    outputs: RegimeErrorAnalysisOutputs,
    *,
    output_report_md: Path,
    output_report_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    for path in (output_report_md, output_report_json, output_csv):
        _ensure_can_write(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(_json(outputs.report), encoding="utf-8")
    output_report_md.write_text(format_error_markdown(outputs.report), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(outputs.csv_rows[0].keys()) if outputs.csv_rows else ["status"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs.csv_rows)


def format_error_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Regime Error Analysis",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "no_final_labels, no_ground_truth_claim, no_production_claim.",
        "",
        f"- report_version: `{report['report_version']}`",
        "- absolute_calibration_insufficient: "
        f"{report['summary']['absolute_calibration_insufficient']}",
        f"- domain_shift_warning: {report['summary']['domain_shift_warning']}",
        "",
        "## Cohort Errors",
    ]
    for row in report["rows"]:
        lines.append(
            f"- {row['cohort']}: residual_mean={row['residual_summary'].get('mean')}, "
            f"pred_target_spearman={row['predicted_vs_target']['spearman']}, "
            f"residual_vs_depth={row['residual_vs_depth']}"
        )
    lines.append("")
    return "\n".join(lines)


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cohort": row["cohort"],
        "feature_set": row["candidate"]["feature_set"],
        "target": row["candidate"]["target"],
        "model": row["candidate"]["model"],
        "sample_count": row["sample_count"],
        "predicted_target_spearman": row["predicted_vs_target"]["spearman"],
        "residual_mean": row["residual_summary"].get("mean"),
        "residual_std": row["residual_summary"].get("std"),
        "residual_vs_depth_spearman": row["residual_vs_depth"],
        "residual_vs_orientation_spearman": row["residual_vs_orientation_confidence"],
        "residual_vs_morphology_spearman": row["residual_vs_morphology"],
    }


def _interval_row(
    index: int,
    depth: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    residual: np.ndarray,
) -> dict[str, Any]:
    return {
        "sample_index": int(index),
        "depth_ft": float(depth[index]),
        "target": float(target[index]),
        "prediction": float(prediction[index]),
        "residual": float(residual[index]),
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p50": float(np.quantile(finite, 0.5)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _validate_artifacts(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    robustness: dict[str, Any],
    ranking: dict[str, Any],
    stratified_decision: dict[str, Any],
) -> None:
    if snapshot["depth"].shape != (7108,):
        raise RegimeErrorAnalysisError(f"Unexpected sample shape: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise RegimeErrorAnalysisError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise RegimeErrorAnalysisError(
            f"Unexpected waveform feature shape: {waveform['waveform_depth_features'].shape}"
        )
    for flag in RESEARCH_FLAGS:
        if not bool(np.asarray(snapshot.get(flag, False)).item()):
            raise RegimeErrorAnalysisError(f"snapshot missing research flag {flag}.")
        if not bool(np.asarray(waveform.get(flag, False)).item()):
            raise RegimeErrorAnalysisError(f"waveform missing research flag {flag}.")
        if not bool(np.asarray(policy_npz.get(flag, False)).item()):
            raise RegimeErrorAnalysisError(f"policy missing research flag {flag}.")
        if robustness.get(flag) is not True:
            raise RegimeErrorAnalysisError(f"robustness missing research flag {flag}.")
        if ranking.get(flag) is not True:
            raise RegimeErrorAnalysisError(f"ranking missing research flag {flag}.")
        if stratified_decision.get(flag) is not True:
            raise RegimeErrorAnalysisError(f"stratified decision missing research flag {flag}.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise RegimeErrorAnalysisError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegimeErrorAnalysisError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RegimeErrorAnalysisError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegimeErrorAnalysisError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


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
