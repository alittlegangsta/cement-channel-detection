from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import kendalltau

from cement_channel.modeling.dependencies import require_sklearn_for_modeling
from cement_channel.modeling.mvp4x_autonomous import evaluate_model_cv
from cement_channel.modeling.mvp4x_model_analysis import build_feature_sets
from cement_channel.modeling.mvp4x_regime_robustness import load_policy
from cement_channel.modeling.mvp4x_stratified_baselines import _feature_matrix

REPORT_VERSION = "mvp4x_ranking_audit_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}


class RankingAuditError(RuntimeError):
    """Raised when MVP-4X ranking audit cannot run safely."""


@dataclass(frozen=True)
class RankingAuditOutputs:
    report: dict[str, Any]
    csv_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_ranking_audit_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    policy_npz: Path | str,
    robustness_json: Path | str,
    config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    overwrite: bool = False,
) -> RankingAuditOutputs:
    snapshot = _load_npz(Path(snapshot_npz))
    waveform = _load_npz(Path(waveform_features_npz))
    policy_npz_data = _load_npz(Path(policy_npz))
    robustness = _read_json(Path(robustness_json))
    config = _load_yaml(Path(config_path))
    outputs = run_ranking_audit(
        snapshot=snapshot,
        waveform=waveform,
        policy_npz=policy_npz_data,
        robustness=robustness,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "policy_npz": str(policy_npz),
            "robustness_json": str(robustness_json),
            "config_path": str(config_path),
        },
    )
    write_ranking_outputs(
        outputs,
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return outputs


def run_ranking_audit(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    robustness: dict[str, Any],
    config: dict[str, Any],
    inputs: dict[str, str],
) -> RankingAuditOutputs:
    _validate_artifacts(snapshot, waveform, policy_npz, robustness)
    sklearn_modules, modeling_environment = require_sklearn_for_modeling()
    ranking_config = _as_dict(config.get("ranking_audit"))
    robustness_config = _as_dict(config.get("robustness"))
    policy = load_policy(policy_npz)
    feature_sets = build_feature_sets(snapshot, waveform)
    depth = np.asarray(snapshot["depth"], dtype=np.float32).reshape(-1)
    rows = []
    csv_rows = []
    for candidate_row in robustness.get("candidate_rows", []):
        candidate = _as_dict(candidate_row.get("candidate"))
        cohort = str(candidate_row.get("cohort"))
        print(
            "MVP-4X ranking audit "
            f"cohort={cohort} feature_set={candidate.get('feature_set')} "
            f"target={candidate.get('target')} model={candidate.get('model')}",
            flush=True,
        )
        result = evaluate_candidate_ranking(
            snapshot=snapshot,
            feature_sets=feature_sets,
            depth=depth,
            masks=policy["masks"],
            cohort=cohort,
            candidate=candidate,
            ranking_config=ranking_config,
            model_config=robustness_config,
            sklearn_modules=sklearn_modules,
        )
        rows.append(result)
        csv_rows.append(_csv_row(result))
    summary = summarize_ranking_support(rows)
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "modeling_environment": modeling_environment.to_dict(),
        "row_count": len(rows),
        "rows": rows,
        "summary": summary,
        "derived_ordinal_audit_only": True,
        **_method_flags(),
    }
    return RankingAuditOutputs(report=report, csv_rows=csv_rows)


def evaluate_candidate_ranking(
    *,
    snapshot: dict[str, np.ndarray],
    feature_sets: dict[str, dict[str, Any]],
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    cohort: str,
    candidate: dict[str, Any],
    ranking_config: dict[str, Any],
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
    yv = y[valid]
    pv = pred[valid]
    top_fractions = _as_float_list(ranking_config.get("top_fraction"), [0.05, 0.10, 0.20])
    bottom_fractions = _as_float_list(ranking_config.get("bottom_fraction"), [0.20])
    ranking = {
        "pairwise_concordance": pairwise_concordance(yv, pv, seed=17),
        "kendall_tau": _kendall_tau(yv, pv),
        "ndcg": ndcg_score(yv, pv),
        "top_k": {
            f"top_{int(frac * 100)}pct": top_k_metrics(yv, pv, frac)
            for frac in top_fractions
        },
        "bottom_k": {
            f"bottom_{int(frac * 100)}pct": bottom_k_metrics(yv, pv, frac)
            for frac in bottom_fractions
        },
    }
    ordinal = ordinal_audit(
        y_true=yv,
        y_pred=pv,
        quantiles=_as_float_list(ranking_config.get("ordinal_quantiles"), [1 / 3, 2 / 3]),
    )
    permutation = ranking_permutation_baseline(
        y_true=yv,
        y_pred=pv,
        top_fraction=0.10,
        repeats=int(ranking_config.get("permutation_count", 20)),
        seed=23,
    )
    return {
        "status": "completed",
        "cohort": cohort,
        "candidate": candidate,
        "sample_count": int(yv.size),
        "ranking": ranking,
        "derived_ordinal_audit": ordinal,
        "permutation_baseline": permutation,
        "ranking_stability": {
            "top10_lift_minus_permutation_mean": _subtract(
                ranking["top_k"]["top_10pct"]["lift"],
                permutation["top10_lift_mean"],
            ),
            "kendall_tau_positive": ranking["kendall_tau"] is not None
            and ranking["kendall_tau"] > 0.0,
            "ordinal_macro_f1_minus_permutation_mean": _subtract(
                ordinal["macro_f1"],
                permutation["ordinal_macro_f1_mean"],
            ),
        },
        "overlap_audit": overlap_audit(snapshot=snapshot, sample_mask=valid, prediction=pv),
        "regime_specific_ranking": regime_specific_ranking(
            regimes=np.asarray(snapshot["broad_regime_id"]).astype(str)[valid],
            y_true=yv,
            y_pred=pv,
        ),
        **_method_flags(),
    }


def pairwise_concordance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seed: int,
    max_pairs: int = 20000,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    n = y_true.size
    if n < 2:
        return {"status": "skipped_low_support"}
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    keep = i != j
    dy = y_true[i[keep]] - y_true[j[keep]]
    dp = y_pred[i[keep]] - y_pred[j[keep]]
    non_tie = (dy != 0.0) & (dp != 0.0)
    if not np.any(non_tie):
        return {"status": "skipped_all_ties"}
    concordant = np.sign(dy[non_tie]) == np.sign(dp[non_tie])
    return {
        "status": "completed",
        "pair_count": int(np.count_nonzero(non_tie)),
        "concordance": float(np.mean(concordant)),
    }


def top_k_metrics(y_true: np.ndarray, y_pred: np.ndarray, fraction: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    n = y_true.size
    k = max(1, int(np.ceil(n * fraction)))
    true_top = set(np.argsort(y_true)[-k:].tolist())
    pred_top = set(np.argsort(y_pred)[-k:].tolist())
    hit = len(true_top & pred_top)
    precision = hit / max(1, len(pred_top))
    recall = hit / max(1, len(true_top))
    base_rate = len(true_top) / max(1, n)
    return {
        "k": int(k),
        "fraction": float(fraction),
        "precision": float(precision),
        "recall": float(recall),
        "enrichment": float(precision / base_rate) if base_rate > 0 else None,
        "lift": float(precision / base_rate) if base_rate > 0 else None,
    }


def bottom_k_metrics(y_true: np.ndarray, y_pred: np.ndarray, fraction: float) -> dict[str, Any]:
    n = y_true.size
    k = max(1, int(np.ceil(n * fraction)))
    true_bottom = set(np.argsort(y_true)[:k].tolist())
    pred_bottom = set(np.argsort(y_pred)[:k].tolist())
    hit = len(true_bottom & pred_bottom)
    precision = hit / max(1, len(pred_bottom))
    recall = hit / max(1, len(true_bottom))
    base_rate = len(true_bottom) / max(1, n)
    return {
        "k": int(k),
        "fraction": float(fraction),
        "precision": float(precision),
        "recall": float(recall),
        "enrichment": float(precision / base_rate) if base_rate > 0 else None,
        "lift": float(precision / base_rate) if base_rate > 0 else None,
    }


def ndcg_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    order = np.argsort(y_pred)[::-1]
    ideal = np.argsort(y_true)[::-1]
    discounts = 1.0 / np.log2(np.arange(2, y_true.size + 2))
    dcg = float(np.sum(np.maximum(y_true[order], 0.0) * discounts))
    idcg = float(np.sum(np.maximum(y_true[ideal], 0.0) * discounts))
    return None if idcg == 0.0 else float(dcg / idcg)


def ordinal_audit(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: list[float],
) -> dict[str, Any]:
    thresholds = np.quantile(y_true, quantiles)
    true_class = np.digitize(y_true, thresholds, right=False)
    pred_class = np.digitize(y_pred, thresholds, right=False)
    labels = np.asarray([0, 1, 2])
    confusion = np.zeros((3, 3), dtype=int)
    for truth, pred in zip(true_class, pred_class, strict=True):
        confusion[int(truth), int(pred)] += 1
    recalls = []
    f1_scores = []
    for label in labels:
        tp = confusion[label, label]
        fn = int(np.sum(confusion[label, :])) - tp
        fp = int(np.sum(confusion[:, label])) - tp
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "derived_ordinal_audit_only": True,
        "thresholds": [float(item) for item in thresholds],
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": confusion.tolist(),
    }


def ranking_permutation_baseline(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_fraction: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    top_lift = []
    macro_f1 = []
    for _ in range(max(1, repeats)):
        shuffled = rng.permutation(y_pred)
        top_lift.append(top_k_metrics(y_true, shuffled, top_fraction)["lift"])
        macro_f1.append(
            ordinal_audit(
                y_true=y_true,
                y_pred=shuffled,
                quantiles=[1 / 3, 2 / 3],
            )["macro_f1"]
        )
    return {
        "status": "completed",
        "repeat_count": int(max(1, repeats)),
        "top10_lift_mean": float(np.mean(top_lift)),
        "top10_lift_p95": float(np.quantile(top_lift, 0.95)),
        "ordinal_macro_f1_mean": float(np.mean(macro_f1)),
        "ordinal_macro_f1_p95": float(np.quantile(macro_f1, 0.95)),
    }


def overlap_audit(
    *,
    snapshot: dict[str, np.ndarray],
    sample_mask: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    selected = np.flatnonzero(sample_mask)
    if selected.size == 0:
        return {"status": "skipped_no_samples"}
    top10_count = max(1, int(np.ceil(prediction.size * 0.10)))
    top10_local = set(np.argsort(prediction)[-top10_count:].tolist())
    top10_global = selected[list(top10_local)]
    special = np.asarray(snapshot["any_special_flag"], dtype=bool)
    morphology_mask = np.zeros(sample_mask.shape, dtype=bool)
    if "morphology_largest_connected_component_fraction" in snapshot:
        morph = np.asarray(
            snapshot["morphology_largest_connected_component_fraction"],
            dtype=np.float32,
        )
        score = np.nanmax(morph, axis=1)
        morphology_mask = score >= np.nanquantile(score, 0.95)
    return {
        "status": "completed",
        "top10_count": int(len(top10_global)),
        "top10_special_fraction": float(np.mean(special[top10_global])),
        "top10_morphology_sensitive_fraction": float(np.mean(morphology_mask[top10_global])),
    }


def regime_specific_ranking(
    *,
    regimes: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for regime in sorted(set(regimes.astype(str).tolist())):
        mask = regimes.astype(str) == regime
        if np.count_nonzero(mask) < 30:
            output[regime] = {"status": "skipped_low_support"}
            continue
        output[regime] = {
            "status": "completed",
            "sample_count": int(np.count_nonzero(mask)),
            "kendall_tau": _kendall_tau(y_true[mask], y_pred[mask]),
            "top10": top_k_metrics(y_true[mask], y_pred[mask], 0.10),
        }
    return output


def summarize_ranking_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = []
    unstable = []
    for row in rows:
        top10 = row["ranking"]["top_k"]["top_10pct"]
        tau = row["ranking"]["kendall_tau"]
        lift_margin = row["ranking_stability"]["top10_lift_minus_permutation_mean"]
        ordinal_margin = row["ranking_stability"]["ordinal_macro_f1_minus_permutation_mean"]
        ok = (
            tau is not None
            and tau > 0.0
            and lift_margin is not None
            and lift_margin > 0.0
            and ordinal_margin is not None
            and ordinal_margin > 0.0
            and top10["lift"] is not None
            and top10["lift"] > 1.0
        )
        (stable if ok else unstable).append(row["cohort"])
    return {
        "ranking_stable_cohorts": stable,
        "ranking_unstable_cohorts": unstable,
        "ranking_more_appropriate_than_absolute_regression": True,
        "reason": "positive rank metrics with known negative R2/calibration limits",
    }


def write_ranking_outputs(
    outputs: RankingAuditOutputs,
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
    output_report_md.write_text(format_ranking_markdown(outputs.report), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(outputs.csv_rows[0].keys()) if outputs.csv_rows else ["status"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs.csv_rows)


def format_ranking_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MVP-4X Ranking Audit",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, "
        "no_final_labels, no_ground_truth_claim, no_production_claim.",
        "",
        f"- report_version: `{report['report_version']}`",
        f"- derived_ordinal_audit_only: {report['derived_ordinal_audit_only']}",
        "",
        "## Cohort Ranking",
    ]
    for row in report["rows"]:
        lines.append(
            f"- {row['cohort']}: kendall_tau={row['ranking']['kendall_tau']}, "
            f"top10_lift={row['ranking']['top_k']['top_10pct']['lift']}, "
            f"ordinal_macro_f1={row['derived_ordinal_audit']['macro_f1']}"
        )
    lines.extend(
        [
            "",
            "## Summary",
            f"- stable: {report['summary']['ranking_stable_cohorts']}",
            f"- unstable: {report['summary']['ranking_unstable_cohorts']}",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cohort": row["cohort"],
        "feature_set": row["candidate"]["feature_set"],
        "target": row["candidate"]["target"],
        "model": row["candidate"]["model"],
        "sample_count": row["sample_count"],
        "kendall_tau": row["ranking"]["kendall_tau"],
        "pairwise_concordance": row["ranking"]["pairwise_concordance"].get("concordance"),
        "ndcg": row["ranking"]["ndcg"],
        "top5_lift": row["ranking"]["top_k"]["top_5pct"]["lift"],
        "top10_lift": row["ranking"]["top_k"]["top_10pct"]["lift"],
        "top20_lift": row["ranking"]["top_k"]["top_20pct"]["lift"],
        "ordinal_balanced_accuracy": row["derived_ordinal_audit"]["balanced_accuracy"],
        "ordinal_macro_f1": row["derived_ordinal_audit"]["macro_f1"],
        "top10_lift_minus_permutation": row["ranking_stability"][
            "top10_lift_minus_permutation_mean"
        ],
        "ordinal_macro_f1_minus_permutation": row["ranking_stability"][
            "ordinal_macro_f1_minus_permutation_mean"
        ],
        "top10_special_fraction": row["overlap_audit"].get("top10_special_fraction"),
        "top10_morphology_sensitive_fraction": row["overlap_audit"].get(
            "top10_morphology_sensitive_fraction"
        ),
    }


def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    value = kendalltau(y_true, y_pred).correlation
    return None if value is None or not np.isfinite(value) else float(value)


def _validate_artifacts(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    policy_npz: dict[str, np.ndarray],
    robustness: dict[str, Any],
) -> None:
    if snapshot["depth"].shape != (7108,):
        raise RankingAuditError(f"Unexpected sample shape: {snapshot['depth'].shape}")
    if snapshot["xsi_features"].shape != (7108, 80):
        raise RankingAuditError(
            f"Unexpected existing feature shape: {snapshot['xsi_features'].shape}"
        )
    if waveform["waveform_depth_features"].shape != (7108, 342):
        raise RankingAuditError(
            f"Unexpected waveform feature shape: {waveform['waveform_depth_features'].shape}"
        )
    if str(snapshot["target_kernel"]) != "triangular_midpoint_weighted":
        raise RankingAuditError(f"Unexpected target kernel: {snapshot['target_kernel']}")
    for flag in RESEARCH_FLAGS:
        if not bool(np.asarray(snapshot.get(flag, False)).item()):
            raise RankingAuditError(f"snapshot missing research flag {flag}.")
        if not bool(np.asarray(waveform.get(flag, False)).item()):
            raise RankingAuditError(f"waveform missing research flag {flag}.")
        if not bool(np.asarray(policy_npz.get(flag, False)).item()):
            raise RankingAuditError(f"policy missing research flag {flag}.")
        if robustness.get(flag) is not True:
            raise RankingAuditError(f"robustness missing research flag {flag}.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise RankingAuditError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RankingAuditError(f"Missing JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RankingAuditError(f"JSON must contain an object: {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RankingAuditError(f"Missing YAML config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _subtract(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


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
