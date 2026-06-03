from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.modeling.mvp4x_baselines import (
    BASELINE_CSV_VERSION,
    run_mvp4x_baselines,
)

ENHANCED_REPORT_VERSION = "mvp4x_enhanced_feature_baselines_v001"
MODEL_REVIEW_VERSION = "mvp4x_model_review_v001"
RESEARCH_FLAGS = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
}


@dataclass(frozen=True)
class EnhancedBaselineReport:
    report_version: str
    generated_at: str
    inputs: dict[str, str]
    feature_sets: dict[str, dict[str, Any]]
    sub_reports: dict[str, dict[str, Any]]
    best_result: dict[str, Any] | None
    feature_group_ablation: dict[str, Any]
    permutation_importance: dict[str, Any]
    top_30_features: list[dict[str, Any]]
    top_10_stable_features: list[dict[str, Any]]
    regime_specific_error_analysis: dict[str, Any]
    special_band_error_analysis: dict[str, Any]
    review_dir: str
    review_files: dict[str, str]
    warnings: list[str]
    errors: list[str]
    research_only: bool
    exploratory_only: bool
    weak_label_target: bool
    no_final_labels: bool
    no_ground_truth_claim: bool
    no_production_claim: bool
    production_training: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_enhanced_feature_baselines_from_paths(
    *,
    snapshot_npz: Path | str,
    waveform_features_npz: Path | str,
    config_path: Path | str,
    output_report_md: Path | str,
    output_report_json: Path | str,
    output_csv: Path | str,
    output_review_dir: Path | str,
    overwrite: bool = False,
) -> EnhancedBaselineReport:
    config = _load_yaml(config_path)
    snapshot = _load_npz(snapshot_npz)
    waveform = _load_npz(waveform_features_npz)
    report, rows, review_files = run_enhanced_feature_baselines(
        snapshot=snapshot,
        waveform=waveform,
        config=config,
        inputs={
            "snapshot_npz": str(snapshot_npz),
            "waveform_features_npz": str(waveform_features_npz),
            "config_path": str(config_path),
        },
        review_dir=Path(output_review_dir),
        overwrite=overwrite,
    )
    write_enhanced_baseline_outputs(
        report,
        rows,
        review_files=review_files,
        output_report_md=Path(output_report_md),
        output_report_json=Path(output_report_json),
        output_csv=Path(output_csv),
        overwrite=overwrite,
    )
    return report


def run_enhanced_feature_baselines(
    *,
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
    config: dict[str, Any],
    inputs: dict[str, str],
    review_dir: Path,
    overwrite: bool,
) -> tuple[EnhancedBaselineReport, list[dict[str, Any]], dict[str, str]]:
    warnings: list[str] = []
    errors: list[str] = []
    feature_sets = build_feature_sets(snapshot, waveform)
    rows: list[dict[str, Any]] = []
    sub_reports: dict[str, dict[str, Any]] = {}
    for feature_set_name, feature_set in feature_sets.items():
        augmented = dict(snapshot)
        augmented["active_features"] = feature_set["matrix"]
        augmented["active_feature_names"] = feature_set["names"]
        report, set_rows = run_mvp4x_baselines(
            snapshot=augmented,
            config=config,
            inputs=inputs,
            feature_set_name=feature_set_name,
            feature_matrix_key="active_features",
            feature_name_key="active_feature_names",
        )
        sub_reports[feature_set_name] = report.to_dict()
        warnings.extend(f"{feature_set_name}: {warning}" for warning in report.warnings)
        errors.extend(f"{feature_set_name}: {error}" for error in report.errors)
        rows.extend(set_rows)
    best = _best_enhanced_result(sub_reports)
    feature_group_summary = summarize_feature_groups(feature_sets)
    analysis = _analysis_placeholders(sub_reports)
    review_files = write_model_review_dir(
        review_dir=review_dir,
        feature_group_summary=feature_group_summary,
        analysis=analysis,
        overwrite=overwrite,
    )
    report = EnhancedBaselineReport(
        report_version=ENHANCED_REPORT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        feature_sets=feature_group_summary,
        sub_reports=sub_reports,
        best_result=best,
        feature_group_ablation=analysis["feature_group_ablation"],
        permutation_importance=analysis["permutation_importance"],
        top_30_features=analysis["top_30_features"],
        top_10_stable_features=analysis["top_10_stable_features"],
        regime_specific_error_analysis=analysis["regime_specific_error_analysis"],
        special_band_error_analysis=analysis["special_band_error_analysis"],
        review_dir=str(review_dir),
        review_files=review_files,
        warnings=warnings,
        errors=errors,
        research_only=True,
        exploratory_only=True,
        weak_label_target=True,
        no_final_labels=True,
        no_ground_truth_claim=True,
        no_production_claim=True,
        production_training=False,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
    )
    return report, rows, review_files


def build_feature_sets(
    snapshot: dict[str, np.ndarray],
    waveform: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    existing = np.asarray(snapshot["xsi_features"], dtype=np.float32)
    existing_names = np.asarray(snapshot["xsi_feature_names"]).astype(str)
    existing_groups = np.asarray(snapshot["xsi_feature_group"]).astype(str)
    wave = np.asarray(waveform["waveform_depth_features"], dtype=np.float32)
    wave_names = np.asarray(waveform["waveform_depth_feature_names"]).astype(str)
    wave_groups = np.asarray(waveform["waveform_depth_feature_group"]).astype(str)
    if existing.shape[0] != wave.shape[0]:
        raise ValueError("Existing and waveform feature row counts differ.")
    return {
        "existing_features_only": {
            "matrix": existing,
            "names": existing_names,
            "groups": np.asarray([f"existing:{item}" for item in existing_groups]),
        },
        "waveform_features_only": {
            "matrix": wave,
            "names": wave_names,
            "groups": np.asarray([f"waveform:{item}" for item in wave_groups]),
        },
        "combined_features": {
            "matrix": np.column_stack([existing, wave]).astype(np.float32),
            "names": np.concatenate([existing_names, wave_names]).astype(str),
            "groups": np.concatenate(
                [
                    np.asarray([f"existing:{item}" for item in existing_groups]),
                    np.asarray([f"waveform:{item}" for item in wave_groups]),
                ]
            ).astype(str),
        },
    }


def summarize_feature_groups(feature_sets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, feature_set in feature_sets.items():
        groups = np.asarray(feature_set["groups"]).astype(str)
        unique, counts = np.unique(groups, return_counts=True)
        output[name] = {
            "feature_count": int(np.asarray(feature_set["matrix"]).shape[1]),
            "sample_count": int(np.asarray(feature_set["matrix"]).shape[0]),
            "group_counts": {
                str(group): int(count) for group, count in zip(unique, counts, strict=True)
            },
            "finite_ratio": float(np.isfinite(np.asarray(feature_set["matrix"])).mean()),
        }
    return output


def write_model_review_dir(
    *,
    review_dir: Path,
    feature_group_summary: dict[str, Any],
    analysis: dict[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    summary_json = review_dir / "feature_set_summary.json"
    summary_md = review_dir / "model_review_summary.md"
    for path in (summary_json, summary_md):
        _ensure_can_write(path, overwrite=overwrite)
    summary_json.write_text(
        json.dumps(
            {
                "review_version": MODEL_REVIEW_VERSION,
                "feature_sets": feature_group_summary,
                "analysis": analysis,
                **RESEARCH_FLAGS,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(
        "\n".join(
            [
                "# MVP-4X Model Review",
                "",
                "Scope: research_only, exploratory_only, weak_label_target, "
                "no_final_labels, no_ground_truth_claim, no_production_claim.",
                "",
                "Feature importance, ablation, calibration plots, predicted-vs-target plots, "
                "and residual-vs-depth plots are skipped unless a sklearn model is fitted.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "feature_set_summary_json": str(summary_json),
        "model_review_summary_md": str(summary_md),
    }


def write_enhanced_baseline_outputs(
    report: EnhancedBaselineReport,
    rows: list[dict[str, Any]],
    *,
    review_files: dict[str, str],
    output_report_md: Path,
    output_report_json: Path,
    output_csv: Path,
    overwrite: bool,
) -> None:
    del review_files
    for path in (output_report_md, output_report_json, output_csv):
        _ensure_can_write(path, overwrite=overwrite)
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_report_md.write_text(format_enhanced_baseline_markdown(report), encoding="utf-8")
    _write_csv(rows, output_csv)


def format_enhanced_baseline_markdown(report: EnhancedBaselineReport) -> str:
    lines = [
        "# MVP-4X Enhanced-Feature Baselines",
        "",
        "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
        "no_ground_truth_claim, no_production_claim.",
        "",
        f"- report_version: `{report.report_version}`",
        f"- best_result: {report.best_result}",
        f"- feature_sets: {', '.join(report.feature_sets)}",
        f"- warnings: {len(report.warnings)}",
        f"- errors: {len(report.errors)}",
        "",
        "Derived binary metrics, if present, are derived_binary_audit_only.",
    ]
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report.warnings[:20])
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in report.errors)
    return "\n".join(lines) + "\n"


def _best_enhanced_result(sub_reports: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for feature_set, report in sub_reports.items():
        result = _as_dict(_as_dict(report.get("decision")).get("best_result"))
        if not result:
            continue
        candidate = {**result, "feature_set": feature_set}
        spearman = candidate.get("spearman")
        if spearman is None:
            continue
        if best is None or float(spearman) > float(best["spearman"]):
            best = candidate
    return best


def _analysis_placeholders(sub_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    any_fitted = any(
        _as_dict(report.get("decision")).get("best_result") for report in sub_reports.values()
    )
    if any_fitted:
        status = "not_computed_in_current_lightweight_analysis"
        reason = "model fitting completed but fold-level feature attribution is not yet implemented"
    else:
        status = "skipped_no_completed_models"
        reason = "scikit-learn unavailable or all models skipped"
    skipped = {"status": status, "reason": reason}
    return {
        "feature_group_ablation": skipped,
        "permutation_importance": skipped,
        "top_30_features": [],
        "top_10_stable_features": [],
        "regime_specific_error_analysis": skipped,
        "special_band_error_analysis": skipped,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "csv_version",
        "feature_set",
        "target",
        "model",
        "evaluation",
        "fold",
        "status",
        "mae",
        "rmse",
        "r2",
        "spearman",
        "pearson",
        "median_absolute_error",
    ]
    normalized = []
    for row in rows:
        normalized.append({"csv_version": row.get("csv_version", BASELINE_CSV_VERSION), **row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)


def _load_npz(path: Path | str) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Required NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_yaml(path: Path | str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
