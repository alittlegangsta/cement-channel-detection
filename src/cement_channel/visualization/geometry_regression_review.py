from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.alignment.xsi_geometry import ReceiverGeometry
from cement_channel.labels.cast_zc_source import load_controlled_cast_zc_source
from cement_channel.visualization.matplotlib_utils import require_pyplot, save_figure

GEOMETRY_REGRESSION_REVIEW_VERSION = "geometry_regression_review_v001"

KERNEL_COMPARISON_FIELDS = [
    "geometry_kernel",
    "sample_count",
    "zero_fraction",
    "nonzero_fraction",
    "spearman_correlation",
    "permutation_spearman_correlation",
    "real_minus_permutation_margin",
    "cross_validated_mae",
    "cross_validated_r2_sanity",
    "fold_stability",
    "depends_on_5700_band",
]

INTERVAL_FIELDS = [
    "review_id",
    "review_type",
    "start_depth",
    "end_depth",
    "primary_kernel",
    "target_fraction",
    "kernel_range",
    "xsi_score",
    "reason",
]


@dataclass(frozen=True)
class GeometryRegressionReviewReport:
    review_version: str
    generated_at: str
    inputs: dict[str, str]
    output_dir: str
    primary_kernel: str
    interval_count: int
    figure_count: int
    figures: dict[str, str]
    kernel_comparison_csv: str
    kernel_comparison_json: str
    interval_regression_target_summary_csv: str
    interval_regression_target_summary_json: str
    selected_interval_review_list_csv: str
    selected_interval_review_list_json: str
    no_model_training: bool
    no_final_labels: bool
    no_stc: bool
    no_apes: bool
    no_deep_learning: bool
    no_mvp4c: bool
    warnings: list[str]
    errors: list[str]
    not_performed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_geometry_regression_review(
    *,
    regression_labels_npz: Path | str,
    regression_audit_json: Path | str,
    raw_cast_zc_npz_candidates: list[Path | str],
    output_dir: Path | str,
    depth_level_features_npz: Path | str,
    cast_baseline_npz: Path | str | None = None,
    old_binary_labels_npz: Path | str | None = None,
    geometry_config_path: Path | str = "configs/xsi_geometry.example.yaml",
    overwrite: bool = False,
) -> GeometryRegressionReviewReport:
    labels = _load_npz(regression_labels_npz)
    audit = _read_json(Path(regression_audit_json))
    features = _load_npz(depth_level_features_npz)
    raw_source = load_controlled_cast_zc_source(raw_cast_zc_npz_candidates)
    baseline = _load_npz(cast_baseline_npz) if cast_baseline_npz else {}
    old_binary = _load_npz(old_binary_labels_npz) if old_binary_labels_npz else {}
    output = Path(output_dir)
    figures_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    _validate_guardrails(labels, audit, features, errors)
    primary_kernel = _primary_kernel(audit, labels)
    kernel_rows = build_kernel_comparison_rows(audit)
    interval_rows = build_selected_interval_rows(
        labels=labels,
        features=features,
        primary_kernel=primary_kernel,
    )
    interval_summary_rows = build_interval_summary_rows(interval_rows)

    kernel_csv = output / "kernel_comparison.csv"
    kernel_json = output / "kernel_comparison.json"
    interval_summary_csv = output / "interval_regression_target_summary.csv"
    interval_summary_json = output / "interval_regression_target_summary.json"
    selected_csv = output / "selected_interval_review_list.csv"
    selected_json = output / "selected_interval_review_list.json"
    review_summary = output / "review_summary.md"
    summary_json = output / "geometry_regression_review_summary_v001.json"
    for path in (
        kernel_csv,
        kernel_json,
        interval_summary_csv,
        interval_summary_json,
        selected_csv,
        selected_json,
        review_summary,
        summary_json,
    ):
        _ensure_can_write(path, overwrite=overwrite)
    _write_csv(kernel_rows, KERNEL_COMPARISON_FIELDS, kernel_csv)
    kernel_json.write_text(json.dumps(kernel_rows, indent=2, ensure_ascii=False) + "\n")
    _write_csv(interval_summary_rows, INTERVAL_FIELDS, interval_summary_csv)
    interval_summary_json.write_text(
        json.dumps(interval_summary_rows, indent=2, ensure_ascii=False) + "\n"
    )
    _write_csv(interval_rows, INTERVAL_FIELDS, selected_csv)
    selected_json.write_text(json.dumps(interval_rows, indent=2, ensure_ascii=False) + "\n")
    figures = write_geometry_regression_review_figures(
        labels=labels,
        audit=audit,
        features=features,
        raw_source=raw_source,
        baseline=baseline,
        old_binary=old_binary,
        output_dir=figures_dir,
        primary_kernel=primary_kernel,
        geometry=ReceiverGeometry.from_yaml(geometry_config_path),
        overwrite=overwrite,
    )
    report = GeometryRegressionReviewReport(
        review_version=GEOMETRY_REGRESSION_REVIEW_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "regression_labels_npz": str(regression_labels_npz),
            "regression_audit_json": str(regression_audit_json),
            "raw_cast_zc_npz_candidates": ", ".join(
                str(path) for path in raw_cast_zc_npz_candidates
            ),
            "cast_baseline_npz": str(cast_baseline_npz or ""),
            "old_binary_labels_npz": str(old_binary_labels_npz or ""),
            "depth_level_features_npz": str(depth_level_features_npz),
            "geometry_config_path": str(geometry_config_path),
        },
        output_dir=str(output),
        primary_kernel=primary_kernel,
        interval_count=len(interval_rows),
        figure_count=len(figures),
        figures=figures,
        kernel_comparison_csv=str(kernel_csv),
        kernel_comparison_json=str(kernel_json),
        interval_regression_target_summary_csv=str(interval_summary_csv),
        interval_regression_target_summary_json=str(interval_summary_json),
        selected_interval_review_list_csv=str(selected_csv),
        selected_interval_review_list_json=str(selected_json),
        no_model_training=True,
        no_final_labels=True,
        no_stc=True,
        no_apes=True,
        no_deep_learning=True,
        no_mvp4c=True,
        warnings=warnings,
        errors=errors,
        not_performed=[
            "ground truth claim",
            "final label approval",
            "MVP-4C",
            "STC",
            "APES",
            "deep learning",
            "model training",
        ],
    )
    review_summary.write_text(format_geometry_regression_review_summary(report), encoding="utf-8")
    summary_json.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_kernel_comparison_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _as_list(audit.get("kernel_summaries")):
        if not isinstance(row, dict):
            continue
        rows.append({field: row.get(field) for field in KERNEL_COMPARISON_FIELDS})
    return rows


def build_selected_interval_rows(
    *,
    labels: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    primary_kernel: str,
) -> list[dict[str, Any]]:
    depth = np.asarray(labels["depth"], dtype=np.float32)
    kernels = labels["geometry_kernel"].astype(str)
    targets = np.asarray(labels["receiver_max"], dtype=np.float32)
    primary_index = _kernel_index(kernels, primary_kernel)
    primary = targets[primary_index]
    kernel_range = np.max(targets, axis=0) - np.min(targets, axis=0)
    feature_values = np.asarray(features["depth_level_xsi_features"], dtype=np.float32)
    xsi_score = _scaled_first_feature(feature_values)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _interval_rows_for_indices(
            "high_fraction_interval",
            np.argsort(primary)[-3:][::-1],
            depth,
            primary,
            kernel_range,
            xsi_score,
            primary_kernel,
            "highest continuous CAST channel fraction",
        )
    )
    rows.extend(
        _interval_rows_for_indices(
            "low_fraction_interval",
            np.argsort(primary)[:3],
            depth,
            primary,
            kernel_range,
            xsi_score,
            primary_kernel,
            "lowest continuous CAST channel fraction",
        )
    )
    local_only = np.flatnonzero((primary > np.percentile(primary, 75.0)) & (xsi_score < 0.4))
    rows.extend(
        _interval_rows_for_indices(
            "local_only_anomaly",
            local_only[:3],
            depth,
            primary,
            kernel_range,
            xsi_score,
            primary_kernel,
            "CAST fraction high while XSI depth feature is low",
        )
    )
    sensitive = np.argsort(kernel_range)[-3:][::-1]
    rows.extend(
        _interval_rows_for_indices(
            "kernel_sensitive_interval",
            sensitive,
            depth,
            primary,
            kernel_range,
            xsi_score,
            primary_kernel,
            "large disagreement across geometry kernels",
        )
    )
    rows.extend(
        _interval_rows_for_indices(
            "xsi_high_cast_fraction_low",
            np.flatnonzero((xsi_score > 0.75) & (primary < 0.1))[:3],
            depth,
            primary,
            kernel_range,
            xsi_score,
            primary_kernel,
            "XSI feature high while CAST continuous target is low",
        )
    )
    rows.extend(
        _interval_rows_for_indices(
            "xsi_low_cast_fraction_high",
            np.flatnonzero((xsi_score < 0.25) & (primary > 0.2))[:3],
            depth,
            primary,
            kernel_range,
            xsi_score,
            primary_kernel,
            "XSI feature low while CAST continuous target is high",
        )
    )
    band = np.flatnonzero((depth >= 5680.0) & (depth <= 5720.0))
    rows.extend(
        _interval_rows_for_indices(
            "5700_band_review",
            band[:3],
            depth,
            primary,
            kernel_range,
            xsi_score,
            primary_kernel,
            "inside 5680-5720 ft review band",
        )
    )
    confidence = np.asarray(
        labels.get("depth_label_confidence", np.ones((targets.shape[0], depth.size, 1))),
        dtype=np.float32,
    )
    if confidence.ndim == 3:
        low_conf = np.argsort(np.nanmean(confidence[primary_index], axis=1))[:3]
        rows.extend(
            _interval_rows_for_indices(
                "uncertainty_review",
                low_conf,
                depth,
                primary,
                kernel_range,
                xsi_score,
                primary_kernel,
                "low regression label confidence",
            )
        )
    return _dedupe_review_rows(rows)


def build_interval_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def write_geometry_regression_review_figures(
    *,
    labels: dict[str, np.ndarray],
    audit: dict[str, Any],
    features: dict[str, np.ndarray],
    raw_source: Any,
    baseline: dict[str, np.ndarray],
    old_binary: dict[str, np.ndarray],
    output_dir: Path,
    primary_kernel: str,
    geometry: ReceiverGeometry,
    overwrite: bool,
) -> dict[str, str]:
    figures = {
        "continuous_channel_fraction_vs_depth": output_dir
        / "continuous_channel_fraction_vs_depth.png",
        "kernel_comparison": output_dir / "kernel_comparison.png",
        "source_r7_receiver_midpoint_schematic": output_dir
        / "source_r7_receiver_midpoint_schematic.png",
        "raw_zc_heatmap": output_dir / "raw_zc_heatmap.png",
        "zc_lt_2p5_binary_cell_mask_heatmap": output_dir
        / "zc_lt_2p5_binary_cell_mask_heatmap.png",
        "relative_drop_heatmap": output_dir / "relative_drop_heatmap.png",
        "weighted_channel_fraction_heatmap_trace": output_dir
        / "weighted_channel_fraction_heatmap_trace.png",
        "selected_interval_before_after_comparison": output_dir
        / "selected_interval_before_after_comparison.png",
        "band_5700_sensitivity": output_dir / "band_5700_sensitivity.png",
        "old_binary_vs_new_continuous_target": output_dir
        / "old_binary_vs_new_continuous_target.png",
    }
    _save_fraction_vs_depth(
        labels,
        primary_kernel,
        figures["continuous_channel_fraction_vs_depth"],
        overwrite,
    )
    _save_kernel_comparison(audit, figures["kernel_comparison"], overwrite)
    _save_schematic(geometry, figures["source_r7_receiver_midpoint_schematic"], overwrite)
    _save_heatmap(raw_source.cast_zc, "Raw CAST Zc", figures["raw_zc_heatmap"], overwrite)
    _save_heatmap(
        raw_source.cast_zc < 2.5,
        "Zc < 2.5 cell mask",
        figures["zc_lt_2p5_binary_cell_mask_heatmap"],
        overwrite,
    )
    relative_drop = np.asarray(
        baseline.get("relative_drop", np.full_like(raw_source.cast_zc, np.nan))
    )
    _save_heatmap(relative_drop, "Relative drop", figures["relative_drop_heatmap"], overwrite)
    _save_weighted_fraction_heatmap(
        labels,
        figures["weighted_channel_fraction_heatmap_trace"],
        overwrite,
    )
    _save_selected_interval_comparison(
        labels,
        features,
        primary_kernel,
        figures["selected_interval_before_after_comparison"],
        overwrite,
    )
    _save_5700(labels, figures["band_5700_sensitivity"], overwrite)
    _save_old_vs_new(
        old_binary,
        labels,
        primary_kernel,
        figures["old_binary_vs_new_continuous_target"],
        overwrite,
    )
    return {key: str(path) for key, path in figures.items()}


def format_geometry_regression_review_summary(report: GeometryRegressionReviewReport) -> str:
    lines = [
        "# Geometry-Aware Regression Manual Review Supplement",
        "",
        "This is a regression weak-label candidate review package. It is not ground "
        "truth, not final labels, and does not approve MVP-4C, STC/APES, deep "
        "learning, or production claims.",
        "",
        f"- primary_kernel: `{report.primary_kernel}`",
        f"- interval_count: {report.interval_count}",
        f"- figure_count: {report.figure_count}",
        f"- no_final_labels: `{report.no_final_labels}`",
        "",
        "## Outputs",
        "",
        f"- kernel_comparison_csv: `{report.kernel_comparison_csv}`",
        f"- selected_interval_review_list_csv: `{report.selected_interval_review_list_csv}`",
        f"- figures_dir: `{Path(report.output_dir) / 'figures'}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(_message_lines(report.warnings))
    lines.extend(["", "## Errors", ""])
    lines.extend(_message_lines(report.errors))
    lines.extend(["", "## Not Performed", ""])
    lines.extend(_message_lines(report.not_performed))
    lines.append("")
    return "\n".join(lines)


def _save_fraction_vs_depth(
    labels: dict[str, np.ndarray],
    kernel: str,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    depth = np.asarray(labels["depth"], dtype=np.float32)
    kernels = labels["geometry_kernel"].astype(str)
    targets = np.asarray(labels["receiver_max"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for index, name in enumerate(kernels):
        ax.plot(depth, targets[index], label=name, linewidth=1.0 if name != kernel else 2.0)
    ax.set_xlabel("depth")
    ax.set_ylabel("continuous channel fraction")
    ax.set_title("Continuous channel fraction vs depth")
    ax.legend(loc="best", fontsize=7)
    save_figure(fig, path, overwrite=overwrite)


def _save_kernel_comparison(audit: dict[str, Any], path: Path, overwrite: bool) -> None:
    plt = require_pyplot()
    rows = build_kernel_comparison_rows(audit)
    labels = [str(row["geometry_kernel"]) for row in rows]
    margins = [_as_float(row.get("real_minus_permutation_margin")) or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(np.arange(len(labels)), margins, color="tab:blue")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_ylabel("real - permutation Spearman margin")
    ax.set_title("Kernel comparison - regression sanity audit")
    save_figure(fig, path, overwrite=overwrite)


def _save_schematic(geometry: ReceiverGeometry, path: Path, overwrite: bool) -> None:
    plt = require_pyplot()
    offsets = geometry.physical_receiver_offsets_ft
    source = geometry.physical_source_offset_ft
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.scatter(offsets, np.zeros_like(offsets), label="receivers", color="tab:blue")
    ax.scatter([source], [0.0], label="source", color="tab:red")
    ax.scatter([(source + offsets[6]) * 0.5], [0.0], label="R7 midpoint", color="tab:green")
    ax.set_xlabel("physical offset relative to R7 (ft; deeper positive)")
    ax.set_yticks([])
    ax.set_title("Source / R7 / receiver / midpoint schematic")
    ax.legend(loc="best")
    save_figure(fig, path, overwrite=overwrite)


def _save_heatmap(values: np.ndarray, title: str, path: Path, overwrite: bool) -> None:
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    image = ax.imshow(np.asarray(values, dtype=np.float32).T, aspect="auto", origin="lower")
    ax.set_xlabel("depth index")
    ax.set_ylabel("azimuth index")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    save_figure(fig, path, overwrite=overwrite)


def _save_weighted_fraction_heatmap(
    labels: dict[str, np.ndarray],
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    values = np.asarray(labels["weighted_channel_fraction_zc_lt_2p5"], dtype=np.float32)
    kernels = labels["geometry_kernel"].astype(str)
    matrix = np.max(values, axis=2)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(kernels)), labels=kernels)
    ax.set_xlabel("depth index")
    ax.set_title("Weighted channel fraction heatmap / trace")
    fig.colorbar(image, ax=ax)
    save_figure(fig, path, overwrite=overwrite)


def _save_selected_interval_comparison(
    labels: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    kernel: str,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    depth = np.asarray(labels["depth"], dtype=np.float32)
    kernels = labels["geometry_kernel"].astype(str)
    target = np.asarray(labels["receiver_max"], dtype=np.float32)[_kernel_index(kernels, kernel)]
    xsi = _scaled_first_feature(np.asarray(features["depth_level_xsi_features"], dtype=np.float32))
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(depth, target, label="new continuous target", color="tab:blue")
    ax.plot(depth, xsi, label="scaled first XSI feature", color="tab:orange", alpha=0.75)
    ax.set_title("Selected interval before/after comparison")
    ax.legend(loc="best")
    save_figure(fig, path, overwrite=overwrite)


def _save_5700(labels: dict[str, np.ndarray], path: Path, overwrite: bool) -> None:
    plt = require_pyplot()
    depth = np.asarray(labels["depth"], dtype=np.float32)
    kernels = labels["geometry_kernel"].astype(str)
    target = np.asarray(labels["receiver_max"], dtype=np.float32)
    band = (depth >= 5680.0) & (depth <= 5720.0)
    values = [float(np.mean(row[band])) if np.any(band) else 0.0 for row in target]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(np.arange(len(kernels)), values, color="tab:purple")
    ax.set_xticks(np.arange(len(kernels)), labels=kernels, rotation=25, ha="right")
    ax.set_ylabel("mean target in 5680-5720 ft")
    ax.set_title("5700 band sensitivity")
    save_figure(fig, path, overwrite=overwrite)


def _save_old_vs_new(
    old_binary: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    kernel: str,
    path: Path,
    overwrite: bool,
) -> None:
    plt = require_pyplot()
    depth = np.asarray(labels["depth"], dtype=np.float32)
    kernels = labels["geometry_kernel"].astype(str)
    new_target = np.asarray(labels["receiver_max"], dtype=np.float32)[
        _kernel_index(kernels, kernel)
    ]
    old = np.asarray(old_binary.get("has_channel_any", np.zeros(depth.size)), dtype=np.float32)
    if old.size != depth.size:
        old = np.zeros(depth.size, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(depth, new_target, label="new continuous target")
    ax.step(depth, old, label="old binary label view", where="mid", alpha=0.75)
    ax.set_title("Old binary label vs new continuous target")
    ax.legend(loc="best")
    save_figure(fig, path, overwrite=overwrite)


def _interval_rows_for_indices(
    review_type: str,
    indices: np.ndarray,
    depth: np.ndarray,
    target: np.ndarray,
    kernel_range: np.ndarray,
    xsi_score: np.ndarray,
    primary_kernel: str,
    reason: str,
) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        idx = int(index)
        if idx < 0 or idx >= depth.size:
            continue
        rows.append(
            {
                "review_id": f"{review_type}_{idx:04d}",
                "review_type": review_type,
                "start_depth": float(depth[idx]),
                "end_depth": float(depth[idx]),
                "primary_kernel": primary_kernel,
                "target_fraction": float(target[idx]),
                "kernel_range": float(kernel_range[idx]),
                "xsi_score": float(xsi_score[idx]),
                "reason": reason,
            }
        )
    return rows


def _dedupe_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, float]] = set()
    output = []
    for row in rows:
        key = (str(row["review_type"]), float(row["start_depth"]))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _scaled_first_feature(features: np.ndarray) -> np.ndarray:
    values = features[:, 0].astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.max(finite) <= np.min(finite):
        return np.zeros(values.size, dtype=np.float32)
    return ((values - np.min(finite)) / (np.max(finite) - np.min(finite))).astype(np.float32)


def _primary_kernel(audit: dict[str, Any], labels: dict[str, np.ndarray]) -> str:
    best = audit.get("best_kernel")
    kernels = set(labels["geometry_kernel"].astype(str))
    if isinstance(best, str) and best in kernels:
        return best
    for candidate in ("triangular_midpoint_weighted", "midpoint_window", "r7_reference_point"):
        if candidate in kernels:
            return candidate
    return str(labels["geometry_kernel"].astype(str)[0])


def _kernel_index(kernels: np.ndarray, kernel: str) -> int:
    matches = np.flatnonzero(kernels.astype(str) == kernel)
    return int(matches[0]) if matches.size else 0


def _validate_guardrails(
    labels: dict[str, np.ndarray],
    audit: dict[str, Any],
    features: dict[str, np.ndarray],
    errors: list[str],
) -> None:
    if not bool(np.asarray(labels.get("no_final_labels", False))):
        errors.append("Regression labels must preserve no_final_labels=true.")
    if audit.get("no_final_labels") is not True:
        errors.append("Regression audit must preserve no_final_labels=true.")
    if not bool(np.asarray(features.get("no_final_labels", False))):
        errors.append("Depth-level features must preserve no_final_labels=true.")


def _write_csv(rows: list[dict[str, Any]], fieldnames: list[str], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path | str | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _message_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["- none"]
    return [f"- {message}" for message in messages]


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
