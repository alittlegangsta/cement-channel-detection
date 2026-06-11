from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cement_channel.visualization.matplotlib_utils import require_pyplot

CAST_HEATMAP_SUPPLEMENT_VERSION = "mvp4x_ls_mw_cast_heatmap_supplement_v001"
DEFAULT_REVIEW_DIR_NAME = "mvp4x_label_semantics_manual_review_v001"
DEFAULT_HEATMAP_DIR_NAME = "interval_cast_heatmaps"
ZC_THRESHOLD_MRAYL = 2.5
METHOD_FLAGS: dict[str, bool] = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
}
RECOGNIZED_CONNECTED_MASK_KEYS = (
    "connected_component_mask",
    "connected_channel_mask",
    "largest_connected_component_mask",
    "connected_mask",
)
REVIEW_CHECKLIST_ADDITIONS = (
    "CAST raw Zc 是否存在局部低于 2.5 MRayl 的区域？",
    "低 Zc 区域是连续结构还是零散像素？",
    "connected mask 是否与肉眼看到的结构一致？",
    "local_worst 是否对应真实局部异常？",
    "receiver_mean 是否稀释局部异常？",
    "p90 是否比 mean 更合理？",
    "是否存在明显 interpolation stripe、boundary artifact 或单行伪影？",
)


@dataclass(frozen=True)
class CastHeatmapSupplementResult:
    version: str
    generated_at: str
    output_dir: str
    selected_interval_count: int
    heatmap_count: int
    failed_intervals: list[str]
    connected_mask_traceable_count: int
    connected_mask_untraceable_count: int
    relative_drop_heatmap_count: int
    zc_color_scale: dict[str, float]
    warnings: list[str]
    inventory_csv: str
    inventory_json: str
    summary_md: str
    summary_json: str
    review_summary: str
    reviewer_checklist: str
    flags: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "output_dir": self.output_dir,
            "selected_interval_count": self.selected_interval_count,
            "heatmap_count": self.heatmap_count,
            "failed_intervals": self.failed_intervals,
            "connected_mask_traceable_count": self.connected_mask_traceable_count,
            "connected_mask_untraceable_count": self.connected_mask_untraceable_count,
            "relative_drop_heatmap_count": self.relative_drop_heatmap_count,
            "zc_color_scale": self.zc_color_scale,
            "warnings": self.warnings,
            "inventory_csv": self.inventory_csv,
            "inventory_json": self.inventory_json,
            "summary_md": self.summary_md,
            "summary_json": self.summary_json,
            "review_summary": self.review_summary,
            "reviewer_checklist": self.reviewer_checklist,
            **self.flags,
        }


def generate_cast_heatmap_supplement(
    *,
    selected_intervals_csv: Path | str,
    cast_label_input_npz: Path | str,
    review_dir: Path | str,
    cast_baseline_npz: Path | str | None = None,
    output_dir: Path | str | None = None,
    connected_mask_npz: Path | str | None = None,
    zc_threshold_mrayl: float = ZC_THRESHOLD_MRAYL,
    depth_half_window_ft: float = 20.0,
    overwrite: bool = False,
) -> CastHeatmapSupplementResult:
    selected_path = Path(selected_intervals_csv)
    label_input_path = Path(cast_label_input_npz)
    review_path = Path(review_dir)
    heatmap_dir = (
        Path(output_dir) if output_dir is not None else review_path / DEFAULT_HEATMAP_DIR_NAME
    )
    if zc_threshold_mrayl != ZC_THRESHOLD_MRAYL:
        raise ValueError("This supplement is fixed to the reviewed 2.5 MRayl threshold.")
    intervals = _read_selected_intervals(selected_path)
    label_input = _load_npz(label_input_path)
    cast_depth = _required_1d(label_input, "cast_depth")
    cast_zc = _required_2d(label_input, "cast_zc")
    cast_azimuth = _azimuth_axis(label_input, cast_zc.shape[1])
    if cast_zc.shape[0] != cast_depth.size:
        raise ValueError("cast_zc first dimension must match cast_depth.")

    relative_drop = None
    baseline_path: Path | None = None
    if cast_baseline_npz is not None and Path(cast_baseline_npz).exists():
        baseline_path = Path(cast_baseline_npz)
        baseline = _load_npz(baseline_path)
        candidate = baseline.get("relative_drop")
        if candidate is not None and np.asarray(candidate).shape == cast_zc.shape:
            relative_drop = np.asarray(candidate, dtype=np.float32)

    connected_mask, connected_source, connected_warning = _load_connected_mask(
        connected_mask_npz,
        expected_shape=cast_zc.shape,
    )
    row_windows = [
        _window_rows(cast_depth, float(row["depth"]), depth_half_window_ft) for row in intervals
    ]
    zc_vmin, zc_vmax = _fixed_zc_color_scale(cast_zc, row_windows, zc_threshold_mrayl)

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    failed: list[str] = []
    warnings: list[str] = []
    if connected_warning:
        warnings.append(connected_warning)

    for interval, rows in zip(intervals, row_windows, strict=True):
        interval_id = _interval_id(interval)
        png_path = heatmap_dir / f"{interval_id}_cast_heatmap.png"
        try:
            record = _write_interval_heatmap(
                interval=interval,
                rows=rows,
                cast_depth=cast_depth,
                cast_azimuth=cast_azimuth,
                cast_zc=cast_zc,
                relative_drop=relative_drop,
                connected_mask=connected_mask,
                connected_source=connected_source,
                output_path=png_path,
                zc_threshold_mrayl=zc_threshold_mrayl,
                zc_vmin=zc_vmin,
                zc_vmax=zc_vmax,
                overwrite=overwrite,
            )
            inventory.append(record)
        except (OSError, ValueError, RuntimeError) as exc:
            failed.append(interval_id)
            warnings.append(f"{interval_id}: {exc}")
            inventory.append(_failed_record(interval, png_path, str(exc)))

    inventory_csv = review_path / "interval_cast_heatmap_inventory.csv"
    inventory_json = review_path / "interval_cast_heatmap_inventory.json"
    summary_md = review_path / "cast_heatmap_generation_summary.md"
    summary_json = review_path / "cast_heatmap_generation_summary.json"
    _write_inventory_csv(inventory, inventory_csv)
    _write_json(inventory_json, {"inventory": inventory, **METHOD_FLAGS})

    connected_traceable_count = int(
        sum(bool(row.get("connected_mask_traceable")) for row in inventory)
    )
    relative_drop_count = int(
        sum(bool(row.get("relative_drop_heatmap_generated")) for row in inventory)
    )
    result = CastHeatmapSupplementResult(
        version=CAST_HEATMAP_SUPPLEMENT_VERSION,
        generated_at=_utc_now(),
        output_dir=str(heatmap_dir),
        selected_interval_count=len(intervals),
        heatmap_count=int(sum(bool(row.get("png_written")) for row in inventory)),
        failed_intervals=failed,
        connected_mask_traceable_count=connected_traceable_count,
        connected_mask_untraceable_count=len(inventory) - connected_traceable_count,
        relative_drop_heatmap_count=relative_drop_count,
        zc_color_scale={"vmin": zc_vmin, "vmax": zc_vmax, "threshold": zc_threshold_mrayl},
        warnings=warnings,
        inventory_csv=str(inventory_csv),
        inventory_json=str(inventory_json),
        summary_md=str(summary_md),
        summary_json=str(summary_json),
        review_summary=str(review_path / "review_summary.md"),
        reviewer_checklist=str(review_path / "reviewer_checklist.md"),
        flags=METHOD_FLAGS.copy(),
    )
    _write_json(summary_json, result.to_dict())
    summary_md.write_text(_format_summary(result, inventory), encoding="utf-8")
    _update_review_summary(review_path / "review_summary.md", result)
    _update_reviewer_checklist(review_path / "reviewer_checklist.md")
    return result


def low_zc_candidate_mask(cast_zc: np.ndarray, threshold: float = ZC_THRESHOLD_MRAYL) -> np.ndarray:
    return np.asarray(cast_zc, dtype=np.float32) < float(threshold)


def connected_mask_traceability(
    connected_mask_npz: Path | str | None,
    *,
    expected_shape: tuple[int, int],
) -> tuple[bool, str | None]:
    mask, source, warning = _load_connected_mask(connected_mask_npz, expected_shape=expected_shape)
    if mask is not None:
        return True, source
    return False, warning


def _write_interval_heatmap(
    *,
    interval: dict[str, Any],
    rows: np.ndarray,
    cast_depth: np.ndarray,
    cast_azimuth: np.ndarray,
    cast_zc: np.ndarray,
    relative_drop: np.ndarray | None,
    connected_mask: np.ndarray | None,
    connected_source: str | None,
    output_path: Path,
    zc_threshold_mrayl: float,
    zc_vmin: float,
    zc_vmax: float,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite.")
    interval_id = _interval_id(interval)
    review_depth = float(interval["depth"])
    selected_depths = cast_depth[rows]
    raw_zc_window = cast_zc[rows]
    candidate = low_zc_candidate_mask(raw_zc_window, zc_threshold_mrayl).astype(np.float32)
    rel_window = None if relative_drop is None else relative_drop[rows]
    connected_window = None if connected_mask is None else connected_mask[rows].astype(np.float32)

    panel_count = 4 if rel_window is not None else 3
    plt = require_pyplot()
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(4.2 * panel_count, 5.2),
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    _plot_image_panel(
        fig,
        axes_array[0],
        raw_zc_window,
        selected_depths,
        cast_azimuth,
        title="raw Zc",
        colorbar_label="Zc MRayl",
        vmin=zc_vmin,
        vmax=zc_vmax,
        review_depth=review_depth,
        cmap="viridis",
    )
    axes_array[0].text(
        0.02,
        0.98,
        f"zc_threshold = {zc_threshold_mrayl:.1f} MRayl",
        transform=axes_array[0].transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "linewidth": 0.0},
    )
    _plot_image_panel(
        fig,
        axes_array[1],
        candidate,
        selected_depths,
        cast_azimuth,
        title="low-Zc candidate mask\nZc < 2.5 MRayl",
        colorbar_label="candidate 0/1",
        vmin=0.0,
        vmax=1.0,
        review_depth=review_depth,
        cmap="gray_r",
    )
    if connected_window is None:
        _plot_connected_unavailable_panel(
            axes_array[2],
            review_depth=review_depth,
            depth_min=float(np.min(selected_depths)),
            depth_max=float(np.max(selected_depths)),
        )
    else:
        _plot_image_panel(
            fig,
            axes_array[2],
            connected_window,
            selected_depths,
            cast_azimuth,
            title="connected-component mask\ntraceable saved 2D source",
            colorbar_label="connected 0/1",
            vmin=0.0,
            vmax=1.0,
            review_depth=review_depth,
            cmap="gray_r",
        )
    if rel_window is not None:
        rel_vmin, rel_vmax = _finite_limits(rel_window, default=(0.0, 1.0))
        _plot_image_panel(
            fig,
            axes_array[3],
            rel_window,
            selected_depths,
            cast_azimuth,
            title="relative-drop heatmap\nlegal baseline field",
            colorbar_label="relative drop",
            vmin=rel_vmin,
            vmax=rel_vmax,
            review_depth=review_depth,
            cmap="magma",
        )

    for axis in axes_array:
        axis.set_xlabel("CAST azimuth degree")
    axes_array[0].set_ylabel("physical depth ft")
    title = (
        f"{interval_id} | {interval.get('selection_reason', 'unknown_reason')} | "
        f"review depth {review_depth:.2f} ft"
    )
    fig.suptitle(title + "\nresearch-only weak-label audit; no final labels", fontsize=10)
    metadata = {
        "Title": title,
        "Description": json.dumps(
            {
                "version": CAST_HEATMAP_SUPPLEMENT_VERSION,
                "interval_id": interval_id,
                "review_depth_ft": review_depth,
                "zc_threshold_mrayl": zc_threshold_mrayl,
                "connected_mask_traceable": connected_window is not None,
                "connected_component_rule": _connected_rule_note(connected_window is not None),
                **METHOD_FLAGS,
            },
            sort_keys=True,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", metadata=metadata)
    plt.close(fig)

    return {
        "interval_id": interval_id,
        "review_rank": int(interval["review_rank"]),
        "sample_index": int(interval["sample_index"]),
        "review_depth_ft": review_depth,
        "selection_reason": interval.get("selection_reason", ""),
        "png_path": str(output_path),
        "png_written": True,
        "depth_min_ft": float(np.min(selected_depths)),
        "depth_max_ft": float(np.max(selected_depths)),
        "depth_row_count": int(rows.size),
        "azimuth_bin_count": int(cast_azimuth.size),
        "raw_zc_heatmap_generated": True,
        "low_zc_candidate_mask_generated": True,
        "low_zc_candidate_fraction": float(np.mean(candidate)),
        "connected_mask_traceable": connected_window is not None,
        "connected_mask_source": connected_source,
        "connected_component_rule": _connected_rule_note(connected_window is not None),
        "connected_mask_warning": ""
        if connected_window is not None
        else _connected_unavailable_warning(),
        "relative_drop_heatmap_generated": rel_window is not None,
        "azimuth_width_support_mask_generated": False,
        "azimuth_width_support_mask_warning": (
            "No saved legal 2D azimuth-width support mask field was provided."
        ),
        **METHOD_FLAGS,
    }


def _plot_image_panel(
    fig: Any,
    axis: Any,
    values: np.ndarray,
    depth: np.ndarray,
    azimuth: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    review_depth: float,
    cmap: str,
) -> None:
    image = axis.imshow(
        np.asarray(values, dtype=np.float32),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            float(np.min(azimuth)),
            float(np.max(azimuth)),
            float(np.min(depth)),
            float(np.max(depth)),
        ),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    axis.axhline(review_depth, color="cyan", linewidth=1.0)
    axis.set_title(title, fontsize=9)
    fig.colorbar(image, ax=axis, label=colorbar_label, fraction=0.046, pad=0.04)


def _plot_connected_unavailable_panel(
    axis: Any,
    *,
    review_depth: float,
    depth_min: float,
    depth_max: float,
) -> None:
    axis.set_title("connected-component mask\nnot traceable to saved 2D field", fontsize=9)
    axis.set_xlim(0.0, 358.0)
    axis.set_ylim(depth_min, depth_max)
    axis.axhline(review_depth, color="cyan", linewidth=1.0)
    axis.set_xlabel("CAST azimuth degree")
    axis.text(
        0.5,
        0.5,
        "WARNING\nconnected_channel_fraction is saved as\n"
        "aggregate morphology fractions only.\nNo 2D component label map was saved.\n"
        "No mask is fabricated here.",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff3cd", "edgecolor": "#856404"},
    )


def _failed_record(interval: dict[str, Any], png_path: Path, error: str) -> dict[str, Any]:
    return {
        "interval_id": _interval_id(interval),
        "review_rank": int(interval["review_rank"]),
        "sample_index": int(interval["sample_index"]),
        "review_depth_ft": float(interval["depth"]),
        "selection_reason": interval.get("selection_reason", ""),
        "png_path": str(png_path),
        "png_written": False,
        "error": error,
        **METHOD_FLAGS,
    }


def _read_selected_intervals(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"selected_intervals.csv does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            parsed["review_rank"] = int(parsed["review_rank"])
            parsed["sample_index"] = int(parsed["sample_index"])
            parsed["depth"] = float(parsed["depth"])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No selected intervals found in {path}")
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _required_1d(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Required field missing: {key}")
    values = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Required field is empty: {key}")
    return values


def _required_2d(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Required field missing: {key}")
    values = np.asarray(data[key], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Required field must be 2-D: {key}")
    return values


def _azimuth_axis(data: dict[str, np.ndarray], width: int) -> np.ndarray:
    if "cast_azimuth_deg" in data:
        axis = np.asarray(data["cast_azimuth_deg"], dtype=np.float32).reshape(-1)
        if axis.size == width:
            return axis
    return np.linspace(0.0, 360.0, num=width, endpoint=False, dtype=np.float32)


def _window_rows(depth: np.ndarray, review_depth: float, half_window_ft: float) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32).reshape(-1)
    mask = np.abs(values - float(review_depth)) <= float(half_window_ft)
    rows = np.flatnonzero(mask)
    if rows.size < 3:
        center = int(np.argmin(np.abs(values - float(review_depth))))
        start = max(0, center - 2)
        stop = min(values.size, center + 3)
        rows = np.arange(start, stop, dtype=np.int64)
    return rows[np.argsort(values[rows])]


def _fixed_zc_color_scale(
    cast_zc: np.ndarray,
    row_windows: list[np.ndarray],
    threshold: float,
) -> tuple[float, float]:
    if not row_windows:
        return 0.0, 1.0
    samples = [np.asarray(cast_zc[rows], dtype=np.float32).reshape(-1) for rows in row_windows]
    values = np.concatenate(samples)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    vmin = min(float(vmin), float(threshold))
    vmax = max(float(vmax), float(threshold) + 0.25)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _finite_limits(values: np.ndarray, *, default: tuple[float, float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default
    vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return default
    return float(vmin), float(vmax)


def _load_connected_mask(
    path: Path | str | None,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray | None, str | None, str | None]:
    if path is None:
        return None, None, _connected_unavailable_warning()
    source = Path(path)
    if not source.exists():
        return None, None, f"Connected mask source does not exist: {source}"
    data = _load_npz(source)
    for key in RECOGNIZED_CONNECTED_MASK_KEYS:
        if key not in data:
            continue
        values = np.asarray(data[key])
        if values.shape != expected_shape:
            return None, None, (
                f"Connected mask field {key} shape {values.shape} does not match {expected_shape}."
            )
        return values.astype(bool), f"{source}:{key}", None
    return None, None, (
        "Connected mask source lacks recognized 2D field: "
        + ", ".join(RECOGNIZED_CONNECTED_MASK_KEYS)
    )


def _connected_unavailable_warning() -> str:
    return (
        "connected_channel_fraction is stored as aggregate morphology fractions; "
        "no saved 2D connected-component mask is available, so no connected mask is fabricated."
    )


def _connected_rule_note(traceable: bool) -> str:
    if traceable:
        return (
            "saved 2D connected-component mask; neighborhood/min-component rule must be read "
            "from the mask-producing manifest"
        )
    return (
        "untraceable_to_2d_mask; aggregate connected_channel_fraction came from morphology "
        "arrays, but component labels/neighborhood/min-area map were not saved"
    )


def _interval_id(interval: dict[str, Any]) -> str:
    return f"interval_{int(interval['review_rank']):02d}"


def _write_inventory_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_summary(result: CastHeatmapSupplementResult, inventory: list[dict[str, Any]]) -> str:
    failed = ", ".join(result.failed_intervals) if result.failed_intervals else "none"
    warnings = result.warnings or ["none"]
    return "\n".join(
        [
            "# MVP-4X LS/MW CAST Heatmap Supplement",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            f"- version: `{result.version}`",
            f"- selected_interval_count: `{result.selected_interval_count}`",
            f"- heatmap_count: `{result.heatmap_count}`",
            f"- output_dir: `{result.output_dir}`",
            f"- zc_threshold_mrayl: `{result.zc_color_scale['threshold']}`",
            f"- raw_zc_fixed_color_scale: `{result.zc_color_scale}`",
            f"- relative_drop_heatmap_count: `{result.relative_drop_heatmap_count}`",
            f"- connected_mask_traceable_count: `{result.connected_mask_traceable_count}`",
            f"- connected_mask_untraceable_count: `{result.connected_mask_untraceable_count}`",
            f"- failed_intervals: `{failed}`",
            "",
            "Connected-component traceability:",
            "",
            "- The current manual-review pack stores `connected_channel_fraction` as aggregate "
            "morphology fractions.",
            "- No saved per-cell 2D connected-component label map was found for these intervals.",
            "- The connected panel therefore records a warning and does not fabricate a mask.",
            "",
            "Warnings:",
            "",
            *[f"- {warning}" for warning in warnings],
            "",
            f"Inventory rows: `{len(inventory)}`",
            "",
        ]
    )


def _update_review_summary(path: Path, result: CastHeatmapSupplementResult) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Review Summary\n"
    section = "\n".join(
        [
            "## CAST 2D Heatmap Supplement",
            "",
            f"- heatmap_count: `{result.heatmap_count}`",
            f"- output_dir: `{result.output_dir}`",
            f"- zc_threshold_mrayl: `{result.zc_color_scale['threshold']}`",
            f"- relative_drop_heatmap_count: `{result.relative_drop_heatmap_count}`",
            f"- connected_mask_traceable_count: `{result.connected_mask_traceable_count}`",
            f"- connected_mask_untraceable_count: `{result.connected_mask_untraceable_count}`",
            "- connected_mask_note: `no saved 2D component mask; warning panel only`",
            "",
        ]
    )
    path.write_text(
        _replace_section(existing, "## CAST 2D Heatmap Supplement", section),
        encoding="utf-8",
    )


def _update_reviewer_checklist(path: Path) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Reviewer Checklist\n"
    section = "\n".join(
        [
            "## CAST 2D Heatmap Supplement Questions",
            "",
            *[f"- {item}" for item in REVIEW_CHECKLIST_ADDITIONS],
            "",
        ]
    )
    path.write_text(
        _replace_section(existing, "## CAST 2D Heatmap Supplement Questions", section),
        encoding="utf-8",
    )


def _replace_section(existing: str, heading: str, section: str) -> str:
    lines = existing.rstrip().splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index
            break
    if start is None:
        return existing.rstrip() + "\n\n" + section
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join([*lines[:start], section.rstrip(), *lines[end:]]) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
