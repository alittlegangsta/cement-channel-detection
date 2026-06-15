from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STRIPE_MANUAL_REVIEW_INTEGRATION_VERSION = "mvp4x_stripe_aware_manual_review_v001"
DEFAULT_HEATMAP_HALF_WINDOW_FT = 20.0
DEFAULT_NEAR_DISTANCE_FT = 3.0
METHOD_FLAGS: dict[str, bool] = {
    "research_only": True,
    "exploratory_only": True,
    "weak_label_target": True,
    "no_final_labels": True,
    "no_ground_truth_claim": True,
    "no_production_claim": True,
    "not_validated_for_deployment": True,
    "no_raw_mat_modified": True,
    "no_formal_mask_applied": True,
    "no_threshold_change": True,
    "no_label_formula_change": True,
    "no_geometry_sign_change": True,
    "no_relbearing_policy_change": True,
    "no_model_training": True,
    "no_stc": True,
    "no_apes": True,
    "no_deep_learning": True,
}
REVIEW_ACTIONS = (
    "review_now",
    "review_with_qc_note",
    "defer_pending_horizontal_stripe_qc",
)


@dataclass(frozen=True)
class StripeManualReviewConfig:
    heatmap_half_window_ft: float = DEFAULT_HEATMAP_HALF_WINDOW_FT
    near_distance_ft: float = DEFAULT_NEAR_DISTANCE_FT


@dataclass(frozen=True)
class StripeManualReviewIntegrationResult:
    version: str
    generated_at: str
    selected_interval_count: int
    stripe_event_count: int
    stripe_target_sensitivity_max: float | None
    heatmap_dir: str
    heatmap_png_count: int
    action_counts: dict[str, int]
    stripe_in_heatmap_window_count: int
    stripe_crosses_review_depth_count: int
    stripe_near_review_depth_count: int
    output_csv: str
    output_json: str
    summary_md: str
    reviewer_checklist: str
    reviewer_decision_template: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), **METHOD_FLAGS}


def integrate_stripe_qc_with_manual_review_from_paths(
    *,
    selected_intervals_csv: Path | str,
    stripe_inventory_csv: Path | str,
    stripe_sensitivity_csv: Path | str,
    manual_review_dir: Path | str,
    heatmap_dir: Path | str,
    overwrite: bool = False,
    config: StripeManualReviewConfig | None = None,
) -> StripeManualReviewIntegrationResult:
    review_dir = Path(manual_review_dir)
    outputs = {
        "csv": review_dir / "selected_intervals_stripe_qc.csv",
        "json": review_dir / "selected_intervals_stripe_qc.json",
        "summary": review_dir / "stripe_qc_review_summary.md",
    }
    for path in outputs.values():
        _ensure_can_write(path, overwrite=overwrite)
    selected = _read_csv(Path(selected_intervals_csv))
    stripes = _read_csv(Path(stripe_inventory_csv))
    sensitivity = _read_csv(Path(stripe_sensitivity_csv))
    result, rows, summary = integrate_stripe_qc_with_manual_review(
        selected_intervals=selected,
        stripe_inventory=stripes,
        stripe_sensitivity=sensitivity,
        manual_review_dir=review_dir,
        heatmap_dir=Path(heatmap_dir),
        config=config,
    )
    _write_csv(rows, outputs["csv"])
    _write_json(outputs["json"], {"summary": summary, "selected_intervals": rows, **METHOD_FLAGS})
    outputs["summary"].write_text(format_stripe_qc_review_summary(summary), encoding="utf-8")
    update_reviewer_checklist(review_dir / "reviewer_checklist.md")
    update_reviewer_decision_template(review_dir / "reviewer_decision_template.md")
    return StripeManualReviewIntegrationResult(
        **{
            **asdict(result),
            "output_csv": str(outputs["csv"]),
            "output_json": str(outputs["json"]),
            "summary_md": str(outputs["summary"]),
            "reviewer_checklist": str(review_dir / "reviewer_checklist.md"),
            "reviewer_decision_template": str(review_dir / "reviewer_decision_template.md"),
        }
    )


def integrate_stripe_qc_with_manual_review(
    *,
    selected_intervals: list[dict[str, str]],
    stripe_inventory: list[dict[str, str]],
    stripe_sensitivity: list[dict[str, str]],
    manual_review_dir: Path,
    heatmap_dir: Path,
    config: StripeManualReviewConfig | None = None,
) -> tuple[StripeManualReviewIntegrationResult, list[dict[str, Any]], dict[str, Any]]:
    cfg = config or StripeManualReviewConfig()
    warnings: list[str] = []
    if not selected_intervals:
        raise ValueError("selected_intervals.csv is empty.")
    if not stripe_inventory:
        warnings.append("Stripe inventory is empty; all intervals default to review_now.")
    if not heatmap_dir.exists():
        warnings.append(f"CAST heatmap directory does not exist: {heatmap_dir}")
    heatmap_png_count = len(list(heatmap_dir.glob("*.png"))) if heatmap_dir.exists() else 0
    sensitivity_max = max_target_sensitivity(stripe_sensitivity)
    stripe_events = [_parse_stripe_event(row) for row in stripe_inventory]
    rows = [
        augment_selected_interval(
            row,
            stripe_events=stripe_events,
            sensitivity_max=sensitivity_max,
            config=cfg,
        )
        for row in selected_intervals
    ]
    action_counts = {action: 0 for action in REVIEW_ACTIONS}
    for row in rows:
        action_counts[str(row["recommended_review_action"])] += 1
    summary = {
        "version": STRIPE_MANUAL_REVIEW_INTEGRATION_VERSION,
        "generated_at": _utc_now(),
        "selected_interval_count": len(rows),
        "stripe_event_count": len(stripe_events),
        "stripe_target_sensitivity_max": sensitivity_max,
        "heatmap_dir": str(heatmap_dir),
        "heatmap_png_count": heatmap_png_count,
        "heatmap_half_window_ft": cfg.heatmap_half_window_ft,
        "near_distance_ft": cfg.near_distance_ft,
        "action_counts": action_counts,
        "stripe_in_heatmap_window_count": sum(
            _as_bool(row["stripe_in_heatmap_window"]) for row in rows
        ),
        "stripe_crosses_review_depth_count": sum(
            _as_bool(row["stripe_crosses_review_depth"]) for row in rows
        ),
        "stripe_near_review_depth_count": sum(
            _as_bool(row["stripe_near_review_depth"]) for row in rows
        ),
        "warnings": warnings,
        "review_rules": {
            "crosses_review_depth": "defer_pending_horizontal_stripe_qc",
            "nearest_distance_lte_3ft": "defer_pending_horizontal_stripe_qc",
            "in_heatmap_window_distance_gt_3ft": "review_with_qc_note",
            "no_nearby_stripe": "review_now",
        },
        **METHOD_FLAGS,
    }
    result = StripeManualReviewIntegrationResult(
        version=STRIPE_MANUAL_REVIEW_INTEGRATION_VERSION,
        generated_at=str(summary["generated_at"]),
        selected_interval_count=len(rows),
        stripe_event_count=len(stripe_events),
        stripe_target_sensitivity_max=sensitivity_max,
        heatmap_dir=str(heatmap_dir),
        heatmap_png_count=heatmap_png_count,
        action_counts=action_counts,
        stripe_in_heatmap_window_count=int(summary["stripe_in_heatmap_window_count"]),
        stripe_crosses_review_depth_count=int(summary["stripe_crosses_review_depth_count"]),
        stripe_near_review_depth_count=int(summary["stripe_near_review_depth_count"]),
        output_csv="",
        output_json="",
        summary_md="",
        reviewer_checklist=str(manual_review_dir / "reviewer_checklist.md"),
        reviewer_decision_template=str(manual_review_dir / "reviewer_decision_template.md"),
        warnings=warnings,
    )
    return result, rows, summary


def augment_selected_interval(
    interval: dict[str, str],
    *,
    stripe_events: list[dict[str, Any]],
    sensitivity_max: float | None,
    config: StripeManualReviewConfig,
) -> dict[str, Any]:
    depth = _float(interval["depth"])
    nearest = nearest_stripe_event(depth, stripe_events)
    output: dict[str, Any] = dict(interval)
    if nearest is None:
        output.update(
            {
                "nearest_stripe_depth_ft": None,
                "nearest_stripe_distance_ft": None,
                "stripe_in_heatmap_window": False,
                "stripe_crosses_review_depth": False,
                "stripe_near_review_depth": False,
                "stripe_event_type": "",
                "stripe_azimuth_coverage": None,
                "stripe_thickness_rows": None,
                "stripe_target_sensitivity_max": sensitivity_max,
                "recommended_review_action": "review_now",
                "reviewer_note_template": _note_template("review_now", depth, None),
                **METHOD_FLAGS,
            }
        )
        return output
    distance = float(nearest["distance_ft"])
    crosses = bool(distance == 0.0)
    near = bool(distance <= config.near_distance_ft)
    in_window = bool(distance <= config.heatmap_half_window_ft)
    if crosses or near:
        action = "defer_pending_horizontal_stripe_qc"
    elif in_window:
        action = "review_with_qc_note"
    else:
        action = "review_now"
    event = nearest["event"]
    output.update(
        {
            "nearest_stripe_depth_ft": event["stripe_depth_ft"],
            "nearest_stripe_distance_ft": distance,
            "stripe_in_heatmap_window": in_window,
            "stripe_crosses_review_depth": crosses,
            "stripe_near_review_depth": near,
            "stripe_event_type": event["event_class"],
            "stripe_event_family": event["event_family"],
            "stripe_event_id": event["event_id"],
            "stripe_azimuth_coverage": event["azimuth_coverage"],
            "stripe_thickness_rows": event["row_count"],
            "stripe_target_sensitivity_max": sensitivity_max,
            "recommended_review_action": action,
            "reviewer_note_template": _note_template(action, depth, event, distance),
            **METHOD_FLAGS,
        }
    )
    return output


def nearest_stripe_event(
    depth_ft: float,
    stripe_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not stripe_events:
        return None
    best: dict[str, Any] | None = None
    for event in stripe_events:
        distance = _distance_to_event(depth_ft, event)
        if best is None or distance < float(best["distance_ft"]):
            best = {"event": event, "distance_ft": distance}
    return best


def max_target_sensitivity(rows: list[dict[str, str]]) -> float | None:
    values: list[float] = []
    for row in rows:
        if row.get("recompute_status") != "recomputed_audit_only":
            continue
        if row.get("scenario") == "keep_all_rows":
            continue
        value = _optional_float(row.get("max_abs_delta_vs_keep_all", ""))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def format_stripe_qc_review_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stripe-Aware Manual Review Integration",
            "",
            "Scope: research_only, exploratory_only, weak_label_target, no_final_labels, "
            "no_ground_truth_claim, no_production_claim, not_validated_for_deployment.",
            "",
            "This integrates CAST horizontal stripe QC into the manual review pack only. "
            "It does not mask data, change labels, change thresholds, change geometry, train "
            "models, or approve production use.",
            "",
            f"- selected_interval_count: `{summary['selected_interval_count']}`",
            f"- stripe_event_count: `{summary['stripe_event_count']}`",
            f"- heatmap_png_count: `{summary['heatmap_png_count']}`",
            f"- stripe_target_sensitivity_max: `{summary['stripe_target_sensitivity_max']}`",
            f"- stripe_in_heatmap_window_count: `{summary['stripe_in_heatmap_window_count']}`",
            "- stripe_crosses_review_depth_count: "
            f"`{summary['stripe_crosses_review_depth_count']}`",
            f"- stripe_near_review_depth_count: `{summary['stripe_near_review_depth_count']}`",
            f"- action_counts: `{json.dumps(summary['action_counts'], sort_keys=True)}`",
            "",
            "## Review Actions",
            "",
            "- `defer_pending_horizontal_stripe_qc`: review depth crosses a stripe "
            "or is within 3 ft.",
            "- `review_with_qc_note`: stripe appears within the CAST heatmap window "
            "but is >3 ft away.",
            "- `review_now`: no stripe in the CAST heatmap window.",
            "",
            "## Warnings",
            "",
            *_message_lines(summary.get("warnings", [])),
            "",
        ]
    )


def update_reviewer_checklist(path: Path) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Reviewer Checklist\n"
    section = "\n".join(
        [
            "## Stripe-Aware Manual Review Integration",
            "",
            "- Check `recommended_review_action` before interpreting CAST evidence.",
            "- If `defer_pending_horizontal_stripe_qc`, do not adjudicate label semantics until "
            "horizontal stripe processing/casing metadata review is resolved.",
            "- If `review_with_qc_note`, inspect the stripe location in the CAST heatmap and note "
            "whether it may bias receiver_mean, p90, local-worst, connectedness, or persistence.",
            "- If `review_now`, proceed with ordinary weak-label semantics review.",
            "- Never use stripe QC to mask rows, change the 2.5 MRayl threshold, "
            "or approve final labels.",
            "",
        ]
    )
    path.write_text(
        _replace_section(existing, "## Stripe-Aware Manual Review Integration", section),
        encoding="utf-8",
    )


def update_reviewer_decision_template(path: Path) -> None:
    existing = (
        path.read_text(encoding="utf-8") if path.exists() else "# Reviewer Decision Template\n"
    )
    section = "\n".join(
        [
            "## Stripe QC Decision Fields",
            "",
            "- recommended_review_action: review_now / review_with_qc_note / "
            "defer_pending_horizontal_stripe_qc",
            "- qc_flag_horizontal_stripe affects interpretation: yes/no/uncertain",
            "- casing_collar_metadata_required: yes/no",
            "- processing_review_required_before_label_decision: yes/no",
            "- final_label_approval: no",
            "- notes:",
            "",
        ]
    )
    path.write_text(
        _replace_section(existing, "## Stripe QC Decision Fields", section),
        encoding="utf-8",
    )


def _parse_stripe_event(row: dict[str, str]) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id", ""),
        "event_family": row.get("event_family", ""),
        "event_class": row.get("event_class", ""),
        "stripe_depth_ft": _float(row["stripe_depth_ft"]),
        "depth_min_ft": _float(row["depth_min_ft"]),
        "depth_max_ft": _float(row["depth_max_ft"]),
        "azimuth_coverage": _float(row["azimuth_coverage"]),
        "row_count": int(float(row["row_count"])),
    }


def _distance_to_event(depth_ft: float, event: dict[str, Any]) -> float:
    low = min(float(event["depth_min_ft"]), float(event["depth_max_ft"]))
    high = max(float(event["depth_min_ft"]), float(event["depth_max_ft"]))
    if low <= depth_ft <= high:
        return 0.0
    return min(abs(depth_ft - low), abs(depth_ft - high))


def _note_template(
    action: str,
    depth_ft: float,
    event: dict[str, Any] | None,
    distance_ft: float | None = None,
) -> str:
    if event is None:
        return (
            f"Depth {depth_ft:.2f} ft: no nearby horizontal high-Zc stripe in CAST heatmap "
            "window; proceed with weak-label semantics review only."
        )
    return (
        f"Depth {depth_ft:.2f} ft: nearest stripe {event['event_id']} "
        f"({event['event_class']}) is {distance_ft:.2f} ft away; action={action}. "
        "Do not mask data or change label formula from this QC flag alone."
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


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


def _float(value: str) -> float:
    return float(value)


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _message_lines(messages: list[str]) -> list[str]:
    return [f"- {message}" for message in messages] if messages else ["- none"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
