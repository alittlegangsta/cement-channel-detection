from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(root: Path) -> Path:
    reports = root / "reports"
    review = reports / "mvp4x_label_semantics_manual_review_v001"
    heatmaps = review / "interval_cast_heatmaps"
    heatmaps.mkdir(parents=True, exist_ok=True)
    (heatmaps / "interval_01_cast_heatmap.png").write_bytes(b"png")
    _write_csv(
        review / "selected_intervals.csv",
        [
            {"review_rank": 1, "sample_index": 1, "depth": 100.5, "selection_reason": "a"},
            {"review_rank": 2, "sample_index": 2, "depth": 115.0, "selection_reason": "b"},
            {"review_rank": 3, "sample_index": 3, "depth": 150.0, "selection_reason": "c"},
        ],
    )
    _write_csv(
        reports / "cast_horizontal_stripe_inventory_v001.csv",
        [
            {
                "event_id": "cast_hstripe_0001",
                "event_family": "near_360_high_zc",
                "event_class": "multi-row band",
                "stripe_depth_ft": 100.5,
                "depth_min_ft": 100.0,
                "depth_max_ft": 101.0,
                "azimuth_coverage": 0.98,
                "row_count": 7,
            }
        ],
    )
    _write_csv(
        reports / "cast_horizontal_stripe_target_sensitivity_v001.csv",
        [
            {
                "target": "receiver_mean_v1_reference",
                "scenario": "exclude_stripe_rows_pm1",
                "recompute_status": "recomputed_audit_only",
                "max_abs_delta_vs_keep_all": 0.4,
            }
        ],
    )
    (review / "reviewer_checklist.md").write_text("# Checklist\n", encoding="utf-8")
    (review / "reviewer_decision_template.md").write_text(
        "# Reviewer Decision Template\n",
        encoding="utf-8",
    )
    paths = root / "paths.yaml"
    paths.write_text(
        "\n".join(
            [
                "data:",
                f"  reports: {reports}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


def test_integrate_stripe_qc_manual_review_cli_writes_augmented_outputs(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path / "data")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07ab_integrate_stripe_qc_manual_review.py",
            "--paths",
            str(paths),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "selected=3" in result.stdout
    review = tmp_path / "data" / "reports" / "mvp4x_label_semantics_manual_review_v001"
    output = json.loads((review / "selected_intervals_stripe_qc.json").read_text())
    actions = [row["recommended_review_action"] for row in output["selected_intervals"]]
    assert actions == [
        "defer_pending_horizontal_stripe_qc",
        "review_with_qc_note",
        "review_now",
    ]
    assert output["summary"]["stripe_target_sensitivity_max"] == 0.4
    assert "Stripe-Aware Manual Review Integration" in (
        review / "reviewer_checklist.md"
    ).read_text(encoding="utf-8")
    assert "Stripe QC Decision Fields" in (
        review / "reviewer_decision_template.md"
    ).read_text(encoding="utf-8")
