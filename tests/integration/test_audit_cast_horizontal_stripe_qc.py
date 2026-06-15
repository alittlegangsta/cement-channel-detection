from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _write_inputs(root: Path) -> Path:
    interim = root / "interim"
    reports = root / "reports"
    features = root / "features"
    manifests = root / "manifests"
    review = reports / "mvp4x_label_semantics_manual_review_v001"
    for path in (interim, reports, features, manifests, review):
        path.mkdir(parents=True, exist_ok=True)
    cast_depth = np.arange(20, dtype=np.float32)
    cast_zc = np.full((20, 8), 3.2, dtype=np.float32)
    cast_zc[10, :] = 8.0
    cast_zc[14, :3] = 8.0
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=cast_depth,
        cast_zc=cast_zc,
        cast_azimuth_deg=np.linspace(0.0, 360.0, 8, endpoint=False, dtype=np.float32),
    )
    depth = cast_depth[::2]
    receiver_count = 2
    kernel_count = 1
    target = np.zeros((kernel_count, depth.size, receiver_count), dtype=np.float32)
    interval_min = np.repeat((depth - 0.5)[None, :, None], receiver_count, axis=2)
    interval_max = np.repeat((depth + 0.5)[None, :, None], receiver_count, axis=2)
    midpoint = np.repeat(depth[None, :, None], receiver_count, axis=2)
    total = np.full_like(target, 8, dtype=np.int32)
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=np.asarray(["triangular_midpoint_weighted"]),
        receiver_index=np.asarray([1, 2], dtype=np.int16),
        weighted_channel_fraction_zc_lt_2p5=target,
        interval_min_depth=interval_min,
        interval_max_depth=interval_max,
        midpoint_depth=midpoint,
        total_cell_count=total,
    )
    zeros = np.zeros(depth.size, dtype=np.float32)
    np.savez_compressed(
        interim / "mvp4x_parallel_label_candidates_v001.npz",
        receiver_mean_v1_reference=zeros,
        receiver_p90_robust_candidate=zeros,
        local_worst_sector_fraction=zeros,
        connected_channel_fraction=zeros,
        interval_persistence_weighted_fraction=zeros,
    )
    (review / "reviewer_checklist.md").write_text("# Checklist\n", encoding="utf-8")
    paths = root / "paths.yaml"
    paths.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root}",
                f"  interim: {interim}",
                f"  features: {features}",
                f"  reports: {reports}",
                f"  manifests: {manifests}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


def test_cast_horizontal_stripe_qc_cli_writes_reports(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path / "data")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06ao_audit_cast_horizontal_stripe_qc.py",
            "--paths",
            str(paths),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "near360_events=1" in result.stdout
    reports = tmp_path / "data" / "reports"
    report = json.loads((reports / "cast_horizontal_stripe_qc_report_v001.json").read_text())
    assert report["near360_event_count"] == 1
    assert report["partial_event_count"] == 1
    with (reports / "cast_horizontal_stripe_inventory_v001.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["event_class"] for row in rows} >= {
        "one-row stripe",
        "partial-azimuth high-Zc event",
    }
    sensitivity = list(
        csv.DictReader(
            (reports / "cast_horizontal_stripe_target_sensitivity_v001.csv").open(
                encoding="utf-8",
                newline="",
            )
        )
    )
    assert any(row["target"] == "receiver_mean_v1_reference" for row in sensitivity)
    checklist = (
        reports / "mvp4x_label_semantics_manual_review_v001" / "reviewer_checklist.md"
    ).read_text(encoding="utf-8")
    assert "qc_flag_horizontal_stripe" in checklist
    assert (reports / "cast_horizontal_stripe_review_v001" / "stripe_depth_inventory.png").exists()
    assert list((reports / "cast_horizontal_stripe_review_v001").glob("example_*_heatmap.png"))
