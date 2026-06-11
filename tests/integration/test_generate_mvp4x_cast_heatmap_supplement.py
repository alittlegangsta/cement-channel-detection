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
    reports = root / "reports" / "mvp4x_label_semantics_manual_review_v001"
    features = root / "features"
    manifests = root / "manifests"
    for path in (interim, reports, features, manifests):
        path.mkdir(parents=True, exist_ok=True)
    depth = np.linspace(1000.0, 1009.0, 10, dtype=np.float32)
    azimuth = np.asarray([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    zc = np.full((10, 4), 3.5, dtype=np.float32)
    zc[3:6, 1:3] = 2.1
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=depth,
        cast_zc=zc,
        cast_azimuth_deg=azimuth,
    )
    np.savez_compressed(
        interim / "cast_zc_baseline_v001.npz",
        cast_depth=depth,
        cast_azimuth_deg=azimuth,
        relative_drop=np.linspace(0.0, 1.0, 40, dtype=np.float32).reshape(10, 4),
    )
    with (reports / "selected_intervals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_rank", "sample_index", "depth", "selection_reason"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "review_rank": 1,
                "sample_index": 3,
                "depth": 1003.0,
                "selection_reason": "local_worst_high_mean_low",
            }
        )
        writer.writerow(
            {
                "review_rank": 2,
                "sample_index": 6,
                "depth": 1006.0,
                "selection_reason": "random_control",
            }
        )
    (reports / "review_summary.md").write_text("# Review Summary\n", encoding="utf-8")
    (reports / "reviewer_checklist.md").write_text("# Checklist\n", encoding="utf-8")
    paths_config = root / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  interim: {interim}",
                f"  features: {features}",
                f"  reports: {root / 'reports'}",
                f"  manifests: {manifests}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths_config


def test_generate_mvp4x_cast_heatmap_supplement_cli_writes_inventory(
    tmp_path: Path,
) -> None:
    paths_config = _write_inputs(tmp_path / "data")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07z_generate_mvp4x_cast_heatmap_supplement.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "heatmaps=2" in result.stdout
    review_dir = tmp_path / "data" / "reports" / "mvp4x_label_semantics_manual_review_v001"
    assert (review_dir / "interval_cast_heatmaps" / "interval_01_cast_heatmap.png").exists()
    assert (review_dir / "interval_cast_heatmaps" / "interval_02_cast_heatmap.png").exists()
    summary = json.loads(
        (review_dir / "cast_heatmap_generation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["heatmap_count"] == 2
    assert summary["connected_mask_traceable_count"] == 0
    assert summary["relative_drop_heatmap_count"] == 2
    inventory = json.loads(
        (review_dir / "interval_cast_heatmap_inventory.json").read_text(encoding="utf-8")
    )
    assert len(inventory["inventory"]) == 2
    assert inventory["inventory"][0]["connected_mask_traceable"] is False
    checklist = (review_dir / "reviewer_checklist.md").read_text(encoding="utf-8")
    assert "CAST raw Zc 是否存在局部低于 2.5 MRayl 的区域？" in checklist
