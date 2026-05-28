from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _write_inputs(root_dir: Path) -> None:
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports" / "depth_level_manual_review_v001"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    depth = np.arange(100.0, 125.0, dtype=np.float32)
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
        depth=depth,
        depth_has_channel_any=(depth.astype(np.int32) % 2) == 0,
        depth_label_confidence=np.ones(depth.size, dtype=np.float32) * 0.75,
        depth_review_band_mask=(depth >= 118.0) & (depth <= 122.0),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.column_stack(
            [
                np.linspace(2.5e6, 2.8e6, depth.size),
                np.linspace(3.0e5, 4.0e5, depth.size),
                np.linspace(1.0e12, 2.0e12, depth.size),
            ]
        ).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(
            ["receiver_mean_peak_abs", "side_mean_rms_energy", "side_max_early_energy"]
        ),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    intervals = [
        {
            "review_id": "DLR-001",
            "start_depth": 101.0,
            "end_depth": 103.0,
            "interval_type": "clear_negative_like",
            "5700_band_flag": False,
            "prediction_score_summary": {"score_mean": 0.2, "available": True},
            "confidence_summary": {"depth_label_confidence_mean": 0.8},
            "plus_minus_disagreement_summary": {"plus_minus_disagreement_max": 0.0},
            "cast_label_summary": {
                "weak_label_candidate_summary": {"presence_plus_fraction": 0.0},
                "cast_zc_summary": {"zc_p05": 4.0},
            },
            "xsi_feature_summary": {
                "top_feature_values": [{"feature_name": "f0", "mean": 1.0}]
            },
        }
    ]
    (reports_dir / "review_intervals.json").write_text(
        json.dumps(
            {
                "report": {
                    "review_pack_version": "depth_level_manual_review_v001",
                    "no_final_labels": True,
                },
                "intervals": intervals,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_generate_depth_level_manual_review_figures_cli_writes_outputs(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    _write_inputs(root_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {root_dir / 'interim'}",
                f"  reports: {root_dir / 'reports'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06ac_generate_depth_level_manual_review_figures.py",
            "--paths",
            str(paths_config),
            "--max-interval-panels",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level manual review figures errors=0" in result.stdout
    output_dir = root_dir / "reports" / "depth_level_manual_review_v001"
    assert (output_dir / "overview_depth_label_score_confidence.png").exists()
    assert (
        output_dir / "interval_xsi_feature_panels" / "DLR-001_xsi_raw_feature_multiples.png"
    ).exists()
    assert (
        output_dir / "interval_xsi_feature_panels" / "DLR-001_xsi_normalized_feature_panel.png"
    ).exists()
    assert (output_dir / "interval_cast_panels" / "DLR-001_cast_review_panels.png").exists()
    with (output_dir / "interval_xsi_feature_summary_table.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        assert "global_percentile_of_interval_mean" in (csv.DictReader(handle).fieldnames or [])
    summary = json.loads(
        (output_dir / "depth_level_manual_review_figures_summary_v001.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["review_figure_version"] == "depth_level_manual_review_figures_v001"
    assert summary["interval_xsi_panel_count"] == 2
