from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_inputs(root_dir: Path) -> None:
    interim = root_dir / "interim"
    reports = root_dir / "reports"
    review_dir = reports / "depth_level_manual_review_v001"
    interim.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    depth = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
    np.savez_compressed(
        interim / "geometry_aware_depth_labels_v001.npz",
        depth=depth,
        mode=np.asarray(["r7_reference_depth", "source_receiver_interval"]),
        sign=np.asarray([1, -1], dtype=np.int8),
        candidate_fraction=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.4, 0.5]], dtype=np.float32),
        has_channel_any=np.asarray([[False, False, False], [False, True, True]]),
        max_severity=np.asarray([[0, 0, 0], [0, 2, 3]], dtype=np.int8),
        max_confidence=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.8, 0.9]], dtype=np.float32),
        max_relative_drop=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.7]], dtype=np.float32),
        depth_label_confidence=np.asarray([[1.0, 1.0, 1.0], [1.0, 0.8, 0.9]], dtype=np.float32),
        high_confidence_positive_mask=np.asarray([[False, False, False], [False, True, True]]),
        no_final_labels=np.asarray(True),
    )
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.ones((3, 1), dtype=np.float32),
        depth_level_xsi_feature_names=np.asarray(["receiver_mean_peak_abs"]),
        no_final_labels=np.asarray(True),
    )
    (reports / "geometry_alignment_audit_v001.json").write_text(
        json.dumps(
            {
                "no_final_labels": True,
                "combo_summaries": [
                    {
                        "mode": "r7_reference_depth",
                        "sign": 1,
                        "status": "runnable",
                        "positive_count": 1,
                        "negative_count": 2,
                        "real_minus_permutation_margin": 0.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review_dir.joinpath("review_intervals.json").write_text(
        json.dumps(
            {
                "intervals": [
                    {
                        "review_id": "DLR-009",
                        "interval_type": "clear_negative_like",
                        "start_depth": 10.0,
                        "end_depth": 12.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    review_dir.joinpath("interval_cast_evidence_summary_table.json").write_text(
        json.dumps([{"interval_id": "DLR-009", "evidence_category": "clear_negative_evidence"}]),
        encoding="utf-8",
    )


def test_generate_geometry_alignment_review_cli_writes_supplement(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    _write_inputs(root_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {root_dir / 'interim'}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06af_generate_geometry_alignment_review.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Geometry-aware manual review supplement errors=0" in result.stdout
    output_dir = reports_dir / "geometry_aware_manual_review_v001"
    assert (output_dir / "review_summary.md").exists()
    assert (output_dir / "geometry_mode_comparison.csv").exists()
    assert (output_dir / "interval_geometry_comparison.json").exists()
    assert (output_dir / "alignment_mode_comparison_overview.png").exists()
