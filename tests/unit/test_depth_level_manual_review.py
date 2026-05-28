from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from cement_channel.visualization.depth_level_manual_review import (  # noqa: E402
    generate_depth_level_manual_review_figures,
)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    depth = np.arange(100.0, 130.0, dtype=np.float32)
    labels_path = tmp_path / "depth_level_labels_v001.npz"
    features_path = tmp_path / "depth_level_xsi_features_v001.npz"
    np.savez_compressed(
        labels_path,
        depth=depth,
        depth_has_channel_any=(depth.astype(np.int32) % 3) == 0,
        depth_label_confidence=np.linspace(0.4, 0.9, depth.size, dtype=np.float32),
        depth_review_band_mask=(depth >= 118.0) & (depth <= 122.0),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    np.savez_compressed(
        features_path,
        depth=depth,
        depth_level_xsi_features=np.column_stack(
            [
                np.linspace(2.5e6, 2.7e6, depth.size),
                np.linspace(3.5e5, 3.9e5, depth.size),
                np.linspace(1.0e12, 1.7e12, depth.size),
                np.linspace(1.5e8, 2.0e8, depth.size),
                np.linspace(0.1, 0.3, depth.size),
            ]
        ).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(
            [
                "receiver_mean_peak_abs",
                "side_mean_rms_energy",
                "side_max_early_energy",
                "side_mean_late_energy",
                "side_max_late_over_early_ratio",
            ]
        ),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    review_json = tmp_path / "review_intervals.json"
    intervals = [
        {
            "review_id": "DLR-001",
            "start_depth": 101.0,
            "end_depth": 103.0,
            "interval_type": "true_positive_like",
            "5700_band_flag": False,
            "prediction_score_summary": {"score_mean": 0.8, "available": True},
            "confidence_summary": {"depth_label_confidence_mean": 0.8},
            "plus_minus_disagreement_summary": {"plus_minus_disagreement_max": 0.1},
            "cast_label_summary": {
                "weak_label_candidate_summary": {
                    "presence_plus_fraction": 0.5,
                    "severity_plus_max": 2,
                    "label_confidence_plus_mean": 0.7,
                },
                "cast_zc_summary": {"zc_p05": 2.4, "low_inc_fraction": 0.0},
            },
            "xsi_feature_summary": {
                "top_feature_values": [
                    {"feature_name": "f0", "mean": 1.0},
                    {"feature_name": "f1", "mean": 2.0},
                ]
            },
        },
        {
            "review_id": "DLR-002",
            "start_depth": 118.0,
            "end_depth": 122.0,
            "interval_type": "5700_band_review",
            "5700_band_flag": True,
            "prediction_score_summary": {"available": False},
            "confidence_summary": {"depth_label_confidence_mean": 0.6},
            "plus_minus_disagreement_summary": {"plus_minus_disagreement_max": 0.2},
            "cast_label_summary": {"weak_label_candidate_summary": {}, "cast_zc_summary": {}},
            "xsi_feature_summary": {"top_feature_values": []},
        },
    ]
    review_json.write_text(
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
    refinement_report = tmp_path / "depth_level_refinement_report_v001.json"
    refinement_report.write_text(
        json.dumps(
            {
                "best_result": {"scenario_id": "best"},
                "top_features": {
                    "best": [
                        {
                            "feature_name": "side_max_early_energy",
                            "mean_coefficient": 0.8,
                        },
                        {
                            "feature_name": "receiver_mean_peak_abs",
                            "mean_coefficient": -0.6,
                        },
                    ]
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "labels": labels_path,
        "features": features_path,
        "review_json": review_json,
        "refinement_report": refinement_report,
    }


def test_generate_depth_level_manual_review_figures_writes_outputs(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    report = generate_depth_level_manual_review_figures(
        review_intervals_json=paths["review_json"],
        depth_level_labels_npz=paths["labels"],
        depth_level_features_npz=paths["features"],
        output_dir=output_dir,
        overwrite=True,
        max_interval_panels=2,
        cast_weak_label_candidates_npz=tmp_path / "missing_cast_candidates.npz",
        refinement_report_json=paths["refinement_report"],
    )

    assert report.errors == []
    assert report.no_final_labels is True
    assert report.interval_cast_panel_count == 2
    assert report.interval_xsi_raw_panel_count == 2
    assert report.interval_xsi_normalized_panel_count == 2
    assert any("Optional manual review NPZ not found" in warning for warning in report.warnings)
    assert (output_dir / "overview_depth_label_score_confidence.png").read_bytes().startswith(
        b"\x89PNG"
    )
    assert (output_dir / "5700_band_sensitivity.png").exists()
    assert (output_dir / "interval_cast_panels" / "DLR-001_cast_review_panels.png").exists()
    assert (
        output_dir / "interval_xsi_feature_panels" / "DLR-001_xsi_raw_feature_multiples.png"
    ).exists()
    assert (
        output_dir / "interval_xsi_feature_panels" / "DLR-001_xsi_normalized_feature_panel.png"
    ).exists()
    xsi_rows = json.loads(
        (output_dir / "interval_xsi_feature_summary_table.json").read_text(encoding="utf-8")
    )
    finite_z = [
        row["robust_z_of_interval_mean"]
        for row in xsi_rows
        if row["feature_name"] == "side_max_early_energy"
    ]
    assert finite_z
    assert all(np.isfinite(value) for value in finite_z if value is not None)
    with (output_dir / "interval_cast_evidence_summary_table.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    assert fieldnames is not None
    assert {
        "interval_id",
        "presence_fraction",
        "severity_max",
        "zc_p05",
        "evidence_category",
        "has_candidate_mask",
    } <= set(fieldnames)
