from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cement_channel.visualization.geometry_regression_formal_review import (
    generate_geometry_regression_formal_review,
)


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    depth = np.linspace(2350.0, 5750.0, 36, dtype=np.float32)
    kernels = np.asarray(
        [
            "r7_reference_point",
            "midpoint_window",
            "uniform_source_receiver_interval",
            "triangular_midpoint_weighted",
        ]
    )
    base = np.r_[
        np.ones(12) * 0.34,
        np.ones(12) * 0.07,
        np.linspace(0.06, 0.14, 12),
    ].astype(np.float32)
    targets = np.vstack([base * 0.9, base * 0.85, base * 0.8, base]).astype(np.float32)
    cube = np.repeat(targets[:, :, None], 13, axis=2)
    labels_npz = root / "labels.npz"
    np.savez_compressed(
        labels_npz,
        depth=depth,
        geometry_kernel=kernels,
        receiver_mean=targets * 0.7,
        receiver_p90=targets * 0.9,
        receiver_max=targets,
        full_360_fraction=np.clip(targets * 1.2, 0.0, 1.0),
        receiver_std=targets * 0.25,
        weighted_channel_fraction_zc_lt_2p5=cube * 0.8,
        largest_connected_component_fraction=cube,
        max_azimuth_channel_fraction=np.clip(cube * 2.5, 0.0, 1.0),
        relative_anomaly_fraction=cube * 0.7,
        combined_channel_fraction=cube,
        depth_label_confidence=np.ones_like(cube, dtype=np.float32),
        total_cell_count=np.ones_like(cube, dtype=np.int32),
        no_final_labels=np.asarray(True),
    )
    feature0 = np.linspace(0.0, 0.1, depth.size, dtype=np.float32)
    feature0[2] = 10.0
    feature1 = np.linspace(0.05, 0.2, depth.size, dtype=np.float32)
    feature2 = np.linspace(0.2, 0.05, depth.size, dtype=np.float32)
    features_npz = root / "features.npz"
    np.savez_compressed(
        features_npz,
        depth=depth,
        depth_level_xsi_features=np.column_stack([feature0, feature1, feature2]).astype(
            np.float32
        ),
        depth_level_xsi_feature_names=np.asarray(
            [
                "near_far_ratio_mean_early_energy",
                "near_far_ratio_mean_rms_energy",
                "near_far_ratio_mean_peak_abs",
            ]
        ),
        feature_group_names=np.asarray(["near_far_receiver_ratio"]),
        feature_group_counts_json=np.asarray(
            json.dumps({"near_far_receiver_ratio": 3})
        ),
        no_final_labels=np.asarray(True),
    )
    cast_depth = np.linspace(2350.0, 5750.0, 90, dtype=np.float32)
    cast_zc = np.ones((90, 8), dtype=np.float32) * 3.0
    cast_zc[2, 0] = -1.0
    cast_npz = root / "cast.npz"
    np.savez_compressed(
        cast_npz,
        cast_depth=cast_depth,
        cast_zc=cast_zc,
        orientation_confidence=np.linspace(1.0, 0.55, cast_depth.size).astype(np.float32),
        inc_deg=np.linspace(3.0, 8.0, cast_depth.size).astype(np.float32),
        relbearing_deg=np.linspace(0.0, 90.0, cast_depth.size).astype(np.float32),
    )
    return labels_npz, features_npz, cast_npz


def test_formal_review_pack_generates_required_outputs(tmp_path: Path) -> None:
    labels_npz, features_npz, cast_npz = _write_inputs(tmp_path)
    output_dir = tmp_path / "formal_review"
    pack = generate_geometry_regression_formal_review(
        labels_npz=labels_npz,
        features_npz=features_npz,
        cast_npz=cast_npz,
        policy_config="configs/geometry_regression_depth_regime_review.example.yaml",
        output_dir=output_dir,
        decision_pack_md=tmp_path / "decision.md",
        decision_pack_json=tmp_path / "decision.json",
    )

    assert pack.no_formal_cv_split_change is True
    assert pack.no_model_training is True
    assert pack.selected_manual_review_interval_count >= 28
    assert (output_dir / "reviewer_checklist.md").exists()
    assert (output_dir / "selected_manual_review_intervals.csv").exists()
    assert (output_dir / "boundary_2582p79_ft.png").exists()
    assert (output_dir / "boundary_4219p52_ft.png").exists()
    assert (output_dir / "boundary_5680p00_ft.png").exists()
    intervals = json.loads(
        (output_dir / "selected_manual_review_intervals.json").read_text(encoding="utf-8")
    )
    required = {
        "interval_id",
        "review_type",
        "depth_min",
        "depth_max",
        "regime_id",
        "receiver_mean",
        "receiver_p90",
        "receiver_max",
        "full_360_fraction",
        "receiver_std",
        "morphology_summary",
        "orientation_confidence",
        "inclination",
        "near_far_ratio_summary",
        "recommended_human_question",
        "review_only",
        "no_final_labels",
    }
    assert required.issubset(intervals[0])
    assert all(row["review_only"] is True for row in intervals)
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["manual_review_interval_count"] == len(intervals)
    assert decision["stage_10_stop_still_valid"] is True
    assert "formal CV split change" in decision["not_authorized"]

