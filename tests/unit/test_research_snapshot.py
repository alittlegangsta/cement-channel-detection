from __future__ import annotations

import numpy as np

from cement_channel.experiments.research_snapshot import build_research_snapshot


def _config() -> dict:
    return {
        "schema_version": "schema_v001",
        "scope": {
            "research_only": True,
            "exploratory_only": True,
            "weak_label_target": True,
            "no_final_labels": True,
            "no_ground_truth_claim": True,
            "no_production_claim": True,
        },
        "snapshot": {
            "target_kernel": "triangular_midpoint_weighted",
            "broad_regimes": [
                {"id": "A", "min_depth_ft": 0.0, "max_depth_ft": 10.0},
                {"id": "B", "min_depth_ft": 10.0, "max_depth_ft": 20.0},
            ],
            "special_bands": {
                "saturation_platform": {"min_depth_ft": 2.0, "max_depth_ft": 4.0},
                "transition_2582": {"center_depth_ft": 12.0, "half_width_ft": 1.0},
            },
            "low_orientation_confidence_threshold": 0.5,
        },
    }


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    depth = np.arange(6, dtype=np.float32)
    kernels = np.asarray(["r7_reference_point", "triangular_midpoint_weighted"])
    target = np.vstack([depth, depth + 10.0]).astype(np.float32)
    receiver_cube = np.ones((2, 6, 3), dtype=np.float32)
    labels = {
        "depth": depth,
        "geometry_kernel": kernels,
        "receiver_p90": target,
        "receiver_mean": target + 1.0,
        "receiver_max": target + 2.0,
        "full_360_fraction": target + 3.0,
        "receiver_std": target + 4.0,
        "depth_label_confidence": receiver_cube,
        "min_zc": receiver_cube,
        "p05_zc": receiver_cube,
        "p10_zc": receiver_cube,
        "max_relative_drop": receiver_cube,
        "largest_connected_component_fraction": receiver_cube,
        "max_azimuth_channel_fraction": receiver_cube,
        "candidate_cell_count": receiver_cube.astype(np.int32),
        "total_cell_count": receiver_cube.astype(np.int32),
    }
    features = {
        "depth": depth,
        "depth_level_xsi_features": np.column_stack([depth, depth + 1.0]).astype(np.float32),
        "depth_level_xsi_feature_names": np.asarray(["side_mean_energy", "receiver_max_rms"]),
        "feature_group_names": np.asarray(["side_mean", "receiver_max"]),
        "feature_group_counts_json": np.asarray('{"receiver_max": 1, "side_mean": 1}'),
    }
    cast = {
        "cast_depth": depth,
        "orientation_confidence": np.asarray([1.0, 0.9, 0.4, 1.0, 1.0, 1.0], dtype=np.float32),
        "inc_deg": np.ones(6, dtype=np.float32) * 20.0,
        "low_inc_mask": np.zeros(6, dtype=bool),
        "orientation_uncertain": np.zeros(6, dtype=bool),
    }
    return labels, features, cast


def test_build_research_snapshot_selects_kernel_and_metadata_only_flags() -> None:
    labels, features, cast = _arrays()

    arrays, report, manifest = build_research_snapshot(
        regression_labels=labels,
        depth_level_features=features,
        cast_label_input=cast,
        config=_config(),
        inputs={"regression_labels_npz": "labels.npz"},
    )

    assert report.errors == []
    assert report.target_kernel_index == 1
    assert np.allclose(arrays["receiver_p90"], np.arange(6, dtype=np.float32) + 10.0)
    assert arrays["model_feature_mask"].tolist() == [True, True]
    assert arrays["broad_regime_id"].tolist() == ["A"] * 6
    assert arrays["saturation_platform_flag"].tolist() == [False, False, True, True, True, False]
    assert arrays["low_orientation_confidence_flag"].tolist()[2] is True
    assert manifest["scope"]["research_only"] is True
    assert report.no_ground_truth_claim is True

