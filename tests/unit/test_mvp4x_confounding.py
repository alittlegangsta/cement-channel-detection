from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_confounding import (
    build_cf_masks,
    build_common_support_audit,
    build_depth_target_matched_controls,
)


def _snapshot() -> dict[str, np.ndarray]:
    depth = np.asarray([1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350], dtype=np.float32)
    low = np.asarray([True, True, False, False, True, False, False, False])
    regimes = np.asarray(["A", "A", "B", "B", "B", "C", "C", "C"])
    features = np.arange(depth.size * 3, dtype=np.float32).reshape(depth.size, 3)
    return {
        "depth": depth,
        "low_orientation_confidence_flag": low,
        "broad_regime_id": regimes,
        "broad_regime_code": np.asarray([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.float32),
        "saturation_platform_flag": np.zeros(depth.size, dtype=bool),
        "special_5680_flag": np.asarray([False, False, False, False, False, False, True, False]),
        "any_special_flag": np.asarray([False, False, False, False, False, False, True, False]),
        "xsi_features": features,
        "orientation_confidence": np.where(low, 0.1, 0.9).astype(np.float32),
        "receiver_mean": np.linspace(0.0, 0.2, depth.size, dtype=np.float32),
        "receiver_p90": np.linspace(0.0, 0.3, depth.size, dtype=np.float32),
        "receiver_max": np.linspace(0.0, 0.4, depth.size, dtype=np.float32),
        "full_360_fraction": np.linspace(0.0, 0.5, depth.size, dtype=np.float32),
        "receiver_std": np.linspace(0.0, 0.1, depth.size, dtype=np.float32),
    }


def test_build_cf_masks_includes_regime_orientation_crosses() -> None:
    masks = build_cf_masks(_snapshot())

    assert masks["all_samples"].sum() == 8
    assert masks["low_orientation"].sum() == 3
    assert masks["high_orientation"].sum() == 5
    assert masks["regime_a_high_orientation"].sum() == 0
    assert masks["regime_b_low_orientation"].sum() == 1
    assert masks["regime_b_high_orientation"].sum() == 2
    assert masks["regime_c_low_orientation"].sum() == 0
    assert masks["high_orientation_exclude_5680"].sum() == 4


def test_common_support_marks_orientation_effect_not_identifiable() -> None:
    snapshot = _snapshot()
    masks = build_cf_masks(snapshot)
    waveform = {"waveform_depth_features": np.ones((8, 2), dtype=np.float32)}

    report = build_common_support_audit(
        snapshot=snapshot,
        waveform=waveform,
        masks=masks,
        config={
            "common_support_depth_bin_width_ft": 100.0,
            "common_support_min_bin_samples": 1,
            "common_support_min_group_samples": 1,
        },
    )

    assert report["answers"]["1_regime_a_completely_lacks_high_orientation_support"]
    assert report["answers"]["2_regime_c_completely_lacks_low_orientation_support"]
    assert report["orientation_effect_not_identifiable_from_current_well"]
    assert (
        "orientation_effect_not_identifiable_from_current_well" in report["identifiability_warning"]
    )


def test_depth_target_matched_controls_preserve_reference_count() -> None:
    snapshot = _snapshot()
    masks = build_cf_masks(snapshot)
    depth = snapshot["depth"]
    target = snapshot["receiver_mean"]
    reference = masks["high_orientation"]

    controls, rows = build_depth_target_matched_controls(
        depth=depth,
        target=target,
        reference_mask=reference,
        candidate_mask=np.ones(depth.size, dtype=bool),
        repeat_count=3,
        seed=7,
        bin_width_ft=200.0,
        target_quantile_count=2,
        high_orientation_mask=masks["high_orientation"],
        regimes=snapshot["broad_regime_id"],
    )

    assert len(controls) == 3
    assert len(rows) == 3
    assert all(mask.sum() <= reference.sum() for mask in controls)
    assert all(0.0 <= row["high_orientation_fraction"] <= 1.0 for row in rows)
