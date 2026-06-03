from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_autonomous import (
    build_autonomous_masks,
    build_depth_matched_controls,
    build_subgroup_audit,
)


def _snapshot(sample_count: int = 12) -> dict[str, np.ndarray]:
    depth = np.linspace(2350.0, 5750.0, sample_count, dtype=np.float32)
    low_orientation = np.zeros(sample_count, dtype=bool)
    low_orientation[::3] = True
    regime = np.asarray(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    if sample_count != 12:
        regime = np.resize(regime, sample_count)
    return {
        "depth": depth,
        "orientation_confidence": np.linspace(0.1, 0.9, sample_count, dtype=np.float32),
        "low_orientation_confidence_flag": low_orientation,
        "saturation_platform_flag": (depth >= 2400.0) & (depth <= 2500.0),
        "transition_2582_flag": np.zeros(sample_count, dtype=bool),
        "transition_4219_flag": np.zeros(sample_count, dtype=bool),
        "special_5680_flag": depth >= 5680.0,
        "any_special_flag": (depth >= 5680.0) | ((depth >= 2400.0) & (depth <= 2500.0)),
        "broad_regime_id": regime,
        "receiver_mean": np.linspace(0.0, 1.0, sample_count, dtype=np.float32),
        "receiver_p90": np.linspace(0.0, 1.0, sample_count, dtype=np.float32),
        "receiver_max": np.linspace(0.0, 1.0, sample_count, dtype=np.float32),
        "full_360_fraction": np.linspace(0.0, 1.0, sample_count, dtype=np.float32),
        "receiver_std": np.linspace(0.0, 1.0, sample_count, dtype=np.float32),
        "xsi_features": np.ones((sample_count, 2), dtype=np.float32),
        "morphology_min_zc": np.linspace(1.0, 2.0, sample_count, dtype=np.float32),
    }


def test_build_autonomous_masks_includes_high_orientation_and_regimes() -> None:
    masks = build_autonomous_masks(_snapshot())

    assert masks["all_samples"].sum() == 12
    assert masks["low_orientation"].sum() == 4
    assert masks["high_orientation"].sum() == 8
    assert masks["regime_a"].sum() == 4
    assert masks["regime_a_high_orientation"].sum() == 2
    assert "high_orientation_exclude_all_special" in masks


def test_build_depth_matched_controls_preserves_reference_count() -> None:
    depth = np.arange(20, dtype=np.float32)
    reference = np.zeros(20, dtype=bool)
    reference[[1, 2, 5, 8, 13, 17]] = True
    candidate = np.ones(20, dtype=bool)

    controls = build_depth_matched_controls(
        depth=depth,
        reference_mask=reference,
        candidate_mask=candidate,
        repeat_count=3,
        seed=42,
        bin_count=4,
    )

    assert len(controls) == 3
    assert [int(mask.sum()) for mask in controls] == [int(reference.sum())] * 3


def test_build_subgroup_audit_reports_counts_and_finite_ratios() -> None:
    snapshot = _snapshot()
    waveform = {"waveform_depth_features": np.ones((12, 3), dtype=np.float32)}
    audit = build_subgroup_audit(
        snapshot=snapshot,
        waveform=waveform,
        masks=build_autonomous_masks(snapshot),
    )

    high = audit["groups"]["high_orientation"]
    assert audit["research_only"] is True
    assert high["sample_count"] == 8
    assert high["finite_ratio"]["existing_features"] == 1.0
    assert high["finite_ratio"]["waveform_depth_features"] == 1.0
    assert "receiver_mean" in high["target_distribution"]
