from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cement_channel.training.preprocessing_diagnostics import (
    diagnose_feature_preprocessing,
    standardized_differences_by_subset,
)

pytest.importorskip("matplotlib")


def test_standardized_differences_by_subset_computes_high_confidence_difference() -> None:
    transformed = np.array([[2.0], [3.0], [0.0], [1.0]], dtype=np.float32)
    presence = np.array([1, 1, 0, 0], dtype=np.int8)

    result = standardized_differences_by_subset(
        transformed=transformed,
        transformed_names=["robust_scaled_rms_energy"],
        presence=presence,
        valid_azimuthal=np.ones(4, dtype=bool),
        disagreement=np.zeros(4, dtype=bool),
    )

    assert result["high_confidence:robust_scaled_rms_energy"]["candidate_count"] == 2
    assert result["high_confidence:robust_scaled_rms_energy"]["non_candidate_count"] == 2
    assert result["high_confidence:robust_scaled_rms_energy"]["standardized_difference"] > 0.0


def test_diagnose_feature_preprocessing_writes_figures(tmp_path: Path) -> None:
    sample_npz = tmp_path / "baseline_sample_table_v001.npz"
    features = np.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0], [6.0, 7.0]], dtype=np.float32)
    transformed = np.column_stack(
        [
            np.log1p(features),
            np.array([[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32),
        ]
    )
    np.savez_compressed(
        sample_npz,
        features=features,
        feature_names=np.array(["rms_energy", "late_over_early_ratio"]),
        transformed_features=transformed.astype(np.float32),
        transformed_feature_names=np.array(
            [
                "log1p_rms_energy",
                "log1p_late_over_early_ratio",
                "robust_scaled_rms_energy",
                "robust_scaled_late_over_early_ratio",
            ]
        ),
        label_presence_plus=np.array([0, 0, 1, 1], dtype=np.int8),
        valid_for_azimuthal_validation=np.ones(4, dtype=bool),
        plus_minus_disagreement=np.zeros(4, dtype=bool),
        sample_weight=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        depth_match_error=np.zeros(4, dtype=np.float32),
        transform_stats_json=np.asarray(
            '{"rms_energy":{"clipped_count":1,"clip_low":0.0,"clip_high":2.0},'
            '"late_over_early_ratio":{"clipped_count":0,"clip_low":0.0,"clip_high":2.0}}'
        ),
        no_model_training=np.asarray(True),
        no_final_labels=np.asarray(True),
    )

    report = diagnose_feature_preprocessing(
        sample_table_npz=sample_npz,
        output_dir=tmp_path / "diagnostics",
        overwrite=False,
        max_samples=100,
    )

    assert report.errors == []
    assert report.no_model_training
    assert report.no_final_labels
    assert len(report.figures) == 5
    assert report.outliers["rms_energy"]["clipped_count_before"] == 1
    for figure in report.figures.values():
        assert Path(figure).read_bytes().startswith(b"\x89PNG")
