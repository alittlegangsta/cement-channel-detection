from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cement_channel.visualization.mvp4a_review import generate_mvp4a_review_figures

pytest.importorskip("matplotlib")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    label_npz = tmp_path / "xsi_label_samples_v001.npz"
    feature_npz = tmp_path / "xsi_basic_features_v001.npz"
    correlation_csv = tmp_path / "xsi_cast_correlation_v001.csv"
    presence = np.zeros((6, 8), dtype=np.int8)
    presence[2:4, 2:4] = 1
    severity = np.where(presence == 1, 2, 0).astype(np.int8)
    np.savez_compressed(
        label_npz,
        xsi_depth=np.arange(6, dtype=np.float32),
        xsi_side_azimuth_deg=np.arange(8, dtype=np.float32) * 45.0,
        label_presence_plus=presence,
        label_severity_plus=severity,
        label_confidence_plus=np.where(presence == 1, 0.8, 0.1).astype(np.float32),
        valid_for_azimuthal_validation=np.ones_like(presence, dtype=bool),
        plus_minus_disagreement=np.roll(presence, 1, axis=1) != presence,
        no_final_labels=np.asarray(True),
    )
    values = np.ones((6, 8, 2), dtype=np.float32)
    values[..., 0] += presence * 3.0
    np.savez_compressed(
        feature_npz,
        xsi_basic_features_by_side=values,
        feature_names=np.array(["rms_energy", "mean_abs"]),
        no_model_training=np.asarray(True),
    )
    correlation_csv.write_text(
        "\n".join(
            [
                "label_convention,subset,feature,point_biserial_effect_size,weighted_difference_fraction",
                "plus_primary,high_confidence,rms_energy,0.5,0.2",
                "plus_primary,all_known,mean_abs,0.1,0.05",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return label_npz, feature_npz, correlation_csv


def test_generate_mvp4a_review_figures_writes_expected_outputs(tmp_path: Path) -> None:
    label_npz, feature_npz, correlation_csv = _write_inputs(tmp_path)
    output_dir = tmp_path / "review"

    report = generate_mvp4a_review_figures(
        label_samples_npz=label_npz,
        basic_features_npz=feature_npz,
        correlation_csv=correlation_csv,
        output_dir=output_dir,
        overwrite=False,
        max_depth_pixels=10,
        max_distribution_samples=100,
    )

    assert report.errors == []
    assert report.no_model_training
    assert report.no_final_labels
    assert len(report.figures) == 7
    for figure in report.figures.values():
        assert Path(figure).read_bytes().startswith(b"\x89PNG")
    assert (output_dir / "review_summary_template.md").exists()
    assert (output_dir / "mvp4a_review_summary_v001.json").exists()
