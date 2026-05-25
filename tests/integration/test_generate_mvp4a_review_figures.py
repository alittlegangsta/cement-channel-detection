from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def test_generate_mvp4a_review_figures_cli_outputs_review_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    features_dir = root_dir / "features"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    features_dir.mkdir()
    reports_dir.mkdir()
    presence = np.zeros((6, 8), dtype=np.int8)
    presence[2:4, 2:4] = 1
    np.savez_compressed(
        interim_dir / "xsi_label_samples_v001.npz",
        xsi_depth=np.arange(6, dtype=np.float32),
        xsi_side_azimuth_deg=np.arange(8, dtype=np.float32) * 45.0,
        label_presence_plus=presence,
        label_severity_plus=np.where(presence == 1, 2, 0).astype(np.int8),
        label_confidence_plus=np.where(presence == 1, 0.8, 0.1).astype(np.float32),
        valid_for_azimuthal_validation=np.ones_like(presence, dtype=bool),
        plus_minus_disagreement=np.roll(presence, 1, axis=1) != presence,
        no_final_labels=np.asarray(True),
    )
    values = np.ones((6, 8, 1), dtype=np.float32)
    values[..., 0] += presence * 2.0
    np.savez_compressed(
        features_dir / "xsi_basic_features_v001.npz",
        xsi_basic_features_by_side=values,
        feature_names=np.array(["rms_energy"]),
        no_model_training=np.asarray(True),
    )
    (reports_dir / "xsi_cast_correlation_v001.csv").write_text(
        "\n".join(
            [
                "label_convention,subset,feature,point_biserial_effect_size,weighted_difference_fraction",
                "plus_primary,high_confidence,rms_energy,0.5,0.2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {interim_dir}",
                f"  features: {features_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/05d_generate_mvp4a_review_figures.py",
            "--paths",
            str(paths_config),
            "--max-depth-pixels",
            "10",
            "--max-distribution-samples",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4A review figures errors=0" in result.stdout
    review_dir = reports_dir / "mvp4a_review_v001"
    summary = json.loads((review_dir / "mvp4a_review_summary_v001.json").read_text())
    assert summary["no_model_training"] is True
    assert summary["no_final_labels"] is True
    assert len(summary["figures"]) == 7
    assert (review_dir / "01_label_coverage_vs_depth.png").read_bytes().startswith(b"\x89PNG")
    assert (review_dir / "review_summary_template.md").exists()
