from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_inputs(root: Path) -> None:
    interim = root / "interim"
    features = root / "features"
    depth = np.arange(12, dtype=np.float32)
    np.savez_compressed(
        interim / "mvp4x_research_snapshot_v001.npz",
        depth=depth,
        xsi_features=np.ones((12, 2), dtype=np.float32),
        xsi_feature_names=np.asarray(["xsi_a", "xsi_b"]),
        xsi_feature_group=np.asarray(["xsi_g1", "xsi_g2"]),
        receiver_p90=depth / 12.0,
        receiver_mean=depth / 12.0,
        receiver_max=depth / 12.0,
        broad_regime_id=np.asarray(["A"] * 4 + ["B"] * 4 + ["C"] * 4),
        saturation_platform_flag=np.zeros(12, dtype=bool),
        special_5680_flag=np.zeros(12, dtype=bool),
        low_orientation_confidence_flag=np.zeros(12, dtype=bool),
        any_special_flag=np.zeros(12, dtype=bool),
        research_only=np.asarray(True),
        exploratory_only=np.asarray(True),
        weak_label_target=np.asarray(True),
        no_final_labels=np.asarray(True),
        no_ground_truth_claim=np.asarray(True),
        no_production_claim=np.asarray(True),
    )
    np.savez_compressed(
        features / "mvp4x_waveform_features_v001.npz",
        depth=depth,
        waveform_depth_features=np.ones((12, 3), dtype=np.float32),
        waveform_depth_feature_names=np.asarray(["wf_a", "wf_b", "wf_c"]),
        waveform_depth_feature_group=np.asarray(["wf_g1", "wf_g1", "wf_g2"]),
        research_only=np.asarray(True),
        exploratory_only=np.asarray(True),
        weak_label_target=np.asarray(True),
        no_final_labels=np.asarray(True),
        no_ground_truth_claim=np.asarray(True),
        no_production_claim=np.asarray(True),
    )


def test_run_enhanced_feature_baselines_cli_writes_reports(tmp_path: Path) -> None:
    root = tmp_path / "data"
    interim = root / "interim"
    features = root / "features"
    reports = root / "reports"
    interim.mkdir(parents=True)
    features.mkdir()
    reports.mkdir()
    _write_inputs(root)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root}",
                f"  interim: {interim}",
                f"  features: {features}",
                f"  reports: {reports}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07d_run_mvp4x_enhanced_feature_baselines.py",
            "--paths",
            str(paths),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4X enhanced-feature baselines" in result.stdout
    output_json = reports / "mvp4x_enhanced_feature_baselines_v001.json"
    output_csv = reports / "mvp4x_enhanced_feature_baselines_v001.csv"
    review_json = reports / "mvp4x_model_review_v001" / "feature_set_summary.json"
    assert output_json.exists()
    assert output_csv.exists()
    assert review_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["research_only"] is True
    assert set(report["feature_sets"]) == {
        "existing_features_only",
        "waveform_features_only",
        "combined_features",
    }
    assert output_csv.read_text(encoding="utf-8").splitlines()[0].startswith("csv_version")
