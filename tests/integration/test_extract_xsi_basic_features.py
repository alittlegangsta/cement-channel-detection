from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_extract_xsi_basic_features_cli_from_tiny_waveform_npz(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    features_dir = root_dir / "features"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    features_dir.mkdir()
    reports_dir.mkdir()
    np.savez_compressed(
        interim_dir / "xsi_label_samples_v001.npz",
        xsi_depth=np.array([100.0, 99.0], dtype=np.float32),
        xsi_depth_index=np.array([0, 1], dtype=np.int32),
        xsi_side_azimuth_deg=np.arange(8, dtype=np.float32) * 45.0,
    )
    waveform = np.ones((2, 2, 8, 4), dtype=np.float32)
    waveform[1, :, 2, :] = 2.0
    np.savez_compressed(interim_dir / "tiny_waveform.npz", xsi_waveform=waveform)
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
            "scripts/05b_extract_xsi_basic_features.py",
            "--paths",
            str(paths_config),
            "--input-waveform-npz",
            str(interim_dir / "tiny_waveform.npz"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "XSI basic features errors=0" in result.stdout
    output_npz = features_dir / "xsi_basic_features_v001.npz"
    output_json = reports_dir / "xsi_basic_features_report_v001.json"
    with np.load(output_npz) as data:
        assert data["xsi_basic_features"].shape == (2, 2, 8, 6)
        assert data["xsi_basic_features_by_side"].shape == (2, 8, 6)
        assert data["feature_names"].astype(str).tolist()[-1] == "late_over_early_ratio"
        assert bool(data["no_model_training"].reshape(()))
        assert bool(data["no_stc"].reshape(()))
        assert bool(data["no_apes"].reshape(()))
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["feature_version"] == "xsi_basic_features_v001"
    assert report["no_model_training"] is True
    assert report["no_stc"] is True
    assert report["no_apes"] is True
    assert (reports_dir / "xsi_basic_features_report_v001.md").exists()
