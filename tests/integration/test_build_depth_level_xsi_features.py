from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_depth_level_xsi_features_cli_writes_npz_and_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    features_dir = root_dir / "features"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    features_dir.mkdir(parents=True)
    interim_dir.mkdir()
    reports_dir.mkdir()
    features = np.ones((3, 4, 3, 2), dtype=np.float32)
    features[:, :, :, 1] = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
    np.savez_compressed(
        features_dir / "xsi_basic_features_v001.npz",
        xsi_depth=np.asarray([100.0, 101.0, 102.0], dtype=np.float32),
        feature_names=np.asarray(["rms_energy", "late_over_early_ratio"]),
        xsi_basic_features=features,
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "baseline_sample_table_receiver_enhanced_v001.npz",
        depth=np.repeat(np.asarray([100.0, 101.0, 102.0], dtype=np.float32), 3),
        label_presence_plus=np.zeros(9, dtype=np.int8),
        no_final_labels=np.asarray(True),
    )
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  features: {features_dir}",
                f"  interim: {interim_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06s_build_depth_level_xsi_features.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level XSI features errors=0" in result.stdout
    output_npz = interim_dir / "depth_level_xsi_features_v001.npz"
    output_json = reports_dir / "depth_level_xsi_features_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    with np.load(output_npz, allow_pickle=False) as data:
        assert data["depth_level_xsi_features"].shape[0] == 3
        assert bool(data["no_stc"]) is True
        assert bool(data["no_apes"]) is True
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["feature_version"] == "depth_level_xsi_features_v001"
    assert report["used_label_information_for_feature_construction"] is False
