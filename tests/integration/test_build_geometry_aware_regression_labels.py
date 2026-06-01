from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_inputs(root_dir: Path) -> None:
    interim = root_dir / "interim"
    reports = root_dir / "reports"
    interim.mkdir(parents=True)
    reports.mkdir()
    cast_depth = np.arange(96.0, 108.5, 0.5, dtype=np.float32)
    cast_azimuth = np.arange(6, dtype=np.float32) * 60.0
    cast_zc = np.full((cast_depth.size, cast_azimuth.size), 4.0, dtype=np.float32)
    cast_zc[(cast_depth >= 100.0) & (cast_depth <= 101.0), :2] = 2.1
    cast_zc[(cast_depth >= 103.0) & (cast_depth <= 105.0), 3:5] = 2.0
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=cast_depth,
        cast_azimuth_deg=cast_azimuth,
        cast_zc=cast_zc,
    )
    relative_drop = np.zeros_like(cast_zc)
    relative_drop[(cast_depth >= 103.0) & (cast_depth <= 104.0), 3:5] = 0.5
    np.savez_compressed(
        interim / "cast_zc_baseline_v001.npz",
        cast_depth=cast_depth,
        cast_azimuth_deg=cast_azimuth,
        relative_drop=relative_drop,
        zc_ratio=np.ones_like(cast_zc),
    )
    depth = np.asarray([100.0, 103.0, 106.0], dtype=np.float32)
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.ones((depth.size, 2), dtype=np.float32),
        depth_level_xsi_feature_names=np.asarray(["energy", "attenuation"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )


def test_build_geometry_aware_regression_labels_cli_writes_outputs(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    _write_inputs(root_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {root_dir / 'interim'}",
                f"  reports: {root_dir / 'reports'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06ah_build_geometry_aware_regression_labels.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Geometry-aware regression labels errors=0" in result.stdout
    output_npz = root_dir / "interim" / "geometry_aware_regression_labels_v001.npz"
    output_json = root_dir / "reports" / "geometry_aware_regression_labels_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    with np.load(output_npz, allow_pickle=False) as data:
        assert data["weighted_channel_fraction_zc_lt_2p5"].shape == (4, 3, 13)
        assert data["receiver_mean"].shape == (4, 3)
        assert bool(data["no_final_labels"]) is True
        assert "derived_positive_at_fraction_0p10" in data.files
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["raw_zc_available"] is True
    assert report["geometry_sign_status"]["sign_convention_status"] == "human_confirmed"
    assert report["no_final_labels"] is True
