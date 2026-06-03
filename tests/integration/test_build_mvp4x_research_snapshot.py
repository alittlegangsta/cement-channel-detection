from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_inputs(root: Path) -> None:
    interim = root / "interim"
    interim.mkdir(parents=True)
    depth = np.arange(9, dtype=np.float32)
    kernels = np.asarray(["r7_reference_point", "triangular_midpoint_weighted"])
    target = np.vstack([depth, depth + 1.0]).astype(np.float32)
    cube = np.ones((2, 9, 2), dtype=np.float32)
    np.savez_compressed(
        interim / "geometry_aware_regression_labels_v001.npz",
        depth=depth,
        geometry_kernel=kernels,
        receiver_p90=target,
        receiver_mean=target,
        receiver_max=target,
        full_360_fraction=target,
        receiver_std=target,
        depth_label_confidence=cube,
        min_zc=cube,
        p05_zc=cube,
        p10_zc=cube,
        max_relative_drop=cube,
        largest_connected_component_fraction=cube,
        max_azimuth_channel_fraction=cube,
        candidate_cell_count=cube.astype(np.int32),
        total_cell_count=cube.astype(np.int32),
    )
    np.savez_compressed(
        interim / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.column_stack([depth, depth + 2.0]).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["side_mean_energy", "receiver_std_rms"]),
        feature_group_names=np.asarray(["side_mean", "receiver_std"]),
        feature_group_counts_json=np.asarray('{"receiver_std": 1, "side_mean": 1}'),
    )
    np.savez_compressed(
        interim / "cast_label_input_v001.npz",
        cast_depth=depth,
        orientation_confidence=np.ones(9, dtype=np.float32),
        inc_deg=np.ones(9, dtype=np.float32) * 18.0,
        low_inc_mask=np.zeros(9, dtype=bool),
        orientation_uncertain=np.zeros(9, dtype=bool),
    )


def test_build_mvp4x_research_snapshot_cli_writes_outputs(tmp_path: Path) -> None:
    root = tmp_path / "data"
    reports = root / "reports"
    manifests = root / "manifests"
    reports.mkdir(parents=True)
    manifests.mkdir()
    _write_inputs(root)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root}",
                f"  interim: {root / 'interim'}",
                f"  reports: {reports}",
                f"  manifests: {manifests}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rapid = yaml.safe_load(Path("configs/mvp4x_rapid_exploratory.example.yaml").read_text())
    rapid["snapshot"]["broad_regimes"] = [
        {"id": "A", "min_depth_ft": 0.0, "max_depth_ft": 4.0},
        {"id": "B", "min_depth_ft": 4.0, "max_depth_ft": 10.0},
    ]
    rapid_path = tmp_path / "rapid.yaml"
    rapid_path.write_text(yaml.safe_dump(rapid, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07a_build_mvp4x_research_snapshot.py",
            "--paths",
            str(paths),
            "--rapid-config",
            str(rapid_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4X research snapshot samples=9" in result.stdout
    output_npz = root / "interim" / "mvp4x_research_snapshot_v001.npz"
    output_json = reports / "mvp4x_research_snapshot_report_v001.json"
    output_manifest = manifests / "mvp4x_research_snapshot_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    assert output_manifest.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["research_only"] is True
    assert report["no_ground_truth_claim"] is True
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["inputs"]["regression_labels_npz"]["sha256"]
