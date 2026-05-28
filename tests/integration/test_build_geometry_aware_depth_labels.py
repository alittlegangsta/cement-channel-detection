from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_inputs(root_dir: Path) -> None:
    labels_dir = root_dir / "labels"
    interim_dir = root_dir / "interim"
    labels_dir.mkdir(parents=True)
    interim_dir.mkdir()
    presence = np.zeros((8, 6), dtype=np.int8)
    severity = np.zeros_like(presence)
    confidence = np.full((8, 6), 0.7, dtype=np.float32)
    presence[3, :] = 1
    severity[3, :] = 2
    np.savez_compressed(
        labels_dir / "cast_weak_label_candidates_v001.npz",
        cast_depth=np.arange(8, dtype=np.float32),
        presence_plus=presence,
        severity_plus=severity,
        label_confidence_plus=confidence,
        presence_minus_ablation=np.zeros_like(presence),
        relative_drop_plus=np.where(presence == 1, 0.5, 0.0).astype(np.float32),
        orientation_confidence_on_cast_depth_plus=np.ones_like(confidence),
        no_final_labels=np.asarray(True),
    )
    depth = np.asarray([3.0, 6.0], dtype=np.float32)
    np.savez_compressed(
        interim_dir / "depth_level_labels_v001.npz",
        depth=depth,
        depth_has_channel_any=np.asarray([True, False]),
        depth_candidate_fraction=np.asarray([1.0, 0.0], dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "depth_level_xsi_features_v001.npz",
        depth=depth,
        depth_level_xsi_features=np.ones((2, 1), dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )


def _write_geometry(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "xsi_geometry:",
                "  reference_receiver_index: 7",
                "  receiver_count: 13",
                "  receiver_spacing_ft: 0.5",
                "  r1_source_distance_ft: 1.0",
                "  depth_axis_sign: audit_both",
                "  sign_convention_status: requires_audit",
                "  source_offset_relative_to_r7_ft: -4.0",
                "  alignment_modes:",
                "    - r7_reference_depth",
                "    - receiver_depth_shifted",
                "    - source_receiver_midpoint",
                "    - source_receiver_interval",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_build_geometry_aware_depth_labels_cli_writes_outputs(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(root_dir)
    geometry_config = tmp_path / "xsi_geometry.yaml"
    _write_geometry(geometry_config)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  labels: {root_dir / 'labels'}",
                f"  interim: {root_dir / 'interim'}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06ad_build_geometry_aware_depth_labels.py",
            "--paths",
            str(paths_config),
            "--geometry-config",
            str(geometry_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Geometry-aware depth labels errors=0" in result.stdout
    output_npz = root_dir / "interim" / "geometry_aware_depth_labels_v001.npz"
    output_json = reports_dir / "geometry_aware_depth_labels_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    with np.load(output_npz, allow_pickle=False) as data:
        assert data["candidate_fraction"].shape == (8, 2)
        assert data["receiver_depths"].shape == (8, 2, 13)
        assert bool(data["no_final_labels"]) is True
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["label_version"] == "geometry_aware_depth_labels_v001"
    assert report["raw_zc_available"] is False
