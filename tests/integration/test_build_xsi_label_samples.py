from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_xsi_label_samples_cli_writes_index_and_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    labels_dir = root_dir / "labels"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    labels_dir.mkdir()
    reports_dir.mkdir()
    presence = np.zeros((3, 16), dtype=np.int8)
    severity = np.zeros_like(presence)
    confidence = np.full((3, 16), 0.2, dtype=np.float32)
    presence[1, 4] = 1
    severity[1, 4] = 2
    confidence[1, 4] = 0.9
    np.savez_compressed(
        labels_dir / "cast_weak_label_candidates_v001.npz",
        cast_depth=np.array([101.0, 100.0, 99.0], dtype=np.float32),
        cast_azimuth_aligned_deg=np.arange(16, dtype=np.float32) * 22.5,
        presence_plus=presence,
        severity_plus=severity,
        label_confidence_plus=confidence,
        presence_minus_ablation=np.zeros_like(presence),
        severity_minus_ablation=np.zeros_like(severity),
        label_confidence_minus_ablation=np.full_like(confidence, 0.2),
        no_final_labels=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "depth_only_v001.npz",
        xsi_depth_by_receiver=np.tile(np.array([101.0, 100.0, 99.0], dtype=np.float32), (13, 1)),
    )
    np.savez_compressed(
        interim_dir / "orientation_confidence_v001.npz",
        pose_depth=np.array([101.0, 100.0, 99.0], dtype=np.float32),
        orientation_confidence=np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {interim_dir}",
                f"  labels: {labels_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/05a_build_xsi_label_samples.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "XSI label samples errors=0" in result.stdout
    output_npz = interim_dir / "xsi_label_samples_v001.npz"
    output_json = reports_dir / "xsi_label_samples_report_v001.json"
    with np.load(output_npz) as data:
        assert data["label_presence_plus"].shape == (3, 8)
        assert data["label_presence_plus"][1, 2] == 1
        assert bool(data["valid_for_azimuthal_validation"][1, 2])
        assert bool(data["no_final_labels"].reshape(()))
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["sample_index_version"] == "xsi_label_samples_v001"
    assert report["no_final_labels"] is True
    assert (reports_dir / "xsi_label_samples_report_v001.md").exists()
