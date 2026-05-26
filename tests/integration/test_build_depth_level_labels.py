from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_inputs(root_dir: Path) -> None:
    labels_dir = root_dir / "labels"
    interim_dir = root_dir / "interim"
    labels_dir.mkdir(parents=True)
    interim_dir.mkdir()
    presence = np.zeros((4, 12), dtype=np.int8)
    severity = np.zeros_like(presence)
    confidence = np.full((4, 12), 0.2, dtype=np.float32)
    presence[0, [0, 1, 11]] = 1
    severity[0, [0, 1, 11]] = 2
    confidence[0, [0, 1, 11]] = 0.9
    presence[2, [4, 5, 6]] = 1
    severity[2, [4, 5, 6]] = 2
    confidence[2, [4, 5, 6]] = 0.8
    np.savez_compressed(
        labels_dir / "cast_weak_label_candidates_v001.npz",
        cast_depth=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float32),
        cast_azimuth_aligned_deg=np.arange(12, dtype=np.float32) * 30.0,
        presence_plus=presence,
        severity_plus=severity,
        label_confidence_plus=confidence,
        presence_minus_ablation=np.zeros_like(presence),
        relative_drop_plus=np.where(presence == 1, 0.5, 0.0).astype(np.float32),
        no_final_labels=np.asarray(True),
    )
    np.savez_compressed(
        interim_dir / "xsi_label_samples_v001.npz",
        xsi_depth=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float32),
        cast_depth_index=np.asarray([0, 1, 2, 3], dtype=np.int32),
        orientation_confidence_depth=np.ones(4, dtype=np.float32),
        orientation_confidence=np.ones((4, 8), dtype=np.float32),
        no_final_labels=np.asarray(True),
    )


def _write_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/depth_level_label.example.yaml").read_text())
    config["quality_policy"]["review_intervals"] = [
        {
            "name": "review_horizontal_severe_band_5700ft",
            "depth_min_ft": 5680.0,
            "depth_max_ft": 5720.0,
            "reason": "outside synthetic sample but required by schema",
            "apply_by_default": True,
        }
    ]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_build_depth_level_labels_cli_writes_npz_and_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(root_dir)
    depth_config = tmp_path / "depth_level_label.yaml"
    _write_config(depth_config)
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
            "scripts/06r_build_depth_level_labels.py",
            "--paths",
            str(paths_config),
            "--depth-level-config",
            str(depth_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Depth-level labels errors=0" in result.stdout
    output_npz = root_dir / "interim" / "depth_level_labels_v001.npz"
    output_json = reports_dir / "depth_level_labels_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    with np.load(output_npz, allow_pickle=False) as data:
        assert data["depth_has_channel_any"].shape == (4,)
        assert int(np.count_nonzero(data["depth_strong_positive_mask"])) == 2
        assert int(np.count_nonzero(data["depth_clear_negative_mask"])) == 2
        assert bool(data["no_final_labels"]) is True
        assert bool(data["no_stc"]) is True
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["label_version"] == "depth_level_labels_v001"
    assert report["positive_fraction"] == 0.5
    assert report["no_final_labels"] is True
