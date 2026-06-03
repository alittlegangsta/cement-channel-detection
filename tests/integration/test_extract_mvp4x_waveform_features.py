from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io
import yaml


def _write_receiver_mat(raw_dir: Path) -> None:
    receiver_dir = raw_dir / "XSILMR"
    receiver_dir.mkdir(parents=True)
    depth = np.arange(6, dtype=np.float64)
    time_depth_a = np.tile(np.arange(8, dtype=np.int32)[:, None], (1, depth.size))
    time_depth_b = time_depth_a * 2
    scipy.io.savemat(
        receiver_dir / "XSILMR01.mat",
        {
            "XSILMR01": {
                "Depth": depth.reshape(1, -1),
                "Tad": np.asarray([[1.0]], dtype=np.float32),
                "WaveRng01SideA": time_depth_a,
                "WaveRng01SideB": time_depth_b,
            }
        },
    )


def _write_snapshot(interim: Path) -> None:
    depth = np.arange(6, dtype=np.float32)
    np.savez_compressed(
        interim / "mvp4x_research_snapshot_v001.npz",
        depth=depth,
        research_only=np.asarray(True),
        exploratory_only=np.asarray(True),
        weak_label_target=np.asarray(True),
        no_final_labels=np.asarray(True),
        no_ground_truth_claim=np.asarray(True),
        no_production_claim=np.asarray(True),
    )


def _write_mapping(path: Path) -> None:
    mapping = {
        "xsi": {
            "receiver_dir": "XSILMR",
            "expected_receiver_files": 1,
            "expected_side_count": 2,
            "expected_time_sample_count": 8,
            "depth_variable_pattern": "XSILMR{receiver:02d}.Depth",
            "time_variable_pattern": "XSILMR{receiver:02d}.Tad",
            "waveform_variable_pattern": "XSILMR{receiver:02d}.WaveRng{receiver:02d}Side{side}",
            "side_labels": ["A", "B"],
            "depth_source_shape_order": ["depth"],
            "waveform_source_shape_order": ["time", "depth"],
            "waveform_canonical_shape_order": ["depth", "time"],
        }
    }
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")


def test_extract_mvp4x_waveform_features_cli_writes_outputs(tmp_path: Path) -> None:
    root = tmp_path / "data"
    raw = root / "raw"
    interim = root / "interim"
    features = root / "features"
    reports = root / "reports"
    raw.mkdir(parents=True)
    interim.mkdir()
    features.mkdir()
    reports.mkdir()
    _write_receiver_mat(raw)
    _write_snapshot(interim)
    mapping = tmp_path / "mapping.yaml"
    _write_mapping(mapping)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root}",
                f"  raw: {raw}",
                f"  interim: {interim}",
                f"  features: {features}",
                f"  reports: {reports}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rapid = yaml.safe_load(Path("configs/mvp4x_rapid_exploratory.example.yaml").read_text())
    rapid["waveform_features"]["receiver_count"] = 1
    rapid["waveform_features"]["side_labels"] = ["A", "B"]
    rapid["waveform_features"]["max_time_samples"] = 8
    rapid["waveform_features"]["chunk_memory_cap_mb"] = 1
    rapid_path = tmp_path / "rapid.yaml"
    rapid_path.write_text(yaml.safe_dump(rapid, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07c_extract_mvp4x_waveform_features.py",
            "--paths",
            str(paths),
            "--mapping",
            str(mapping),
            "--rapid-config",
            str(rapid_path),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4X waveform features samples=6" in result.stdout
    output_npz = features / "mvp4x_waveform_features_v001.npz"
    output_json = reports / "mvp4x_waveform_feature_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["research_only"] is True
    assert report["no_full_waveform_loaded"] is True
    assert report["chunk_count"] == 1
    with np.load(output_npz, allow_pickle=False) as data:
        assert data["waveform_receiver_side_features"].shape[:3] == (6, 1, 2)
