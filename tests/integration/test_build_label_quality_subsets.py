from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_sample_table(path: Path) -> None:
    depth_values = np.arange(5600.0, 5640.0, 2.0, dtype=np.float32)
    depth = np.repeat(depth_values, 4)
    side = np.tile(np.arange(4, dtype=np.int16), depth_values.size)
    label = np.zeros(depth.size, dtype=np.int8)
    severity = np.zeros(depth.size, dtype=np.int8)
    confidence = np.full(depth.size, 0.05, dtype=np.float32)
    positive = (depth < 5612.0) & np.isin(side, [0, 1, 3])
    label[positive] = 1
    severity[positive] = 2
    confidence[positive] = 0.8
    np.savez_compressed(
        path,
        sample_id=np.arange(depth.size, dtype=np.int64),
        depth=depth,
        side_index=side,
        side_azimuth_deg=side.astype(np.float32) * 90.0,
        label_presence_plus=label,
        label_severity_plus=severity,
        label_confidence_plus=confidence,
        label_presence_minus_audit=label.copy(),
        plus_minus_disagreement=np.zeros(depth.size, dtype=bool),
        orientation_confidence=np.full(depth.size, 0.9, dtype=np.float32),
        depth_match_error=np.zeros(depth.size, dtype=np.float32),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def _write_label_quality_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/mvp4b_label_quality_subsets.example.yaml").read_text())
    config["quality_policy"]["min_subset_samples_per_class"] = 4
    config["subsets"]["exclude_review_intervals"] = [
        {
            "name": "review_horizontal_severe_band_5700ft",
            "depth_min_ft": 5680.0,
            "depth_max_ft": 5720.0,
            "reason": "outside synthetic sample but required by schema",
            "apply_by_default": True,
        }
    ]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_build_label_quality_subsets_cli_writes_npz_and_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_sample_table(interim_dir / "baseline_sample_table_receiver_enhanced_v001.npz")
    label_quality_config = tmp_path / "label_quality.yaml"
    _write_label_quality_config(label_quality_config)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
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
            "scripts/06o_build_label_quality_subsets.py",
            "--paths",
            str(paths_config),
            "--label-quality-config",
            str(label_quality_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Label-quality subsets errors=0" in result.stdout
    output_npz = interim_dir / "label_quality_subsets_v001.npz"
    output_json = reports_dir / "label_quality_subsets_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["subset_version"] == "label_quality_subsets_v001"
    assert report["subset_counts"]["quality_strong_positive"]["sample_count"] > 0
    assert report["subset_counts"]["quality_clear_negative"]["sample_count"] > 0
    assert report["no_final_labels"] is True
    with np.load(output_npz, allow_pickle=False) as data:
        assert "quality_strong_positive_mask" in data.files
        assert bool(data["no_stc"]) is True
