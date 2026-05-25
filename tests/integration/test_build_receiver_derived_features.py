from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_basic_features(path: Path) -> None:
    depth = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    receiver = np.arange(13, dtype=np.float32)
    features = np.zeros((3, 13, 8, 6), dtype=np.float32)
    for depth_index in range(3):
        for side in range(8):
            base = 1.0 + depth_index + side * 0.1
            features[depth_index, :, side, :] = (
                base + receiver[:, None] * np.linspace(0.01, 0.06, 6)[None, :]
            )
    np.savez_compressed(
        path,
        xsi_depth=depth,
        receiver_index=np.arange(1, 14, dtype=np.int16),
        side_labels=np.array(list("ABCDEFGH")),
        feature_names=np.array(
            [
                "rms_energy",
                "peak_abs",
                "mean_abs",
                "early_energy",
                "late_energy",
                "late_over_early_ratio",
            ]
        ),
        xsi_basic_features=features,
        no_model_training=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def _write_sample_table(path: Path) -> None:
    depth = np.repeat(np.array([100.0, 101.0, 102.0], dtype=np.float32), 8)
    side = np.tile(np.arange(8, dtype=np.int16), 3)
    labels = (side == 1).astype(np.int8)
    np.savez_compressed(
        path,
        sample_id=np.arange(depth.size, dtype=np.int64),
        depth=depth,
        side_index=side,
        label_presence_plus=labels,
        label_presence_minus_audit=labels.copy(),
        label_confidence_plus=np.ones(depth.size, dtype=np.float32),
        valid_for_azimuthal_validation=np.ones(depth.size, dtype=bool),
        plus_minus_disagreement=np.zeros(depth.size, dtype=bool),
        exclude_large_depth_match_error=np.zeros(depth.size, dtype=bool),
        sample_weight=np.ones(depth.size, dtype=np.float32),
        sample_weight_capped_class_balanced_confidence=np.ones(depth.size, dtype=np.float32),
        transformed_features=np.ones((depth.size, 2), dtype=np.float32),
        transformed_feature_names=np.array(["a", "b"]),
        no_model_training=np.asarray(True),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )


def test_build_receiver_derived_features_cli_writes_table_and_report(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    features_dir = root_dir / "features"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    features_dir.mkdir(parents=True)
    interim_dir.mkdir()
    reports_dir.mkdir()
    _write_basic_features(features_dir / "xsi_basic_features_v001.npz")
    _write_sample_table(interim_dir / "baseline_sample_table_enhanced_v001.npz")
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
            "scripts/06l_build_receiver_derived_features.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Receiver-derived features errors=0" in result.stdout
    output_npz = interim_dir / "baseline_sample_table_receiver_enhanced_v001.npz"
    output_json = reports_dir / "receiver_derived_feature_report_v001.json"
    assert output_npz.exists()
    assert output_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["raw_receiver_feature_count"] == 90
    assert report["finite_ratio"]["transformed_receiver_features"] == 1.0
    assert report["used_label_information_for_feature_construction"] is False
    with np.load(output_npz, allow_pickle=False) as data:
        assert data["receiver_features_added"].shape == (24, 90)
        assert data["transformed_features"].shape[1] > 2
        assert bool(data["no_final_labels"])
