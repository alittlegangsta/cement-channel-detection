from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_evaluate_xsi_cast_correlation_cli_outputs_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    features_dir = root_dir / "features"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    features_dir.mkdir()
    reports_dir.mkdir()
    presence = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int8)
    values = np.zeros((4, 2, 1), dtype=np.float32)
    values[..., 0] = np.where(presence == 1, 5.0, 1.0)
    np.savez_compressed(
        interim_dir / "xsi_label_samples_v001.npz",
        label_presence_plus=presence,
        label_presence_minus_audit=np.zeros_like(presence),
        label_severity_plus=np.where(presence == 1, 2, 0).astype(np.int8),
        label_confidence_plus=np.ones_like(presence, dtype=np.float32),
        label_confidence_minus_audit=np.ones_like(presence, dtype=np.float32),
        valid_for_azimuthal_validation=np.ones_like(presence, dtype=bool),
        valid_for_non_azimuthal_summary=np.ones_like(presence, dtype=bool),
        plus_minus_disagreement=presence == 1,
        no_final_labels=np.asarray(True),
    )
    np.savez_compressed(
        features_dir / "xsi_basic_features_v001.npz",
        xsi_basic_features_by_side=values,
        feature_names=np.array(["rms_energy"]),
        no_model_training=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
    )
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
            "scripts/05c_evaluate_xsi_cast_correlation.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "XSI-CAST correlation errors=0" in result.stdout
    report = json.loads(
        (reports_dir / "xsi_cast_correlation_report_v001.json").read_text(encoding="utf-8")
    )
    assert report["correlation_version"] == "xsi_cast_correlation_v001"
    assert report["no_model_training"] is True
    assert report["no_final_labels"] is True
    with (reports_dir / "xsi_cast_correlation_v001.csv").open(encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows
    assert rows[0]["feature"] == "rms_energy"
    assert (reports_dir / "xsi_cast_correlation_report_v001.md").exists()
