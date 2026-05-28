from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_inputs(interim_dir: Path, reports_dir: Path) -> None:
    n = 30
    positive = np.zeros((2, n), dtype=bool)
    negative = np.zeros((2, n), dtype=bool)
    positive[:, :10] = True
    negative[:, 10:25] = True
    np.savez_compressed(
        interim_dir / "geometry_aware_depth_labels_v001.npz",
        depth=np.arange(n, dtype=np.float32),
        mode=np.asarray(["r7_reference_depth", "source_receiver_interval"]),
        sign=np.asarray([1, 1], dtype=np.int8),
        high_confidence_positive_mask=positive,
        clear_negative_mask=negative,
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    signal = np.zeros(n, dtype=np.float32)
    signal[:10] = 2.0
    signal[10:25] = -1.0
    np.savez_compressed(
        interim_dir / "depth_level_xsi_features_v001.npz",
        depth=np.arange(n, dtype=np.float32),
        depth_level_xsi_features=np.column_stack([signal]).astype(np.float32),
        depth_level_xsi_feature_names=np.asarray(["receiver_mean_peak_abs"]),
        no_final_labels=np.asarray(True),
        no_stc=np.asarray(True),
        no_apes=np.asarray(True),
        no_deep_learning=np.asarray(True),
        no_mvp4c=np.asarray(True),
    )
    reports_dir.joinpath("depth_level_refinement_report_v001.json").write_text(
        json.dumps({"no_final_labels": True}),
        encoding="utf-8",
    )


def test_audit_geometry_alignment_cli_writes_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    interim_dir = root_dir / "interim"
    reports_dir = root_dir / "reports"
    interim_dir.mkdir(parents=True)
    reports_dir.mkdir()
    _write_inputs(interim_dir, reports_dir)
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
            "scripts/06ae_audit_geometry_alignment.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Geometry alignment audit errors=0" in result.stdout
    output_json = reports_dir / "geometry_alignment_audit_v001.json"
    output_csv = reports_dir / "geometry_alignment_audit_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_version"] == "geometry_alignment_audit_v001"
    assert report["no_final_labels"] is True
