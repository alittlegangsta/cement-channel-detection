from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_relbearing_validation_overlap_report_can_be_generated(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    interim_dir.mkdir()
    reports_dir.mkdir()
    preview_npz = interim_dir / "depth_resample_overlap_preview_v001.npz"
    small_summary = interim_dir / "small_slice_overlap_summary_v001.json"
    resample_report = reports_dir / "depth_resample_overlap_preview_report.json"
    output_json = reports_dir / "relbearing_sign_validation_overlap_report.json"
    output_md = reports_dir / "relbearing_sign_validation_overlap_report.md"
    output_config = tmp_path / "alignment.relbearing.example.yaml"
    paths_config = tmp_path / "paths.yaml"
    np.savez_compressed(
        preview_npz,
        canonical_depth=np.array([104.0, 105.0, 106.0], dtype=np.float32),
        inc_deg_on_grid=np.array([2.0, 3.0, 6.0], dtype=np.float32),
        relbearing_deg_on_grid=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        small_slice_cast_zc_on_preview=np.array(
            [[3.0, 2.0, 3.0, 4.0], [3.0, 1.5, 3.0, 4.0], [3.0, 2.5, 3.0, 4.0]],
            dtype=np.float32,
        ),
        small_slice_xsi_waveform_on_preview=np.ones((3, 2, 4, 5), dtype=np.float32),
    )
    small_summary.write_text(json.dumps({"warnings": []}), encoding="utf-8")
    resample_report.write_text(
        json.dumps({"small_slice": {"status": "completed"}, "warnings": [], "errors": []}),
        encoding="utf-8",
    )
    paths_config.write_text(
        "\n".join(["data:", f"  interim: {interim_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03f_validate_relbearing_sign.py",
            "--paths",
            str(paths_config),
            "--overlap-targeted",
            "--output-config",
            str(output_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=insufficient_evidence" in result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["candidate_metrics"]["plus"]["evidence_available"] is True
    assert report["decision"] == "insufficient_evidence"
    assert output_md.exists()
