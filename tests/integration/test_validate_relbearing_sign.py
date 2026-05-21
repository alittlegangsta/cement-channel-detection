from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_validate_relbearing_sign_cli_outputs_report_and_config(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    interim_dir.mkdir()
    reports_dir.mkdir()
    preview_npz = interim_dir / "depth_resample_preview_v001.npz"
    small_summary = interim_dir / "small_slice_summary_v001.json"
    resample_report = reports_dir / "depth_resample_preview_report.json"
    output_md = reports_dir / "relbearing_sign_validation_report.md"
    output_json = reports_dir / "relbearing_sign_validation_report.json"
    output_config = tmp_path / "alignment.relbearing.example.yaml"
    paths_config = tmp_path / "paths.yaml"
    np.savez_compressed(
        preview_npz,
        canonical_depth=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        inc_deg_on_grid=np.array([1.0, 3.0, 6.0], dtype=np.float32),
        relbearing_deg_on_grid=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        small_slice_cast_zc_on_preview=np.empty((0, 0), dtype=np.float32),
    )
    small_summary.write_text(json.dumps({"warnings": ["unit unknown"]}), encoding="utf-8")
    resample_report.write_text(
        json.dumps(
            {
                "small_slice": {"status": "skipped_no_common_overlap"},
                "warnings": ["no common overlap"],
                "errors": [],
            }
        ),
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
            "--depth-resample-preview-npz",
            str(preview_npz),
            "--small-slice-summary-json",
            str(small_summary),
            "--depth-resample-report-json",
            str(resample_report),
            "--output-report-md",
            str(output_md),
            "--output-report-json",
            str(output_json),
            "--output-config",
            str(output_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=insufficient_evidence" in result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["selected_convention"] is None
    assert report["manual_confirmation_required"] is True
    assert output_md.exists()
    assert "selected_convention: unconfirmed" in output_config.read_text(encoding="utf-8")
