from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cement_channel.data.manifest import build_manifest


def test_qc_skeleton_script_writes_reports(tmp_path: Path) -> None:
    raw_dir = Path("tests/fixtures/tiny_sample/raw").resolve()
    manifest_path = tmp_path / "data_manifest_v001.json"
    output_dir = tmp_path / "qc"
    config_path = tmp_path / "paths.test.yaml"
    manifest = build_manifest(
        {
            "schema_version": "schema_v001",
            "data": {"raw": str(raw_dir), "manifests": str(tmp_path)},
            "raw_layout": {
                "organization": "single_well_flat",
                "well_id": "D2",
                "cast_files": ["CAST.fake_mat"],
                "pose_files": ["D2_XSI_RelBearing_Inclination.fake_mat"],
                "xsi_receiver_dir": "XSILMR",
                "xsi_receiver_file_patterns": ["XSILMR*.fake_mat"],
                "expected_xsi_receiver_files": 13,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "schema_version: schema_v001",
                "data:",
                f"  manifests: {tmp_path}",
                f"  reports: {tmp_path / 'reports'}",
                "outputs:",
                f"  data_manifest_json: {manifest_path}",
                f"  qc_report_dir: {output_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/02_run_qc.py",
            "--paths",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Manifest schema validation passed" in result.stdout
    report_json = output_dir / "qc_skeleton_report.json"
    report_md = output_dir / "qc_skeleton_report.md"
    assert report_json.exists()
    assert report_md.exists()

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["validation"]["is_valid"] is True
    assert report["validation"]["errors"] == []
    assert "large .mat content reads" in report["not_performed"]
