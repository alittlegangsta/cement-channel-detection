from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _synthetic_metadata(path: Path) -> None:
    files = [
        {
            "path": "/tmp/CAST.mat",
            "filename": "CAST.mat",
            "file_role": "cast",
            "receiver_index": None,
            "can_open": True,
            "mat_format": "matlab_v5_or_v7",
            "variables": [
                {
                    "name": "Zc",
                    "shape": [4, 180],
                    "dtype_or_class": "double",
                    "is_numeric": True,
                    "element_count": 720,
                    "role_hint": "cast_zc_candidate",
                },
                {
                    "name": "depth",
                    "shape": [4, 1],
                    "dtype_or_class": "double",
                    "is_numeric": True,
                    "element_count": 4,
                    "role_hint": "depth_candidate",
                },
            ],
            "warnings": [],
            "errors": [],
        },
        {
            "path": "/tmp/D2_XSI_RelBearing_Inclination.mat",
            "filename": "D2_XSI_RelBearing_Inclination.mat",
            "file_role": "pose",
            "receiver_index": None,
            "can_open": True,
            "mat_format": "matlab_v5_or_v7",
            "variables": [
                {
                    "name": "depth",
                    "shape": [4, 1],
                    "dtype_or_class": "double",
                    "is_numeric": True,
                    "element_count": 4,
                    "role_hint": "depth_candidate",
                },
                {
                    "name": "Inc",
                    "shape": [4, 1],
                    "dtype_or_class": "double",
                    "is_numeric": True,
                    "element_count": 4,
                    "role_hint": "inclination_candidate",
                },
                {
                    "name": "RelBearing",
                    "shape": [4, 1],
                    "dtype_or_class": "double",
                    "is_numeric": True,
                    "element_count": 4,
                    "role_hint": "relbearing_candidate",
                },
            ],
            "warnings": [],
            "errors": [],
        },
    ]
    for receiver_index in range(1, 14):
        files.append(
            {
                "path": f"/tmp/XSILMR{receiver_index:02d}.mat",
                "filename": f"XSILMR{receiver_index:02d}.mat",
                "file_role": "xsi_receiver",
                "receiver_index": receiver_index,
                "can_open": True,
                "mat_format": "matlab_v5_or_v7",
                "variables": [
                    {
                        "name": "waveform",
                        "shape": [4, 1024],
                        "dtype_or_class": "double",
                        "is_numeric": True,
                        "element_count": 4096,
                        "role_hint": "xsi_waveform_candidate",
                    }
                ],
                "warnings": [],
                "errors": [],
            }
        )
    path.write_text(
        json.dumps(
            {
                "metadata_version": "mat_metadata_v001",
                "schema_version": "schema_v001",
                "data_version": "data_v001",
                "files": files,
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_audit_raw_metadata_cli_writes_reports_and_template(tmp_path: Path) -> None:
    metadata_json = tmp_path / "mat_metadata_v001.json"
    report_md = tmp_path / "raw_metadata_report.md"
    report_json = tmp_path / "raw_metadata_report.json"
    mapping_template = tmp_path / "raw_variable_mapping.example.yaml"
    paths_config = tmp_path / "paths.test.yaml"
    _synthetic_metadata(metadata_json)
    paths_config.write_text(
        "\n".join(
            [
                "schema_version: schema_v001",
                "data:",
                f"  manifests: {tmp_path}",
                f"  reports: {tmp_path / 'reports'}",
                "raw_layout:",
                "  well_id: D2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/01c_audit_raw_metadata.py",
            "--paths",
            str(paths_config),
            "--metadata-json",
            str(metadata_json),
            "--output-report-md",
            str(report_md),
            "--output-report-json",
            str(report_json),
            "--output-mapping-template",
            str(mapping_template),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Raw metadata audit status=pass" in result.stdout
    assert report_md.exists()
    assert report_json.exists()
    assert mapping_template.exists()
    assert "Raw MAT Metadata Audit Report" in report_md.read_text(encoding="utf-8")
    audit = json.loads(report_json.read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["statistics"]["xsi_receiver_files"] == 13
    template_text = mapping_template.read_text(encoding="utf-8")
    assert "status: draft_requires_human_review" in template_text
    assert "zc_variable: TODO_CONFIRM" in template_text
    assert "waveform_variable: TODO_CONFIRM" in template_text
