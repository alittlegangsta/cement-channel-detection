from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_manifest_script_writes_inventory_and_json(tmp_path: Path) -> None:
    raw_dir = Path("tests/fixtures/tiny_sample/raw").resolve()
    output_csv = tmp_path / "raw_file_inventory.csv"
    output_json = tmp_path / "data_manifest_v001.json"
    config_output_csv = tmp_path / "config_raw_file_inventory.csv"
    config_output_json = tmp_path / "config_data_manifest_v001.json"
    config_path = tmp_path / "paths.test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: schema_v001",
                "data:",
                f"  raw: {raw_dir}",
                f"  manifests: {tmp_path}",
                "raw_layout:",
                "  organization: single_well_flat",
                "  well_id: D2",
                "  cast_files:",
                "    - CAST.fake_mat",
                "  pose_files:",
                "    - D2_XSI_RelBearing_Inclination.fake_mat",
                "  xsi_receiver_dir: XSILMR",
                "  xsi_receiver_file_patterns:",
                '    - "XSILMR*.fake_mat"',
                "  expected_xsi_receiver_files: 13",
                "outputs:",
                f"  raw_inventory_csv: {config_output_csv}",
                f"  data_manifest_json: {config_output_json}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/01_build_manifest.py",
            "--paths",
            str(config_path),
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Scanned 15 raw files" in result.stdout
    assert output_csv.exists()
    assert output_json.exists()
    assert not config_output_csv.exists()
    assert not config_output_json.exists()

    with output_csv.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 15
    assert rows[0].keys() >= {
        "well_id",
        "file_role",
        "receiver_index",
        "path",
        "filename",
        "extension",
        "size_bytes",
        "modified_time",
        "matched_pattern",
    }

    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["summary"]["files_by_role"]["xsi_receiver"] == 13
    assert manifest["warnings"] == []
