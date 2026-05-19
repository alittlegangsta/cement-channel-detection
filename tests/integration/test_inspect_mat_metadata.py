from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_inspect_mat_metadata_cli_outputs_json(tmp_path: Path) -> None:
    mat_path = tmp_path / "synthetic.mat"
    mat_path.write_bytes(b"synthetic placeholder; not a real MAT file")
    manifest_path = tmp_path / "data_manifest_v001.json"
    output_json = tmp_path / "mat_metadata_v001.json"
    paths_config = tmp_path / "paths.test.yaml"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "data_manifest_v001",
                "schema_version": "schema_v001",
                "data_version": "data_v001",
                "created_at": "2026-05-19T00:00:00+00:00",
                "files": [
                    {
                        "path": str(mat_path),
                        "filename": mat_path.name,
                        "file_role": "cast",
                        "receiver_index": None,
                    }
                ],
                "wells": [
                    {
                        "well_id": "D2",
                        "files": [
                            {
                                "path": str(mat_path),
                                "filename": mat_path.name,
                                "file_role": "cast",
                                "receiver_index": None,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    paths_config.write_text(
        "\n".join(
            [
                "schema_version: schema_v001",
                "data:",
                f"  manifests: {tmp_path}",
                "outputs:",
                f"  data_manifest_json: {manifest_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/01b_inspect_mat_metadata.py",
            "--paths",
            str(paths_config),
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Inspected 1 MAT file(s)" in result.stdout
    assert output_json.exists()
    metadata = json.loads(output_json.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == "mat_metadata_v001"
    assert metadata["summary"]["file_count"] == 1
    assert metadata["files"][0]["filename"] == "synthetic.mat"
    assert metadata["files"][0]["variables"] == []
    assert metadata["files"][0]["errors"] or metadata["files"][0]["warnings"]
