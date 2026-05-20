from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat


def test_probe_mat_structs_cli_outputs_json_and_report(tmp_path: Path) -> None:
    mat_path = tmp_path / "cast_struct.mat"
    savemat(
        mat_path,
        {
            "CAST": {
                "Zc": np.arange(12, dtype=float).reshape(3, 4),
                "depth": np.array([1.0, 2.0, 3.0]),
            }
        },
    )
    manifest_path = tmp_path / "data_manifest_v001.json"
    metadata_path = tmp_path / "mat_metadata_v001.json"
    output_json = tmp_path / "mat_struct_probe_v001.json"
    output_report = tmp_path / "mat_struct_probe_report.md"
    paths_config = tmp_path / "paths.test.yaml"
    manifest_path.write_text("{}", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": str(mat_path),
                        "filename": mat_path.name,
                        "file_role": "cast",
                        "receiver_index": None,
                        "can_open": True,
                        "mat_format": "matlab_v5_or_v7",
                        "variables": [
                            {
                                "name": "CAST",
                                "shape": [1, 1],
                                "dtype_or_class": "struct",
                                "role_hint": "unknown",
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
                "data:",
                f"  manifests: {tmp_path}",
                f"  reports: {tmp_path}",
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
            "scripts/01d_probe_mat_structs.py",
            "--paths",
            str(paths_config),
            "--manifest",
            str(manifest_path),
            "--metadata-json",
            str(metadata_path),
            "--output-json",
            str(output_json),
            "--output-report-md",
            str(output_report),
            "--max-files",
            "1",
            "--max-variables-per-file",
            "1",
            "--max-field-depth",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Struct probe files=1" in result.stdout
    assert output_json.exists()
    assert output_report.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    field_paths = {
        field["field_path"] for field in payload["files"][0]["fields"]
    }
    assert "CAST.Zc" in field_paths
    assert "CAST.depth" in field_paths
