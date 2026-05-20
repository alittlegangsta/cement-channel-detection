from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _field(field_path: str, shape: list[int], role_hint: str) -> dict:
    return {
        "top_variable": field_path.split(".")[0],
        "field_path": field_path,
        "shape": shape,
        "dtype_or_class": "float32",
        "role_hint": role_hint,
        "element_count": 1,
        "preview_stats": {},
    }


def _write_probe_json(path: Path) -> None:
    fields = [
        _field("XSILMR01.Depth", [1, 4], "depth_candidate"),
        _field("XSILMR01.Tad", [1, 1], "xsi_time_candidate"),
    ]
    fields.extend(
        _field(f"XSILMR01.WaveRng01Side{side}", [1024, 4], "xsi_waveform_candidate")
        for side in "ABCDEFGH"
    )
    path.write_text(
        json.dumps(
            {
                "probe_version": "mat_struct_probe_v001",
                "files": [
                    {
                        "path": "/tmp/CAST.mat",
                        "filename": "CAST.mat",
                        "file_role": "cast",
                        "receiver_index": None,
                        "can_probe": True,
                        "mat_format": "matlab_v5_or_v7",
                        "probed_variables": ["CAST"],
                        "fields": [
                            _field("CAST.Zc", [180, 4], "cast_zc_candidate"),
                            _field("CAST.Depth", [1, 4], "depth_candidate"),
                        ],
                        "warnings": [],
                        "errors": [],
                    },
                    {
                        "path": "/tmp/D2_XSI_RelBearing_Inclination.mat",
                        "filename": "D2_XSI_RelBearing_Inclination.mat",
                        "file_role": "pose",
                        "receiver_index": None,
                        "can_probe": True,
                        "mat_format": "matlab_v5_or_v7",
                        "probed_variables": ["Depth_inc", "Inc", "RelBearing"],
                        "fields": [
                            _field("Depth_inc", [1, 4], "depth_candidate"),
                            _field("Inc", [1, 4], "inclination_candidate"),
                            _field("RelBearing", [1, 4], "relbearing_candidate"),
                        ],
                        "warnings": [],
                        "errors": [],
                    },
                    {
                        "path": "/tmp/XSILMR01.mat",
                        "filename": "XSILMR01.mat",
                        "file_role": "xsi_receiver",
                        "receiver_index": 1,
                        "can_probe": True,
                        "mat_format": "matlab_v5_or_v7",
                        "probed_variables": ["XSILMR01"],
                        "fields": fields,
                        "warnings": [],
                        "errors": [],
                    },
                ],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_suggest_raw_mapping_cli_outputs_suggestions(tmp_path: Path) -> None:
    probe_json = tmp_path / "mat_struct_probe_v001.json"
    report_md = tmp_path / "raw_variable_mapping_suggestions.md"
    report_json = tmp_path / "raw_variable_mapping_suggestions.json"
    draft_yaml = tmp_path / "raw_variable_mapping.draft.yaml"
    paths_config = tmp_path / "paths.test.yaml"
    _write_probe_json(probe_json)
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  manifests: {tmp_path}",
                f"  reports: {tmp_path}",
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
            "scripts/01e_suggest_raw_mapping.py",
            "--paths",
            str(paths_config),
            "--struct-probe-json",
            str(probe_json),
            "--output-report-md",
            str(report_md),
            "--output-report-json",
            str(report_json),
            "--output-draft-yaml",
            str(draft_yaml),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Raw mapping suggestions" in result.stdout
    assert report_md.exists()
    assert report_json.exists()
    assert draft_yaml.exists()
    suggestions = json.loads(report_json.read_text(encoding="utf-8"))
    assert suggestions["recommendations"]["cast.zc_variable"]["variable_path"] == "CAST.Zc"
    assert (
        suggestions["recommendations"]["xsi.waveform_variable"]["variable_path"]
        == "XSILMR01.WaveRng01Side{A-H}"
    )
    draft_text = draft_yaml.read_text(encoding="utf-8")
    assert "status: draft_requires_human_review" in draft_text
    assert "zc_variable: CAST.Zc" in draft_text
    assert "waveform_variable: XSILMR01.WaveRng01Side{A-H}" in draft_text
