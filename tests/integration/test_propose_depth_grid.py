from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_audit(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "decision": "conditional_go",
                "depth_unit": "unknown_to_verify",
                "warnings": ["depth unit is unknown_to_verify"],
                "no_go_blockers": [],
                "common_overlap_interval": {"min": 100.0, "max": 103.0, "length": 3.0},
                "cast_depth": {"median_step": 0.25},
                "pose_depth": {"median_step": 0.5},
                "xsi_depth_by_receiver": {
                    "receiver_01": {"median_step": 1.0},
                    "receiver_02": {"median_step": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )


def test_propose_depth_grid_cli_outputs_reports_and_config(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    audit_json = reports_dir / "depth_axis_audit_report.json"
    output_md = reports_dir / "depth_grid_proposal.md"
    output_json = reports_dir / "depth_grid_proposal.json"
    output_config = tmp_path / "alignment.depth_grid.example.yaml"
    paths_config = tmp_path / "paths.yaml"
    _write_audit(audit_json)
    paths_config.write_text(
        "\n".join(["data:", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03b_propose_depth_grid.py",
            "--paths",
            str(paths_config),
            "--audit-report-json",
            str(audit_json),
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

    assert "Depth grid proposal decision=conditional_go" in result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["depth_step"] == 1.0
    assert report["sample_count"] == 4
    assert output_md.exists()
    assert "canonical_depth_grid" in output_config.read_text(encoding="utf-8")
