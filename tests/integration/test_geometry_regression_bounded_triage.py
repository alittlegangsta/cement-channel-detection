from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.integration.test_geometry_regression_contract_qa import _write_inputs


def test_geometry_regression_bounded_triage_cli_writes_outputs(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    _write_inputs(root_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {root_dir}",
                f"  interim: {root_dir / 'interim'}",
                f"  reports: {root_dir / 'reports'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/06ai_audit_geometry_regression.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/06al_generate_geometry_regression_contract_qa.py",
            "--paths",
            str(paths_config),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    audit_json = root_dir / "reports" / "geometry_regression_audit_v001.json"
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    audit["recommendation"] = "stop"
    audit_json.write_text(json.dumps(audit), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06am_generate_geometry_regression_bounded_triage.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Geometry regression bounded triage" in result.stdout
    output_json = root_dir / "reports" / "geometry_regression_bounded_triage_v001.json"
    output_csv = root_dir / "reports" / "geometry_regression_bounded_triage_v001.csv"
    assert output_json.exists()
    assert output_csv.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["no_final_labels"] is True
    assert report["eligibility"]["contract_invariants_passed"] is True
