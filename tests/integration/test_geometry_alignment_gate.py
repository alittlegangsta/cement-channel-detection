from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_inputs(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True)
    reports_dir.joinpath("geometry_alignment_audit_v001.json").write_text(
        json.dumps(
            {
                "recommendation": "conditional_go",
                "errors": [],
                "best_combo": {
                    "mode": "source_receiver_midpoint",
                    "sign": -1,
                    "real_minus_permutation_margin": 0.08,
                    "predicted_positive_rate": 0.5,
                    "depends_on_5700_band": False,
                    "suspicious_leakage": False,
                },
                "r7_reference_summary": {
                    "mode": "r7_reference_depth",
                    "sign": 1,
                    "real_minus_permutation_margin": 0.06,
                },
                "source_receiver_interval_summary": {
                    "mode": "source_receiver_interval",
                    "sign": 1,
                    "sample_count_collapse": True,
                    "real_minus_permutation_margin": None,
                },
                "combo_summaries": [
                    {"mode": "r7_reference_depth", "sign": 1, "sample_count_collapse": False},
                    {
                        "mode": "source_receiver_interval",
                        "sign": 1,
                        "sample_count_collapse": True,
                    },
                ],
                "sign_sensitivity": [
                    {"mode": "source_receiver_midpoint", "sign_sensitive": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    reports_dir.joinpath("geometry_aware_depth_labels_report_v001.json").write_text(
        json.dumps({"raw_zc_available": False}),
        encoding="utf-8",
    )
    review_dir = reports_dir / "geometry_aware_manual_review_v001"
    review_dir.mkdir()
    review_dir.joinpath("review_summary.md").write_text(
        "- evidence_category_changed_interval_count: 3\n- revisit_interval_count: 2\n",
        encoding="utf-8",
    )
    review_dir.joinpath("geometry_alignment_review_summary_v001.json").write_text(
        json.dumps(
            {
                "evidence_category_changed_interval_count": 3,
                "revisit_interval_count": 2,
            }
        ),
        encoding="utf-8",
    )
    reports_dir.joinpath("depth_level_refinement_gate_report.json").write_text(
        json.dumps({"decision": "go"}),
        encoding="utf-8",
    )


def test_geometry_alignment_gate_cli_writes_conditional_go(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    _write_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06ag_generate_geometry_alignment_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    output_json = reports_dir / "geometry_alignment_gate_report.json"
    assert output_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["mvp4c_allowed"] is False
    assert report["final_labels_allowed"] is False
    assert report["depth_axis_sign_confirmation_required"] is True
    assert report["manual_review_revisit_required"] is True
