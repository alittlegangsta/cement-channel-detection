from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_gate_inputs(reports_dir: Path, *, signal: bool = True, high_conf: bool = True) -> None:
    _write_json(
        reports_dir / "xsi_label_samples_report_v001.json",
        {
            "sample_index_version": "xsi_label_samples_v001",
            "shape": {"depth": 4, "side": 8},
            "coverage": {
                "valid_for_azimuthal_validation_count": 20 if high_conf else 0,
                "valid_for_non_azimuthal_summary_count": 32,
                "high_confidence_candidate_count": 10 if high_conf else 0,
                "high_confidence_non_candidate_count": 10 if high_conf else 0,
            },
            "no_final_labels": True,
            "warnings": [],
            "errors": [],
        },
    )
    _write_json(
        reports_dir / "xsi_basic_features_report_v001.json",
        {
            "feature_version": "xsi_basic_features_v001",
            "summaries": {
                "xsi_basic_features_by_side": {
                    "shape": [4, 8, 6],
                    "finite_ratio": 1.0,
                }
            },
            "no_model_training": True,
            "no_stc": True,
            "no_apes": True,
            "warnings": [],
            "errors": [],
        },
    )
    _write_json(
        reports_dir / "xsi_cast_correlation_report_v001.json",
        {
            "correlation_version": "xsi_cast_correlation_v001",
            "gate_observations": {
                "high_confidence_subset_exists": high_conf,
                "interpretable_signal_separation": signal,
                "low_confidence_policy_respected": True,
                "no_model_training": True,
                "no_final_labels": True,
            },
            "subset_counts": {},
            "no_model_training": True,
            "no_final_labels": True,
            "warnings": [],
            "errors": [],
        },
    )
    _write_json(
        reports_dir / "mvp4a_review_v001" / "mvp4a_review_summary_v001.json",
        {
            "review_version": "mvp4a_review_v001",
            "figures": {f"figure_{index}": f"{index}.png" for index in range(7)},
            "no_model_training": True,
            "no_final_labels": True,
            "warnings": [],
            "errors": [],
        },
    )


def test_mvp4a_gate_report_cli_allows_mvp4b_when_conditions_pass(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_gate_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/05e_generate_mvp4a_gate_report.py", "--paths", str(paths_config)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4A gate decision=go" in result.stdout
    report = json.loads((reports_dir / "mvp4a_gate_report.json").read_text(encoding="utf-8"))
    assert report["decision"] == "go"
    assert report["mvp4b_allowed"] is True
    assert report["gate_conditions"]["no_model_training"] is True
    assert (reports_dir / "mvp4a_gate_report.md").exists()


def test_mvp4a_gate_report_cli_blocks_missing_signal(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_gate_inputs(reports_dir, signal=False, high_conf=True)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/05e_generate_mvp4a_gate_report.py", "--paths", str(paths_config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "decision=no_go" in result.stdout
    report = json.loads((reports_dir / "mvp4a_gate_report.json").read_text(encoding="utf-8"))
    assert report["decision"] == "no_go"
    assert report["mvp4b_allowed"] is False
    assert any("no interpretable separation" in item for item in report["blocking_issues"])
