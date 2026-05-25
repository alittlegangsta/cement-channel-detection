from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_report(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_inputs(
    tmp_path: Path,
    *,
    audit_errors: list[str] | None = None,
    plus_minus_disagreement: float = 0.0,
) -> dict[str, Path]:
    reports_dir = tmp_path / "reports"
    review_dir = reports_dir / "label_review_v001"
    reports_dir.mkdir()
    review_dir.mkdir()
    paths = {
        "config": tmp_path / "paths.yaml",
        "input": reports_dir / "cast_label_input_summary_v001.json",
        "baseline": reports_dir / "cast_zc_baseline_report_v001.json",
        "weak": reports_dir / "cast_weak_label_candidates_report_v001.json",
        "audit": reports_dir / "cast_weak_label_audit_v001.json",
        "review": review_dir / "label_review_summary_v001.json",
        "output_md": reports_dir / "mvp3_gate_report.md",
        "output_json": reports_dir / "mvp3_gate_report.json",
    }
    paths["config"].write_text(
        "\n".join(["data:", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )
    _write_report(
        paths["input"],
        {
            "cast_label_input_version": "cast_label_input_v001",
            "arrays": {"cast_zc": {"shape": [4, 8]}},
            "errors": [],
            "warnings": [],
        },
    )
    _write_report(
        paths["baseline"],
        {
            "cast_baseline_version": "cast_zc_baseline_v001",
            "baseline_valid_ratio": 1.0,
            "arrays": {"zc_base": {"shape": [4, 8]}},
            "errors": [],
            "warnings": [],
        },
    )
    _write_report(
        paths["weak"],
        {
            "cast_weak_label_candidate_version": "cast_weak_label_candidates_v001",
            "label_version": "cast_weak_v001",
            "convention_status": "specification_preferred_plus_data_unresolved",
            "no_final_labels": True,
            "threshold": {"zc_min_limit_status": "requires_human_threshold_confirmation"},
            "coverage": {
                "plus": 0.1,
                "minus_ablation": 0.1,
                "plus_minus_disagreement": plus_minus_disagreement,
            },
            "confidence": {"plus": {}, "minus_ablation": {}},
            "errors": [],
            "warnings": [],
        },
    )
    _write_report(
        paths["audit"],
        {
            "label_audit_version": "cast_weak_label_audit_v001",
            "coverage": {"plus": 0.1, "minus_ablation": 0.1},
            "components": {"plus": {"component_count": 1}},
            "no_final_labels": True,
            "errors": audit_errors or [],
            "warnings": [],
        },
    )
    _write_report(
        paths["review"],
        {
            "label_review_version": "label_review_v001",
            "figures": {f"figure_{index}": f"{index}.png" for index in range(9)},
            "review_summary_template": "review_summary_template.md",
            "no_final_labels": True,
            "errors": [],
            "warnings": [],
        },
    )
    return paths


def test_mvp3_gate_report_conditional_go_for_threshold_confirmation(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04f_generate_mvp3_gate_report.py",
            "--paths",
            str(paths["config"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["mvp4_allowed"] is False
    assert report["recommended_parameter_set"]["status"] == "human_reviewed_candidate_v001"
    assert report["recommended_parameter_set"]["alpha"] == 0.35
    assert report["recommended_parameter_set"]["zc_min_limit"] == 2.5
    assert report["recommended_parameter_set"]["severity_thresholds"] == [0.30, 0.45, 0.60]
    assert report["recommended_parameter_set"]["final_label"] is False
    assert report["relbearing_label_policy"]["primary"] == "plus"
    assert report["relbearing_label_policy"]["primary_status"] == "human_specification_approved"
    assert report["relbearing_label_policy"]["minus_usage"] == "audit_only"
    assert report["relbearing_label_policy"]["single_sign_final_label_approved"] is False
    assert report["no_final_labels"] is True
    assert report["plus_primary_minus_ablation_preserved"] is True
    assert "MVP-4 requires separate approval" in report["mvp4_allowed_reason"]
    assert any("domain confirmation note" in item for item in report["warnings"])
    assert paths["output_md"].exists()


def test_mvp3_gate_report_no_go_on_audit_errors(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, audit_errors=["coverage extreme"])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/04f_generate_mvp3_gate_report.py",
            "--paths",
            str(paths["config"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["decision"] == "no_go"
    assert report["mvp4_allowed"] is False


def test_mvp3_gate_report_keeps_mvp4_blocked_for_non_negligible_disagreement(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, plus_minus_disagreement=0.20975005836610816)

    subprocess.run(
        [
            sys.executable,
            "scripts/04f_generate_mvp3_gate_report.py",
            "--paths",
            str(paths["config"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    assert report["decision"] == "conditional_go"
    assert report["mvp4_allowed"] is False
    assert report["plus_minus_disagreement"] == 0.20975005836610816
    assert any("plus/minus disagreement" in item for item in report["warnings"])
