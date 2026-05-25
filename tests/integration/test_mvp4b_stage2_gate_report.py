from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _simple_baseline_report(
    *,
    passes_permutation: bool = True,
    leakage_suspected: bool = False,
) -> dict:
    real_balanced = 0.66
    permutation_balanced = 0.54 if passes_permutation else 0.68
    return {
        "report_version": "simple_baseline_v001",
        "class_balance": {
            "candidate_count": 80,
            "non_candidate_count": 120,
            "candidate_fraction": 0.4,
        },
        "split": {
            "method": "depth_block_group_split",
            "n_splits": 3,
            "folds": [
                {
                    "train_candidate_count": 50,
                    "train_non_candidate_count": 80,
                    "validation_candidate_count": 30,
                    "validation_non_candidate_count": 40,
                },
                {
                    "train_candidate_count": 55,
                    "train_non_candidate_count": 75,
                    "validation_candidate_count": 25,
                    "validation_non_candidate_count": 45,
                },
                {
                    "train_candidate_count": 55,
                    "train_non_candidate_count": 85,
                    "validation_candidate_count": 25,
                    "validation_non_candidate_count": 35,
                },
            ],
        },
        "aggregate_metrics": {
            "logistic_regression": {
                "balanced_accuracy": real_balanced,
                "f1": 0.62,
                "weighted_accuracy": 0.64,
            }
        },
        "permutation_check": {
            "logistic_regression": {
                "real_balanced_accuracy": real_balanced,
                "permutation_balanced_accuracy": permutation_balanced,
                "balanced_accuracy_margin": real_balanced - permutation_balanced,
                "required_margin": 0.02,
                "passes_margin": passes_permutation,
            }
        },
        "minus_audit_comparison": {"logistic_regression": {"balanced_accuracy": 0.60}},
        "coefficient_summary": {
            "logistic_regression:late_energy": {"mean_coefficient": 0.2}
        },
        "production_training": False,
        "no_final_labels": True,
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "no_production_model": True,
        "leakage_suspected": leakage_suspected,
        "warnings": [],
        "errors": [],
    }


def _review_summary() -> dict:
    return {
        "review_version": "simple_baseline_review_v001",
        "figures": {f"figure_{index}": f"{index}.png" for index in range(7)},
        "review_summary_template": "review_summary_template.md",
        "no_final_labels": True,
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "no_production_model": True,
        "warnings": [],
        "errors": [],
    }


def _stage1_gate() -> dict:
    return {
        "decision": "go",
        "mvp4b_stage2_allowed": True,
        "warnings": [],
        "errors": [],
    }


def _write_gate_inputs(
    reports_dir: Path,
    *,
    passes_permutation: bool = True,
    leakage_suspected: bool = False,
) -> None:
    _write_json(
        reports_dir / "simple_baseline_report_v001.json",
        _simple_baseline_report(
            passes_permutation=passes_permutation,
            leakage_suspected=leakage_suspected,
        ),
    )
    _write_json(
        reports_dir
        / "simple_baseline_review_v001"
        / "simple_baseline_review_summary_v001.json",
        _review_summary(),
    )
    _write_json(reports_dir / "mvp4b_stage1_gate_report.json", _stage1_gate())


def test_mvp4b_stage2_gate_report_cli_allows_mvp4c_when_conditions_pass(
    tmp_path: Path,
) -> None:
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
        [
            sys.executable,
            "scripts/06f_generate_mvp4b_stage2_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4B Stage 2 gate decision=go" in result.stdout
    report = json.loads(
        (reports_dir / "mvp4b_stage2_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "go"
    assert report["mvp4c_allowed"] is True
    assert (reports_dir / "mvp4b_stage2_gate_report.md").exists()


def test_mvp4b_stage2_gate_report_cli_blocks_failed_permutation_check(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_gate_inputs(reports_dir, passes_permutation=False)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06f_generate_mvp4b_stage2_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "mvp4b_stage2_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert any("permutation" in issue for issue in report["blocking_issues"])


def test_mvp4b_stage2_gate_report_cli_blocks_leakage_suspicion(tmp_path: Path) -> None:
    root_dir = tmp_path / "data"
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True)
    _write_gate_inputs(reports_dir, leakage_suspected=True)
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06f_generate_mvp4b_stage2_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "mvp4b_stage2_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert any("leakage" in issue for issue in report["blocking_issues"])
