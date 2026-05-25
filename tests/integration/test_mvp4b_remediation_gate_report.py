from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _failure_diagnostics() -> dict:
    return {
        "diagnostics_version": "baseline_failure_diagnostics_v001",
        "no_go_confirmed": True,
        "no_go_reason_classes": ["sample_weight_failure", "feature_weakness"],
        "no_final_labels": True,
    }


def _weight_report(candidate_fraction: float = 0.5) -> dict:
    return {
        "report_version": "sample_weight_policy_v001",
        "policy_summary": {
            "capped_class_balanced_confidence": {
                "candidate_effective_weight_fraction": candidate_fraction
            }
        },
        "no_final_labels": True,
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "warnings": [],
        "errors": [],
    }


def _feature_report(finite_ratio: float = 1.0) -> dict:
    return {
        "report_version": "mvp4b_enhanced_features_v001",
        "enhanced_transformed_feature_finite_ratio": finite_ratio,
        "used_label_information_for_features": False,
        "no_final_labels": True,
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "warnings": [],
        "errors": [],
    }


def _ablation_report(
    *,
    margin: float = 0.04,
    class_balanced_success: bool = True,
) -> dict:
    best = {
        "scenario_name": "enhanced_capped_exclude_disagreement_d05",
        "model_type": "linear_probe",
        "feature_set": "enhanced_normalized",
        "weight_policy": "capped_class_balanced_confidence",
        "balanced_accuracy": 0.54,
        "permutation_balanced_accuracy": 0.50,
        "real_minus_permutation_margin": margin,
        "predicted_positive_rate": 0.36,
        "degenerate_prediction": False,
        "folds_above_permutation": 2,
    }
    return {
        "report_version": "baseline_remediation_ablation_v001",
        "best_non_degenerate_scenario": best,
        "class_balanced_non_degenerate_above_permutation": class_balanced_success,
        "only_confidence_only_effective": False,
        "no_go_reasons": [] if class_balanced_success else ["class_balanced_margin_not_met"],
        "no_final_labels": True,
        "no_deep_learning": True,
        "no_stc": True,
        "no_apes": True,
        "no_mvp4c": True,
        "warnings": [],
        "errors": [],
    }


def _write_inputs(
    reports_dir: Path,
    *,
    margin: float = 0.04,
    class_balanced_success: bool = True,
) -> None:
    _write_json(reports_dir / "baseline_failure_diagnostics_v001.json", _failure_diagnostics())
    _write_json(reports_dir / "sample_weight_policy_report_v001.json", _weight_report())
    _write_json(reports_dir / "enhanced_feature_report_v001.json", _feature_report())
    _write_json(
        reports_dir / "baseline_remediation_ablation_v001.json",
        _ablation_report(margin=margin, class_balanced_success=class_balanced_success),
    )


def _write_paths_config(path: Path, reports_dir: Path) -> None:
    root_dir = reports_dir.parent
    path.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )


def test_mvp4b_remediation_gate_report_allows_consideration_when_conditions_pass(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06k_generate_mvp4b_remediation_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MVP-4B remediation gate decision=go" in result.stdout
    report = json.loads(
        (reports_dir / "mvp4b_remediation_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "go"
    assert report["mvp4c_consideration_allowed"] is True


def test_mvp4b_remediation_gate_report_blocks_small_margin(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(reports_dir, margin=0.019)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06k_generate_mvp4b_remediation_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "mvp4b_remediation_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert report["mvp4c_consideration_allowed"] is False
    assert any("margin" in issue for issue in report["blocking_issues"])


def test_mvp4b_remediation_gate_report_blocks_missing_class_balanced_success(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(reports_dir, class_balanced_success=False)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06k_generate_mvp4b_remediation_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "mvp4b_remediation_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert any("class-balanced" in issue for issue in report["blocking_issues"])
