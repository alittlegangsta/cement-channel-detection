from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _receiver_feature_report() -> dict:
    return {
        "report_version": "receiver_derived_feature_report_v001",
        "finite_ratio": {
            "raw_receiver_features": 1.0,
            "transformed_receiver_features": 1.0,
            "output_transformed_features": 1.0,
        },
        "used_label_information_for_feature_construction": False,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
        "warnings": [],
        "errors": [],
    }


def _receiver_ablation(margin: float = 0.04, feature_set: str = "receiver_derived_only") -> dict:
    best = {
        "scenario_name": "receiver_only_exclude_disagreement",
        "model_type": "linear_probe",
        "feature_set": feature_set,
        "weight_policy": "capped_class_balanced_confidence",
        "balanced_accuracy": 0.54,
        "permutation_balanced_accuracy": 0.50,
        "real_minus_permutation_margin": margin,
        "predicted_positive_rate": 0.40,
        "degenerate_prediction": False,
        "folds_above_permutation": 2,
        "leakage_suspected": False,
    }
    return {
        "report_version": "receiver_feature_ablation_v001",
        "summary_rows": [best],
        "best_non_degenerate_scenario": best,
        "best_non_degenerate_margin": margin,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
        "warnings": [],
        "errors": [],
    }


def _remediation_gate() -> dict:
    return {"decision": "no_go", "mvp4c_consideration_allowed": False}


def _write_inputs(
    reports_dir: Path,
    *,
    margin: float = 0.04,
    feature_set: str = "receiver_derived_only",
) -> None:
    _write_json(
        reports_dir / "receiver_derived_feature_report_v001.json",
        _receiver_feature_report(),
    )
    _write_json(
        reports_dir / "receiver_feature_ablation_v001.json",
        _receiver_ablation(margin=margin, feature_set=feature_set),
    )
    _write_json(reports_dir / "mvp4b_remediation_gate_report.json", _remediation_gate())


def _write_paths_config(path: Path, reports_dir: Path) -> None:
    root_dir = reports_dir.parent
    path.write_text(
        "\n".join(["data:", f"  root: {root_dir}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )


def test_receiver_feature_gate_report_allows_consideration_when_conditions_pass(
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
            "scripts/06n_generate_receiver_feature_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Receiver feature gate decision=go" in result.stdout
    report = json.loads(
        (reports_dir / "receiver_feature_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "go"
    assert report["mvp4c_consideration_allowed"] is True


def test_receiver_feature_gate_report_blocks_small_receiver_margin(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(reports_dir, margin=0.016)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06n_generate_receiver_feature_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "receiver_feature_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert any("margin" in issue for issue in report["blocking_issues"])


def test_receiver_feature_gate_report_blocks_side_only_best_result(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    _write_inputs(reports_dir, margin=0.04, feature_set="side_level_enhanced_only")
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06n_generate_receiver_feature_gate_report.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "receiver_feature_gate_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "no_go"
    assert any("receiver-derived" in issue for issue in report["blocking_issues"])
