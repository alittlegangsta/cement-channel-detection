from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _scenario(*, margin: float = 0.08, group: str = "all_depth_features") -> dict:
    return {
        "scenario_id": f"{group}__exclude5700_true__conf_0p5__split_3__logistic_regression",
        "feature_group": group,
        "exclude_5700_band": True,
        "confidence_threshold": 0.5,
        "n_splits": 3,
        "model_type": "logistic_regression",
        "balanced_accuracy_mean": 0.7,
        "permutation_balanced_accuracy_mean": 0.7 - margin,
        "margin_mean": margin,
        "margin_std": 0.01,
        "predicted_positive_rate": 0.5,
        "folds_above_permutation_fraction": 1.0,
        "degenerate_prediction": False,
        "passes_gate_thresholds": margin >= 0.05,
    }


def _refinement_report(
    *,
    margin: float = 0.08,
    single_group: bool = False,
    suspicious: bool = False,
) -> dict:
    best = _scenario(margin=margin)
    groups = (
        ["all_depth_features"]
        if single_group
        else ["all_depth_features", "receiver_summary_features"]
    )
    return {
        "report_version": "depth_level_refinement_v001",
        "recommendation": "go",
        "best_result": best,
        "best_feature_group": best["feature_group"],
        "passing_scenario_count": len(groups) * 2,
        "robustness_summary": {
            "passing_scenario_count": len(groups) * 2,
            "passing_feature_groups": groups,
            "passing_confidence_thresholds": [0.4, 0.5, 0.6],
            "passing_depth_block_splits": [3, 5],
            "passing_exclude_5700_values": [False, True],
            "depends_on_single_feature_group": single_group,
            "depends_on_single_confidence_threshold": False,
            "depends_on_single_split": False,
            "depends_on_5700_band": False,
            "exclude_5700_still_passes": True,
            "stable_over_permutation": True,
            "suspicious_leakage": suspicious,
        },
        "manual_confirmation_items": [],
        "production_training": False,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
        "warnings": [],
        "errors": [],
    }


def _baseline_report() -> dict:
    return {
        "report_version": "depth_level_baseline_v001",
        "best_result": {"balanced_accuracy_margin": 0.08},
        "production_training": False,
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
    }


def _review_summary() -> dict:
    return {
        "review_version": "depth_level_refinement_review_v001",
        "figures": {"a": "a.png"},
        "manual_confirmation_items": [],
        "no_final_labels": True,
        "no_stc": True,
        "no_apes": True,
        "no_deep_learning": True,
        "no_mvp4c": True,
        "warnings": [],
        "errors": [],
    }


def _write_inputs(
    reports_dir: Path,
    *,
    margin: float = 0.08,
    single_group: bool = False,
    suspicious: bool = False,
) -> None:
    _write_json(
        reports_dir / "depth_level_refinement_report_v001.json",
        _refinement_report(margin=margin, single_group=single_group, suspicious=suspicious),
    )
    _write_json(reports_dir / "depth_level_baseline_report_v001.json", _baseline_report())
    _write_json(
        reports_dir
        / "depth_level_refinement_review_v001"
        / "depth_level_refinement_review_summary_v001.json",
        _review_summary(),
    )


def _write_paths_config(path: Path, reports_dir: Path) -> None:
    path.write_text(
        "\n".join(["data:", f"  root: {reports_dir.parent}", f"  reports: {reports_dir}", ""]),
        encoding="utf-8",
    )


def test_depth_level_refinement_gate_go_when_robust(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    _write_inputs(reports_dir)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06aa_generate_depth_level_refinement_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=go" in result.stdout
    report = json.loads(
        (reports_dir / "depth_level_refinement_gate_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["decision"] == "go"
    assert report["mvp4c_allowed"] is False
    assert report["stc_allowed"] is False
    assert report["final_labels_allowed"] is False
    assert report["next_branch_requires_human_approval"] is True


def test_depth_level_refinement_gate_conditional_for_single_group(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    _write_inputs(reports_dir, single_group=True)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06aa_generate_depth_level_refinement_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=conditional_go" in result.stdout
    report = json.loads(
        (reports_dir / "depth_level_refinement_gate_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["manual_confirmation_required"] is True
    assert report["mvp4c_allowed"] is False


def test_depth_level_refinement_gate_no_go_for_suspicious_leakage(tmp_path: Path) -> None:
    reports_dir = tmp_path / "data" / "reports"
    _write_inputs(reports_dir, suspicious=True)
    paths_config = tmp_path / "paths.yaml"
    _write_paths_config(paths_config, reports_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/06aa_generate_depth_level_refinement_gate.py",
            "--paths",
            str(paths_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(
        (reports_dir / "depth_level_refinement_gate_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["decision"] == "no_go"
    assert any("leakage" in issue for issue in report["blocking_issues"])
